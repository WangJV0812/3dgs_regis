#!/usr/bin/env python
"""Test Top-K Sampler registration method.

Compares three registration methods:
1. MLE - GMM MLE-based registration
2. Sampler - Traditional ICP with random sampling
3. TopK Sampler - ICP with Top-K sphere sampling
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from pathlib import Path
from time import time

import taichi as ti
ti.init(arch=ti.cuda)

device = 'cuda'

# =============================================================================
# Load Data
# =============================================================================

print("=" * 70)
print("Top-K Sampler Registration Test")
print("=" * 70)

# Load Gaussian scene
from misc.hier_IO import load_hier_to_torch
hier_path = Path('data/merged.hier')
hier_scene = load_hier_to_torch(hier_path, device=device)
scene = hier_scene.gaussian_scene

print(f"\nLoaded Gaussian scene:")
print(f"  Spheres: {len(scene.position)}")
print(f"  Position range: [{scene.position.min().item():.2f}, {scene.position.max().item():.2f}]")
print(f"  Scale range: [{scene.scales.min().item():.2f}, {scene.scales.max().item():.2f}]")

# Load point cloud from PLY
def read_ply_xyz(ply_path: Path) -> torch.Tensor:
    """Simple binary PLY reader that extracts XYZ coordinates."""
    with open(ply_path, 'rb') as f:
        # Parse header
        line = f.readline().decode('ascii').strip()
        assert line == "ply", f"Not a PLY file: {line}"

        vertex_count = None
        properties = []
        format_type = None

        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith("format"):
                parts = line.split()
                format_type = parts[1]
            elif line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            elif line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break

        assert vertex_count is not None
        assert format_type == "binary_little_endian"

        # Find XYZ property indices
        xyz_indices = []
        for i, (dtype, name) in enumerate(properties):
            if name in ['x', 'y', 'z']:
                xyz_indices.append(i)

        dtype_map = {
            'char': 1, 'uchar': 1, 'short': 2, 'ushort': 2,
            'int': 4, 'uint': 4, 'float': 4, 'double': 8,
            'float32': 4, 'float64': 8, 'int32': 4, 'uint32': 4,
        }

        prop_sizes = [dtype_map.get(dtype, 4) for dtype, _ in properties]
        vertex_stride = sum(prop_sizes)
        data = f.read()

    # Extract XYZ coordinates
    points = np.zeros((vertex_count, 3), dtype=np.float32)
    xyz_offsets = [sum(prop_sizes[:i]) for i in xyz_indices]

    for i in range(vertex_count):
        offset = i * vertex_stride
        for j, (prop_idx, prop_offset) in enumerate(zip(xyz_indices, xyz_offsets)):
            prop_dtype = properties[prop_idx][0]
            if prop_dtype in ['float', 'float32']:
                val = np.frombuffer(data[offset + prop_offset:offset + prop_offset + 4], dtype=np.float32)[0]
            elif prop_dtype == 'double':
                val = np.frombuffer(data[offset + prop_offset:offset + prop_offset + 8], dtype=np.float64)[0]
            else:
                continue
            points[i, j] = val

    return torch.from_numpy(points).to(device)

ply_path = Path('data/points3D.ply')
pointcloud = read_ply_xyz(ply_path)

print(f"\nLoaded point cloud:")
print(f"  Points: {len(pointcloud)}")
print(f"  Position range: [{pointcloud.min().item():.2f}, {pointcloud.max().item():.2f}]")

# Sample for faster processing
if len(pointcloud) > 5000:
    indices = torch.randperm(len(pointcloud))[:5000]
    pointcloud = pointcloud[indices]
    print(f"  Sampled to: {len(pointcloud)} points")

# =============================================================================
# Apply Known Transform for Testing
# =============================================================================

from gmm_point_alignment.transform_utils import se3_exp

# Apply a known transform to test registration
np.random.seed(42)
xi_true = torch.tensor([0.3, -0.2, 0.1, 0.05, -0.03, 0.02], device=device)
T_true = se3_exp(xi_true)
R_true = T_true[:3, :3]
t_true = T_true[:3, 3]

pointcloud_transformed = (R_true @ pointcloud.T).T + t_true

print(f"\nApplied transform:")
print(f"  Translation: {t_true.cpu().numpy()}")
print(f"  Rotation angle: {np.degrees(torch.acos((R_true.trace()-1)/2).item()):.2f}°")

# Ground truth inverse
T_true_inv = torch.inverse(T_true)
R_true_inv = T_true_inv[:3, :3]
t_true_inv = T_true_inv[:3, 3]

def compute_errors(result):
    """Compute translation and rotation errors."""
    t_error = (result.t - t_true_inv).norm().item()

    R_result = result.R
    R_rel = R_result.T @ R_true_inv
    cos_angle = torch.clamp((R_rel.trace() - 1) / 2, -1, 1)
    angle_error = torch.acos(cos_angle).item() * 180 / np.pi

    return t_error, angle_error

# =============================================================================
# Test 1: MLE Registration
# =============================================================================

print(f"\n{'=' * 70}")
print("Test 1: MLE Registration (GMM-based)")
print(f"{'=' * 70}")

from gmm_point_alignment.unified_registration import (
    UnifiedRegistration,
    UnifiedConfig,
    RegistrationMethod,
)

config_mle = UnifiedConfig(
    method=RegistrationMethod.MLE,
    mle_voxel_strategy="median_radius",
    mle_voxel_factor=1.0,
    mle_num_iters=100,
    mle_lr=0.01,
    mle_use_scale=False,
    mle_multi_init=True,
    mle_num_init=5,
)

start_time = time()
reg_mle = UnifiedRegistration(config_mle)
result_mle = reg_mle.register(scene, pointcloud_transformed)
mle_time = time() - start_time

t_err_mle, r_err_mle = compute_errors(result_mle)

print(f"\nMLE Results:")
print(f"  Time: {mle_time:.2f}s")
print(f"  Converged: {result_mle.converged}")
print(f"  Loss: {result_mle.error:.4f}")
print(f"  Iters: {result_mle.num_iters}")
print(f"  Translation error: {t_err_mle:.4f}m")
print(f"  Rotation error: {r_err_mle:.2f}°")

# =============================================================================
# Test 2: Sampler Registration
# =============================================================================

print(f"\n{'=' * 70}")
print("Test 2: Sampler Registration (ICP-based)")
print(f"{'=' * 70}")

config_sampler = UnifiedConfig(
    method=RegistrationMethod.SAMPLER,
    sampler_method="svd_icp",
    sampler_max_iters=100,
    sampler_num_points=5000,
    sampler_multi_init=True,
    sampler_num_init=5,
)

start_time = time()
reg_sampler = UnifiedRegistration(config_sampler)
result_sampler = reg_sampler.register(scene, pointcloud_transformed)
sampler_time = time() - start_time

t_err_sampler, r_err_sampler = compute_errors(result_sampler)

print(f"\nSampler Results:")
print(f"  Time: {sampler_time:.2f}s")
print(f"  Converged: {result_sampler.converged}")
print(f"  RMSE: {result_sampler.error:.4f}")
print(f"  Iters: {result_sampler.num_iters}")
print(f"  Translation error: {t_err_sampler:.4f}m")
print(f"  Rotation error: {r_err_sampler:.2f}°")

# =============================================================================
# Test 3: Top-K Sampler Registration (NEW!)
# =============================================================================

print(f"\n{'=' * 70}")
print("Test 3: Top-K Sampler Registration (NEW!)")
print(f"{'=' * 70}")

config_topk = UnifiedConfig(
    method=RegistrationMethod.TOPK_SAMPLER,
    topk_sampler_k=16,
    topk_sampler_samples_per_sphere=20,
    topk_sampler_sampling_mode="random",
    topk_sampler_voxel_strategy="median_radius",
    topk_sampler_voxel_factor=1.0,
    topk_sampler_reg_method="open3d_icp_point_to_point",
    topk_sampler_max_iters=100,
    topk_sampler_multi_init=True,
    topk_sampler_num_init=5,
)

start_time = time()
reg_topk = UnifiedRegistration(config_topk)
result_topk = reg_topk.register(scene, pointcloud_transformed)
topk_time = time() - start_time

t_err_topk, r_err_topk = compute_errors(result_topk)

print(f"\nTop-K Sampler Results:")
print(f"  Time: {topk_time:.2f}s")
print(f"  Converged: {result_topk.converged}")
print(f"  RMSE: {result_topk.error:.4f}")
print(f"  Iters: {result_topk.num_iters}")
print(f"  Translation error: {t_err_topk:.4f}m")
print(f"  Rotation error: {r_err_topk:.2f}°")
print(f"  Method: {result_topk.method}")

# =============================================================================
# Summary Comparison
# =============================================================================

print(f"\n{'=' * 70}")
print("Summary Comparison")
print(f"{'=' * 70}")
print(f"{'Metric':<30} {'MLE':<15} {'Sampler':<15} {'TopK-Sampler':<15}")
print(f"{'-' * 70}")
print(f"{'Time (s)':<30} {mle_time:<15.2f} {sampler_time:<15.2f} {topk_time:<15.2f}")
print(f"{'Error metric':<30} {result_mle.error:<15.4f} {result_sampler.error:<15.4f} {result_topk.error:<15.4f}")
print(f"{'Converged':<30} {str(result_mle.converged):<15} {str(result_sampler.converged):<15} {str(result_topk.converged):<15}")
print(f"{'Translation error (m)':<30} {t_err_mle:<15.4f} {t_err_sampler:<15.4f} {t_err_topk:<15.4f}")
print(f"{'Rotation error (°)':<30} {r_err_mle:<15.2f} {r_err_sampler:<15.2f} {r_err_topk:<15.2f}")

# =============================================================================
# Test with different Top-K configurations
# =============================================================================

print(f"\n{'=' * 70}")
print("Test 4: Top-K Sampler with Different Configurations")
print(f"{'=' * 70}")

configs_to_test = [
    ("K=4, Samples=5", {"topk_sampler_k": 4, "topk_sampler_samples_per_sphere": 5}),
    ("K=8, Samples=5", {"topk_sampler_k": 8, "topk_sampler_samples_per_sphere": 5}),
    ("K=16, Samples=5", {"topk_sampler_k": 16, "topk_sampler_samples_per_sphere": 5}),
    ("K=8, Samples=20", {"topk_sampler_k": 8, "topk_sampler_samples_per_sphere": 20}),
]

print(f"\n{'Config':<25} {'Time(s)':<10} {'T-Err(m)':<12} {'R-Err(°)':<12} {'RMSE':<10}")
print("-" * 70)

for name, kwargs in configs_to_test:
    config = UnifiedConfig(
        method=RegistrationMethod.TOPK_SAMPLER,
        topk_sampler_k=kwargs["topk_sampler_k"],
        topk_sampler_samples_per_sphere=kwargs["topk_sampler_samples_per_sphere"],
        topk_sampler_sampling_mode="random",
        topk_sampler_voxel_strategy="median_radius",
        topk_sampler_voxel_factor=1.0,
        topk_sampler_reg_method="svd_icp",
        topk_sampler_max_iters=100,
        topk_sampler_multi_init=True,
        topk_sampler_num_init=3,
    )

    start_time = time()
    reg = UnifiedRegistration(config)
    result = reg.register(scene, pointcloud_transformed)
    elapsed = time() - start_time

    t_err, r_err = compute_errors(result)

    print(f"{name:<25} {elapsed:<10.2f} {t_err:<12.4f} {r_err:<12.2f} {result.error:<10.4f}")

print(f"\n{'=' * 70}")
print("All tests completed!")
print(f"{'=' * 70}")
