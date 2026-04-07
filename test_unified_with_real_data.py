#!/usr/bin/env python
"""Test unified registration with real data from ./data directory.

Uses:
- merged.hier: Gaussian scene
- points3D.ply: Point cloud to register
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
print("Unified Registration Test - Real Data")
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

# Compute errors
R_recovered = result_mle.R
# Correct rotation error: theta = acos((trace(R_rel) - 1) / 2)
R_rel = R_recovered.T @ R_true
cos_angle = torch.clamp((R_rel.trace() - 1) / 2, -1, 1)
angle_error = torch.acos(cos_angle)

print(f"\nMLE Results:")
print(f"  Time: {mle_time:.2f}s")
print(f"  Method: {result_mle.method}")
print(f"  Converged: {result_mle.converged}")
print(f"  Loss: {result_mle.error:.4f}")
print(f"  Iters: {result_mle.num_iters}")
print(f"  Scale: {result_mle.scale:.4f}")
print(f"  Translation error: {(result_mle.t - t_true).norm().item():.4f}m")
print(f"  Rotation error: {np.degrees(angle_error.item()):.2f}°")

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

# Compute errors
R_recovered = result_sampler.R
R_rel = R_recovered.T @ R_true
cos_angle = torch.clamp((R_rel.trace() - 1) / 2, -1, 1)
angle_error = torch.acos(cos_angle)

print(f"\nSampler Results:")
print(f"  Time: {sampler_time:.2f}s")
print(f"  Method: {result_sampler.method}")
print(f"  Converged: {result_sampler.converged}")
print(f"  RMSE: {result_sampler.error:.4f}")
print(f"  Iters: {result_sampler.num_iters}")
print(f"  Scale: {result_sampler.scale:.4f}")
print(f"  Translation error: {(result_sampler.t - t_true).norm().item():.4f}m")
print(f"  Rotation error: {np.degrees(angle_error.item()):.2f}°")

# =============================================================================
# Summary
# =============================================================================

print(f"\n{'=' * 70}")
print("Summary Comparison")
print(f"{'=' * 70}")
print(f"{'Metric':<30} {'MLE':<20} {'Sampler':<20}")
print(f"{'-' * 70}")
print(f"{'Time (s)':<30} {mle_time:<20.2f} {sampler_time:<20.2f}")
print(f"{'Error metric':<30} {result_mle.error:<20.4f} {result_sampler.error:<20.4f}")
print(f"{'Scale':<30} {result_mle.scale:<20.4f} {result_sampler.scale:<20.4f}")
print(f"{'Converged':<30} {str(result_mle.converged):<20} {str(result_sampler.converged):<20}")
print(f"{'Translation error (m)':<30} {(result_mle.t - t_true).norm().item():<20.4f} {(result_sampler.t - t_true).norm().item():<20.4f}")

# =============================================================================
# Test 3: MLE with Scale Optimization
# =============================================================================

print(f"\n{'=' * 70}")
print("Test 3: MLE with Scale Optimization")
print(f"{'=' * 70}")

# Apply scale
scale_true = 1.1
pointcloud_scaled = scale_true * (R_true @ pointcloud.T).T + t_true

config_mle_scale = UnifiedConfig(
    method=RegistrationMethod.MLE,
    mle_voxel_strategy="median_radius",
    mle_voxel_factor=1.0,
    mle_num_iters=100,
    mle_lr=0.01,
    mle_use_scale=True,  # Enable scale
    mle_multi_init=True,
    mle_num_init=5,
)

start_time = time()
reg_mle_scale = UnifiedRegistration(config_mle_scale)
result_mle_scale = reg_mle_scale.register(scene, pointcloud_scaled)
mle_scale_time = time() - start_time

print(f"\nMLE with Scale Results:")
print(f"  Time: {mle_scale_time:.2f}s")
print(f"  True scale: {scale_true:.4f}")
print(f"  Recovered scale: {result_mle_scale.scale:.4f}")
print(f"  Scale error: {abs(result_mle_scale.scale - scale_true):.4f}")
print(f"  Converged: {result_mle_scale.converged}")

print(f"\n{'=' * 70}")
print("All tests completed!")
print(f"{'=' * 70}")
