#!/usr/bin/env python
"""Test improved MLE registration with robust kernels and PCA initialization.

Compares:
1. Baseline MLE
2. MLE with Huber kernel
3. MLE with Cauchy kernel
4. MLE with PCA initialization
5. MLE with Sim(3) scale estimation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from time import time

import taichi as ti
ti.init(arch=ti.cuda)

device = 'cuda'

# =============================================================================
# Load Data
# =============================================================================

print("=" * 70)
print("Robust MLE Registration Test")
print("=" * 70)

from misc.hier_IO import load_hier_to_torch
hier_path = Path(__file__).parent.parent.parent / 'data' / 'merged.hier'
hier_scene = load_hier_to_torch(hier_path, device=device)
scene = hier_scene.gaussian_scene

print(f"\nLoaded Gaussian scene: {len(scene.position)} spheres")

from tests.utils import read_ply_xyz

ply_path = Path(__file__).parent.parent.parent / 'data' / 'points3D.ply'
pointcloud = read_ply_xyz(ply_path)

# Sample for faster processing
if len(pointcloud) > 5000:
    indices = torch.randperm(len(pointcloud))[:5000]
    pointcloud = pointcloud[indices]

print(f"Loaded point cloud: {len(pointcloud)} points")

# =============================================================================
# Apply Transform (with scale)
# =============================================================================

from gmm_point_alignment.transform_utils import se3_exp

np.random.seed(42)

# Apply known transform WITH SCALE
xi_true = torch.tensor([0.3, -0.2, 0.1, 0.05, -0.03, 0.02], device=device)
scale_true = 1.5  # VGGT has unknown scale

T_true = se3_exp(xi_true)
R_true = T_true[:3, :3]
t_true = T_true[:3, 3]

# Apply scale to rotation
pointcloud_transformed = scale_true * (R_true @ pointcloud.T).T + t_true

print(f"\nApplied transform:")
print(f"  Scale: {scale_true}")
print(f"  Translation: {t_true.cpu().numpy()}")
print(f"  Rotation: {np.degrees(torch.acos((R_true.trace()-1)/2).item()):.2f}°")

# Ground truth inverse (for error computation)
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
# Test Different Configurations
# =============================================================================

from gmm_point_alignment.unified_registration import (
    UnifiedRegistration,
    UnifiedConfig,
    RegistrationMethod,
)

configs = [
    ("Baseline MLE", {
        'mle_use_scale': False,
        'mle_robust_kernel': 'none',
        'mle_use_pca_init': False,
        'mle_top_k': 8,
    }),
    ("MLE + Huber", {
        'mle_use_scale': False,
        'mle_robust_kernel': 'huber',
        'mle_kernel_threshold': 0.1,
        'mle_use_pca_init': False,
        'mle_top_k': 8,
    }),
    ("MLE + Cauchy", {
        'mle_use_scale': False,
        'mle_robust_kernel': 'cauchy',
        'mle_kernel_threshold': 0.1,
        'mle_use_pca_init': False,
        'mle_top_k': 8,
    }),
    ("MLE + Geman-McClure", {
        'mle_use_scale': False,
        'mle_robust_kernel': 'geman_mcclure',
        'mle_kernel_threshold': 0.1,
        'mle_use_pca_init': False,
        'mle_top_k': 8,
    }),
    ("MLE + TopK=16", {
        'mle_use_scale': False,
        'mle_robust_kernel': 'none',
        'mle_use_pca_init': False,
        'mle_top_k': 16,
    }),
    ("MLE + PCA Init", {
        'mle_use_scale': False,
        'mle_robust_kernel': 'none',
        'mle_use_pca_init': True,
        'mle_top_k': 8,
    }),
    ("MLE + Sim(3)", {
        'mle_use_scale': True,
        'mle_robust_kernel': 'none',
        'mle_use_pca_init': False,
        'mle_top_k': 8,
    }),
    ("MLE + Sim(3) + PCA Init", {
        'mle_use_scale': True,
        'mle_robust_kernel': 'none',
        'mle_use_pca_init': True,
        'mle_top_k': 8,
    }),
    ("MLE + Sim(3) + Huber + PCA", {
        'mle_use_scale': True,
        'mle_robust_kernel': 'huber',
        'mle_kernel_threshold': 0.1,
        'mle_use_pca_init': True,
        'mle_top_k': 16,
    }),
]

print("\n" + "=" * 70)
print("Testing Different MLE Configurations")
print("=" * 70)
print(f"\n{'Config':<35} {'Time(s)':<10} {'T-Err(m)':<12} {'R-Err(°)':<12} {'Scale':<10} {'Loss':<10}")
print("-" * 100)

results = []

for name, kwargs in configs:
    config = UnifiedConfig(
        method=RegistrationMethod.MLE,
        mle_voxel_strategy="median_radius",
        mle_voxel_factor=1.0,
        mle_num_iters=100,
        mle_lr=0.01,
        mle_lr_translation=0.01,
        mle_lr_rotation=0.001,
        mle_multi_init=True,
        mle_num_init=5,
        **kwargs
    )

    try:
        start_time = time()
        reg = UnifiedRegistration(config)
        result = reg.register(scene, pointcloud_transformed)
        elapsed = time() - start_time

        t_err, r_err = compute_errors(result)
        scale_str = f"{result.scale:.2f}" if hasattr(result, 'scale') else "1.00"

        print(f"{name:<35} {elapsed:<10.2f} {t_err:<12.4f} {r_err:<12.2f} {scale_str:<10} {result.error:<10.2f}")

        results.append({
            'name': name,
            'time': elapsed,
            't_err': t_err,
            'r_err': r_err,
            'scale': result.scale if hasattr(result, 'scale') else 1.0,
            'loss': result.error,
            'converged': result.converged,
        })
    except Exception as e:
        print(f"{name:<35} FAILED: {str(e)[:50]}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(f"\nTrue scale: {scale_true}")
print(f"\nBest by translation error:")
best_t = min(results, key=lambda x: x['t_err'])
print(f"  {best_t['name']}: {best_t['t_err']:.4f}m")

print(f"\nBest by rotation error:")
best_r = min(results, key=lambda x: x['r_err'])
print(f"  {best_r['name']}: {best_r['r_err']:.2f}°")

if any(r['scale'] != 1.0 for r in results):
    print(f"\nBest scale estimate:")
    best_s = min([r for r in results if r['scale'] != 1.0],
                 key=lambda x: abs(x['scale'] - scale_true))
    print(f"  {best_s['name']}: {best_s['scale']:.2f} (true: {scale_true})")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
