#!/usr/bin/env python
"""Test MLE registration with debug visualization."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

import taichi as ti
ti.init(arch=ti.cuda)

device = 'cuda'

# Load data
from misc.hier_IO import load_hier_to_torch
hier_path = Path(__file__).parent.parent.parent / 'data' / 'merged.hier'
hier_scene = load_hier_to_torch(hier_path, device=device)
scene = hier_scene.gaussian_scene

print("=" * 70)
print("MLE Registration with Debug Visualization")
print("=" * 70)

# Sample point cloud from scene
np.random.seed(42)
sample_indices = torch.randperm(len(scene.position))[:1000]
pointcloud = scene.position[sample_indices].clone()

# Apply known transform for testing
from gmm_point_alignment.transform_utils import se3_exp

np.random.seed(42)
xi_true = torch.tensor([0.3, -0.2, 0.1, 0.05, -0.03, 0.02], device=device)
T_true = se3_exp(xi_true)
R_true = T_true[:3, :3]
t_true = T_true[:3, 3]

pointcloud_transformed = (R_true @ pointcloud.T).T + t_true

print(f"\nTrue transform:")
print(f"  Translation: {t_true.cpu().numpy()}")
print(f"  Rotation angle: {np.degrees(torch.acos((R_true.trace()-1)/2).item()):.2f}°")

# =============================================================================
# Build CSR Grid and Plot Sphere Distribution Histogram
# =============================================================================

print(f"\n{'=' * 70}")
print("Building CSR Grid and Analyzing Sphere Distribution")
print(f"{'=' * 70}")

from gmm_point_alignment.mle_registration import CSRGridBuilder, CSRGridBuilderConfig

grid_config = CSRGridBuilderConfig(
    voxel_size_strategy="median_radius",
    voxel_size_factor=1.0,
)
grid_builder = CSRGridBuilder(grid_config)
grid_data = grid_builder.build(scene)

# Plot spheres per voxel histogram
grid_builder.plot_spheres_per_voxel_histogram(
    grid_data,
    output_path="spheres_per_voxel_histogram.png"
)

print(f"  Grid histogram saved to: spheres_per_voxel_histogram.png")

# Test with debug mode enabled
from gmm_point_alignment.unified_registration import (
    UnifiedRegistration,
    UnifiedConfig,
    RegistrationMethod,
)

print(f"\n{'=' * 70}")
print("Running MLE with Debug Visualization")
print(f"{'=' * 70}")

config = UnifiedConfig(
    method=RegistrationMethod.MLE,
    mle_voxel_strategy="median_radius",
    mle_voxel_factor=0.5,
    mle_num_iters=10000,
    mle_lr=0.001,
    mle_top_k=32,
    mle_lr_translation=0.005,  # Much smaller LR for translation
    mle_lr_rotation=0.005,      # Larger LR for rotation
    mle_use_scale=True,
    mle_multi_init=True,
    mle_num_init=5,
    mle_debug=True,
    mle_debug_gt_transform=torch.inverse(T_true),
)

reg = UnifiedRegistration(config)
result = reg.register(scene, pointcloud_transformed)

print(f"\nResults:")
print(f"  Converged: {result.converged}")
print(f"  Loss: {result.error:.4f}")
print(f"  Scale: {result.scale:.4f}")

# Compute ground truth inverse transform
# If pointcloud_transformed = T_true(pointcloud),
# then to align back: T_inv(pointcloud_transformed) = pointcloud
T_true_inv = torch.inverse(T_true)
R_true_inv = T_true_inv[:3, :3]
t_true_inv = T_true_inv[:3, 3]

print(f"\n  Expected transform (inverse of T_true):")
print(f"    Translation: {t_true_inv.cpu().numpy()}")
print(f"    Rotation angle: {np.degrees(torch.acos((R_true_inv.trace()-1)/2).item()):.2f}°")

print(f"\n  Recovered transform:")
print(f"    Translation: {result.t.cpu().numpy()}")
R_result = result.R
print(f"    Rotation angle: {np.degrees(torch.acos((R_result.trace()-1)/2).item()):.2f}°")

t_error = (result.t - t_true_inv).norm().item()
R_rel = R_result.T @ R_true_inv
cos_angle = torch.clamp((R_rel.trace() - 1) / 2, -1, 1)
angle_error = torch.acos(cos_angle).item() * 180 / np.pi

print(f"\n  Errors (comparing to inverse transform):")
print(f"    Translation error: {t_error:.4f}m")
print(f"    Rotation error: {angle_error:.2f}°")

print(f"\n{'=' * 70}")
print("Debug visualization saved to: registration_debug.png")
print(f"{'=' * 70}")
