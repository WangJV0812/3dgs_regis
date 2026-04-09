#!/usr/bin/env python
"""Visualization script for GMM point cloud registration results.

Shows before/after comparison of point cloud alignment with Gaussian scene.
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from time import time

import taichi as ti

from misc.hier_IO import load_hier_to_torch
from gmm_point_alignment.gmm_point_alignment import (
    GMMPointAlignment,
    GMMPointAlignmentConfig,
)
from gmm_point_alignment.csr_grid_builder import CSRGridBuilderConfig, VoxelSizeStrategy
from gmm_point_alignment.sphere_mle_loss import RegistrationConfig
from gmm_point_alignment.transform_utils import se3_exp


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

        assert vertex_count is not None, "No vertex count found"
        assert format_type == "binary_little_endian", f"Unsupported format: {format_type}"

        # Find XYZ property indices
        xyz_indices = []
        for i, (dtype, name) in enumerate(properties):
            if name in ['x', 'y', 'z']:
                xyz_indices.append(i)

        assert len(xyz_indices) == 3, f"Expected x,y,z properties, got {len(xyz_indices)}"

        # Calculate property sizes
        dtype_map = {
            'char': 1, 'uchar': 1, 'short': 2, 'ushort': 2,
            'int': 4, 'uint': 4, 'float': 4, 'double': 8,
            'float32': 4, 'float64': 8, 'int32': 4, 'uint32': 4,
        }

        prop_sizes = []
        for dtype, name in properties:
            size = dtype_map.get(dtype, 4)
            prop_sizes.append(size)

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

    return torch.from_numpy(points)


def covariance_from_scale_rotation(scale, rotation):
    """Compute covariance matrix from scale and rotation quaternion."""
    # Ensure float32
    scale = np.asarray(scale, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)

    # Rotation matrix from quaternion
    q = rotation / (np.linalg.norm(rotation) + 1e-8)
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ], dtype=np.float32)

    # Scale matrix
    S = np.diag(scale**2).astype(np.float32)

    # Covariance
    return R @ S @ R.T


def draw_ellipsoid(ax, center, covariance, color='blue', alpha=0.1, n_points=20):
    """Draw a 3D ellipsoid given center and covariance."""
    # Ensure float32 for linalg
    covariance = np.asarray(covariance, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Radii (3 sigma for 99% confidence)
    radii = 3 * np.sqrt(np.maximum(eigenvalues, 0))

    # Generate unit sphere
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Transform to ellipsoid
    points = np.array([x.flatten(), y.flatten(), z.flatten()])
    ellipsoid_points = eigenvectors @ np.diag(radii) @ points + center.reshape(3, 1)

    x_ell = ellipsoid_points[0, :].reshape(n_points, n_points)
    y_ell = ellipsoid_points[1, :].reshape(n_points, n_points)
    z_ell = ellipsoid_points[2, :].reshape(n_points, n_points)

    ax.plot_surface(x_ell, y_ell, z_ell, color=color, alpha=alpha, linewidth=0)


def draw_gaussian_scene(ax, scene, max_spheres=100, color='blue', alpha=0.05):
    """Draw Gaussian scene as ellipsoids."""
    # Convert to numpy and ensure float32 (linalg doesn't support float16)
    positions = scene.position.cpu().numpy().astype(np.float32)
    scales = scene.scales.cpu().numpy().astype(np.float32)
    rotations = scene.rotation.cpu().numpy().astype(np.float32)
    opacities = scene.opacities.cpu().numpy().astype(np.float32)

    # Sort by opacity and take top max_spheres
    indices = np.argsort(opacities)[::-1][:max_spheres]

    for i in indices:
        center = positions[i]
        scale = scales[i]
        rotation = rotations[i]
        opacity = opacities[i]

        # Skip very transparent gaussians
        if opacity < 0.1:
            continue

        # Clamp opacity to [0, 1] for visualization
        opacity = np.clip(opacity, 0.0, 1.0)

        # Compute covariance
        cov = covariance_from_scale_rotation(scale, rotation)

        # Draw ellipsoid with opacity-weighted alpha (clamped to valid range)
        final_alpha = np.clip(alpha * opacity, 0.0, 1.0)
        draw_ellipsoid(ax, center, cov, color=color, alpha=final_alpha)


def visualize_registration(
    scene,
    points_before,
    points_after,
    transform,
    save_path=None,
    max_spheres=50,
    max_points=1000,
):
    """Create visualization of registration results."""

    fig = plt.figure(figsize=(18, 6))

    # Convert to numpy
    scene_pos = scene.position.cpu().numpy()
    points_before_np = points_before.cpu().numpy()
    points_after_np = points_after.cpu().numpy()

    # Sample points for visualization
    if len(points_before_np) > max_points:
        indices = np.random.choice(len(points_before_np), max_points, replace=False)
        points_before_np = points_before_np[indices]
        points_after_np = points_after_np[indices]

    # Compute bounds
    all_points = np.vstack([scene_pos, points_before_np, points_after_np])
    center = all_points.mean(axis=0)
    max_range = np.abs(all_points - center).max() * 1.2

    # Subplot 1: Before Registration
    ax1 = fig.add_subplot(131, projection='3d')
    draw_gaussian_scene(ax1, scene, max_spheres=max_spheres, color='blue', alpha=0.05)
    ax1.scatter(
        points_before_np[:, 0],
        points_before_np[:, 1],
        points_before_np[:, 2],
        c='red',
        s=1,
        alpha=0.5,
        label='Point Cloud (Before)'
    )
    ax1.set_title('Before Registration\n(Red = Point Cloud, Blue = Scene)', fontsize=12)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(center[0] - max_range, center[0] + max_range)
    ax1.set_ylim(center[1] - max_range, center[1] + max_range)
    ax1.set_zlim(center[2] - max_range, center[2] + max_range)

    # Subplot 2: After Registration
    ax2 = fig.add_subplot(132, projection='3d')
    draw_gaussian_scene(ax2, scene, max_spheres=max_spheres, color='blue', alpha=0.05)
    ax2.scatter(
        points_after_np[:, 0],
        points_after_np[:, 1],
        points_after_np[:, 2],
        c='green',
        s=1,
        alpha=0.5,
        label='Point Cloud (After)'
    )
    ax2.set_title('After Registration\n(Green = Aligned Point Cloud)', fontsize=12)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(center[0] - max_range, center[0] + max_range)
    ax2.set_ylim(center[1] - max_range, center[1] + max_range)
    ax2.set_zlim(center[2] - max_range, center[2] + max_range)

    # Subplot 3: Overlay Comparison
    ax3 = fig.add_subplot(133, projection='3d')
    draw_gaussian_scene(ax3, scene, max_spheres=max_spheres, color='blue', alpha=0.03)

    # Draw before points with lower alpha
    ax3.scatter(
        points_before_np[:, 0],
        points_before_np[:, 1],
        points_before_np[:, 2],
        c='red',
        s=1,
        alpha=0.2,
        label='Before'
    )

    # Draw after points
    ax3.scatter(
        points_after_np[:, 0],
        points_after_np[:, 1],
        points_after_np[:, 2],
        c='green',
        s=1,
        alpha=0.6,
        label='After'
    )

    # Draw transformation arrows for a few points
    step = max(1, len(points_before_np) // 20)
    for i in range(0, len(points_before_np), step):
        p1 = points_before_np[i]
        p2 = points_after_np[i]
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                'k--', alpha=0.3, linewidth=0.5)

    ax3.set_title('Overlay: Before (Red) → After (Green)\nBlack arrows show transformation', fontsize=12)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim(center[0] - max_range, center[0] + max_range)
    ax3.set_ylim(center[1] - max_range, center[1] + max_range)
    ax3.set_zlim(center[2] - max_range, center[2] + max_range)
    ax3.legend()

    # Add transform info as text
    if transform is not None:
        T = transform.cpu().numpy()
        translation = T[:3, 3]
        # Extract rotation angle from rotation matrix
        trace = np.trace(T[:3, :3])
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

        info_text = f'Transform Info:\n'
        info_text += f'Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]\n'
        info_text += f'Rotation Angle: {np.degrees(angle):.2f}°\n'
        info_text += f'Rotation Matrix:\n{T[:3, :3]}'

        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.15, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    plt.show()
    return fig


def create_synthetic_test_case():
    """Create synthetic test case for visualization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create synthetic scene (grid of spheres)
    from misc.hier_IO import GaussianScenes

    n_per_dim = 5
    positions = []
    for x in np.linspace(-5, 5, n_per_dim):
        for y in np.linspace(-5, 5, n_per_dim):
            for z in np.linspace(-5, 5, n_per_dim):
                positions.append([x, y, z])

    positions = torch.tensor(positions, dtype=torch.float32, device=device)
    num_spheres = len(positions)

    scene = GaussianScenes(
        position=positions,
        rotation=torch.tensor([[1, 0, 0, 0]] * num_spheres, dtype=torch.float32, device=device),
        scales=torch.ones(num_spheres, 3, device=device) * 0.5,
        opacities=torch.ones(num_spheres, device=device),
        shs=torch.randn(num_spheres, 3, 16, device=device),
    )

    # Create point cloud (subset of scene centers + noise)
    indices = torch.randperm(num_spheres)[:30]
    pointcloud = positions[indices] + torch.randn(30, 3, device=device) * 0.3

    # Apply known transform
    xi_true = torch.tensor([2.0, -1.0, 0.5, 0.2, -0.1, 0.1], device=device)
    T_true = se3_exp(xi_true)

    R = T_true[:3, :3]
    t = T_true[:3, 3]
    pointcloud_transformed = (R @ pointcloud.T).T + t

    return scene, pointcloud_transformed, T_true


def main():
    print("=" * 70)
    print("GMM Point Cloud Registration - Visualization")
    print("=" * 70)

    # Initialize Taichi
    ti.init(arch=ti.cuda if torch.cuda.is_available() else ti.cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check for real data
    data_dir = Path("./data")
    hier_path = data_dir / "merged.hier"
    ply_path = data_dir / "points3D.ply"

    use_real_data = hier_path.exists() and ply_path.exists()

    if use_real_data:
        print("\nUsing real data...")
        hier_scene = load_hier_to_torch(hier_path, device=device)
        scene = hier_scene.gaussian_scene
        pointcloud = read_ply_xyz(ply_path).to(device)

        # Sample for faster processing
        pointcloud = pointcloud[torch.randperm(len(pointcloud))[:2000]]
        print(f"Scene: {scene.position.shape[0]} spheres")
        print(f"Pointcloud (sampled): {pointcloud.shape[0]} points")
    else:
        print("\nReal data not found, using synthetic test case...")
        scene, pointcloud, T_true = create_synthetic_test_case()
        print(f"Synthetic scene: {scene.position.shape[0]} spheres")
        print(f"Synthetic pointcloud: {pointcloud.shape[0]} points")

    # Build alignment with optimal voxel strategy
    print("\nBuilding grid...")
    config = GMMPointAlignmentConfig(
        grid_config=CSRGridBuilderConfig(
            voxel_size_strategy=VoxelSizeStrategy.MEDIAN_RADIUS,
            voxel_size_factor=1.0,
        ),
        reg_config=RegistrationConfig(
            num_iters=100,
            lr=0.01,
            multi_init=True,
            num_init=5,
            verbose=True,
            use_scale=True,  # Enable scale optimization
            scale_lr=0.001,
        )
    )

    aligner = GMMPointAlignment(config)
    aligner.build_grid(scene)

    # Store original points
    points_before = pointcloud.clone()

    # Run registration
    print("\nRunning registration...")
    reg_result = aligner.register(pointcloud)

    # Apply recovered transform to get after state
    T_recovered = reg_result.transform
    R = T_recovered[:3, :3]
    t = T_recovered[:3, 3]
    points_after = (R @ points_before.T).T + t

    print("\n" + "=" * 70)
    print("Registration Results:")
    print("=" * 70)
    print(f"Loss: {reg_result.loss:.4f}")
    print(f"Inlier Ratio: {reg_result.inlier_ratio:.4f}")
    print(f"Iterations: {reg_result.num_iters}")
    print(f"Converged: {reg_result.converged}")

    # Create visualization
    print("\nCreating visualization...")
    visualize_registration(
        scene,
        points_before,
        points_after,
        T_recovered,
        save_path="registration_result.png",
        max_spheres=100,
        max_points=500,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
