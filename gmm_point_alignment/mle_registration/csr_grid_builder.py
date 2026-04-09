"""CSR Grid Builder - Build compressed sparse row grid for Gaussian spheres.

This module constructs a CSR-format spatial index for efficient point-to-sphere
query. The grid uses a two-level lookup (L1/L2) to balance memory and speed.

Example:
    >>> config = CSRGridBuilderConfig()
    >>> builder = CSRGridBuilder(config)
    >>> grid_data = builder.build(scene)
    >>> print(f"Built grid with {grid_data.total_pairs} pairs")
"""

import torch
import taichi as ti
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from time import time

from misc.hier_IO import GaussianScenes
from .gs_scene_aabb import (
    robust_global_scene_aabb,
    gaussian_scene_aabb,
)
from .morton_code import grid_coords_to_morton

# =============================================================================
# Constants
# =============================================================================

EPS_F32 = 1e-7
MAX_GRID_SIZE = 1024
L1_GRID_SIZE = 32
L2_GRID_SIZE = MAX_GRID_SIZE // L1_GRID_SIZE  # 32


# =============================================================================
# Configuration
# =============================================================================

class VoxelSizeStrategy:
    """Strategy for computing adaptive voxel size."""
    MEDIAN_RADIUS = "median_radius"           # Original: median of max axes
    SHORT_AXIS_MEDIAN = "short_axis_median"   # Median of shortest axes
    SHORT_AXIS_MODE = "short_axis_mode"       # Mode of shortest axes (most common)
    VOLUME_BASED = "volume_based"             # Based on median volume
    PERCENTILE_DENSE = "percentile_dense"     # P10 of small scales (dense regions)


@dataclass
class CSRGridBuilderConfig:
    """Configuration for CSR Grid Builder.

    Args:
        confidence_level: Confidence level for AABB computation (default: 0.95)
        voxel_size_strategy: Strategy for computing voxel size
        voxel_size_factor: Multiplier for computed voxel size (default: 2.0)
        target_spheres_per_voxel: Target number of spheres per voxel (default: 10)
        max_grid_size: Maximum grid resolution per dimension (default: 1024)
        oversized_threshold_voxels: Threshold for oversized spheres (default: 64)
        l1_grid_size: Coarse grid dimension (default: 32)
        use_two_level_lookup: Enable L1/L2 hierarchy (default: True)
        global_aabb_padding_factor: Padding factor for global AABB (default: 0.1)
        global_aabb_clip_quantile: Quantile for outlier clipping (default: 0.01)
    """
    confidence_level: float = 0.95
    voxel_size_strategy: str = VoxelSizeStrategy.SHORT_AXIS_MEDIAN
    voxel_size_factor: float = 2.0
    target_spheres_per_voxel: float = 10.0
    max_grid_size: int = 1024
    oversized_threshold_voxels: int = 64
    l1_grid_size: int = 32
    use_two_level_lookup: bool = True
    global_aabb_padding_factor: float = 0.1
    global_aabb_clip_quantile: float = 0.01


@dataclass
class CSRGridData:
    """Container for CSR grid data.

    Attributes:
        pairs_morton: Sorted morton codes [total_pairs], int64
        pairs_sphere_id: Corresponding sphere IDs [total_pairs], int32
        l1_offsets: L1 lookup table [32, 32, 32], offset into L2 or -1
        l2_blocks: List of L2 block tensors, each [N, 2] (morton, sphere_id)
        oversized_sphere_ids: IDs of oversized spheres [num_oversized], int32
        global_aabb_min: Global AABB minimum [3], float32
        voxel_size: Voxel size, float
        grid_dims: Grid dimensions (Gx, Gy, Gz)
        total_pairs: Total number of sphere-voxel pairs
        num_unique_voxels: Number of unique occupied voxels
        cov_inv: Precomputed covariance inverse [M, 3, 3], float32
        norm_factor: Precomputed normalization factors [M], float32
        sphere_centers: Sphere centers [M, 3], float32
    """
    pairs_morton: torch.Tensor
    pairs_sphere_id: torch.Tensor
    l1_offsets: torch.Tensor
    l2_blocks: List[torch.Tensor]
    oversized_sphere_ids: torch.Tensor
    global_aabb_min: torch.Tensor
    voxel_size: float
    grid_dims: Tuple[int, int, int]
    total_pairs: int
    num_unique_voxels: int
    cov_inv: torch.Tensor
    norm_factor: torch.Tensor
    sphere_centers: torch.Tensor


# =============================================================================
# Taichi Functions (Layer 3)
# =============================================================================

@ti.func
def expand_bits_10(v: ti.u32) -> ti.u32:
    """Expand 10-bit integer to 30-bit for morton encoding."""
    x = ti.cast(v, ti.u32)
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x <<  8)) & 0x0300F00F
    x = (x | (x <<  4)) & 0x030C30C3
    x = (x | (x <<  2)) & 0x09249249
    return x


@ti.func
def encode_morton_ti(x: ti.i32, y: ti.i32, z: ti.i32) -> ti.u32:
    """Encode grid coordinates to morton code."""
    xi = ti.cast(ti.max(0, ti.min(x, 1023)), ti.u32)
    yi = ti.cast(ti.max(0, ti.min(y, 1023)), ti.u32)
    zi = ti.cast(ti.max(0, ti.min(z, 1023)), ti.u32)
    return (expand_bits_10(xi) << 2) | (expand_bits_10(yi) << 1) | expand_bits_10(zi)


@ti.func
def grid_coord_from_point_ti(
    point: ti.math.vec3,
    global_min: ti.math.vec3,
    voxel_size: ti.f32,
) -> ti.math.ivec3:
    """Convert world coordinate to grid coordinate."""
    rel = (point - global_min) / voxel_size
    return ti.cast(ti.floor(rel), ti.i32)


# =============================================================================
# Taichi Kernels (Layer 2)
# =============================================================================

@ti.kernel
def count_voxels_kernel(
    min_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),
    max_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),
    global_min: ti.types.ndarray(dtype=ti.f32, ndim=1),
    voxel_size: ti.f32,
    grid_size: ti.i32,
    oversized_threshold: ti.i32,
    voxel_counts: ti.types.ndarray(dtype=ti.i32, ndim=1),
    out_is_oversized: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    """Count voxels per sphere and mark oversized (Pass 1)."""
    sphere_count = min_corners.shape[0]

    for sphere_id in range(sphere_count):
        p_min = ti.math.vec3([
            min_corners[sphere_id, 0],
            min_corners[sphere_id, 1],
            min_corners[sphere_id, 2]
        ])
        p_max = ti.math.vec3([
            max_corners[sphere_id, 0],
            max_corners[sphere_id, 1],
            max_corners[sphere_id, 2]
        ])

        g_min = grid_coord_from_point_ti(p_min, ti.math.vec3([global_min[0], global_min[1], global_min[2]]), voxel_size)
        g_max = grid_coord_from_point_ti(p_max, ti.math.vec3([global_min[0], global_min[1], global_min[2]]), voxel_size)

        g_min.x = ti.max(0, ti.min(g_min.x, grid_size - 1))
        g_min.y = ti.max(0, ti.min(g_min.y, grid_size - 1))
        g_min.z = ti.max(0, ti.min(g_min.z, grid_size - 1))
        g_max.x = ti.max(0, ti.min(g_max.x, grid_size - 1))
        g_max.y = ti.max(0, ti.min(g_max.y, grid_size - 1))
        g_max.z = ti.max(0, ti.min(g_max.z, grid_size - 1))

        extent = g_max - g_min + 1
        num_voxels = extent.x * extent.y * extent.z

        if num_voxels > oversized_threshold:
            out_is_oversized[sphere_id] = 1
            voxel_counts[sphere_id] = 0
        else:
            out_is_oversized[sphere_id] = 0
            voxel_counts[sphere_id] = num_voxels


@ti.kernel
def enumerate_pairs_kernel(
    min_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),
    max_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),
    global_min: ti.types.ndarray(dtype=ti.f32, ndim=1),
    voxel_size: ti.f32,
    grid_size: ti.i32,
    offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    out_morton: ti.types.ndarray(dtype=ti.i64, ndim=1),
    out_sphere_id: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    """Enumerate sphere-voxel pairs using pre-computed offsets (Pass 2).

    Each sphere writes to its designated slot [offset[i], offset[i+1]).
    No atomic operations needed.
    """
    sphere_count = min_corners.shape[0]

    for sphere_id in range(sphere_count):
        start_idx = offsets[sphere_id]
        end_idx = offsets[sphere_id + 1]

        if end_idx <= start_idx:
            continue

        p_min = ti.math.vec3([
            min_corners[sphere_id, 0],
            min_corners[sphere_id, 1],
            min_corners[sphere_id, 2]
        ])
        p_max = ti.math.vec3([
            max_corners[sphere_id, 0],
            max_corners[sphere_id, 1],
            max_corners[sphere_id, 2]
        ])

        g_min = grid_coord_from_point_ti(p_min, ti.math.vec3([global_min[0], global_min[1], global_min[2]]), voxel_size)
        g_max = grid_coord_from_point_ti(p_max, ti.math.vec3([global_min[0], global_min[1], global_min[2]]), voxel_size)

        g_min.x = ti.max(0, ti.min(g_min.x, grid_size - 1))
        g_min.y = ti.max(0, ti.min(g_min.y, grid_size - 1))
        g_min.z = ti.max(0, ti.min(g_min.z, grid_size - 1))
        g_max.x = ti.max(0, ti.min(g_max.x, grid_size - 1))
        g_max.y = ti.max(0, ti.min(g_max.y, grid_size - 1))
        g_max.z = ti.max(0, ti.min(g_max.z, grid_size - 1))

        extent = g_max - g_min + 1

        idx = start_idx
        for dx in range(extent.x):
            for dy in range(extent.y):
                for dz in range(extent.z):
                    gx = g_min.x + dx
                    gy = g_min.y + dy
                    gz = g_min.z + dz

                    morton = encode_morton_ti(gx, gy, gz)
                    out_morton[idx] = ti.cast(morton, ti.i64)
                    out_sphere_id[idx] = sphere_id
                    idx += 1


# =============================================================================
# Python Module (Layer 1)
# =============================================================================

class CSRGridBuilder(torch.nn.Module):
    """Build CSR grid for Gaussian scene.

    Args:
        config: Builder configuration
    """

    def __init__(self, config: CSRGridBuilderConfig = CSRGridBuilderConfig()):
        super().__init__()
        self.config = config

    def build(self, scene: GaussianScenes) -> CSRGridData:
        """Build CSR grid from Gaussian scene.

        Pipeline:
            1. Compute per-sphere AABB and voxel size
            2. Enumerate sphere-voxel pairs
            3. Sort pairs by morton code
            4. Build two-level lookup table
            5. Precompute sphere data for query

        Args:
            scene: Gaussian scene with positions, scales, rotations

        Returns:
            CSRGridData containing all grid structures
        """
        start_time = time()
        device = scene.position.device

        # Step 1: Compute voxel size and per-sphere AABB
        print(f"[CSRGrid] Computing AABB for {scene.position.shape[0]} spheres...")
        min_corners, max_corners, voxel_size, global_min = self._compute_voxel_size_and_aabb(scene)

        # Step 2: Enumerate sphere-voxel pairs
        print(f"[CSRGrid] Enumerating sphere-voxel pairs...")
        pairs_morton, pairs_sphere_id, oversized_ids = self._enumerate_sphere_voxel_pairs(
            min_corners, max_corners, global_min, voxel_size
        )

        # Step 3: Build two-level lookup table
        print(f"[CSRGrid] Building L1/L2 lookup table...")
        l1_offsets, l2_blocks = self._build_two_level_lookup(
            pairs_morton, pairs_sphere_id
        )

        # Step 4: Precompute sphere data
        print(f"[CSRGrid] Precomputing sphere covariance data...")
        cov_inv, norm_factor = self._precompute_sphere_data(scene)

        # Count unique voxels and spheres per voxel
        unique_mortons, inverse_indices, counts = torch.unique(
            pairs_morton, return_inverse=True, return_counts=True
        )
        num_unique_voxels = len(unique_mortons)

        # Compute spheres per voxel statistics
        spheres_per_voxel = counts.cpu().numpy()

        elapsed = time() - start_time
        print(f"[CSRGrid] Build complete in {elapsed:.2f}s")
        print(f"  - Total pairs: {len(pairs_morton)}")
        print(f"  - Unique voxels: {num_unique_voxels}")
        print(f"  - Oversized spheres: {len(oversized_ids)}")
        print(f"  - Voxel size: {voxel_size:.4f}")
        print(f"  - Spheres per voxel: min={spheres_per_voxel.min()}, "
              f"max={spheres_per_voxel.max()}, mean={spheres_per_voxel.mean():.2f}, "
              f"median={np.median(spheres_per_voxel):.2f}")

        return CSRGridData(
            pairs_morton=pairs_morton,
            pairs_sphere_id=pairs_sphere_id,
            l1_offsets=l1_offsets,
            l2_blocks=l2_blocks,
            oversized_sphere_ids=oversized_ids,
            global_aabb_min=global_min,
            voxel_size=voxel_size,
            grid_dims=(self.config.max_grid_size,) * 3,
            total_pairs=len(pairs_morton),
            num_unique_voxels=num_unique_voxels,
            cov_inv=cov_inv,
            norm_factor=norm_factor,
            sphere_centers=scene.position.contiguous(),
        )

    def _compute_voxel_size_and_aabb(
        self,
        scene: GaussianScenes
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """Compute voxel size and per-sphere AABB.

        Uses adaptive voxel size computation based on configured strategy.

        Args:
            scene: Gaussian scene

        Returns:
            min_corners: [M, 3] per-sphere AABB min
            max_corners: [M, 3] per-sphere AABB max
            voxel_size: float
            global_min: [3] global AABB minimum
        """
        device = scene.position.device
        M = scene.position.shape[0]

        min_corners = torch.zeros((M, 3), dtype=torch.float32, device=device)
        max_corners = torch.zeros((M, 3), dtype=torch.float32, device=device)
        radius = torch.zeros((M, 3), dtype=torch.float32, device=device)

        # Compute per-sphere AABB using Taichi kernel
        gaussian_scene_aabb(
            centers=scene.position.contiguous().float(),
            scales=scene.scales.contiguous().float(),
            quaternions=scene.rotation.contiguous().float(),
            min_corners=min_corners,
            max_corners=max_corners,
            radius=radius,
            confidence_level=self.config.confidence_level,
        )

        # Compute robust global AABB
        global_min, global_max = robust_global_scene_aabb(
            min_corners=min_corners,
            max_corners=max_corners,
            clip_quantile=self.config.global_aabb_clip_quantile,
            padding_factor=self.config.global_aabb_padding_factor,
        )

        # Compute adaptive voxel size based on strategy
        voxel_size = self._compute_adaptive_voxel_size(scene, radius)

        return min_corners, max_corners, voxel_size, global_min

    def _compute_adaptive_voxel_size(
        self,
        scene: GaussianScenes,
        radius: torch.Tensor,
    ) -> float:
        """Compute adaptive voxel size based on scene characteristics.

        Args:
            scene: Gaussian scene
            radius: [M, 3] per-sphere radii

        Returns:
            voxel_size: float
        """
        strategy = self.config.voxel_size_strategy
        factor = self.config.voxel_size_factor

        # Get real scales (exp of log scales stored in hierarchy)
        scales = scene.scales.float()
        real_scales = torch.exp(scales)  # [M, 3]

        if strategy == VoxelSizeStrategy.MEDIAN_RADIUS:
            # Original: use median of max radii
            base_size = torch.median(radius[:, 0]).item()

        elif strategy == VoxelSizeStrategy.SHORT_AXIS_MEDIAN:
            # Use median of shortest axes
            short_axes = real_scales.min(dim=1)[0]  # [M]
            # Filter out extremely small values (numerical noise)
            valid_short = short_axes[short_axes > 1e-6]
            if len(valid_short) > 0:
                base_size = torch.median(valid_short).item()
            else:
                base_size = torch.median(short_axes).item()

        elif strategy == VoxelSizeStrategy.SHORT_AXIS_MODE:
            # Use mode (most common) of shortest axes
            short_axes = real_scales.min(dim=1)[0]  # [M]
            # Filter outliers and find histogram peak
            valid_short = short_axes[short_axes > 1e-6]
            if len(valid_short) > 0:
                # Use histogram to find mode
                hist, bin_edges = torch.histogram(valid_short.cpu(), bins=100)
                mode_idx = hist.argmax().item()
                base_size = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]).item() / 2
            else:
                base_size = torch.median(short_axes).item()

        elif strategy == VoxelSizeStrategy.VOLUME_BASED:
            # Based on median volume (cube root)
            volumes = real_scales.prod(dim=1)  # [M]
            median_volume = torch.median(volumes).item()
            base_size = median_volume ** (1/3)

        elif strategy == VoxelSizeStrategy.PERCENTILE_DENSE:
            # Use lower percentile to focus on dense regions
            short_axes = real_scales.min(dim=1)[0]
            # P10 - only consider small gaussians (dense regions)
            base_size = torch.quantile(short_axes, 0.1).item()

        else:
            # Default to median radius
            base_size = torch.median(radius[:, 0]).item()

        voxel_size = base_size * factor

        # Sanity check: ensure voxel size is reasonable
        if voxel_size < 1e-6:
            print(f"[CSRGrid] Warning: computed voxel size {voxel_size:.2e} too small, using fallback")
            voxel_size = torch.median(radius[:, 0]).item() * factor

        print(f"[CSRGrid] Voxel size strategy: {strategy}")
        print(f"[CSRGrid] Base size: {base_size:.6f}, Factor: {factor}, Final: {voxel_size:.6f}")

        return voxel_size

    def _enumerate_sphere_voxel_pairs(
        self,
        min_corners: torch.Tensor,
        max_corners: torch.Tensor,
        global_min: torch.Tensor,
        voxel_size: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enumerate all sphere-voxel pairs where sphere overlaps voxel.

        Two-pass algorithm:
        1. Count voxels per sphere
        2. Prefix sum to compute offsets
        3. Enumerate pairs without atomic conflicts

        Args:
            min_corners: [M, 3] per-sphere AABB min
            max_corners: [M, 3] per-sphere AABB max
            global_min: [3] global AABB minimum
            voxel_size: Voxel size

        Returns:
            pairs_morton: [total_pairs] sorted morton codes
            pairs_sphere_id: [total_pairs] corresponding sphere IDs
            oversized_ids: [num_oversized] oversized sphere IDs
        """
        device = min_corners.device
        M = min_corners.shape[0]

        # Pass 1: Count voxels per sphere
        voxel_counts = torch.zeros((M,), dtype=torch.int32, device=device)
        out_is_oversized = torch.zeros((M,), dtype=torch.int32, device=device)

        count_voxels_kernel(
            min_corners=min_corners,
            max_corners=max_corners,
            global_min=global_min,
            voxel_size=voxel_size,
            grid_size=self.config.max_grid_size,
            oversized_threshold=self.config.oversized_threshold_voxels,
            voxel_counts=voxel_counts,
            out_is_oversized=out_is_oversized,
        )

        ti.sync()

        # Get oversized sphere IDs
        oversized_ids = torch.nonzero(out_is_oversized, as_tuple=False).squeeze(-1).to(torch.int32)

        # Compute prefix sum for exact memory allocation
        # offsets[i] = sum(voxel_counts[0:i]), offsets[M] = total_pairs
        offsets = torch.cat([
            torch.zeros((1,), dtype=torch.int32, device=device),
            torch.cumsum(voxel_counts, dim=0, dtype=torch.int32)
        ])
        total_pairs = offsets[-1].item()

        # Pass 2: Enumerate pairs using pre-computed offsets (no atomic conflicts)
        out_morton = torch.zeros((total_pairs,), dtype=torch.int64, device=device)
        out_sphere_id = torch.zeros((total_pairs,), dtype=torch.int32, device=device)

        enumerate_pairs_kernel(
            min_corners=min_corners,
            max_corners=max_corners,
            global_min=global_min,
            voxel_size=voxel_size,
            grid_size=self.config.max_grid_size,
            offsets=offsets,
            out_morton=out_morton,
            out_sphere_id=out_sphere_id,
        )

        ti.sync()

        # Sort by morton code
        sorted_indices = torch.argsort(out_morton)
        pairs_morton = out_morton[sorted_indices]
        pairs_sphere_id = out_sphere_id[sorted_indices]

        return pairs_morton, pairs_sphere_id, oversized_ids

    def _build_two_level_lookup(
        self,
        pairs_morton: torch.Tensor,
        pairs_sphere_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Build L1/L2 two-level lookup table (fully vectorized).

        Args:
            pairs_morton: [P] sorted morton codes
            pairs_sphere_id: [P] corresponding sphere IDs

        Returns:
            l1_offsets: [32, 32, 32] offset into L2 or -1
            l2_blocks: List of L2 block tensors
        """
        device = pairs_morton.device
        l1_size = self.config.l1_grid_size
        l2_size = self.config.max_grid_size // l1_size

        # Initialize L1 table with -1 (empty)
        l1_offsets = torch.full((l1_size, l1_size, l1_size), -1, dtype=torch.int32, device=device)
        l2_blocks: List[torch.Tensor] = []

        if len(pairs_morton) == 0:
            return l1_offsets, l2_blocks

        # Decode morton codes to grid coordinates
        grid_coords = self._decode_morton_batch(pairs_morton)

        # Compute L1 indices
        l1_coords = grid_coords // l2_size  # [P, 3]

        # Group pairs by L1 block using vectorized operations
        l1_keys = l1_coords[:, 0] * l1_size * l1_size + l1_coords[:, 1] * l1_size + l1_coords[:, 2]

        # Sort by l1_keys to group
        sorted_order = torch.argsort(l1_keys)
        sorted_keys = l1_keys[sorted_order]
        sorted_morton = pairs_morton[sorted_order]
        sorted_sphere_ids = pairs_sphere_id[sorted_order]

        # Find boundaries between different L1 blocks
        key_changes = torch.cat([
            torch.tensor([True], device=device),
            sorted_keys[1:] != sorted_keys[:-1]
        ])
        block_starts = torch.nonzero(key_changes, as_tuple=False).squeeze(1)
        block_ends = torch.cat([block_starts[1:], torch.tensor([len(sorted_keys)], device=device)])

        # Get unique L1 keys
        unique_l1 = sorted_keys[key_changes]

        # Build L2 blocks using vectorized slicing
        for block_idx, (start, end, l1_key) in enumerate(zip(block_starts, block_ends, unique_l1)):
            # Direct slice (no mask) - much faster
            block_tensor = torch.stack([sorted_morton[start:end], sorted_sphere_ids[start:end]], dim=1)
            l2_blocks.append(block_tensor)

            # Set L1 offset (vectorized indexing)
            l1_x = (l1_key // (l1_size * l1_size)).long()
            l1_y = ((l1_key // l1_size) % l1_size).long()
            l1_z = (l1_key % l1_size).long()
            l1_offsets[l1_x, l1_y, l1_z] = block_idx

        return l1_offsets, l2_blocks

    def plot_spheres_per_voxel_histogram(
        self,
        grid_data: CSRGridData,
        output_path: str = "spheres_per_voxel_histogram.png",
        max_bins: int = 100,
    ):
        """Plot histogram of sphere counts per voxel.

        Args:
            grid_data: CSR grid data from build()
            output_path: Path to save the histogram plot
            max_bins: Maximum number of bins for histogram
        """
        import matplotlib.pyplot as plt

        # Get unique morton codes and their counts
        unique_mortons, counts = torch.unique(
            grid_data.pairs_morton, return_counts=True
        )
        spheres_per_voxel = counts.cpu().numpy()

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Spheres per Voxel Distribution", fontsize=14, fontweight='bold')

        # Plot 1: Full histogram (with capped bins)
        ax = axes[0, 0]
        max_count = int(spheres_per_voxel.max())
        bins = min(max_bins, max_count + 1)
        ax.hist(spheres_per_voxel, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Number of Spheres per Voxel')
        ax.set_ylabel('Number of Voxels')
        ax.set_title(f'Full Distribution (max={max_count})')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = (
            f"Total voxels: {len(spheres_per_voxel)}\n"
            f"Mean: {spheres_per_voxel.mean():.2f}\n"
            f"Median: {np.median(spheres_per_voxel):.2f}\n"
            f"Std: {spheres_per_voxel.std():.2f}\n"
            f"Min: {spheres_per_voxel.min()}\n"
            f"Max: {spheres_per_voxel.max()}"
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        # Plot 2: Zoomed histogram (capped at 95th percentile)
        ax = axes[0, 1]
        p95 = np.percentile(spheres_per_voxel, 95)
        capped_data = spheres_per_voxel[spheres_per_voxel <= p95]
        ax.hist(capped_data, bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax.set_xlabel('Number of Spheres per Voxel')
        ax.set_ylabel('Number of Voxels')
        ax.set_title(f'Distribution (≤95th percentile, ≤{p95:.1f})')
        ax.grid(True, alpha=0.3)

        # Plot 3: Log scale histogram
        ax = axes[1, 0]
        ax.hist(spheres_per_voxel, bins=bins, edgecolor='black', alpha=0.7, color='seagreen')
        ax.set_xlabel('Number of Spheres per Voxel')
        ax.set_ylabel('Number of Voxels (log scale)')
        ax.set_yscale('log')
        ax.set_title('Distribution (Log Scale)')
        ax.grid(True, alpha=0.3, which='both')

        # Plot 4: Cumulative distribution
        ax = axes[1, 1]
        sorted_counts = np.sort(spheres_per_voxel)
        cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
        ax.plot(sorted_counts, cumulative, linewidth=2, color='purple')
        ax.set_xlabel('Number of Spheres per Voxel')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Cumulative Distribution')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, np.percentile(sorted_counts, 99))

        # Add percentile markers
        for p in [50, 90, 95, 99]:
            val = np.percentile(spheres_per_voxel, p)
            ax.axvline(val, color='red', linestyle='--', alpha=0.5)
            ax.annotate(f'{p}%\n({val:.1f})', xy=(val, p), fontsize=8, ha='center')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[CSRGrid] Saved spheres per voxel histogram to: {output_path}")
        plt.close()

    def get_spheres_per_voxel_counts(self, grid_data: CSRGridData) -> np.ndarray:
        """Get the number of spheres per voxel as a numpy array.

        Args:
            grid_data: CSR grid data from build()

        Returns:
            Array of sphere counts per voxel
        """
        _, counts = torch.unique(grid_data.pairs_morton, return_counts=True)
        return counts.cpu().numpy()

    def _decode_morton_batch(self, morton_codes: torch.Tensor) -> torch.Tensor:
        """Decode morton codes to grid coordinates (batch).

        Args:
            morton_codes: [N] morton codes

        Returns:
            grid_coords: [N, 3] grid coordinates
        """
        # Use existing morton_code module function
        from .morton_code import morton_to_grid_coords
        return morton_to_grid_coords(morton_codes)

    def _precompute_sphere_data(
        self,
        scene: GaussianScenes
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute covariance inverse and normalization factors (vectorized).

        Uses batched PyTorch operations for efficiency with large scenes.

        Args:
            scene: Gaussian scene

        Returns:
            cov_inv: [M, 3, 3] inverse covariance matrices
            norm_factor: [M] normalization factors
        """
        device = scene.position.device
        M = scene.position.shape[0]

        # Batch process in chunks to avoid OOM
        chunk_size = 10000
        cov_inv_list = []
        norm_factor_list = []

        scales = scene.scales.float()
        quaternions = scene.rotation.float()

        for start_idx in range(0, M, chunk_size):
            end_idx = min(start_idx + chunk_size, M)
            chunk_size_actual = end_idx - start_idx

            scale_chunk = scales[start_idx:end_idx]  # [B, 3]
            q_chunk = quaternions[start_idx:end_idx]  # [B, 4]

            # Compute covariance matrices in batch
            cov_chunk = self._compute_covariance_batched(scale_chunk, q_chunk)  # [B, 3, 3]

            # Compute inverse and determinant in batch
            try:
                cov_inv_chunk = torch.inverse(cov_chunk)  # [B, 3, 3]
                det_chunk = torch.det(cov_chunk)  # [B]
                norm_factor_chunk = 1.0 / (torch.sqrt((2 * 3.14159265) ** 3 * det_chunk) + EPS_F32)
            except:
                # Fallback for singular matrices
                cov_inv_chunk = torch.eye(3, device=device).unsqueeze(0).repeat(chunk_size_actual, 1, 1)
                cov_inv_chunk = cov_inv_chunk / (scale_chunk.max(dim=1, keepdim=True)[0].unsqueeze(-1) ** 2 + EPS_F32)
                norm_factor_chunk = torch.ones(chunk_size_actual, device=device)

            cov_inv_list.append(cov_inv_chunk)
            norm_factor_list.append(norm_factor_chunk)

        cov_inv = torch.cat(cov_inv_list, dim=0)
        norm_factor = torch.cat(norm_factor_list, dim=0)

        return cov_inv, norm_factor

    def _compute_covariance_batched(
        self,
        scales: torch.Tensor,  # [B, 3]
        quaternions: torch.Tensor,  # [B, 4]
    ) -> torch.Tensor:
        """Compute Gaussian covariance matrices in batch.

        Args:
            scales: Covariance scales [B, 3]
            quaternions: Rotation quaternions [B, 4] (w, x, y, z)

        Returns:
            covariance: [B, 3, 3] covariance matrices
        """
        B = scales.shape[0]
        device = scales.device

        # Build scale matrices S [B, 3, 3]
        S = torch.diag_embed(scales ** 2)  # [B, 3, 3]

        # Normalize quaternions
        q = quaternions
        norm = torch.sqrt(q[:, 0]**2 + q[:, 1]**2 + q[:, 2]**2 + q[:, 3]**2)
        w, x, y, z = q[:, 0]/norm, q[:, 1]/norm, q[:, 2]/norm, q[:, 3]/norm

        # Build rotation matrices R [B, 3, 3]
        R = torch.zeros((B, 3, 3), dtype=torch.float32, device=device)

        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)

        # Covariance: Sigma = R @ S @ R^T [B, 3, 3]
        covariance = R @ S @ R.transpose(-2, -1)

        return covariance

# =============================================================================

if __name__ == "__main__":
    ti.init(arch=ti.cuda if torch.cuda.is_available() else ti.cpu)

    # Create dummy scene
    class DummyScene:
        def __init__(self, num_spheres, device='cuda'):
            self.position = torch.randn(num_spheres, 3, device=device) * 10.0
            self.scales = torch.rand(num_spheres, 3, device=device) * 0.5 + 0.1
            self.rotation = torch.randn(num_spheres, 4, device=device)
            self.rotation = self.rotation / self.rotation.norm(dim=1, keepdim=True)

    scene = DummyScene(1000)

    # Build grid
    config = CSRGridBuilderConfig()
    builder = CSRGridBuilder(config)

    grid_data = builder.build(scene)

    print(f"\nGrid data summary:")
    print(f"  Total pairs: {grid_data.total_pairs}")
    print(f"  Unique voxels: {grid_data.num_unique_voxels}")
    print(f"  Voxel size: {grid_data.voxel_size:.4f}")
    print(f"  Oversized spheres: {len(grid_data.oversized_sphere_ids)}")

    # Plot histogram
    print("\nGenerating spheres per voxel histogram...")
    builder.plot_spheres_per_voxel_histogram(grid_data, output_path="spheres_per_voxel_histogram.png")
