"""CSR Grid Querier - Query Top-K spheres for points using CSR grid.

This module provides efficient point-to-sphere queries using the pre-built
CSR grid structure. Supports both PyTorch and Taichi implementations.

Example:
    >>> config = CSRGridQuerierConfig(top_k=8)
    >>> querier = CSRGridQuerier(grid_data, config)
    >>> result = querier.query(points)
    >>> print(f"Top-K spheres: {result.topk_sphere_ids.shape}")
"""

import torch
import taichi as ti
from dataclasses import dataclass
from typing import Optional, Tuple
from time import time

from gmm_point_alignment.csr_grid_builder import CSRGridData
from gmm_point_alignment.morton_code import grid_coords_to_morton

# =============================================================================
# Constants
# =============================================================================

EPS_F32 = 1e-7


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CSRGridQuerierConfig:
    """Configuration for CSR Grid Querier.

    Args:
        top_k: Number of top spheres to return per point (default: 8)
        max_candidates_per_point: Maximum candidates to consider (default: 64)
        use_taichi: Use Taichi kernels for query (default: True)
        use_taichi_densities: Use Taichi kernel for density computation (default: True)
        batch_size: Batch size for memory efficiency (default: 10000)
    """
    top_k: int = 8
    max_candidates_per_point: int = 64
    use_taichi: bool = True
    use_taichi_densities: bool = True
    batch_size: int = 10000


@dataclass
class QueryResult:
    """Container for query results.

    Attributes:
        topk_sphere_ids: Top-K sphere IDs [N, K], int32, -1 for padding
        topk_densities: Top-K Gaussian densities [N, K], float32
        topk_distances: Optional distances [N, K], float32
    """
    topk_sphere_ids: torch.Tensor
    topk_densities: torch.Tensor
    topk_distances: Optional[torch.Tensor] = None


# =============================================================================
# Taichi Functions (Layer 3)
# =============================================================================

@ti.func
def grid_coord_from_point_ti(
    point: ti.math.vec3,
    global_min: ti.math.vec3,
    voxel_size: ti.f32,
) -> ti.math.ivec3:
    """Convert world coordinate to grid coordinate."""
    rel = (point - global_min) / voxel_size
    return ti.cast(ti.floor(rel), ti.i32)


@ti.func
def encode_morton_ti(x: ti.i32, y: ti.i32, z: ti.i32) -> ti.u32:
    """Encode grid coordinates to morton code."""
    def expand_bits(v: ti.u32) -> ti.u32:
        x = ti.cast(v, ti.u32)
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3
        x = (x | (x << 2)) & 0x09249249
        return x

    xi = ti.cast(ti.max(0, ti.min(x, 1023)), ti.u32)
    yi = ti.cast(ti.max(0, ti.min(y, 1023)), ti.u32)
    zi = ti.cast(ti.max(0, ti.min(z, 1023)), ti.u32)
    return (expand_bits(xi) << 2) | (expand_bits(yi) << 1) | expand_bits(zi)


@ti.func
def unnormalized_gaussian_density_ti(
    diff: ti.math.vec3,
    cov_inv: ti.math.mat3,
) -> ti.f32:
    """Compute unnormalized Gaussian density."""
    mahalanobis = diff.dot(cov_inv @ diff)
    return ti.exp(-0.5 * mahalanobis)


@ti.func
def insert_topk_ti(
    sphere_id: ti.i32,
    density: ti.f32,
    topk_ids: ti.types.ndarray(ti.i32, 1),
    topk_scores: ti.types.ndarray(ti.f32, 1),
    k: ti.i32,
):
    """Insert a candidate into sorted Top-K list (descending order)."""
    # Find insertion position
    pos = k - 1
    for i in range(k - 1):
        if density > topk_scores[i]:
            pos = i
            break

    # Shift elements and insert
    if pos < k:
        for i in range(k - 1, pos, -1):
            topk_scores[i] = topk_scores[i - 1]
            topk_ids[i] = topk_ids[i - 1]
        topk_scores[pos] = density
        topk_ids[pos] = sphere_id


# =============================================================================
# Taichi Kernels (Layer 2)
# =============================================================================

@ti.kernel
def gather_candidates_kernel(
    morton_codes: ti.types.ndarray(ti.i64, 1),      # [B]
    pairs_morton: ti.types.ndarray(ti.i64, 1),      # [P]
    pairs_sphere_id: ti.types.ndarray(ti.i32, 1),   # [P]
    left_indices: ti.types.ndarray(ti.i32, 1),      # [B]
    right_indices: ti.types.ndarray(ti.i32, 1),     # [B]
    max_cand: ti.i32,
    candidate_ids: ti.types.ndarray(ti.i32, 2),     # [B, max_cand]
    candidate_counts: ti.types.ndarray(ti.i32, 1),  # [B]
):
    """Gather candidate sphere IDs for each point (parallel).

    Each thread processes one point, copying candidates from pairs array.
    """
    num_points = morton_codes.shape[0]

    for point_idx in range(num_points):
        left = left_indices[point_idx]
        right = right_indices[point_idx]
        count = right - left

        if count > max_cand:
            count = max_cand

        candidate_counts[point_idx] = count

        # Copy candidates
        for j in range(count):
            candidate_ids[point_idx, j] = pairs_sphere_id[left + j]

        # Fill rest with -1
        for j in range(count, max_cand):
            candidate_ids[point_idx, j] = -1


@ti.kernel
def compute_densities_kernel(
    points: ti.types.ndarray(ti.f32, 2),          # [B, 3]
    candidate_ids: ti.types.ndarray(ti.i32, 2),   # [B, C]
    sphere_centers: ti.types.ndarray(ti.f32, 2),  # [S, 3]
    sphere_cov_inv: ti.types.ndarray(ti.f32, 3),  # [S, 3, 3]
    id_to_idx: ti.types.ndarray(ti.i32, 1),       # [max_sphere_id]
    densities: ti.types.ndarray(ti.f32, 2),       # [B, C]
):
    """Compute Gaussian densities in parallel using Taichi.

    Each thread processes one (point, candidate) pair.
    """
    B = points.shape[0]
    C = candidate_ids.shape[1]

    for point_idx in range(B):
        # Load point coordinates
        px = points[point_idx, 0]
        py = points[point_idx, 1]
        pz = points[point_idx, 2]

        for cand_idx in range(C):
            sphere_id = candidate_ids[point_idx, cand_idx]

            if sphere_id < 0:
                densities[point_idx, cand_idx] = 0.0
                continue

            # Map to unique index
            unique_idx = id_to_idx[sphere_id]
            if unique_idx < 0:
                densities[point_idx, cand_idx] = 0.0
                continue

            # Load sphere center
            cx = sphere_centers[unique_idx, 0]
            cy = sphere_centers[unique_idx, 1]
            cz = sphere_centers[unique_idx, 2]

            # Compute diff
            dx = px - cx
            dy = py - cy
            dz = pz - cz

            # Load cov_inv and compute mahalanobis: diff^T @ cov_inv @ diff
            # cov_inv: [3, 3]
            c00 = sphere_cov_inv[unique_idx, 0, 0]
            c01 = sphere_cov_inv[unique_idx, 0, 1]
            c02 = sphere_cov_inv[unique_idx, 0, 2]
            c10 = sphere_cov_inv[unique_idx, 1, 0]
            c11 = sphere_cov_inv[unique_idx, 1, 1]
            c12 = sphere_cov_inv[unique_idx, 1, 2]
            c20 = sphere_cov_inv[unique_idx, 2, 0]
            c21 = sphere_cov_inv[unique_idx, 2, 1]
            c22 = sphere_cov_inv[unique_idx, 2, 2]

            # temp = cov_inv @ diff
            tx = c00 * dx + c01 * dy + c02 * dz
            ty = c10 * dx + c11 * dy + c12 * dz
            tz = c20 * dx + c21 * dy + c22 * dz

            # mahalanobis = diff @ temp
            mahal = dx * tx + dy * ty + dz * tz

            # density = exp(-0.5 * mahal)
            densities[point_idx, cand_idx] = ti.exp(-0.5 * mahal)


@ti.kernel
def select_topk_kernel(
    densities: ti.types.ndarray(ti.f32, 2),         # [B, C]
    candidate_ids: ti.types.ndarray(ti.i32, 2),     # [B, C]
    candidate_counts: ti.types.ndarray(ti.i32, 1),  # [B]
    topk_k: ti.i32,
    topk_ids: ti.types.ndarray(ti.i32, 2),          # [B, K]
    topk_scores: ti.types.ndarray(ti.f32, 2),       # [B, K]
):
    """Select Top-K candidates for each point (parallel).

    Each thread processes one point using simple selection sort for top-k.
    """
    num_points = densities.shape[0]
    max_cand = densities.shape[1]

    for point_idx in range(num_points):
        count = candidate_counts[point_idx]

        # Initialize with -1 and 0
        for k in range(topk_k):
            topk_ids[point_idx, k] = -1
            topk_scores[point_idx, k] = 0.0

        if count == 0:
            continue

        # Simple selection: find top-k
        # Use local arrays for efficiency
        k = ti.min(count, topk_k)

        for ki in range(k):
            max_val = -1.0
            max_idx = -1
            max_sphere_id = -1

            # Find max in remaining candidates
            for j in range(max_cand):
                if candidate_ids[point_idx, j] < 0:
                    continue

                val = densities[point_idx, j]
                if val > max_val:
                    max_val = val
                    max_idx = j
                    max_sphere_id = candidate_ids[point_idx, j]

            if max_idx >= 0:
                topk_ids[point_idx, ki] = max_sphere_id
                topk_scores[point_idx, ki] = max_val
                # Mark as used
                candidate_ids[point_idx, max_idx] = -1


@ti.kernel
def query_topk_kernel(
    # Grid data
    pairs_morton: ti.types.ndarray(ti.i64, 1),
    pairs_sphere_id: ti.types.ndarray(ti.i32, 1),
    l1_offsets: ti.types.ndarray(ti.i32, 3),
    # Sphere data
    sphere_centers: ti.types.ndarray(ti.f32, 2),
    sphere_cov_inv: ti.types.ndarray(ti.f32, 3),
    oversized_ids: ti.types.ndarray(ti.i32, 1),
    # Query data
    points: ti.types.ndarray(ti.f32, 2),
    # Grid params
    global_min: ti.types.ndarray(ti.f32, 1),
    voxel_size: ti.f32,
    l1_size: ti.i32,
    l2_size: ti.i32,
    # Output
    topk_ids: ti.types.ndarray(ti.i32, 2),
    topk_scores: ti.types.ndarray(ti.f32, 2),
    topk_k: ti.i32,
    max_candidates: ti.i32,
):
    """Query Top-K spheres for each point using Taichi.

    Each thread processes one point.
    """
    num_points = points.shape[0]

    for point_idx in range(num_points):
        point = ti.math.vec3([
            points[point_idx, 0],
            points[point_idx, 1],
            points[point_idx, 2],
        ])

        # Compute grid coordinate and morton code
        grid_coord = grid_coord_from_point_ti(point, ti.math.vec3([
            global_min[0], global_min[1], global_min[2]
        ]), voxel_size)

        # Clamp to valid grid
        grid_coord.x = ti.max(0, ti.min(grid_coord.x, l1_size * l2_size - 1))
        grid_coord.y = ti.max(0, ti.min(grid_coord.y, l1_size * l2_size - 1))
        grid_coord.z = ti.max(0, ti.min(grid_coord.z, l1_size * l2_size - 1))

        # Compute L1 index
        l1_coord = grid_coord // l2_size

        # Initialize Top-K arrays
        for k in range(topk_k):
            topk_ids[point_idx, k] = -1
            topk_scores[point_idx, k] = 0.0

        # Get candidates from grid (simplified - full impl needs L2 block lookup)
        # This is a placeholder for the full L1/L2 lookup logic
        # In practice, we'd need to pass L2 blocks as well

        # Also check oversized spheres
        for i in range(oversized_ids.shape[0]):
            sphere_id = oversized_ids[i]
            center = ti.math.vec3([
                sphere_centers[sphere_id, 0],
                sphere_centers[sphere_id, 1],
                sphere_centers[sphere_id, 2],
            ])
            diff = point - center

            # Load covariance inverse
            cov_inv = ti.math.mat3([
                [sphere_cov_inv[sphere_id, 0, 0], sphere_cov_inv[sphere_id, 0, 1], sphere_cov_inv[sphere_id, 0, 2]],
                [sphere_cov_inv[sphere_id, 1, 0], sphere_cov_inv[sphere_id, 1, 1], sphere_cov_inv[sphere_id, 1, 2]],
                [sphere_cov_inv[sphere_id, 2, 0], sphere_cov_inv[sphere_id, 2, 1], sphere_cov_inv[sphere_id, 2, 2]],
            ])

            density = unnormalized_gaussian_density_ti(diff, cov_inv)
            insert_topk_ti(sphere_id, density, topk_ids[point_idx], topk_scores[point_idx], topk_k)


# =============================================================================
# Python Module (Layer 1)
# =============================================================================

class CSRGridQuerier:
    """Query Top-K spheres for points using CSR grid.

    Args:
        grid_data: Pre-built CSR grid data
        config: Querier configuration
    """

    def __init__(
        self,
        grid_data: CSRGridData,
        config: CSRGridQuerierConfig = CSRGridQuerierConfig(),
    ):
        self.grid_data = grid_data
        self.config = config

        # Cache grid parameters
        self.voxel_size = grid_data.voxel_size
        self.global_min = grid_data.global_aabb_min
        self.l1_size = 32
        self.l2_size = 32

    def query(
        self,
        points: torch.Tensor,
        point_transform: Optional[torch.Tensor] = None,
    ) -> QueryResult:
        """Query Top-K spheres for each point.

        Args:
            points: Query points [N, 3]
            point_transform: Optional point transformation [4, 4]

        Returns:
            QueryResult with topk_sphere_ids and topk_densities
        """
        start_time = time()
        device = points.device
        N = points.shape[0]

        # Transform points if needed
        if point_transform is not None:
            points = self._transform_points(points, point_transform)

        # Process in batches for memory efficiency
        all_topk_ids = []
        all_topk_densities = []

        for i in range(0, N, self.config.batch_size):
            batch_end = min(i + self.config.batch_size, N)
            batch_points = points[i:batch_end]

            batch_result = self._query_batch_torch(batch_points)
            all_topk_ids.append(batch_result.topk_sphere_ids)
            all_topk_densities.append(batch_result.topk_densities)

        # Concatenate results
        topk_ids = torch.cat(all_topk_ids, dim=0)
        topk_densities = torch.cat(all_topk_densities, dim=0)

        elapsed = time() - start_time
        print(f"[CSRQuerier] Query complete in {elapsed:.3f}s for {N} points")

        return QueryResult(
            topk_sphere_ids=topk_ids,
            topk_densities=topk_densities,
        )

    def _transform_points(
        self,
        points: torch.Tensor,
        transform: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transformation to points.

        Args:
            points: [N, 3] points
            transform: [4, 4] transformation matrix

        Returns:
            Transformed points [N, 3]
        """
        # Add homogeneous coordinate
        ones = torch.ones((points.shape[0], 1), device=points.device)
        points_h = torch.cat([points, ones], dim=1)

        # Transform
        points_transformed = (transform @ points_h.T).T

        # Remove homogeneous coordinate
        return points_transformed[:, :3]

    def _query_batch_torch(self, points: torch.Tensor) -> QueryResult:
        """PyTorch/Taichi implementation for batch query.

        Args:
            points: [B, 3] query points

        Returns:
            QueryResult
        """
        device = points.device
        B = points.shape[0]
        K = self.config.top_k

        # Step 1: Compute grid coordinates and morton codes
        morton_codes = self._points_to_morton(points)

        # Step 2: Lookup candidates (Taichi or PyTorch)
        if self.config.use_taichi:
            candidate_ids, candidate_counts = self._lookup_candidates_taichi(morton_codes)
        else:
            candidate_ids, candidate_counts = self._lookup_candidates_torch(morton_codes)

        # Step 3: Compute Gaussian densities (Taichi or PyTorch)
        if self.config.use_taichi and self.config.use_taichi_densities:
            densities = self._compute_densities_taichi(points, candidate_ids)
        else:
            densities = self._compute_densities_torch(points, candidate_ids)

        # Step 4: Select Top-K (Taichi or PyTorch)
        if self.config.use_taichi:
            topk_ids, topk_scores = self._select_topk_taichi(
                densities, candidate_ids, candidate_counts
            )
        else:
            topk_ids, topk_scores = self._select_topk_torch(
                densities, candidate_ids, candidate_counts
            )

        return QueryResult(
            topk_sphere_ids=topk_ids,
            topk_densities=topk_scores,
        )

    def _points_to_morton(self, points: torch.Tensor) -> torch.Tensor:
        """Convert points to morton codes.

        Args:
            points: [B, 3] points

        Returns:
            morton_codes: [B] morton codes
        """
        # Convert to grid coordinates
        grid_coords = ((points - self.global_min) / self.voxel_size).long()

        # Clamp to valid range
        max_grid = self.grid_data.grid_dims[0]
        grid_coords = torch.clamp(grid_coords, 0, max_grid - 1)

        # Encode to morton
        return grid_coords_to_morton(grid_coords)

    def _lookup_candidates_torch(
        self,
        morton_codes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lookup candidate spheres for given morton codes (vectorized).

        Uses batched binary search on the sorted pairs array.

        Args:
            morton_codes: [B] morton codes

        Returns:
            candidate_ids: [B, max_candidates] sphere IDs, -1 for padding
            candidate_counts: [B] actual candidate count per point
        """
        device = morton_codes.device
        B = morton_codes.shape[0]
        max_cand = self.config.max_candidates_per_point

        candidate_ids = torch.full((B, max_cand), -1, dtype=torch.int32, device=device)
        candidate_counts = torch.zeros((B,), dtype=torch.int32, device=device)

        pairs_morton = self.grid_data.pairs_morton
        pairs_sphere_id = self.grid_data.pairs_sphere_id

        # Ensure same dtype for searchsorted
        morton_codes_i64 = morton_codes.long()

        # Batch binary search (vectorized)
        left = torch.searchsorted(pairs_morton, morton_codes_i64, right=False)
        right = torch.searchsorted(pairs_morton, morton_codes_i64, right=True)
        counts = torch.minimum((right - left).long(), torch.tensor(max_cand, device=device))

        candidate_counts = counts

        # Gather candidates (vectorized per point)
        for i in range(B):
            count = counts[i].item()
            if count > 0:
                candidate_ids[i, :count] = pairs_sphere_id[left[i]:left[i] + count]

        return candidate_ids, candidate_counts

    def _compute_densities_torch(
        self,
        points: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian densities for point-sphere pairs (fully vectorized).

        Uses fully vectorized PyTorch operations without Python loops.

        Args:
            points: [B, 3] query points
            candidate_ids: [B, C] candidate sphere IDs

        Returns:
            densities: [B, C] Gaussian densities
        """
        device = points.device
        B, C = candidate_ids.shape

        densities = torch.zeros((B, C), dtype=torch.float32, device=device)
        valid_mask = candidate_ids >= 0

        if not valid_mask.any():
            return densities

        # Get unique sphere IDs and gather data
        all_valid_ids = candidate_ids[valid_mask].unique()
        if len(all_valid_ids) == 0:
            return densities

        unique_centers = self.grid_data.sphere_centers[all_valid_ids]
        unique_cov_inv = self.grid_data.cov_inv[all_valid_ids]

        # Build ID mapping
        max_sphere_id = all_valid_ids.max().item() + 1
        id_to_idx = torch.full((max_sphere_id,), -1, dtype=torch.int64, device=device)
        id_to_idx[all_valid_ids] = torch.arange(len(all_valid_ids), device=device)

        # Map candidates to unique indices
        candidate_idx = id_to_idx[candidate_ids.clamp(min=0)]
        valid_candidate_idx = candidate_idx.clamp(min=0)

        # Gather all centers and cov_inv at once [B, C, 3] and [B, C, 3, 3]
        centers_flat = unique_centers[valid_candidate_idx]
        cov_inv_flat = unique_cov_inv[valid_candidate_idx]

        # Compute diff [B, C, 3]
        diff = points.unsqueeze(1) - centers_flat

        # Compute mahalanobis: diff @ cov_inv @ diff.T
        # Reshape for batch matrix multiply: [B*C, 3, 1] and [B*C, 3, 3]
        diff_flat = diff.view(-1, 3, 1)  # [B*C, 3, 1]
        cov_flat = cov_inv_flat.view(-1, 3, 3)  # [B*C, 3, 3]

        # cov @ diff: [B*C, 3, 3] @ [B*C, 3, 1] -> [B*C, 3, 1]
        temp = torch.bmm(cov_flat, diff_flat).squeeze(-1).view(B, C, 3)

        # diff @ temp: sum over last dim
        mahalanobis = (diff * temp).sum(dim=-1)  # [B, C]

        # Compute densities and mask
        densities = torch.exp(-0.5 * mahalanobis)
        densities = densities * valid_mask.float()

        return densities

    def _compute_densities_taichi(
        self,
        points: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian densities using Taichi kernel (parallel).

        Args:
            points: [B, 3] query points
            candidate_ids: [B, C] candidate sphere IDs

        Returns:
            densities: [B, C] Gaussian densities
        """
        device = points.device
        B, C = candidate_ids.shape

        densities = torch.zeros((B, C), dtype=torch.float32, device=device)
        valid_mask = candidate_ids >= 0

        if not valid_mask.any():
            return densities

        # Get unique sphere IDs and gather data
        all_valid_ids = candidate_ids[valid_mask].unique()
        if len(all_valid_ids) == 0:
            return densities

        unique_centers = self.grid_data.sphere_centers[all_valid_ids]
        unique_cov_inv = self.grid_data.cov_inv[all_valid_ids]

        # Build ID mapping
        max_sphere_id = all_valid_ids.max().item() + 1
        id_to_idx = torch.full((max_sphere_id,), -1, dtype=torch.int32, device=device)
        id_to_idx[all_valid_ids] = torch.arange(len(all_valid_ids), device=device, dtype=torch.int32)

        # Launch Taichi kernel
        compute_densities_kernel(
            points=points,
            candidate_ids=candidate_ids,
            sphere_centers=unique_centers,
            sphere_cov_inv=unique_cov_inv,
            id_to_idx=id_to_idx,
            densities=densities,
        )

        return densities

    def _select_topk_torch(
        self,
        densities: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select Top-K highest density associations.

        Args:
            densities: [B, C] densities
            candidate_ids: [B, C] candidate sphere IDs
            candidate_counts: [B] valid candidate counts

        Returns:
            topk_ids: [B, K] top-K sphere IDs
            topk_scores: [B, K] top-K densities
        """
        device = densities.device
        B = densities.shape[0]
        K = self.config.top_k

        topk_ids = torch.full((B, K), -1, dtype=torch.int32, device=device)
        topk_scores = torch.zeros((B, K), dtype=torch.float32, device=device)

        for i in range(B):
            count = candidate_counts[i].item()

            if count == 0:
                continue

            # Get valid candidates
            valid_densities = densities[i, :count]
            valid_ids = candidate_ids[i, :count]

            # Top-K selection
            k = min(K, count)
            topk_values, topk_indices = torch.topk(valid_densities, k)

            topk_ids[i, :k] = valid_ids[topk_indices]
            topk_scores[i, :k] = topk_values

        return topk_ids, topk_scores

    def _lookup_candidates_taichi(
        self,
        morton_codes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lookup candidate spheres using Taichi kernel (parallel).

        Args:
            morton_codes: [B] morton codes

        Returns:
            candidate_ids: [B, max_candidates] sphere IDs, -1 for padding
            candidate_counts: [B] actual candidate count per point
        """
        device = morton_codes.device
        B = morton_codes.shape[0]
        max_cand = self.config.max_candidates_per_point

        # Pre-allocate outputs
        candidate_ids = torch.full((B, max_cand), -1, dtype=torch.int32, device=device)
        candidate_counts = torch.zeros((B,), dtype=torch.int32, device=device)

        pairs_morton = self.grid_data.pairs_morton
        pairs_sphere_id = self.grid_data.pairs_sphere_id

        # Batch binary search (still done in PyTorch - it's fast)
        morton_codes_i64 = morton_codes.long()
        left = torch.searchsorted(pairs_morton, morton_codes_i64, right=False).to(torch.int32)
        right = torch.searchsorted(pairs_morton, morton_codes_i64, right=True).to(torch.int32)

        # Use Taichi kernel for gather (parallel per point)
        gather_candidates_kernel(
            morton_codes=morton_codes_i64,
            pairs_morton=pairs_morton,
            pairs_sphere_id=pairs_sphere_id,
            left_indices=left,
            right_indices=right,
            max_cand=max_cand,
            candidate_ids=candidate_ids,
            candidate_counts=candidate_counts,
        )

        return candidate_ids, candidate_counts

    def _select_topk_taichi(
        self,
        densities: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select Top-K using Taichi kernel (parallel).

        Args:
            densities: [B, C] densities
            candidate_ids: [B, C] candidate sphere IDs
            candidate_counts: [B] valid candidate counts

        Returns:
            topk_ids: [B, K] top-K sphere IDs
            topk_scores: [B, K] top-K densities
        """
        device = densities.device
        B = densities.shape[0]
        K = self.config.top_k

        # Pre-allocate outputs
        topk_ids = torch.full((B, K), -1, dtype=torch.int32, device=device)
        topk_scores = torch.zeros((B, K), dtype=torch.float32, device=device)

        # Need to copy candidate_ids since kernel modifies it
        candidate_ids_copy = candidate_ids.clone()

        # Use Taichi kernel for parallel top-k selection
        select_topk_kernel(
            densities=densities,
            candidate_ids=candidate_ids_copy,
            candidate_counts=candidate_counts,
            topk_k=K,
            topk_ids=topk_ids,
            topk_scores=topk_scores,
        )

        return topk_ids, topk_scores


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    ti.init(arch=ti.cuda if torch.cuda.is_available() else ti.cpu)

    # Create dummy grid data
    class DummyGridData:
        def __init__(self, num_spheres, device='cuda'):
            self.pairs_morton = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64, device=device)
            self.pairs_sphere_id = torch.tensor([0, 1, 0, 2, 1], dtype=torch.int32, device=device)
            self.l1_offsets = torch.full((32, 32, 32), -1, dtype=torch.int32, device=device)
            self.l2_blocks = []
            self.oversized_sphere_ids = torch.tensor([], dtype=torch.int32, device=device)
            self.global_aabb_min = torch.tensor([-10.0, -10.0, -10.0], device=device)
            self.voxel_size = 1.0
            self.grid_dims = (1024, 1024, 1024)
            self.sphere_centers = torch.randn(num_spheres, 3, device=device)
            self.cov_inv = torch.eye(3, device=device).unsqueeze(0).repeat(num_spheres, 1, 1)

    grid_data = DummyGridData(10)

    # Create querier
    config = CSRGridQuerierConfig(top_k=3)
    querier = CSRGridQuerier(grid_data, config)

    # Query
    points = torch.randn(100, 3, device='cuda')
    result = querier.query(points)

    print(f"Top-K sphere IDs shape: {result.topk_sphere_ids.shape}")
    print(f"Top-K densities shape: {result.topk_densities.shape}")
