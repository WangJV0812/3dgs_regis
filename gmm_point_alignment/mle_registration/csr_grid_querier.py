"""CSR Grid Querier - Query Top-K spheres for points using CSR grid.

This module provides efficient point-to-sphere queries using the pre-built
CSR grid structure with Taichi GPU acceleration.

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

from .csr_grid_builder import CSRGridData
from .morton_code import grid_coords_to_morton

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CSRGridQuerierConfig:
    """Configuration for CSR Grid Querier.

    Args:
        top_k: Number of top spheres to return per point (default: 8)
        max_candidates_per_point: Maximum candidates to consider (default: 64)
        batch_size: Batch size for memory efficiency (default: 10000)
    """
    top_k: int = 8
    max_candidates_per_point: int = 64
    batch_size: int = 10000


@dataclass
class QueryResult:
    """Container for query results.

    Attributes:
        topk_sphere_ids: Top-K sphere IDs [N, K], int32, -1 for padding
        topk_densities: Top-K Gaussian densities [N, K], float32
    """
    topk_sphere_ids: torch.Tensor
    topk_densities: torch.Tensor


# =============================================================================
# Taichi Kernels (Layer 2)
# =============================================================================

@ti.kernel
def gather_candidates_kernel(
    morton_codes: ti.types.ndarray(ti.i64, 1),
    pairs_morton: ti.types.ndarray(ti.i64, 1),
    pairs_sphere_id: ti.types.ndarray(ti.i32, 1),
    left_indices: ti.types.ndarray(ti.i32, 1),
    right_indices: ti.types.ndarray(ti.i32, 1),
    max_cand: ti.i32,
    candidate_ids: ti.types.ndarray(ti.i32, 2),
    candidate_counts: ti.types.ndarray(ti.i32, 1),
):
    """Gather candidate sphere IDs for each point (parallel)."""
    num_points = morton_codes.shape[0]

    for point_idx in range(num_points):
        left = left_indices[point_idx]
        right = right_indices[point_idx]
        count = right - left

        if count > max_cand:
            count = max_cand

        candidate_counts[point_idx] = count

        for j in range(count):
            candidate_ids[point_idx, j] = pairs_sphere_id[left + j]

        for j in range(count, max_cand):
            candidate_ids[point_idx, j] = -1


@ti.kernel
def compute_densities_kernel(
    points: ti.types.ndarray(ti.f32, 2),
    candidate_ids: ti.types.ndarray(ti.i32, 2),
    sphere_centers: ti.types.ndarray(ti.f32, 2),
    sphere_cov_inv: ti.types.ndarray(ti.f32, 3),
    id_to_idx: ti.types.ndarray(ti.i32, 1),
    densities: ti.types.ndarray(ti.f32, 2),
):
    """Compute Gaussian densities in parallel using Taichi."""
    B = points.shape[0]
    C = candidate_ids.shape[1]

    for point_idx in range(B):
        px = points[point_idx, 0]
        py = points[point_idx, 1]
        pz = points[point_idx, 2]

        for cand_idx in range(C):
            sphere_id = candidate_ids[point_idx, cand_idx]

            if sphere_id < 0:
                densities[point_idx, cand_idx] = 0.0
                continue

            unique_idx = id_to_idx[sphere_id]
            if unique_idx < 0:
                densities[point_idx, cand_idx] = 0.0
                continue

            cx = sphere_centers[unique_idx, 0]
            cy = sphere_centers[unique_idx, 1]
            cz = sphere_centers[unique_idx, 2]

            dx = px - cx
            dy = py - cy
            dz = pz - cz

            c00 = sphere_cov_inv[unique_idx, 0, 0]
            c01 = sphere_cov_inv[unique_idx, 0, 1]
            c02 = sphere_cov_inv[unique_idx, 0, 2]
            c10 = sphere_cov_inv[unique_idx, 1, 0]
            c11 = sphere_cov_inv[unique_idx, 1, 1]
            c12 = sphere_cov_inv[unique_idx, 1, 2]
            c20 = sphere_cov_inv[unique_idx, 2, 0]
            c21 = sphere_cov_inv[unique_idx, 2, 1]
            c22 = sphere_cov_inv[unique_idx, 2, 2]

            tx = c00 * dx + c01 * dy + c02 * dz
            ty = c10 * dx + c11 * dy + c12 * dz
            tz = c20 * dx + c21 * dy + c22 * dz

            mahal = dx * tx + dy * ty + dz * tz
            densities[point_idx, cand_idx] = ti.exp(-0.5 * mahal)


@ti.kernel
def select_topk_kernel(
    densities: ti.types.ndarray(ti.f32, 2),
    candidate_ids: ti.types.ndarray(ti.i32, 2),
    candidate_counts: ti.types.ndarray(ti.i32, 1),
    topk_k: ti.i32,
    topk_ids: ti.types.ndarray(ti.i32, 2),
    topk_scores: ti.types.ndarray(ti.f32, 2),
):
    """Select Top-K candidates for each point (parallel)."""
    num_points = densities.shape[0]
    max_cand = densities.shape[1]

    for point_idx in range(num_points):
        count = candidate_counts[point_idx]

        for k in range(topk_k):
            topk_ids[point_idx, k] = -1
            topk_scores[point_idx, k] = 0.0

        if count == 0:
            continue

        k = ti.min(count, topk_k)

        for ki in range(k):
            max_val = -1.0
            max_idx = -1
            max_sphere_id = -1

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
                candidate_ids[point_idx, max_idx] = -1


# =============================================================================
# Python Module (Layer 1)
# =============================================================================

class CSRGridQuerier:
    """Query Top-K spheres for points using CSR grid with Taichi acceleration.

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
        self.voxel_size = grid_data.voxel_size
        self.global_min = grid_data.global_aabb_min

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

        if point_transform is not None:
            points = self._transform_points(points, point_transform)

        all_topk_ids = []
        all_topk_densities = []

        for i in range(0, N, self.config.batch_size):
            batch_end = min(i + self.config.batch_size, N)
            batch_points = points[i:batch_end]

            batch_result = self._query_batch(batch_points)
            all_topk_ids.append(batch_result.topk_sphere_ids)
            all_topk_densities.append(batch_result.topk_densities)

        topk_ids = torch.cat(all_topk_ids, dim=0)
        topk_densities = torch.cat(all_topk_densities, dim=0)

        elapsed = time() - start_time
        # print(f"[CSRQuerier] Query complete in {elapsed:.3f}s for {N} points")

        return QueryResult(topk_sphere_ids=topk_ids, topk_densities=topk_densities)

    def _transform_points(
        self,
        points: torch.Tensor,
        transform: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transformation to points."""
        ones = torch.ones((points.shape[0], 1), device=points.device)
        points_h = torch.cat([points, ones], dim=1)
        points_transformed = (transform @ points_h.T).T
        return points_transformed[:, :3]

    def _query_batch(self, points: torch.Tensor) -> QueryResult:
        """Process batch query using optimized Taichi kernels."""
        morton_codes = self._points_to_morton(points)
        candidate_ids, candidate_counts = self._lookup_candidates(morton_codes)
        densities = self._compute_densities(points, candidate_ids)
        topk_ids, topk_scores = self._select_topk(densities, candidate_ids, candidate_counts)

        return QueryResult(topk_sphere_ids=topk_ids, topk_densities=topk_scores)

    def _points_to_morton(self, points: torch.Tensor) -> torch.Tensor:
        """Convert points to morton codes."""
        grid_coords = ((points - self.global_min) / self.voxel_size).long()
        max_grid = self.grid_data.grid_dims[0]
        grid_coords = torch.clamp(grid_coords, 0, max_grid - 1)
        return grid_coords_to_morton(grid_coords)

    def _lookup_candidates(
        self,
        morton_codes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lookup candidate spheres using Taichi kernel."""
        device = morton_codes.device
        B = morton_codes.shape[0]
        max_cand = self.config.max_candidates_per_point

        candidate_ids = torch.full((B, max_cand), -1, dtype=torch.int32, device=device)
        candidate_counts = torch.zeros((B,), dtype=torch.int32, device=device)

        pairs_morton = self.grid_data.pairs_morton
        pairs_sphere_id = self.grid_data.pairs_sphere_id

        morton_codes_i64 = morton_codes.long()
        left = torch.searchsorted(pairs_morton, morton_codes_i64, right=False).to(torch.int32)
        right = torch.searchsorted(pairs_morton, morton_codes_i64, right=True).to(torch.int32)

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

    def _compute_densities(
        self,
        points: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian densities using Taichi kernel."""
        device = points.device
        B, C = candidate_ids.shape

        densities = torch.zeros((B, C), dtype=torch.float32, device=device)
        valid_mask = candidate_ids >= 0

        if not valid_mask.any():
            return densities

        all_valid_ids = candidate_ids[valid_mask].unique()
        if len(all_valid_ids) == 0:
            return densities

        unique_centers = self.grid_data.sphere_centers[all_valid_ids]
        unique_cov_inv = self.grid_data.cov_inv[all_valid_ids]

        max_sphere_id = all_valid_ids.max().item() + 1
        id_to_idx = torch.full((max_sphere_id,), -1, dtype=torch.int32, device=device)
        id_to_idx[all_valid_ids] = torch.arange(len(all_valid_ids), device=device, dtype=torch.int32)

        compute_densities_kernel(
            points=points.contiguous(),
            candidate_ids=candidate_ids.contiguous(),
            sphere_centers=unique_centers.contiguous(),
            sphere_cov_inv=unique_cov_inv.contiguous(),
            id_to_idx=id_to_idx.contiguous(),
            densities=densities.contiguous(),
        )

        return densities

    def _select_topk(
        self,
        densities: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select Top-K using Taichi kernel."""
        device = densities.device
        B = densities.shape[0]
        K = self.config.top_k

        topk_ids = torch.full((B, K), -1, dtype=torch.int32, device=device)
        topk_scores = torch.zeros((B, K), dtype=torch.float32, device=device)
        candidate_ids_copy = candidate_ids.clone()

        select_topk_kernel(
            densities=densities,
            candidate_ids=candidate_ids_copy,
            candidate_counts=candidate_counts,
            topk_k=K,
            topk_ids=topk_ids,
            topk_scores=topk_scores,
        )

        return topk_ids, topk_scores
