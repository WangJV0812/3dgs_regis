"""Detailed profiling for CSR Grid Querier stages.

Analyzes time spent in each sub-stage of the query process.
"""

import torch
import taichi as ti
import sys
from pathlib import Path
from time import time
from dataclasses import dataclass, field
from typing import List
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

from misc.hier_IO import load_hier_to_torch
from gmm_point_alignment.csr_grid_builder import CSRGridBuilder
from gmm_point_alignment.csr_grid_querier import CSRGridQuerier, CSRGridQuerierConfig


@dataclass
class StageTiming:
    """Timing for a single stage."""
    name: str
    elapsed_ms: float
    throughput: str = ""
    extra: str = ""


@dataclass
class ProfileReport:
    """Complete profile report."""
    total_time_ms: float
    stages: List[StageTiming] = field(default_factory=list)

    def print(self):
        print(f"\n{'='*80}")
        print(f"Querier Stage-by-Stage Performance")
        print(f"{'='*80}")
        print(f"{'Stage':<40}{'Time (ms)':<15}{'% Total':<10}{'Details':<20}")
        print("-" * 80)

        for s in self.stages:
            pct = (s.elapsed_ms / self.total_time_ms) * 100 if self.total_time_ms > 0 else 0
            print(f"{s.name:<40}{s.elapsed_ms:<15.3f}{pct:<10.2f}{s.extra:<20}")

        print("-" * 80)
        print(f"{'TOTAL':<40}{self.total_time_ms:<15.3f}{100.0:<10.2f}")


@contextmanager
def gpu_sync_timer(name: str, stages: List, extra: str = ""):
    """Timer with GPU synchronization."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time() - start) * 1000
    stages.append(StageTiming(name, elapsed, extra=extra))


class DetailedQuerierProfiler:
    """Profile each stage of the querier in detail."""

    def __init__(self, grid_data, config: CSRGridQuerierConfig):
        self.grid_data = grid_data
        self.config = config
        self.querier = CSRGridQuerier(grid_data, config)

    def profile_full_query(self, points: torch.Tensor) -> ProfileReport:
        """Profile complete query pipeline."""
        stages = []
        device = points.device
        B = points.shape[0]

        print(f"\nProfiling {B:,} query points...")
        print(f"Grid: {self.grid_data.sphere_centers.shape[0]:,} spheres")
        print(f"Top-K: {self.config.top_k}")

        # Stage 1: Transform points (if needed)
        with gpu_sync_timer("1. Point transform", stages):
            points_work = points  # No transform in this test

        # Stage 2: Points to morton codes
        with gpu_sync_timer("2. Points -> Morton codes", stages):
            morton_codes = self._profile_points_to_morton(points_work)

        # Stage 3: Lookup candidates
        with gpu_sync_timer("3. Lookup candidates", stages):
            candidate_ids, candidate_counts = self._profile_lookup_candidates(morton_codes)

        total_candidates = candidate_counts.sum().item()
        avg_cands = total_candidates / B
        stages[-1].extra = f"{total_candidates:,} total, {avg_cands:.1f} avg/pt"

        # Stage 4: Compute densities
        with gpu_sync_timer("4. Compute densities", stages):
            densities = self._profile_compute_densities(points_work, candidate_ids)

        # Stage 5: Select top-k
        with gpu_sync_timer("5. Select Top-K", stages):
            topk_ids, topk_scores = self._profile_select_topk(
                densities, candidate_ids, candidate_counts
            )

        # Calculate total
        total_time = sum(s.elapsed_ms for s in stages)

        return ProfileReport(total_time, stages)

    def _profile_points_to_morton(self, points: torch.Tensor) -> torch.Tensor:
        """Profile points to morton conversion."""
        # Convert to grid coordinates
        grid_coords = ((points - self.querier.global_min) / self.querier.voxel_size).long()
        max_grid = self.grid_data.grid_dims[0]
        grid_coords = torch.clamp(grid_coords, 0, max_grid - 1)

        # Encode to morton
        from gmm_point_alignment.morton_code import grid_coords_to_morton
        return grid_coords_to_morton(grid_coords)

    def _profile_lookup_candidates(self, morton_codes: torch.Tensor):
        """Profile candidate lookup."""
        device = morton_codes.device
        B = morton_codes.shape[0]
        max_cand = self.config.max_candidates_per_point

        candidate_ids = torch.full((B, max_cand), -1, dtype=torch.int32, device=device)
        candidate_counts = torch.zeros((B,), dtype=torch.int32, device=device)

        pairs_morton = self.grid_data.pairs_morton
        pairs_sphere_id = self.grid_data.pairs_sphere_id

        # Batch binary search
        morton_codes_i64 = morton_codes.long()
        left = torch.searchsorted(pairs_morton, morton_codes_i64, right=False)
        right = torch.searchsorted(pairs_morton, morton_codes_i64, right=True)
        counts = torch.minimum((right - left).long(), torch.tensor(max_cand, device=device))

        candidate_counts = counts

        for i in range(B):
            count = counts[i].item()
            if count > 0:
                candidate_ids[i, :count] = pairs_sphere_id[left[i]:left[i] + count]

        return candidate_ids, candidate_counts

    def _profile_compute_densities(self, points: torch.Tensor, candidate_ids: torch.Tensor):
        """Profile density computation with internal breakdown."""
        device = points.device
        B, C = candidate_ids.shape

        densities = torch.zeros((B, C), dtype=torch.float32, device=device)
        valid_mask = candidate_ids >= 0

        # Get unique sphere IDs
        all_valid_ids = candidate_ids[valid_mask].unique()
        if len(all_valid_ids) == 0:
            return densities

        # Gather unique centers and cov_inv
        unique_centers = self.grid_data.sphere_centers[all_valid_ids]
        unique_cov_inv = self.grid_data.cov_inv[all_valid_ids]

        # Build ID mapping
        max_sphere_id = all_valid_ids.max().item() + 1
        id_to_idx = torch.full((max_sphere_id,), -1, dtype=torch.int64, device=device)
        id_to_idx[all_valid_ids] = torch.arange(len(all_valid_ids), device=device)

        candidate_idx = id_to_idx[candidate_ids.clamp(min=0)]

        # Process in chunks
        chunk_size = 512
        for chunk_start in range(0, B, chunk_size):
            chunk_end = min(chunk_start + chunk_size, B)

            chunk_points = points[chunk_start:chunk_end]
            chunk_mask = valid_mask[chunk_start:chunk_end]
            chunk_idx = candidate_idx[chunk_start:chunk_end]

            valid_chunk_idx = chunk_idx.clamp(min=0)

            # Gather
            centers_flat = unique_centers[valid_chunk_idx]
            cov_inv_flat = unique_cov_inv[valid_chunk_idx]

            # Compute
            diff = chunk_points.unsqueeze(1) - centers_flat
            temp = torch.einsum('bcij,bcj->bci', cov_inv_flat, diff)
            mahalanobis = (diff * temp).sum(dim=-1)
            chunk_densities = torch.exp(-0.5 * mahalanobis)
            chunk_densities = chunk_densities * chunk_mask.float()

            densities[chunk_start:chunk_end] = chunk_densities

        return densities

    def _profile_select_topk(self, densities, candidate_ids, candidate_counts):
        """Profile top-k selection."""
        device = densities.device
        B = densities.shape[0]
        K = self.config.top_k

        topk_ids = torch.full((B, K), -1, dtype=torch.int32, device=device)
        topk_scores = torch.zeros((B, K), dtype=torch.float32, device=device)

        for i in range(B):
            count = candidate_counts[i].item()
            if count == 0:
                continue

            valid_densities = densities[i, :count]
            valid_ids = candidate_ids[i, :count]

            k = min(K, count)
            topk_values, topk_indices = torch.topk(valid_densities, k)

            topk_ids[i, :k] = valid_ids[topk_indices]
            topk_scores[i, :k] = topk_values

        return topk_ids, topk_scores


def main():
    """Main profiling entry."""
    print("="*80)
    print("CSR Grid Querier Detailed Performance Profiler")
    print("="*80)

    ti.init(arch=ti.cuda)

    if not torch.cuda.is_available():
        print("WARNING: Using CPU")
    else:
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load scene
    hier_path = Path("data/merged.hier")
    print(f"\nLoading scene...")
    scene_data = load_hier_to_torch(hier_path, torch.device("cuda"))

    # Sample for testing
    sample_size = 50000
    full_scene = scene_data.gaussian_scene
    indices = torch.randperm(full_scene.position.shape[0])[:sample_size]

    class SampledScene:
        pass

    sampled = SampledScene()
    sampled.position = full_scene.position[indices].float()
    sampled.rotation = full_scene.rotation[indices].float()
    sampled.scales = full_scene.scales[indices].float()
    sampled.opacities = full_scene.opacities[indices]
    sampled.shs = full_scene.shs[indices]

    print(f"Sampled: {sample_size:,} spheres")

    # Build grid
    print("\nBuilding grid...")
    grid_data = CSRGridBuilder().build(sampled)

    # Profile different query sizes
    test_sizes = [1000, 5000, 10000]

    for num_points in test_sizes:
        points = torch.randn(num_points, 3, device='cuda') * 5.0

        profiler = DetailedQuerierProfiler(
            grid_data,
            CSRGridQuerierConfig(top_k=8, max_candidates_per_point=64)
        )

        report = profiler.profile_full_query(points)
        report.print()

    print("\n" + "="*80)
    print("Profiling complete!")
    print("="*80)


if __name__ == "__main__":
    main()
