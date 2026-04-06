"""Detailed GPU performance profiling for CSR Grid.

Analyzes time spent in each phase to identify bottlenecks.
"""

import torch
import taichi as ti
import sys
from pathlib import Path
from time import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from misc.hier_IO import load_hier_to_torch
from gmm_point_alignment.csr_grid_builder import CSRGridBuilder, CSRGridBuilderConfig
from gmm_point_alignment.csr_grid_querier import CSRGridQuerier, CSRGridQuerierConfig


@dataclass
class TimingStats:
    """Timing statistics for a single operation."""
    name: str
    elapsed_ms: float
    throughput: str = ""
    extra_info: str = ""


@dataclass
class ProfileResult:
    """Complete profiling result."""
    total_time_ms: float
    timings: List[TimingStats] = field(default_factory=list)

    def print_report(self):
        print(f"\n{'='*70}")
        print(f"Performance Profile Report")
        print(f"{'='*70}")
        print(f"{'Operation':<30}{'Time (ms)':<15}{'% Total':<10}{'Throughput':<20}")
        print("-" * 70)

        for t in self.timings:
            pct = (t.elapsed_ms / self.total_time_ms) * 100
            print(f"{t.name:<30}{t.elapsed_ms:<15.2f}{pct:<10.1f}{t.throughput:<20}")

        print("-" * 70)
        print(f"{'TOTAL':<30}{self.total_time_ms:<15.2f}{100.0:<10.1f}")


@contextmanager
def gpu_timer(name: str, stats_list: List, item_count: int = 0, item_name: str = ""):
    """Context manager for GPU timing with automatic synchronization."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time() - start) * 1000  # Convert to ms

    throughput = ""
    if item_count > 0:
        throughput = f"{item_count / (elapsed/1000):,.0f} {item_name}/s"

    stats_list.append(TimingStats(name, elapsed, throughput))


def profile_grid_builder(scene, sample_size: int) -> ProfileResult:
    """Profile CSR Grid Builder step by step."""
    print(f"\n{'='*70}")
    print(f"Profiling CSR Grid Builder ({sample_size:,} spheres)")
    print(f"{'='*70}")

    timings = []
    builder = CSRGridBuilder()

    # Step 1: AABB computation
    with gpu_timer("1. AABB + Voxel Size", timings, sample_size, "spheres"):
        min_corners, max_corners, voxel_size, global_min = builder._compute_voxel_size_and_aabb(scene)

    # Step 2: Enumerate pairs
    with gpu_timer("2. Enumerate sphere-voxel pairs", timings):
        pairs_morton, pairs_sphere_id, oversized_ids = builder._enumerate_sphere_voxel_pairs(
            min_corners, max_corners, global_min, voxel_size
        )
    num_pairs = len(pairs_morton)
    timings[-1].extra_info = f"{num_pairs:,} pairs"
    timings[-1].throughput = f"{num_pairs / (timings[-1].elapsed_ms/1000):,.0f} pairs/s"

    # Step 3: Sort pairs (internal to step 2, but let's measure)
    # Actually sorting is done in step 2

    # Step 4: Build L1/L2 lookup
    with gpu_timer("3. Build L1/L2 lookup", timings, num_pairs, "pairs"):
        l1_offsets, l2_blocks = builder._build_two_level_lookup(pairs_morton, pairs_sphere_id)
    timings[-1].extra_info = f"{len(l2_blocks)} L2 blocks"

    # Step 5: Precompute covariance
    with gpu_timer("4. Precompute covariance", timings, sample_size, "spheres"):
        cov_inv, norm_factor = builder._precompute_sphere_data(scene)

    # Calculate total
    total_time = sum(t.elapsed_ms for t in timings)

    result = ProfileResult(total_time, timings)
    result.print_report()

    # Memory stats
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / (1024**2)
        mem_reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"\n[GPU Memory]")
        print(f"  Allocated: {mem_allocated:.2f} MB")
        print(f"  Reserved:  {mem_reserved:.2f} MB")

    return result


def profile_grid_querier(grid_data, num_points: int) -> ProfileResult:
    """Profile CSR Grid Querier."""
    print(f"\n{'='*70}")
    print(f"Profiling CSR Grid Querier ({num_points:,} points)")
    print(f"{'='*70}")

    timings = []

    # Generate query points on GPU
    device = grid_data.sphere_centers.device
    points = torch.randn(num_points, 3, device=device) * 5.0

    # Build querier
    querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=8, batch_size=5000))

    # Step 1: Points to morton
    with gpu_timer("1. Points to morton", timings, num_points, "points"):
        morton_codes = querier._points_to_morton(points)

    # Step 2: Lookup candidates
    with gpu_timer("2. Lookup candidates", timings, num_points, "points"):
        candidate_ids, candidate_counts = querier._lookup_candidates_torch(morton_codes)

    # Step 3: Compute densities
    with gpu_timer("3. Compute densities", timings):
        densities = querier._compute_densities_torch(points, candidate_ids)
    timings[-1].throughput = f"{num_points / (timings[-1].elapsed_ms/1000):,.0f} points/s"

    # Step 4: Select top-k
    with gpu_timer("4. Select Top-K", timings, num_points, "points"):
        topk_ids, topk_scores = querier._select_topk_torch(
            densities, candidate_ids, candidate_counts
        )

    # Calculate total
    total_time = sum(t.elapsed_ms for t in timings)

    result = ProfileResult(total_time, timings)
    result.print_report()

    return result


def main():
    """Main profiling entry."""
    print("Initializing Taichi...")
    ti.init(arch=ti.cuda)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name()}")

    # Load scene
    hier_path = Path("data/merged.hier")
    if not hier_path.exists():
        print(f"Error: {hier_path} not found")
        return

    print(f"\nLoading scene from {hier_path}...")
    scene_data = load_hier_to_torch(hier_path, torch.device("cuda"))
    full_scene = scene_data.gaussian_scene
    print(f"Full scene: {full_scene.position.shape[0]:,} spheres")

    # Test different scales
    test_sizes = [1000, 10000, 50000]

    for size in test_sizes:
        # Sample scene
        indices = torch.randperm(full_scene.position.shape[0])[:size]

        class SampledScene:
            pass

        sampled = SampledScene()
        sampled.position = full_scene.position[indices].float()
        sampled.rotation = full_scene.rotation[indices].float()
        sampled.scales = full_scene.scales[indices].float()
        sampled.opacities = full_scene.opacities[indices]
        sampled.shs = full_scene.shs[indices]

        # Profile builder
        builder_result = profile_grid_builder(sampled, size)

        # Build grid for querier test
        grid_data = CSRGridBuilder().build(sampled)

        # Profile querier
        query_size = min(size, 10000)  # Cap query size
        querier_result = profile_grid_querier(grid_data, query_size)

        # Cleanup
        del grid_data
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("Profiling complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
