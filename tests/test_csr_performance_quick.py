"""Quick performance tests for CSR Grid using sampled real data.

Tests CSR Grid performance on sampled real-world datasets for faster execution.
"""

import pytest
import torch
import taichi as ti
import numpy as np
from pathlib import Path
from time import time
import gc

from misc.hier_IO import load_hier_to_torch, GaussianScenes
from gmm_point_alignment.csr_grid_builder import CSRGridBuilder, CSRGridBuilderConfig
from gmm_point_alignment.csr_grid_querier import CSRGridQuerier, CSRGridQuerierConfig


# Path to test data
DATA_DIR = Path(__file__).parent.parent / "data"
HIER_PATH = DATA_DIR / "merged.hier"


@pytest.fixture(scope="module", autouse=True)
def init_taichi():
    """Initialize Taichi once for all tests."""
    ti.init(arch=ti.cuda if torch.cuda.is_available() else ti.cpu)
    yield


@pytest.fixture(scope="module")
def sampled_scene():
    """Load sampled Gaussian scene from hier file."""
    if not HIER_PATH.exists():
        pytest.skip(f"Hier file not found: {HIER_PATH}")

    print(f"\n[PerfTest] Loading scene from {HIER_PATH}")
    start = time()
    scene_data = load_hier_to_torch(
        hier_path=HIER_PATH,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    elapsed = time() - start

    full_scene = scene_data.gaussian_scene
    num_spheres = full_scene.position.shape[0]
    print(f"[PerfTest] Full scene loaded in {elapsed:.2f}s")
    print(f"[PerfTest] Full scene has {num_spheres:,} spheres")

    # Sample subset for faster testing
    sample_size = min(10000, num_spheres)
    indices = torch.randperm(num_spheres)[:sample_size]

    class SampledScene:
        def __init__(self, original, indices):
            self.position = original.position[indices]
            self.rotation = original.rotation[indices]
            self.scales = original.scales[indices]
            self.opacities = original.opacities[indices]
            self.shs = original.shs[indices]

    sampled = SampledScene(full_scene, indices)
    print(f"[PerfTest] Sampled {sample_size:,} spheres for testing")

    return sampled


class TestCSRGridBuilderQuick:
    """Quick performance tests for CSR Grid Builder."""

    def test_build_grid_performance(self, sampled_scene):
        """Test grid building performance."""
        print(f"\n{'='*60}")
        print("CSR Grid Builder Performance Test")
        print(f"{'='*60}")

        num_spheres = sampled_scene.position.shape[0]
        print(f"\nDataset: {num_spheres:,} spheres")

        config = CSRGridBuilderConfig(
            voxel_size_factor=3.0,
            max_grid_size=1024,
            oversized_threshold_voxels=64,
        )
        builder = CSRGridBuilder(config)

        # Build grid
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start = time()
        grid_data = builder.build(sampled_scene)
        elapsed = time() - start

        print(f"\n[Results] Grid Building:")
        print(f"  - Build time: {elapsed:.2f}s")
        print(f"  - Throughput: {num_spheres/elapsed:,.0f} spheres/s")
        print(f"  - Total pairs: {grid_data.total_pairs:,}")
        print(f"  - Unique voxels: {grid_data.num_unique_voxels:,}")
        print(f"  - Oversized spheres: {len(grid_data.oversized_sphere_ids)}")
        print(f"  - Voxel size: {grid_data.voxel_size:.4f}")

        # Memory estimate
        pairs_memory = grid_data.total_pairs * 12 / (1024**2)
        cov_inv_memory = num_spheres * 3 * 3 * 4 / (1024**2)
        total_memory_mb = pairs_memory + cov_inv_memory + num_spheres * 3 * 4 / (1024**2)

        print(f"  - Estimated memory: {total_memory_mb:.2f} MB")

        # Assertions
        assert elapsed < 30, "Grid build should complete in under 30s"
        assert grid_data.total_pairs > 0, "Should have sphere-voxel pairs"

        # Store for next test
        TestCSRGridBuilderQuick.grid_data = grid_data
        TestCSRGridBuilderQuick.scene = sampled_scene

    def test_voxel_factor_comparison(self, sampled_scene):
        """Compare different voxel size factors."""
        print(f"\n{'='*60}")
        print("Voxel Size Factor Comparison")
        print(f"{'='*60}")

        factors = [1.0, 2.0, 3.0]
        results = []

        for factor in factors:
            config = CSRGridBuilderConfig(voxel_size_factor=factor)
            builder = CSRGridBuilder(config)

            gc.collect()

            start = time()
            grid_data = builder.build(sampled_scene)
            elapsed = time() - start

            results.append({
                'factor': factor,
                'voxel_size': grid_data.voxel_size,
                'total_pairs': grid_data.total_pairs,
                'unique_voxels': grid_data.num_unique_voxels,
                'time': elapsed,
            })

        print(f"\n[Results] Voxel Size Factor Comparison:")
        print(f"{'Factor':<10}{'Voxel Size':<15}{'Total Pairs':<15}{'Unique Voxels':<15}{'Time (s)':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r['factor']:<10.1f}{r['voxel_size']:<15.4f}{r['total_pairs']:<15,}{r['unique_voxels']:<15,}{r['time']:<10.2f}")


class TestCSRGridQuerierQuick:
    """Quick performance tests for CSR Grid Querier."""

    def test_query_performance(self, sampled_scene):
        """Test query performance."""
        print(f"\n{'='*60}")
        print("CSR Grid Querier Performance Test")
        print(f"{'='*60}")

        # Build grid first
        builder = CSRGridBuilder()
        grid_data = builder.build(sampled_scene)

        # Generate random query points
        num_points = 10000
        query_points = torch.randn(num_points, 3, device=sampled_scene.position.device) * 5.0

        print(f"\nDataset:")
        print(f"  - Spheres: {sampled_scene.position.shape[0]:,}")
        print(f"  - Query points: {num_points:,}")

        # Test different top-k values
        topk_values = [1, 4, 8]

        for topk in topk_values:
            querier_config = CSRGridQuerierConfig(top_k=topk, batch_size=2000)
            querier = CSRGridQuerier(grid_data, querier_config)

            gc.collect()

            start = time()
            result = querier.query(query_points)
            elapsed = time() - start

            print(f"\n[Results] Top-K = {topk}:")
            print(f"  - Query time: {elapsed:.3f}s")
            print(f"  - Throughput: {num_points/elapsed:,.0f} points/s")
            print(f"  - Per-point time: {elapsed*1000/num_points:.3f} ms")

    def test_batch_size_comparison(self, sampled_scene):
        """Compare different batch sizes."""
        print(f"\n{'='*60}")
        print("Batch Size Comparison")
        print(f"{'='*60}")

        # Build grid
        builder = CSRGridBuilder()
        grid_data = builder.build(sampled_scene)

        # Generate query points
        num_points = 5000
        query_points = torch.randn(num_points, 3, device=sampled_scene.position.device) * 5.0

        batch_sizes = [500, 1000, 2000, 5000]
        results = []

        for batch_size in batch_sizes:
            querier_config = CSRGridQuerierConfig(top_k=4, batch_size=batch_size)
            querier = CSRGridQuerier(grid_data, querier_config)

            gc.collect()

            start = time()
            result = querier.query(query_points)
            elapsed = time() - start

            results.append({
                'batch_size': batch_size,
                'time': elapsed,
                'throughput': num_points / elapsed,
            })

        print(f"\n[Results] Batch Size Comparison ({num_points:,} points):")
        print(f"{'Batch Size':<15}{'Time (s)':<12}{'Throughput (pts/s)':<20}")
        print("-" * 50)
        for r in results:
            print(f"{r['batch_size']:<15,}{r['time']:<12.3f}{r['throughput']:<20,.0f}")

    def test_csr_vs_naive(self, sampled_scene):
        """Compare CSR vs naive brute-force."""
        print(f"\n{'='*60}")
        print("CSR Grid vs Naive Brute-Force")
        print(f"{'='*60}")

        # Build grid
        builder = CSRGridBuilder()
        grid_data = builder.build(sampled_scene)

        # Sample small subset for naive
        num_points = 500
        query_points = torch.randn(num_points, 3, device=sampled_scene.position.device) * 5.0

        # CSR query
        querier_config = CSRGridQuerierConfig(top_k=4)
        querier = CSRGridQuerier(grid_data, querier_config)

        start = time()
        csr_result = querier.query(query_points)
        csr_time = time() - start

        # Naive brute-force
        start = time()
        num_spheres = sampled_scene.position.shape[0]
        naive_densities = torch.zeros((num_points, num_spheres), device=query_points.device)

        for i in range(num_points):
            diff = query_points[i:i+1] - sampled_scene.position
            dist_sq = (diff ** 2).sum(dim=1)
            naive_densities[i] = torch.exp(-0.5 * dist_sq)

        naive_topk = torch.topk(naive_densities, k=4, dim=1)
        naive_time = time() - start

        speedup = naive_time / csr_time

        print(f"\n[Results] CSR vs Naive ({num_points:,} points, {num_spheres:,} spheres):")
        print(f"  - CSR time: {csr_time:.3f}s")
        print(f"  - Naive time: {naive_time:.3f}s")
        print(f"  - Speedup: {speedup:.1f}x")

        assert speedup > 1, "CSR should be faster than naive"


class TestEndToEndQuick:
    """End-to-end quick tests."""

    def test_full_pipeline(self, sampled_scene):
        """Test complete pipeline."""
        print(f"\n{'='*60}")
        print("End-to-End Pipeline Test")
        print(f"{'='*60}")

        num_spheres = sampled_scene.position.shape[0]
        num_points = 5000
        query_points = torch.randn(num_points, 3, device=sampled_scene.position.device) * 5.0

        print(f"\nDataset: {num_spheres:,} spheres, {num_points:,} points")

        # Phase 1: Build
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        build_start = time()
        builder = CSRGridBuilder()
        grid_data = builder.build(sampled_scene)
        build_time = time() - build_start

        # Phase 2: Query
        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=4))

        query_start = time()
        result = querier.query(query_points)
        query_time = time() - query_start

        total_time = build_time + query_time

        print(f"\n[Results] Pipeline Performance:")
        print(f"  - Build time: {build_time:.2f}s")
        print(f"  - Query time: {query_time:.2f}s")
        print(f"  - Total time: {total_time:.2f}s")
        print(f"  - Build throughput: {num_spheres/build_time:,.0f} spheres/s")
        print(f"  - Query throughput: {num_points/query_time:,.0f} points/s")

        print(f"\n[Amortized] Assuming 100 query iterations:")
        print(f"  - Build (amortized): {build_time/100:.3f}s")
        print(f"  - Query: {query_time:.3f}s")
        print(f"  - Total per iter: {build_time/100 + query_time:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
