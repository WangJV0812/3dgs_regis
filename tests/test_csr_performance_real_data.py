"""Performance tests for CSR Grid using real 3DGS data.

Tests CSR Grid Builder and Querier performance on real-world datasets.
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
POINTCLOUD_PATH = DATA_DIR / "points3D.ply"


def read_ply_xyz(ply_path: Path, device='cpu') -> torch.Tensor:
    """Read PLY file and extract XYZ coordinates."""
    import struct

    with open(ply_path, 'rb') as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse element vertex count
        num_vertices = None
        properties = []
        for line in header_lines:
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[2])
            elif line.startswith("property "):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))

        # Calculate offset to xyz
        format_map = {
            'char': 1, 'uchar': 1,
            'short': 2, 'ushort': 2,
            'int': 4, 'uint': 4,
            'float': 4, 'double': 8
        }

        xyz_offset = 0
        xyz_stride = 0
        for name, ptype in properties:
            size = format_map.get(ptype, 4)
            if name in ['x', 'y', 'z']:
                xyz_stride += size
            elif xyz_stride == 0:  # Before xyz
                xyz_offset += size

        # Read vertex data
        vertex_size = sum(format_map.get(ptype, 4) for _, ptype in properties)

        points = []
        for i in range(num_vertices):
            f.seek(len(b''.join(l.encode() + b'\n' for l in header_lines)) + i * vertex_size + xyz_offset)
            x, y, z = struct.unpack('<fff', f.read(12))
            points.append([x, y, z])

        return torch.tensor(points, dtype=torch.float32, device=device)


@pytest.fixture(scope="module", autouse=True)
def init_taichi():
    """Initialize Taichi once for all tests."""
    ti.init(arch=ti.cuda if torch.cuda.is_available() else ti.cpu)
    yield


@pytest.fixture(scope="module")
def real_scene():
    """Load real Gaussian scene from hier file."""
    if not HIER_PATH.exists():
        pytest.skip(f"Hier file not found: {HIER_PATH}")

    print(f"\n[PerfTest] Loading scene from {HIER_PATH}")
    start = time()
    scene = load_hier_to_torch(hier_path=HIER_PATH, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    elapsed = time() - start
    print(f"[PerfTest] Scene loaded in {elapsed:.2f}s")
    print(f"[PerfTest] Scene has {scene.gaussian_scene.position.shape[0]} spheres")
    return scene.gaussian_scene


@pytest.fixture(scope="module")
def real_pointcloud():
    """Load real point cloud from PLY file."""
    if not POINTCLOUD_PATH.exists():
        pytest.skip(f"Pointcloud file not found: {POINTCLOUD_PATH}")

    print(f"\n[PerfTest] Loading pointcloud from {POINTCLOUD_PATH}")
    start = time()
    points = read_ply_xyz(POINTCLOUD_PATH, device='cuda' if torch.cuda.is_available() else 'cpu')
    elapsed = time() - start
    print(f"[PerfTest] Pointcloud loaded in {elapsed:.2f}s")
    print(f"[PerfTest] Pointcloud has {points.shape[0]} points")
    return points


class TestCSRGridBuilderPerformance:
    """Performance tests for CSR Grid Builder on real data."""

    def test_build_grid_full_scene(self, real_scene):
        """Test grid building performance on full scene."""
        print(f"\n{'='*60}")
        print("CSR Grid Builder Performance Test - Full Scene")
        print(f"{'='*60}")

        config = CSRGridBuilderConfig(
            voxel_size_factor=3.0,
            max_grid_size=1024,
            oversized_threshold_voxels=64,
        )
        builder = CSRGridBuilder(config)

        # Build grid and measure time
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start = time()
        grid_data = builder.build(real_scene)
        elapsed = time() - start

        num_spheres = real_scene.position.shape[0]

        print(f"\n[Results] Grid Building Performance:")
        print(f"  - Scene size: {num_spheres:,} spheres")
        print(f"  - Build time: {elapsed:.2f}s")
        print(f"  - Throughput: {num_spheres/elapsed:,.0f} spheres/s")
        print(f"  - Total pairs: {grid_data.total_pairs:,}")
        print(f"  - Unique voxels: {grid_data.num_unique_voxels:,}")
        print(f"  - Oversized spheres: {len(grid_data.oversized_sphere_ids)}")
        print(f"  - Voxel size: {grid_data.voxel_size:.4f}")

        # Memory estimate
        pairs_memory = grid_data.total_pairs * (8 + 4)  # morton (int64) + sphere_id (int32)
        l1_memory = 32 * 32 * 32 * 4  # int32
        cov_inv_memory = num_spheres * 3 * 3 * 4  # float32
        norm_factor_memory = num_spheres * 4
        total_memory_mb = (pairs_memory + l1_memory + cov_inv_memory + norm_factor_memory) / (1024 * 1024)

        print(f"  - Estimated memory: {total_memory_mb:.2f} MB")

        # Assertions for sanity checks
        assert elapsed < 60, "Grid build should complete in under 60s"
        assert grid_data.total_pairs > 0, "Should have sphere-voxel pairs"

        # Store for later tests
        TestCSRGridBuilderPerformance.grid_data = grid_data

    def test_build_grid_different_voxel_factors(self, real_scene):
        """Test grid building with different voxel size factors."""
        print(f"\n{'='*60}")
        print("CSR Grid Builder - Voxel Size Factor Comparison")
        print(f"{'='*60}")

        factors = [1.0, 2.0, 3.0, 5.0]
        results = []

        for factor in factors:
            config = CSRGridBuilderConfig(voxel_size_factor=factor)
            builder = CSRGridBuilder(config)

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start = time()
            grid_data = builder.build(real_scene)
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


class TestCSRGridQuerierPerformance:
    """Performance tests for CSR Grid Querier on real data."""

    def test_query_performance_full_pointcloud(self, real_scene, real_pointcloud):
        """Test query performance on full point cloud."""
        print(f"\n{'='*60}")
        print("CSR Grid Querier Performance Test - Full Pointcloud")
        print(f"{'='*60}")

        # First build grid
        builder_config = CSRGridBuilderConfig()
        builder = CSRGridBuilder(builder_config)
        grid_data = builder.build(real_scene)

        # Test different top-k values
        topk_values = [1, 4, 8, 16]

        print(f"\nPointcloud size: {real_pointcloud.shape[0]:,} points")

        for topk in topk_values:
            querier_config = CSRGridQuerierConfig(top_k=topk, batch_size=10000)
            querier = CSRGridQuerier(grid_data, querier_config)

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start = time()
            result = querier.query(real_pointcloud)
            elapsed = time() - start

            num_points = real_pointcloud.shape[0]

            print(f"\n[Results] Top-K = {topk}:")
            print(f"  - Query time: {elapsed:.3f}s")
            print(f"  - Throughput: {num_points/elapsed:,.0f} points/s")
            print(f"  - Per-point time: {elapsed*1000/num_points:.3f} ms")

    def test_query_performance_different_batch_sizes(self, real_scene, real_pointcloud):
        """Test query performance with different batch sizes."""
        print(f"\n{'='*60}")
        print("CSR Grid Querier - Batch Size Comparison")
        print(f"{'='*60}")

        # Build grid once
        builder = CSRGridBuilder()
        grid_data = builder.build(real_scene)

        # Sample subset for faster testing
        sample_size = min(50000, real_pointcloud.shape[0])
        sample_points = real_pointcloud[:sample_size]

        batch_sizes = [1000, 5000, 10000, 20000, 50000, 100000, 200000]
        results = []

        for batch_size in batch_sizes:
            querier_config = CSRGridQuerierConfig(top_k=8, batch_size=batch_size)
            querier = CSRGridQuerier(grid_data, querier_config)

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start = time()
            result = querier.query(sample_points)
            elapsed = time() - start

            results.append({
                'batch_size': batch_size,
                'time': elapsed,
                'throughput': sample_size / elapsed,
            })

        print(f"\n[Results] Batch Size Comparison (sample: {sample_size:,} points):")
        print(f"{'Batch Size':<15}{'Time (s)':<12}{'Throughput (pts/s)':<20}")
        print("-" * 50)
        for r in results:
            print(f"{r['batch_size']:<15,}{r['time']:<12.3f}{r['throughput']:<20,.0f}")

    def test_query_vs_naive_comparison(self, real_scene, real_pointcloud):
        """Compare CSR query vs naive brute-force query."""
        print(f"\n{'='*60}")
        print("CSR Grid vs Naive Brute-Force Comparison")
        print(f"{'='*60}")

        # Build grid
        builder = CSRGridBuilder()
        grid_data = builder.build(real_scene)

        # Sample small subset for naive comparison
        sample_size = min(1000, real_pointcloud.shape[0])
        sample_points = real_pointcloud[:sample_size]

        # CSR query
        querier_config = CSRGridQuerierConfig(top_k=8)
        querier = CSRGridQuerier(grid_data, querier_config)

        start = time()
        csr_result = querier.query(sample_points)
        csr_time = time() - start

        # Naive brute-force query (compute distance to ALL spheres)
        start = time()
        num_points = sample_points.shape[0]
        num_spheres = real_scene.position.shape[0]

        # Compute distances for all point-sphere pairs
        naive_densities = torch.zeros((num_points, num_spheres), device=sample_points.device)
        for i in range(num_points):
            diff = sample_points[i:i+1] - real_scene.position  # [1, 3] - [N, 3] = [N, 3]
            # Simple squared distance as proxy
            dist_sq = (diff ** 2).sum(dim=1)
            naive_densities[i] = torch.exp(-0.5 * dist_sq)

        # Get top-k
        naive_topk = torch.topk(naive_densities, k=8, dim=1)
        naive_time = time() - start

        speedup = naive_time / csr_time

        print(f"\n[Results] CSR Grid vs Naive (sample: {sample_size:,} points):")
        print(f"  - CSR Grid time: {csr_time:.3f}s")
        print(f"  - Naive time: {naive_time:.3f}s")
        print(f"  - Speedup: {speedup:.1f}x")

        # Sanity check: CSR should be faster
        assert speedup > 1, "CSR Grid should be faster than naive"


class TestEndToEndPerformance:
    """End-to-end performance tests."""

    def test_full_pipeline(self, real_scene, real_pointcloud):
        """Test complete pipeline: build + query."""
        print(f"\n{'='*60}")
        print("End-to-End Pipeline Performance Test")
        print(f"{'='*60}")

        num_spheres = real_scene.position.shape[0]
        num_points = real_pointcloud.shape[0]

        print(f"\nDataset:")
        print(f"  - Spheres: {num_spheres:,}")
        print(f"  - Points: {num_points:,}")

        # Phase 1: Build Grid
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        build_start = time()
        builder = CSRGridBuilder()
        grid_data = builder.build(real_scene)
        build_time = time() - build_start

        # Phase 2: Query
        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=8))

        query_start = time()
        result = querier.query(real_pointcloud)
        query_time = time() - query_start

        total_time = build_time + query_time

        print(f"\n[Results] Pipeline Performance:")
        print(f"  - Build time: {build_time:.2f}s")
        print(f"  - Query time: {query_time:.2f}s")
        print(f"  - Total time: {total_time:.2f}s")
        print(f"  - Build throughput: {num_spheres/build_time:,.0f} spheres/s")
        print(f"  - Query throughput: {num_points/query_time:,.0f} points/s")

        # Per-operation breakdown
        print(f"\n[Breakdown] Assuming 100 iterations:")
        print(f"  - Build (amortized): {build_time/100:.3f}s per iter")
        print(f"  - Query: {query_time:.3f}s per iter")
        print(f"  - Total per iter: {build_time/100 + query_time:.3f}s")

    def test_memory_usage(self, real_scene, real_pointcloud):
        """Test memory usage of CSR Grid."""
        print(f"\n{'='*60}")
        print("Memory Usage Analysis")
        print(f"{'='*60}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Build grid
        builder = CSRGridBuilder()
        grid_data = builder.build(real_scene)

        if torch.cuda.is_available():
            build_memory = torch.cuda.max_memory_allocated() / (1024**2)
            torch.cuda.reset_peak_memory_stats()
        else:
            build_memory = 0

        # Query
        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=8))
        result = querier.query(real_pointcloud)

        if torch.cuda.is_available():
            query_memory = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            query_memory = 0

        # Calculate theoretical sizes
        num_spheres = real_scene.position.shape[0]
        num_pairs = grid_data.total_pairs

        pairs_size = num_pairs * 12 / (1024**2)  # morton (8) + sphere_id (4)
        l1_table_size = 32 * 32 * 32 * 4 / (1024**2)
        cov_inv_size = num_spheres * 3 * 3 * 4 / (1024**2)
        norm_factor_size = num_spheres * 4 / (1024**2)
        centers_size = num_spheres * 3 * 4 / (1024**2)

        theoretical_total = pairs_size + l1_table_size + cov_inv_size + norm_factor_size + centers_size

        print(f"\n[Results] Memory Usage:")
        print(f"  - Pairs array: {pairs_size:.2f} MB ({num_pairs:,} pairs)")
        print(f"  - L1 lookup table: {l1_table_size:.2f} MB")
        print(f"  - Covariance inverse: {cov_inv_size:.2f} MB")
        print(f"  - Norm factors: {norm_factor_size:.2f} MB")
        print(f"  - Sphere centers: {centers_size:.2f} MB")
        print(f"  - Theoretical total: {theoretical_total:.2f} MB")

        if torch.cuda.is_available():
            print(f"  - Actual GPU peak (build): {build_memory:.2f} MB")
            print(f"  - Actual GPU peak (query): {query_memory:.2f} MB")

        # Compare with naive approach
        naive_pairs = num_spheres * 1000  # Assume 1000 voxels per sphere
        naive_memory = naive_pairs * 12 / (1024**2)
        print(f"\n[Comparison] Naive dense grid would need: ~{naive_memory:.0f} MB")
        print(f"[Comparison] CSR saves: {(naive_memory - theoretical_total)/naive_memory*100:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
