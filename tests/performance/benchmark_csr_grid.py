"""Final benchmark for optimized CSR Grid with Taichi.

Tests full pipeline: Build + Query on real 3DGS dataset.
"""

import torch
import taichi as ti
import sys
from pathlib import Path
from time import time
from dataclasses import dataclass, field
from typing import List, Dict
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from misc.hier_IO import load_hier_to_torch
from gmm_point_alignment.csr_grid_builder import CSRGridBuilder
from gmm_point_alignment.csr_grid_querier import CSRGridQuerier, CSRGridQuerierConfig


@dataclass
class BenchmarkResult:
    """Result for a single benchmark run."""
    num_spheres: int
    num_points: int
    build_time_ms: float
    query_time_ms: float
    total_time_ms: float
    throughput: float  # points/s
    memory_mb: float


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    device: str
    gpu_name: str
    results: List[BenchmarkResult] = field(default_factory=list)

    def print_report(self):
        """Print formatted benchmark report."""
        print("\n" + "="*80)
        print(f"CSR Grid Performance Benchmark Report")
        print(f"Device: {self.device}")
        if self.gpu_name:
            print(f"GPU: {self.gpu_name}")
        print("="*80)

        # Group by num_spheres
        sphere_counts = sorted(set(r.num_spheres for r in self.results))

        for num_spheres in sphere_counts:
            print(f"\n{'─'*80}")
            print(f"Scene: {num_spheres:,} spheres")
            print(f"{'─'*80}")

            sphere_results = [r for r in self.results if r.num_spheres == num_spheres]
            point_counts = sorted(set(r.num_points for r in sphere_results))

            for num_points in point_counts:
                r = next(r for r in sphere_results if r.num_points == num_points)
                print(f"\n  Query Points: {num_points:,}")
                print(f"    Build: {r.build_time_ms:.2f} ms")
                print(f"    Query: {r.query_time_ms:.2f} ms")
                print(f"    Total: {r.total_time_ms:.2f} ms")
                print(f"    Throughput: {r.throughput:,.0f} pts/s")

        print("\n" + "="*80)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "device": self.device,
            "gpu_name": self.gpu_name,
            "results": [
                {
                    "num_spheres": r.num_spheres,
                    "num_points": r.num_points,
                    "build_time_ms": r.build_time_ms,
                    "query_time_ms": r.query_time_ms,
                    "total_time_ms": r.total_time_ms,
                    "throughput": r.throughput,
                    "memory_mb": r.memory_mb,
                }
                for r in self.results
            ]
        }


class SampledScene:
    """Sampled scene wrapper."""
    def __init__(self, original, indices):
        self.position = original.position[indices].float()
        self.rotation = original.rotation[indices].float()
        self.scales = original.scales[indices].float()
        self.opacities = original.opacities[indices]
        self.shs = original.shs[indices]


def run_benchmark(scene, num_points: int) -> BenchmarkResult:
    """Run benchmark for a scene configuration."""
    device = scene.position.device
    num_spheres = scene.position.shape[0]

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Build grid
    start = time()
    builder = CSRGridBuilder()
    grid_data = builder.build(scene)
    build_time = (time() - start) * 1000

    # Generate query points
    torch.manual_seed(42)
    query_points = torch.randn(num_points, 3, device=device) * 5.0

    # Query
    querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=8, batch_size=10000))

    # Warmup
    for _ in range(2):
        _ = querier.query(query_points[:min(1000, num_points)])

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Benchmark
    times = []
    for _ in range(5):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time()
        result = querier.query(query_points)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time() - start) * 1000)

    query_time = min(times)

    memory_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    del grid_data, querier, result
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return BenchmarkResult(
        num_spheres=num_spheres,
        num_points=num_points,
        build_time_ms=build_time,
        query_time_ms=query_time,
        total_time_ms=build_time + query_time,
        throughput=num_points / (query_time / 1000),
        memory_mb=memory_mb,
    )


def main():
    """Main benchmark entry."""
    print("="*80)
    print("CSR Grid Performance Benchmark (Taichi Optimized)")
    print("="*80)

    ti.init(arch=ti.cuda)

    device_name = "CPU"
    gpu_name = ""
    if torch.cuda.is_available():
        device_name = "CUDA"
        gpu_name = torch.cuda.get_device_name()
        print(f"GPU: {gpu_name}")

    hier_path = Path("data/merged.hier")
    if not hier_path.exists():
        print(f"Error: {hier_path} not found")
        return

    print(f"\nLoading dataset from {hier_path}...")
    scene_data = load_hier_to_torch(hier_path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    full_scene = scene_data.gaussian_scene
    total_spheres = full_scene.position.shape[0]
    print(f"Total scene: {total_spheres:,} spheres")

    suite = BenchmarkSuite(device=device_name, gpu_name=gpu_name)

    test_configs = [
        (1000, [1000]),
        (5000, [1000, 5000]),
        (10000, [5000, 10000]),
        (50000, [10000]),
    ]

    for num_spheres, point_counts in test_configs:
        if num_spheres > total_spheres:
            continue

        print(f"\n{'='*80}")
        print(f"Benchmarking: {num_spheres:,} spheres")
        print(f"{'='*80}")

        torch.manual_seed(42)
        indices = torch.randperm(total_spheres)[:num_spheres]
        sampled = SampledScene(full_scene, indices)

        for num_points in point_counts:
            print(f"\n  Query points: {num_points:,}", end="", flush=True)
            result = run_benchmark(sampled, num_points)
            print(f" -> Total: {result.total_time_ms:.1f}ms")
            suite.results.append(result)

        del sampled
        torch.cuda.empty_cache()

    suite.print_report()

    output_path = Path("benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(suite.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
