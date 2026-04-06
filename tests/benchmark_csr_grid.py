"""Comprehensive benchmark for CSR Grid (PyTorch vs Taichi).

Compares performance on real 3DGS dataset with detailed metrics.
"""

import torch
import taichi as ti
import sys
from pathlib import Path
from time import time
from dataclasses import dataclass, field
from typing import List, Dict
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from misc.hier_IO import load_hier_to_torch
from gmm_point_alignment.csr_grid_builder import CSRGridBuilder, CSRGridBuilderConfig
from gmm_point_alignment.csr_grid_querier import CSRGridQuerier, CSRGridQuerierConfig


@dataclass
class BenchmarkResult:
    """Result for a single benchmark run."""
    name: str
    num_spheres: int
    num_points: int
    build_time_ms: float
    query_time_ms: float
    total_time_ms: float
    throughput: float  # points/s
    memory_mb: float
    use_taichi: bool


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    device: str
    gpu_name: str
    results: List[BenchmarkResult] = field(default_factory=list)

    def print_report(self):
        """Print formatted benchmark report."""
        print("\n" + "="*90)
        print(f"CSR Grid Performance Benchmark Report")
        print(f"Device: {self.device}")
        if self.gpu_name:
            print(f"GPU: {self.gpu_name}")
        print("="*90)

        # Group by num_spheres
        sphere_counts = sorted(set(r.num_spheres for r in self.results))

        for num_spheres in sphere_counts:
            print(f"\n{'─'*90}")
            print(f"Scene: {num_spheres:,} spheres")
            print(f"{'─'*90}")

            # Get results for this sphere count
            sphere_results = [r for r in self.results if r.num_spheres == num_spheres]

            # Group by num_points
            point_counts = sorted(set(r.num_points for r in sphere_results))

            for num_points in point_counts:
                print(f"\n  Query Points: {num_points:,}")
                print(f"  {'Version':<15}{'Build (ms)':<15}{'Query (ms)':<15}{'Total (ms)':<15}{'Throughput':<20}{'Memory (MB)':<15}")
                print(f"  {'─'*85}")

                pt_results = [r for r in sphere_results if r.num_points == num_points]

                for r in sorted(pt_results, key=lambda x: x.use_taichi):
                    version = "Taichi" if r.use_taichi else "PyTorch"
                    print(f"  {version:<15}{r.build_time_ms:<15.2f}{r.query_time_ms:<15.2f}"
                          f"{r.total_time_ms:<15.2f}{r.throughput:<20,.0f}{r.memory_mb:<15.2f}")

                # Calculate speedup
                pt_time = next((r.total_time_ms for r in pt_results if not r.use_taichi), None)
                ti_time = next((r.total_time_ms for r in pt_results if r.use_taichi), None)

                if pt_time and ti_time:
                    speedup = pt_time / ti_time
                    print(f"\n  ⚡ Taichi Speedup: {speedup:.2f}x")

        print("\n" + "="*90)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "device": self.device,
            "gpu_name": self.gpu_name,
            "results": [
                {
                    "name": r.name,
                    "num_spheres": r.num_spheres,
                    "num_points": r.num_points,
                    "build_time_ms": r.build_time_ms,
                    "query_time_ms": r.query_time_ms,
                    "total_time_ms": r.total_time_ms,
                    "throughput": r.throughput,
                    "memory_mb": r.memory_mb,
                    "use_taichi": r.use_taichi,
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


def run_single_benchmark(
    scene,
    num_points: int,
    use_taichi: bool,
    top_k: int = 8,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    device = scene.position.device
    num_spheres = scene.position.shape[0]

    name = f"{'Taichi' if use_taichi else 'PyTorch'}_{num_spheres}sph_{num_points}pts"

    # Clear cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Build grid
    start = time()
    builder = CSRGridBuilder()
    grid_data = builder.build(scene)
    build_time = (time() - start) * 1000  # ms

    # Generate query points
    query_points = torch.randn(num_points, 3, device=device) * 5.0

    # Query
    querier = CSRGridQuerier(
        grid_data,
        CSRGridQuerierConfig(top_k=top_k, use_taichi=use_taichi, batch_size=5000)
    )

    start = time()
    result = querier.query(query_points)
    query_time = (time() - start) * 1000  # ms

    # Memory
    memory_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    # Cleanup
    del grid_data, querier, result
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return BenchmarkResult(
        name=name,
        num_spheres=num_spheres,
        num_points=num_points,
        build_time_ms=build_time,
        query_time_ms=query_time,
        total_time_ms=build_time + query_time,
        throughput=num_points / (query_time / 1000),
        memory_mb=memory_mb,
        use_taichi=use_taichi,
    )


def main():
    """Main benchmark entry."""
    print("="*90)
    print("CSR Grid Performance Benchmark")
    print("PyTorch vs Taichi Implementation")
    print("="*90)

    # Initialize
    print("\nInitializing Taichi...")
    ti.init(arch=ti.cuda)

    device_name = "CPU"
    gpu_name = ""
    if torch.cuda.is_available():
        device_name = "CUDA"
        gpu_name = torch.cuda.get_device_name()
        print(f"GPU: {gpu_name}")

    # Load dataset
    hier_path = Path("data/merged.hier")
    if not hier_path.exists():
        print(f"Error: {hier_path} not found")
        return

    print(f"\nLoading dataset from {hier_path}...")
    scene_data = load_hier_to_torch(hier_path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    full_scene = scene_data.gaussian_scene
    total_spheres = full_scene.position.shape[0]
    print(f"Total scene: {total_spheres:,} spheres")

    # Create benchmark suite
    suite = BenchmarkSuite(device=device_name, gpu_name=gpu_name)

    # Define test configurations
    test_configs = [
        # (num_spheres, num_points_list)
        (1000, [500, 1000, 2000]),
        (5000, [1000, 5000, 10000]),
        (10000, [1000, 5000, 10000, 20000]),
        (50000, [1000, 5000, 10000]),
    ]

    # Run benchmarks
    for num_spheres, point_counts in test_configs:
        if num_spheres > total_spheres:
            continue

        print(f"\n{'='*90}")
        print(f"Benchmarking: {num_spheres:,} spheres")
        print(f"{'='*90}")

        # Sample scene
        indices = torch.randperm(total_spheres)[:num_spheres]
        sampled = SampledScene(full_scene, indices)

        for num_points in point_counts:
            print(f"\n  Query points: {num_points:,}")

            # PyTorch version
            print("    Running PyTorch...", end="", flush=True)
            pt_result = run_single_benchmark(sampled, num_points, use_taichi=False)
            print(f" {pt_result.total_time_ms:.1f}ms")
            suite.results.append(pt_result)

            # Taichi version
            print("    Running Taichi...", end="", flush=True)
            ti_result = run_single_benchmark(sampled, num_points, use_taichi=True)
            print(f" {ti_result.total_time_ms:.1f}ms")
            suite.results.append(ti_result)

            # Show speedup
            speedup = pt_result.total_time_ms / ti_result.total_time_ms
            print(f"    ⚡ Speedup: {speedup:.2f}x")

        # Cleanup
        del sampled
        torch.cuda.empty_cache()

    # Print final report
    suite.print_report()

    # Save results
    output_path = Path("benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(suite.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
