"""Benchmark density computation: Chunk Loop vs Vectorized vs Taichi.

Direct comparison of the three _compute_densities implementations.
"""

import torch
import taichi as ti
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from gmm_point_alignment.csr_grid_querier import (
    CSRGridQuerier, CSRGridQuerierConfig,
    compute_densities_kernel
)
from gmm_point_alignment.csr_grid_builder import CSRGridBuilder, CSRGridData


def compute_densities_chunk_loop(
    points: torch.Tensor,
    candidate_ids: torch.Tensor,
    sphere_centers: torch.Tensor,
    cov_inv: torch.Tensor,
    chunk_size: int = 512
) -> torch.Tensor:
    """Original chunked implementation (Python loop)."""
    device = points.device
    B, C = candidate_ids.shape
    densities = torch.zeros((B, C), dtype=torch.float32, device=device)
    valid_mask = candidate_ids >= 0

    # Get unique ids
    all_valid_ids = candidate_ids[valid_mask].unique()
    unique_centers = sphere_centers[all_valid_ids]
    unique_cov_inv = cov_inv[all_valid_ids]

    max_sphere_id = all_valid_ids.max().item() + 1
    id_to_idx = torch.full((max_sphere_id,), -1, dtype=torch.int64, device=device)
    id_to_idx[all_valid_ids] = torch.arange(len(all_valid_ids), device=device)
    candidate_idx = id_to_idx[candidate_ids.clamp(min=0)]

    # Chunk loop
    for chunk_start in range(0, B, chunk_size):
        chunk_end = min(chunk_start + chunk_size, B)
        chunk_points = points[chunk_start:chunk_end]
        chunk_mask = valid_mask[chunk_start:chunk_end]
        chunk_idx = candidate_idx[chunk_start:chunk_end]

        valid_chunk_idx = chunk_idx.clamp(min=0)
        centers_flat = unique_centers[valid_chunk_idx]
        cov_inv_flat = unique_cov_inv[valid_chunk_idx]

        diff = chunk_points.unsqueeze(1) - centers_flat
        temp = torch.einsum('bcij,bcj->bci', cov_inv_flat, diff)
        mahalanobis = (diff * temp).sum(dim=-1)
        chunk_densities = torch.exp(-0.5 * mahalanobis)
        chunk_densities = chunk_densities * chunk_mask.float()

        densities[chunk_start:chunk_end] = chunk_densities

    return densities


def compute_densities_vectorized(
    points: torch.Tensor,
    candidate_ids: torch.Tensor,
    sphere_centers: torch.Tensor,
    cov_inv: torch.Tensor,
) -> torch.Tensor:
    """Fully vectorized implementation (no Python loop)."""
    device = points.device
    B, C = candidate_ids.shape

    densities = torch.zeros((B, C), dtype=torch.float32, device=device)
    valid_mask = candidate_ids >= 0

    if not valid_mask.any():
        return densities

    all_valid_ids = candidate_ids[valid_mask].unique()
    if len(all_valid_ids) == 0:
        return densities

    unique_centers = sphere_centers[all_valid_ids]
    unique_cov_inv = cov_inv[all_valid_ids]

    max_sphere_id = all_valid_ids.max().item() + 1
    id_to_idx = torch.full((max_sphere_id,), -1, dtype=torch.int64, device=device)
    id_to_idx[all_valid_ids] = torch.arange(len(all_valid_ids), device=device)

    candidate_idx = id_to_idx[candidate_ids.clamp(min=0)]
    valid_candidate_idx = candidate_idx.clamp(min=0)

    centers_flat = unique_centers[valid_candidate_idx]
    cov_inv_flat = unique_cov_inv[valid_candidate_idx]

    diff = points.unsqueeze(1) - centers_flat

    diff_flat = diff.view(-1, 3, 1)
    cov_flat = cov_inv_flat.view(-1, 3, 3)

    temp = torch.bmm(cov_flat, diff_flat).squeeze(-1).view(B, C, 3)
    mahalanobis = (diff * temp).sum(dim=-1)

    densities = torch.exp(-0.5 * mahalanobis)
    densities = densities * valid_mask.float()

    return densities


def compute_densities_taichi(
    points: torch.Tensor,
    candidate_ids: torch.Tensor,
    sphere_centers: torch.Tensor,
    cov_inv: torch.Tensor,
) -> torch.Tensor:
    """Taichi kernel implementation."""
    device = points.device
    B, C = candidate_ids.shape

    densities = torch.zeros((B, C), dtype=torch.float32, device=device)
    valid_mask = candidate_ids >= 0

    if not valid_mask.any():
        return densities

    all_valid_ids = candidate_ids[valid_mask].unique()
    if len(all_valid_ids) == 0:
        return densities

    unique_centers = sphere_centers[all_valid_ids]
    unique_cov_inv = cov_inv[all_valid_ids]

    max_sphere_id = all_valid_ids.max().item() + 1
    id_to_idx = torch.full((max_sphere_id,), -1, dtype=torch.int32, device=device)
    id_to_idx[all_valid_ids] = torch.arange(len(all_valid_ids), device=device, dtype=torch.int32)

    compute_densities_kernel(
        points=points,
        candidate_ids=candidate_ids.to(torch.int32),
        sphere_centers=unique_centers,
        sphere_cov_inv=unique_cov_inv,
        id_to_idx=id_to_idx,
        densities=densities,
    )

    return densities


def benchmark():
    """Run benchmark comparison."""
    print("="*90)
    print("Density Computation Benchmark: Chunk Loop vs Vectorized vs Taichi")
    print("="*90)

    ti.init(arch=ti.cuda)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Test configurations
    configs = [
        (1000, 64, 10000),   # 1k points, 64 candidates, 10k spheres
        (5000, 64, 50000),   # 5k points, 64 candidates, 50k spheres
        (10000, 64, 50000),  # 10k points, 64 candidates, 50k spheres
        (50000, 64, 50000),  # 50k points, 64 candidates, 50k spheres
    ]

    for num_points, max_cands, num_spheres in configs:
        print(f"\n{'─'*90}")
        print(f"Test: {num_points:,} points, {max_cands} max candidates, {num_spheres:,} spheres")
        print(f"{'─'*90}")

        # Generate data
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        points = torch.randn(num_points, 3, device=device) * 5.0
        sphere_centers = torch.randn(num_spheres, 3, device=device)
        cov_inv = torch.eye(3, device=device).unsqueeze(0).repeat(num_spheres, 1, 1)

        # Generate candidate IDs (random valid spheres)
        candidate_ids = torch.randint(0, num_spheres, (num_points, max_cands), device=device)
        mask = torch.rand(num_points, max_cands, device=device) > 0.1
        candidate_ids = candidate_ids * mask - (~mask).long()

        # Warmup
        for _ in range(3):
            _ = compute_densities_chunk_loop(points, candidate_ids, sphere_centers, cov_inv)
            _ = compute_densities_vectorized(points, candidate_ids, sphere_centers, cov_inv)
            _ = compute_densities_taichi(points, candidate_ids, sphere_centers, cov_inv)

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Benchmark Chunk Loop
        times_chunk = []
        for _ in range(5):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time()
            result_chunk = compute_densities_chunk_loop(points, candidate_ids, sphere_centers, cov_inv)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times_chunk.append((time() - start) * 1000)

        time_chunk = min(times_chunk)

        # Benchmark Vectorized
        times_vec = []
        for _ in range(5):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time()
            result_vec = compute_densities_vectorized(points, candidate_ids, sphere_centers, cov_inv)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times_vec.append((time() - start) * 1000)

        time_vec = min(times_vec)

        # Benchmark Taichi
        times_taichi = []
        for _ in range(5):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time()
            result_taichi = compute_densities_taichi(points, candidate_ids, sphere_centers, cov_inv)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times_taichi.append((time() - start) * 1000)

        time_taichi = min(times_taichi)

        # Verify correctness
        max_diff_vec = (result_chunk - result_vec).abs().max().item()
        max_diff_taichi = (result_chunk - result_taichi).abs().max().item()

        print(f"\n  Chunk Loop (512 chunk):")
        print(f"    Time: {time_chunk:.2f} ms")
        print(f"    Throughput: {num_points/(time_chunk/1000):,.0f} pts/s")

        print(f"\n  Fully Vectorized:")
        print(f"    Time: {time_vec:.2f} ms")
        print(f"    Throughput: {num_points/(time_vec/1000):,.0f} pts/s")

        print(f"\n  Taichi Kernel:")
        print(f"    Time: {time_taichi:.2f} ms")
        print(f"    Throughput: {num_points/(time_taichi/1000):,.0f} pts/s")

        print(f"\n  Speedups:")
        print(f"    Vectorized vs Chunk: {time_chunk/time_vec:.2f}x")
        print(f"    Taichi vs Chunk: {time_chunk/time_taichi:.2f}x")
        print(f"    Taichi vs Vectorized: {time_vec/time_taichi:.2f}x")

        print(f"\n  Correctness:")
        print(f"    Vectorized max diff: {max_diff_vec:.2e}")
        print(f"    Taichi max diff: {max_diff_taichi:.2e}")

    print("\n" + "="*90)


if __name__ == "__main__":
    benchmark()
