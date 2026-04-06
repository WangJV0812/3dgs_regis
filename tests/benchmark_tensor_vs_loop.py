"""Benchmark: PyTorch Tensor vs Python Loop for density computation.

Compares different implementation strategies.
"""

import torch
import taichi as ti
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_densities_loop(
    points: torch.Tensor,
    candidate_ids: torch.Tensor,
    sphere_centers: torch.Tensor,
    cov_inv: torch.Tensor,
    chunk_size: int = 512
) -> torch.Tensor:
    """Current chunked implementation (Python loop)."""
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


def compute_densities_fully_vectorized(
    points: torch.Tensor,
    candidate_ids: torch.Tensor,
    sphere_centers: torch.Tensor,
    cov_inv: torch.Tensor
) -> torch.Tensor:
    """Fully vectorized implementation (no Python loop)."""
    device = points.device
    B, C = candidate_ids.shape

    # Create output
    densities = torch.zeros((B, C), dtype=torch.float32, device=device)
    valid_mask = candidate_ids >= 0

    # Early exit if no valid candidates
    if not valid_mask.any():
        return densities

    # Gather all centers and cov_inv at once using index_select
    # candidate_ids: [B, C] -> flatten to [B*C]
    flat_candidates = candidate_ids.clamp(min=0).view(-1)  # [B*C]

    # Gather: [B*C, 3] and [B*C, 3, 3]
    gathered_centers = sphere_centers[flat_candidates].view(B, C, 3)
    gathered_cov_inv = cov_inv[flat_candidates].view(B, C, 3, 3)

    # Compute diff: [B, C, 3]
    diff = points.unsqueeze(1) - gathered_centers

    # Mahalanobis: [B, C, 3] @ [B, C, 3, 3] @ [B, C, 3]
    # Using bmm for batch matrix multiply
    # cov_inv: [B, C, 3, 3], diff: [B, C, 3]
    # We want: [B, C, 3] @ [B, C, 3, 3] -> [B, C, 3]

    # Reshape for bmm: [B*C, 3, 3] and [B*C, 3, 1]
    diff_flat = diff.view(-1, 3, 1)  # [B*C, 3, 1]
    cov_flat = gathered_cov_inv.view(-1, 3, 3)  # [B*C, 3, 3]

    # cov @ diff: [B*C, 3, 3] @ [B*C, 3, 1] -> [B*C, 3, 1]
    temp = torch.bmm(cov_flat, diff_flat).squeeze(-1)  # [B*C, 3]
    temp = temp.view(B, C, 3)

    # diff @ temp: sum over last dim
    mahalanobis = (diff * temp).sum(dim=-1)  # [B, C]

    # Compute densities and mask
    densities = torch.exp(-0.5 * mahalanobis)
    densities = densities * valid_mask.float()

    return densities


def benchmark():
    """Run benchmark comparison."""
    print("="*80)
    print("PyTorch Loop vs Tensor Benchmark")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Test configurations
    configs = [
        (1000, 64),   # 1k points, 64 candidates
        (5000, 64),   # 5k points, 64 candidates
        (10000, 64),  # 10k points, 64 candidates
        (50000, 64),  # 50k points, 64 candidates
    ]

    num_spheres = 50000

    for num_points, max_cands in configs:
        print(f"\n{'─'*80}")
        print(f"Test: {num_points:,} points, {max_cands} max candidates, {num_spheres:,} spheres")
        print(f"{'─'*80}")

        # Generate data
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        points = torch.randn(num_points, 3, device=device) * 5.0
        sphere_centers = torch.randn(num_spheres, 3, device=device)
        cov_inv = torch.eye(3, device=device).unsqueeze(0).repeat(num_spheres, 1, 1)

        # Generate candidate IDs (random valid spheres)
        candidate_ids = torch.randint(0, num_spheres, (num_points, max_cands), device=device)
        # Make some invalid
        mask = torch.rand(num_points, max_cands, device=device) > 0.1
        candidate_ids = candidate_ids * mask - (~mask).long()

        # Warmup
        for _ in range(3):
            _ = compute_densities_loop(points, candidate_ids, sphere_centers, cov_inv)
            _ = compute_densities_fully_vectorized(points, candidate_ids, sphere_centers, cov_inv)

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Benchmark Loop version
        times_loop = []
        for _ in range(5):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time()
            result_loop = compute_densities_loop(points, candidate_ids, sphere_centers, cov_inv)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times_loop.append((time() - start) * 1000)

        time_loop = min(times_loop)

        # Benchmark Vectorized version
        times_vec = []
        for _ in range(5):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time()
            result_vec = compute_densities_fully_vectorized(points, candidate_ids, sphere_centers, cov_inv)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times_vec.append((time() - start) * 1000)

        time_vec = min(times_vec)

        # Verify correctness
        max_diff = (result_loop - result_vec).abs().max().item()

        print(f"\n  Chunked (Python loop):")
        print(f"    Time: {time_loop:.2f} ms")
        print(f"    Throughput: {num_points/(time_loop/1000):,.0f} pts/s")

        print(f"\n  Fully Vectorized (no loop):")
        print(f"    Time: {time_vec:.2f} ms")
        print(f"    Throughput: {num_points/(time_vec/1000):,.0f} pts/s")

        print(f"\n  ⚡ Speedup: {time_loop/time_vec:.2f}x")
        print(f"  ✅ Correctness: max diff = {max_diff:.2e}")

        # Memory
        if torch.cuda.is_available():
            mem_loop = result_loop.element_size() * result_loop.nelement() / 1024**2
            mem_vec = result_vec.element_size() * result_vec.nelement() / 1024**2
            print(f"\n  Memory: Loop={mem_loop:.2f}MB, Vec={mem_vec:.2f}MB")


if __name__ == "__main__":
    benchmark()
