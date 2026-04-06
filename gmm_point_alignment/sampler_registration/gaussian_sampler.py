"""Gaussian scene sampling utilities.

Provides methods to sample point clouds from Gaussian Splatting scenes.
"""

from dataclasses import dataclass
from typing import Literal

import torch

from misc.hier_IO import GaussianScenes


@dataclass
class SampledPointCloud:
    """Sampled point cloud from Gaussian scene."""
    points: torch.Tensor   # (M, 3)
    weights: torch.Tensor  # (M,)


@dataclass
class SamplingConfig:
    """Configuration for Gaussian scene sampling.

    Args:
        mode: Sampling mode - "mean" uses Gaussian centers, "random" adds noise
        samples_per_gaussian: Number of samples per Gaussian for random mode
        opacity_threshold: Minimum opacity for filtering
        scale_threshold: Maximum scale for filtering
        target_num_points: Target number of points (downsample if needed)
        downsample_strategy: "weighted_random" or "fps" (farthest point sampling)
    """
    mode: Literal["mean", "random"] = "mean"
    samples_per_gaussian: int = 1
    opacity_threshold: float = 0.01
    scale_threshold: float = 1e6
    target_num_points: int | None = None
    downsample_strategy: Literal["weighted_random", "fps"] = "weighted_random"


def _farthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Farthest Point Sampling (FPS) on a point cloud."""
    n = points.shape[0]
    if num_samples >= n:
        return torch.arange(n, device=points.device, dtype=torch.long)

    selected = torch.zeros(num_samples, dtype=torch.long, device=points.device)
    distances = torch.full((n,), float("inf"), device=points.device, dtype=points.dtype)

    # Randomly select the first point
    selected[0] = torch.randint(0, n, (1,), device=points.device).item()

    for i in range(1, num_samples):
        last = points[selected[i - 1]].unsqueeze(0)
        dist = torch.norm(points - last, dim=1)
        distances = torch.minimum(distances, dist)
        selected[i] = distances.argmax()

    return selected


def _downsample_to_fixed_num(
    points: torch.Tensor,
    weights: torch.Tensor,
    target_num: int,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample point cloud to a fixed number of points."""
    n = points.shape[0]
    if target_num is None or n <= target_num:
        return points, weights

    if strategy == "weighted_random":
        probs = weights.clamp(min=1e-8)
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, num_samples=target_num, replacement=False)
    elif strategy == "fps":
        idx = _farthest_point_sample(points, target_num)
    else:
        raise ValueError(f"Unsupported downsample strategy: {strategy}")

    return points[idx], weights[idx]


class GaussianSampler:
    """Sampler for generating point clouds from Gaussian scenes."""

    def __init__(self, config: SamplingConfig = None):
        self.config = config or SamplingConfig()

    def sample(self, scene: GaussianScenes) -> SampledPointCloud:
        """Sample point cloud from Gaussian scene.

        Args:
            scene: Gaussian scene

        Returns:
            SampledPointCloud with points and weights
        """
        device = scene.position.device
        n = scene.position.shape[0]

        # Fixed geometry-aware filtering
        opacity_mask = scene.opacities > self.config.opacity_threshold
        scale_mask = scene.scales.max(dim=-1).values < self.config.scale_threshold
        mask = opacity_mask & scale_mask

        positions = scene.position[mask]
        scales = scene.scales[mask]
        opacities = scene.opacities[mask]
        m = positions.shape[0]

        if m == 0:
            empty = torch.empty((0, 3), dtype=torch.float32, device=device)
            return SampledPointCloud(points=empty, weights=empty)

        # Generate points according to mode
        if self.config.mode == "mean":
            points = positions
            weights = opacities
        elif self.config.mode == "random":
            k = max(1, self.config.samples_per_gaussian)
            positions_rep = positions.unsqueeze(1).repeat(1, k, 1)
            scales_rep = scales.unsqueeze(1).repeat(1, k, 1)
            noise = torch.randn_like(positions_rep) * scales_rep
            points = (positions_rep + noise).reshape(-1, 3)
            weights = opacities.unsqueeze(1).repeat(1, k).reshape(-1)
        else:
            raise ValueError(f"Unsupported sampling mode: {self.config.mode}")

        # Fixed-number sampling
        if self.config.target_num_points is not None and self.config.target_num_points > 0:
            points, weights = _downsample_to_fixed_num(
                points, weights, self.config.target_num_points,
                self.config.downsample_strategy
            )

        return SampledPointCloud(points=points, weights=weights)

    def sample_pair(
        self,
        scene_src: GaussianScenes,
        scene_tgt: GaussianScenes,
    ) -> tuple[SampledPointCloud, SampledPointCloud]:
        """Sample point clouds from a pair of scenes.

        Args:
            scene_src: Source Gaussian scene
            scene_tgt: Target Gaussian scene

        Returns:
            Tuple of (source_sampled, target_sampled)
        """
        return self.sample(scene_src), self.sample(scene_tgt)


def sample_gaussian_scene(
    scene: GaussianScenes,
    config: SamplingConfig,
) -> SampledPointCloud:
    """Sample a point cloud from a Gaussian Splatting scene.

    Args:
        scene: The input Gaussian scene.
        config: Sampling configuration.

    Returns:
        SampledPointCloud: Sampled points and associated weights.
    """
    sampler = GaussianSampler(config)
    return sampler.sample(scene)
