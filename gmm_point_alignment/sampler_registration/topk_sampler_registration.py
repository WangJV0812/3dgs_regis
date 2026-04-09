"""Top-K Sampler-based point cloud registration.

This module provides a registration method that:
1. Queries Top-K spheres for each input point using CSR Grid
2. Samples points only from these Top-K spheres (localized sampling)
3. Registers the input point cloud to the sampled points using ICP

This combines the efficiency of MLE's Top-K querying with traditional ICP.
"""

from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

import torch
import numpy as np

from misc.hier_IO import GaussianScenes
from gmm_point_alignment.mle_registration import (
    CSRGridBuilder,
    CSRGridBuilderConfig,
    CSRGridData,
    CSRGridQuerier,
    CSRGridQuerierConfig,
    VoxelSizeStrategy,
)
from .registration_sampler import (
    RegistrationSamplerConfig,
    RegistrationSamplerResult,
    register_with_sampler,
    SamplerRegistrationMethod,
)


@dataclass
class TopKSamplerConfig:
    """Configuration for Top-K sampler-based registration.

    Args:
        # CSR Grid configuration
        voxel_size_strategy: Strategy for computing voxel size
        voxel_size_factor: Factor for voxel size computation

        # Top-K query configuration
        top_k: Number of nearest spheres to query per point
        max_candidates_per_point: Maximum candidates to consider per point

        # Sampling configuration
        samples_per_sphere: Number of samples to generate per sphere
        sampling_mode: "mean" (sphere center) or "random" (Gaussian random)
        sample_in_local_frame: If True, sample in sphere local frame then transform

        # Registration configuration
        reg_method: Registration method (svd_icp, chamfer_opt, etc.)
        reg_max_iterations: Maximum registration iterations
        reg_tolerance: Convergence tolerance
        reg_lr: Learning rate for gradient-based methods
        reg_multi_init: Use multiple random initializations
        reg_num_init: Number of random initializations
    """
    # Grid config
    voxel_size_strategy: VoxelSizeStrategy = VoxelSizeStrategy.MEDIAN_RADIUS
    voxel_size_factor: float = 1.0

    # Query config
    top_k: int = 8
    max_candidates_per_point: int = 64

    # Sampling config
    samples_per_sphere: int = 10
    sampling_mode: Literal["mean", "random"] = "random"
    sample_in_local_frame: bool = False

    # Registration config
    reg_method: SamplerRegistrationMethod = SamplerRegistrationMethod.SVD_ICP
    reg_max_iterations: int = 100
    reg_tolerance: float = 1e-6
    reg_lr: float = 1e-3
    reg_multi_init: bool = True
    reg_num_init: int = 5


@dataclass
class TopKSamplerResult:
    """Result from Top-K sampler-based registration."""
    R: torch.Tensor          # (3, 3) rotation
    t: torch.Tensor          # (3,) translation
    scale: float             # scale factor (usually 1.0)
    rmse: float
    converged: bool
    num_iters: int

    # Additional info
    num_spheres_queried: int     # Number of unique spheres in Top-K results
    num_sampled_points: int      # Total number of sampled points
    query_time_ms: float         # Time for Top-K query
    sampling_time_ms: float      # Time for sampling
    registration_time_ms: float  # Time for registration


class TopKSamplerRegistration:
    """Top-K sampler-based registration.

    This registration method works by:
    1. Building a CSR Grid for the Gaussian scene
    2. For each input point, querying Top-K nearest spheres
    3. Sampling points only from these Top-K spheres
    4. Registering input points to the sampled point cloud

    Args:
        config: Top-K sampler configuration
    """

    def __init__(self, config: TopKSamplerConfig = None):
        self.config = config or TopKSamplerConfig()
        self._grid_data: Optional[CSRGridData] = None
        self._querier: Optional[CSRGridQuerier] = None

    def build_grid(self, scene: GaussianScenes) -> CSRGridData:
        """Build CSR grid for the scene.

        Args:
            scene: Gaussian scene

        Returns:
            CSRGridData
        """
        grid_config = CSRGridBuilderConfig(
            voxel_size_strategy=self.config.voxel_size_strategy,
            voxel_size_factor=self.config.voxel_size_factor,
        )
        builder = CSRGridBuilder(grid_config)
        self._grid_data = builder.build(scene)

        # Create querier
        query_config = CSRGridQuerierConfig(
            top_k=self.config.top_k,
            max_candidates_per_point=self.config.max_candidates_per_point,
        )
        self._querier = CSRGridQuerier(self._grid_data, query_config)

        return self._grid_data

    def register(
        self,
        scene: GaussianScenes,
        pointcloud: torch.Tensor,
    ) -> TopKSamplerResult:
        """Register point cloud to scene using Top-K sampling.

        Args:
            scene: Gaussian scene
            pointcloud: Input point cloud [N, 3]

        Returns:
            TopKSamplerResult with transformation and metrics
        """
        import time

        device = pointcloud.device

        # Step 1: Build grid if not already built
        if self._grid_data is None:
            t0 = time.time()
            self.build_grid(scene)
            grid_build_time = (time.time() - t0) * 1000
            print(f"[TopKSampler] Grid built in {grid_build_time:.1f}ms")

        # Step 2: Query Top-K spheres for each point
        t0 = time.time()
        query_result = self._querier.query(pointcloud)
        query_time = (time.time() - t0) * 1000

        topk_ids = query_result.topk_sphere_ids  # [N, K]
        topk_densities = query_result.topk_densities  # [N, K]

        # Get unique sphere IDs (excluding -1 padding)
        valid_mask = topk_ids >= 0
        unique_sphere_ids = topk_ids[valid_mask].unique().long()
        num_unique_spheres = len(unique_sphere_ids)

        print(f"[TopKSampler] Queried {num_unique_spheres} unique spheres "
              f"from {pointcloud.shape[0]} points (Top-K={self.config.top_k})")

        # Step 3: Sample points from Top-K spheres
        t0 = time.time()
        sampled_points = self._sample_from_spheres(
            scene, unique_sphere_ids
        )
        sampling_time = (time.time() - t0) * 1000

        print(f"[TopKSampler] Sampled {sampled_points.shape[0]} points "
              f"from {num_unique_spheres} spheres")

        # Step 4: Register input point cloud to sampled points
        t0 = time.time()
        reg_config = RegistrationSamplerConfig(
            method=self.config.reg_method,
            max_iterations=self.config.reg_max_iterations,
            tolerance=self.config.reg_tolerance,
            lr=self.config.reg_lr,
            multi_init=self.config.reg_multi_init,
            num_init=self.config.reg_num_init,
        )

        reg_result = register_with_sampler(pointcloud, sampled_points, reg_config)
        registration_time = (time.time() - t0) * 1000

        print(f"[TopKSampler] Registration completed in {registration_time:.1f}ms, "
              f"RMSE={reg_result.rmse:.4f}, Converged={reg_result.converged}")

        return TopKSamplerResult(
            R=reg_result.R,
            t=reg_result.t,
            scale=reg_result.scale,
            rmse=reg_result.rmse,
            converged=reg_result.converged,
            num_iters=reg_result.num_iters,
            num_spheres_queried=num_unique_spheres,
            num_sampled_points=sampled_points.shape[0],
            query_time_ms=query_time,
            sampling_time_ms=sampling_time,
            registration_time_ms=registration_time,
        )

    def _sample_from_spheres(
        self,
        scene: GaussianScenes,
        sphere_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sample points from specified spheres.

        Args:
            scene: Gaussian scene
            sphere_ids: IDs of spheres to sample from [M]

        Returns:
            Sampled points [M * samples_per_sphere, 3]
        """
        device = scene.position.device
        num_spheres = len(sphere_ids)

        # Get sphere parameters
        positions = scene.position[sphere_ids]      # [M, 3]
        scales = scene.scales[sphere_ids]           # [M, 3]
        rotations = scene.rotation[sphere_ids]      # [M, 4] (quaternions)

        if self.config.sampling_mode == "mean":
            # Just return sphere centers
            return positions

        elif self.config.sampling_mode == "random":
            # Sample from Gaussian distribution for each sphere
            k = self.config.samples_per_sphere

            # Repeat positions and scales for sampling
            positions_rep = positions.unsqueeze(1).repeat(1, k, 1)  # [M, k, 3]
            scales_rep = scales.unsqueeze(1).repeat(1, k, 1)        # [M, k, 3]

            if self.config.sample_in_local_frame:
                # Sample in local frame, then rotate
                noise_local = torch.randn_like(scales_rep) * scales_rep  # [M, k, 3]

                # Convert quaternions to rotation matrices
                R_matrices = self._quaternion_to_matrix(rotations)  # [M, 3, 3]
                R_rep = R_matrices.unsqueeze(1).repeat(1, k, 1, 1)   # [M, k, 3, 3]

                # Rotate noise: [M, k, 3] @ [M, k, 3, 3]
                noise = (R_rep @ noise_local.unsqueeze(-1)).squeeze(-1)  # [M, k, 3]
            else:
                # Sample directly in world frame
                noise = torch.randn_like(scales_rep) * scales_rep

            points = (positions_rep + noise).reshape(-1, 3)
            return points

        else:
            raise ValueError(f"Unknown sampling mode: {self.config.sampling_mode}")

    def _quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices.

        Args:
            q: Quaternions [N, 4] (w, x, y, z)

        Returns:
            Rotation matrices [N, 3, 3]
        """
        # Normalize
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Build rotation matrices
        N = q.shape[0]
        R = torch.zeros(N, 3, 3, device=q.device, dtype=q.dtype)

        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x*y - w*z)
        R[:, 0, 2] = 2 * (x*z + w*y)
        R[:, 1, 0] = 2 * (x*y + w*z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y*z - w*x)
        R[:, 2, 0] = 2 * (x*z - w*y)
        R[:, 2, 1] = 2 * (y*z + w*x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        return R


def register_with_topk_sampler(
    scene: GaussianScenes,
    pointcloud: torch.Tensor,
    config: Optional[TopKSamplerConfig] = None,
) -> TopKSamplerResult:
    """One-shot Top-K sampler registration function.

    Args:
        scene: Gaussian scene
        pointcloud: Input point cloud [N, 3]
        config: Top-K sampler configuration

    Returns:
        TopKSamplerResult

    Example:
        >>> result = register_with_topk_sampler(scene, pointcloud)
        >>> print(f"RMSE: {result.rmse}, Transform: {result.R}, {result.t}")
    """
    registrator = TopKSamplerRegistration(config)
    return registrator.register(scene, pointcloud)
