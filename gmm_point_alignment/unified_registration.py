"""Unified registration interface supporting both MLE and sampler-based methods.

This module provides a unified interface to switch between:
- registration_mle: GMM MLE-based registration using CSR Grid
- registration_sampler: Traditional ICP-based registration using point cloud sampling

Example:
    >>> from gmm_point_alignment.unified_registration import (
    ...     UnifiedRegistration,
    ...     RegistrationMethod,
    ...     UnifiedConfig,
    ... )
    >>> config = UnifiedConfig(method=RegistrationMethod.MLE)
    >>> registrator = UnifiedRegistration(config)
    >>> result = registrator.register(scene, pointcloud)
"""

from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

import torch
from pathlib import Path

from misc.hier_IO import GaussianScenes


class RegistrationMethod(str, Enum):
    """Available registration methods."""
    MLE = "mle"           # GMM MLE-based using CSR Grid
    SAMPLER = "sampler"   # Traditional ICP using point sampling


@dataclass
class UnifiedConfig:
    """Unified configuration for registration.

    Args:
        method: Registration method to use ("mle" or "sampler")
        # MLE-specific configs (used when method="mle")
        mle_voxel_strategy: str = "median_radius"
        mle_voxel_factor: float = 1.0
        mle_num_iters: int = 100
        mle_lr: float = 0.01
        mle_use_scale: bool = False
        mle_multi_init: bool = True
        mle_num_init: int = 5
        mle_debug: bool = False
        mle_debug_gt_transform: Optional[torch.Tensor] = None
        # Sampler-specific configs (used when method="sampler")
        sampler_method: str = "svd_icp"
        sampler_max_iters: int = 100
        sampler_num_points: int = 5000
        sampler_multi_init: bool = True
        sampler_num_init: int = 5
    """
    method: RegistrationMethod = RegistrationMethod.MLE

    # MLE configs
    mle_voxel_strategy: str = "median_radius"
    mle_voxel_factor: float = 1.0
    mle_num_iters: int = 100
    mle_lr: float = 0.01
    mle_lr_translation: float = 0.01
    mle_lr_rotation: float = 0.001
    mle_use_scale: bool = False
    mle_multi_init: bool = True
    mle_num_init: int = 5
    mle_debug: bool = False
    mle_debug_gt_transform: Optional[torch.Tensor] = None

    # Sampler configs
    sampler_method: str = "svd_icp"
    sampler_max_iters: int = 100
    sampler_num_points: int = 5000
    sampler_multi_init: bool = True
    sampler_num_init: int = 5


@dataclass
class UnifiedResult:
    """Unified registration result.

    Attributes:
        transform: 4x4 transformation matrix
        R: 3x3 rotation matrix
        t: 3 translation vector
        scale: scale factor (1.0 for rigid methods)
        converged: whether registration converged
        loss/rmse: error metric (loss for MLE, rmse for sampler)
        method: which method was used
    """
    transform: torch.Tensor  # [4, 4]
    R: torch.Tensor          # [3, 3]
    t: torch.Tensor          # [3,]
    scale: float
    converged: bool
    error: float             # loss or rmse
    num_iters: int
    method: str


class UnifiedRegistration:
    """Unified registration interface.

    Supports switching between MLE-based and sampler-based registration
    through configuration.

    Example:
        >>> config = UnifiedConfig(method=RegistrationMethod.MLE)
        >>> reg = UnifiedRegistration(config)
        >>> result = reg.register(scene, pointcloud)
        >>>
        >>> # Switch to sampler method
        >>> config.method = RegistrationMethod.SAMPLER
        >>> reg = UnifiedRegistration(config)
        >>> result = reg.register(scene, pointcloud)
    """

    def __init__(self, config: UnifiedConfig = None):
        self.config = config or UnifiedConfig()
        self._mle_aligner = None
        self._sampler_config = None

    def _init_mle(self, scene: GaussianScenes):
        """Initialize MLE registration components."""
        from .mle_registration import (
            CSRGridBuilder,
            CSRGridBuilderConfig,
            VoxelSizeStrategy,
            GMMRegistration,
            RegistrationConfig as MLERegistrationConfig,
        )

        # Map strategy string to enum
        strategy_map = {
            "median_radius": VoxelSizeStrategy.MEDIAN_RADIUS,
            "short_axis_median": VoxelSizeStrategy.SHORT_AXIS_MEDIAN,
            "short_axis_mode": VoxelSizeStrategy.SHORT_AXIS_MODE,
            "volume_based": VoxelSizeStrategy.VOLUME_BASED,
            "percentile_dense": VoxelSizeStrategy.PERCENTILE_DENSE,
        }
        strategy = strategy_map.get(
            self.config.mle_voxel_strategy,
            VoxelSizeStrategy.MEDIAN_RADIUS
        )

        # Build grid
        grid_config = CSRGridBuilderConfig(
            voxel_size_strategy=strategy,
            voxel_size_factor=self.config.mle_voxel_factor,
        )
        grid_builder = CSRGridBuilder(grid_config)
        grid_data = grid_builder.build(scene)

        # Create registration
        reg_config = MLERegistrationConfig(
            num_iters=self.config.mle_num_iters,
            lr=self.config.mle_lr,
            lr_translation=self.config.mle_lr_translation,
            lr_rotation=self.config.mle_lr_rotation,
            use_scale=self.config.mle_use_scale,
            multi_init=self.config.mle_multi_init,
            num_init=self.config.mle_num_init,
            verbose=False,
            debug=self.config.mle_debug,
            debug_gt_transform=self.config.mle_debug_gt_transform,
        )
        self._mle_aligner = GMMRegistration(grid_data, reg_config=reg_config)

    def _init_sampler(self):
        """Initialize sampler registration components."""
        from .sampler_registration import (
            RegistrationSamplerConfig,
            SamplerRegistrationMethod,
        )

        # Map method string to enum
        method_map = {
            "svd_icp": SamplerRegistrationMethod.SVD_ICP,
            "chamfer_opt": SamplerRegistrationMethod.CHAMFER_OPT,
            "open3d_icp_point_to_point": SamplerRegistrationMethod.OPEN3D_ICP_POINT_TO_POINT,
            "open3d_icp_point_to_plane": SamplerRegistrationMethod.OPEN3D_ICP_POINT_TO_PLANE,
        }
        method = method_map.get(
            self.config.sampler_method,
            SamplerRegistrationMethod.SVD_ICP
        )

        self._sampler_config = RegistrationSamplerConfig(
            method=method,
            max_iterations=self.config.sampler_max_iters,
            multi_init=self.config.sampler_multi_init,
            num_init=self.config.sampler_num_init,
        )

    def register(
        self,
        scene: GaussianScenes,
        pointcloud: torch.Tensor,
    ) -> UnifiedResult:
        """Register point cloud to scene.

        Args:
            scene: Gaussian scene
            pointcloud: Point cloud to register [N, 3]

        Returns:
            UnifiedResult with transformation
        """
        if self.config.method == RegistrationMethod.MLE:
            return self._register_mle(scene, pointcloud)
        elif self.config.method == RegistrationMethod.SAMPLER:
            return self._register_sampler(scene, pointcloud)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def _register_mle(
        self,
        scene: GaussianScenes,
        pointcloud: torch.Tensor,
    ) -> UnifiedResult:
        """Register using MLE method."""
        if self._mle_aligner is None:
            self._init_mle(scene)

        result = self._mle_aligner.register(pointcloud)

        T = result['transform']
        R = T[:3, :3]
        t = T[:3, 3]
        scale = result.get('scale', torch.tensor(1.0)).item()

        return UnifiedResult(
            transform=T,
            R=R,
            t=t,
            scale=scale,
            converged=result['converged'].item(),
            error=result['loss'].item(),
            num_iters=result['num_iters'],
            method="mle",
        )

    def _register_sampler(
        self,
        scene: GaussianScenes,
        pointcloud: torch.Tensor,
    ) -> UnifiedResult:
        """Register using sampler method."""
        from .sampler_registration import (
            GaussianSampler,
            SamplingConfig,
            register_with_sampler,
        )

        if self._sampler_config is None:
            self._init_sampler()

        # Sample point cloud from scene
        sampling_config = SamplingConfig(
            mode="mean",
            target_num_points=self.config.sampler_num_points,
            downsample_strategy="fps",
        )
        sampler = GaussianSampler(sampling_config)
        scene_sampled = sampler.sample(scene)

        # Register
        result = register_with_sampler(
            pointcloud,
            scene_sampled.points,
            self._sampler_config
        )

        # Build 4x4 transform
        transform = torch.eye(4, device=pointcloud.device)
        transform[:3, :3] = result.R
        transform[:3, 3] = result.t

        return UnifiedResult(
            transform=transform,
            R=result.R,
            t=result.t,
            scale=result.scale,
            converged=result.converged,
            error=result.rmse,
            num_iters=result.num_iters,
            method="sampler",
        )


def register_pointcloud(
    scene: GaussianScenes,
    pointcloud: torch.Tensor,
    method: str = "mle",
    **kwargs
) -> UnifiedResult:
    """One-shot registration function.

    Convenience function for simple use cases.

    Args:
        scene: Gaussian scene
        pointcloud: Point cloud to register [N, 3]
        method: "mle" or "sampler"
        **kwargs: Additional config options

    Returns:
        UnifiedResult

    Example:
        >>> result = register_pointcloud(scene, points, method="mle")
        >>> print(f"Transform: {result.transform}")
    """
    config = UnifiedConfig(method=RegistrationMethod(method))

    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    registrator = UnifiedRegistration(config)
    return registrator.register(scene, pointcloud)
