"""Sampler-based registration methods from point cloud sampling.

This module provides traditional point cloud registration methods that sample
points from the Gaussian scene and use ICP-based alignment.
"""

from .registration_sampler import (
    RegistrationSamplerConfig,
    RegistrationSamplerResult,
    register_with_sampler,
    GaussianSampler,
)

__all__ = [
    "RegistrationSamplerConfig",
    "RegistrationSamplerResult",
    "register_with_sampler",
    "GaussianSampler",
]
