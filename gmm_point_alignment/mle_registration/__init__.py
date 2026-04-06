"""MLE-based registration using CSR Grid spatial indexing.

This module provides GMM MLE registration that operates directly on Gaussian
spheres without point cloud sampling.
"""

from .csr_grid_builder import (
    CSRGridBuilder,
    CSRGridBuilderConfig,
    CSRGridData,
    VoxelSizeStrategy,
)
from .csr_grid_querier import (
    CSRGridQuerier,
    CSRGridQuerierConfig,
)
from .sphere_mle_loss import (
    GMMRegistration,
    MLEAlignmentLoss,
    MLELossConfig,
    RegistrationConfig,
)

__all__ = [
    "CSRGridBuilder",
    "CSRGridBuilderConfig",
    "CSRGridData",
    "VoxelSizeStrategy",
    "CSRGridQuerier",
    "CSRGridQuerierConfig",
    "GMMRegistration",
    "MLEAlignmentLoss",
    "MLELossConfig",
    "RegistrationConfig",
]
