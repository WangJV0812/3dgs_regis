"""GMM Point Alignment V2 - Main orchestrator for point cloud registration.

Integrates CSR Grid construction, Top-K querying, and MLE-based registration
into a unified interface.

Example:
    >>> from gmm_point_alignment.gmm_point_alignment import GMMPointAlignment
    >>> aligner = GMMPointAlignment()
    >>> aligner.build_grid(scene)
    >>> result = aligner.register(pointcloud)
    >>> print(f"Optimized transform: {result['transform']}")
"""

import torch
import taichi as ti
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any
from time import time

from misc.hier_IO import GaussianScenes
from gmm_point_alignment.mle_registration import (
    CSRGridBuilder,
    CSRGridBuilderConfig,
    CSRGridData,
    CSRGridQuerier,
    CSRGridQuerierConfig,
    MLEAlignmentLoss,
    MLELossConfig,
    GMMRegistration,
    RegistrationConfig,
)
from gmm_point_alignment.transform_utils import se3_exp, se3_log


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GMMPointAlignmentConfig:
    """Configuration for GMM Point Alignment.

    Combines configurations for all sub-components.

    Args:
        grid_config: CSR grid builder configuration
        query_config: Grid querier configuration
        loss_config: MLE loss configuration
        reg_config: Registration optimization configuration
    """
    grid_config: CSRGridBuilderConfig = field(default_factory=CSRGridBuilderConfig)
    query_config: CSRGridQuerierConfig = field(default_factory=CSRGridQuerierConfig)
    loss_config: MLELossConfig = field(default_factory=MLELossConfig)
    reg_config: RegistrationConfig = field(default_factory=RegistrationConfig)


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class QueryResult:
    """Result from point-to-sphere query."""
    topk_sphere_ids: torch.Tensor  # [N, K]
    topk_densities: torch.Tensor   # [N, K]
    query_time_ms: float


@dataclass
class RegistrationResult:
    """Result from registration optimization."""
    transform: torch.Tensor           # [4, 4]
    loss: float
    inlier_ratio: float
    num_iters: int
    converged: bool
    optimization_time_ms: float
    scale: float = 1.0                # Scale factor (if optimized)


@dataclass
class AlignmentResult:
    """Complete alignment result combining query and registration."""
    query: QueryResult
    registration: Optional[RegistrationResult] = None
    total_time_ms: float = 0.0


# =============================================================================
# Main Orchestrator
# =============================================================================

class GMMPointAlignment(torch.nn.Module):
    """Main orchestrator for GMM-based point cloud registration.

    Integrates CSR grid building, Top-K querying, and MLE registration
    into a unified workflow.

    Args:
        config: Alignment configuration

    Example:
        >>> aligner = GMMPointAlignment()
        >>> aligner.build_grid(scene)
        >>> result = aligner.register(pointcloud)
        >>> print(f"Optimized transform: {result.transform}")
    """

    def __init__(
        self,
        config: GMMPointAlignmentConfig = GMMPointAlignmentConfig(),
    ):
        super().__init__()
        self.config = config

        # Sub-components (initialized lazily)
        self._grid_builder: Optional[CSRGridBuilder] = None
        self._grid_data: Optional[CSRGridData] = None
        self._querier: Optional[CSRGridQuerier] = None
        self._loss_fn: Optional[MLEAlignmentLoss] = None
        self._registration: Optional[GMMRegistration] = None

        # Cached scene info
        self._scene_built: bool = False
        self._num_spheres: int = 0

    # ========================================================================
    # Grid Building (Phase 1)
    # ========================================================================

    def build_grid(
        self,
        scene: GaussianScenes,
        force_rebuild: bool = False,
    ) -> 'GMMPointAlignment':
        """Build CSR grid for the scene (one-time cost).

        This method must be called before query() or register().
        The grid is cached and can be reused for multiple point clouds.

        Args:
            scene: Gaussian scene with positions, scales, rotations, opacities
            force_rebuild: If True, rebuild even if already built

        Returns:
            self for method chaining

        Example:
            >>> aligner = GMMPointAlignment()
            >>> aligner.build_grid(scene)
            >>> # Now ready for queries
        """
        if self._scene_built and not force_rebuild:
            print("[GMMPointAlignment] Grid already built, skipping. "
                  "Use force_rebuild=True to rebuild.")
            return self

        print("[GMMPointAlignment] Building CSR grid...")
        start_time = time()

        self._grid_builder = CSRGridBuilder(self.config.grid_config)
        self._grid_data = self._grid_builder.build(scene)

        build_time = (time() - start_time) * 1000

        self._num_spheres = scene.position.shape[0]
        self._scene_built = True

        print(f"[GMMPointAlignment] Grid built in {build_time:.1f}ms")
        print(f"  - Spheres: {self._num_spheres}")
        print(f"  - Voxel size: {self._grid_data.voxel_size:.4f}")
        print(f"  - Grid dims: {self._grid_data.grid_dims}")

        # Initialize dependent components
        self._init_querier()
        self._init_registration()

        return self

    def _init_querier(self) -> None:
        """Initialize grid querier."""
        if self._grid_data is None:
            raise RuntimeError("Grid not built. Call build_grid() first.")

        self._querier = CSRGridQuerier(
            self._grid_data,
            self.config.query_config,
        )
        self._loss_fn = MLEAlignmentLoss(
            self._grid_data,
            self.config.loss_config,
        )

    def _init_registration(self) -> None:
        """Initialize registration optimizer."""
        if self._grid_data is None:
            raise RuntimeError("Grid not built. Call build_grid() first.")

        self._registration = GMMRegistration(
            self._grid_data,
            loss_config=self.config.loss_config,
            reg_config=self.config.reg_config,
        )

    # ========================================================================
    # Query (Phase 2)
    # ========================================================================

    def query(
        self,
        points: torch.Tensor,
        transform: Optional[torch.Tensor] = None,
    ) -> QueryResult:
        """Query Top-K spheres for each point.

        Args:
            points: [N, 3] point cloud
            transform: [4, 4] optional transformation to apply to points

        Returns:
            QueryResult with topk_sphere_ids and topk_densities

        Raises:
            RuntimeError: If grid not built
        """
        if not self._scene_built:
            raise RuntimeError(
                "Grid not built. Call build_grid(scene) before query()."
            )

        start_time = time()

        # Transform points if needed
        if transform is not None:
            R = transform[:3, :3]
            t = transform[:3, 3]
            points = (R @ points.T).T + t

        # Query grid
        query_result = self._querier.query(points)

        query_time = (time() - start_time) * 1000

        return QueryResult(
            topk_sphere_ids=query_result.topk_sphere_ids,
            topk_densities=query_result.topk_densities,
            query_time_ms=query_time,
        )

    # ========================================================================
    # Registration (Phase 3)
    # ========================================================================

    def register(
        self,
        points: torch.Tensor,
        initial_transform: Optional[torch.Tensor] = None,
        config: Optional[RegistrationConfig] = None,
    ) -> RegistrationResult:
        """Register point cloud to scene using MLE optimization.

        Args:
            points: [N, 3] point cloud to register
            initial_transform: [4, 4] optional initial guess
            config: Optional override for registration config

        Returns:
            RegistrationResult with optimized transform and statistics

        Raises:
            RuntimeError: If grid not built
        """
        if not self._scene_built:
            raise RuntimeError(
                "Grid not built. Call build_grid(scene) before register()."
            )

        # Use custom config if provided
        if config is not None:
            registration = GMMRegistration(
                self._grid_data,
                loss_config=self.config.loss_config,
                reg_config=config,
            )
        else:
            registration = self._registration

        print("[GMMPointAlignment] Starting registration...")
        start_time = time()

        result = registration.register(points, initial_transform)

        opt_time = (time() - start_time) * 1000

        print(f"[GMMPointAlignment] Registration completed in {opt_time:.1f}ms")
        print(f"  - Loss: {result['loss'].item():.4f}")
        print(f"  - Iters: {result['num_iters']}")
        print(f"  - Converged: {result['converged'].item()}")
        if 'scale' in result:
            print(f"  - Scale: {result['scale'].item():.4f}")

        return RegistrationResult(
            transform=result['transform'],
            loss=result['loss'].item(),
            inlier_ratio=result['inlier_ratio'].item(),
            num_iters=result['num_iters'],
            converged=result['converged'].item(),
            optimization_time_ms=opt_time,
            scale=result.get('scale', torch.tensor(1.0)).item(),
        )

    def register_with_icp_init(
        self,
        points: torch.Tensor,
        icp_transform: torch.Tensor,
        config: Optional[RegistrationConfig] = None,
    ) -> RegistrationResult:
        """Register with ICP initialization for coarse alignment.

        Args:
            points: [N, 3] point cloud
            icp_transform: [4, 4] ICP coarse alignment
            config: Optional override for registration config

        Returns:
            RegistrationResult with refined transform
        """
        if config is None:
            config = RegistrationConfig(
                num_iters=50,
                lr=0.01,
                multi_init=False,
                verbose=True,
            )

        return self.register(points, icp_transform, config)

    # ========================================================================
    # Combined Workflow
    # ========================================================================

    def align(
        self,
        points: torch.Tensor,
        initial_transform: Optional[torch.Tensor] = None,
    ) -> AlignmentResult:
        """Complete alignment workflow: query + registration.

        This is a convenience method that runs the full pipeline:
        1. Query Top-K associations for initial transform
        2. Optimize transform using MLE registration

        Args:
            points: [N, 3] point cloud
            initial_transform: [4, 4] optional initial guess

        Returns:
            AlignmentResult with query and registration results
        """
        total_start = time()

        # Query
        query_result = self.query(points, initial_transform)

        # Register
        reg_result = self.register(points, initial_transform)

        total_time = (time() - total_start) * 1000

        return AlignmentResult(
            query=query_result,
            registration=reg_result,
            total_time_ms=total_time,
        )

    # ========================================================================
    # PyTorch nn.Module Interface
    # ========================================================================

    def forward(
        self,
        scene: GaussianScenes,
        points: torch.Tensor,
        point_transform: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass: build grid if needed, then query.

        This method is compatible with PyTorch nn.Module conventions.

        Args:
            scene: Gaussian scene (grid will be built if needed)
            points: [N, 3] point cloud
            point_transform: [4, 4] optional point transformation

        Returns:
            Dictionary with query results and optionally loss
        """
        # Build grid if needed
        if not self._scene_built:
            self.build_grid(scene)

        # Query
        query_result = self.query(points, point_transform)

        result = {
            'topk_sphere_ids': query_result.topk_sphere_ids,
            'topk_densities': query_result.topk_densities,
            'query_time_ms': query_result.query_time_ms,
        }

        # Compute loss if transform provided
        if point_transform is not None and self._loss_fn is not None:
            with torch.no_grad():
                loss = self._loss_fn(points, point_transform)
                result['loss'] = loss.item()

        return result

    # ========================================================================
    # Utilities
    # ========================================================================

    def get_grid_info(self) -> Dict[str, Any]:
        """Get information about the built grid.

        Returns:
            Dictionary with grid statistics
        """
        if not self._scene_built or self._grid_data is None:
            return {'built': False}

        return {
            'built': True,
            'num_spheres': self._num_spheres,
            'voxel_size': self._grid_data.voxel_size,
            'grid_dims': self._grid_data.grid_dims,
            'total_pairs': self._grid_data.total_pairs,
            'num_unique_voxels': self._grid_data.num_unique_voxels,
        }

    def is_ready(self) -> bool:
        """Check if the aligner is ready for queries.

        Returns:
            True if grid is built and ready
        """
        return self._scene_built and self._grid_data is not None

    def clear_cache(self) -> None:
        """Clear cached grid data to free memory."""
        self._grid_builder = None
        self._grid_data = None
        self._querier = None
        self._loss_fn = None
        self._registration = None
        self._scene_built = False
        self._num_spheres = 0
        print("[GMMPointAlignment] Cache cleared.")


# =============================================================================
# Convenience Functions
# =============================================================================

def register_pointcloud(
    scene: GaussianScenes,
    pointcloud: torch.Tensor,
    config: Optional[GMMPointAlignmentConfig] = None,
    initial_transform: Optional[torch.Tensor] = None,
) -> RegistrationResult:
    """One-shot registration function.

    Convenience function for simple use cases.

    Args:
        scene: Gaussian scene
        pointcloud: [N, 3] point cloud to register
        config: Optional alignment configuration
        initial_transform: [4, 4] optional initial guess

    Returns:
        RegistrationResult

    Example:
        >>> result = register_pointcloud(scene, points)
        >>> print(f"Transform: {result.transform}")
    """
    if config is None:
        config = GMMPointAlignmentConfig()

    aligner = GMMPointAlignment(config)
    aligner.build_grid(scene)
    return aligner.register(pointcloud, initial_transform)


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    # Initialize Taichi
    ti.init(arch=ti.cuda if torch.cuda.is_available() else ti.cpu)

    from misc.hier_IO import GaussianScenes

    # Create dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scene = GaussianScenes(
        position=torch.randn(1000, 3, device=device) * 5.0,
        scales=torch.rand(1000, 3, device=device) * 0.3 + 0.1,
        rotation=torch.randn(1000, 4, device=device),
        opacities=torch.ones(1000, device=device),
        shs=torch.randn(1000, 3, 16, device=device),
    )
    scene.rotation = scene.rotation / scene.rotation.norm(dim=1, keepdim=True)

    pointcloud = torch.randn(500, 3, device=device)

    # Example 1: Basic usage
    print("\n" + "=" * 60)
    print("Example 1: Basic Registration")
    print("=" * 60)

    aligner = GMMPointAlignment()
    aligner.build_grid(scene)

    # Query only
    query_result = aligner.query(pointcloud)
    print(f"\nQuery result:")
    print(f"  Top-K IDs shape: {query_result.topk_sphere_ids.shape}")
    print(f"  Query time: {query_result.query_time_ms:.2f}ms")

    # Full registration
    reg_result = aligner.register(pointcloud)
    print(f"\nRegistration result:")
    print(f"  Loss: {reg_result.loss:.4f}")
    print(f"  Converged: {reg_result.converged}")

    # Example 2: One-shot function
    print("\n" + "=" * 60)
    print("Example 2: One-shot Registration")
    print("=" * 60)

    result = register_pointcloud(scene, pointcloud)
    print(f"Transform:\n{result.transform}")

    # Example 3: With custom config
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)

    config = GMMPointAlignmentConfig(
        query_config=CSRGridQuerierConfig(top_k=16),
        reg_config=RegistrationConfig(
            num_iters=50,
            lr=0.02,
            multi_init=True,
            num_init=3,
        ),
    )

    aligner = GMMPointAlignment(config)
    aligner.build_grid(scene)
    result = aligner.align(pointcloud)

    print(f"Total time: {result.total_time_ms:.2f}ms")
    print(f"Final loss: {result.registration.loss:.4f}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
