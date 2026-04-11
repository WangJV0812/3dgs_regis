"""Tests for CSR Grid Querier module."""

import pytest
import torch
import taichi as ti
import numpy as np

from gmm_point_alignment.csr_grid_querier import (
    CSRGridQuerier,
    CSRGridQuerierConfig,
    QueryResult,
)
from gmm_point_alignment.csr_grid_builder import (
    CSRGridBuilder,
    CSRGridBuilderConfig,
    CSRGridData,
)


@pytest.fixture(scope="module", autouse=True)
def init_taichi():
    """Initialize Taichi once for all tests."""
    ti.init(arch=ti.cpu)
    yield


class DummyGaussianScene:
    """Dummy Gaussian scene for testing."""

    def __init__(self, num_spheres=100, device='cpu'):
        self.device = device
        self.position = torch.randn(num_spheres, 3, device=device) * 5.0
        self.scales = torch.rand(num_spheres, 3, device=device) * 0.3 + 0.1
        self.rotation = torch.randn(num_spheres, 4, device=device)
        self.rotation = self.rotation / self.rotation.norm(dim=1, keepdim=True)


def create_simple_grid_data(num_spheres=10, device='cpu') -> CSRGridData:
    """Create simple grid data for testing without full builder."""
    # Create simple sphere-voxel pairs
    # Each sphere maps to 1-2 voxels
    pairs_morton = []
    pairs_sphere_id = []

    for i in range(num_spheres):
        pairs_morton.append(i % 8)  # 8 different morton codes
        pairs_sphere_id.append(i)
        if i % 3 == 0:  # Some spheres in 2 voxels
            pairs_morton.append((i + 1) % 8)
            pairs_sphere_id.append(i)

    pairs_morton = torch.tensor(pairs_morton, dtype=torch.int64, device=device)
    pairs_sphere_id = torch.tensor(pairs_sphere_id, dtype=torch.int32, device=device)

    # Sort by morton
    sorted_indices = torch.argsort(pairs_morton)
    pairs_morton = pairs_morton[sorted_indices]
    pairs_sphere_id = pairs_sphere_id[sorted_indices]

    # Simple L1/L2 structure
    l1_offsets = torch.full((32, 32, 32), -1, dtype=torch.int32, device=device)
    l1_offsets[0, 0, 0] = 0
    l1_offsets[0, 0, 1] = 1

    l2_blocks = [
        torch.stack([pairs_morton[:5], pairs_sphere_id[:5].long()], dim=1),
        torch.stack([pairs_morton[5:], pairs_sphere_id[5:].long()], dim=1),
    ]

    return CSRGridData(
        pairs_morton=pairs_morton,
        pairs_sphere_id=pairs_sphere_id,
        l1_offsets=l1_offsets,
        l2_blocks=l2_blocks,
        oversized_sphere_ids=torch.tensor([], dtype=torch.int32, device=device),
        global_aabb_min=torch.tensor([-10.0, -10.0, -10.0], device=device),
        voxel_size=1.0,
        grid_dims=(1024, 1024, 1024),
        total_pairs=len(pairs_morton),
        num_unique_voxels=len(torch.unique(pairs_morton)),
        cov_inv=torch.eye(3, device=device).unsqueeze(0).repeat(num_spheres, 1, 1),
        norm_factor=torch.ones(num_spheres, device=device),
        sphere_centers=torch.randn(num_spheres, 3, device=device),
    )


class TestCSRGridQuerierConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CSRGridQuerierConfig()

        assert config.top_k == 8
        assert config.max_candidates_per_point == 64
        assert config.batch_size == 10000

    def test_custom_config(self):
        """Test custom configuration."""
        config = CSRGridQuerierConfig(
            top_k=16,
            max_candidates_per_point=128,
            batch_size=5000,
        )

        assert config.top_k == 16
        assert config.max_candidates_per_point == 128
        assert config.batch_size == 5000


class TestCSRGridQuerierBasic:
    """Test basic querier functionality."""

    def test_querier_initialization(self):
        """Test querier can be initialized."""
        grid_data = create_simple_grid_data(10)
        config = CSRGridQuerierConfig()
        querier = CSRGridQuerier(grid_data, config)

        assert querier.grid_data == grid_data
        assert querier.config == config

    def test_query_returns_correct_types(self):
        """Test that query returns correct result types."""
        grid_data = create_simple_grid_data(10)
        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=4))

        points = torch.randn(20, 3)
        result = querier.query(points)

        assert isinstance(result, QueryResult)
        assert result.topk_sphere_ids.dtype == torch.int32
        assert result.topk_densities.dtype == torch.float32

    def test_query_output_shapes(self):
        """Test query output shapes."""
        grid_data = create_simple_grid_data(10)
        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=4))

        num_points = 50
        points = torch.randn(num_points, 3)
        result = querier.query(points)

        assert result.topk_sphere_ids.shape == (num_points, 4)
        assert result.topk_densities.shape == (num_points, 4)


class TestCSRGridQuerierWithBuilder:
    """Test querier with real grid builder."""

    def test_query_with_built_grid(self):
        """Test query on grid built from real scene."""
        scene = DummyGaussianScene(num_spheres=50)
        builder_config = CSRGridBuilderConfig()
        builder = CSRGridBuilder(builder_config)
        grid_data = builder.build(scene)

        querier_config = CSRGridQuerierConfig(top_k=5)
        querier = CSRGridQuerier(grid_data, querier_config)

        points = torch.randn(30, 3)
        result = querier.query(points)

        assert result.topk_sphere_ids.shape == (30, 5)
        assert result.topk_densities.shape == (30, 5)

    def test_query_sphere_ids_in_valid_range(self):
        """Test that returned sphere IDs are in valid range."""
        scene = DummyGaussianScene(num_spheres=30)
        builder = CSRGridBuilder()
        grid_data = builder.build(scene)

        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=3))

        points = torch.randn(20, 3)
        result = querier.query(points)

        # Check valid IDs (ignoring -1 padding)
        valid_mask = result.topk_sphere_ids >= 0
        if valid_mask.any():
            max_id = result.topk_sphere_ids[valid_mask].max().item()
            assert max_id < 30

    def test_query_densities_positive(self):
        """Test that returned densities are non-negative."""
        scene = DummyGaussianScene(num_spheres=20)
        builder = CSRGridBuilder()
        grid_data = builder.build(scene)

        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=3))

        points = torch.randn(15, 3)
        result = querier.query(points)

        # Densities should be non-negative
        assert torch.all(result.topk_densities >= 0)


class TestCSRGridQuerierBatching:
    """Test batch processing functionality."""

    def test_large_query_batching(self):
        """Test that large queries are processed in batches."""
        grid_data = create_simple_grid_data(10)
        querier = CSRGridQuerier(
            grid_data,
            CSRGridQuerierConfig(top_k=3, batch_size=10)
        )

        # Query more points than batch size
        points = torch.randn(50, 3)
        result = querier.query(points)

        assert result.topk_sphere_ids.shape == (50, 3)

    def test_batch_consistency(self):
        """Test that batched results are consistent."""
        scene = DummyGaussianScene(num_spheres=20)
        builder = CSRGridBuilder()
        grid_data = builder.build(scene)

        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=3, batch_size=15))

        # Same points, different batch sizes should give same results
        points = torch.randn(40, 3)
        result1 = querier.query(points)

        querier2 = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=3, batch_size=25))
        result2 = querier2.query(points)

        # Results should be identical
        assert torch.allclose(result1.topk_sphere_ids.float(), result2.topk_sphere_ids.float())


class TestPointTransformation:
    """Test point transformation functionality."""

    def test_identity_transform(self):
        """Test that identity transform doesn't change points."""
        grid_data = create_simple_grid_data(10)
        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=3))

        points = torch.randn(10, 3)

        # Identity transform
        identity = torch.eye(4)

        result_with_transform = querier.query(points, point_transform=identity)
        result_without_transform = querier.query(points)

        # Should give same results
        assert torch.allclose(result_with_transform.topk_sphere_ids.float(),
                              result_without_transform.topk_sphere_ids.float())

    def test_translation_transform(self):
        """Test that translation changes results appropriately."""
        scene = DummyGaussianScene(num_spheres=20)
        builder = CSRGridBuilder()
        grid_data = builder.build(scene)

        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=3))

        points = torch.randn(10, 3)

        # Translation transform
        translation = torch.eye(4)
        translation[:3, 3] = torch.tensor([1.0, 2.0, 3.0])

        result = querier.query(points, point_transform=translation)

        # Should still return valid results
        assert result.topk_sphere_ids.shape == (10, 3)


class TestTopKSelection:
    """Test Top-K selection logic."""

    def test_topk_sorted_descending(self):
        """Test that Top-K densities are sorted in descending order."""
        scene = DummyGaussianScene(num_spheres=50)
        builder = CSRGridBuilder()
        grid_data = builder.build(scene)

        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=5))

        points = torch.randn(10, 3)
        result = querier.query(points)

        # Check each point's top-k is sorted descending
        for i in range(10):
            densities = result.topk_densities[i]
            valid_mask = densities > 0
            if valid_mask.sum() > 1:
                valid_densities = densities[valid_mask]
                sorted_descending = torch.all(valid_densities[:-1] >= valid_densities[1:])
                assert sorted_descending, f"Point {i}: densities not sorted"

    def test_topk_padding_with_negatives(self):
        """Test that padded entries have sphere_id = -1."""
        grid_data = create_simple_grid_data(5)  # Very few spheres
        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=10))

        points = torch.randn(10, 3)
        result = querier.query(points)

        # Some entries should be padded with -1
        assert torch.any(result.topk_sphere_ids < 0)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_scene(self):
        """Test query on grid with no spheres."""
        grid_data = create_simple_grid_data(0)
        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=3))

        points = torch.randn(5, 3)
        result = querier.query(points)

        assert result.topk_sphere_ids.shape == (5, 3)
        # All should be -1 (no valid spheres)
        assert torch.all(result.topk_sphere_ids == -1)

    def test_single_point(self):
        """Test query with single point."""
        scene = DummyGaussianScene(num_spheres=10)
        builder = CSRGridBuilder()
        grid_data = builder.build(scene)

        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=3))

        points = torch.randn(1, 3)
        result = querier.query(points)

        assert result.topk_sphere_ids.shape == (1, 3)

    def test_points_outside_grid(self):
        """Test query with points outside grid bounds."""
        scene = DummyGaussianScene(num_spheres=20)
        builder = CSRGridBuilder()
        grid_data = builder.build(scene)

        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=3))

        # Points far outside grid
        points = torch.tensor([[1000.0, 1000.0, 1000.0]] * 5)
        result = querier.query(points)

        # Should still return result (clamped to grid bounds)
        assert result.topk_sphere_ids.shape == (5, 3)


class TestDeviceHandling:
    """Test device (CPU/CUDA) handling."""

    def test_cpu_execution(self):
        """Test execution on CPU."""
        scene = DummyGaussianScene(num_spheres=20, device='cpu')
        builder = CSRGridBuilder()
        grid_data = builder.build(scene)

        querier = CSRGridQuerier(grid_data, CSRGridQuerierConfig(top_k=3))

        points = torch.randn(10, 3)
        result = querier.query(points)

        assert result.topk_sphere_ids.device.type == 'cpu'
        assert result.topk_densities.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
