"""Tests for CSR Grid Builder module."""

import pytest
import torch
import taichi as ti
import numpy as np

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
        self.position = torch.randn(num_spheres, 3, device=device) * 10.0
        self.scales = torch.rand(num_spheres, 3, device=device) * 0.5 + 0.1
        # Random normalized quaternions
        self.rotation = torch.randn(num_spheres, 4, device=device)
        self.rotation = self.rotation / self.rotation.norm(dim=1, keepdim=True)


class TestCSRGridBuilderConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CSRGridBuilderConfig()

        assert config.confidence_level == 0.95
        assert config.voxel_size_factor == 3.0
        assert config.max_grid_size == 1024
        assert config.oversized_threshold_voxels == 64
        assert config.l1_grid_size == 32
        assert config.use_two_level_lookup is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = CSRGridBuilderConfig(
            voxel_size_factor=2.0,
            max_grid_size=512,
            oversized_threshold_voxels=32,
        )

        assert config.voxel_size_factor == 2.0
        assert config.max_grid_size == 512
        assert config.oversized_threshold_voxels == 32


class TestCSRGridBuilderBasic:
    """Test basic CSR grid builder functionality."""

    def test_builder_initialization(self):
        """Test builder can be initialized."""
        config = CSRGridBuilderConfig()
        builder = CSRGridBuilder(config)

        assert builder.config == config

    def test_build_small_scene(self):
        """Test building grid for small scene."""
        scene = DummyGaussianScene(num_spheres=10)
        config = CSRGridBuilderConfig()
        builder = CSRGridBuilder(config)

        grid_data = builder.build(scene)

        assert isinstance(grid_data, CSRGridData)
        assert grid_data.total_pairs > 0
        assert grid_data.voxel_size > 0

    def test_build_returns_correct_types(self):
        """Test that build returns data with correct tensor types."""
        scene = DummyGaussianScene(num_spheres=20)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        # Check tensor types
        assert grid_data.pairs_morton.dtype == torch.int64
        assert grid_data.pairs_sphere_id.dtype == torch.int32
        assert grid_data.l1_offsets.dtype == torch.int32
        assert grid_data.oversized_sphere_ids.dtype == torch.int32
        assert grid_data.global_aabb_min.dtype == torch.float32
        assert grid_data.cov_inv.dtype == torch.float32
        assert grid_data.norm_factor.dtype == torch.float32


class TestCSRGridDataStructure:
    """Test the structure and contents of built CSR grid."""

    def test_pairs_are_sorted(self):
        """Test that pairs_morton is sorted."""
        scene = DummyGaussianScene(num_spheres=50)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        # Check morton codes are sorted
        pairs_diff = grid_data.pairs_morton[1:] - grid_data.pairs_morton[:-1]
        assert torch.all(pairs_diff >= 0)

    def test_pairs_have_valid_length(self):
        """Test that pairs arrays have same length."""
        scene = DummyGaussianScene(num_spheres=50)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        assert len(grid_data.pairs_morton) == len(grid_data.pairs_sphere_id)
        assert len(grid_data.pairs_morton) == grid_data.total_pairs

    def test_sphere_ids_in_valid_range(self):
        """Test that sphere IDs are in valid range."""
        scene = DummyGaussianScene(num_spheres=30)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        if grid_data.total_pairs > 0:
            max_id = grid_data.pairs_sphere_id.max().item()
            min_id = grid_data.pairs_sphere_id.min().item()

            assert max_id < 30
            assert min_id >= 0

    def test_l1_offsets_shape(self):
        """Test L1 offsets table has correct shape."""
        scene = DummyGaussianScene(num_spheres=30)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        assert grid_data.l1_offsets.shape == (32, 32, 32)

    def test_l2_blocks_consistency(self):
        """Test L2 blocks are consistent with L1 offsets."""
        scene = DummyGaussianScene(num_spheres=50)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        # Count non-empty L1 entries
        valid_offsets = grid_data.l1_offsets[grid_data.l1_offsets >= 0]

        if len(valid_offsets) > 0:
            # All offsets should be valid indices into l2_blocks
            assert valid_offsets.max().item() < len(grid_data.l2_blocks)


class TestOversizedSpheres:
    """Test handling of oversized spheres."""

    def test_oversized_detection(self):
        """Test that oversized spheres are detected."""
        # Create scene with very large spheres
        scene = DummyGaussianScene(num_spheres=10)
        scene.scales = torch.ones(10, 3, device=scene.device) * 10.0  # Large scales

        config = CSRGridBuilderConfig(oversized_threshold_voxels=8)
        builder = CSRGridBuilder(config)

        grid_data = builder.build(scene)

        # With large spheres and low threshold, should have oversized
        assert len(grid_data.oversized_sphere_ids) > 0

    def test_oversized_not_in_pairs(self):
        """Test that oversized spheres are not in regular pairs."""
        scene = DummyGaussianScene(num_spheres=10)
        scene.scales = torch.ones(10, 3, device=scene.device) * 10.0

        config = CSRGridBuilderConfig(oversized_threshold_voxels=8)
        builder = CSRGridBuilder(config)

        grid_data = builder.build(scene)

        if len(grid_data.oversized_sphere_ids) > 0 and grid_data.total_pairs > 0:
            # Oversized IDs should not appear in pairs
            unique_in_pairs = torch.unique(grid_data.pairs_sphere_id)
            for oid in grid_data.oversized_sphere_ids:
                assert oid.item() not in unique_in_pairs


class TestPrecomputedData:
    """Test precomputed sphere data."""

    def test_cov_inv_shape(self):
        """Test covariance inverse has correct shape."""
        scene = DummyGaussianScene(num_spheres=25)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        assert grid_data.cov_inv.shape == (25, 3, 3)

    def test_norm_factor_shape(self):
        """Test normalization factors have correct shape."""
        scene = DummyGaussianScene(num_spheres=25)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        assert grid_data.norm_factor.shape == (25,)

    def test_cov_inv_is_symmetric(self):
        """Test covariance inverses are symmetric."""
        scene = DummyGaussianScene(num_spheres=10)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        # Check symmetry: cov_inv should equal its transpose
        for i in range(len(grid_data.cov_inv)):
            cov_inv_i = grid_data.cov_inv[i]
            diff = torch.abs(cov_inv_i - cov_inv_i.T)
            assert torch.all(diff < 1e-5)

    def test_norm_factor_positive(self):
        """Test normalization factors are positive."""
        scene = DummyGaussianScene(num_spheres=20)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        assert torch.all(grid_data.norm_factor > 0)


class TestVoxelSizeComputation:
    """Test voxel size computation."""

    def test_voxel_size_positive(self):
        """Test that computed voxel size is positive."""
        scene = DummyGaussianScene(num_spheres=20)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        assert grid_data.voxel_size > 0

    def test_voxel_size_with_different_factors(self):
        """Test voxel size changes with factor."""
        scene = DummyGaussianScene(num_spheres=20)

        config1 = CSRGridBuilderConfig(voxel_size_factor=2.0)
        config2 = CSRGridBuilderConfig(voxel_size_factor=4.0)

        builder1 = CSRGridBuilder(config1)
        builder2 = CSRGridBuilder(config2)

        grid1 = builder1.build(scene)

        # Reset scene for second build (avoid modifying shared data)
        scene2 = DummyGaussianScene(num_spheres=20)
        grid2 = builder2.build(scene2)

        # Higher factor should give larger voxel size
        assert grid2.voxel_size > grid1.voxel_size


class TestGridDimensions:
    """Test grid dimension handling."""

    def test_grid_dims_in_config(self):
        """Test grid dimensions match config."""
        config = CSRGridBuilderConfig(max_grid_size=512)
        builder = CSRGridBuilder(config)

        scene = DummyGaussianScene(num_spheres=10)
        grid_data = builder.build(scene)

        assert grid_data.grid_dims == (512, 512, 512)


class TestEmptyAndEdgeCases:
    """Test edge cases and empty inputs."""

    def test_single_sphere(self):
        """Test with single sphere."""
        scene = DummyGaussianScene(num_spheres=1)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        # Should still produce valid output
        assert grid_data.total_pairs >= 0
        assert grid_data.voxel_size > 0

    def test_large_scene(self):
        """Test with larger number of spheres."""
        scene = DummyGaussianScene(num_spheres=500)
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        assert grid_data.total_pairs > 0
        assert grid_data.cov_inv.shape[0] == 500


class TestDeviceHandling:
    """Test device (CPU/CUDA) handling."""

    def test_cpu_execution(self):
        """Test execution on CPU."""
        scene = DummyGaussianScene(num_spheres=20, device='cpu')
        builder = CSRGridBuilder()

        grid_data = builder.build(scene)

        assert grid_data.pairs_morton.device.type == 'cpu'
        assert grid_data.global_aabb_min.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
