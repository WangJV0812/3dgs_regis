"""Tests for morton_code module with grid coordinate extensions."""

import pytest
import torch
import taichi as ti

from gmm_point_alignment.morton_code import (
    encode_grid_to_morton_ti,
    decode_morton_to_grid_ti,
    grid_coords_to_morton,
    morton_to_grid_coords,
    compute_morton_range,
    Morton3D,
)


@pytest.fixture(scope="module", autouse=True)
def init_taichi():
    """Initialize Taichi once for all tests."""
    ti.init(arch=ti.cpu)
    yield


class TestGridMortonEncoding:
    """Test grid coordinate to morton encoding functions."""

    def test_encode_single_coordinate(self):
        """Test encoding single grid coordinates."""
        # Test (0,0,0)
        coords = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        morton = grid_coords_to_morton(coords)
        assert morton.shape == (1,)
        assert morton[0].item() == 0

    def test_encode_batch_coordinates(self):
        """Test batch encoding of grid coordinates."""
        coords = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=torch.int32)

        morton = grid_coords_to_morton(coords)
        assert morton.shape == (5,)
        assert morton.dtype == torch.uint32

        # Verify interleaving: (1,0,0) -> bits: x=001, y=000, z=000 -> 0b001000 = 8
        assert morton[1].item() == 4  # x=1 at position 2

    def test_encode_boundary_values(self):
        """Test encoding at boundary values (0 and 1023)."""
        coords = torch.tensor([
            [0, 0, 0],
            [1023, 1023, 1023],
            [512, 512, 512],
        ], dtype=torch.int32)

        morton = grid_coords_to_morton(coords)
        assert morton.shape == (3,)

        # Max value should produce valid morton code
        assert morton[1].item() > 0

    def test_encode_clamping(self):
        """Test that values outside [0, 1023] are clamped."""
        coords = torch.tensor([
            [-1, -1, -1],  # Should clamp to 0
            [1024, 1024, 1024],  # Should clamp to 1023
        ], dtype=torch.int32)

        morton = grid_coords_to_morton(coords)

        # Both should be valid (clamped)
        assert morton[0].item() == 0  # Clamped to (0,0,0)
        assert morton[1].item() > 0   # Clamped to (1023,1023,1023)


class TestMortonDecoding:
    """Test morton to grid coordinate decoding functions."""

    def test_decode_single_morton(self):
        """Test decoding single morton code."""
        morton = torch.tensor([0], dtype=torch.uint32)
        coords = morton_to_grid_coords(morton)

        assert coords.shape == (1, 3)
        assert coords[0, 0].item() == 0
        assert coords[0, 1].item() == 0
        assert coords[0, 2].item() == 0

    def test_roundtrip_encode_decode(self):
        """Test that encode -> decode preserves coordinates."""
        original = torch.tensor([
            [0, 0, 0],
            [1, 2, 3],
            [100, 200, 300],
            [1023, 1023, 1023],
        ], dtype=torch.int32)

        morton = grid_coords_to_morton(original)
        decoded = morton_to_grid_coords(morton)

        assert torch.allclose(original, decoded)

    def test_decode_batch(self):
        """Test batch decoding."""
        morton = torch.tensor([0, 1, 2, 4, 8], dtype=torch.uint32)
        coords = morton_to_grid_coords(morton)

        assert coords.shape == (5, 3)


class TestMortonRange:
    """Test morton range computation for AABB."""

    def test_single_voxel_range(self):
        """Test range for single voxel AABB."""
        grid_min = torch.tensor([10, 20, 30], dtype=torch.int32)
        grid_max = torch.tensor([10, 20, 30], dtype=torch.int32)

        morton_min, morton_max = compute_morton_range(grid_min, grid_max)

        # Single voxel: min == max
        assert morton_min == morton_max

    def test_small_aabb_range(self):
        """Test range for small AABB."""
        grid_min = torch.tensor([0, 0, 0], dtype=torch.int32)
        grid_max = torch.tensor([1, 1, 1], dtype=torch.int32)

        morton_min, morton_max = compute_morton_range(grid_min, grid_max)

        # 8 corners, morton codes should be different
        assert morton_max >= morton_min

    def test_large_aabb_range(self):
        """Test range for larger AABB."""
        grid_min = torch.tensor([0, 0, 0], dtype=torch.int32)
        grid_max = torch.tensor([31, 31, 31], dtype=torch.int32)

        morton_min, morton_max = compute_morton_range(grid_min, grid_max)

        # Should cover a range of morton codes
        assert morton_max > morton_min


class TestMorton3DOriginal:
    """Test original Morton3D class for point clouds."""

    def test_pointcloud_encoding(self):
        """Test encoding point cloud coordinates."""
        morton3d = Morton3D()

        pointcloud = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)

        result = morton3d(pointcloud)

        assert 'sorted_pointcloud' in result
        assert 'morton_codes' in result
        assert result['sorted_pointcloud'].shape == pointcloud.shape
        assert result['morton_codes'].shape == (4,)

    def test_pointcloud_with_color(self):
        """Test encoding with color information."""
        morton3d = Morton3D()

        pointcloud = torch.randn(100, 3, dtype=torch.float32)
        color = torch.randn(100, 3, dtype=torch.float32)

        result = morton3d(pointcloud, color=color)

        assert result['sorted_color'] is not None
        assert result['sorted_color'].shape == color.shape


class TestErrorHandling:
    """Test error handling in morton code functions."""

    def test_invalid_grid_coords_shape(self):
        """Test error on invalid grid coordinates shape."""
        with pytest.raises(ValueError):
            coords = torch.randn(10, 4)  # Wrong shape
            grid_coords_to_morton(coords)

    def test_invalid_morton_shape(self):
        """Test error on invalid morton codes shape."""
        with pytest.raises(ValueError):
            morton = torch.randn(10, 3)  # Should be 1D
            morton_to_grid_coords(morton)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
