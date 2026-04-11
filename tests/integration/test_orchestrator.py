"""Test Phase 4: GMM Point Alignment Orchestrator.

Tests the unified GMMPointAlignment interface that integrates:
- Grid building (Phase 1)
- Top-K querying (Phase 2)
- MLE registration (Phase 3)
"""

import sys
sys.path.insert(0, '.')

import pytest
import torch
import taichi as ti
from pathlib import Path

from gmm_point_alignment.gmm_point_alignment import (
    GMMPointAlignment,
    GMMPointAlignmentConfig,
    register_pointcloud,
)
from gmm_point_alignment.csr_grid_builder import CSRGridBuilderConfig
from gmm_point_alignment.csr_grid_querier import CSRGridQuerierConfig
from gmm_point_alignment.sphere_mle_loss import RegistrationConfig
from misc.hier_IO import GaussianScenes


@pytest.fixture(scope="module", autouse=True)
def init_taichi():
    """Initialize Taichi once for all tests."""
    ti.init(arch=ti.cuda if torch.cuda.is_available() else ti.cpu)
    yield


class DummyScene:
    """Dummy Gaussian scene for testing."""

    def __init__(self, num_spheres=100, device='cuda'):
        self.position = torch.randn(num_spheres, 3, device=device) * 5.0
        self.scales = torch.rand(num_spheres, 3, device=device) * 0.3 + 0.1
        self.rotation = torch.randn(num_spheres, 4, device=device)
        self.rotation = self.rotation / self.rotation.norm(dim=1, keepdim=True)
        self.opacities = torch.ones(num_spheres, device=device)
        self.shs = torch.randn(num_spheres, 3, 16, device=device)


@pytest.fixture
def dummy_scene():
    """Create dummy scene."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scene = DummyScene(100, device=device)
    # Convert to GaussianScenes dataclass
    return GaussianScenes(
        position=scene.position,
        rotation=scene.rotation,
        scales=scene.scales,
        opacities=scene.opacities,
        shs=scene.shs,
    )


@pytest.fixture
def dummy_pointcloud():
    """Create dummy point cloud."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.randn(50, 3, device=device)


class TestGMMPointAlignmentBasic:
    """Test basic GMMPointAlignment functionality."""

    def test_initialization(self):
        """Test that aligner initializes correctly."""
        aligner = GMMPointAlignment()
        assert not aligner.is_ready()
        assert aligner.get_grid_info()['built'] is False

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = GMMPointAlignmentConfig(
            query_config=CSRGridQuerierConfig(top_k=16),
            reg_config=RegistrationConfig(num_iters=50),
        )
        aligner = GMMPointAlignment(config)
        assert aligner.config.query_config.top_k == 16
        assert aligner.config.reg_config.num_iters == 50

    def test_build_grid(self, dummy_scene):
        """Test grid building."""
        aligner = GMMPointAlignment()
        result = aligner.build_grid(dummy_scene)

        # Should return self for chaining
        assert result is aligner
        assert aligner.is_ready()

        # Check grid info
        info = aligner.get_grid_info()
        assert info['built'] is True
        assert info['num_spheres'] == 100
        assert info['voxel_size'] > 0

    def test_build_grid_twice(self, dummy_scene):
        """Test that building twice is skipped by default."""
        aligner = GMMPointAlignment()
        aligner.build_grid(dummy_scene)

        # Second build should skip
        aligner.build_grid(dummy_scene)
        assert aligner.is_ready()

    def test_build_grid_force_rebuild(self, dummy_scene):
        """Test force rebuild option."""
        aligner = GMMPointAlignment()
        aligner.build_grid(dummy_scene)

        # Force rebuild
        aligner.build_grid(dummy_scene, force_rebuild=True)
        assert aligner.is_ready()


class TestGMMPointAlignmentQuery:
    """Test querying functionality."""

    def test_query_before_build_raises(self, dummy_pointcloud):
        """Test that query raises error if grid not built."""
        aligner = GMMPointAlignment()
        with pytest.raises(RuntimeError, match="Grid not built"):
            aligner.query(dummy_pointcloud)

    def test_query_basic(self, dummy_scene, dummy_pointcloud):
        """Test basic query."""
        aligner = GMMPointAlignment()
        aligner.build_grid(dummy_scene)

        result = aligner.query(dummy_pointcloud)

        assert result.topk_sphere_ids.shape == (50, 8)  # Default K=8
        assert result.topk_densities.shape == (50, 8)
        assert result.query_time_ms >= 0

    def test_query_with_transform(self, dummy_scene, dummy_pointcloud):
        """Test query with transform."""
        aligner = GMMPointAlignment()
        aligner.build_grid(dummy_scene)

        transform = torch.eye(4, device=dummy_pointcloud.device)
        result = aligner.query(dummy_pointcloud, transform)

        assert result.topk_sphere_ids.shape == (50, 8)

    def test_query_with_custom_k(self, dummy_scene, dummy_pointcloud):
        """Test query with custom top_k."""
        config = GMMPointAlignmentConfig(
            query_config=CSRGridQuerierConfig(top_k=16)
        )
        aligner = GMMPointAlignment(config)
        aligner.build_grid(dummy_scene)

        result = aligner.query(dummy_pointcloud)

        assert result.topk_sphere_ids.shape == (50, 16)


class TestGMMPointAlignmentRegistration:
    """Test registration functionality."""

    def test_register_before_build_raises(self, dummy_pointcloud):
        """Test that register raises error if grid not built."""
        aligner = GMMPointAlignment()
        with pytest.raises(RuntimeError, match="Grid not built"):
            aligner.register(dummy_pointcloud)

    def test_register_basic(self, dummy_scene, dummy_pointcloud):
        """Test basic registration."""
        aligner = GMMPointAlignment()
        aligner.build_grid(dummy_scene)

        result = aligner.register(dummy_pointcloud)

        assert result.transform.shape == (4, 4)
        assert isinstance(result.loss, float)
        assert 0 <= result.inlier_ratio <= 1
        assert result.num_iters > 0
        assert isinstance(result.converged, bool)
        assert result.optimization_time_ms >= 0

    def test_register_with_initial_transform(self, dummy_scene, dummy_pointcloud):
        """Test registration with initial transform."""
        aligner = GMMPointAlignment()
        aligner.build_grid(dummy_scene)

        initial = torch.eye(4, device=dummy_pointcloud.device)
        result = aligner.register(dummy_pointcloud, initial_transform=initial)

        assert result.transform.shape == (4, 4)

    def test_register_with_custom_config(self, dummy_scene, dummy_pointcloud):
        """Test registration with custom config."""
        aligner = GMMPointAlignment()
        aligner.build_grid(dummy_scene)

        custom_config = RegistrationConfig(
            num_iters=20,
            lr=0.05,
            multi_init=False,
        )
        result = aligner.register(dummy_pointcloud, config=custom_config)

        assert result.num_iters <= 20

    def test_register_with_icp_init(self, dummy_scene, dummy_pointcloud):
        """Test registration with ICP initialization."""
        aligner = GMMPointAlignment()
        aligner.build_grid(dummy_scene)

        icp_transform = torch.eye(4, device=dummy_pointcloud.device)
        result = aligner.register_with_icp_init(dummy_pointcloud, icp_transform)

        assert result.transform.shape == (4, 4)


class TestGMMPointAlignmentAlign:
    """Test combined align workflow."""

    def test_align_basic(self, dummy_scene, dummy_pointcloud):
        """Test combined align method."""
        aligner = GMMPointAlignment()
        aligner.build_grid(dummy_scene)

        result = aligner.align(dummy_pointcloud)

        assert result.query is not None
        assert result.registration is not None
        assert result.total_time_ms >= 0

        assert result.query.topk_sphere_ids.shape == (50, 8)
        assert result.registration.transform.shape == (4, 4)


class TestGMMPointAlignmentForward:
    """Test PyTorch nn.Module forward interface."""

    def test_forward_builds_grid(self, dummy_scene, dummy_pointcloud):
        """Test that forward builds grid if needed."""
        aligner = GMMPointAlignment()
        assert not aligner.is_ready()

        result = aligner.forward(dummy_scene, dummy_pointcloud)

        assert aligner.is_ready()
        assert 'topk_sphere_ids' in result
        assert 'topk_densities' in result

    def test_forward_with_transform(self, dummy_scene, dummy_pointcloud):
        """Test forward with transform."""
        aligner = GMMPointAlignment()

        transform = torch.eye(4, device=dummy_pointcloud.device)
        result = aligner.forward(dummy_scene, dummy_pointcloud, transform)

        assert 'topk_sphere_ids' in result
        assert 'loss' in result


class TestGMMPointAlignmentUtilities:
    """Test utility methods."""

    def test_clear_cache(self, dummy_scene, dummy_pointcloud):
        """Test cache clearing."""
        aligner = GMMPointAlignment()
        aligner.build_grid(dummy_scene)
        assert aligner.is_ready()

        aligner.clear_cache()
        assert not aligner.is_ready()
        assert aligner.get_grid_info()['built'] is False


class TestRegisterPointcloudFunction:
    """Test one-shot registration function."""

    def test_register_pointcloud_basic(self, dummy_scene, dummy_pointcloud):
        """Test one-shot registration."""
        result = register_pointcloud(dummy_scene, dummy_pointcloud)

        assert result.transform.shape == (4, 4)
        assert isinstance(result.loss, float)
        assert result.converged is not None

    def test_register_pointcloud_with_config(self, dummy_scene, dummy_pointcloud):
        """Test one-shot with custom config."""
        config = GMMPointAlignmentConfig(
            reg_config=RegistrationConfig(num_iters=20)
        )
        result = register_pointcloud(dummy_scene, dummy_pointcloud, config)

        assert result.num_iters <= 20

    def test_register_pointcloud_with_initial(self, dummy_scene, dummy_pointcloud):
        """Test one-shot with initial transform."""
        initial = torch.eye(4, device=dummy_pointcloud.device)
        result = register_pointcloud(dummy_scene, dummy_pointcloud, initial_transform=initial)

        assert result.transform.shape == (4, 4)


class TestGMMPointAlignmentWithRealData:
    """Test with real data from ./data/."""

    @pytest.fixture
    def real_data(self):
        """Load real data if available."""
        data_dir = Path(__file__).parent.parent.parent / "data"
        hier_path = data_dir / "merged.hier"
        ply_path = data_dir / "points3D.ply"

        if not hier_path.exists() or not ply_path.exists():
            pytest.skip("Real data not available")

        from misc.hier_IO import load_hier_to_torch
        from tests.utils import read_ply_xyz

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hier_scene = load_hier_to_torch(hier_path, device=device)
        pointcloud = read_ply_xyz(ply_path).to(device)

        return {
            'scene': hier_scene.gaussian_scene,
            'pointcloud': pointcloud,
        }

    def test_real_data_build_grid(self, real_data):
        """Test grid building with real data."""
        aligner = GMMPointAlignment()
        aligner.build_grid(real_data['scene'])

        info = aligner.get_grid_info()
        assert info['built'] is True
        assert info['num_spheres'] == 651511

    def test_real_data_query(self, real_data):
        """Test query with real data."""
        aligner = GMMPointAlignment()
        aligner.build_grid(real_data['scene'])

        # Sample points for faster test
        points = real_data['pointcloud'][:1000]
        result = aligner.query(points)

        assert result.topk_sphere_ids.shape == (1000, 8)
        assert result.query_time_ms < 1000  # Should be fast

    def test_real_data_register(self, real_data):
        """Test registration with real data."""
        aligner = GMMPointAlignment()
        aligner.build_grid(real_data['scene'])

        # Sample points for faster test
        points = real_data['pointcloud'][:500]

        config = RegistrationConfig(
            num_iters=30,
            lr=0.01,
            multi_init=False,
            verbose=False,
        )
        result = aligner.register(points, config=config)

        assert result.transform.shape == (4, 4)
        assert result.num_iters <= 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
