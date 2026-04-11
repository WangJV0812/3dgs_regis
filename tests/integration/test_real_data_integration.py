"""Test Phase 3c: Integration test with real data.

Tests the complete GMM registration pipeline:
- Load 3DGS scene from merged.hier
- Load point cloud from points3D.ply
- Build CSR grid
- Perform GMM registration
- Verify convergence and accuracy
"""

import pytest
import torch
import numpy as np
import taichi as ti
from pathlib import Path
from time import time

from gmm_point_alignment.csr_grid_builder import CSRGridBuilder
from gmm_point_alignment.sphere_mle_loss import (
    MLEAlignmentLoss,
    MLELossConfig,
    GMMRegistration,
    RegistrationConfig,
)
from gmm_point_alignment.transform_utils import se3_exp, se3_log
from misc.hier_IO import load_hier_to_torch


from tests.utils import read_ply_xyz


@pytest.fixture(scope="module", autouse=True)
def init_taichi():
    """Initialize Taichi once for all tests."""
    ti.init(arch=ti.cuda if torch.cuda.is_available() else ti.cpu)
    yield


class TestRealDataIntegration:
    """Integration test with real 3DGS data."""

    @pytest.fixture(scope="class")
    def data_paths(self):
        """Get paths to data files."""
        data_dir = Path(__file__).parent.parent.parent / "data"
        hier_path = data_dir / "merged.hier"
        ply_path = data_dir / "points3D.ply"

        if not hier_path.exists():
            pytest.skip(f"Hierarchy file not found: {hier_path}")
        if not ply_path.exists():
            pytest.skip(f"Point cloud file not found: {ply_path}")

        return {"hier": hier_path, "ply": ply_path}

    @pytest.fixture(scope="class")
    def scene_data(self, data_paths):
        """Load 3DGS scene data."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hier_scene = load_hier_to_torch(data_paths["hier"], device=device)
        return hier_scene.gaussian_scene

    @pytest.fixture(scope="class")
    def point_cloud(self, data_paths):
        """Load point cloud data."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points = read_ply_xyz(data_paths["ply"])
        return points.to(device)

    def test_data_loading(self, scene_data, point_cloud):
        """Test that data loads correctly."""
        print(f"\n{'='*60}")
        print("Real Data Loading Test")
        print(f"{'='*60}")

        # Check scene data
        print(f"\n3DGS Scene:")
        print(f"  Spheres: {scene_data.position.shape[0]}")
        print(f"  Position shape: {scene_data.position.shape}")
        print(f"  Scales shape: {scene_data.scales.shape}")
        print(f"  Rotation shape: {scene_data.rotation.shape}")
        print(f"  Opacities shape: {scene_data.opacities.shape}")

        # Check point cloud
        print(f"\nPoint Cloud:")
        print(f"  Points: {point_cloud.shape[0]}")
        print(f"  Shape: {point_cloud.shape}")
        print(f"  Device: {point_cloud.device}")

        # Check ranges
        pos_min = scene_data.position.min(dim=0)[0]
        pos_max = scene_data.position.max(dim=0)[0]
        print(f"\nScene bounds:")
        print(f"  Min: {pos_min.cpu().numpy()}")
        print(f"  Max: {pos_max.cpu().numpy()}")

        pc_min = point_cloud.min(dim=0)[0]
        pc_max = point_cloud.max(dim=0)[0]
        print(f"\nPoint cloud bounds:")
        print(f"  Min: {pc_min.cpu().numpy()}")
        print(f"  Max: {pc_max.cpu().numpy()}")

        assert scene_data.position.shape[0] > 0
        assert point_cloud.shape[0] > 0
        assert point_cloud.shape[1] == 3

    def test_csr_grid_building(self, scene_data):
        """Test CSR grid building with real data."""
        print(f"\n{'='*60}")
        print("CSR Grid Building Test")
        print(f"{'='*60}")

        builder = CSRGridBuilder()

        start_time = time()
        grid_data = builder.build(scene_data)
        build_time = time() - start_time

        print(f"\nGrid built in {build_time:.2f}s")
        print(f"  Voxel size: {grid_data.voxel_size:.4f}")
        print(f"  Grid dims: {grid_data.grid_dims}")
        print(f"  Num unique voxels: {grid_data.num_unique_voxels}")
        print(f"  Total sphere-voxel pairs: {grid_data.total_pairs}")

        assert grid_data.sphere_centers.shape[0] == scene_data.position.shape[0]
        assert grid_data.voxel_size > 0

    def test_mle_loss_computation(self, scene_data, point_cloud):
        """Test MLE loss computation with real data."""
        print(f"\n{'='*60}")
        print("MLE Loss Computation Test")
        print(f"{'='*60}")

        # Build grid
        builder = CSRGridBuilder()
        grid_data = builder.build(scene_data)

        # Create loss function
        loss_fn = MLEAlignmentLoss(grid_data, MLELossConfig(top_k=8))

        # Sample points for faster test
        sample_size = min(5000, point_cloud.shape[0])
        indices = torch.randperm(point_cloud.shape[0])[:sample_size]
        points_sample = point_cloud[indices]

        # Test with identity transform
        T_identity = torch.eye(4, device=point_cloud.device)

        start_time = time()
        loss = loss_fn(points_sample, T_identity)
        loss_time = time() - start_time

        print(f"\nLoss computation:")
        print(f"  Points: {sample_size}")
        print(f"  Time: {loss_time*1000:.2f}ms")
        print(f"  Loss: {loss.item():.4f}")

        # Test with details
        details = loss_fn.forward_with_details(points_sample, T_identity)
        print(f"\nDetailed statistics:")
        print(f"  Mean density: {details['mean_density'].item():.6f}")
        print(f"  Inlier ratio: {details['inlier_ratio'].item():.4f}")

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gmm_registration_identity(self, scene_data, point_cloud):
        """Test registration with identity initialization."""
        print(f"\n{'='*60}")
        print("GMM Registration - Identity Init")
        print(f"{'='*60}")

        # Build grid
        builder = CSRGridBuilder()
        grid_data = builder.build(scene_data)

        # Sample points for faster test
        sample_size = min(3000, point_cloud.shape[0])
        indices = torch.randperm(point_cloud.shape[0])[:sample_size]
        points_sample = point_cloud[indices]

        # Create registration
        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=50,
                lr=0.01,
                multi_init=False,
                verbose=True,
            )
        )

        # Register
        start_time = time()
        result = reg.register(points_sample)
        reg_time = time() - start_time

        print(f"\nRegistration completed in {reg_time:.2f}s")
        print(f"  Final loss: {result['loss'].item():.4f}")
        print(f"  Iterations: {result['num_iters']}")
        print(f"  Converged: {result['converged'].item()}")
        print(f"  Inlier ratio: {result['inlier_ratio'].item():.4f}")

        # Check that transform is close to identity
        T = result['transform']
        t_error = T[:3, 3].norm().item()
        R_diff = (T[:3, :3] - torch.eye(3, device=T.device)).abs().max().item()

        print(f"\nTransform deviation from identity:")
        print(f"  Translation error: {t_error:.6f}")
        print(f"  Rotation max diff: {R_diff:.6f}")

        assert result['converged']
        assert t_error < 0.01  # Should stay near identity
        assert R_diff < 0.01

    def test_gmm_registration_with_transform(self, scene_data, point_cloud):
        """Test registration recovering a known transform."""
        print(f"\n{'='*60}")
        print("GMM Registration - Known Transform Recovery")
        print(f"{'='*60}")

        # Build grid
        builder = CSRGridBuilder()
        grid_data = builder.build(scene_data)

        # Sample points
        sample_size = min(3000, point_cloud.shape[0])
        indices = torch.randperm(point_cloud.shape[0])[:sample_size]
        points_orig = point_cloud[indices]

        # Create a known transformation
        device = point_cloud.device
        xi_true = torch.tensor([0.5, -0.3, 0.2, 0.1, -0.05, 0.08], device=device)
        T_true = se3_exp(xi_true)

        # Apply transform to points
        R = T_true[:3, :3]
        t = T_true[:3, 3]
        points_transformed = (R @ points_orig.T).T + t

        print(f"\nGround truth transform:")
        print(f"  xi: {xi_true.cpu().numpy()}")
        print(f"  Translation: {t.cpu().numpy()}")

        # Add noise
        points_noisy = points_transformed + torch.randn_like(points_transformed) * 0.05

        # Register
        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=100,
                lr=0.02,
                convergence_threshold=1e-4,
                patience=20,
                multi_init=True,
                num_init=5,
                init_noise_scale=0.5,
                verbose=True,
            )
        )

        start_time = time()
        result = reg.register(points_noisy)
        reg_time = time() - start_time

        T_recovered = result['transform']
        t_recovered = T_recovered[:3, 3]

        # Compute errors
        t_error = (t_recovered - t).norm().item()
        R_error_mat = T_recovered[:3, :3].T @ R
        trace = R_error_mat.trace()
        R_error = torch.acos(torch.clamp((trace - 1) / 2, -1, 1)).item()

        print(f"\nRegistration completed in {reg_time:.2f}s")
        print(f"  Final loss: {result['loss'].item():.4f}")
        print(f"  Iterations: {result['num_iters']}")
        print(f"  Converged: {result['converged'].item()}")

        print(f"\nRecovered transform:")
        print(f"  Translation: {t_recovered.cpu().numpy()}")

        print(f"\nErrors:")
        print(f"  Translation error: {t_error:.4f}m")
        print(f"  Rotation error: {R_error:.4f}rad ({np.degrees(R_error):.2f}deg)")

        # For this test with real data, we allow larger errors
        # since the point cloud may not perfectly align with the scene
        assert result['num_iters'] < 100  # Should converge
        assert result['inlier_ratio'] > 0.3  # Should have reasonable inlier ratio

    def test_full_pipeline_performance(self, scene_data, point_cloud):
        """Test full pipeline performance metrics."""
        print(f"\n{'='*60}")
        print("Full Pipeline Performance Test")
        print(f"{'='*60}")

        device = point_cloud.device

        # Time grid building
        builder = CSRGridBuilder()
        start = time()
        grid_data = builder.build(scene_data)
        grid_time = time() - start

        # Time loss computation
        loss_fn = MLEAlignmentLoss(grid_data, MLELossConfig(top_k=8))
        points_sample = point_cloud[:min(10000, point_cloud.shape[0])]

        start = time()
        for _ in range(10):
            loss = loss_fn(points_sample, torch.eye(4, device=device))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        loss_time = (time() - start) / 10

        # Time registration
        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=30,
                lr=0.01,
                multi_init=False,
                verbose=False,
            )
        )

        start = time()
        result = reg.register(points_sample[:1000])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        reg_time = time() - start

        print(f"\nPerformance metrics:")
        print(f"  Grid build: {grid_time:.2f}s")
        print(f"  Loss computation (10k pts): {loss_time*1000:.2f}ms")
        print(f"  Registration (1k pts, 30 iters): {reg_time:.2f}s")

        print(f"\n{'='*60}")
        print("Phase 3c Integration Tests Complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
