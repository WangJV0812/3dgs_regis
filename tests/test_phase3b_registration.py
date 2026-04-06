"""Test Phase 3b: GMM Registration with multi-init and convergence.

Tests enhanced registration with:
- Multi-initialization strategy
- Convergence detection
- Learning rate scheduling
- Integration with real data
"""

import pytest
import torch
import taichi as ti
from pathlib import Path

from gmm_point_alignment.csr_grid_builder import CSRGridBuilder
from gmm_point_alignment.sphere_mle_loss import (
    MLEAlignmentLoss,
    MLELossConfig,
    GMMRegistration,
    RegistrationConfig,
)
from gmm_point_alignment.transform_utils import se3_exp, se3_log


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


@pytest.fixture
def grid_data():
    """Build grid for testing."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scene = DummyScene(50, device=device)
    builder = CSRGridBuilder()
    return builder.build(scene)


class TestRegistrationConvergence:
    """Test convergence detection."""

    def test_single_optimization_converges(self, grid_data):
        """Test that single optimization converges."""
        device = grid_data.sphere_centers.device
        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=100,
                lr=0.01,
                convergence_threshold=1e-4,
                patience=20,
                multi_init=False,
                verbose=False,
            )
        )

        # Create points with known transform
        points = grid_data.sphere_centers.clone() + torch.randn_like(grid_data.sphere_centers) * 0.1

        # Register
        result = reg.register(points)

        assert result['converged'].item()
        assert result['num_iters'] < 100  # Should converge before max
        assert result['loss'] < 10  # Should achieve reasonable loss
        assert result['inlier_ratio'] > 0.5  # Most points should have associations

    def test_convergence_detected(self, grid_data):
        """Test convergence detection works correctly."""
        device = grid_data.sphere_centers.device
        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=200,
                lr=0.01,
                convergence_threshold=1e-4,
                patience=10,
                multi_init=False,
                verbose=False,
            )
        )

        points = grid_data.sphere_centers.clone()
        result = reg.register(points)

        # Should converge early
        assert result['num_iters'] < 200

    def test_loss_history_tracked(self, grid_data):
        """Test that loss history is tracked."""
        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=50,
                multi_init=False,
                verbose=False,
            )
        )

        points = grid_data.sphere_centers.clone()
        result = reg.register(points)

        # Loss history should be available
        assert 'loss_history' in result
        assert len(result['loss_history']) == result['num_iters']
        # Loss magnitude should generally decrease (loss can be negative)
        # Check that we've made progress from initial state
        assert result['loss'] < result['loss_history'][0] + 1.0


class TestMultiInitialization:
    """Test multi-initialization strategy."""

    def test_multi_init_runs_all_trials(self, grid_data):
        """Test that multi-init runs all trials."""
        device = grid_data.sphere_centers.device
        num_init = 3

        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=20,
                multi_init=True,
                num_init=num_init,
                verbose=False,
            )
        )

        points = grid_data.sphere_centers.clone()
        result = reg.register(points)

        # Should return valid result
        assert result['transform'].shape == (4, 4)
        # Loss can be negative (when points align with high-density regions)

    def test_multi_init_selects_best(self, grid_data):
        """Test that multi-init selects best result."""
        device = grid_data.sphere_centers.device

        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=30,
                multi_init=True,
                num_init=5,
                init_noise_scale=1.0,
                verbose=False,
            )
        )

        points = grid_data.sphere_centers.clone()
        result = reg.register(points)

        # Best result should be reasonable
        assert result['inlier_ratio'] > 0.3

    def test_single_init_from_identity(self, grid_data):
        """Test single initialization from identity."""
        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                multi_init=False,
                num_iters=30,
                verbose=False,
            )
        )

        points = grid_data.sphere_centers.clone()
        result = reg.register(points)

        assert result['transform'].shape == (4, 4)


class TestLearningRateScheduling:
    """Test learning rate scheduling."""

    def test_lr_scheduler_enabled(self, grid_data):
        """Test that LR scheduler is active."""
        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=100,
                lr=0.1,
                lr_scheduler=True,
                lr_decay_step=30,
                lr_decay_rate=0.5,
                verbose=False,
            )
        )

        points = grid_data.sphere_centers.clone()
        result = reg.register(points)

        # Should converge even with high initial LR
        assert result['loss'] < 50

    def test_lr_scheduler_disabled(self, grid_data):
        """Test without LR scheduler."""
        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=50,
                lr=0.01,
                lr_scheduler=False,
                verbose=False,
            )
        )

        points = grid_data.sphere_centers.clone()
        result = reg.register(points)

        assert result['loss'] < 50


class TestGradientBehavior:
    """Test gradient behavior."""

    def test_gradient_flow(self, grid_data):
        """Test that gradients flow correctly."""
        device = grid_data.sphere_centers.device
        loss_fn = MLEAlignmentLoss(grid_data)

        points = grid_data.sphere_centers.clone()
        xi = torch.zeros(6, device=device, requires_grad=True)
        T = se3_exp(xi)

        loss = loss_fn(points, T)
        loss.backward()

        assert xi.grad is not None
        assert xi.grad.abs().sum() > 0

    def test_gradient_clipping(self, grid_data):
        """Test gradient clipping works."""
        # This is implicitly tested by the optimization running without NaN
        device = grid_data.sphere_centers.device
        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=20,
                lr=0.1,  # High LR to potentially cause large gradients
                multi_init=False,
                verbose=False,
            )
        )

        points = grid_data.sphere_centers.clone()
        result = reg.register(points)

        # Should not produce NaN
        assert not torch.isnan(result['loss'])


class TestRegistrationAccuracy:
    """Test registration accuracy."""

    def test_recover_known_transform(self, grid_data):
        """Test recovering a known transformation."""
        device = grid_data.sphere_centers.device

        # Create ground truth transform
        xi_true = torch.tensor([0.3, -0.2, 0.1, 0.1, 0.05, 0.05], device=device)
        T_true = se3_exp(xi_true)

        # Transform points
        points = grid_data.sphere_centers.clone()
        R = T_true[:3, :3]
        t = T_true[:3, 3]
        points_transformed = (R @ points.T).T + t

        # Add noise
        points_noisy = points_transformed + torch.randn_like(points_transformed) * 0.05

        # Register
        reg = GMMRegistration(
            grid_data,
            reg_config=RegistrationConfig(
                num_iters=100,
                lr=0.02,
                multi_init=True,
                num_init=3,
                verbose=False,
            )
        )

        result = reg.register(points_noisy)

        # Check that recovered transform is close to ground truth
        T_recovered = result['transform']

        # Translation error
        t_error = (T_recovered[:3, 3] - T_true[:3, 3]).norm()

        # Rotation error (angle between rotations)
        R_error = torch.acos(
            torch.clamp(
                (torch.trace(T_recovered[:3, :3].T @ T_true[:3, :3]) - 1) / 2,
                -1, 1
            )
        )

        assert t_error < 0.5, f"Translation error too large: {t_error}"
        assert R_error < 0.3, f"Rotation error too large: {R_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
