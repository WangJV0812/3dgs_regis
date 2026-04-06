"""MLE Loss for point cloud registration using 3D Gaussian Splatting.

Implements negative log-likelihood loss for point-to-sphere registration.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from .csr_grid_builder import CSRGridData
from .csr_grid_querier import CSRGridQuerier, CSRGridQuerierConfig
from gmm_point_alignment.transform_utils import transform_points, se3_exp, sim3_exp


@dataclass
class MLELossConfig:
    """Configuration for MLE alignment loss.

    Args:
        top_k: Number of top spheres to consider per point (default: 8)
        min_opacity: Minimum opacity threshold for valid spheres (default: 1e-3)
        use_weighted: Use opacity as mixture weight (default: True)
        batch_size: Batch size for processing large point clouds (default: 10000)
    """
    top_k: int = 8
    min_opacity: float = 1e-3
    use_weighted: bool = True
    batch_size: int = 10000


class MLEAlignmentLoss(nn.Module):
    """Negative log-likelihood loss for point-to-sphere registration.

    Uses soft GMM assignment with Top-K approximation for efficiency.

    Example:
        >>> loss_fn = MLEAlignmentLoss(grid_data, MLELossConfig(top_k=8))
        >>> points = torch.randn(1000, 3)
        >>> transform = torch.eye(4)
        >>> loss = loss_fn(points, transform)
        >>> loss.backward()
    """

    def __init__(
        self,
        grid_data: CSRGridData,
        config: Optional[MLELossConfig] = None,
    ):
        super().__init__()
        self.grid_data = grid_data
        self.config = config or MLELossConfig()

        # Create querier
        self.querier = CSRGridQuerier(
            grid_data,
            CSRGridQuerierConfig(
                top_k=self.config.top_k,
                batch_size=self.config.batch_size
            )
        )

        # Precompute log normalization factors
        # Ensure positive values for log
        norm_factor_safe = torch.clamp(grid_data.norm_factor, min=1e-10)
        self.register_buffer('log_norm_factors', torch.log(norm_factor_safe))

        # Precompute log opacities if using weighted mixture
        if hasattr(grid_data, 'opacities') and grid_data.opacities is not None:
            opacities_safe = torch.clamp(grid_data.opacities, min=1e-10)
            self.register_buffer('log_opacities', torch.log(opacities_safe))
            self.has_opacities = True
        else:
            self.has_opacities = False

    def forward(
        self,
        points: torch.Tensor,
        transform: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            points: [N, 3] query points
            transform: [4, 4] transformation matrix (differentiable)

        Returns:
            loss: scalar (mean NLL over all points)
        """
        # Transform points
        points_transformed = transform_points(points, transform)

        # Compute NLL
        nll = self._compute_nll(points_transformed)

        return nll

    def forward_with_details(
        self,
        points: torch.Tensor,
        transform: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with detailed statistics.

        Args:
            points: [N, 3] query points
            transform: [4, 4] transformation matrix

        Returns:
            dict with:
                loss: scalar
                nll_per_point: [N] per-point NLL
                mean_density: scalar
                inlier_ratio: scalar (fraction with valid associations)
        """
        points_transformed = transform_points(points, transform)

        nll, details = self._compute_nll_with_details(points_transformed)

        return {
            'loss': nll,
            'nll_per_point': details['nll_per_point'],
            'mean_density': details['mean_density'],
            'inlier_ratio': details['inlier_ratio'],
        }

    def _compute_nll(self, points: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood (vectorized).

        Args:
            points: [N, 3] transformed points

        Returns:
            nll: scalar (mean)
        """
        # Query Top-K candidates
        query_result = self.querier.query(points)

        N = points.shape[0]
        K = self.config.top_k

        # Handle empty result
        if query_result.topk_sphere_ids.shape[1] == 0:
            return torch.tensor(float('inf'), device=points.device)

        # Gather sphere parameters [N, K, ...]
        sphere_ids = query_result.topk_sphere_ids  # [N, K]
        sphere_ids_clamped = sphere_ids.clamp(min=0)  # Replace -1 with 0 for indexing

        # Centers [N, K, 3]
        centers = self.grid_data.sphere_centers[sphere_ids_clamped]

        # Covariance inverse [N, K, 3, 3]
        cov_inv = self.grid_data.cov_inv[sphere_ids_clamped]

        # Compute Mahalanobis distance
        diff = points.unsqueeze(1) - centers  # [N, K, 3]

        # Mahalanobis: (x-mu)^T @ Sigma^-1 @ (x-mu)
        # Using einsum for batch matrix-vector product
        temp = torch.einsum('nkij,nkj->nki', cov_inv, diff)  # [N, K, 3]
        mahalanobis = (diff * temp).sum(dim=-1)  # [N, K]

        # Log densities
        log_norm = self.log_norm_factors[sphere_ids_clamped]  # [N, K]
        log_densities = -0.5 * mahalanobis + log_norm  # [N, K]

        # Create mask for valid associations (sphere_id >= 0)
        valid_mask = (sphere_ids >= 0).float()  # [N, K]

        # Set invalid entries to very negative log density
        log_densities = log_densities * valid_mask + (-1000) * (1 - valid_mask)

        # Weights
        if self.config.use_weighted and self.has_opacities:
            log_weights = self.log_opacities[sphere_ids_clamped]  # [N, K]
            # Mask invalid weights
            log_weights = log_weights * valid_mask + (-1000) * (1 - valid_mask)
        else:
            # Uniform weights among valid candidates
            num_valid = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [N, 1]
            log_weights = torch.log(valid_mask / num_valid + 1e-10)  # [N, K]

        # Numerically stable log-sum-exp for mixture likelihood
        # log(sum(w_i * exp(log_d_i))) = log_sum_exp(log_w_i + log_d_i)
        log_weighted = log_weights + log_densities  # [N, K]
        log_likelihood = torch.logsumexp(log_weighted, dim=-1)  # [N]

        # Negative log-likelihood
        nll = -log_likelihood  # [N]

        # Handle points with no valid associations
        has_valid = (valid_mask.sum(dim=-1) > 0).float()  # [N]
        nll = nll * has_valid + (1 - has_valid) * 1000  # Large penalty for no association

        return nll.mean()

    def _compute_nll_with_details(
        self,
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute NLL with detailed statistics."""
        query_result = self.querier.query(points)

        N = points.shape[0]
        K = self.config.top_k

        # Handle empty result
        if query_result.topk_sphere_ids.shape[1] == 0:
            inf_tensor = torch.tensor(float('inf'), device=points.device)
            return inf_tensor, {
                'nll_per_point': torch.full((N,), float('inf'), device=points.device),
                'mean_density': torch.tensor(0.0, device=points.device),
                'inlier_ratio': torch.tensor(0.0, device=points.device),
            }

        sphere_ids = query_result.topk_sphere_ids
        sphere_ids_clamped = sphere_ids.clamp(min=0)
        valid_mask = (sphere_ids >= 0).float()

        centers = self.grid_data.sphere_centers[sphere_ids_clamped]
        cov_inv = self.grid_data.cov_inv[sphere_ids_clamped]

        diff = points.unsqueeze(1) - centers
        temp = torch.einsum('nkij,nkj->nki', cov_inv, diff)
        mahalanobis = (diff * temp).sum(dim=-1)

        log_norm = self.log_norm_factors[sphere_ids_clamped]
        log_densities = -0.5 * mahalanobis + log_norm
        log_densities = log_densities * valid_mask + (-1000) * (1 - valid_mask)

        # Compute actual densities for statistics
        densities = torch.exp(log_densities) * valid_mask

        if self.config.use_weighted and self.has_opacities:
            log_weights = self.log_opacities[sphere_ids_clamped]
            log_weights = log_weights * valid_mask + (-1000) * (1 - valid_mask)
        else:
            num_valid = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            log_weights = torch.log(valid_mask / num_valid + 1e-10)

        log_weighted = log_weights + log_densities
        log_likelihood = torch.logsumexp(log_weighted, dim=-1)
        nll = -log_likelihood

        has_valid = (valid_mask.sum(dim=-1) > 0).float()
        nll = nll * has_valid + (1 - has_valid) * 1000

        # Statistics
        details = {
            'nll_per_point': nll,
            'mean_density': densities.sum() / valid_mask.sum().clamp(min=1),
            'inlier_ratio': has_valid.mean(),
        }

        return nll.mean(), details


@dataclass
class RegistrationConfig:
    """Configuration for GMM registration optimization.

    Args:
        num_iters: Maximum optimization iterations (default: 100)
        lr: Learning rate (default: 0.01)
        convergence_threshold: Loss change threshold for convergence (default: 1e-4)
        patience: Iterations without improvement before early stop (default: 20)
        multi_init: Use multiple random initializations (default: True)
        num_init: Number of random initializations (default: 5)
        init_noise_scale: Scale of random noise for initialization (default: 0.5)
        verbose: Print progress (default: True)
        lr_scheduler: Use learning rate decay (default: True)
        lr_decay_step: Step size for LR decay (default: 30)
        lr_decay_rate: LR decay factor (default: 0.5)
        use_scale: Optimize scale factor in addition to R and t (default: False)
        scale_lr: Learning rate for scale optimization (default: 0.001)
    """
    num_iters: int = 1000
    lr: float = 0.01
    convergence_threshold: float = 1e-4
    patience: int = 20
    multi_init: bool = True
    num_init: int = 5
    init_noise_scale: float = 0.5
    verbose: bool = True
    lr_scheduler: bool = True
    lr_decay_step: int = 30
    lr_decay_rate: float = 0.5
    use_scale: bool = False
    scale_lr: float = 0.001


class GMMRegistration:
    """Complete GMM-based registration pipeline with multi-init and convergence.

    Combines CSR grid querying with MLE optimization.

    Example:
        >>> reg = GMMRegistration(grid_data)
        >>> points = torch.randn(1000, 3, device='cuda')
        >>> result = reg.register(points)  # Multi-init optimization
        >>> print(f"Best loss: {result['loss']:.4f}")
    """

    def __init__(
        self,
        grid_data: CSRGridData,
        loss_config: Optional[MLELossConfig] = None,
        reg_config: Optional[RegistrationConfig] = None,
    ):
        self.grid_data = grid_data
        self.loss_fn = MLEAlignmentLoss(grid_data, loss_config)
        self.config = reg_config or RegistrationConfig()

    def register(
        self,
        points: torch.Tensor,
        initial_transform: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Register point cloud to scene using MLE with multi-init.

        Args:
            points: [N, 3] point cloud
            initial_transform: [4, 4] initial guess (optional, overrides multi_init)

        Returns:
            dict with:
                transform: [4, 4] optimal transformation
                loss: final loss value
                inlier_ratio: fraction of points with valid associations
                mean_density: mean density of associations
                num_iters: iterations performed
                converged: whether optimization converged
        """
        device = points.device

        if initial_transform is not None:
            # Single optimization from provided initial
            result = self._optimize_single(points, initial_transform)
            return result

        if self.config.multi_init:
            # Multiple random initializations
            return self._register_multi_init(points)
        else:
            # Single optimization from identity
            result = self._optimize_single(points, torch.eye(4, device=device))
            return result

    def _optimize_single(
        self,
        points: torch.Tensor,
        initial_transform: torch.Tensor,
        initial_log_scale: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Single optimization run with convergence detection."""
        from gmm_point_alignment.transform_utils import se3_log, sim3_log, sim3_exp

        device = points.device

        # Initialize parameters
        if self.config.use_scale:
            # Extract initial scale from transform if not identity
            xi, log_scale = sim3_log(initial_transform)
            xi = xi.clone().detach().requires_grad_(True)
            log_scale = log_scale.clone().detach().requires_grad_(True)
            params = [xi, log_scale]
        else:
            xi = se3_log(initial_transform)
            xi = xi.clone().detach().requires_grad_(True)
            params = [xi]
            log_scale = None

        # Optimizer with different LR for scale
        if self.config.use_scale:
            optimizer = torch.optim.Adam([
                {'params': [xi], 'lr': self.config.lr},
                {'params': [log_scale], 'lr': self.config.scale_lr},
            ])
        else:
            optimizer = torch.optim.Adam([xi], lr=self.config.lr)

        # LR scheduler
        if self.config.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_decay_step,
                gamma=self.config.lr_decay_rate
            )

        # Tracking
        best_loss = float('inf')
        best_xi = xi.clone().detach()
        best_log_scale = log_scale.clone().detach() if self.config.use_scale else None
        patience_counter = 0
        loss_history = []
        converged = False

        for i in range(self.config.num_iters):
            optimizer.zero_grad()

            # Forward
            if self.config.use_scale:
                T = sim3_exp(xi, log_scale)
            else:
                T = se3_exp(xi)
            loss = self.loss_fn(points, T)

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(xi, max_norm=1.0)
            if self.config.use_scale:
                torch.nn.utils.clip_grad_norm_(log_scale, max_norm=0.1)

            optimizer.step()

            if self.config.lr_scheduler:
                scheduler.step()

            # Tracking
            loss_val = loss.item()
            loss_history.append(loss_val)

            # Update best
            if loss_val < best_loss - self.config.convergence_threshold:
                best_loss = loss_val
                best_xi = xi.clone().detach()
                if self.config.use_scale:
                    best_log_scale = log_scale.clone().detach()
                patience_counter = 0
            else:
                patience_counter += 1

            # Convergence check
            if patience_counter >= self.config.patience:
                converged = True
                if self.config.verbose:
                    print(f"  Converged at iter {i}")
                break

            # Logging
            if self.config.verbose and i % 10 == 0:
                with torch.no_grad():
                    if self.config.use_scale:
                        T_debug = sim3_exp(xi, log_scale)
                        scale_val = torch.exp(log_scale).item()
                    else:
                        T_debug = se3_exp(xi)
                        scale_val = 1.0
                    details = self.loss_fn.forward_with_details(points, T_debug)
                    lr_current = optimizer.param_groups[0]['lr']
                    scale_info = f", s={scale_val:.4f}" if self.config.use_scale else ""
                    print(f"  Iter {i:3d}: loss={loss_val:.4f}, "
                          f"inlier={details['inlier_ratio'].item():.3f}{scale_info}, "
                          f"lr={lr_current:.4e}")

        # Return best result
        with torch.no_grad():
            if self.config.use_scale:
                T_best = sim3_exp(best_xi, best_log_scale)
                final_scale = torch.exp(best_log_scale)
            else:
                T_best = se3_exp(best_xi)
                final_scale = torch.tensor(1.0)
            final_details = self.loss_fn.forward_with_details(points, T_best)

        result = {
            'transform': T_best,
            'loss': torch.tensor(best_loss),
            'inlier_ratio': final_details['inlier_ratio'],
            'mean_density': final_details['mean_density'],
            'num_iters': len(loss_history),
            'converged': torch.tensor(converged),
            'loss_history': torch.tensor(loss_history),
            'scale': final_scale,
        }

        return result

    def _register_multi_init(
        self,
        points: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Multi-initialization strategy."""
        device = points.device

        if self.config.verbose:
            print(f"Starting multi-init registration ({self.config.num_init} trials, scale={self.config.use_scale})")

        # Generate random initializations
        initializations = []
        scale_inits = []

        # Always include identity
        initializations.append(torch.eye(4, device=device))
        scale_inits.append(0.0)  # log_scale = 0 -> scale = 1.0

        # Add random perturbations
        for _ in range(self.config.num_init - 1):
            xi = torch.randn(6, device=device) * self.config.init_noise_scale
            T = se3_exp(xi)
            initializations.append(T)
            if self.config.use_scale:
                # Random log scale: -0.2 to 0.2 (scale 0.8 to 1.2)
                log_s = (torch.rand(1, device=device).item() - 0.5) * 0.4
                scale_inits.append(log_s)
            else:
                scale_inits.append(0.0)

        # Run optimization for each
        results = []
        for i, (T_init, s_init) in enumerate(zip(initializations, scale_inits)):
            if self.config.verbose:
                scale_str = f" (scale={torch.exp(torch.tensor(s_init)).item():.3f})" if self.config.use_scale else ""
                print(f"\nTrial {i+1}/{len(initializations)}:{scale_str}")

            result = self._optimize_single(points, T_init, s_init)
            results.append(result)

        # Select best result
        losses = torch.stack([r['loss'] for r in results])
        best_idx = losses.argmin()
        best_result = results[best_idx]

        if self.config.verbose:
            print(f"\n{'='*60}")
            scale_str = f", scale={best_result['scale'].item():.4f}" if self.config.use_scale else ""
            print(f"Best result from trial {best_idx+1}: loss={best_result['loss'].item():.4f}{scale_str}")
            print(f"{'='*60}")

        return best_result

    def register_with_icp_init(
        self,
        points: torch.Tensor,
        icp_transform: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Register with ICP initialization for coarse alignment.

        Args:
            points: [N, 3] point cloud
            icp_transform: [4, 4] ICP coarse alignment

        Returns:
            dict with registration results
        """
        if self.config.verbose:
            print("Starting GMM refinement from ICP initialization")

        return self._optimize_single(points, icp_transform)
