"""MLE Loss for point cloud registration using 3D Gaussian Splatting.

Implements negative log-likelihood loss for point-to-sphere registration.
"""

import torch
import torch.nn as nn
import numpy as np
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
        # Robust kernel options
        robust_kernel: str = "none"  # "none", "huber", "cauchy", "geman_mcclure"
        kernel_threshold: float = 0.1  # Threshold for robust kernels
        use_point_confidence: bool = False  # Use per-point confidence weights
    """
    top_k: int = 8
    min_opacity: float = 1e-3
    use_weighted: bool = True
    batch_size: int = 10000
    robust_kernel: str = "none"
    kernel_threshold: float = 0.1
    use_point_confidence: bool = False


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

    def _apply_robust_kernel(self, nll: torch.Tensor) -> torch.Tensor:
        """Apply robust kernel to negative log-likelihood.

        Args:
            nll: Raw NLL values [N]

        Returns:
            Robust NLL values [N]
        """
        kernel = self.config.robust_kernel
        c = self.config.kernel_threshold

        if kernel == "none" or c <= 0:
            return nll
        elif kernel == "huber":
            # Huber loss: quadratic for small values, linear for large
            mask = nll <= c
            robust_nll = torch.where(
                mask,
                nll,  # quadratic region
                2 * c * torch.sqrt(nll) - c  # linear region
            )
            return robust_nll
        elif kernel == "cauchy":
            # Cauchy loss: c^2 * log(1 + (x/c)^2)
            return c**2 * torch.log(1 + nll / (c**2))
        elif kernel == "geman_mcclure":
            # Geman-McClure: x^2 / (c^2 + x^2) * c^2
            return nll * c**2 / (c**2 + nll)
        else:
            return nll

    def forward(
        self,
        points: torch.Tensor,
        transform: torch.Tensor,
        point_confidence: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            points: [N, 3] query points
            transform: [4, 4] transformation matrix (differentiable)
            point_confidence: [N] optional per-point confidence weights

        Returns:
            loss: scalar (mean NLL over all points)
        """
        # Transform points
        points_transformed = transform_points(points, transform)

        # Compute NLL
        nll = self._compute_nll(points_transformed, point_confidence)

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

    def _compute_nll(self, points: torch.Tensor, point_confidence: torch.Tensor = None) -> torch.Tensor:
        """Compute negative log-likelihood (vectorized).

        Args:
            points: [N, 3] transformed points
            point_confidence: [N] optional per-point confidence weights

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

        # Apply robust kernel
        nll = self._apply_robust_kernel(nll)

        # Apply point confidence weights
        if point_confidence is not None and self.config.use_point_confidence:
            nll = nll * point_confidence
            return nll.sum() / (point_confidence.sum() + 1e-8)

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
        lr: Base learning rate (default: 0.01)
        lr_translation: Learning rate for translation (default: 0.01)
        lr_rotation: Learning rate for rotation (default: 0.001)
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
        debug: Enable debug mode with visualization (default: False)
        debug_gt_transform: Ground truth transform for error computation in debug mode (default: None)
        use_pca_init: Use PCA-based initialization (default: False)
        pca_scale_range: Search range for scale initialization [min, max] (default: [0.1, 10.0])
    """
    num_iters: int = 1000
    lr: float = 0.01
    lr_translation: float = 0.01
    lr_rotation: float = 0.001
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
    debug: bool = False
    debug_gt_transform: Optional[torch.Tensor] = None
    use_pca_init: bool = False
    pca_scale_range: tuple = (0.1, 10.0)


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
        point_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Register point cloud to scene using MLE with multi-init.

        Args:
            points: [N, 3] point cloud
            initial_transform: [4, 4] initial guess (optional, overrides multi_init)
            point_confidence: [N] optional per-point confidence weights

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
            result = self._optimize_single(points, initial_transform, point_confidence=point_confidence)
            return result

        if self.config.multi_init:
            # Multiple random initializations
            return self._register_multi_init(points, point_confidence=point_confidence)
        else:
            # Single optimization from identity
            result = self._optimize_single(points, torch.eye(4, device=device), point_confidence=point_confidence)
            return result

    def _optimize_single(
        self,
        points: torch.Tensor,
        initial_transform: torch.Tensor,
        initial_log_scale: float = 0.0,
        point_confidence: Optional[torch.Tensor] = None,
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

        # Split xi into translation (first 3) and rotation (last 3)
        xi_t = xi[:3].clone().detach().requires_grad_(True)
        xi_r = xi[3:].clone().detach().requires_grad_(True)

        # Optimizer with different LR for translation, rotation, and scale
        # Translation usually needs larger LR, rotation needs smaller LR
        lr_t = getattr(self.config, 'lr_translation', self.config.lr)
        lr_r = getattr(self.config, 'lr_rotation', self.config.lr * 0.1)

        if self.config.use_scale:
            log_scale = log_scale.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([
                {'params': [xi_t], 'lr': lr_t},
                {'params': [xi_r], 'lr': lr_r},
                {'params': [log_scale], 'lr': self.config.scale_lr},
            ])
        else:
            optimizer = torch.optim.Adam([
                {'params': [xi_t], 'lr': lr_t},
                {'params': [xi_r], 'lr': lr_r},
            ])

        # LR scheduler
        if self.config.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_decay_step,
                gamma=self.config.lr_decay_rate
            )

        # Tracking
        best_loss = float('inf')
        best_xi_t = xi_t.clone().detach()
        best_xi_r = xi_r.clone().detach()
        best_log_scale = log_scale.clone().detach() if self.config.use_scale else None
        patience_counter = 0
        loss_history = []
        converged = False

        # Debug tracking
        if self.config.debug:
            debug_history = {
                'loss': [],
                'rotation_error': [],
                'translation_error': [],
                'scale_error': [],
                'scale': [],
            }

        for i in range(self.config.num_iters):
            optimizer.zero_grad()

            # Reconstruct xi from split parameters
            xi = torch.cat([xi_t, xi_r])

            # Forward
            if self.config.use_scale:
                T = sim3_exp(xi, log_scale)
            else:
                T = se3_exp(xi)
            loss = self.loss_fn(points, T, point_confidence)

            # Backward
            loss.backward()

            # Gradient clipping (separate for translation and rotation)
            torch.nn.utils.clip_grad_norm_(xi_t, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(xi_r, max_norm=0.5)  # Smaller clip for rotation
            if self.config.use_scale:
                torch.nn.utils.clip_grad_norm_(log_scale, max_norm=0.1)

            optimizer.step()

            if self.config.lr_scheduler:
                scheduler.step()

            # Tracking
            loss_val = loss.item()
            loss_history.append(loss_val)

            # Debug tracking
            if self.config.debug and self.config.debug_gt_transform is not None:
                with torch.no_grad():
                    # Reconstruct xi and get current transform
                    xi = torch.cat([xi_t, xi_r])
                    if self.config.use_scale:
                        T_current = sim3_exp(xi, log_scale)
                        scale_val = torch.exp(log_scale).item()
                    else:
                        T_current = se3_exp(xi)
                        scale_val = 1.0

                    # Extract R, t from current transform
                    R_current = T_current[:3, :3]
                    t_current = T_current[:3, 3]

                    # Extract ground truth
                    R_gt = self.config.debug_gt_transform[:3, :3]
                    t_gt = self.config.debug_gt_transform[:3, 3]
                    scale_gt = torch.det(R_gt) ** (1/3)  # Assuming scale is encoded in R

                    # Compute errors
                    # Rotation error
                    R_rel = R_current.T @ R_gt
                    cos_angle = torch.clamp((R_rel.trace() - 1) / 2, -1, 1)
                    rot_error = torch.acos(cos_angle).item() * 180 / 3.14159

                    # Translation error
                    trans_error = (t_current - t_gt).norm().item()

                    # Scale error
                    if self.config.use_scale:
                        scale_err = abs(scale_val - scale_gt.item())
                    else:
                        scale_err = 0.0

                    debug_history['loss'].append(loss_val)
                    debug_history['rotation_error'].append(rot_error)
                    debug_history['translation_error'].append(trans_error)
                    debug_history['scale_error'].append(scale_err)
                    debug_history['scale'].append(scale_val)

            # Update best
            if loss_val < best_loss - self.config.convergence_threshold:
                best_loss = loss_val
                best_xi_t = xi_t.clone().detach()
                best_xi_r = xi_r.clone().detach()
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
                    xi = torch.cat([xi_t, xi_r])
                    if self.config.use_scale:
                        T_debug = sim3_exp(xi, log_scale)
                        scale_val = torch.exp(log_scale).item()
                    else:
                        T_debug = se3_exp(xi)
                        scale_val = 1.0
                    details = self.loss_fn.forward_with_details(points, T_debug)
                    lr_t = optimizer.param_groups[0]['lr']
                    lr_r = optimizer.param_groups[1]['lr']
                    scale_info = f", s={scale_val:.4f}" if self.config.use_scale else ""
                    print(f"  Iter {i:3d}: loss={loss_val:.4f}, "
                          f"inlier={details['inlier_ratio'].item():.3f}{scale_info}, "
                          f"lr_t={lr_t:.4e}, lr_r={lr_r:.4e}")

        # Return best result
        with torch.no_grad():
            # Reconstruct best xi from split parameters
            best_xi = torch.cat([best_xi_t, best_xi_r])
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

        # Plot debug visualization
        if self.config.debug and self.config.debug_gt_transform is not None:
            self._plot_debug_curves(debug_history)

        return result

    def _plot_debug_curves(self, debug_history: dict):
        """Plot training curves for debugging."""
        import matplotlib.pyplot as plt

        num_plots = 4 if self.config.use_scale else 3
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
        if num_plots == 1:
            axes = [axes]

        iters = range(len(debug_history['loss']))

        # Loss curve
        ax = axes[0]
        ax.plot(iters, debug_history['loss'], 'b-', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curve')
        ax.grid(True, alpha=0.3)

        # Rotation error
        ax = axes[1]
        ax.plot(iters, debug_history['rotation_error'], 'r-', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Rotation Error (°)')
        ax.set_title('Rotation Error')
        ax.grid(True, alpha=0.3)

        # Translation error
        ax = axes[2]
        ax.plot(iters, debug_history['translation_error'], 'g-', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Translation Error (m)')
        ax.set_title('Translation Error')
        ax.grid(True, alpha=0.3)

        # Scale error (if applicable)
        if self.config.use_scale:
            ax = axes[3]
            ax.plot(iters, debug_history['scale_error'], 'm-', linewidth=1.5, label='Error')
            ax_twin = ax.twinx()
            ax_twin.plot(iters, debug_history['scale'], 'c--', linewidth=1.5, label='Scale')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Scale Error', color='m')
            ax_twin.set_ylabel('Scale Value', color='c')
            ax.set_title('Scale Error & Value')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelcolor='m')
            ax_twin.tick_params(axis='y', labelcolor='c')

        plt.tight_layout()
        plt.savefig('registration_debug.png', dpi=150, bbox_inches='tight')
        print(f"  [Debug] Saved training curves to registration_debug.png")
        plt.close()

    def _register_multi_init(
        self,
        points: torch.Tensor,
        point_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Multi-initialization strategy."""
        device = points.device

        if self.config.verbose:
            print(f"Starting multi-init registration ({self.config.num_init} trials, scale={self.config.use_scale})")

        # Generate random initializations
        initializations = []
        scale_inits = []

        # Option 1: PCA-based initialization (if enabled)
        if self.config.use_pca_init:
            R_pca, scale_pca = self._compute_pca_init(points, self.grid_data.sphere_centers)
            T_pca = torch.eye(4, device=device)
            T_pca[:3, :3] = R_pca
            initializations.append(T_pca)
            scale_inits.append(np.log(scale_pca) if self.config.use_scale else 0.0)
            if self.config.verbose:
                print(f"  PCA init: scale={scale_pca:.3f}")
        else:
            # Always include identity
            initializations.append(torch.eye(4, device=device))
            scale_inits.append(0.0)  # log_scale = 0 -> scale = 1.0

        # Add random perturbations
        num_random = self.config.num_init - len(initializations)
        for _ in range(num_random):
            xi = torch.randn(6, device=device) * self.config.init_noise_scale
            T = se3_exp(xi)
            initializations.append(T)
            if self.config.use_scale:
                # Random log scale within pca_scale_range
                scale_min, scale_max = self.config.pca_scale_range
                log_s = np.log(scale_min) + torch.rand(1).item() * np.log(scale_max / scale_min)
                scale_inits.append(log_s)
            else:
                scale_inits.append(0.0)

        # Run optimization for each
        results = []
        for i, (T_init, s_init) in enumerate(zip(initializations, scale_inits)):
            if self.config.verbose:
                scale_str = f" (scale={torch.exp(torch.tensor(s_init)).item():.3f})" if self.config.use_scale else ""
                print(f"\nTrial {i+1}/{len(initializations)}:{scale_str}")

            result = self._optimize_single(points, T_init, s_init, point_confidence)
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

    def _compute_pca_init(
        self,
        points: torch.Tensor,
        scene_centers: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Compute PCA-based initialization for scale and rotation.

        Uses PCA on point clouds to align principal axes.

        Args:
            points: [N, 3] input point cloud
            scene_centers: [M, 3] Gaussian scene centers

        Returns:
            R_init: [3, 3] initial rotation matrix
            scale_init: initial scale factor
        """
        device = points.device

        # Center the point clouds
        p_center = points.mean(dim=0)
        s_center = scene_centers.mean(dim=0)

        p_centered = points - p_center
        s_centered = scene_centers - s_center

        # Compute covariance matrices
        p_cov = p_centered.T @ p_centered / points.shape[0]
        s_cov = s_centered.T @ scene_centers / scene_centers.shape[0]

        # Eigendecomposition
        p_eigvals, p_eigvecs = torch.linalg.eigh(p_cov)
        s_eigvals, s_eigvecs = torch.linalg.eigh(s_cov)

        # Sort by eigenvalue magnitude (descending)
        p_idx = torch.argsort(p_eigvals, descending=True)
        s_idx = torch.argsort(s_eigvals, descending=True)

        p_eigvecs = p_eigvecs[:, p_idx]
        s_eigvecs = s_eigvecs[:, s_idx]

        p_eigvals = p_eigvals[p_idx]
        s_eigvals = s_eigvals[s_idx]

        # Estimate scale from eigenvalue ratios
        # Use the largest eigenvalue ratio as scale estimate
        scale_estimate = torch.sqrt(s_eigvals[0] / (p_eigvals[0] + 1e-8))

        # Clamp scale to reasonable range
        scale_min, scale_max = self.config.pca_scale_range
        scale_estimate = torch.clamp(scale_estimate, scale_min, scale_max)

        # Compute rotation: align principal axes
        # R @ p_axes = s_axes  =>  R = s_axes @ p_axes^T
        R_init = s_eigvecs @ p_eigvecs.T

        # Ensure proper rotation (det = 1)
        if torch.det(R_init) < 0:
            R_init[:, 2] *= -1

        return R_init, scale_estimate.item()

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
