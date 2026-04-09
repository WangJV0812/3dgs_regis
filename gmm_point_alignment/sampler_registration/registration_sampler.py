"""Sampler-based point cloud registration methods.

Provides traditional ICP-based registration that samples point clouds from
Gaussian scenes and aligns them using various ICP variants.
"""

from dataclasses import dataclass
from typing import Literal, Optional
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


class SamplerRegistrationMethod(str, Enum):
    """Registration methods available for sampler-based approach."""
    SVD_ICP = "svd_icp"
    CHAMFER_OPT = "chamfer_opt"
    OPEN3D_ICP_POINT_TO_POINT = "open3d_icp_point_to_point"
    OPEN3D_ICP_POINT_TO_PLANE = "open3d_icp_point_to_plane"


@dataclass
class RegistrationSamplerConfig:
    """Configuration for sampler-based registration.

    Args:
        method: Registration method to use
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance
        lr: Learning rate for gradient-based methods
        init_noise_scale: Scale of random noise for initialization
        multi_init: Use multiple random initializations
        num_init: Number of random initializations
    """
    method: SamplerRegistrationMethod = SamplerRegistrationMethod.SVD_ICP
    max_iterations: int = 100
    tolerance: float = 1e-6
    lr: float = 1e-3
    init_noise_scale: float = 0.5
    multi_init: bool = True
    num_init: int = 5


@dataclass
class RegistrationSamplerResult:
    """Result from sampler-based registration."""
    R: torch.Tensor          # (3, 3) rotation
    t: torch.Tensor          # (3,) translation
    scale: float             # scale factor (usually 1.0 for rigid methods)
    rmse: float
    converged: bool
    num_iters: int


def _so3_to_matrix(so3_vec: torch.Tensor) -> torch.Tensor:
    """Convert an so(3) axis-angle vector to a rotation matrix."""
    theta = so3_vec.norm()
    if theta < 1e-6:
        return torch.eye(3, device=so3_vec.device, dtype=so3_vec.dtype)

    K = torch.zeros(3, 3, device=so3_vec.device, dtype=so3_vec.dtype)
    K[0, 1] = -so3_vec[2]
    K[0, 2] = so3_vec[1]
    K[1, 2] = -so3_vec[0]
    K = K - K.T
    K = K / theta

    I = torch.eye(3, device=so3_vec.device, dtype=so3_vec.dtype)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


def _find_nearest_neighbors(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Find nearest neighbor indices from src to tgt."""
    dists = torch.cdist(src, tgt)  # (N, M)
    nn_idx = dists.argmin(dim=-1)  # (N,)
    return nn_idx


def _compute_rmse(src: torch.Tensor, tgt_aligned: torch.Tensor) -> float:
    """Compute root-mean-square error between two point clouds."""
    return torch.sqrt(((src - tgt_aligned) ** 2).sum() / src.shape[0]).item()


def _chamfer_distance(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Compute symmetric Chamfer distance between two point clouds."""
    dists = torch.cdist(src, tgt)  # (N, M)
    min_src_to_tgt = dists.min(dim=1)[0].mean()
    min_tgt_to_src = dists.min(dim=0)[0].mean()
    return min_src_to_tgt + min_tgt_to_src


def register_svd_icp(
    src: torch.Tensor,
    tgt: torch.Tensor,
    config: RegistrationSamplerConfig,
) -> RegistrationSamplerResult:
    """Register source to target using iterative closest point with SVD."""
    device = src.device
    dtype = src.dtype

    R = torch.eye(3, device=device, dtype=dtype)
    t = torch.zeros(3, device=device, dtype=dtype)

    prev_rmse = float("inf")
    converged = False

    for i in range(config.max_iterations):
        # Transform source
        src_transformed = src @ R.T + t

        # Find nearest neighbors
        nn_idx = _find_nearest_neighbors(src_transformed, tgt)
        tgt_matched = tgt[nn_idx]

        # Compute centroids
        src_centroid = src_transformed.mean(dim=0)
        tgt_centroid = tgt_matched.mean(dim=0)

        # Centered coordinates
        src_centered = src_transformed - src_centroid
        tgt_centered = tgt_matched - tgt_centroid

        # Cross-covariance matrix
        H = tgt_centered.T @ src_centered  # (3, 3)

        # SVD
        U, S, Vt = torch.linalg.svd(H)
        d = torch.det(U @ Vt)
        diag = torch.eye(3, device=device, dtype=dtype)
        diag[2, 2] = d
        R_step = U @ diag @ Vt

        if torch.det(R_step) < 0:
            diag[2, 2] = -1.0
            R_step = U @ diag @ Vt

        t_step = tgt_centroid - R_step @ src_centroid

        # Update cumulative transformation
        R = R_step @ R
        t = R_step @ t + t_step

        # Check convergence
        rmse = _compute_rmse(src_transformed, tgt_matched)
        if abs(prev_rmse - rmse) < config.tolerance:
            converged = True
            break
        prev_rmse = rmse

    final_src = src @ R.T + t
    final_rmse = _compute_rmse(final_src, tgt[_find_nearest_neighbors(final_src, tgt)])

    return RegistrationSamplerResult(
        R=R, t=t, scale=1.0, rmse=final_rmse,
        converged=converged, num_iters=i+1
    )


def register_chamfer_opt(
    src: torch.Tensor,
    tgt: torch.Tensor,
    config: RegistrationSamplerConfig,
) -> RegistrationSamplerResult:
    """Register source to target by optimizing Chamfer distance."""
    device = src.device
    dtype = src.dtype

    so3 = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
    t = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)

    optimizer = torch.optim.Adam([so3, t], lr=config.lr)

    prev_loss = float("inf")
    converged = False

    for i in range(config.max_iterations):
        optimizer.zero_grad()
        R = _so3_to_matrix(so3)
        src_transformed = src @ R.T + t
        loss = _chamfer_distance(src_transformed, tgt)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if abs(prev_loss - loss_val) < config.tolerance:
            converged = True
            break
        prev_loss = loss_val

    R_final = _so3_to_matrix(so3.detach())
    t_final = t.detach()
    with torch.no_grad():
        src_transformed = src @ R_final.T + t_final
        rmse = _compute_rmse(src_transformed, tgt[_find_nearest_neighbors(src_transformed, tgt)])

    return RegistrationSamplerResult(
        R=R_final, t=t_final, scale=1.0, rmse=rmse,
        converged=converged, num_iters=i+1
    )


def _torch_to_o3d(points: torch.Tensor) -> "o3d.geometry.PointCloud":
    """Convert a torch tensor to an Open3D point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    return pcd


def _o3d_to_torch_Rt(trans: "o3d.pipelines.registration.RegistrationResult", device, dtype) -> tuple:
    """Extract R and t from an Open3D registration result."""
    T = trans.transformation  # 4x4 numpy array
    R = torch.from_numpy(T[:3, :3].copy()).to(device=device, dtype=dtype)
    t = torch.from_numpy(T[:3, 3].copy()).to(device=device, dtype=dtype)
    return R, t


def _open3d_icp(
    src: torch.Tensor,
    tgt: torch.Tensor,
    config: RegistrationSamplerConfig,
    method: str,
) -> RegistrationSamplerResult:
    """Internal helper for Open3D-based ICP registration."""
    if not _HAS_OPEN3D:
        raise ImportError(
            "Open3D is required for this registration method. "
            "Install it with: pip install open3d"
        )

    if method == "point_to_plane":
        src_o3d = _torch_to_o3d(src)
        tgt_o3d = _torch_to_o3d(tgt)
        src_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        tgt_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        o3d_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        src_o3d = _torch_to_o3d(src)
        tgt_o3d = _torch_to_o3d(tgt)
        o3d_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    threshold = 0.1

    trans = o3d.pipelines.registration.registration_icp(
        src_o3d,
        tgt_o3d,
        max_correspondence_distance=threshold,
        init=np.eye(4),
        estimation_method=o3d_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=config.max_iterations,
            relative_fitness=1e-6,
            relative_rmse=1e-6,
        ),
    )

    R, t = _o3d_to_torch_Rt(trans, src.device, src.dtype)
    aligned = src @ R.T + t
    rmse = _compute_rmse(aligned, tgt[_find_nearest_neighbors(aligned, tgt)])

    return RegistrationSamplerResult(
        R=R, t=t, scale=1.0, rmse=rmse,
        converged=len(trans.correspondence_set) > 0,
        num_iters=config.max_iterations
    )


def _register_single_init(
    src: torch.Tensor,
    tgt: torch.Tensor,
    config: RegistrationSamplerConfig,
) -> RegistrationSamplerResult:
    """Single initialization registration."""
    if config.method == SamplerRegistrationMethod.SVD_ICP:
        return register_svd_icp(src, tgt, config)
    elif config.method == SamplerRegistrationMethod.CHAMFER_OPT:
        return register_chamfer_opt(src, tgt, config)
    elif config.method == SamplerRegistrationMethod.OPEN3D_ICP_POINT_TO_POINT:
        return _open3d_icp(src, tgt, config, method="point_to_point")
    elif config.method == SamplerRegistrationMethod.OPEN3D_ICP_POINT_TO_PLANE:
        return _open3d_icp(src, tgt, config, method="point_to_plane")
    else:
        raise ValueError(f"Unknown registration method: {config.method}")


def register_with_sampler(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    config: Optional[RegistrationSamplerConfig] = None,
) -> RegistrationSamplerResult:
    """Register source point cloud to target point cloud using sampler-based methods.

    Args:
        src_points: Source points (N, 3)
        tgt_points: Target points (M, 3)
        config: Registration configuration

    Returns:
        RegistrationSamplerResult with transformation and metrics
    """
    if config is None:
        config = RegistrationSamplerConfig()

    device = src_points.device

    if config.multi_init:
        # Generate random initializations
        best_result = None
        best_rmse = float('inf')

        for i in range(config.num_init):
            # Add random perturbation
            if i > 0:
                noise = torch.randn(3, device=device) * config.init_noise_scale
                src_perturbed = src_points + noise
            else:
                src_perturbed = src_points

            result = _register_single_init(src_perturbed, tgt_points, config)

            if result.rmse < best_rmse:
                best_rmse = result.rmse
                best_result = result

        return best_result
    else:
        return _register_single_init(src_points, tgt_points, config)
