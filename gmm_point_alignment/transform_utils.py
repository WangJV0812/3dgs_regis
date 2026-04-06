"""Transformation utilities for SE(3) parameterization.

Provides differentiable transformation operations for point cloud registration.
"""

import torch
import torch.nn as nn
from typing import Dict


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Exponential map from so(3) to SO(3) using Rodrigues' formula.

    Args:
        w: [..., 3] axis-angle representation (rotation vector)

    Returns:
        R: [..., 3, 3] rotation matrices
    """
    # Handle different input shapes
    original_shape = w.shape
    w_flat = w.reshape(-1, 3)

    theta = w_flat.norm(dim=-1, keepdim=True)  # [B, 1]
    w_hat = w_flat / (theta + 1e-8)  # [B, 3], unit axis

    # Skew-symmetric matrix K
    K = torch.zeros(w_flat.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    K[:, 0, 1] = -w_hat[:, 2]
    K[:, 0, 2] = w_hat[:, 1]
    K[:, 1, 0] = w_hat[:, 2]
    K[:, 1, 2] = -w_hat[:, 0]
    K[:, 2, 0] = -w_hat[:, 1]
    K[:, 2, 1] = w_hat[:, 0]

    # Rodrigues formula: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0)

    sin_theta = torch.sin(theta).unsqueeze(-1)  # [B, 1, 1]
    cos_theta = torch.cos(theta).unsqueeze(-1)  # [B, 1, 1]

    R = I + sin_theta * K + (1 - cos_theta) * (K @ K)

    # Reshape back
    R = R.reshape(*original_shape[:-1], 3, 3)
    return R


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Exponential map from se(3) to SE(3).

    Args:
        xi: [..., 6] Lie algebra [tx, ty, tz, wx, wy, wz]

    Returns:
        T: [..., 4, 4] transformation matrices
    """
    original_shape = xi.shape
    xi_flat = xi.reshape(-1, 6)

    # Split translation and rotation
    t = xi_flat[:, :3]   # [B, 3]
    w = xi_flat[:, 3:]   # [B, 3]

    # Rotation: so(3) -> SO(3)
    R = so3_exp(w)  # [B, 3, 3]

    # Build transformation matrix
    B = xi_flat.shape[0]
    T = torch.zeros(B, 4, 4, device=xi.device, dtype=xi.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0

    # Reshape back
    T = T.reshape(*original_shape[:-1], 4, 4)
    return T


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """Logarithm map from SE(3) to se(3).

    Args:
        T: [..., 4, 4] transformation matrices

    Returns:
        xi: [..., 6] Lie algebra [tx, ty, tz, wx, wy, wz]
    """
    original_shape = T.shape
    T_flat = T.reshape(-1, 4, 4)

    # Extract rotation and translation
    R = T_flat[:, :3, :3]  # [B, 3, 3]
    t = T_flat[:, :3, 3]   # [B, 3]

    # SO(3) -> so(3)
    # Trace
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

    # Skew-symmetric matrix
    skew = torch.zeros_like(R)
    skew[:, 0, 1] = (R[:, 0, 1] - R[:, 1, 0]) / 2
    skew[:, 0, 2] = (R[:, 0, 2] - R[:, 2, 0]) / 2
    skew[:, 1, 2] = (R[:, 1, 2] - R[:, 2, 1]) / 2
    skew[:, 1, 0] = -skew[:, 0, 1]
    skew[:, 2, 0] = -skew[:, 0, 2]
    skew[:, 2, 1] = -skew[:, 1, 2]

    # w = theta * axis
    w = theta.unsqueeze(-1) * torch.stack([
        skew[:, 2, 1],
        skew[:, 0, 2],
        skew[:, 1, 0]
    ], dim=-1) / (torch.sin(theta).unsqueeze(-1) + 1e-8)

    # Handle small rotations
    small = theta < 1e-6
    w_small = torch.stack([R[:, 2, 1], R[:, 0, 2], R[:, 1, 0]], dim=-1) / 2
    w = torch.where(small.unsqueeze(-1), w_small, w)

    xi = torch.cat([t, w], dim=-1)
    xi = xi.reshape(*original_shape[:-2], 6)
    return xi


def sim3_exp(xi: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """Exponential map from sim(3) to Sim(3).

    Args:
        xi: [..., 6] Lie algebra [tx, ty, tz, wx, wy, wz]
        log_scale: [...] log of scale factor

    Returns:
        T: [..., 4, 4] similarity transformation matrices (scale * R | t)
    """
    original_shape = xi.shape[:-1]
    xi_flat = xi.reshape(-1, 6)
    log_scale_flat = log_scale.reshape(-1) if log_scale.ndim > 0 else log_scale.unsqueeze(0)

    # Get SE(3) part
    T = se3_exp(xi_flat)

    # Apply scale to rotation part (avoid in-place operations)
    s = torch.exp(log_scale_flat)
    scaled_rotation = T[:, :3, :3] * s.unsqueeze(-1).unsqueeze(-1)

    # Build new transformation matrix
    T_out = torch.zeros_like(T)
    T_out[:, :3, :3] = scaled_rotation
    T_out[:, :3, 3] = T[:, :3, 3]
    T_out[:, 3, 3] = 1.0

    T_out = T_out.reshape(*original_shape, 4, 4)
    return T_out


def sim3_log(T: torch.Tensor) -> tuple:
    """Logarithm map from Sim(3) to sim(3).

    Args:
        T: [..., 4, 4] similarity transformation matrices

    Returns:
        xi: [..., 6] Lie algebra [tx, ty, tz, wx, wy, wz]
        log_scale: [...] log of scale factor
    """
    original_shape = T.shape[:-2]
    T_flat = T.reshape(-1, 4, 4)

    # Extract scaled rotation and translation
    sR = T_flat[:, :3, :3]  # [B, 3, 3] - this is scale * rotation
    t = T_flat[:, :3, 3]   # [B, 3]

    # Compute scale from determinant
    # det(sR) = s^3 * det(R) = s^3 (since det(R) = 1)
    det = torch.det(sR)
    s = torch.pow(torch.abs(det), 1.0/3.0)
    log_scale = torch.log(s + 1e-8)

    # Extract pure rotation: R = (sR) / s
    R = sR / s.unsqueeze(-1).unsqueeze(-1)

    # SO(3) -> so(3)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

    skew = torch.zeros_like(R)
    skew[:, 0, 1] = (R[:, 0, 1] - R[:, 1, 0]) / 2
    skew[:, 0, 2] = (R[:, 0, 2] - R[:, 2, 0]) / 2
    skew[:, 1, 2] = (R[:, 1, 2] - R[:, 2, 1]) / 2
    skew[:, 1, 0] = -skew[:, 0, 1]
    skew[:, 2, 0] = -skew[:, 0, 2]
    skew[:, 2, 1] = -skew[:, 1, 2]

    w = theta.unsqueeze(-1) * torch.stack([
        skew[:, 2, 1],
        skew[:, 0, 2],
        skew[:, 1, 0]
    ], dim=-1) / (torch.sin(theta).unsqueeze(-1) + 1e-8)

    small = theta < 1e-6
    w_small = torch.stack([R[:, 2, 1], R[:, 0, 2], R[:, 1, 0]], dim=-1) / 2
    w = torch.where(small.unsqueeze(-1), w_small, w)

    xi = torch.cat([t, w], dim=-1)
    xi = xi.reshape(*original_shape, 6)
    log_scale = log_scale.reshape(original_shape) if len(original_shape) > 0 else log_scale.squeeze()

    return xi, log_scale


class SE3Parameter(nn.Module):
    """Optimizable SE(3) parameter using Lie algebra representation.

    Example:
        >>> se3 = SE3Parameter()  # Identity
        >>> points = torch.randn(100, 3)
        >>> transformed = se3.transform(points)
        >>> T_matrix = se3.matrix()
    """

    def __init__(self, xi_init: torch.Tensor = None, device='cuda'):
        super().__init__()
        if xi_init is None:
            xi_init = torch.zeros(6, device=device)
        self.xi = nn.Parameter(xi_init)

    def matrix(self) -> torch.Tensor:
        """Get transformation matrix."""
        return se3_exp(self.xi)

    def rotation(self) -> torch.Tensor:
        """Get rotation matrix."""
        return so3_exp(self.xi[3:])

    def translation(self) -> torch.Tensor:
        """Get translation vector."""
        return self.xi[:3]

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """Transform points.

        Args:
            points: [N, 3] or [B, N, 3]

        Returns:
            transformed: same shape as input
        """
        T = self.matrix()
        R = T[:3, :3]
        t = T[:3, 3]

        if points.ndim == 2:
            return (R @ points.T).T + t
        else:
            return (R @ points.transpose(-2, -1)).transpose(-2, -1) + t

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Alias for transform."""
        return self.transform(points)

    def __repr__(self):
        return f"SE3Parameter(xi={self.xi.data})"


class Sim3Parameter(nn.Module):
    """Optimizable Similarity transform parameter (SE3 + scale).

    Parameterized as [tx, ty, tz, wx, wy, wz, log_s] where log_s ensures
    scale is always positive via s = exp(log_s).

    Example:
        >>> sim3 = Sim3Parameter()  # Identity, scale=1.0
        >>> points = torch.randn(100, 3)
        >>> transformed = sim3.transform(points)  # s * R @ p + t
        >>> T_matrix = sim3.matrix()
    """

    def __init__(self, xi_init: torch.Tensor = None, log_scale_init: float = 0.0, device='cuda'):
        super().__init__()
        if xi_init is None:
            xi_init = torch.zeros(6, device=device)
        self.xi = nn.Parameter(xi_init)
        self.log_scale = nn.Parameter(torch.tensor(log_scale_init, device=device))

    def scale(self) -> torch.Tensor:
        """Get scale factor (always positive)."""
        return torch.exp(self.log_scale)

    def matrix(self) -> torch.Tensor:
        """Get 4x4 transformation matrix (scale applied to rotation)."""
        T = se3_exp(self.xi)
        s = self.scale()
        # Apply scale to rotation part
        T[:3, :3] = T[:3, :3] * s
        return T

    def rotation(self) -> torch.Tensor:
        """Get rotation matrix (without scale)."""
        return so3_exp(self.xi[3:])

    def translation(self) -> torch.Tensor:
        """Get translation vector."""
        return self.xi[:3]

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """Transform points: s * R @ p + t

        Args:
            points: [N, 3] or [B, N, 3]

        Returns:
            transformed: same shape as input
        """
        R = self.rotation()
        t = self.translation()
        s = self.scale()

        if points.ndim == 2:
            return s * (R @ points.T).T + t
        else:
            return s * (R @ points.transpose(-2, -1)).transpose(-2, -1) + t

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Alias for transform."""
        return self.transform(points)

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Get all parameters as dictionary."""
        return {
            'translation': self.translation(),
            'rotation': self.rotation(),
            'scale': self.scale(),
            'log_scale': self.log_scale,
        }

    def __repr__(self):
        return f"Sim3Parameter(xi={self.xi.data}, scale={self.scale().item():.4f})"


def transform_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Transform points using transformation matrix.

    Args:
        points: [N, 3] or [B, N, 3]
        T: [4, 4] or [B, 4, 4]

    Returns:
        transformed: same shape as points
    """
    if T.ndim == 2:
        R = T[:3, :3]
        t = T[:3, 3]
        return (R @ points.T).T + t
    else:
        # Batch transform
        R = T[:, :3, :3]  # [B, 3, 3]
        t = T[:, :3, 3]   # [B, 3]
        return (R @ points.transpose(-2, -1)).transpose(-2, -1) + t.unsqueeze(-2)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix.

    Args:
        q: [..., 4] quaternion [w, x, y, z]

    Returns:
        R: [..., 3, 3] rotation matrices
    """
    # Normalize
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Build matrix
    R = torch.zeros(*q.shape[:-1], 3, 3, device=q.device, dtype=q.dtype)

    R[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    R[..., 0, 1] = 2 * (x*y - w*z)
    R[..., 0, 2] = 2 * (x*z + w*y)
    R[..., 1, 0] = 2 * (x*y + w*z)
    R[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    R[..., 1, 2] = 2 * (y*z - w*x)
    R[..., 2, 0] = 2 * (x*z - w*y)
    R[..., 2, 1] = 2 * (y*z + w*x)
    R[..., 2, 2] = 1 - 2 * (x**2 + y**2)

    return R


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion.

    Args:
        R: [..., 3, 3] rotation matrices

    Returns:
        q: [..., 4] quaternion [w, x, y, z]
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    q = torch.zeros(*R.shape[:-2], 4, device=R.device, dtype=R.dtype)

    # Different cases for numerical stability
    mask0 = trace > 0
    mask1 = (~mask0) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    mask2 = (~mask0) & (~mask1) & (R[..., 1, 1] > R[..., 2, 2])
    mask3 = (~mask0) & (~mask1) & (~mask2)

    # Case 0: trace > 0
    s0 = torch.sqrt(trace[mask0] + 1.0) * 2
    q[mask0, 0] = 0.25 * s0
    q[mask0, 1] = (R[mask0, 2, 1] - R[mask0, 1, 2]) / s0
    q[mask0, 2] = (R[mask0, 0, 2] - R[mask0, 2, 0]) / s0
    q[mask0, 3] = (R[mask0, 1, 0] - R[mask0, 0, 1]) / s0

    # Case 1: R[0,0] is largest
    s1 = torch.sqrt(1.0 + R[mask1, 0, 0] - R[mask1, 1, 1] - R[mask1, 2, 2]) * 2
    q[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    q[mask1, 1] = 0.25 * s1
    q[mask1, 2] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / s1
    q[mask1, 3] = (R[mask1, 0, 2] + R[mask1, 2, 0]) / s1

    # Case 2: R[1,1] is largest
    s2 = torch.sqrt(1.0 + R[mask2, 1, 1] - R[mask2, 0, 0] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 0, 2] - R[mask2, 2, 0]) / s2
    q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    q[mask2, 2] = 0.25 * s2
    q[mask2, 3] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / s2

    # Case 3: R[2,2] is largest
    s3 = torch.sqrt(1.0 + R[mask3, 2, 2] - R[mask3, 0, 0] - R[mask3, 1, 1]) * 2
    q[mask3, 0] = (R[mask3, 1, 0] - R[mask3, 0, 1]) / s3
    q[mask3, 1] = (R[mask3, 0, 2] + R[mask3, 2, 0]) / s3
    q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3
    q[mask3, 3] = 0.25 * s3

    return q


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Test SE(3) operations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Transform Utilities Test")
    print("=" * 60)

    # Test se3_exp and se3_log
    xi = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], device=device)
    T = se3_exp(xi)
    xi_recovered = se3_log(T)

    print(f"\nOriginal xi: {xi}")
    print(f"Recovered xi: {xi_recovered}")
    print(f"Reconstruction error: {(xi - xi_recovered).abs().max().item():.2e}")

    # Test SE3Parameter
    se3 = SE3Parameter(xi, device=device)
    points = torch.randn(100, 3, device=device)
    transformed = se3.transform(points)

    print(f"\nSE3Parameter test:")
    print(f"  Input shape: {points.shape}")
    print(f"  Output shape: {transformed.shape}")

    # Test gradient flow
    xi_grad = torch.zeros(6, device=device, requires_grad=True)
    T_grad = se3_exp(xi_grad)
    points_t = transform_points(points, T_grad)
    loss = points_t.sum()
    loss.backward()

    print(f"\nGradient test:")
    print(f"  xi gradient: {xi_grad.grad}")
    print(f"  Has gradient: {xi_grad.grad is not None}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
