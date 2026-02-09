import taichi as ti
import torch
from misc.geometry import quaternion_to_rotation_ti
from gmm_point_alignment.gs_scene_radius import approximate_chi_2_critical_value


@ti.func
def gaussian_sphere_aabb(
    center: ti.math.vec3,
    scales: ti.math.vec3,
    quaternion: ti.math.vec4,
    critical_value: ti.f32 = 7.8147
) -> (ti.math.vec3, ti.math.vec3):

    min_corner = center
    max_corner = center
    
    rotation = quaternion_to_rotation_ti(quaternion)
    
    sqrt_c = ti.sqrt(critical_value)
    
    for axis in ti.static(range(3)):
        L = 0.0
        for i in ti.static(range(3)):
            R_ai = rotation[axis, i]
            L += R_ai * R_ai * scales[i] * scales[i]
        
        half_extent = sqrt_c * ti.sqrt(L)

        min_corner[axis] -= half_extent
        max_corner[axis] += half_extent
    
    return min_corner, max_corner


@ti.kernel
def gaussian_scene_aabb(
    centers: ti.types.ndarray(dtype=ti.f32, ndim=2),        # [num_gaussians, 3]
    scales: ti.types.ndarray(dtype=ti.f32, ndim=2),         # [num_gaussians, 3]
    quaternions: ti.types.ndarray(dtype=ti.f32, ndim=2),    # [num_gaussians, 4]
    # output
    min_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),    # [num_gaussians, 3]
    max_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),    # [num_gaussians, 3]
    confidence_level: float = 0.95
):
    critical_value = approximate_chi_2_critical_value(confidence_level)
    sphere_counts = centers.shape[0]
    
    for idx in range(sphere_counts):
        center = ti.math.vec3([
            centers[idx][0], centers[idx][1], centers[idx][2]    
        ])
        scale = ti.math.vec3([
            scales[idx][0], scales[idx][1], scales[idx][2]    
        ])
        quaternion = ti.math.vec4([
            quaternions[idx][0], quaternions[idx][1], quaternions[idx][2], quaternions[idx][3]    
        ])
        
        min_corner, max_corner = gaussian_sphere_aabb(
            center=center,
            scales=scale,
            quaternion=quaternion,
            critical_value=critical_value,
        )
        
        for i in ti.static(range(3)):
            min_corners[idx][i] = min_corner[i]
            max_corners[idx][i] = max_corner[i]
            

def global_scene_aabb(
    min_corners: torch.Tensor,  # [num_gaussians, 3]
    max_corners: torch.Tensor,  # [num_gaussians, 3]
) -> (torch.Tensor, torch.Tensor):
    """compute the global AABB of the Gaussian scene

    Args:
        min_corners (torch.Tensor): minimum corners of the Gaussian spheres, shape (num_gaussians, 3)
        max_corners (torch.Tensor): maximum corners of the Gaussian spheres, shape (num_gaussians, 3)

    Returns:
        global_min_corner (torch.Tensor): minimum corner of the global AABB, shape (3,)
        global_max_corner (torch.Tensor): maximum corner of the global AABB, shape (3,)
    """
    global_min_corner = torch.min(min_corners, dim=0)[0]
    global_max_corner = torch.max(max_corners, dim=0)[0]

    return global_min_corner, global_max_corner


def robust_global_scene_aabb(
    min_corners: torch.Tensor,  # [num_gaussians, 3]
    max_corners: torch.Tensor,  # [num_gaussians, 3]
    clip_quantile: float = 0.01,
    padding_factor: float = 0.1,
) -> (torch.Tensor, torch.Tensor):
    """ignore outlier sphere in scene to calculate scene's aabb

    Args:
        min_corners (torch.Tensor): minimum corners of the Gaussian spheres, shape (num_gaussians, 3)
        max_corners (torch.Tensor): maximum corners of the Gaussian spheres, shape (num_gaussians, 3)
        clip_quantile (float, optional): quantile for clipping outliers. Defaults to 0.01.
        padding_factor (float, optional): factor for padding the AABB. Defaults to 0.1.

    Returns:
        global_min_corner (torch.Tensor): minimum corner of the robust global AABB, shape (3,)
        global_max_corner (torch.Tensor): maximum corner of the robust global AABB, shape (3,)
    """
    
    all_corners = torch.cat([min_corners, max_corners], dim=0)
    
    q_min = torch.quantile(all_corners, clip_quantile, dim=0)
    q_max = torch.quantile(all_corners, (1 - clip_quantile), dim=0)
    
    center = (q_min + q_max) / 2.0
    size = (q_max - q_min) * (1.0 + padding_factor)
    global_min_corner = center - size / 2.0
    global_max_corner = center + size / 2.0
    
    return global_min_corner, global_max_corner