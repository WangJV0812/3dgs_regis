import taichi as ti
import torch
import math
from misc.hier_IO import GaussianScenes


@ti.func
def gaussian_sphere_radius(
    scales: ti.math.vec3,
    critical_value: ti.f32 = 7.8147,  # chi-squared distribution with 3 degrees of freedom at 95% confidence
) -> ti.math.vec3:
    a = scales.x
    b = scales.y
    c = scales.z
    max_v = ti.max(a, ti.max(b, c))
    min_v = ti.min(a, ti.min(b, c))
    mid_v = a + b + c - max_v - min_v
    sqrt_critical_value = ti.sqrt(critical_value)
    

    return ti.math.vec3([
        sqrt_critical_value * max_v,
        sqrt_critical_value * mid_v,
        sqrt_critical_value * min_v,
    ])
    
@ti.func
def approximate_chi_2_critical_value_ti(
    degrees_of_freedom: int, 
    confidence_level: float = 0.95
) -> ti.f32:
    # using wilson-hilferty transformation to approximate chi-squared critical value
    # c \approx df * (1 - 2/(9*df) + z * sqrt(2/(9*df)))^3
    
    z = 1 / (ti.abs(confidence_level) + 1e-10)  # avoid division by zero, though confidence_level should never be 0 or 1
    df = degrees_of_freedom
    
    return df * (1 - 2/(9*df) + z * ti.sqrt(2/(9*df)))**3


def approximate_chi_2_critical_value(
    degrees_of_freedom: int, 
    confidence_level: float = 0.95
) -> ti.f32:
    # using wilson-hilferty transformation to approximate chi-squared critical value
    # c \approx df * (1 - 2/(9*df) + z * sqrt(2/(9*df)))^3
    
    z = 1 / (confidence_level + 1e-10)  # avoid division by zero, though confidence_level should never be 0 or 1
    df = degrees_of_freedom
    
    return df * (1 - 2/(9*df) + z * math.sqrt(2/(9*df)))**3



@ti.kernel
def gaussian_scene_radius(
    scales: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [num_gaussians, 3]
    radius: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [num_gaussians, 3]
    confidence_level: float
):
    """calcualte gaussian ellipsoid radii for three axes at confidence level, sorted as max, mid, min. The radius is calculated by multiplying the square root of the chi-squared critical value with the scales, which represent the covariance of the Gaussian distribution. The chi-squared critical value is determined by the confidence level and the degrees of freedom (which is 3 for 3D space).

    Args:
        scales (ti.types.ndarray, optional): gaussian covariance scales, calculated by \Sigma = RSS^TR^T. Defaults to ti.f32, ndim=2).
        radius (ti.types.ndarray, optional): gaussian ellipsoid radii for three axes at confidence level, sorted as max, mid, min. Defaults to ti.f32, ndim=2).
        confidence_level (float, optional): confidence level for the chi-squared distribution. Defaults to 0.95.
    """
    sphere_counts = scales.shape[0]
    critical_value = approximate_chi_2_critical_value_ti(3, confidence_level)
    
    for idx in range(sphere_counts):
        sphere_scale = ti.math.vec3([
            scales[idx, 0], scales[idx, 1], scales[idx, 2]    
        ])
        
        radii = gaussian_sphere_radius(
            scales=sphere_scale,
            critical_value=critical_value,
        )
        
        for i in ti.static(range(3)):
            radius[idx, i] = radii[i]
        
        
def compute_gaussian_scene_average_max_radii(
    scene: GaussianScenes,
    confidence_level: float = 0.95
) -> float:
    """compute the average maximum radius of the Gaussian scene, which can be used as a reference for point cloud alignment.

    Args:
        scene (GaussianScenes): the input Gaussian scene
        confidence_level (float, optional): confidence level for the chi-squared distribution. Defaults to 0.95.
    """
    
    radius = torch.zeros_like(scene.scales)
    
    gaussian_scene_radius(
        scales=scene.scales,
        radius=radius,
        confidence_level=confidence_level,
    )
    
    average_max_radius = torch.mean(radius[:, 0])

    return average_max_radius.item()


def compute_gaussian_scene_median_max_radii(
    scene: GaussianScenes,
    confidence_level: float = 0.95
) -> float:
    """compute the median maximum radius of the Gaussian scene, which can be used as a reference for point cloud alignment.

    Args:
        scene (GaussianScenes): the input Gaussian scene
        confidence_level (float, optional): confidence level for the chi-squared distribution. Defaults to 0.95.
    """
    
    radius = torch.zeros_like(scene.scales)
    
    gaussian_scene_radius(
        scales=scene.scales,
        radius=radius,
        confidence_level=confidence_level,
    )
    
    median_max_radius = torch.median(radius[:, 0])

    return median_max_radius.item()