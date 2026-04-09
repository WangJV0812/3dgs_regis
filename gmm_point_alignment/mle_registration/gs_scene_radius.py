import taichi as ti
import torch
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
def approximate_chi_2_critical_value(
    degrees_of_freedom: ti.i32,
    confidence_level: ti.f32,
) -> ti.f32:
    """Approximate chi-squared critical value using Wilson-Hilferty transformation.

    Uses approximation: c ≈ df * (1 - 2/(9*df) + z * sqrt(2/(9*df)))^3
    where z is the standard normal quantile for the confidence level.

    For common confidence levels:
    - 0.95 -> z ≈ 1.645
    - 0.99 -> z ≈ 2.326
    """
    # Approximate z-score for given confidence level
    # Using simplified approximation for z = Phi^-1(p)
    p = confidence_level
    # Rational approximation for inverse normal CDF
    # z ≈ 4.91 * (p - 0.5) works reasonably well for 0.5 < p < 0.99
    z = 4.91 * (p - 0.5)

    df = ti.cast(degrees_of_freedom, ti.f32)

    return df * (1.0 - 2.0/(9.0*df) + z * ti.sqrt(2.0/(9.0*df)))**3


@ti.kernel
def gaussian_scene_radius(
    scales: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [num_gaussians, 3]
    radius: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [num_gaussians, 3]
    confidence_level: float,
):
    """calcualte gaussian ellipsoid radii for three axes at confidence level, sorted as max, mid, min. The radius is calculated by multiplying the square root of the chi-squared critical value with the scales, which represent the covariance of the Gaussian distribution. The chi-squared critical value is determined by the confidence level and the degrees of freedom (which is 3 for 3D space).

    Args:
        scales (ti.types.ndarray, optional): gaussian covariance scales, calculated by \Sigma = RSS^TR^T. Defaults to ti.f32, ndim=2).
        radius (ti.types.ndarray, optional): gaussian ellipsoid radii for three axes at confidence level, sorted as max, mid, min. Defaults to ti.f32, ndim=2).
        confidence_level (float, optional): confidence level for the chi-squared distribution. Defaults to 0.95.
    """
    sphere_counts = scales.shape[0]
    critical_value = approximate_chi_2_critical_value(ti.cast(3, ti.i32), ti.cast(confidence_level, ti.f32))
    
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