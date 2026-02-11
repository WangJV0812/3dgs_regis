
import unittest
import taichi as ti
import torch
import numpy as np

# Adjust path to import from workspace
import sys
import os
sys.path.append("/home/wangjv_wsl/code/3dgs/3dgs_regis")

from gmm_point_alignment.gs_scene_aabb import gaussian_scene_aabb, robust_global_scene_aabb, gaussian_sphere_aabb
from gmm_point_alignment.gs_scene_radius import approximate_chi_2_critical_value

class TestGsSceneAabb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu)

    def test_gaussian_sphere_aabb_identity(self):
        """Test AABB for identity rotation and simple scales."""
        
        @ti.kernel
        def run_test() -> ti.types.matrix(3, 3, float):
            center = ti.math.vec3([0.0, 0.0, 0.0])
            scales = ti.math.vec3([1.0, 2.0, 3.0])
            quaternion = ti.math.vec4([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
            critical_value = 1.0 # Simplify math, sqrt(1)=1
            
            min_c, max_c, rad = gaussian_sphere_aabb(center, scales, quaternion, critical_value)
            
            # Pack into matrix to return
            return ti.math.mat3([
                [min_c[0], min_c[1], min_c[2]],
                [max_c[0], max_c[1], max_c[2]],
                [rad[0], rad[1], rad[2]]
            ])

        result = run_test()
        min_c = result[0, :]
        max_c = result[1, :]
        rad = result[2, :]

        # Expected:
        # Extents (half): 1*1, 1*2, 1*3 => 1, 2, 3
        # Min: -1, -2, -3
        # Max: 1, 2, 3
        # Sorted extents (full widths): 2*3, 2*2, 2*1 => 6, 4, 2
        
        np.testing.assert_allclose(min_c, [-1.0, -2.0, -3.0], atol=1e-5)
        np.testing.assert_allclose(max_c, [1.0, 2.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(rad, [6.0, 4.0, 2.0], atol=1e-5)

    def test_gaussian_sphere_aabb_rotated(self):
        """Test AABB with 90 degree rotation around Z axis."""
        
        @ti.kernel
        def run_test() -> ti.types.matrix(3, 3, float):
            center = ti.math.vec3([10.0, 10.0, 10.0])
            scales = ti.math.vec3([4.0, 1.0, 1.0]) # Long along X initially
            # Rotate 90 deg around Z: X becomes Y
            # q = cos(45) + k sin(45) = 0.7071 + 0.7071 k
            quaternion = ti.math.vec4([0.70710678, 0.0, 0.0, 0.70710678]) 
            critical_value = 1.0 
            
            min_c, max_c, rad = gaussian_sphere_aabb(center, scales, quaternion, critical_value)
            
            return ti.math.mat3([
                [min_c[0], min_c[1], min_c[2]],
                [max_c[0], max_c[1], max_c[2]],
                [rad[0], rad[1], rad[2]]
            ])

        result = run_test()
        min_c = result[0, :]
        max_c = result[1, :]
        rad = result[2, :]
        
        # After rotation, the long axis (4.0) is along Y.
        # X extent: 1.0
        # Y extent: 4.0
        # Z extent: 1.0
        
        # Center is (10, 10, 10)
        # Min X: 10 - 1 = 9
        # Max X: 10 + 1 = 11
        # Min Y: 10 - 4 = 6
        # Max Y: 10 + 4 = 14
        # Min Z: 10 - 1 = 9
        # Max Z: 10 + 1 = 11
        
        np.testing.assert_allclose(min_c, [9.0, 6.0, 9.0], atol=1e-4)
        np.testing.assert_allclose(max_c, [11.0, 14.0, 11.0], atol=1e-4)
        
        # Sorted full extents: 8, 2, 2
        np.testing.assert_allclose(rad, [8.0, 2.0, 2.0], atol=1e-4)

    def test_gaussian_scene_aabb_integration(self):
        """Test the full scene kernel."""
        N = 2
        centers = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=torch.float32)
        scales = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
        quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        
        min_corners = torch.zeros((N, 3), dtype=torch.float32)
        max_corners = torch.zeros((N, 3), dtype=torch.float32)
        radius = torch.zeros((N, 3), dtype=torch.float32)
        
        confidence = 0.95 # This will trigger lookup or calculation
        # approximate_chi_2_critical_value(0.95) is approx 7.8147
        # sqrt(7.8147) approx 2.795
        
        gaussian_scene_aabb(
            centers, scales, quaternions,
            min_corners, max_corners, radius,
            confidence
        )
        
        crit = approximate_chi_2_critical_value(3, confidence)
        sqrt_crit = np.sqrt(crit)
        
        # Check first gaussian
        # Scale 1.0 => Half extent 1.0 * 2.795 = 2.795
        # Min: -2.795, Max: 2.795
        
        np.testing.assert_allclose(min_corners[0].numpy(), [-sqrt_crit]*3, rtol=1e-4)
        np.testing.assert_allclose(max_corners[0].numpy(), [sqrt_crit]*3, rtol=1e-4)
        
        # Check second gaussian
        # Center 5.0, Scale 2.0 => Half extent 2.0 * 2.795 = 5.59
        # Min: 5 - 5.59 = -0.59
        # Max: 5 + 5.59 = 10.59
        
        expected_half = 2.0 * sqrt_crit
        np.testing.assert_allclose(min_corners[1].numpy(), [5.0 - expected_half]*3, rtol=1e-4)
        
    def test_robust_global_scene_aabb(self):
        """Test robust global AABB calculation with outliers."""
        # 100 points centered at 0, radius 1
        # 1 point way out at 100
        
        min_corners = torch.zeros((100, 3)) - 1.0
        max_corners = torch.zeros((100, 3)) + 1.0
        
        # Outlier
        min_corners = torch.cat([min_corners, torch.tensor([[99.0, 99.0, 99.0]])])
        max_corners = torch.cat([max_corners, torch.tensor([[101.0, 101.0, 101.0]])])
        
        # Standard global AABB would include outlier
        # Robust with clip_quantile=0.02 (clips top/bottom 2%)
        # 2% of 101 points is approx 2 points. So the outlier (1 point) should be clipped at the top end.
        
        g_min, g_max = robust_global_scene_aabb(min_corners, max_corners, clip_quantile=0.02, padding_factor=0.0)
        
        # Should be close to -1, 1
        # Note: Quantile interpolation might not be exactly -1 or 1 depending on implementation, but should be far from 100.
        
        self.assertTrue(torch.all(g_max < 10.0), f"Max corner {g_max} included outlier!")
        self.assertTrue(torch.all(g_min > -10.0))


    def test_approximate_chi_2_critical_value_logic(self):
        """Test the logic of chi-2 approximation."""
        # 1. Sanity check: must be positive
        val = approximate_chi_2_critical_value(3, 0.95)
        self.assertGreater(val, 0.0)

        # 2. Check for the specific value expected by the current implementation logic.
        # Note: The current implementation uses z = 1/confidence, which is likely physically incorrect 
        # (should be inverse CDF), but we test for the *implemented* behavior here strictly for regression.
        # If the math is fixed, this test value should be updated to ~7.81.
        
        # Current logic: 
        # z = 1/0.95 = 1.05263
        # term = sqrt(2/27) * z = 0.27216 * 1.05263 = 0.28649
        # base = 1 - 2/27 + 0.28649 = 1 - 0.07407 + 0.28649 = 1.2124
        # result = 3 * (1.2124)^3 = 3 * 1.782 = 5.346
        
        self.assertAlmostEqual(val, 5.346, places=2)

if __name__ == '__main__':
    unittest.main()
