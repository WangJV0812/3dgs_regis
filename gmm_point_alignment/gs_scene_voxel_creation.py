import taichi as ti
import torch
from dataclasses import dataclass
from typing import Optional
from time import time

from misc.hier_IO import GaussianScenes
from gmm_point_alignment.gs_scene_aabb import (
    robust_global_scene_aabb,
    gaussian_scene_aabb,
)
from misc.geometry import (
    compute_gaussian_covariance,
    gaussian_density_cov_inv_ti,
    compute_gaussian_normalized_factor_cov_inv_ti
)
from gmm_point_alignment.morton_code import Morton3D


@dataclass
class GaussianSceneGridCreatorConfig:
    confidence_level: float = 0.95
    voxel_size_factor: float = 3.0
    max_active_cells: int = 2 ** 20
    grid_max_sphere_count: int = 1024  # maximum number of spheres might in each voxel
    global_aabb_padding_factor: float = 0.1
    global_aabb_clip_quantile: float = 0.01
    oversized_sphere_voxels_threshold: int = 64  # if a sphere overlaps with more than this number of voxels, it will be considered as oversized
    max_grid_size: int = 2**10

@ti.data_oriented
class GaussianSceneGridCreator:
    
    def __init__(
        self,
        config: GaussianSceneGridCreatorConfig = GaussianSceneGridCreatorConfig(),
    ):
        self.config = config
        self.grid_g_ids = ti.field(dtype=ti.i32)
        
        self.voxel_size_field = ti.field(dtype=ti.f32, shape=())
        
        # normal sphere radius, which using a sparse data structure to store
        # Optimization: Split into two levels to avoid large contiguous allocation for pointer table (OOM fix)
        # A single pointer table for 1024^3 would be ~8GB. Splitting reduces this drastically.
        l1_size = 32
        l2_size = self.config.max_grid_size // l1_size
        
        self.root = ti.root.pointer(
            ti.ijk, 
            (l1_size, l1_size, l1_size)
        )
        block = self.root.pointer(
            ti.ijk,
            (l2_size, l2_size, l2_size)
        )
        self.pixel = block.dynamic(ti.l, self.config.grid_max_sphere_count)
        self.pixel.place(self.grid_g_ids)
        
        # oversized sphere, using a special list to store
        # for thread load balance
        self.oversized_g_ids = ti.field(ti.i32)
        self.oversized_list = ti.root.dynamic(ti.i, 1024, chunk_size=32)
        self.oversized_list.place(self.oversized_g_ids)

        self.scene_min_offset = ti.field(dtype=ti.f32, shape=(3,))

        
    def _calculate_voxel_size_and_sphere_aabb(
        self,
        scene: GaussianScenes,
    ) -> float:
        
        self.min_corners = torch.zeros_like(scene.position)
        self.max_corners = torch.zeros_like(scene.position)
        self.radius = torch.zeros_like(scene.scales)

        gaussian_scene_aabb(
            centers=scene.position.contiguous(),
            scales=scene.scales.contiguous(),
            quaternions=scene.rotation.contiguous(),
            min_corners=self.min_corners,
            max_corners=self.max_corners,
            radius=self.radius,
            confidence_level=self.config.confidence_level,
        )
        
        global_min_corner, global_max_corner = robust_global_scene_aabb(
            min_corners=self.min_corners,
            max_corners=self.max_corners,
            clip_quantile=self.config.global_aabb_clip_quantile,
            padding_factor=self.config.global_aabb_padding_factor,
        )
        
                
        global_scene_size = global_max_corner - global_min_corner
        
        print(f"global scene size: {global_scene_size}")
        
        median_radius = torch.median(self.radius[:, 0]).item()
        voxel_size = median_radius * self.config.voxel_size_factor 

        print(f"voxel size: {voxel_size}")
        
        self.scene_min_offset.from_torch(global_min_corner)
        
        return voxel_size

    
    def _clear_grid(self):
        self.oversized_list.deactivate_all()
        
        self.root.deactivate_all()


    def _estimate_insert_cost_torch(self, voxel_size: float):
        """Cheap, pre-kernel estimate of how much work insert_gaussian_sphere_kernel will do.

        This is purely for diagnostics: it estimates how many voxel references will be appended.
        """
        if not hasattr(self, 'min_corners') or not hasattr(self, 'max_corners'):
            return

        device = self.min_corners.device
        offset = self.scene_min_offset.to_torch(device=device).view(1, 3)

        # Compute in-bounds grid range per sphere.
        grid_min = torch.floor((self.min_corners - offset) / voxel_size).to(torch.int32)
        grid_max = torch.floor((self.max_corners - offset) / voxel_size).to(torch.int32)

        max_grid = self.config.max_grid_size
        grid_min = torch.clamp(grid_min, 0, max_grid - 1)
        grid_max = torch.clamp(grid_max, 0, max_grid - 1)

        extent = grid_max - grid_min + 1
        extent = torch.clamp(extent, min=0)
        num_voxels = (extent[:, 0].to(torch.int64)
                     * extent[:, 1].to(torch.int64)
                     * extent[:, 2].to(torch.int64))

        threshold = int(self.config.oversized_sphere_voxels_threshold)
        oversized_mask = num_voxels > threshold
        oversized_count = int(oversized_mask.sum().item())

        # In the kernel: oversized -> 1 append; normal -> num_voxels appends.
        total_refs_normal = int(num_voxels[~oversized_mask].sum().item())
        est_total_appends = total_refs_normal + oversized_count

        # A few robust stats help explain runtime.
        def q(v, p):
            if v.numel() == 0:
                return 0
            return int(torch.quantile(v.to(torch.float32), p).item())

        print(
            "[grid_estimate] spheres={}, oversized={} ({:.2f}%), ".format(
                num_voxels.numel(), oversized_count, 100.0 * oversized_count / max(1, num_voxels.numel())
            )
            + "num_voxels per sphere: median={}, p90={}, p99={}, max={}, ".format(
                q(num_voxels, 0.50), q(num_voxels, 0.90), q(num_voxels, 0.99), int(num_voxels.max().item())
            )
            + "est_total_appends={} (normal_refs={}, oversized_appends={})".format(
                est_total_appends, total_refs_normal, oversized_count
            )
        )


    @ti.kernel
    def insert_gaussian_sphere_kernel(
        self,
        min_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [num_gaussians, 3]
        max_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [num_gaussians, 3]
        voxel_size: ti.f32,
    ):
        sphere_counts = min_corners.shape[0]

        # Hoist constants to reduce per-sphere overhead.
        offset = ti.math.vec3([self.scene_min_offset[0], self.scene_min_offset[1], self.scene_min_offset[2]])
        max_grid = ti.cast(self.config.max_grid_size, ti.i32)
        oversized_threshold = ti.cast(self.config.oversized_sphere_voxels_threshold, ti.i32)

        ti.loop_config(block_dim=256)

        for idx in range(sphere_counts):
            p_min = ti.math.vec3([min_corners[idx, 0], min_corners[idx, 1], min_corners[idx, 2]])
            p_max = ti.math.vec3([max_corners[idx, 0], max_corners[idx, 1], max_corners[idx, 2]])

            grid_min = ti.floor((p_min - offset) / voxel_size).cast(ti.i32)
            grid_max = ti.floor((p_max - offset) / voxel_size).cast(ti.i32)

            # Clamp to in-bounds upfront to avoid inner-loop bounds checks.
            grid_min = ti.math.ivec3([
                ti.max(0, grid_min.x),
                ti.max(0, grid_min.y),
                ti.max(0, grid_min.z),
            ])
            grid_max = ti.math.ivec3([
                ti.min(max_grid - 1, grid_max.x),
                ti.min(max_grid - 1, grid_max.y),
                ti.min(max_grid - 1, grid_max.z),
            ])

            if grid_min.x > grid_max.x or grid_min.y > grid_max.y or grid_min.z > grid_max.z:
                continue
            
            extent = grid_max - grid_min + 1
            num_voxels = extent.x * extent.y * extent.z
            
            if num_voxels > oversized_threshold:
                # insert into oversized list
                self.oversized_g_ids.append(idx)
            else:
                for dx, dy, dz in ti.ndrange((0, extent.x), (0, extent.y), (0, extent.z)):
                    gx = grid_min.x + dx
                    gy = grid_min.y + dy
                    gz = grid_min.z + dz
                    self.grid_g_ids[gx, gy, gz].append(idx)


    @ti.kernel
    def get_oversized_count(self) -> ti.i32:
        return ti.length(self.oversized_list, [])

    def grid_build(
        self,
        scene: GaussianScenes,
    ):
        # self._clear_grid()
        ti.deactivate_all_snodes()
        ti.sync()
        
        start_time = time()
        print(f'there are {scene.position.shape[0]} spheres in the scene.')
        voxel_size = self._calculate_voxel_size_and_sphere_aabb(scene)
        self.voxel_size_field[None] = voxel_size

        # Pre-kernel diagnostic: how many appends are we about to do?
        # This helps explain why insert_gaussian_sphere_kernel can take seconds.
        self._estimate_insert_cost_torch(voxel_size)
        
        ti.profiler.print_kernel_profiler_info('trace')
        ti.profiler.clear_kernel_profiler_info()  # Clears all records

        self.insert_gaussian_sphere_kernel(
            min_corners=self.min_corners.contiguous(),
            max_corners=self.max_corners.contiguous(),
            voxel_size=voxel_size,
        )

        ti.sync()
        ti.profiler.print_kernel_profiler_info()  # The default mode: 'count'

        end_time = time()
        print(f'grid building completed in {end_time - start_time:.2f} seconds.')



    @ti.func
    def position_to_grid(self, position: ti.math.vec3) -> ti.math.ivec3:
        offset = ti.math.vec3([self.scene_min_offset[0], self.scene_min_offset[1], self.scene_min_offset[2]])
        return ti.floor((position - offset) / self.voxel_size_field[None]).cast(ti.i32)
    

    @ti.kernel
    def query_topk_sphere_kernel(
        self,
        positions: ti.types.ndarray(dtype=ti.f32, ndim=2),              # [num_points, 3]
        scales: ti.types.ndarray(dtype=ti.f32, ndim=2),                 # [num_gaussians, 3]
        quaternions: ti.types.ndarray(dtype=ti.f32, ndim=2),            # [num_gaussians, 4]
        sorted_points: ti.types.ndarray(dtype=ti.f32, ndim=2),          # [num_points, 3]
        # middle
        tmp_sphere_covariance_inv: ti.types.ndarray(dtype=ti.f32, ndim=3),  # [num_gaussians, 3, 3]
        tmp_sphere_normalized_factor: ti.types.ndarray(dtype=ti.f32, ndim=1), # [num_gaussians]
        # output
        topk_sphere_relations: ti.types.ndarray(dtype=ti.i32, ndim=2),   # [num_points, k]    
        topk_probabilities: ti.types.ndarray(dtype=ti.f32, ndim=2),      # [num_points, k]    
        K: ti.template(),
    ):
        """
        build a point-to-sphere association list for each point, witch will find the spheres with highest confidence for each point, and store the relation in the sphere_relation list.
        
        :param positions: gaussian sphere centers, shape [num_gaussians, 3]
        :type positions: ti.types.ndarray(dtype=ti.f32, ndim=2)
        :param scales: gaussian sphere scales, shape [num_gaussians, 3]
        :type scales: ti.types.ndarray(dtype=ti.f32, ndim=2)
        :param quaternions: quaternions of the rotation of gaussian sphere's covariance , shape [num_gaussians, 4]
        :type quaternions: ti.types.ndarray(dtype=ti.f32, ndim=2)
        :param sorted_points: points in pointcloud sorted by mortoncode (sorted for memory locality) , shape [num_points, 3]
        :type sorted_points: ti.types.ndarray(dtype=ti.f32, ndim=2)
        :param tmp_sphere_covariance_inv: temporary storage for pre-computed covariance inverse for each sphere, shape [num_gaussians, 3, 3]
        :type tmp_sphere_covariance_inv: ti.types.ndarray(dtype=ti.f32, ndim=3)
        :param tmp_sphere_normalized_factor: temporary storage for pre-computed normalized factor for each sphere, shape [num_gaussians]
        :type tmp_sphere_normalized_factor: ti.types.ndarray(dtype=ti.f32, ndim=1)
        :param topk_sphere_relations: output point-to-sphere association list, which will store the top-k most related sphere id for each point, shape [num_points, k]
        :type topk_sphere_relations: ti.types.ndarray(dtype=ti.i32, ndim=2)
        :param topk_probabilities: output confidence list for the top-k most related sphere for each point, shape [num_points, k]
        :type topk_probabilities: ti.types.ndarray(dtype=ti.f32, ndim=2)
        :param K: number of top-k most related spheres to store for each point, default is 8
        :type K: ti.i32
        """
        
        # step 1: pre-compute the inverse of covariance matrix and normalized factor for each sphere, which will be used to calculate the confidence for each point-sphere pair.
        sphere_counts = scales.shape[0]
        
        ti.loop_config(block_dim=128)
        for sphere_idx in range(sphere_counts):
            scale = ti.math.vec3([
                scales[sphere_idx, 0], scales[sphere_idx, 1], scales[sphere_idx, 2]    
            ])
            quaternion = ti.math.vec4([
                quaternions[sphere_idx, 0], quaternions[sphere_idx, 1], quaternions[sphere_idx, 2], quaternions[sphere_idx, 3]    
            ])
            
            covariance_inv = compute_gaussian_covariance(scale, quaternion).inverse()
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                tmp_sphere_covariance_inv[sphere_idx, i, j] = covariance_inv[i, j]
                
            tmp_sphere_normalized_factor[sphere_idx] = compute_gaussian_normalized_factor_cov_inv_ti(covariance_inv)
                

        # step 2: per-point top-k search
        point_counts = sorted_points.shape[0]

        for point_idx in range(point_counts):
            point = ti.math.vec3([
                sorted_points[point_idx, 0], sorted_points[point_idx, 1], sorted_points[point_idx, 2]
            ])

            grid_coord = self.position_to_grid(point)

            # init topk
            topk_ids = ti.Vector([-1 for _ in range(32)])
            topk_vals = ti.Vector([0.0 for _ in range(32)])

            if 0 <= grid_coord.x < self.config.max_grid_size and 0 <= grid_coord.y < self.config.max_grid_size and 0 <= grid_coord.z < self.config.max_grid_size:
                grid_sphere_counts = ti.length(self.pixel, [grid_coord.x, grid_coord.y, grid_coord.z])

                # scan voxel spheres
                for sphere_local_idx in range(grid_sphere_counts):
                    g_id = self.grid_g_ids[grid_coord.x, grid_coord.y, grid_coord.z, sphere_local_idx]

                    local_covariance_inv = ti.Matrix.zero(ti.f32, 3, 3)
                    for i, j in ti.static(ti.ndrange(3, 3)):
                        local_covariance_inv[i, j] = tmp_sphere_covariance_inv[g_id, i, j]

                    diff = point - ti.math.vec3([
                        positions[g_id, 0], positions[g_id, 1], positions[g_id, 2]
                    ])

                    # topk needed normalized confidence
                    confidence = gaussian_density_cov_inv_ti(
                        diff=diff,
                        cov_inv=local_covariance_inv,
                        cov_det=tmp_sphere_normalized_factor[g_id],
                    )

                    # insert into topk
                    is_inserted = 0
                    for k in ti.static(range(K)):
                        if is_inserted == 0 and confidence > topk_vals[k]:
                            for j in ti.static(range(K - 1, k, -1)):
                                topk_vals[j] = topk_vals[j - 1]
                                topk_ids[j] = topk_ids[j - 1]
                            topk_vals[k] = confidence
                            topk_ids[k] = g_id
                            is_inserted = 1

            # oversized spheres
            oversized_counts = ti.length(self.oversized_list, [])
            for oversized_idx in range(oversized_counts):
                g_id = self.oversized_g_ids[oversized_idx]

                local_covariance_inv = ti.Matrix.zero(ti.f32, 3, 3)
                for i, j in ti.static(ti.ndrange(3, 3)):
                    local_covariance_inv[i, j] = tmp_sphere_covariance_inv[g_id, i, j]

                diff = point - ti.math.vec3([
                    positions[g_id, 0], positions[g_id, 1], positions[g_id, 2]
                ])

                confidence = gaussian_density_cov_inv_ti(
                    diff=diff, 
                    cov_inv=local_covariance_inv, 
                    cov_det=tmp_sphere_normalized_factor[g_id]
                )

                is_inserted = 0
                for k in ti.static(range(K)):
                    if is_inserted == 0 and confidence > topk_vals[k]:
                        for j in ti.static(range(K - 1, k, -1)):
                            topk_vals[j] = topk_vals[j - 1]
                            topk_ids[j] = topk_ids[j - 1]
                        topk_vals[k] = confidence
                        topk_ids[k] = g_id
                        is_inserted = 1

            # write back
            for k in ti.static(range(K)):
                topk_sphere_relations[point_idx, k] = ti.cast(topk_ids[k], ti.i32)
                topk_probabilities[point_idx, k] = topk_vals[k]


    @ti.kernel
    def compute_statistics_kernel(
        self,
        # Inputs
        min_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [num_gaussians, 3]
        max_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [num_gaussians, 3]
        voxel_size: ti.f32,
        max_bins: ti.i32,
        # output
        hist_spheres_per_voxel: ti.types.ndarray(dtype=ti.i32, ndim=1), # [max_bins]
        hist_voxels_per_sphere: ti.types.ndarray(dtype=ti.i32, ndim=1), # [max_bins]
        grid_usage_info: ti.types.ndarray(dtype=ti.i32, ndim=1),        # [4]: [active_voxels, max_spheres_in_voxel, total_refs, 0]
    ):
        # task 1. statistic the number of voxels each sphere occupies, and build a histogram of the voxel counts for all spheres. This can help us to understand the distribution of sphere sizes and choose appropriate thresholds for oversized spheres.
        sphere_counts = min_corners.shape[0]
        ti.loop_config(block_dim=128)
        for idx in range(sphere_counts):
            offset = ti.math.vec3([self.scene_min_offset[0], self.scene_min_offset[1], self.scene_min_offset[2]])
            p_min = ti.math.vec3([min_corners[idx, 0], min_corners[idx, 1], min_corners[idx, 2]])
            p_max = ti.math.vec3([max_corners[idx, 0], max_corners[idx, 1], max_corners[idx, 2]])

            grid_min = ti.floor((p_min - offset) / voxel_size).cast(ti.i32)
            grid_max = ti.floor((p_max - offset) / voxel_size).cast(ti.i32)
            
            extent = grid_max - grid_min + 1
            num_voxels = extent.x * extent.y * extent.z
            
            bin_idx = ti.min(num_voxels, max_bins - 1)

            ti.atomic_add(hist_voxels_per_sphere[bin_idx], 1)

        # task 2. statistic the number of spheres in each voxel, and build a histogram of the sphere counts for all voxels. This can help us to understand the distribution of sphere density in the grid and choose appropriate thresholds for grid optimization.
        max_s_in_v = 0
        total_refs = 0
        
        active_voxel_count = 0 

        for I in ti.grouped(self.pixel):
            count = ti.length(self.pixel, I)
            
            if count > 0:
                ti.atomic_add(active_voxel_count, 1)
                ti.atomic_max(max_s_in_v, count)
                ti.atomic_add(total_refs, count)
                
                bin_idx = ti.min(count, max_bins - 1)
                ti.atomic_add(hist_spheres_per_voxel[bin_idx], 1)
        
        grid_usage_info[0] = active_voxel_count
        grid_usage_info[1] = max_s_in_v
        grid_usage_info[2] = total_refs


@dataclass
class GaussianPointMatcherConfig:
    topk_K: int = 8
    scene_grid_config: GaussianSceneGridCreatorConfig = GaussianSceneGridCreatorConfig()
    


class GaussianPointMatcher(torch.nn.Module):
    def __init__(
        self,
        config: GaussianPointMatcherConfig = GaussianPointMatcherConfig(),
    ):
        super().__init__()
        
        self.config = config
        
        self.grid_creator = GaussianSceneGridCreator(
            config=self.config.scene_grid_config,
        )
        self.morton_code_calculator = Morton3D()
    

    def forward(
        self,
        scene: GaussianScenes,
        pointcloud: torch.Tensor,  # [num_points, 3]
        pointcloud_color: Optional[torch.Tensor]=None,  # [num_points, 3]
    ):
        """compute the point-to-sphere association for each point in the pointcloud, which will be used for later point-to-Gaussian registration.

        Args:
            scene (GaussianScenes): gaussian scene conytains the Gaussian spheres, including their positions, scales and quaternions.
            pointcloud (torch.Tensor): point cloud tensor of shape [num_points, 3]
            pointcloud_color (Optional[torch.Tensor], optional): optional color information for each point in the point cloud. Defaults to None.

        Returns:
            dict: a dictionary containing sorted pointcloud, sphere relations, best probabilities, and grid information.
        """
        
        # step 1. build the grid for gaussian spheres
        print("Building grid for Gaussian spheres...")
        self.grid_creator.grid_build(scene)
        
        
        # step 2. sort the pointcloud by morton code
        print("Sorting pointcloud by Morton code...")
        print(f'there are {pointcloud.shape[0]} points in the pointcloud.')
        sorted_code_dict = self.morton_code_calculator(
            pointcloud=pointcloud,
            color=pointcloud_color,
        )
        
        # step 3. query the grid to find the sphere relation for each point
        print("Querying grid for point-to-sphere association...")
        topk_sphere_relations = torch.full((pointcloud.shape[0], self.config.topk_K), -1, dtype=torch.int32)
        topk_probabilities = torch.zeros((pointcloud.shape[0], self.config.topk_K), dtype=torch.float32)
        
        tmp_covariances_inv = torch.zeros((scene.position.shape[0], 3, 3), dtype=torch.float32)
        tmp_normalized_factor = torch.zeros((scene.position.shape[0]), dtype=torch.float32)
        
        # self.grid_creator.query_topk_sphere_kernel(
        #     positions=scene.position.contiguous(),
        #     scales=scene.scales.contiguous(),
        #     quaternions=scene.rotation.contiguous(),
        #     sorted_points=sorted_code_dict['sorted_pointcloud'].contiguous(),
        #     tmp_sphere_covariance_inv=tmp_covariances_inv,
        #     tmp_sphere_normalized_factor=tmp_normalized_factor,
        #     topk_sphere_relations=topk_sphere_relations,
        #     topk_probabilities=topk_probabilities,
        #     K=self.config.topk_K,
        # )
        
        return {
            **sorted_code_dict,
            'topk_sphere_relations': topk_sphere_relations,
            'topk_probabilities': topk_probabilities,
            'tmp_covariances_inv': tmp_covariances_inv,
            'tmp_normalized_factor': tmp_normalized_factor,
            'grid_info': {
                'voxel_size': self.grid_creator.voxel_size_field[None],
                'oversized_sphere_count': self.grid_creator.get_oversized_count(),
            }
        }
        
        
    def compute_statistics(self, scene: GaussianScenes, max_bins: int = 128):
        device = scene.position.device

        # step 1. prepare output tensors for statistics
        hist_spheres_per_voxel = torch.zeros(max_bins, dtype=torch.int32, device=device)
        hist_voxels_per_sphere = torch.zeros(max_bins, dtype=torch.int32, device=device)
        grid_usage_info = torch.zeros(4, dtype=torch.int32, device=device) # [active_voxels, max_spheres, total_refs, padding]

        # step 2. run the statistics kernel
        voxel_size = self.voxel_size_field[None]

        self.compute_statistics_kernel(
            min_corners=self.min_corners.contiguous(),
            max_corners=self.max_corners.contiguous(),
            voxel_size=voxel_size,
            max_bins=max_bins,
            hist_spheres_per_voxel=hist_spheres_per_voxel,
            hist_voxels_per_sphere=hist_voxels_per_sphere,
            grid_usage_info=grid_usage_info
        )
        
        active_voxels = grid_usage_info[0].item()
        max_spheres_in_voxel = grid_usage_info[1].item()
        total_refs = grid_usage_info[2].item()
        oversized_count = self.grid_creator.get_oversized_count()
        
        return {
            "histograms": {
                "spheres_per_voxel": hist_spheres_per_voxel, # Tensor
                "voxels_per_sphere": hist_voxels_per_sphere, # Tensor
                "bins": torch.arange(max_bins, device=device)
            },
            "scalars": {
                "voxel_size": voxel_size,
                "active_voxels": active_voxels,
                "max_spheres_in_one_voxel": max_spheres_in_voxel,
                "oversized_sphere_count": oversized_count,
                "avg_spheres_per_voxel": total_refs / max(1, active_voxels),
                "avg_voxels_per_sphere": total_refs / max(1, scene.position.shape[0])
            }
        }


if __name__ == "__main__":
    from pathlib import Path
    from misc.hier_IO import load_hier_to_torch
    from misc.colmap_read import read_colmap_points3d_ply
    
    ti.init(arch=ti.cuda, kernel_profiler = True)
    
    hier_path = Path('/home/wangjv_wsl/data/3dgs_dataset/hierachy/replica/office1/output/merged.hier')
    pointcloud_path = Path('/home/wangjv_wsl/data/3dgs_dataset/hierachy/replica/office1/camera_calibration/aligned/sparse/0/points3D.ply')
    
    hier_scene = load_hier_to_torch(
        hier_path=hier_path,
        device=torch.device("cuda"),
    )
    
    gaussian_scene = hier_scene.gaussian_scene
    
    pointcloud_dict = read_colmap_points3d_ply(
        pointcloud_path, 
        device='cuda'
    )

    config = GaussianPointMatcherConfig(
        topk_K=8,
        scene_grid_config=GaussianSceneGridCreatorConfig(
            confidence_level=0.95,
            voxel_size_factor=1.0,
            max_active_cells=2 ** 20,
            grid_max_sphere_count=1024,
            global_aabb_padding_factor=0.1,
            global_aabb_clip_quantile=0.01,
            oversized_sphere_voxels_threshold=64,
            max_grid_size=2**10
        )
    )
    
    matcher = GaussianPointMatcher(config=config)
    
    matcher_dict = matcher(
        scene=gaussian_scene,
        pointcloud=pointcloud_dict["pointcloud"],
    )
    
    