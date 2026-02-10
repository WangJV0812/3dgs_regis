import taichi as ti
import torch
from dataclasses import dataclass
from typing import Optional

from misc.hier_IO import GaussianScenes
from gmm_point_alignment.gs_scene_aabb import (
    robust_global_scene_aabb,
    gaussian_scene_aabb,
)
from misc.geometry import (
    compute_gaussian_covariance,
    unnormalized_gaussian_density_ti
)
from gmm_point_alignment.morton_code import Morton3D


@dataclass
class GaussianSceneGridCreatorConfig:
    confidence_level: float = 0.95
    voxel_size_factor: float = 3.0
    max_active_cells: int = 2 ** 20
    voxel_max_sphere_count: int = 1024  # maximum number of spheres might in each voxel
    global_aabb_padding_factor: float = 0.1
    global_aabb_clip_quantile: float = 0.01
    oversized_sphere_voxels_threshold: int = 64  # if a sphere overlaps with more than this number of voxels, it will be considered as oversized    

@ti.data_oriented
class GaussianSceneGridCreator:
    
    def __init__(
        self,
        config: GaussianSceneGridCreatorConfig = GaussianSceneGridCreatorConfig(),
    ):
        self.config = config
        self.grid_g_ids = ti.field(dtype=ti.i32)
        
        self.voxel_size_field = ti.field(dtype=ti.f32, shape=())
        
        # normal sphere radius, which using a hash snode to store
        self.root = ti.root.hash(ti.ijk, self.config.max_active_cells)
        self.pixel = self.root.dynamic(ti.l, self.config.voxel_max_sphere_count, chunksize=32)
        self.pixel.place(self.grid_g_ids)
        
        # oversized sphere, using a special list to store
        # for thread load balance
        self.oversized_g_ids = ti.field(ti.i32)
        self.oversized_list = ti.root.dynamic(ti.i, 1024, chunk_size=32)
        self.oversized_list.place(self.oversized_g_ids)

        
        
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
        return voxel_size


    @ti.kernel
    def _clear_grid_kernel(self):
        for I in ti.grouped(self.pixel):
            ti.deactivate(self.pixel, I)
        
        ti.deactivate(self.oversized_list, slice(0))


    @ti.kernel
    def insert_gaussian_sphere_kernel(
        self,
        min_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [num_gaussians, 3]
        max_corners: ti.types.ndarray(dtype=ti.f32, ndim=2),  # [num_gaussians, 3]
        voxel_size: ti.f32,
    ):
        sphere_counts = min_corners.shape[0]
        
        ti.loop_config(block_dim=128)
        
        for idx in range(sphere_counts):
            grid_min = ti.floor(
                ti.math.vec3([min_corners[idx, 0], min_corners[idx, 1], min_corners[idx, 2]]) / voxel_size  
            ).cast(ti.i32)
            
            grid_max = ti.floor(
                ti.math.vec3([max_corners[idx, 0], max_corners[idx, 1], max_corners[idx, 2]]) / voxel_size  
            ).cast(ti.i32)
            
            extent = grid_max - grid_min + 1
            num_voxels = extent.x * extent.y * extent.z
            
            if num_voxels > self.config.oversized_sphere_voxels_threshold:
                # insert into oversized list
                oversized_idx = ti.atomic_add(self.oversized_list.length(), 1)
                self.oversized_g_ids[oversized_idx] = idx
            else:
                for x, y, z in ti.ndrange(extent.x, extent.y, extent.z):
                    self.pixel[x, y, z].append(idx)


    def grid_build(
        self,
        scene: GaussianScenes,
    ):
        self._clear_grid_kernel()
        
        voxel_size = self._calculate_voxel_size_and_sphere_aabb(scene)
        self.voxel_size_field[None] = voxel_size

        self.insert_gaussian_sphere_kernel(
            min_corners=self.min_corners.contiguous(),
            max_corners=self.max_corners.contiguous(),
            voxel_size=voxel_size,
        )


    @ti.func
    def position_to_grid(self, position: ti.math.vec3) -> ti.math.ivec3:
        return ti.floor(position / self.voxel_size_field[None]).cast(ti.i32)
    

    @ti.kernel
    def query_topk_sphere_kernel(
        self,
        positions: ti.types.ndarray(dtype=ti.f32, ndim=2),              # [num_points, 3]
        scales: ti.types.ndarray(dtype=ti.f32, ndim=2),                 # [num_gaussians, 3]
        quaternions: ti.types.ndarray(dtype=ti.f32, ndim=2),            # [num_gaussians, 4]
        sorted_points: ti.types.ndarray(dtype=ti.f32, ndim=2),          # [num_points, 3]
        # middle
        tmp_sphere_covariance: ti.types.ndarray(dtype=ti.f32, ndim=3),  # [num_gaussians, 3, 3]
        # output
        topk_sphere_relation: ti.types.ndarray(dtype=ti.f32, ndim=2),   # [num_points, k]    
        topk_probabilitys: ti.types.ndarray(dtype=ti.f32, ndim=2),      # [num_points, k]    
        K: ti.i32 = 8,
    ):
        """ build a point-to-sphere association list for each point, witch will find the spheres with highest confidence for each point, and store the relation in the sphere_relation list.

        Args:
            positions (ti.types.ndarray): positions of the Gaussian spheres, shape (num_gaussians, 3).
            scales (ti.types.ndarray): scales of the Gaussian spheres, shape (num_gaussians, 3).
            quaternions (ti.types.ndarray): quaternions of the Gaussian spheres, shape (num_gaussians, 4).
            sorted_points (ti.types.ndarray): points list sorted by mortoncode for spatial locality. shape (num_points, 3).
            tmp_sphere_covariance (ti.types.ndarray): temporary storage for sphere covariance matrices, needed torch help to manage this memory. shape (num_gaussians, 3, 3).
            topk_sphere_relation (ti.types.ndarray): output lists of sphere id for each point sorted in mortoncode order. shape (num_points, k).
            topk_probabilitys (ti.types.ndarray): output list of best confidence for each point. shape (num_points, k).
            K (int): number of top-k spheres to find for each point, can use ti.static forcing taichi kernel expand the loop. Defaults to 8.
        """
        
        # step 1: pre-compute the covariance matrix for each sphere, which will be used to calculate the confidence for each point-sphere pair.
        sphere_counts = scales.shape[0]
        
        ti.loop_config(block_dim=128)
        for sphere_idx in range(sphere_counts):
            scale = ti.math.vec3([
                scales[sphere_idx, 0], scales[sphere_idx, 1], scales[sphere_idx, 2]    
            ])
            quaternion = ti.math.vec4([
                quaternions[sphere_idx, 0], quaternions[sphere_idx, 1], quaternions[sphere_idx, 2], quaternions[sphere_idx, 3]    
            ])
            
            covariance = compute_gaussian_covariance(scale, quaternion)
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                tmp_sphere_covariance[sphere_idx, i, j] = covariance[i, j]
                

        # step 2: per-point top-k search
        point_counts = sorted_points.shape[0]

        for point_idx in range(point_counts):
            point = ti.math.vec3([
                sorted_points[point_idx, 0], sorted_points[point_idx, 1], sorted_points[point_idx, 2]
            ])

            grid_coord = self.position_to_grid(point)

            # init topk
            topk_ids = ti.Vector([-1 for _ in range(32)])  # assume K <= 32
            topk_vals = ti.Vector([0.0 for _ in range(32)])

            grid_sphere_counts = self.pixel[grid_coord.x, grid_coord.y, grid_coord.z].length()

            # scan voxel spheres
            for sphere_local_idx in range(grid_sphere_counts):
                g_id = self.pixel[grid_coord.x, grid_coord.y, grid_coord.z][sphere_local_idx]

                local_covariance = ti.Matrix.zero(ti.f32, 3, 3)
                for i, j in ti.static(ti.ndrange(3, 3)):
                    local_covariance[i, j] = tmp_sphere_covariance[g_id, i, j]

                diff = point - ti.math.vec3([
                    positions[g_id, 0], positions[g_id, 1], positions[g_id, 2]
                ])

                confidence = unnormalized_gaussian_density_ti(diff=diff, covariance=local_covariance)

                # insert into topk
                for k in range(K):
                    if confidence > topk_vals[k]:
                        for j in range(K - 1, k, -1):
                            topk_vals[j] = topk_vals[j - 1]
                            topk_ids[j] = topk_ids[j - 1]
                        topk_vals[k] = confidence
                        topk_ids[k] = g_id
                        break

            # oversized spheres
            oversized_counts = self.oversized_list.length()
            for oversized_idx in range(oversized_counts):
                g_id = self.oversized_g_ids[oversized_idx]

                local_covariance = ti.Matrix.zero(ti.f32, 3, 3)
                for i, j in ti.static(ti.ndrange(3, 3)):
                    local_covariance[i, j] = tmp_sphere_covariance[g_id, i, j]

                diff = point - ti.math.vec3([
                    positions[g_id, 0], positions[g_id, 1], positions[g_id, 2]
                ])

                confidence = unnormalized_gaussian_density_ti(diff=diff, covariance=local_covariance)

                for k in range(K):
                    if confidence > topk_vals[k]:
                        for j in range(K - 1, k, -1):
                            topk_vals[j] = topk_vals[j - 1]
                            topk_ids[j] = topk_ids[j - 1]
                        topk_vals[k] = confidence
                        topk_ids[k] = g_id
                        break

            # write back
            for k in range(K):
                topk_sphere_relation[point_idx, k] = ti.cast(topk_ids[k], ti.f32)
                topk_probabilitys[point_idx, k] = topk_vals[k]



@dataclass
class GaussianPointMatcherConfig:
    grid_creator_config: GaussianSceneGridCreatorConfig = GaussianSceneGridCreatorConfig()


class GaussianPointMatcher(torch.nn.Module):
    def __init__(
        self,
        config: GaussianPointMatcherConfig = GaussianPointMatcherConfig(),
    ):
        super().__init__()
        
        self.config = config
        
        self.grid_creator = GaussianSceneGridCreator()
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
        self.grid_creator.grid_build(scene)
        
        # step 2. sort the pointcloud by morton code
        sorted_code_dict = self.morton_code_calculator(
            pointcloud=pointcloud,
            color=pointcloud_color,
        )
        
        # step 3. query the grid to find the sphere relation for each point
        K = 8
        sphere_relations = torch.full((pointcloud.shape[0], K), -1, dtype=torch.int32)
        best_probabilitys = torch.zeros((pointcloud.shape[0], K), dtype=torch.float32)
        
        tmp_covariances = torch.zeros((scene.position.shape[0], 3, 3), dtype=torch.float32)
        
        self.grid_creator.query_topk_sphere_kernel(
            positions=scene.position.contiguous(),
            scales=scene.scales.contiguous(),
            quaternions=scene.rotation.contiguous(),
            sorted_points=sorted_code_dict['sorted_pointcloud'].contiguous(),
            tmp_sphere_covariance=tmp_covariances,
            topk_sphere_relation=sphere_relations,
            topk_probabilitys=best_probabilitys,
        )
        
        del tmp_covariances
        
        return {
            **sorted_code_dict,
            'sphere_relations': sphere_relations,
            'best_probabilitys': best_probabilitys,
            'grid_info': {
                'voxel_size': self.grid_creator.voxel_size_field[None],
                'oversized_sphere_count': self.grid_creator.oversized_list.length(),
            }
        }