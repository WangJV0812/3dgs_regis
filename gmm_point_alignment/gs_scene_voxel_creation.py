import taichi as ti
import torch
from dataclasses import dataclass

from misc.hier_IO import GaussianScenes
from gmm_point_alignment.gs_scene_aabb import (
    robust_global_scene_aabb,
    gaussian_scene_aabb,
)


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
        
        self.current_voxel_size = ti.field(dtype=ti.f32, shape=())
        self.corrent_voxel_size = 1.0
        
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
            
    def build(
        self,
        scene: GaussianScenes,
    ):
        self._clear_grid_kernel()
        
        self.voxel_size = self._calculate_voxel_size_and_sphere_aabb(scene)
        
        self.insert_gaussian_sphere_kernel(
            min_corners=self.min_corners.contiguous(),
            max_corners=self.max_corners.contiguous(),
            voxel_size=self.voxel_size,
        )
        

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
    
    