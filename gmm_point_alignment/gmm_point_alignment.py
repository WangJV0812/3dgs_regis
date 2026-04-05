import taichi as ti
import torch
from dataclasses import dataclass

from morton_code import (
    expand_bits_10,
    compact_bits_10
)
from gmm_point_alignment.gs_scene_aabb import (
    robust_global_scene_aabb,
    gaussian_scene_aabb,
)
from misc.hier_IO import GaussianScenes
from misc.geometry import (
    gaussian_density_ti,
    compute_gaussian_covariance,
    compute_gaussian_normalized_factor_cov_inv_ti
)


@dataclass
class GMMPointAlignmentConfig:
    confidence_level: float = 0.95
    global_aabb_clip_quantile: float = 0.01
    global_aabb_padding_factor: float = 0.1
    voxel_size_factor: float = 0.1
    gaussian_confidence_threshold: float = 0.01


def calculate_voxel_size_and_sphere_aabb(
    scene: GaussianScenes,
    confidence_level: float = 0.95,
    global_aabb_clip_quantile: float = 0.01,
    global_aabb_padding_factor: float = 0.1,
    voxel_size_factor: float = 0.1,
) -> float:
    
    min_corners = torch.zeros_like(scene.position)
    max_corners = torch.zeros_like(scene.position)
    radius = torch.zeros_like(scene.scales)

    gaussian_scene_aabb(
        centers=scene.position.contiguous(),
        scales=scene.scales.contiguous(),
        quaternions=scene.rotation.contiguous(),
        min_corners=min_corners,
        max_corners=max_corners,
        radius=radius,
        confidence_level=confidence_level,
    )
    
    global_min_corner, global_max_corner = robust_global_scene_aabb(
        min_corners=min_corners,
        max_corners=max_corners,
        clip_quantile=global_aabb_clip_quantile,
        padding_factor=global_aabb_padding_factor,
    )
       
    global_scene_size = global_max_corner - global_min_corner
    
    print(f"global scene size: {global_scene_size}")
    
    median_radius = torch.median(radius[:, 0]).item()
    voxel_size = median_radius * voxel_size_factor 

    print(f"voxel size: {voxel_size}")
        
    return {
        'spheres': {
            'aabb_min_corners': min_corners,
            'aabb_max_corners': max_corners,
            'radius': radius,
        },
        'global': {
            'aabb_min_corner': global_min_corner,
            'aabb_max_corner': global_max_corner,
            'voxel_size': voxel_size,
        },
    }
    
    
def sphere_grid_counts(
    aabb_min: torch.Tensor,   # (N,3)
    aabb_max: torch.Tensor,   # (N,3)
    voxel_size: float,
):
    """sphere_grid_counts compute the number of voxels each sphere occupies in the grid, as well as the prefix sum for indexing.

    Args:
        aabb_min (torch.Tensor): min corner of the AABB for each sphere, shape (N, 3)
        aabb_max (torch.Tensor): max corner of the AABB for each sphere, shape (N, 3)
        voxel_size (float): the size of each voxel in the grid

    Returns:
        dict: {
            "grid_min": grid_min,  # (N, 3) the min grid coordinate for each sphere
            "grid_max": grid_max,  # (N, 3) the max grid coordinate for each sphere
            "counts": counts,      # (N,) the number of voxels each sphere occupies
            "prefix": prefix_exclusive,  # (N,) the exclusive prefix sum for indexing
            "total_pairs": total, # int, the total number of sphere-voxel pairs
        } 
    """
    grid_min = torch.floor(aabb_min / voxel_size).int()
    grid_max = torch.floor(aabb_max / voxel_size).int()

    extent = grid_max - grid_min + 1
    counts = extent.prod(dim=1)                     # (N,)

    prefix = torch.cumsum(counts, dim=0)
    prefix_exclusive = prefix - counts              # (N,)

    total = counts.sum()

    return {
        "grid_min": grid_min,
        "grid_max": grid_max,
        "counts": counts,
        "prefix": prefix_exclusive,
        "total_pairs": total,
    }


@ti.func
def morton3D_encoder(
    point: ti.math.vec3,
    bounding_extent: ti.math.vec3,
    bounding_min: ti.math.vec3,
)-> ti.u32:
    normalized_point = (point - bounding_min) / (bounding_extent + 1e-7)
    x, y, z = normalized_point.x, normalized_point.y, normalized_point.z

    x = ti.cast(x * 1024, ti.u32)
    y = ti.cast(y * 1024, ti.u32)
    z = ti.cast(z * 1024, ti.u32)

    return expand_bits_10(x) | (expand_bits_10(y) << 1) | (expand_bits_10(z) << 2)


@ti.func
def morton3D_decoder(
    code: ti.u32,
    bounding_extent: ti.math.vec3,
    bounding_min: ti.math.vec3,
) -> ti.math.vec3:
    x = compact_bits_10(code >> 2)
    y = compact_bits_10(code >> 1)
    z = compact_bits_10(code)

    normalized_point = ti.math.vec3(x, y, z) / 1024.0
    point = normalized_point * bounding_extent + bounding_min
    return point


@ti.kernel
def gmm_grid_creation(
    sphere_grid_min: ti.types.ndarray(ti.i32, 2),
    sphere_grid_max: ti.types.ndarray(ti.i32, 2),
    sphere_presum: ti.types.ndarray(ti.i32, 1),
    global_min: ti.types.ndarray(ti.f32, 1),  # (3,)
    global_extent: ti.types.ndarray(ti.f32, 1),  # (3)
    
    out_pairs: ti.types.ndarray(ti.i32, 2)  # (total_pairs,2) -> (hash_idx, sphere_id)
):

    sphere_counts = sphere_grid_min.shape[0]

    for sphere_idx in range(sphere_counts):
        minx = sphere_grid_min[sphere_idx, 0]
        miny = sphere_grid_min[sphere_idx, 1]
        minz = sphere_grid_min[sphere_idx, 2]

        maxx = sphere_grid_max[sphere_idx, 0]
        maxy = sphere_grid_max[sphere_idx, 1]
        maxz = sphere_grid_max[sphere_idx, 2]

        dy = maxy - miny + 1
        dz = maxz - minz + 1

        base = sphere_presum[sphere_idx]
        bounding_extent = ti.math.vec3(global_extent[0], global_extent[1], global_extent[2])
        bounding_min = ti.math.vec3(global_min[0], global_min[1], global_min[2])

        for x in range(minx, maxx + 1):
            for y in range(miny, maxy + 1):
                for z in range(minz, maxz + 1):

                    local_idx = (x - minx) * dy * dz + (y - miny) * dz + (z - minz)
                    global_idx = base + local_idx

                    hash_index = morton3D_encoder(
                        ti.math.vec3(x, y, z),
                        bounding_extent,
                        bounding_min, 
                    )

                    out_pairs[global_idx, 0] = hash_index
                    out_pairs[global_idx, 1] = sphere_idx


@ti.func
def get_grid_index(pos: ti.math.vec3, global_min: ti.math.vec3, grid_size: ti.math.vec3) -> ti.math.uvec3:

    normalized = (pos - global_min) / grid_size

    idx = ti.cast(ti.floor(normalized * 1024), ti.u32)
    return ti.math.clamp(idx, 0, 1023)


@ti.func
def binary_search_range(
    code: ti.u32, 
    pairs: ti.types.ndarray(ti.i32, 2),
    total_pairs: ti.i32
) -> ti.math.vec2:
    
    l, r = 0, total_pairs
    while l < r:
        mid = l + (r - l) // 2
        if ti.cast(pairs[mid, 0], ti.u32) < code:
            l = mid + 1
        else:
            r = mid
    start = l

    l, r = start, total_pairs
    while l < r:
        mid = l + (r - l) // 2
        if ti.cast(pairs[mid, 0], ti.u32) <= code:
            l = mid + 1
        else:
            r = mid
    end = l
    
    return ti.math.vec2(start, end)


@ti.kernel
def query_top_k(
    K: ti.i32,
    gaussian_confidence_threshold: ti.f32,
    pointcloud: ti.types.ndarray(ti.f32, 2),  # (N, 3)
    global_extent: ti.types.ndarray(ti.f32, 1),  # (3,)
    global_min: ti.types.ndarray(ti.f32, 1),  # (3,)
    sphere_grid_pairs: ti.types.ndarray(ti.i32, 2),  # (total_pairs, 2) -> (hash_idx, sphere_id)
    position: ti.types.ndarray(ti.f32, 2),  # (M, 3)
    rotation: ti.types.ndarray(ti.f32, 2),  # (M, 4)
    scles: ti.types.ndarray(ti.f32, 2),  # (M, 3)
    # middle variables 
    covariance_inv_buffer: ti.types.ndarray(ti.f32, 3),  # (M, 3, 3)
    normalized_distance_buffer: ti.types.ndarray(ti.f32, 1),  # (N, M)
    # output
    topk_sphere_ids: ti.types.ndarray(ti.i32, 2),  # (N, K)
):
    sphere_counts = position.shape[0]
    for sphere_idx in range(sphere_counts):
        local_rotation = ti.math.vec4([
            rotation[sphere_idx, 0],
            rotation[sphere_idx, 1],
            rotation[sphere_idx, 2],
            rotation[sphere_idx, 3],
        ])
        
        local_scale = ti.math.vec3([
            scles[sphere_idx, 0],
            scles[sphere_idx, 1],
            scles[sphere_idx, 2],
        ])
        
        local_cov_inv = compute_gaussian_covariance(local_scale, local_rotation).inverse()
        
        for i, j in ti.static(ti.ndrange(3, 3)):
            covariance_inv_buffer[sphere_idx, i, j] = local_cov_inv[i, j]
        
        normalized_distance_buffer[sphere_idx] = compute_gaussian_normalized_factor_cov_inv_ti(
            local_cov_inv
        )
        
    
    
    point_counts = pointcloud.shape[0]
    global_extent_ti = ti.math.vec3(global_extent[0], global_extent[1], global_extent[2])
    global_min_ti = ti.math.vec3(global_min[0], global_min[1], global_min[2])
    total_pairs = sphere_grid_pairs.shape[0]
    
    for point_idx in range(point_counts):
        local_point = ti.math.vec3([
            pointcloud[point_idx, 0], 
            pointcloud[point_idx, 1], 
            pointcloud[point_idx, 2]
        ])
        
        point_hash = morton3D_encoder(
            local_point,
            global_extent_ti,
            global_min_ti,
        )
        
        range = binary_search_range(
            point_hash,
            sphere_grid_pairs,
            total_pairs,
        )

        range_start, range_end = range.x, range.y
        
        for pairs_idx in range(range_start, range_end):
            sphere_id = sphere_grid_pairs[pairs_idx, 1]
            
            local_cov_inv = ti.Matrix.zeros(ti.f32, 3, 3)
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                local_cov_inv[i, j] = covariance_inv_buffer[sphere_id, i, j]
                
            local_pos = ti.math.vec3([
                position[sphere_id, 0],
                position[sphere_id, 1],
                position[sphere_id, 2],
            ])
            
            diff = local_point - local_pos
            normalized_distance = diff.dot(local_cov_inv @ d iff) + normalized_distance_buffer[sphere_id]
            
            if normalized_distance < gaussian_confidence_threshold:
                topk_sphere_ids[point_idx, 0] = sphere_id
                break
            
            



class GMMPointAlignment(torch.nn.Module):
    def __init__(
        self,
        config: GMMPointAlignmentConfig = GMMPointAlignmentConfig(),
    ):
        super(GMMPointAlignment, self).__init__()
        self.config = config
        
    
    def forward(
        self,
        scene: GaussianScenes,
    ):
        scene_aabb_dict = calculate_voxel_size_and_sphere_aabb(
            scene=scene,
            confidence_level=self.config.confidence_level,
            global_aabb_clip_quantile=self.config.global_aabb_clip_quantile,
            global_aabb_padding_factor=self.config.global_aabb_padding_factor,
            voxel_size_factor=self.config.voxel_size_factor,
        )
        
        sphere_counts = scene.position.shape[0]
        
        
