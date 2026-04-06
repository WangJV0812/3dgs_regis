import taichi as ti
import torch
from typing import Optional, Dict, Optional

@ti.func
def expand_bits_10(v: ti.u32) -> ti.u32:
    x = ti.cast(v, ti.u32)
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x <<  8)) & 0x0300F00F
    x = (x | (x <<  4)) & 0x030C30C3
    x = (x | (x <<  2)) & 0x09249249
    return x


@ti.func
def compact_bits_10(x: ti.u32) -> ti.u32:
    x &= 0x09249249
    x = (x ^ (x >> 2)) & 0x030C30C3
    x = (x ^ (x >> 4)) & 0x0300F00F
    x = (x ^ (x >> 8)) & 0x030000FF
    x = (x ^ (x >> 16)) & 0x000003FF
    return x


@ti.func
def decode_morton32(code: ti.u32):
    return compact_bits_10(code >> 2), compact_bits_10(code >> 1), compact_bits_10(code)


@ti.kernel
def Morton3D_kernel(
    pointcloud: ti.types.ndarray(dtype=ti.f32, ndim=2),
    morton_codes: ti.types.ndarray(dtype=ti.u32, ndim=1),
    bounding_box: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    """_summary_

    Args:
        pointcloud (ti.types.ndarray, optional): input, pointcloud to calculate morton code for each point. \in R^(N, 3)
        morton_codes (ti.types.ndarray, optional): output, morton code for each points. \in R^{N}
        bounding_box (ti.types.ndarray, optional): bounding_box of pointcloud, in order of x_max, y_max, z_max, x_min, y_min, z_min \in R^{2, 3}
    """
    
    max_val = 1024
    
    x_max, y_max, z_max = bounding_box[0, 0], bounding_box[0, 1], bounding_box[0, 2]
    x_min, y_min, z_min = bounding_box[1, 0], bounding_box[1, 1], bounding_box[1, 2]
    
    for i in range(pointcloud.shape[0]):
        x, y, z = pointcloud[i, 0], pointcloud[i, 1], pointcloud[i, 2]
        
        scale_x = (max_val - 1) / (x_max - x_min + 1e-7)
        scale_y = (max_val - 1) / (y_max - y_min + 1e-7)
        scale_z = (max_val - 1) / (z_max - z_min + 1e-7)

        x_idx = ti.cast(ti.floor((x - x_min) * scale_x), ti.u32)
        y_idx = ti.cast(ti.floor((y - y_min) * scale_y), ti.u32)
        z_idx = ti.cast(ti.floor((z - z_min) * scale_z), ti.u32)

        x_idx = ti.min(x_idx, max_val - 1) 
        y_idx = ti.min(y_idx, max_val - 1)
        z_idx = ti.min(z_idx, max_val - 1)

        code = (expand_bits_10(x_idx) << 2) | (expand_bits_10(y_idx) << 1) | expand_bits_10(z_idx)
        morton_codes[i] = code
    
    
# =============================================================================
# Grid Coordinate Morton Encoding (for CSR Grid)
# =============================================================================

@ti.func
def encode_grid_to_morton_ti(x: ti.i32, y: ti.i32, z: ti.i32) -> ti.u32:
    """Encode grid coordinates (0-1023) directly to morton code.

    Args:
        x, y, z: Grid coordinates, must be in range [0, 1023]

    Returns:
        30-bit morton code
    """
    xi = ti.cast(ti.max(0, ti.min(x, 1023)), ti.u32)
    yi = ti.cast(ti.max(0, ti.min(y, 1023)), ti.u32)
    zi = ti.cast(ti.max(0, ti.min(z, 1023)), ti.u32)

    return (expand_bits_10(xi) << 2) | (expand_bits_10(yi) << 1) | expand_bits_10(zi)


@ti.func
def decode_morton_to_grid_ti(code: ti.i64) -> ti.math.ivec3:
    """Decode morton code to grid coordinates.

    Args:
        code: 30-bit morton code (passed as i64 for compatibility)

    Returns:
        Grid coordinates (x, y, z)
    """
    code_u32 = ti.cast(code, ti.u32)
    x = compact_bits_10(code_u32 >> 2)
    y = compact_bits_10(code_u32 >> 1)
    z = compact_bits_10(code_u32)
    return ti.math.ivec3([ti.cast(x, ti.i32), ti.cast(y, ti.i32), ti.cast(z, ti.i32)])


@ti.kernel
def grid_to_morton_kernel(
    grid_coords: ti.types.ndarray(dtype=ti.i32, ndim=2),  # [N, 3]
    morton_codes: ti.types.ndarray(dtype=ti.u32, ndim=1),  # [N]
):
    """Batch convert grid coordinates to morton codes.

    Args:
        grid_coords: Grid coordinates [N, 3], each in range [0, 1023]
        morton_codes: Output morton codes [N]
    """
    for i in range(grid_coords.shape[0]):
        x = grid_coords[i, 0]
        y = grid_coords[i, 1]
        z = grid_coords[i, 2]
        morton_codes[i] = encode_grid_to_morton_ti(x, y, z)


@ti.kernel
def morton_to_grid_kernel(
    morton_codes: ti.types.ndarray(dtype=ti.i64, ndim=1),  # [N] - use i64 to match pairs_morton
    grid_coords: ti.types.ndarray(dtype=ti.i32, ndim=2),   # [N, 3]
):
    """Batch convert morton codes to grid coordinates.

    Args:
        morton_codes: Morton codes [N]
        grid_coords: Output grid coordinates [N, 3]
    """
    for i in range(morton_codes.shape[0]):
        coord = decode_morton_to_grid_ti(morton_codes[i])
        grid_coords[i, 0] = coord[0]
        grid_coords[i, 1] = coord[1]
        grid_coords[i, 2] = coord[2]


def grid_coords_to_morton(grid_coords: torch.Tensor) -> torch.Tensor:
    """Convert grid coordinates to morton codes (Python wrapper).

    Args:
        grid_coords: Grid coordinates [N, 3], values in [0, 1023]

    Returns:
        Morton codes [N], dtype=torch.uint32
    """
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
        raise ValueError(f"Expected [N, 3], got {grid_coords.shape}")

    device = grid_coords.device
    grid_coords = grid_coords.contiguous().to(torch.int32)
    morton_codes = torch.empty((grid_coords.shape[0],), dtype=torch.uint32, device=device)

    grid_to_morton_kernel(grid_coords, morton_codes)

    return morton_codes


def morton_to_grid_coords(morton_codes: torch.Tensor) -> torch.Tensor:
    """Convert morton codes to grid coordinates (Python wrapper).

    Args:
        morton_codes: Morton codes [N], dtype=torch.int64 (from pairs_morton)

    Returns:
        Grid coordinates [N, 3], dtype=torch.int32
    """
    if morton_codes.ndim != 1:
        raise ValueError(f"Expected [N], got {morton_codes.shape}")

    device = morton_codes.device
    morton_codes = morton_codes.contiguous().to(torch.int64)
    grid_coords = torch.empty((morton_codes.shape[0], 3), dtype=torch.int32, device=device)

    morton_to_grid_kernel(morton_codes, grid_coords)

    return grid_coords


def compute_morton_range(
    grid_min: torch.Tensor,  # [3]
    grid_max: torch.Tensor,  # [3]
) -> tuple[int, int]:
    """Compute the morton code range for a grid AABB.

    This computes the minimum and maximum morton codes that could
    be generated from coordinates within the given grid bounds.

    Args:
        grid_min: Minimum grid coordinates [3]
        grid_max: Maximum grid coordinates [3]

    Returns:
        (morton_min, morton_max) tuple

    Example:
        >>> grid_min = torch.tensor([0, 0, 0])
        >>> grid_max = torch.tensor([1, 1, 1])
        >>> morton_min, morton_max = compute_morton_range(grid_min, grid_max)
    """
    device = grid_min.device

    # Clamp to valid range
    grid_min = torch.clamp(grid_min, 0, 1023).to(torch.int32)
    grid_max = torch.clamp(grid_max, 0, 1023).to(torch.int32)

    # Compute corners
    corners = torch.zeros((8, 3), dtype=torch.int32, device=device)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                corners[idx, 0] = grid_min[0] if i == 0 else grid_max[0]
                corners[idx, 1] = grid_min[1] if j == 0 else grid_max[1]
                corners[idx, 2] = grid_min[2] if k == 0 else grid_max[2]
                idx += 1

    morton_codes = grid_coords_to_morton(corners)
    # Convert to int64 for min/max operations (uint32 not supported)
    morton_codes_int64 = morton_codes.to(torch.int64)
    morton_min = int(morton_codes_int64.min().item())
    morton_max = int(morton_codes_int64.max().item())

    return morton_min, morton_max


# =============================================================================
# Original Point Cloud Morton Encoding
# =============================================================================

class Morton3D(torch.nn.Module):
    def __init__(
        self,
    ):
        super(Morton3D, self).__init__()

    def forward(
        self,
        pointcloud: torch.Tensor,
        color: Optional[torch.Tensor]=None,
    ) -> Dict:
        """_summary_
            
        Args:
            pointcloud (torch.Tensor): pointcloud to calculate morton code for each point. \in R^(N, 3)
            color (Optional[torch.Tensor], optional): optional color information for each point. \in R^(N, 3). Defaults to None.
            device (Optional[torch.device], optional): device to run the computation on. Defaults to torch.device('cuda').

        Returns:
            Dict: sorted pointcloud，sorted color(optional), sorted morton codes
        """
        
        N = pointcloud.shape[0]
        
        device = pointcloud.device
        print(f"Morton3D running on device: {device}")
        
        bounding_box = torch.cat((
                torch.max(pointcloud, dim=0).values.view(1, 3),
                torch.min(pointcloud, dim=0).values.view(1, 3),
            ), dim=0 
        )
        
        morton_codes = torch.zeros((N,), dtype=torch.uint32, device=device)
        
        Morton3D_kernel(
            pointcloud=pointcloud,
            morton_codes=morton_codes,
            bounding_box=bounding_box,
        )
        
        sorted_codes_long, indices = torch.sort(morton_codes.long())
        
        sorted_codes = sorted_codes_long.to(torch.uint32)

        sorted_pointcloud = pointcloud[indices, :]
        if color is not None:
            sorted_color = color[indices, :]
        else:
            sorted_color = None
            
        return {
            'sorted_pointcloud': sorted_pointcloud,
            'sorted_color': sorted_color,
            'morton_codes': sorted_codes,
        }

