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

