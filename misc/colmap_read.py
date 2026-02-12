
import numpy as np
import torch
from plyfile import PlyData
from pathlib import Path
from typing import Union

def read_colmap_points3d_ply(path: Union[str, Path], device: Union[str, torch.device] = "cpu"):
    """
    Reads a COLMAP points3D.ply file and returns point cloud and colors as Tensors.

    Args:
        path (Union[str, Path]): Path to the points3D.ply file.
        device (Union[str, torch.device]): Device to place the tensors on.

    Returns:
        dict:
            - pointcloud (torch.Tensor): Shape (N, 3) with float32 coordinates.
            - pointcloud_color (torch.Tensor): Shape (N, 3) with uint8 RGB values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    plydata = PlyData.read(str(path))
    
    # Check for vertex element
    if 'vertex' not in plydata:
        raise ValueError("PLY file does not contain 'vertex' element")
        
    vertex = plydata['vertex']

    # Extract positions (x, y, z)
    x = np.asarray(vertex['x'])
    y = np.asarray(vertex['y'])
    z = np.asarray(vertex['z'])
    pointcloud = np.stack([x, y, z], axis=1).astype(np.float32)
    pointcloud = torch.from_numpy(pointcloud).to(device)

    # Extract colors (red, green, blue)
    # Using getattr to handle missing properties gracefully logic inside try-except
    try:
        r = np.asarray(vertex['red'])
        g = np.asarray(vertex['green'])
        b = np.asarray(vertex['blue'])
        pointcloud_color = np.stack([r, g, b], axis=1).astype(np.uint8)
        pointcloud_color = torch.from_numpy(pointcloud_color).to(device)
    except (ValueError, KeyError):
        print(f"Warning: Color attributes not found in {path}. Using black color.")
        pointcloud_color = torch.zeros_like(pointcloud, dtype=torch.uint8, device=device)

    return {
        "pointcloud": pointcloud, 
        "pointcloud_color": pointcloud_color
    }

if __name__ == "__main__":
    import argparse
    

    path = Path('/home/wangjv_wsl/data/3dgs_dataset/hierachy/replica/office1/camera_calibration/aligned/sparse/0/points3D.ply')
    try:
        data = read_colmap_points3d_ply(path, device='cpu')
        points = data["pointcloud"]
        colors = data["pointcloud_color"]
        print(f"Successfully loaded {path}")
        print(f" - Point cloud shape: {points.shape}, Device: {points.device}, Type: {points.dtype}")
        print(f" - Color shape:       {colors.shape}, Device: {colors.device}, Type: {colors.dtype}")
    except Exception as e:
        print(f"Error: {e}")
