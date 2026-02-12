import torch
import numpy as np
import pandas as pd
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Dict


@dataclass
class GaussianScenes:
    position:   torch.Tensor    # (N, 3)
    rotation:   torch.Tensor    # (N, 4)
    scales:     torch.Tensor    # (N, 3)
    opacities:  torch.Tensor    # (N,)
    shs:        torch.Tensor    # (N, 3, 16)
    
@dataclass
class HierarchyNodes:
    depth:          torch.Tensor    # (M,)
    parent:         torch.Tensor    # (M,)
    start:          torch.Tensor    # (M,)
    count_leafs:    torch.Tensor    # (M,)
    count_merged:   torch.Tensor    # (M,)
    start_children: torch.Tensor    # (M,)
    count_children: torch.Tensor    # (M,)
    
@dataclass
class Boxes:
    mins:   torch.Tensor    # (M, 4)
    maxs:   torch.Tensor    # (M, 4)

@dataclass
class HierarchyGaussianScene:
    gaussian_scene: GaussianScenes
    nodes:          HierarchyNodes
    boxes:          Boxes



def read_int32(f):
    return struct.unpack('<i', f.read(4))[0]


def read_int16(f):
    return struct.unpack('<h', f.read(2))[0]


def read_array(f, dtype, count):
    return np.fromfile(f, dtype=dtype, count=count)


def read_gaussian_scene(f, gaussian_num, is_compressed):
    gaussian_scene = {}
    
    # read gaussian parameters
    gaussian_scene['position'] = read_array(f, np.float32, gaussian_num * 3).reshape(gaussian_num, 3)
    
    if is_compressed:
        gaussian_scene['rotation']   = read_array(f, np.float16, gaussian_num * 4).reshape(gaussian_num, 4)
        gaussian_scene['scales']     = read_array(f, np.float16, gaussian_num * 3).reshape(gaussian_num, 3)
        gaussian_scene['opacities']  = read_array(f, np.float16, gaussian_num)
        gaussian_scene['shs']        = read_array(f, np.float16, gaussian_num * 48).reshape(gaussian_num, 48)
    else:
        gaussian_scene['rotation']   = read_array(f, np.float32, gaussian_num * 4).reshape(gaussian_num, 4)
        gaussian_scene['scales']     = read_array(f, np.float32, gaussian_num * 3).reshape(gaussian_num, 3)
        gaussian_scene['opacities']  = read_array(f, np.float32, gaussian_num)
        gaussian_scene['shs']        = read_array(f, np.float32, gaussian_num * 48).reshape(gaussian_num, 3, 16)

    return gaussian_scene


def read_hierarchy_nodes(f, node_num, is_compressed):
    if is_compressed:
        node_dtype = np.dtype([
            ('parent', np.int32),
            ('start', np.int32),
            ('start_children', np.int32),
            ('depth', np.int16),
            ('count_children', np.int16),
            ('count_leafs', np.int16),
            ('count_merged', np.int16),
        ])
    else:
        node_dtype = np.dtype([
            ('depth', np.int32),
            ('parent', np.int32),
            ('start', np.int32),
            ('count_leafs', np.int32),
            ('count_merged', np.int32),
            ('start_children', np.int32),
            ('count_children', np.int32),
        ])                	
        
    node_data = read_array(f, dtype=node_dtype, count=node_num)
    
    return node_data


def read_boxes(f, node_num, is_compressed):
    if is_compressed:
        boxes = read_array(f, np.float16, node_num * 8).reshape(node_num, 2, 4)
    else:
        boxes = read_array(f, np.float32, node_num * 8).reshape(node_num, 2, 4)

    return boxes


def load_hier_to_numpy(
    hier_path: Path,
) -> Dict:
    """Load hierarchy data from a .hier file into numpy arrays.

    Args:
        hier_path (Path): Path to the .hier file to be loaded.

    Raises:
        FileNotFoundError: If the specified .hier file does not exist.
        ValueError: If the file extension is not .hier.
    Returns:
        Dict: A dictionary containing the Gaussian scene, hierarchy nodes, and boxes as numpy arrays.
        contains keys: "gaussian_scene", "hier_nodes", "boxes"
    """
    if hier_path.exists() is False:
        raise FileNotFoundError(f"Hierarchy file not found: {hier_path.absolute()}")
    if hier_path.suffix != ".hier":
        raise ValueError(f"Invalid hierarchy file format: {hier_path.suffix}")
    
    with hier_path.open("rb") as f:
        gaussian_num = read_int32(f)
        
        is_compressed = True if gaussian_num < 0 else False

        # read gaussian scene from .hier file
        gaussian_num = abs(gaussian_num)
        print(f"Loading hierarchy file: {hier_path.name}, number of gaussians: {gaussian_num}, compressed: {is_compressed}")
        gaussian_scene = read_gaussian_scene(f, gaussian_num, is_compressed)
        
        # read gaussian scene from .hier file
        node_num = read_int32(f)
        print(f'Number of hierarchy nodes: {node_num}')
        hier_nodes = read_hierarchy_nodes(f, node_num, is_compressed)

        # read boxes from .hier file
        boxes = read_boxes(f, node_num, is_compressed)
        
    return {
        "gaussian_scene": gaussian_scene,
        "hier_nodes": hier_nodes,
        "boxes": boxes,
    }


def load_hier_to_torch(
    hier_path: Path,
    device: torch.device = torch.device("cpu"),
) -> HierarchyGaussianScene:
    """load hierarchy 3dgs dataset to torch tensors

    Args:
        hier_path (Path): Path to the .hier file to be loaded.
        device (torch.device, optional): Device on which to load the tensors. Defaults to torch.device("cpu").

    Returns:
        HierarchyGaussianScene: The hierarchy Gaussian scene data loaded into torch tensors.
    """
    hier_data_np = load_hier_to_numpy(hier_path)
    
    gaussian_scene_np = hier_data_np["gaussian_scene"]
    hier_nodes_np     = hier_data_np["hier_nodes"]
    boxes_np          = hier_data_np["boxes"]
    
    gaussian_scene = GaussianScenes(
        position   = torch.from_numpy(gaussian_scene_np['position']).to(device, dtype=torch.float32),
        rotation   = torch.from_numpy(gaussian_scene_np['rotation']).to(device, dtype=torch.float32),
        scales     = torch.from_numpy(gaussian_scene_np['scales']).to(device, dtype=torch.float32),
        opacities  = torch.from_numpy(gaussian_scene_np['opacities']).to(device, dtype=torch.float32),
        shs        = torch.from_numpy(gaussian_scene_np['shs']).to(device, dtype=torch.float32),
    )
    
    hier_nodes = HierarchyNodes(
        depth          = torch.from_numpy(hier_nodes_np['depth']).to(device),
        parent         = torch.from_numpy(hier_nodes_np['parent']).to(device),
        start          = torch.from_numpy(hier_nodes_np['start']).to(device),
        count_leafs    = torch.from_numpy(hier_nodes_np['count_leafs']).to(device),
        count_merged   = torch.from_numpy(hier_nodes_np['count_merged']).to(device),
        start_children = torch.from_numpy(hier_nodes_np['start_children']).to(device),
        count_children = torch.from_numpy(hier_nodes_np['count_children']).to(device),
    )
    
    boxes = Boxes(
        mins = torch.from_numpy(boxes_np[:,0,:]).to(device),
        maxs = torch.from_numpy(boxes_np[:,1,:]).to(device),
    )
    
    hier_gaussian_scene = HierarchyGaussianScene(
        gaussian_scene = gaussian_scene,
        nodes     = hier_nodes,
        boxes     = boxes,
    )
    
    return hier_gaussian_scene


def write_hier_to_parquet(
    hier_gaussian_scene: HierarchyGaussianScene,
    save_path: Path,
):
    if save_path.parent.exists() is False:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.suffix != ".parquet":
        raise ValueError(f"Invalid save file format: {save_path.suffix}")
    
    # prepare data to pandas dataframe
    data_dict = {}
    # gaussian scene
    data_dict['gaussian_position']  = hier_gaussian_scene.gaussian_scene.position.cpu().numpy()
    data_dict['gaussian_rotation']  = hier_gaussian_scene.gaussian_scene.rotation.cpu().numpy()
    data_dict['gaussian_scales']    = hier_gaussian_scene.gaussian_scene.scales.cpu().numpy()
    data_dict['gaussian_opacities'] = hier_gaussian_scene.gaussian_scene.opacities.cpu().numpy()
    data_dict['gaussian_shs']       = hier_gaussian_scene.gaussian_scene.shs.cpu().numpy()
    # hierarchy nodes
    data_dict['node_depth']          = hier_gaussian_scene.nodes.depth.cpu().numpy()
    data_dict['node_parent']         = hier_gaussian_scene.nodes.parent.cpu().numpy()
    data_dict['node_start']          = hier_gaussian_scene.nodes.start.cpu().numpy()
    data_dict['node_count_leafs']    = hier_gaussian_scene.nodes.count_leafs.cpu().numpy()
    data_dict['node_count_merged']   = hier_gaussian_scene.nodes.count_merged.cpu().numpy()
    data_dict['node_start_children'] = hier_gaussian_scene.nodes.start_children.cpu().numpy()
    data_dict['node_count_children'] = hier_gaussian_scene.nodes.count_children.cpu().numpy()
    # boxes
    data_dict['box_mins'] = hier_gaussian_scene.boxes.mins.cpu().numpy()
    data_dict['box_maxs'] = hier_gaussian_scene.boxes.maxs.cpu().numpy()
    
    df = pd.DataFrame([data_dict])
    df.to_parquet(save_path, index=False)




if __name__ == "__main__":
    hier_path = Path("data/hierarchy.hier")
    hier_scene = load_hier_to_torch(hier_path, device=torch.device("cpu"))
    # print(f'gaussian scene is {hier_scene.gaussian_scene}')
    # print(f'hierarchy nodes is {hier_scene.nodes}')
    # print(f'boxes is {hier_scene.boxes}')
