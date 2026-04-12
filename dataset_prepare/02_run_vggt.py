#!/usr/bin/env python
"""
Step 2: VGGT reconstruction

This script runs VGGT (Visual Geometry Grounded Transformer) to estimate camera poses
and dense point cloud without COLMAP. VGGT is a feed-forward method that works well
for scenes where COLMAP struggles.

Usage:
    python 02_run_vggt.py --scene_dir /path/to/scene --max_frames 50 --use_ba

Output structure:
    scene_dir/
    ├── vggt_result/
    │   ├── sparse/           # VGGT reconstruction (COLMAP format)
    │   │   ├── cameras.bin
    │   │   ├── images.bin
    │   │   ├── points3D.bin
    │   │   └── points.ply    # Point cloud visualization
    │   ├── intermediate/     # Intermediate VGGT features
    │   │   ├── tokens.pt     # Final backbone tokens
    │   │   ├── depth_maps.npy
    │   │   ├── point_maps.npy
    │   │   └── metadata.json
    │   └── depth_vis/        # Depth map visualizations
    │       ├── frame_0000_depth.jpg
    │       └── ...
    └── input/images/         # Input images (must exist)

Requirements:
    - VGGT submodule: submodule/vggt/
    - CUDA-capable GPU
    - ~10GB GPU memory
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add VGGT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "submodule" / "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_vggt_model(model, images, dtype, resolution=518):
    """
    Run VGGT model to estimate cameras and depth.

    Args:
        model: VGGT model instance
        images: Input images [B, 3, H, W]
        dtype: Data type for inference
        resolution: Resolution for VGGT (fixed at 518)

    Returns:
        extrinsic: Camera extrinsics [B, 4, 4]
        intrinsic: Camera intrinsics [B, 3, 3]
        depth_map: Predicted depth maps [B, H, W]
        depth_conf: Depth confidence maps [B, H, W]
        tokens: Final aggregated tokens from VGGT backbone
    """
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # Resize to VGGT fixed resolution
    images = F.interpolate(images, size=(resolution, resolution),
                          mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # Add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # Predict depth
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    tokens = aggregated_tokens_list[-1].cpu()  # Save final layer tokens

    return extrinsic, intrinsic, depth_map, depth_conf, tokens


def run_vggt_reconstruction(scene_dir: Path, use_ba: bool = False,
                            conf_threshold: float = 5.0,
                            vis_threshold: float = 0.2,
                            max_query_pts: int = 4096,
                            max_frames: int = None,
                            seed: int = 42):
    """
    Run VGGT reconstruction on a scene.

    Args:
        scene_dir: Scene directory containing input/images/
        use_ba: Whether to use bundle adjustment (slower but more accurate)
        conf_threshold: Confidence threshold for depth filtering (without BA)
        vis_threshold: Visibility threshold for tracks (with BA)
        max_query_pts: Maximum query points for tracking
        max_frames: Maximum number of frames to process (uniformly sampled)
        seed: Random seed

    Returns:
        sparse_dir: Path to COLMAP-format sparse reconstruction
    """
    # Setup paths
    image_dir = scene_dir / "input" / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    vggt_result_dir = scene_dir / "vggt_result" / str(seed)
    sparse_dir = vggt_result_dir / "sparse"
    vggt_result_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Get image paths
    image_path_list = sorted(glob.glob(str(image_dir / "*")))
    image_path_list = [p for p in image_path_list
                       if Path(p).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")

    n_total = len(image_path_list)

    # Set seed early so frame sampling is reproducible
    set_seed(seed)

    # Random consecutive frame sampling for memory control
    if max_frames is not None and n_total > max_frames:
        start_idx = random.randint(0, n_total - max_frames)
        indices = list(range(start_idx, start_idx + max_frames))
        image_path_list = [image_path_list[i] for i in indices]
        print(f"  !! Frame limit active: sampled {len(image_path_list)} consecutive frames "
              f"starting at index {start_idx} / {n_total}")

    base_image_path_list = [os.path.basename(p) for p in image_path_list]

    print(f"\n{'='*70}")
    print(f"VGGT Reconstruction")
    print(f"{'='*70}")
    print(f"Scene: {scene_dir}")
    print(f"Images: {len(image_path_list)}")
    print(f"Bundle Adjustment: {use_ba}")

    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")

    # Load VGGT model
    print("\n[1/3] Loading VGGT model...")
    model = VGGT()
    model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_url, progress=True))
    model.eval()
    model = model.to(device)
    print("  Model loaded successfully")

    # Load images
    print("\n[2/3] Loading images...")
    vggt_resolution = 518
    load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"  Loaded {len(images)} images at resolution {load_resolution}")

    # Run VGGT inference
    print("\n[3/3] Running VGGT inference...")
    extrinsic, intrinsic, depth_map, depth_conf, tokens = run_vggt_model(model, images, dtype, vggt_resolution)

    # Unproject depth to 3D points
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # Get point colors from images
    images_resized = F.interpolate(images, size=(vggt_resolution, vggt_resolution),
                                   mode="bilinear", align_corners=False)
    points_rgb = (images_resized.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    if use_ba:
        print("  Running with Bundle Adjustment...")
        reconstruction = run_vggt_with_ba(
            model, images, points_3d, depth_conf, extrinsic, intrinsic,
            original_coords, base_image_path_list, vggt_resolution, load_resolution,
            vis_threshold, max_query_pts, dtype
        )
        reconstruction_resolution = load_resolution
    else:
        print("  Running feed-forward (no BA)...")
        reconstruction = run_vggt_feedforward(
            points_3d, depth_conf, extrinsic, intrinsic, points_rgb,
            conf_threshold, max_points=100000
        )
        reconstruction_resolution = vggt_resolution

    # Rescale and rename
    reconstruction = rescale_and_rename_reconstruction(
        reconstruction, base_image_path_list, original_coords.cpu().numpy(),
        reconstruction_resolution, shift_point2d_to_original_res=True
    )

    # Save reconstruction
    print(f"\nSaving reconstruction to {sparse_dir}")
    reconstruction.write(str(sparse_dir))

    # Save point cloud for visualization
    try:
        import trimesh
        points_for_vis = points_3d.reshape(-1, 3)
        colors_for_vis = points_rgb.reshape(-1, 3)
        # Filter valid points
        valid_mask = ~(np.isnan(points_for_vis).any(axis=1) | np.isinf(points_for_vis).any(axis=1))
        trimesh.PointCloud(
            points_for_vis[valid_mask],
            colors=colors_for_vis[valid_mask]
        ).export(str(sparse_dir / "points.ply"))
        print(f"  Saved visualization point cloud: {sparse_dir / 'points.ply'}")
    except ImportError:
        print("  Note: trimesh not installed, skipping PLY export")

    # Save intermediate features (tokens, depth maps, point maps)
    intermediate_dir = vggt_result_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving intermediate features to {intermediate_dir}")

    torch.save(tokens, intermediate_dir / "tokens.pt")
    np.save(intermediate_dir / "depth_maps.npy", depth_map)
    np.save(intermediate_dir / "point_maps.npy", points_3d)

    import json
    metadata = {
        "frame_names": base_image_path_list,
        "num_frames": len(base_image_path_list),
        "tokens_shape": list(tokens.shape),
        "depth_maps_shape": list(depth_map.shape),
        "point_maps_shape": list(points_3d.shape),
        "vggt_resolution": vggt_resolution,
        "load_resolution": load_resolution,
    }
    with open(intermediate_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  tokens.pt        {tokens.shape}    (~{tokens.element_size() * tokens.nelement() / 1024**2:.1f} MB)")
    print(f"  depth_maps.npy   {depth_map.shape}")
    print(f"  point_maps.npy   {points_3d.shape}")
    print(f"  metadata.json")

    # Save depth visualizations as JPEG with colormap
    depth_vis_dir = vggt_result_dir / "depth_vis"
    depth_vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving depth visualizations to {depth_vis_dir}")

    import matplotlib.pyplot as plt
    cmap = plt.colormaps.get_cmap('gray')

    for i, frame_name in enumerate(base_image_path_list):
        d = depth_map[i, ..., 0] if depth_map.ndim == 4 else depth_map[i]
        d_min, d_max = d.min(), d.max()
        d_norm = (d - d_min) / (d_max - d_min + 1e-8)
        colored = cmap(d_norm)[:, :, :3]  # RGB
        colored_uint8 = (colored * 255).astype(np.uint8)
        out_name = Path(frame_name).stem + "_depth.jpg"
        Image.fromarray(colored_uint8).save(str(depth_vis_dir / out_name), quality=95)

    print(f"  Saved {len(base_image_path_list)} depth visualization JPEGs")

    # Count reconstructed cameras
    num_cameras = len(reconstruction.cameras)
    num_images = len(reconstruction.images)
    num_points = len(reconstruction.points3D)

    print(f"\n{'='*70}")
    print(f"VGGT Reconstruction Complete!")
    print(f"{'='*70}")
    print(f"Cameras: {num_cameras}")
    print(f"Images: {num_images}")
    print(f"3D Points: {num_points}")
    print(f"Output: {sparse_dir}")

    return sparse_dir


def run_vggt_with_ba(model, images, points_3d, depth_conf, extrinsic, intrinsic,
                     original_coords, image_paths, vggt_resolution, load_resolution,
                     vis_threshold, max_query_pts, dtype):
    """Run VGGT with bundle adjustment."""
    import pycolmap

    image_size = np.array(images.shape[-2:])
    scale = load_resolution / vggt_resolution

    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict tracks using VGGSfM tracker
        pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d,
            masks=None,
            max_query_pts=max_query_pts,
            query_frame_num=8,
            keypoint_extractor="aliked+sp",
            fine_tracking=True,
        )

    torch.cuda.empty_cache()

    # Rescale intrinsic to original resolution
    intrinsic[:, :2, :] *= scale
    track_mask = pred_vis_scores > vis_threshold

    # Convert to pycolmap format
    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        points_3d,
        extrinsic,
        intrinsic,
        pred_tracks,
        image_size,
        masks=track_mask,
        max_reproj_error=8.0,
        shared_camera=False,
        camera_type="PINHOLE",
        points_rgb=points_rgb,
    )

    if reconstruction is None:
        raise ValueError("No reconstruction can be built with BA")

    # Bundle adjustment
    print("  Running bundle adjustment...")
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    return reconstruction


def run_vggt_feedforward(points_3d, depth_conf, extrinsic, intrinsic, points_rgb,
                         conf_threshold, max_points=100000):
    """Run VGGT in feed-forward mode without BA."""
    num_frames, height, width, _ = points_3d.shape
    image_size = np.array([height, width])

    # Create pixel coordinate grid
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    # Filter by confidence
    conf_mask = depth_conf >= conf_threshold
    conf_mask = randomly_limit_trues(conf_mask, max_points)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print(f"  Points after filtering: {len(points_3d)}")

    # Convert to COLMAP format
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=False,
        camera_type="PINHOLE",
    )

    return reconstruction


def rescale_and_rename_reconstruction(reconstruction, image_paths, original_coords,
                                      img_size, shift_point2d_to_original_res=False):
    """Rescale camera parameters to original image resolution."""
    import copy

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        # Rescale camera parameters
        real_image_size = original_coords[pyimageid - 1, -2:]
        resize_ratio = max(real_image_size) / img_size
        pred_params = copy.deepcopy(pycamera.params)
        pred_params = pred_params * resize_ratio
        real_pp = real_image_size / 2
        pred_params[-2:] = real_pp

        pycamera.params = pred_params
        pycamera.width = int(real_image_size[0])
        pycamera.height = int(real_image_size[1])

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

    return reconstruction


def main():
    parser = argparse.ArgumentParser(description="Run VGGT reconstruction")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Scene directory containing input/images/")
    parser.add_argument("--use_ba", action="store_true",
                        help="Use bundle adjustment (slower but more accurate)")
    parser.add_argument("--conf_threshold", type=float, default=5.0,
                        help="Confidence threshold for depth filtering (without BA)")
    parser.add_argument("--vis_threshold", type=float, default=0.2,
                        help="Visibility threshold for tracks (with BA)")
    parser.add_argument("--max_query_pts", type=int, default=4096,
                        help="Maximum number of query points")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames to process (random consecutive window). Recommended: 50 for 8GB, 80 for 16GB, 100 for 24GB VRAM")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available. VGGT requires GPU.", file=sys.stderr)
        sys.exit(1)

    scene_dir = Path(args.scene_dir).resolve()

    try:
        run_vggt_reconstruction(
            scene_dir=scene_dir,
            use_ba=args.use_ba,
            conf_threshold=args.conf_threshold,
            vis_threshold=args.vis_threshold,
            max_query_pts=args.max_query_pts,
            max_frames=args.max_frames,
            seed=args.seed
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
