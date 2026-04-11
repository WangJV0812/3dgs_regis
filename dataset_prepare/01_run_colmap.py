#!/usr/bin/env python
"""
Step 1: COLMAP SfM reconstruction

This script runs COLMAP structure-from-motion to reconstruct camera poses and sparse point cloud.
Output is saved in standard COLMAP format (cameras.bin, images.bin, points3D.bin).

Usage:
    python 01_run_colmap.py --scene_dir /path/to/scene --camera_type PINHOLE

Output structure:
    scene_dir/
    ├── colmap_result/
    │   ├── sparse/           # COLMAP sparse reconstruction
    │   │   ├── cameras.bin
    │   │   ├── images.bin
    │   │   └── points3D.bin
    │   └── database.db       # COLMAP feature database
    └── input/images/         # Input images (must exist)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run shell command and print output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")
    return result


def check_colmap_installed():
    """Check if COLMAP is installed."""
    try:
        result = subprocess.run(["colmap", "--help"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ COLMAP is installed")
            return True
    except FileNotFoundError:
        pass
    print("✗ COLMAP not found. Please install COLMAP first.")
    print("  Installation: https://colmap.github.io/install.html")
    return False


def run_colmap_sfm(scene_dir: Path, camera_type: str = "PINHOLE",
                   matcher_type: str = "exhaustive", vocab_tree_path: str = None):
    """
    Run full COLMAP SfM pipeline.

    Args:
        scene_dir: Scene directory containing input/images/
        camera_type: Camera model type (PINHOLE, OPENCV, SIMPLE_PINHOLE, etc.)
        matcher_type: Feature matcher type (exhaustive or sequential)
        vocab_tree_path: Path to vocab tree for sequential matcher (optional)
    """
    input_dir = scene_dir / "input" / "images"
    if not input_dir.exists():
        raise FileNotFoundError(f"Input images not found: {input_dir}")

    # Setup output directories
    colmap_result_dir = scene_dir / "colmap_result"
    sparse_dir = colmap_result_dir / "sparse"
    database_path = colmap_result_dir / "database.db"

    colmap_result_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Get image paths
    image_paths = list(input_dir.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {input_dir}")

    print(f"\n{'='*70}")
    print(f"COLMAP SfM Reconstruction")
    print(f"{'='*70}")
    print(f"Scene: {scene_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Camera type: {camera_type}")
    print(f"Matcher: {matcher_type}")

    # Step 1: Feature extraction
    print(f"\n[1/6] Feature extraction...")
    run_command(
        f"colmap feature_extractor "
        f"--database_path {database_path} "
        f"--image_path {input_dir} "
        f"--ImageReader.camera_model {camera_type} "
        f"--ImageReader.single_camera 1"
    )

    # Step 2: Feature matching
    print(f"\n[2/6] Feature matching ({matcher_type})...")
    if matcher_type == "exhaustive":
        run_command(
            f"colmap exhaustive_matcher "
            f"--database_path {database_path}"
        )
    elif matcher_type == "sequential":
        match_cmd = (f"colmap sequential_matcher "
                    f"--database_path {database_path}")
        if vocab_tree_path and Path(vocab_tree_path).exists():
            match_cmd += f" --SequentialMatching.vocab_tree_path {vocab_tree_path}"
        run_command(match_cmd)
    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")

    # Step 3: Mapper (SfM)
    print(f"\n[3/6] Running mapper...")
    run_command(
        f"colmap mapper "
        f"--database_path {database_path} "
        f"--image_path {input_dir} "
        f"--output_path {sparse_dir}"
    )

    # Find the largest reconstruction (usually named "0")
    recon_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
    if len(recon_dirs) == 0:
        raise RuntimeError("COLMAP reconstruction failed - no sparse model generated")

    # Use the first (and usually only) reconstruction
    recon_dir = recon_dirs[0]
    print(f"  Using reconstruction: {recon_dir.name}")

    # Step 4: Bundle adjustment
    print(f"\n[4/6] Bundle adjustment...")
    run_command(
        f"colmap bundle_adjuster "
        f"--input_path {recon_dir} "
        f"--output_path {recon_dir}"
    )

    # Step 5: Convert to PLY for visualization
    print(f"\n[5/6] Exporting PLY...")
    ply_path = sparse_dir / "points.ply"
    run_command(
        f"colmap model_converter "
        f"--input_path {recon_dir} "
        f"--output_path {ply_path} "
        f"--output_type PLY"
    )

    # Step 6: Copy files to standard location
    print(f"\n[6/6] Organizing output...")
    for fname in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = recon_dir / fname
        dst = sparse_dir / fname
        if src.exists():
            # Move to parent sparse directory
            import shutil
            shutil.copy2(src, dst)

    # Count reconstructed cameras and points
    result = subprocess.run(
        f"colmap model_analyzer --path {recon_dir}",
        shell=True, capture_output=True, text=True
    )
    print(f"\n{result.stdout}")

    print(f"\n{'='*70}")
    print(f"COLMAP SfM Complete!")
    print(f"{'='*70}")
    print(f"Output: {sparse_dir}")
    print(f"  - cameras.bin: Camera intrinsics")
    print(f"  - images.bin: Camera poses (extrinsics)")
    print(f"  - points3D.bin: Sparse 3D points")
    print(f"  - points.ply: Visualization point cloud")

    return sparse_dir


def main():
    parser = argparse.ArgumentParser(description="Run COLMAP SfM reconstruction")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Scene directory containing input/images/")
    parser.add_argument("--camera_type", type=str, default="PINHOLE",
                        choices=["PINHOLE", "OPENCV", "SIMPLE_PINHOLE", "SIMPLE_RADIAL"],
                        help="Camera model type")
    parser.add_argument("--matcher", type=str, default="exhaustive",
                        choices=["exhaustive", "sequential"],
                        help="Feature matcher type")
    parser.add_argument("--vocab_tree", type=str, default=None,
                        help="Path to vocab tree for sequential matcher")

    args = parser.parse_args()

    # Check COLMAP is installed
    if not check_colmap_installed():
        sys.exit(1)

    scene_dir = Path(args.scene_dir).resolve()

    try:
        run_colmap_sfm(
            scene_dir=scene_dir,
            camera_type=args.camera_type,
            matcher_type=args.matcher,
            vocab_tree_path=args.vocab_tree
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
