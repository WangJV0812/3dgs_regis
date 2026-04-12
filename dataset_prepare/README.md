# HIER-3DGS-VGGT Dataset Preparation

This directory contains scripts for preparing datasets for VGGT-to-3DGS registration experiments.

## Overview

The data preparation pipeline consists of three steps:

1. **COLMAP SfM** (`01_run_colmap.py`) - Traditional SfM reconstruction
2. **VGGT Reconstruction** (`02_run_vggt.py`) - Feed-forward pose/point estimation
3. **HIER 3DGS Training** (`03_train_hier.py`) - Train 3D Gaussian Splatting model

## Dataset Structure

```
scene_dir/
├── input/
│   └── images/              # Input RGB images (required)
├── colmap_result/           # Step 1 output
│   ├── sparse/
│   │   ├── cameras.bin      # Camera intrinsics
│   │   ├── images.bin       # Camera poses
│   │   ├── points3D.bin     # Sparse 3D points
│   │   └── points.ply       # Visualization point cloud
│   └── database.db          # Feature database
├── vggt_result/             # Step 2 output
│   ├── sparse/
│   │   ├── cameras.bin      # VGGT camera intrinsics
│   │   ├── images.bin       # VGGT camera poses
│   │   ├── points3D.bin     # VGGT dense points
│   │   └── points.ply       # Visualization point cloud
│   └── (depth_maps/)        # Optional: predicted depth
├── 3dgs_result/             # Step 3 output
│   ├── model/
│   │   ├── final.hier       # Final trained 3DGS model
│   │   └── ckpt/            # Training checkpoints
│   ├── render/              # Rendered validation imagz   └── log/                 # Training logs
└── config.json              # Dataset configuration
```

## Requirements

### Software

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- COLMAP (for Step 1)

### Python Packages

```bash
# Install VGGT dependencies
cd ../submodule/vggt
pip install -r requirements.txt

# Install HIER 3DGS dependencies
cd ../hier_3dgs
pip install -r requirements.txt
```

### Hardware

- **Step 1 (COLMAP)**: CPU-only or GPU optional
- **Step 2 (VGGT)**: NVIDIA GPU with 10GB+ VRAM
- **Step 3 (HIER 3DGS)**: NVIDIA GPU with 24GB+ VRAM

## Usage

### Step 0: Prepare Input Images

Place your images in the scene directory:

```bash
mkdir -p /path/to/scene/input/images
cp /path/to/your/images/*.jpg /path/to/scene/input/images/
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

### Step 1: COLMAP SfM

Run COLMAP structure-from-motion reconstruction:

```bash
python 01_run_colmap.py \
    --scene_dir /path/to/scene \
    --camera_type PINHOLE \
    --matcher exhaustive
```

Options:
- `--camera_type`: Camera model (`PINHOLE`, `OPENCV`, `SIMPLE_PINHOLE`, etc.)
- `--matcher`: Feature matcher (`exhaustive` or `sequential`)
- `--vocab_tree`: Path to vocab tree for sequential matcher

### Step 2: VGGT Reconstruction

Run VGGT feed-forward reconstruction:

```bash
# Fast mode (feed-forward only)
python 02_run_vggt.py \
    --scene_dir /path/to/scene \
    --conf_threshold 5.0

# With Bundle Adjustment (more accurate, slower)
python 02_run_vggt.py \
    --scene_dir /path/to/scene \
    --use_ba \
    --vis_threshold 0.2 \
    --max_query_pts 4096
```

Options:
- `--use_ba`: Enable bundle adjustment (slower but more accurate)
- `--conf_threshold`: Depth confidence threshold (0-10, higher = more points)
- `--vis_threshold`: Track visibility threshold for BA (0-1)
- `--max_query_pts`: Maximum query points for tracking

**Note**: VGGT outputs are scale-ambiguous (no absolute scale), which is important for registration experiments.

### Step 3: Train HIER 3DGS

Train 3D Gaussian Splatting from COLMAP or VGGT reconstruction:

```bash
# Train from COLMAP
python 03_train_hier.py \
    --scene_dir /path/to/scene \
    --source colmap \
    --iterations 30000

# Train from VGGT
python 03_train_hier.py \
    --scene_dir /path/to/scene \
    --source vggt \
    --iterations 30000 \
    --use_depth
```

Options:
- `--source`: Source reconstruction (`colmap` or `vggt`)
- `--iterations`: Training iterations (default: 30000)
- `--save_iterations`: Additional save points (default: [7000, 15000])
- `--use_depth`: Enable depth supervision (if depth maps available)
- `--depth_weight_init`: Initial depth loss weight (default: 0.1)
- `--depth_weight_final`: Final depth loss weight (default: 0.01)

## Complete Example

```bash
# Setup scene directory
SCENE_DIR="/data/scenes/office0"
mkdir -p $SCENE_DIR/input/images
cp /data/images/office0/*.jpg $SCENE_DIR/input/images/

# Step 1: COLMAP
cd /path/to/3dgs_regis
python dataset_prepare/01_run_colmap.py --scene_dir $SCENE_DIR

# Step 2: VGGT
conda activate vggt  # If using separate environment
python dataset_prepare/02_run_vggt.py --scene_dir $SCENE_DIR --use_ba

# Step 3: Train 3DGS from COLMAP
conda activate gs_reg  # Back to main environment
python dataset_prepare/03_train_hier.py \
    --scene_dir $SCENE_DIR \
    --source colmap \
    --iterations 30000

# Also train from VGGT for comparison
python dataset_prepare/03_train_hier.py \
    --scene_dir $SCENE_DIR \
    --source vggt \
    --iterations 30000
```

## Registration Experiments

After dataset preparation, you can run registration experiments:

```python
from misc.hier_IO import load_hier_to_torch
from gmm_point_alignment.unified_registration import (
    UnifiedRegistration, UnifiedConfig, RegistrationMethod
)

# Load 3DGS scene trained from COLMAP (with scale)
scene_colmap = load_hier_to_torch(
    f"{SCENE_DIR}/3dgs_result/model/final.hier"
).gaussian_scene

# Load VGGT point cloud (without scale)
import open3d as o3d
pcd_vggt = o3d.io.read_point_cloud(
    f"{SCENE_DIR}/vggt_result/sparse/points.ply"
)
points_vggt = np.asarray(pcd_vggt.points)

# Register VGGT to COLMAP-trained 3DGS (estimate Sim(3) transform)
config = UnifiedConfig(
    method=RegistrationMethod.MLE,
    mle_use_scale=True,           # Important: VGGT has unknown scale
    mle_robust_kernel="huber",    # Handle outliers
    mle_num_iters=100,
)
reg = UnifiedRegistration(config)
result = reg.register(scene_colmap, points_vggt)

print(f"Estimated scale: {result.scale:.3f}")
print(f"Transform:\n{result.transform}")
```

## Troubleshooting

### COLMAP fails to reconstruct

- Try different `--camera_type` (e.g., `OPENCV` for distorted images)
- Use `--matcher sequential` for video sequences
- Check image quality (blur, exposure, texture)

### VGGT runs out of memory

- Reduce `--max_query_pts` (default: 4096)
- Use feed-forward mode without `--use_ba`
- Close other GPU processes

### HIER 3DGS training fails

- Check COLMAP/VGGT reconstruction exists
- Verify images symlink: `ls -la {colmap,vggt}_result/images`
- Ensure sufficient GPU memory

### Registration fails

- Verify 3DGS model: `python -c "from misc.hier_IO import load_hier_to_torch; load_hier_to_torch('path/to/model.hier')"`
- Check point cloud has valid points
- Try increasing `--mle_num_iters`

## Data Format Specifications

### COLMAP Format

Standard COLMAP sparse reconstruction format:
- `cameras.bin`: Camera intrinsics (focal length, principal point, distortion)
- `images.bin`: Image metadata and camera poses (quaternion + translation)
- `points3D.bin`: 3D points with visibility information

### VGGT Format

Compatible with COLMAP format but:
- Points are dense (from depth unprojection)
- Scale is arbitrary (unknown metric scale)
- Cameras may have different intrinsics per image

### HIER Format

Custom compressed 3D Gaussian format:
- `.hier`: Binary format with position, rotation, scale, opacity, SH coefficients
- Viewable with custom renderer
- Supports hierarchical level-of-detail

## Citation

If you use this dataset preparation pipeline, please cite:

```bibtex
@article{vggt2025,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={...},
  year={2025}
}

@article{hier3dgs2024,
  title={Hierarchical 3D Gaussian Splatting},
  author={...},
  year={2024}
}
```
