# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **3D Gaussian Splatting (3DGS) Registration** project that aligns point clouds to Gaussian scenes. It implements GMM-based registration using a novel CSR Grid spatial indexing approach and supports both MLE-based and traditional ICP-based registration methods.

## Architecture

### Core Modules

- **`gmm_point_alignment/`** - Main registration codebase
  - `unified_registration.py` - Unified interface supporting both MLE and sampler methods
  - `gmm_point_alignment.py` - Main orchestrator integrating CSR grid, querying, and MLE registration
  - `transform_utils.py` - SE(3)/Sim(3) Lie algebra operations with differentiable exponential maps
  - `mle_registration/` - MLE-based registration using CSR Grid
    - `csr_grid_builder.py` - Builds compressed sparse row (CSR) grid for spatial indexing of Gaussian spheres
    - `csr_grid_querier.py` - Top-K nearest sphere queries using the CSR grid
    - `sphere_mle_loss.py` - GMM MLE loss computation and registration optimization
    - `morton_code.py` - 3D Morton coding for spatial hashing
  - `sampler_registration/` - Traditional ICP-based registration
    - `gaussian_sampler.py` - Point cloud sampling from Gaussian scenes
    - `registration_sampler.py` - ICP registration methods (SVD, Chamfer, Open3D)

- **`misc/`** - Utilities
  - `hier_IO.py` - Loads `.hier` files (custom 3DGS format) into `GaussianScenes` dataclass
  - `geometry.py` - Geometric utilities

- **`taichi_3d_gaussian_splatting/`** - Third-party 3DGS training/rendering module (unmodified)

### Data Structures

- `GaussianScenes` - Dataclass holding 3D Gaussian parameters (position, rotation, scales, opacities, SHs)
- `CSRGridData` - Spatial index structure for fast sphere queries
- `UnifiedConfig` - Configuration for switching between MLE and sampler registration methods

## Common Commands

### Setup

```bash
# Install dependencies
pip install -r taichi_3d_gaussian_splatting/requirements.txt

# Install the taichi_3d_gaussian_splatting package
pip install -e taichi_3d_gaussian_splatting/
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/unit/test_csr_grid_builder.py -v
python -m pytest tests/unit/test_morton_code.py -v
python -m pytest tests/integration/test_registration.py -v

# Run the unified registration test with real data
python tests/scripts/test_unified_registration.py

# Run MLE debug test
python tests/scripts/test_mle_debug.py
```

### Running Registration

```bash
# Unified registration test (compares MLE vs Sampler methods)
python tests/scripts/test_unified_registration.py

# Visualization of registration results
python visualize_registration.py

# Parameter sweep experiments
python experiments/run_param_sweep.py
```

### Using the Registration API

```python
import taichi as ti
ti.init(arch=ti.cuda)

from misc.hier_IO import load_hier_to_torch
from gmm_point_alignment.unified_registration import (
    UnifiedRegistration, UnifiedConfig, RegistrationMethod
)

# Load data
scene = load_hier_to_torch("data/merged.hier").gaussian_scene
pointcloud = torch.randn(1000, 3).cuda()  # Your point cloud

# MLE registration
config = UnifiedConfig(method=RegistrationMethod.MLE)
reg = UnifiedRegistration(config)
result = reg.register(scene, pointcloud)

# Access results
print(result.transform)  # 4x4 transformation matrix
print(result.R)          # 3x3 rotation matrix
print(result.t)          # 3 translation vector
```

## Key Technical Details

### CSR Grid Spatial Indexing

The MLE registration uses a custom CSR (Compressed Sparse Row) grid for efficient spatial indexing:

1. **Grid Construction** (`csr_grid_builder.py`): Voxelizes space based on Gaussian sphere radii, assigns each sphere to voxels it intersects
2. **Morton Coding** (`morton_code.py`): 3D spatial hashing for sorting voxels
3. **Querying** (`csr_grid_querier.py`): For each query point, finds candidate spheres in neighboring voxels and returns Top-K nearest

### Registration Methods

**MLE Method (`RegistrationMethod.MLE`)**:
- Directly optimizes alignment between point cloud and Gaussian spheres
- Uses GMM negative log-likelihood as loss function
- Supports SE(3) (rigid) and Sim(3) (similarity) transformations
- Parameters: `mle_lr_translation`, `mle_lr_rotation`, `mle_num_iters`, `mle_use_scale`

**Sampler Method (`RegistrationMethod.SAMPLER`)**:
- Samples point cloud from Gaussian scene
- Applies traditional ICP (Iterative Closest Point)
- Methods: SVD-ICP, Chamfer distance optimization, Open3D ICP
- Parameters: `sampler_method`, `sampler_num_points`

### Coordinate Systems

- All positions/scales in the same coordinate system as the input `.hier` file
- Transformations follow standard SE(3) convention: `p' = R @ p + t`
- Uses Lie algebra parameterization for optimization (`se3_exp`, `sim3_exp` in `transform_utils.py`)

## Test Data

Real test data is located in `data/`:
- `merged.hier` - Gaussian scene (~100MB, compressed 3DGS format)
- `points3D.ply` - Point cloud to register (PLY format)

Load with:
```python
from misc.hier_IO import load_hier_to_torch
hier_scene = load_hier_to_torch("data/merged.hier", device="cuda")
scene = hier_scene.gaussian_scene  # GaussianScenes object
```

## Performance Notes

- Taichi kernel compilation happens on first run (can take 10-30 seconds)
- Grid building is O(N) but takes significant time for large scenes
- Query performance depends on voxel size strategy (see `VoxelSizeStrategy`)
- MLE method is generally more accurate than sampler-based methods
