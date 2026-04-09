# MLE-based Point Cloud Registration Theory

## Overview

MLE (Maximum Likelihood Estimation) registration treats point cloud alignment as a probabilistic inference problem. Instead of finding nearest neighbors (ICP), we maximize the likelihood of observed points under the Gaussian mixture model represented by the 3DGS scene.

## Mathematical Foundation

### 1. Gaussian Scene Representation

Each 3D Gaussian in the scene is defined by:
- **Center**: $\mu_i \in \mathbb{R}^3$
- **Covariance**: $\Sigma_i = R_i \cdot S_i^2 \cdot R_i^T$
  - $R_i$: rotation matrix from quaternion
  - $S_i = \text{diag}(s_i)$: scale matrix (diagonal)
- **Opacity**: $\alpha_i \in [0, 1]$

The probability density of point $x$ under Gaussian $i$:
$$\mathcal{N}(x | \mu_i, \Sigma_i) = \frac{1}{(2\pi)^{3/2} |\Sigma_i|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu_i)^T \Sigma_i^{-1} (x-\mu_i)\right)$$

### 2. GMM Likelihood

For a point cloud $P = \{p_1, p_2, ..., p_N\}$, the likelihood under the Gaussian scene is:

$$L(T; P, G) = \prod_{j=1}^N \sum_{i=1}^M w_i \cdot \mathcal{N}(T(p_j) | \mu_i, \Sigma_i)$$

Where:
- $T: \mathbb{R}^3 \to \mathbb{R}^3$ is the transformation (SE(3) or Sim(3))
- $w_i = \frac{\alpha_i}{\sum_k \alpha_k}$ is the mixture weight
- $M$ is the number of Gaussians in the scene

### 3. Negative Log-Likelihood Loss

We minimize the negative log-likelihood:

$$\mathcal{L}(T) = -\sum_{j=1}^N \log \left( \sum_{i=1}^M w_i \cdot \mathcal{N}(T(p_j) | \mu_i, \Sigma_i) \right)$$

This is equivalent to minimizing the cross-entropy between:
- The empirical distribution of transformed points
- The GMM represented by the scene

### 4. Top-K Approximation

Computing all $M$ Gaussians for each point is expensive ($O(N \cdot M)$). We use **Top-K approximation**:

$$\mathcal{L}(T) \approx -\sum_{j=1}^N \log \left( \sum_{i \in \mathcal{K}(T(p_j))} w_i \cdot \mathcal{N}(T(p_j) | \mu_i, \Sigma_i) \right)$$

Where $\mathcal{K}(p)$ returns the K nearest Gaussians to point $p$.

**Why this works:**
- Gaussians are spatially localized
- Far-away Gaussians contribute negligibly to the likelihood
- Typically $K=8$ captures >99% of the probability mass

## Data Processing Pipeline

### Phase 1: CSR Grid Construction (One-time)

```
Input: GaussianScene (M Gaussians)
       ├── positions: (M, 3)
       ├── scales: (M, 3) [log scale]
       ├── rotations: (M, 4) [quaternions]
       └── opacities: (M,)

Step 1: Compute AABB for each Gaussian
        - Convert log scales to real scales: $s = \exp(\hat{s})$
        - Compute covariance: $\Sigma = R \cdot S^2 \cdot R^T$
        - Compute AABB with confidence interval (default: 97%, $\chi^2_3 = 7.81$)

Step 2: Build Voxel Grid
        - Compute global scene AABB
        - Determine voxel size (adaptive strategy)
        - Create uniform 3D grid

Step 3: Enumerate Sphere-Voxel Pairs
        - For each Gaussian, find overlapping voxels
        - Store (voxel_id, sphere_id) pairs

Step 4: Build CSR Structure
        - Sort pairs by voxel_id
        - Build row pointers (CSR format)
        - Build L1/L2 lookup table for O(1) queries

Step 5: Precompute Covariance Data
        - $\Sigma^{-1}$ (inverse covariance)
        - $\log |\Sigma|^{1/2}$ (normalization factor)

Output: CSRGridData
        ├── voxel_size: float
        ├── grid_dims: (3,)
        ├── voxel_to_spheres: CSR format indices
        ├── sphere_centers: (M, 3)
        ├── cov_inv: (M, 3, 3)
        └── norm_factor: (M,)
```

### Phase 2: Top-K Query (Per Iteration)

```
Input: PointCloud P (N points)
       Transformation T (current estimate)

Step 1: Transform Points
        P' = T(P)  [Apply current SE(3)/Sim(3)]

Step 2: Voxel Hashing
        For each point p':
        - Compute voxel coordinates: $v = \lfloor (p' - \text{origin}) / \text{voxel_size} \rfloor$
        - Hash to linear index

Step 3: Retrieve Candidates
        For each voxel v:
        - Lookup L1 table → L2 range
        - Get sphere IDs from CSR structure

Step 4: Compute Distances
        For each point p' and its candidate Gaussians:
        - Compute Mahalanobis distance: $d^2 = (p' - \mu_i)^T \Sigma_i^{-1} (p' - \mu_i)$
        - Score = opacity × exp(-0.5 × d²) / normalization

Step 5: Select Top-K
        - Keep K Gaussians with highest scores per point
        - Handle empty voxels (pad with -1)

Output: QueryResult
        ├── topk_sphere_ids: (N, K)
        └── topk_densities: (N, K) [computed scores]
```

### Phase 3: MLE Optimization

```
Input: PointCloud P (N points)
       CSRGridData G
       QueryResult (Top-K associations)

Step 1: Initialize Transform
        - Default: identity
        - Multi-init: random perturbations around identity
        - Or: use ICP coarse initialization

Step 2: Parameterize Transformation
        SE(3): $T = \exp(\xi)$ where $\xi \in \mathfrak{se}(3)$ (6-DOF)
        Sim(3): $T = s \cdot \exp(\xi)$ where $s = \exp(\log_s)$ (7-DOF)

Step 3: Optimization Loop
        For iteration t = 1 to T:
        
        a. Forward Pass:
           - Transform points: $P' = T(P)$
           - Query Top-K Gaussians for P'
           - Compute NLL loss:
             $$
             \mathcal{L} = -\frac{1}{N} \sum_{j=1}^N \text{logsumexp}_{i \in \mathcal{K}_j} \left( \log w_i - \frac{1}{2} d_{ij}^2 + \log Z_i \right)
             $$
        
        b. Backward Pass:
           - Compute gradients: $\nabla_\xi \mathcal{L}$
           - Clip gradients for stability
        
        c. Update:
           - $\xi \leftarrow \xi - \eta \cdot \nabla_\xi \mathcal{L}$
           - Update learning rate (scheduler)
        
        d. Convergence Check:
           - Loss change < threshold?
           - Patience counter > max?

Step 4: Return Result
        - Best transform T* = exp(ξ*)
        - Final loss, inlier ratio, convergence flag

Output: RegistrationResult
        ├── transform: (4, 4)
        ├── loss: float
        ├── inlier_ratio: float
        ├── converged: bool
        └── num_iters: int
```

## Key Advantages Over ICP

### 1. Soft Correspondences
- **ICP**: Hard nearest neighbor (binary 0/1)
- **MLE**: Soft probabilistic weights (continuous [0,1])
- **Benefit**: Robust to noise, handles partial overlap

### 2. Covariance-Aware
- **ICP**: Treats all points as isotropic
- **MLE**: Respects anisotropic Gaussian shapes
- **Benefit**: Better alignment for elongated/flat structures

### 3. Natural Outlier Handling
- **ICP**: Outliers can pull solution away
- **MLE**: Low-opacity Gaussians have low weight; uniform background model possible
- **Benefit**: More robust to outliers

### 4. Scale Optimization
- **ICP**: Requires separate scale estimation
- **MLE**: Scale is part of the likelihood (Sim(3))
- **Benefit**: Joint optimization of all parameters

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Grid Build | $O(M \cdot \bar{v})$ | $\bar{v}$ = avg voxels per Gaussian, one-time |
| Top-K Query | $O(N \cdot \bar{c})$ | $\bar{c}$ = candidates per voxel, ~10-100 |
| Loss Compute | $O(N \cdot K)$ | K = 8 (constant) |
| Gradient | $O(N \cdot K)$ | Backprop through transform |
| **Total/Iter** | **$O(N \cdot K)$** | ~linear in point count |

Compare to ICP:
- **ICP**: $O(N \cdot M)$ for nearest neighbor search
- **MLE**: $O(N \cdot K)$ where $K \ll M$ (thanks to spatial indexing)

## Loss Landscape Visualization

```
Loss
  │
  │    ╭────╮
  │   ╱      ╲     ← Sharp minimum (ICP)
  │  ╱        ╲
  │ ╱          ╲
  │╱            ╲____
  └───────────────────── Transform Parameter
       │
     Ground Truth


  │
  │       ╭────────╮
  │     ╱            ╲    ← Smooth basin (MLE)
  │    ╱              ╲
  │   ╱                ╲
  │  ╱                  ╲___
  └──────────────────────────
       │
     Ground Truth
```

MLE 的 soft assignment  creates a smoother loss landscape with fewer local minima, making optimization more robust.
