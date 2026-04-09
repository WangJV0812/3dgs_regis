# CSR Grid Point-to-Sphere Matching 代码规划文档

**版本**: 1.0  
**日期**: 2026-04-04  
**目标**: 实现基于 CSR (Compressed Sparse Row) 格式的高斯球体素网格索引，支持高效的 point-to-sphere Top-K 查询

---

## 1. 架构概览

### 1.1 整体流程

```
Input: GaussianScenes (M spheres), PointCloud (N points)

Phase 1: Grid Construction (One-time)
├── 1.1 Compute global AABB & voxel size
├── 1.2 Calculate AABB per sphere → (grid_min, grid_max) per sphere
├── 1.3 Enumerate sphere-voxel pairs → (morton_code, sphere_id) pairs
├── 1.4 Sort pairs by morton_code
├── 1.5 Build CSR lookup table: morton_hash → (start_idx, count)
└── 1.6 Precompute sphere covariance inverse & normalization factor

Phase 2: Query for PointCloud (Per-registration or Per-iteration)
├── 2.1 Transform points (if during optimization)
├── 2.2 Compute morton_code for each point
├── 2.3 Lookup candidate spheres via CSR table
├── 2.4 Calculate Gaussian density for candidates
├── 2.5 Select Top-K spheres per point
└── 2.6 Return association matrix [N, K]

Phase 3: MLE Registration Optimization
├── 3.1 Build association from Phase 2
├── 3.2 Compute negative log-likelihood loss
└── 3.3 Optimize pointcloud pose (or scene alignment)
```

### 1.2 与现有代码的关系

| 现有模块 | 复用/替换 | 说明 |
|---------|----------|------|
| `gs_scene_aabb.py` | 复用 | AABB计算保持不变 |
| `morton_code.py` | 扩展 | 增加3D grid morton编码函数 |
| `gs_scene_voxel_creation.py` | 替换 | Dynamic SNode → CSR 结构 |
| `misc/geometry.py` | 复用 | 协方差、密度计算函数 |
| `gmm_point_alignment.py` | 重构 | 集成CSR查询和MLE优化 |

---

## 2. 核心数据结构

### 2.1 CSR Grid 数据结构

```python
# Python/Torch Scope
csr_grid_data = {
    # Primary CSR arrays
    "pairs_morton": torch.Tensor,      # [total_pairs], dtype=torch.int64, sorted morton codes
    "pairs_sphere_id": torch.Tensor,   # [total_pairs], dtype=torch.int32, corresponding sphere ids
    
    # Lookup table: dense array for O(1) lookup
    # Size: 2^30 (for 10-bit per dim, 30-bit morton) → too large
    # Alternative: use sparse hash map or bucketed approach
    "lookup_table": torch.Tensor,      # [num_unique_hashes, 2], (morton_hash, start_idx)
    "lookup_starts": torch.Tensor,     # [num_buckets], start index in pairs for each bucket
    "lookup_counts": torch.Tensor,     # [num_buckets], number of spheres in each bucket
    
    # Alternative: Two-level lookup for 1024^3 grid
    "l1_table": torch.Tensor,          # [1024/32, 1024/32, 1024/32] → L2 block indices
    "l2_blocks": List[torch.Tensor],   # Dynamic L2 blocks containing actual pairs
    
    # Metadata
    "global_aabb_min": torch.Tensor,   # [3], float32
    "global_aabb_max": torch.Tensor,   # [3], float32
    "voxel_size": float,
    "grid_dims": Tuple[int, int, int], # (Gx, Gy, Gz)
    "total_pairs": int,
    "num_unique_voxels": int,
}

# Precomputed sphere data for fast density calculation
sphere_cache = {
    "cov_inv": torch.Tensor,           # [M, 3, 3], float32, inverse covariance matrices
    "norm_factor": torch.Tensor,       # [M], float32, normalization factors
    "center": torch.Tensor,            # [M, 3], float32, sphere centers
    "is_oversized": torch.Tensor,      # [M], bool, oversized sphere flags
}
```

### 2.2 Two-Level Lookup Table Design

为避免 1024³ = 1B 大小的 dense lookup table，采用二级索引：

```
L1 Grid: 32 x 32 x 32 = 32,768 cells (coarse)
L2 Grid: 32 x 32 x 32 per L1 cell (fine)
Total: 1024 x 1024 x 1024 fine cells

L1 table stores:
- Offset into L2 block list (or -1 if empty)
- Pointer to L2 block data

L2 block stores:
- List of (morton_hash, sphere_id) pairs for that region
- Sorted by morton_hash for binary search
```

---

## 3. 模块设计

### 3.1 模块划分

```
gmm_point_alignment/
├── csr_grid_builder.py          # Phase 1: Grid construction
│   ├── CSRGridBuilder (class)
│   └── build_csr_grid() → CSRGridData
│
├── csr_grid_querier.py          # Phase 2: Point query
│   ├── CSRGridQuerier (class)
│   ├── query_topk_cpu()         # PyTorch fallback
│   └── query_topk_taichi()      # Taichi kernel (optional)
│
├── sphere_mle_loss.py           # Phase 3: MLE optimization
│   ├── MLEAlignmentLoss (nn.Module)
│   └── compute_registration_loss()
│
└── gmm_point_alignment_v2.py    # Main orchestrator
    └── GMMPointAlignmentV2 (class)
```

---

## 4. 函数详细设计

### 4.1 csr_grid_builder.py

#### Python Scope Functions

```python
# Class: CSRGridBuilderConfig
@dataclass
class CSRGridBuilderConfig:
    confidence_level: float = 0.95           # For AABB computation
    voxel_size_factor: float = 3.0           # Median radius multiplier
    max_grid_size: int = 1024                # Per dimension
    oversized_threshold_voxels: int = 64     # Threshold for oversized spheres
    l1_grid_size: int = 32                   # Coarse grid dimension
    use_two_level_lookup: bool = True        # Enable L1/L2 hierarchy

# Class: CSRGridData
@dataclass
class CSRGridData:
    pairs_morton: torch.Tensor          # [total_pairs], int64
    pairs_sphere_id: torch.Tensor       # [total_pairs], int32
    l1_offsets: torch.Tensor            # [32, 32, 32], int32, offset into L2 or -1
    l2_blocks: List[torch.Tensor]       # List of L2 block tensors
    oversized_sphere_ids: torch.Tensor  # [num_oversized], int32
    global_aabb_min: torch.Tensor       # [3], float32
    voxel_size: float
    grid_dims: Tuple[int, int, int]
    
    # Precomputed sphere data
    cov_inv: torch.Tensor               # [M, 3, 3], float32
    norm_factor: torch.Tensor           # [M], float32

# Main builder class
class CSRGridBuilder:
    def __init__(self, config: CSRGridBuilderConfig):
        """Initialize builder with configuration."""
        pass
    
    def build(self, scene: GaussianScenes) -> CSRGridData:
        """
        Main entry: Build CSR grid from Gaussian scene.
        
        Steps:
        1. Compute per-sphere AABB
        2. Compute global AABB and voxel size
        3. Enumerate sphere-voxel pairs
        4. Sort and build CSR structure
        5. Precompute sphere covariance data
        """
        pass
    
    def _compute_voxel_size_and_aabb(
        self, 
        scene: GaussianScenes
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute voxel size and per-sphere AABB.
        
        Returns:
            min_corners: [M, 3], per-sphere AABB min
            max_corners: [M, 3], per-sphere AABB max
            voxel_size: float
        """
        pass
    
    def _enumerate_sphere_voxel_pairs(
        self,
        min_corners: torch.Tensor,      # [M, 3]
        max_corners: torch.Tensor,      # [M, 3]
        global_min: torch.Tensor,       # [3]
        voxel_size: float,
        grid_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Enumerate all (sphere, voxel) pairs where sphere overlaps voxel.
        
        Returns:
            pairs_morton: [total_pairs], int64
            pairs_sphere_id: [total_pairs], int32
            oversized_ids: [num_oversized], int32
        
        Complexity: O(total_pairs), pairs count depends on sphere sizes
        """
        pass
    
    def _build_two_level_lookup(
        self,
        pairs_morton: torch.Tensor,     # [P], sorted
        pairs_sphere_id: torch.Tensor,  # [P]
        grid_size: int,
        l1_size: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Build L1/L2 two-level lookup table.
        
        Returns:
            l1_offsets: [l1_size, l1_size, l1_size], offset or -1
            l2_blocks: List of tensors, each containing (hash, sphere_id) pairs
        """
        pass
    
    def _precompute_sphere_data(
        self,
        scene: GaussianScenes
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute covariance inverse and normalization factors.
        
        Returns:
            cov_inv: [M, 3, 3]
            norm_factor: [M]
        """
        pass
```

#### Taichi Scope Functions (Optional, for enumeration)

```python
# Taichi kernel for parallel pair enumeration
@ti.kernel
def enumerate_pairs_kernel(
    min_corners: ti.types.ndarray(ti.f32, 2),      # [M, 3]
    max_corners: ti.types.ndarray(ti.f32, 2),      # [M, 3]
    global_min: ti.types.ndarray(ti.f32, 1),       # [3]
    voxel_size: ti.f32,
    grid_size: ti.i32,
    # Output (pre-allocated with max size)
    out_morton: ti.types.ndarray(ti.i64, 1),       # [max_pairs]
    out_sphere_id: ti.types.ndarray(ti.i32, 1),    # [max_pairs]
    out_counts: ti.types.ndarray(ti.i32, 1),       # [M], pairs per sphere
    oversized_flags: ti.types.ndarray(ti.i32, 1),  # [M], 1 if oversized
) -> ti.i32:  # Returns total pairs written
    """
    Parallel enumeration of sphere-voxel pairs.
    Each thread processes one sphere.
    """
    pass
```

### 4.2 csr_grid_querier.py

#### Python Scope Functions

```python
# Class: CSRGridQuerierConfig
@dataclass
class CSRGridQuerierConfig:
    top_k: int = 8                           # Number of top spheres per point
    max_candidates_per_point: int = 64       # Limit candidates for efficiency
    use_taichi: bool = True                  # Use Taichi kernels
    batch_size: int = 10000                  # Batch processing for memory

# Class: QueryResult
@dataclass  
class QueryResult:
    topk_sphere_ids: torch.Tensor    # [N, K], int32, -1 for padding
    topk_densities: torch.Tensor     # [N, K], float32
    topk_distances: torch.Tensor     # [N, K], float32 (optional)

class CSRGridQuerier:
    def __init__(
        self,
        grid_data: CSRGridData,
        config: CSRGridQuerierConfig,
    ):
        """Initialize with pre-built grid data."""
        pass
    
    def query(
        self,
        points: torch.Tensor,           # [N, 3]
        point_transform: Optional[torch.Tensor] = None,  # [4, 4] pose
    ) -> QueryResult:
        """
        Query Top-K spheres for each point.
        
        Steps:
        1. Transform points if transform provided
        2. Compute morton codes for points
        3. Lookup candidate spheres via CSR table
        4. Calculate Gaussian density
        5. Select Top-K
        """
        pass
    
    def _query_batch_torch(
        self,
        points: torch.Tensor,           # [B, 3]
    ) -> QueryResult:
        """
        PyTorch implementation for batch query.
        Uses vectorized operations where possible.
        """
        pass
    
    def _lookup_candidates(
        self,
        morton_codes: torch.Tensor,     # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lookup candidate spheres for given morton codes.
        
        Returns:
            candidate_sphere_ids: [B, max_candidates], -1 for padding
            candidate_counts: [B], actual candidate count per point
        """
        pass
    
    def _compute_gaussian_densities(
        self,
        points: torch.Tensor,           # [B, 3]
        candidate_ids: torch.Tensor,    # [B, C]
    ) -> torch.Tensor:                  # [B, C]
        """
        Compute Gaussian density for point-sphere pairs.
        """
        pass
    
    def _select_topk(
        self,
        densities: torch.Tensor,        # [B, C]
        candidate_ids: torch.Tensor,    # [B, C]
        valid_mask: torch.Tensor,       # [B, C], bool
    ) -> QueryResult:
        """
        Select Top-K highest density associations.
        """
        pass
```

#### Taichi Scope Functions

```python
@ti.kernel
def query_topk_kernel(
    # Grid data
    pairs_morton: ti.types.ndarray(ti.i64, 1),
    pairs_sphere_id: ti.types.ndarray(ti.i32, 1),
    l1_offsets: ti.types.ndarray(ti.i32, 3),
    # Sphere data
    sphere_centers: ti.types.ndarray(ti.f32, 2),     # [M, 3]
    sphere_cov_inv: ti.types.ndarray(ti.f32, 3),     # [M, 3, 3]
    sphere_norm: ti.types.ndarray(ti.f32, 1),        # [M]
    oversized_ids: ti.types.ndarray(ti.i32, 1),
    # Query data
    points: ti.types.ndarray(ti.f32, 2),             # [N, 3]
    # Grid params
    global_min: ti.types.ndarray(ti.f32, 1),
    voxel_size: ti.f32,
    l1_size: ti.i32,
    l2_size: ti.i32,
    # Output
    topk_ids: ti.types.ndarray(ti.i32, 2),           # [N, K]
    topk_scores: ti.types.ndarray(ti.f32, 2),        # [N, K]
    K: ti.template(),
):
    """
    Taichi kernel for parallel Top-K query.
    Each thread processes one point.
    """
    pass

@ti.func
def lookup_candidates_ti(
    point_morton: ti.i64,
    l1_offsets: ti.template(),
    pairs_morton: ti.template(),
    pairs_sphere_id: ti.template(),
    out_candidates: ti.template(),
) -> ti.i32:
    """
    Lookup candidate sphere IDs for a point.
    Returns number of candidates found.
    """
    pass

@ti.func
def insert_topk(
    sphere_id: ti.i32,
    density: ti.f32,
    topk_ids: ti.template(),
    topk_scores: ti.template(),
    K: ti.template(),
):
    """
    Insert a candidate into sorted Top-K list.
    """
    pass
```

### 4.3 sphere_mle_loss.py

```python
# Class: MLELossConfig
@dataclass
class MLELossConfig:
    top_k: int = 8
    min_density_threshold: float = 1e-6
    use_soft_assignment: bool = False    # Hard vs soft top-k
    temperature: float = 0.1             # For soft assignment

class MLEAlignmentLoss(nn.Module):
    def __init__(
        self,
        grid_data: CSRGridData,
        config: MLELossConfig,
    ):
        """
        Initialize MLE loss for point-to-sphere registration.
        """
        pass
    
    def forward(
        self,
        points: torch.Tensor,               # [N, 3]
        point_transform: torch.Tensor,      # [4, 4] or [B, 4, 4]
        querier: CSRGridQuerier,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute negative log-likelihood loss.
        
        Returns:
            loss: scalar
            log_likelihood: scalar
            inlier_ratio: scalar
            mean_topk_density: scalar
        """
        pass
    
    def _compute_negative_log_likelihood(
        self,
        points_transformed: torch.Tensor,   # [N, 3]
        topk_ids: torch.Tensor,             # [N, K]
        topk_densities: torch.Tensor,       # [N, K]
    ) -> torch.Tensor:
        """
        Compute NLL assuming Gaussian mixture.
        
        For each point: log p(point) = log(sum_k w_k * N(point | sphere_k))
        where w_k are mixture weights (can be uniform or based on density).
        """
        pass
```

### 4.4 gmm_point_alignment_v2.py (Orchestrator)

```python
class GMMPointAlignmentV2(nn.Module):
    """
    Main orchestrator class combining all components.
    """
    
    def __init__(
        self,
        grid_config: CSRGridBuilderConfig = CSRGridBuilderConfig(),
        query_config: CSRGridQuerierConfig = CSRGridQuerierConfig(),
        loss_config: MLELossConfig = MLELossConfig(),
    ):
        super().__init__()
        self.grid_builder = CSRGridBuilder(grid_config)
        self.grid_data: Optional[CSRGridData] = None
        self.querier: Optional[CSRGridQuerier] = None
        self.loss_fn = None  # Initialized after grid build
    
    def build_grid(self, scene: GaussianScenes):
        """Build CSR grid for scene (one-time cost)."""
        self.grid_data = self.grid_builder.build(scene)
        self.querier = CSRGridQuerier(self.grid_data, self.query_config)
        self.loss_fn = MLEAlignmentLoss(self.grid_data, self.loss_config)
    
    def forward(
        self,
        scene: GaussianScenes,
        pointcloud: torch.Tensor,
        pointcloud_transform: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            topk_ids: [N, K]
            topk_densities: [N, K]
            loss: scalar (if transform provided)
        """
        if self.grid_data is None:
            self.build_grid(scene)
        
        # Query associations
        query_result = self.querier.query(pointcloud, pointcloud_transform)
        
        # Compute loss if transform provided
        if pointcloud_transform is not None:
            loss_dict = self.loss_fn(pointcloud, pointcloud_transform, self.querier)
            return {
                **query_result.__dict__,
                **loss_dict,
            }
        
        return query_result.__dict__
    
    def optimize_alignment(
        self,
        scene: GaussianScenes,
        pointcloud: torch.Tensor,
        initial_transform: Optional[torch.Tensor] = None,
        num_iterations: int = 100,
        lr: float = 0.01,
    ) -> torch.Tensor:
        """
        Optimize pointcloud alignment using MLE.
        
        Returns:
            optimized_transform: [4, 4]
        """
        pass
```

---

## 5. 数据流图

### 5.1 Grid Construction Flow

```
GaussianScenes (M spheres)
    ↓
[gaussian_scene_aabb] → min_corners[M,3], max_corners[M,3]
    ↓
[enumerate_sphere_voxel_pairs]
    For each sphere:
        - Convert AABB to grid coordinates
        - If voxels > threshold: mark oversized
        - Else: enumerate all (morton, sphere_id) pairs
    ↓
pairs: [(morton_0, id_0), (morton_1, id_1), ...]
    ↓
[sort by morton]
    ↓
Sorted pairs + [build_two_level_lookup]
    ↓
CSRGridData
```

### 5.2 Query Flow

```
PointCloud [N, 3] + Transform[4,4] (optional)
    ↓
[Transform points]
    ↓
[Compute morton codes]
    ↓
For each point:
    [Lookup L1] → L2 block index
    [Lookup L2] → Candidate sphere IDs
    [Collect candidates]
    ↓
Candidates [N, C_max]
    ↓
[Compute densities] using precomputed cov_inv
    density = N(point | center, covariance)
    ↓
[Top-K selection]
    ↓
QueryResult: topk_ids [N,K], topk_densities [N,K]
```

---

## 6. 复杂度分析

### 6.1 空间复杂度

| 组件 | 大小 | 说明 |
|-----|------|------|
| pairs_morton | P × 8 bytes | P = total sphere-voxel overlaps |
| pairs_sphere_id | P × 4 bytes | |
| L1 lookup table | 32K × 4 bytes | Fixed small size |
| L2 blocks | ~P × 4 bytes (overhead) | Depends on distribution |
| sphere_cache | M × (36 + 4 + 12) bytes | cov_inv + norm + center |
| **Total** | **~12P + 52M + 128KB** | |

**典型场景估算**:
- M = 100K spheres, 平均每球覆盖 8 voxels → P = 800K
- Total ≈ 12 × 800K + 52 × 100K = 9.6MB + 5.2MB = **~15MB**
- 对比 Dynamic SNode: 可能有 8GB+ 指针表开销

### 6.2 时间复杂度

| 操作 | 复杂度 | 说明 |
|-----|--------|------|
| Build AABB | O(M) | Parallel per-sphere |
| Enumerate pairs | O(P) | P = total overlaps |
| Sort pairs | O(P log P) | Dominant cost in build |
| Build lookup | O(P) | Linear scan |
| **Total Build** | **O(P log P)** | One-time cost |
| Query per point | O(1) lookup + O(C log K) | C = candidates, K = topk |
| Density compute | O(C) | Per point |
| **Total Query** | **O(N × C log K)** | C typically small (< 50) |

---

## 7. 边界情况处理

### 7.1 超大球 (Oversized Spheres)

```python
# During enumeration
if num_voxels > oversized_threshold:
    oversized_ids.append(sphere_id)
    # Do NOT insert into CSR pairs

# During query
candidates = lookup_csr(point_morton)  # Normal spheres
for oid in oversized_ids:              # Always check oversized
    candidates.append(oid)
```

### 7.2 空 Voxel 查询

```python
# L1 lookup returns -1 for empty regions
def lookup_candidates(morton_code):
    l1_idx = morton_to_l1(morton_code)
    l2_offset = l1_table[l1_idx]
    if l2_offset == -1:
        return []  # Empty voxel
    # ... binary search in L2 block
```

### 7.3 点在 Scene AABB 外

```python
# Clamp to valid grid coordinates
grid_coord = torch.clamp(
    ((point - global_min) / voxel_size).long(),
    0, grid_size - 1
)
```

---

## 8. 实现优先级

### Phase 1: Core CSR (Must Have)
- [ ] `CSRGridBuilderConfig`, `CSRGridData` dataclasses
- [ ] `CSRGridBuilder.build()` main flow
- [ ] `_enumerate_sphere_voxel_pairs()` (PyTorch version first)
- [ ] `_build_two_level_lookup()`
- [ ] Unit tests for grid construction

### Phase 2: Query (Must Have)
- [ ] `CSRGridQuerier.query()` main flow
- [ ] `_query_batch_torch()` vectorized implementation
- [ ] `_lookup_candidates()` with two-level lookup
- [ ] `_select_topk()`
- [ ] Integration test

### Phase 3: MLE Loss (Must Have)
- [ ] `MLEAlignmentLoss` basic version
- [ ] Negative log-likelihood computation
- [ ] End-to-end registration test

### Phase 4: Optimization (Nice to Have)
- [ ] Taichi kernels for enumeration
- [ ] Taichi kernels for query
- [ ] Batch processing for large pointclouds
- [ ] Memory-mapped storage for very large scenes

---

## 9. 代码依赖关系

```
gmm_point_alignment_v2.py (main API)
    ├── csr_grid_builder.py
    │   ├── gs_scene_aabb.py (existing)
    │   ├── morton_code.py (existing + extend)
    │   └── misc/geometry.py (existing)
    │
    ├── csr_grid_querier.py
    │   └── csr_grid_builder.py (CSRGridData)
    │
    └── sphere_mle_loss.py
        └── csr_grid_querier.py
```

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| P (total pairs) 过大导致内存不足 | 高 | 监控 P/M ratio，调整 voxel_size_factor |
| Sort O(P log P) 构建时间过长 | 中 | 使用 torch.sort (GPU accelerated)，考虑增量更新 |
| 查询时某些 voxel 候选过多 | 中 | 设置 max_candidates_per_point 上限 |
| Morton code collision (不同坐标相同hash) | 低 | 在 L2 block 中存储精确坐标验证 |

---

## 附录: 关键算法伪代码

### A.1 Two-Level Lookup Build

```python
def build_two_level_lookup(pairs_morton, pairs_sphere_id, grid_size=1024, l1_size=32):
    l2_size = grid_size // l1_size  # 32
    
    # Group pairs by L1 block
    l1_to_pairs = defaultdict(list)
    for morton, sid in zip(pairs_morton, pairs_sphere_id):
        x, y, z = decode_morton(morton)
        l1_x, l1_y, l1_z = x // l2_size, y // l2_size, z // l2_size
        l1_to_pairs[(l1_x, l1_y, l1_z)].append((morton, sid))
    
    # Build L1 offset table and L2 blocks
    l1_offsets = torch.full((l1_size, l1_size, l1_size), -1, dtype=torch.int32)
    l2_blocks = []
    
    for (l1_x, l1_y, l1_z), block_pairs in l1_to_pairs.items():
        # Sort within L2 block by morton
        block_pairs.sort(key=lambda x: x[0])
        
        # Store offset
        l1_offsets[l1_x, l1_y, l1_z] = len(l2_blocks)
        
        # Create L2 block tensor
        block_tensor = torch.tensor(block_pairs, dtype=torch.int64)  # [N, 2]
        l2_blocks.append(block_tensor)
    
    return l1_offsets, l2_blocks
```

### A.2 Query with Two-Level Lookup

```python
def lookup_candidates(point_morton, l1_offsets, l2_blocks, grid_size=1024, l1_size=32):
    x, y, z = decode_morton(point_morton)
    l2_size = grid_size // l1_size
    
    l1_x, l1_y, l1_z = x // l2_size, y // l2_size, z // l2_size
    
    l2_idx = l1_offsets[l1_x, l1_y, l1_z]
    if l2_idx == -1:
        return []  # Empty
    
    l2_block = l2_blocks[l2_idx]  # [N, 2] (morton, sphere_id)
    
    # Binary search for exact morton match
    mortons = l2_block[:, 0]
    left = torch.searchsorted(mortons, point_morton, right=False)
    right = torch.searchsorted(mortons, point_morton, right=True)
    
    return l2_block[left:right, 1].tolist()  # sphere_ids
```

---

**文档结束**
