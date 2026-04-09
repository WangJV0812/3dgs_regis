# CSR Grid Point-to-Sphere Matching 开发计划 (修正版)

**版本**: 2.0  
**日期**: 2026-04-06  
**更新**: 基于 Phase 0-2 实际开发经验修正

---

## 经验总结 (Lessons Learned)

### Phase 0: Morton Code
| 原计划 | 实际问题 | 最终方案 |
|-------|---------|---------|
| 仅点云 morton | 需要 grid 坐标编码 | 增加 `grid_coords_to_morton` |
| 无 | uint32 min/max 报错 | 转换为 int64 后运算 |

### Phase 1: CSR Grid Builder
| 原计划 | 实际问题 | 最终方案 |
|-------|---------|---------|
| Python loop 建表 | 太慢 (数分钟) | Vectorized PyTorch + Taichi |
| M*100 预分配 | 内存浪费/可能溢出 | Two-pass: count → prefix sum |
| Single kernel | atomic_add 冲突 | Two kernels: count + enumerate |
| 忽略 oversized | 大 sphere 丢失 | 单独处理 oversized spheres |

### Phase 2: CSR Grid Querier
| 原计划 | 实际问题 | 最终方案 |
|-------|---------|---------|
| PyTorch einsum | 7s+ 太慢 | Taichi kernel 并行化 |
| Chunk loop (512) | GPU 利用率低 | Full parallel per (point, cand) |
| use_taichi flag | 配置复杂 | 只保留高效实现 |
| 无 contiguous | kernel 报错 | 强制 `.contiguous()` |
| Single batch | OOM 风险 | Batch processing (10k) |

---

## 修正后的 Phase 3: MLE Registration

### 3.1 设计原则 (基于经验)

```
1. 优先 PyTorch vectorized，必要时 Taichi kernel
2. 内存预分配必须精确 (prefix sum)，不用 estimate
3. 保持代码简洁，不保留冗余实现
4. 自动求导优先，手动梯度作为备选
5. 数值稳定性优先 (logsumexp, epsilon)
```

### 3.2 核心架构

```python
# Layer 1: Python Control (torch.nn.Module)
class MLEAlignmentLoss(nn.Module):
    def forward(self, points, transform_matrix, csr_querier):
        # 1. Transform points (PyTorch)
        # 2. Query Top-K (CSRGridQuerier - Taichi accelerated)
        # 3. Compute NLL loss (PyTorch vectorized)
        return loss

# Layer 2: PyTorch Operations (vectorized)
def compute_nll_loss(log_densities, weights):
    # Numerically stable with logsumexp
    return -torch.logsumexp(log_densities + torch.log(weights), dim=-1)

# No Layer 3 Taichi needed for MLE (PyTorch sufficient)
```

### 3.3 修正后的实现方案

#### 方案对比

| 方案 | 对应关系 | 计算复杂度 | 数值稳定性 | 推荐 |
|-----|---------|-----------|-----------|------|
| **Soft GMM** (推荐) | 加权所有 K 个 | O(N×K) | logsumexp | ✅ |
| Hard NN | 仅最近 | O(N×K) | 简单 | ❌ |
| Full GMM | 所有 M 个 | O(N×M) | 不可行 | ❌ |

**选择**: Soft GMM with Top-K (K=8)

#### 损失函数实现

```python
def forward(self, points, transform):
    """
    Args:
        points: [N, 3] query points
        transform: [4, 4] transformation matrix (differentiable)
    Returns:
        loss: scalar (negative log-likelihood)
    """
    # Step 1: Transform points
    points_transformed = transform_points(points, transform)  # [N, 3]
    
    # Step 2: Query Top-K (Taichi accelerated)
    result = self.querier.query(points_transformed)  # [N, K]
    
    # Step 3: Gather Gaussian parameters
    # centers: [N, K, 3], cov_inv: [N, K, 3, 3]
    centers = self.grid_data.sphere_centers[result.topk_sphere_ids]
    cov_inv = self.grid_data.cov_inv[result.topk_sphere_ids]
    
    # Step 4: Compute densities (vectorized)
    diff = points_transformed.unsqueeze(1) - centers  # [N, K, 3]
    mahalanobis = (diff @ cov_inv @ diff.transpose(-2, -1)).squeeze(-1)  # [N, K]
    log_densities = -0.5 * mahalanobis + self.log_norm_factors  # [N, K]
    
    # Step 5: NLL with numerical stability
    weights = self.opacities[result.topk_sphere_ids]  # [N, K]
    nll = -torch.logsumexp(log_densities + torch.log(weights + 1e-8), dim=-1)
    
    return nll.mean()
```

### 3.4 关键修正点

#### 修正 1: 不使用 Taichi for Loss
**原因**: 
- PyTorch vectorized 已经足够快 (< 1ms for 10k points)
- 需要自动求导，Taichi kernel 需要手动 backward
- 保持简单，统一使用 PyTorch

#### 修正 2: 精确内存管理
```python
# ❌ 原计划: 预分配 [N, K, 3, 3] 可能很大
# ✅ 修正: 用 index_select 动态 gather，不预存
# PyTorch 的索引是 O(1) 开销，内存按需
```

#### 修正 3: 数值稳定性
```python
# ❌ 原设计: 直接 exp 再 log
# densities = torch.exp(-0.5 * mahalanobis)
# loss = -torch.log(torch.sum(weights * densities))

# ✅ 修正: logsumexp
def stable_nll(log_densities, log_weights):
    return -torch.logsumexp(log_densities + log_weights, dim=-1)
```

#### 修正 4: 梯度检查
```python
# 变换参数化使用李代数或四元数
# 验证: torch.autograd.gradcheck
```

### 3.5 模块设计 (修正后)

```
gmm_point_alignment/
├── sphere_mle_loss.py           # Phase 3: MLE 配准
│   └── MLEAlignmentLoss
│       ├── forward()            # 主入口
│       ├── _compute_nll()       # 负对数似然
│       └── _transform_points()  # 可微分变换
│
├── transform_utils.py           # 新增: 变换工具
│   ├── se3_exp()                # 李代数 → SE(3)
│   ├── se3_log()                # SE(3) → 李代数
│   └── quaternion_to_matrix()   # 四元数 → 旋转矩阵
│
└── gmm_registration.py          # 主控器 (简化版)
    └── GMMRegistration
        ├── __init__(grid_data)
        ├── compute_loss(points, transform)
        └── optimize(points, initial_transform=None)
```

### 3.6 文件详细设计

#### sphere_mle_loss.py

```python
@dataclass
class MLELossConfig:
    """修正: 简化配置"""
    top_k: int = 8                    # Top-K 候选
    min_opacity: float = 1e-3         # 忽略透明度过低的高斯
    use_weighted: bool = True         # 使用不透明度加权
    # ❌ 删除: temperature, use_soft_assignment (过度设计)

class MLEAlignmentLoss(nn.Module):
    """
    修正: 简化实现，仅用 PyTorch vectorized
    """
    def __init__(self, grid_data: CSRGridData, config: MLELossConfig = None):
        super().__init__()
        self.grid_data = grid_data
        self.config = config or MLELossConfig()
        
        # 预计算 log 归一化因子
        self.register_buffer('log_norm_factors', 
                            torch.log(grid_data.norm_factor + 1e-10))
        self.register_buffer('log_opacities',
                            torch.log(grid_data.opacities + 1e-10))
    
    def forward(self, points: torch.Tensor, 
                transform: torch.Tensor) -> torch.Tensor:
        """
        计算 NLL 损失
        
        Args:
            points: [N, 3] 点云
            transform: [4, 4] 变换矩阵
        
        Returns:
            loss: scalar
        """
        # 变换点 (可微分)
        points_t = self._transform(points, transform)
        
        # 查询 Top-K (使用已有的 CSRQuerier)
        # 注意: querier 内部是 Taichi，但输出是 PyTorch tensor
        result = self.querier.query(points_t)
        
        # 计算损失 (PyTorch vectorized)
        loss = self._compute_nll(points_t, result)
        
        return loss
    
    def _compute_nll(self, points: torch.Tensor, 
                     query_result: QueryResult) -> torch.Tensor:
        """向量化的 NLL 计算"""
        N, K = query_result.topk_sphere_ids.shape
        
        # Gather parameters
        centers = self.grid_data.sphere_centers[query_result.topk_sphere_ids]  # [N, K, 3]
        cov_inv = self.grid_data.cov_inv[query_result.topk_sphere_ids]          # [N, K, 3, 3]
        
        # Mahalanobis distance
        diff = points.unsqueeze(1) - centers  # [N, K, 3]
        temp = torch.einsum('nkij,nkj->nki', cov_inv, diff)  # [N, K, 3]
        mahal = (diff * temp).sum(dim=-1)  # [N, K]
        
        # Log densities
        log_densities = -0.5 * mahal + self.log_norm_factors[query_result.topk_sphere_ids]
        
        # Weights
        log_weights = self.log_opacities[query_result.topk_sphere_ids]
        
        # Numerically stable NLL
        nll = -torch.logsumexp(log_densities + log_weights, dim=-1)
        
        return nll.mean()
```

#### transform_utils.py (新增)

```python
"""变换工具: SE(3) 参数化"""

import torch
import torch.nn as nn

def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """
    李代数 se(3) → SE(3) 矩阵
    
    Args:
        xi: [..., 6] [tx, ty, tz, wx, wy, wz]
    
    Returns:
        T: [..., 4, 4] 变换矩阵
    """
    # 分割平移和旋转
    t = xi[..., :3]   # [..., 3]
    w = xi[..., 3:]   # [..., 3] 轴角
    
    # so(3) → SO(3) 使用罗德里格斯公式
    theta = w.norm(dim=-1, keepdim=True)  # [...]
    w_hat = w / (theta + 1e-8)            # 单位轴
    
    # 叉积矩阵
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    K[..., 0, 1] = -w_hat[..., 2]
    K[..., 0, 2] = w_hat[..., 1]
    K[..., 1, 0] = w_hat[..., 2]
    K[..., 1, 2] = -w_hat[..., 0]
    K[..., 2, 0] = -w_hat[..., 1]
    K[..., 2, 1] = w_hat[..., 0]
    
    # Rodrigues
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    R = I + torch.sin(theta).unsqueeze(-1) * K + \
        (1 - torch.cos(theta)).unsqueeze(-1) * (K @ K)
    
    # 构建变换矩阵
    T = torch.zeros(*xi.shape[:-1], 4, 4, device=xi.device, dtype=xi.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    
    return T

class SE3Parameter(nn.Module):
    """可优化的 SE(3) 参数"""
    def __init__(self, xi_init=None):
        super().__init__()
        if xi_init is None:
            xi_init = torch.zeros(6)
        self.xi = nn.Parameter(xi_init)
    
    def matrix(self):
        return se3_exp(self.xi)
    
    def forward(self, points):
        """变换点"""
        T = self.matrix()
        return (T[:3, :3] @ points.T).T + T[:3, 3]
```

#### gmm_registration.py (简化主控器)

```python
class GMMRegistration:
    """
    简化版主控器: 只做配准，不重复建 grid
    """
    def __init__(self, grid_data: CSRGridData):
        self.grid_data = grid_data
        self.querier = CSRGridQuerier(grid_data)
        self.loss_fn = MLEAlignmentLoss(grid_data)
    
    def register(self, 
                 points: torch.Tensor,
                 initial_transform: torch.Tensor = None,
                 num_iters: int = 100,
                 lr: float = 0.01) -> torch.Tensor:
        """
        配准点云到场景
        
        Args:
            points: [N, 3] 点云
            initial_transform: [4, 4] 初始变换 (可选)
            num_iters: 优化迭代次数
            lr: 学习率
        
        Returns:
            transform: [4, 4] 最优变换
        """
        # 初始化
        if initial_transform is None:
            xi = torch.zeros(6, device=points.device, requires_grad=True)
        else:
            # 从矩阵转换到李代数 (近似)
            xi = self._matrix_to_se3(initial_transform)
            xi.requires_grad_(True)
        
        optimizer = torch.optim.Adam([xi], lr=lr)
        
        # 优化
        for i in range(num_iters):
            optimizer.zero_grad()
            
            T = se3_exp(xi)
            loss = self.loss_fn(points, T)
            
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Iter {i}: loss = {loss.item():.4f}")
        
        return se3_exp(xi)
```

### 3.7 开发优先级 (修正)

#### Phase 3a: 核心 MLE Loss (Must Have)
- [ ] `transform_utils.py` - SE(3) 参数化
- [ ] `sphere_mle_loss.py` - NLL 损失 (Soft GMM)
- [ ] 单元测试: 梯度检查，数值稳定性

#### Phase 3b: 优化器集成 (Must Have)
- [ ] `gmm_registration.py` - 简化主控器
- [ ] Adam/SGD 优化循环
- [ ] 收敛性判断

#### Phase 3c: 鲁棒性 (Nice to Have)
- [ ] 多尺度金字塔 ( coarse-to-fine )
- [ ] 异常值剔除 (robust kernel)
- [ ] 多初始值策略

#### ❌ 删除 (过度设计)
- [ ] ~~Hard assignment 模式~~ - Soft 足够好
- [ ] ~~Temperature annealing~~ - 不需要
- [ ] ~~手动 backward~~ - PyTorch autograd 足够
- [ ] ~~Taichi kernel for loss~~ - PyTorch 足够快

### 3.8 风险与缓解 (更新)

| 风险 | 概率 | 影响 | 缓解 |
|-----|------|------|------|
| 局部最优 | 高 | 配准失败 | 好的初始化 (ICP coarse) |
| 梯度消失 | 中 | 收敛慢 | 检查数值范围，使用 logsumexp |
| Top-K 不足 | 低 | 似然估计偏差 | K=8 足够，可用 K=16 验证 |
| 内存 OOM | 低 | 崩溃 | batch processing (已支持) |

---

## 附录: 与数学文档的关联

本文档的 Phase 3 实现对应 `mle_registration_math.md` 中的:
- **公式 (7)**: Top-K 近似 NLL
- **公式 (11)**: 梯度计算 (PyTorch 自动处理)
- **Section 3.2.1**: Soft Assignment 方案

实现时参考数学文档的数值稳定性建议 (logsumexp)。

---

**文档结束**
