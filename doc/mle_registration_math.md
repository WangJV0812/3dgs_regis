# MLE (最大似然估计) 点云配准数学原理

**文档版本**: 1.0  
**日期**: 2026-04-06  
**目标**: 分析使用 MLE 进行 3DGS 场景与点云配准的数学原理和可行性

---

## 1. 问题定义

### 1.1 符号定义

| 符号 | 含义 | 维度 |
|-----|------|------|
| $\mathcal{G}$ | 3D Gaussian Splatting 场景 | - |
| $M$ | 场景中高斯球数量 | scalar |
| $\mathbf{\mu}_i$ | 第 $i$ 个高斯球的中心位置 | $\mathbb{R}^3$ |
| $\Sigma_i$ | 第 $i$ 个高斯球的协方差矩阵 | $\mathbb{R}^{3 \times 3}$ |
| $\mathbf{P}$ | 点云 | - |
| $N$ | 点云中的点数 | scalar |
| $\mathbf{p}_j$ | 第 $j$ 个点 | $\mathbb{R}^3$ |
| $\mathbf{T}$ | 点云到场景的变换矩阵 (待优化) | $\mathbb{R}^{4 \times 4}$ |
| $\mathbf{p}'_j$ | 变换后的点: $\mathbf{p}'_j = \mathbf{T} \mathbf{p}_j$ | $\mathbb{R}^3$ |

### 1.2 问题陈述

**输入**:
- 3DGS 场景 $\mathcal{G}$ 包含 $M$ 个各向异性 3D 高斯球
- 点云 $\mathbf{P} = \{\mathbf{p}_1, \mathbf{p}_2, ..., \mathbf{p}_N\}$

**输出**:
- 最优刚性变换 $\mathbf{T}^* \in SE(3)$，使得点云与场景对齐

---

## 2. 数学模型

### 2.1 单个高斯球的概率密度

第 $i$ 个高斯球定义了三维空间中的概率密度函数:

$$
g_i(\mathbf{x}) = \frac{1}{(2\pi)^{3/2} |\Sigma_i|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \mathbf{\mu}_i)^T \Sigma_i^{-1} (\mathbf{x} - \mathbf{\mu}_i)\right)
$$

其中:
- $\mathbf{\mu}_i \in \mathbb{R}^3$: 高斯中心
- $\Sigma_i \in \mathbb{R}^{3 \times 3}$: 协方差矩阵 (对称正定)
- $|\Sigma_i|$: 协方差矩阵的行列式

### 2.2 场景的混合高斯表示

整个 3DGS 场景可以看作一个**高斯混合模型 (GMM)**:

$$
p_{\mathcal{G}}(\mathbf{x}) = \sum_{i=1}^{M} w_i \cdot g_i(\mathbf{x})
$$

其中 $w_i$ 是混合权重，满足 $\sum_{i=1}^{M} w_i = 1$。

**两种权重选择**:
1. **均匀权重**: $w_i = \frac{1}{M}$
2. **不透明度加权**: $w_i = \frac{\alpha_i}{\sum_j \alpha_j}$，其中 $\alpha_i$ 是 3DGS 的不透明度

### 2.3 配准的似然函数

**核心假设**: 观测到的点云是从场景分布中采样得到的。

给定变换 $\mathbf{T}$，变换后的点 $\mathbf{p}'_j = \mathbf{T} \mathbf{p}_j$ 在场景中的似然为:

$$
L(\mathbf{T}) = p_{\mathcal{G}}(\mathbf{P} | \mathbf{T}) = \prod_{j=1}^{N} p_{\mathcal{G}}(\mathbf{p}'_j) = \prod_{j=1}^{N} \sum_{i=1}^{M} w_i \cdot g_i(\mathbf{p}'_j)
$$

### 2.4 负对数似然 (NLL) 损失

为了数值稳定性，使用负对数似然:

$$
\mathcal{L}_{\text{NLL}}(\mathbf{T}) = -\log L(\mathbf{T}) = -\sum_{j=1}^{N} \log \left( \sum_{i=1}^{M} w_i \cdot g_i(\mathbf{p}'_j) \right)
$$

**优化目标**:

$$
\mathbf{T}^* = \arg\min_{\mathbf{T} \in SE(3)} \mathcal{L}_{\text{NLL}}(\mathbf{T})
$$

---

## 3. 近似与简化

### 3.1 Top-K 近似

直接计算所有 $M$ 个高斯球的贡献计算量过大 ($O(N \times M)$)。

**关键观察**: 对于任意点 $\mathbf{p}'_j$，只有附近的高斯球有显著贡献。

**Top-K 近似**:

$$
p_{\mathcal{G}}(\mathbf{p}'_j) \approx \sum_{i \in \mathcal{K}_j} w_i \cdot g_i(\mathbf{p}'_j)
$$

其中 $\mathcal{K}_j$ 是距离 $\mathbf{p}'_j$ 最近的 $K$ 个高斯球的索引集合。

近似后的损失:

$$
\mathcal{L}_{\text{approx}}(\mathbf{T}) = -\sum_{j=1}^{N} \log \left( \sum_{i \in \mathcal{K}_j} w_i \cdot g_i(\mathbf{p}'_j) \right)
$$

### 3.2 Hard vs Soft Assignment

#### 3.2.1 Soft Assignment (GMM 方式)

保留所有 $K$ 个候选的贡献，使用 log-sum-exp 技巧数值稳定计算:

```
log(sum(exp(log_densities))) = log_sum_exp(log_densities)
```

PyTorch: `torch.logsumexp(log_densities, dim=-1)`

#### 3.2.2 Hard Assignment (最近邻方式)

只使用最近的高斯球:

$$
\mathcal{L}_{\text{hard}}(\mathbf{T}) = -\sum_{j=1}^{N} \log g_{i^*_j}(\mathbf{p}'_j)
$$

其中 $i^*_j = \arg\max_{i \in \mathcal{K}_j} g_i(\mathbf{p}'_j)$

---

## 4. 梯度推导

### 4.1 变换参数的梯度

变换 $\mathbf{T} \in SE(3)$ 可以用李代数 $\mathfrak{se}(3)$ 表示为 6 维向量 $\boldsymbol{\xi} = [\mathbf{t}; \boldsymbol{\omega}]$:

- $\mathbf{t} \in \mathbb{R}^3$: 平移
- $\boldsymbol{\omega} \in \mathbb{R}^3$: 旋转 (轴角表示)

点变换:

$$
\mathbf{p}'_j = \mathbf{R}(\boldsymbol{\omega}) \mathbf{p}_j + \mathbf{t}
$$

### 4.2 链式法则

对于每个点 $\mathbf{p}_j$，损失对其变换后位置的梯度:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{p}'_j} = -\frac{\sum_{i \in \mathcal{K}_j} w_i \cdot g_i(\mathbf{p}'_j) \cdot \Sigma_i^{-1} (\mathbf{p}'_j - \mathbf{\mu}_i)}{\sum_{i \in \mathcal{K}_j} w_i \cdot g_i(\mathbf{p}'_j)}
$$

这个梯度可以理解为**加权残差向量**，指向高斯中心。

### 4.3 变换参数的梯度

使用链式法则:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{t}} = \sum_{j=1}^{N} \frac{\partial \mathcal{L}}{\partial \mathbf{p}'_j}
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\omega}} = \sum_{j=1}^{N} \left( \mathbf{p}_j \times \frac{\partial \mathcal{L}}{\partial \mathbf{p}'_j} \right)
$$

---

## 5. 与现有方法的对比

### 5.1 与 ICP (Iterative Closest Point) 对比

| 特性 | ICP | MLE (GMM) |
|-----|-----|-----------|
| 对应关系 | 最近点 (Hard) | 软权重 (Soft) |
| 损失函数 | 点-点/点-平面距离 | 负对数似然 |
| 收敛性 | 局部最优，对初始化敏感 | 更平滑的损失面 |
| 处理部分重叠 | 需要鲁棒核函数 | 自然处理 (低似然区域) |
| 计算复杂度 | $O(N \log M)$ | $O(N \times K)$，$K \ll M$ |

### 5.2 与 NDT (Normal Distributions Transform) 对比

| 特性 | NDT | MLE (3DGS) |
|-----|-----|------------|
| 体素化 | 是 (固定网格) | 否 (高斯中心自适应) |
| 分布表示 | 体素内点云的协方差 | 3DGS 高斯球 |
| 多尺度 | 需要金字塔 | 自然多尺度 (不同大小高斯) |
| 方向性 | 各向异性 (体素级别) | 各向异性 (每个高斯) |

---

## 6. 数学可行性分析

### 6.1 理论保证

**定理**: 在 3DGS 场景准确表示几何表面的前提下，MLE 配准的损失函数 $\mathcal{L}_{\text{NLL}}$ 在真实对齐变换 $\mathbf{T}^*$ 处取得最小值。

**证明思路**:
1. 3DGS 的高斯球密集覆盖表面 → 场景分布 $p_{\mathcal{G}}$ 在表面处概率密度高
2. 当点云 $\mathbf{P}$ 采样自表面时，在 $\mathbf{T}^*$ 处每个 $\mathbf{p}'_j$ 位于高概率区域
3. 因此 $p_{\mathcal{G}}(\mathbf{p}'_j)$ 在 $\mathbf{T}^*$ 处最大化，$\mathcal{L}_{\text{NLL}}$ 最小化

### 6.2 实际考虑

#### 6.2.1 优势

1. **概率解释清晰**: 损失有明确的统计意义
2. **软对应**: 自动处理模糊对应关系
3. **自然处理遮挡**: 点落在低概率区域时损失增大
4. **梯度平滑**: 相对于 ICP 的硬最近邻，GMM 提供更平滑的梯度

#### 6.2.2 挑战

1. **多模态**: 损失函数可能有多个局部最优
   - **缓解**: 多尺度优化或 good initialization

2. **数值稳定性**: log-sum-exp 中的指数可能溢出
   - **缓解**: 使用 `logsumexp` 技巧，减去最大值

3. **Top-K 截断误差**: 忽略远处高斯球的贡献
   - **缓解**: 选择足够大的 $K$ (如 8-16)

4. **各向异性影响**: 高斯球的形状影响概率密度
   - **缓解**: 使用完整的协方差信息

---

## 7. 实现细节

### 7.1 数值稳定性

```python
def stable_nll_loss(log_densities, weights):
    """
    log_densities: [N, K] log g_i(p'_j)
    weights: [N, K] mixture weights
    """
    # Numerically stable log-sum-exp
    log_weighted = log_densities + torch.log(weights)
    loss = -torch.logsumexp(log_weighted, dim=-1)
    return loss.mean()
```

### 7.2 变换参数化

```python
# 使用李代数 se(3) 表示
xi = torch.zeros(6, requires_grad=True)  # [tx, ty, tz, wx, wy, wz]
T = se3_exp(xi)  # 指数映射到 SE(3)

# 或使用四元数 + 平移
quat = torch.zeros(4)  # [w, x, y, z]
trans = torch.zeros(3)  # [tx, ty, tz]
```

### 7.3 优化策略

1. **粗到精**: 从 large voxel 查询开始，逐步细化
2. **随机重启**: 多个初始变换，选择最优结果
3. **与 ICP 结合**: 先用 ICP 粗略对齐，再用 MLE 细化

---

## 8. 总结

### 8.1 可行性结论

**数学上完全可行**。MLE 配准有以下特点:

- ✅ 理论基础扎实 (GMM, MLE)
- ✅ 利用 3DGS 的各向异性信息
- ✅ 软对应更鲁棒
- ✅ 梯度可微，可用 PyTorch 自动求导
- ⚠️ 需要良好的初始化避免局部最优
- ⚠️ Top-K 近似引入小误差，但可控

### 8.2 推荐实现

1. **Soft Assignment + Top-K (K=8)**
2. **使用 logsumexp 保证数值稳定**
3. **结合旋转参数化 (四元数或李代数)**
4. **多尺度或 good initialization 策略**

### 8.3 与计划文档的关联

本文档是 Phase 3 (MLE Registration Optimization) 的数学基础。根据上述分析:

- `sphere_mle_loss.py` 应该实现公式 (7) 的近似损失
- `MLEAlignmentLoss` 需要支持 soft/hard assignment 选项
- 梯度计算使用 PyTorch 自动微分，无需手动推导

---

## 参考文献

1. Myronenko & Song (2010). "Point Set Registration: Coherent Point Drift"
2. Eckart et al. (2018). "HGMR: Hierarchical Gaussian Mixtures for Adaptive 3D Registration"
3. Kerbl et al. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
4. Stoyanov et al. (2012). "Fast and Accurate Scan Registration through Minimization of the Distance between Compact 3D NDT Representations"

---

**文档结束**
