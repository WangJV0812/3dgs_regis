# Replica 数据集实验计划

## 1. 数据集结构

原始数据集位于 `/home/wangjv_wsl/data/RGBD/replica`：

```
/home/wangjv_wsl/data/RGBD/replica/
├── office0/results
├── office1/results
├── office2/results
├── office3/results
├── office4/results
├── room0/results
├── room1/results
└── room2/results
```

其中 `results/` 目录包含实验所需的图片帧。

## 2. 实验目标

对每个场景 (`office0`–`office4`, `room0`–`room2`)，使用多种随机种子运行 VGGT 重建和 HIER 3DGS 训练，以评估方法稳定性。

- **随机种子列表**: `[42, 100, 123, 125, 128, 236]`
- `--max_frames 50` 作为默认配置（控制显存占用）

## 3. 工作目录与环境

```bash
cd /home/wangjv_wsl/code/3dgs/3dgs_regis
conda activate gs_reg
```

## 4. 单场景准备

以 `office1` 为例（`office0` 已完成，见第 6 节兼容性说明）。

### 4.1 创建场景目录并复制图片

```bash
SCENE=office1
SRC=/home/wangjv_wsl/data/RGBD/replica/${SCENE}/results
DST=/home/wangjv_wsl/data/3dgs_dataset/geosplate_dataset/replica/${SCENE}

mkdir -p ${DST}/input/images
rsync -av ${SRC}/ ${DST}/input/images/
```

复制完成后，目录结构应为：

```
/home/wangjv_wsl/data/3dgs_dataset/geosplate_dataset/replica/office1/
└── input/images/
    └── frame_0000.jpg
    └── frame_0001.jpg
    └── ...
```

## 5. 实验流程（以单场景单种子为例）

### 5.1 运行 COLMAP（可选，但建议执行）

```bash
python dataset_prepare/01_run_colmap.py \
    --scene_dir /home/wangjv_wsl/data/3dgs_dataset/geosplate_dataset/replica/office1
```

### 5.2 运行 VGGT（按种子隔离输出）

```bash
SEED=42
python dataset_prepare/02_run_vggt.py \
    --scene_dir /home/wangjv_wsl/data/3dgs_dataset/geosplate_dataset/replica/office1 \
    --max_frames 50 \
    --seed ${SEED}
```

输出目录：

```
office1/vggt_result/42/
├── sparse/
│   ├── cameras.bin
│   ├── images.bin
│   ├── points3D.bin
│   └── points.ply
├── intermediate/
│   ├── tokens.pt
│   ├── depth_maps.npy
│   ├── point_maps.npy
│   └── metadata.json
└── depth_vis/
    └── ...
```

### 5.3 训练 HIER 3DGS（读取对应 seed 的 VGGT 结果）

```bash
SEED=42
python dataset_prepare/03_train_hier.py \
    --scene_dir /home/wangjv_wsl/data/3dgs_dataset/geosplate_dataset/replica/office1 \
    --source vggt \
    --seed ${SEED} \
    --build_hierarchy \
    --post_opt
```

输出目录：

```
office1/3dgs_result/42/
├── log/
└── model/
    ├── ckpt/
    ├── point_cloud/iteration_7000/
    ├── point_cloud/iteration_15000/
    ├── point_cloud/iteration_30000/
    ├── hierarchy.hier
    ├── hierarchy.hier_opt
    └── final.hier
```

### 5.4 完整单场景实验脚本

```bash
SCENE=office1
SCENE_DIR=/home/wangjv_wsl/data/3dgs_dataset/geosplate_dataset/replica/${SCENE}

# Step 1: COLMAP
python dataset_prepare/01_run_colmap.py --scene_dir ${SCENE_DIR}

# Step 2 & 3: VGGT + HIER for each seed
for SEED in 42 100 123 125 128 236; do
    echo "======================================"
    echo "Processing ${SCENE} with seed ${SEED}"
    echo "======================================"

    python dataset_prepare/02_run_vggt.py \
        --scene_dir ${SCENE_DIR} \
        --max_frames 50 \
        --seed ${SEED}

    python dataset_prepare/03_train_hier.py \
        --scene_dir ${SCENE_DIR} \
        --source vggt \
        --seed ${SEED} \
        --build_hierarchy \
        --post_opt
done
```

## 6. `office0` 兼容性说明

`office0` 此前已完成实验，旧结构为：

```
office0/vggt_result/          # 无 seed 子目录
office0/3dgs_result/          # 无 seed 子目录
```

如需继续使用旧结果，必须手动迁移：

```bash
OLDDIR=/home/wangjv_wsl/data/3dgs_dataset/geosplate_dataset/replica/office0

# 迁移 VGGT 结果
mkdir -p ${OLDDIR}/vggt_result/42
mv ${OLDDIR}/vggt_result/sparse      ${OLDDIR}/vggt_result/42/
mv ${OLDDIR}/vggt_result/intermediate ${OLDDIR}/vggt_result/42/
mv ${OLDDIR}/vggt_result/depth_vis    ${OLDDIR}/vggt_result/42/

# 迁移 3DGS 结果
mkdir -p ${OLDDIR}/3dgs_result/42
mv ${OLDDIR}/3dgs_result/log         ${OLDDIR}/3dgs_result/42/
mv ${OLDDIR}/3dgs_result/model       ${OLDDIR}/3dgs_result/42/
mv ${OLDDIR}/3dgs_result/config.json ${OLDDIR}/3dgs_result/42/
```

迁移后，`03_train_hier.py --seed 42` 即可正常读取。

## 7. 批量执行所有场景

```bash
#!/bin/bash
SEEDS=(42 100 123 125 128 236)
SCENES=(office1 office2 office3 office4 room0 room1 room2)
BASE_DIR=/home/wangjv_wsl/data/3dgs_dataset/geosplate_dataset/replica

cd /home/wangjv_wsl/code/3dgs/3dgs_regis
conda activate gs_reg

for SCENE in "${SCENES[@]}"; do
    SCENE_DIR=${BASE_DIR}/${SCENE}

    # 若为首次运行该场景，先复制图片
    if [ ! -d "${SCENE_DIR}/input/images" ]; then
        mkdir -p ${SCENE_DIR}/input/images
        rsync -av /home/wangjv_wsl/data/RGBD/replica/${SCENE}/results/ ${SCENE_DIR}/input/images/
    fi

    # 运行 COLMAP（若不启用可注释掉）
    python dataset_prepare/01_run_colmap.py --scene_dir ${SCENE_DIR}

    for SEED in "${SEEDS[@]}"; do
        echo "======================================"
        echo "Processing ${SCENE} with seed ${SEED}"
        echo "======================================"

        python dataset_prepare/02_run_vggt.py \
            --scene_dir ${SCENE_DIR} \
            --max_frames 50 \
            --seed ${SEED}

        python dataset_prepare/03_train_hier.py \
            --scene_dir ${SCENE_DIR} \
            --source vggt \
            --seed ${SEED} \
            --build_hierarchy \
            --post_opt
    done
done
```

## 8. `--max_frames` VRAM 建议

| GPU 显存 | 建议 `--max_frames` |
|---------|--------------------|
| 8 GB    | 40–50              |
| 12 GB   | 60–80              |
| 16 GB   | 80–100             |
| 24 GB+  | 100–150            |

如需处理完整序列，可省略 `--max_frames`。

## 9. 最终完整目录结构示例

以 `office1` 为例，完成全部 6 个种子的实验后：

```
/home/wangjv_wsl/data/3dgs_dataset/geosplate_dataset/replica/office1/
├── input/
│   └── images/
│       └── frame_*.jpg
├── colmap_result/
│   └── sparse/
│       └── 0/
├── vggt_result/
│   ├── 42/
│   │   ├── sparse/
│   │   ├── intermediate/
│   │   └── depth_vis/
│   ├── 100/
│   │   ├── sparse/
│   │   ├── intermediate/
│   │   └── depth_vis/
│   ├── 123/
│   ├── 125/
│   ├── 128/
│   └── 236/
└── 3dgs_result/
    ├── 42/
    │   ├── log/
    │   └── model/
    │       ├── point_cloud/
    │       ├── hierarchy.hier
    │       ├── hierarchy.hier_opt
    │       └── final.hier
    ├── 100/
    ├── 123/
    ├── 125/
    ├── 128/
    └── 236/
```

## 10. 关键脚本清单

| 脚本 | 作用 |
|------|------|
| `dataset_prepare/01_run_colmap.py` | COLMAP 稀疏重建 |
| `dataset_prepare/02_run_vggt.py` | VGGT 位姿估计与点云生成（输出到 `vggt_result/<seed>/`） |
| `dataset_prepare/03_train_hier.py` | HIER 3DGS 训练（从 `vggt_result/<seed>/` 读取，写入 `3dgs_result/<seed>/`） |
