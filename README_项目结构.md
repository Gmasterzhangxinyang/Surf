# BEVFusion Mac 复现版 — 项目结构说明

## 一、目录结构

```
bevfusion/
├── config.py                 # 统一配置：BEV网格参数、相机参数、LiDAR参数等
├── train.py                  # 训练入口脚本
├── run_inference.py          # 完整 BEVFusion 推理入口
├── run_ogm.py                # OGM 占用栅格图生成（无需训练）
├── requirements.txt         # 依赖列表
├── 项目进展汇报.md           # 项目进展文档
├── full_pipeline.png         # 推理输出：完整 pipeline 可视化
│
├── models/                   # 模型结构（纯 PyTorch，无 CUDA 依赖）
│   ├── __init__.py
│   ├── bevfusion.py          # 主模型：组装 camera + lidar + fusion + head
│   ├── camera_encoder.py     # 相机分支：ResNet-50 + FPN + LSS 视角变换
│   ├── lidar_encoder.py      # LiDAR 分支：PointPillars 编码器
│   ├── fusion.py             # 融合模块：ConvFuser（cat + conv）
│   └── heads.py              # BEV 分割头
│
├── data/                     # 数据处理
│   ├── __init__.py
│   ├── nuscenes_loader.py    # nuScenes 数据加载器（6路相机 + LiDAR + 标定）
│   └── bev_gt.py             # BEV 地面真值生成（从3D框投影到BEV网格）
│
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── geometry.py           # 几何变换：create_frustum / frustum_to_world / points_to_bev_indices
│   ├── ogm.py                # OGM 生成：直接投影 / 概率 / 高度图 + 可视化
│   └── visualize.py          # BEV 结果可视化
│
├── v1.0-mini/                # nuScenes mini 数据集（本地副本）
│
├── output/                   # 推理输出目录
│   ├── full_pipeline.png
│   └── bev_segmentation.png
│
├── ogm_output/               # OGM 输出目录（5个样本）
│   ├── overview_*.png        # 概览图（6路相机 + LiDAR俯视 + OGM）
│   ├── ogm_sample_*.png      # OGM 详细图（概率 + 点数热力 + 高度）
│   └── ogm_binary_*.png      # 二值 OGM（黑=占用，白=空闲）
│
├── train_output/             # 训练输出
├── inference_output/         # 推理输出（另一版本）
└── output_real/               # 真实推理输出
```

---

## 二、核心实现说明

### 2.1 模型架构

```
摄像头图像 (6路)              LiDAR 点云
      ↓                           ↓
 ResNet-50 + FPN              PointPillars
 (pretrained, stride 8)         pillarize
      ↓                           ↓
 LSS View Transform           Pillar Scatter
 (depth → 3D volume           (pillars → BEV伪图)
  → scatter_add 到 BEV)
      ↓                           ↓
 Camera BEV 特征  ←—ConvFuser—→  LiDAR BEV 特征
                    ↓
               BEV 分割头
                    ↓
            (B, num_classes, H, W)
```

**关键设计决策：去 CUDA 化**

| 模块 | 原版 BEVFusion | 本实现 |
|------|----------------|--------|
| View Transform | LSS + CUDA `bev_pool` | LSS + `scatter_add`（纯 PyTorch） |
| LiDAR Encoder | VoxelNet + `spconv` | PointPillars（纯 PyTorch） |
| 所有算子 | CUDA 专用 | CPU / MPS 通用 |

### 2.2 各模块详解

#### `camera_encoder.py`

相机分支将 2D 图像特征"提升"到 3D 再投影到 BEV：

1. **ResNet-50 backbone** — 提取多尺度特征（C2/3/4/5）
2. **FPN** — 融合 C2(512ch)、C3(1024ch)、C4(2048ch) → 64ch
3. **LSS View Transform**（核心）：
   - `DepthNet`：预测每个像素的深度分布（39 个离散去散桶）
   - 外积 `feature * depth_prob` → 3D 特征体积 (C, D, fH, fW)
   - `frustum_to_world` 将每个 frustum 点从图像坐标经相机内参/外参变换到自车坐标系
   - `points_to_bev_indices` 计算每个 3D 点落在哪个 BEV 网格
   - `scatter_add` 将特征累加到对应的 BEV 格子

#### `lidar_encoder.py`

将稀疏点云转换为密集 BEV 伪图：

1. **`pillarize`**：将点云分配到pillar网格（0.5m × 0.5m），每 pillar 最多 32 点
2. **PillarFeatureNet**：每点提取 10 维特征 `[x,y,z,intensity,ring, x-cx,y-cy,z-cz, grid_x, grid_y]`，PointNet 风格 max-pooling 得pillar特征
3. **PointPillarsScatter**：`scatter_add` 将pillar特征放到对应的2D网格位置
4. **BEVBackbone2D**：3层Conv2D处理BEV伪图

#### `fusion.py`

`ConvFuser`：Camera BEV + LiDAR BEV 做 concat，再过两层 Conv-BN-ReLU。

#### `heads.py`

`BEVSegHead`：两层 Conv，最后 1×1 conv 输出 `num_classes` 通道（默认 2 类：空闲/占用）。

---

## 三、数据流程

### 训练

```
nuScenes 数据
    ↓
 nuscenes_loader.py
    ↓ (images, intrinsics, extrinsics, lidar_points, lidar_mask)
 BEVFusion.forward()
    ↓ (cam_bev, lidar_bev, fused, logits)
 bev_gt.py generate_bev_gt() → BEV 语义 mask
    ↓
 CrossEntropyLoss(logits, gt)
    ↓
 Backprop + Adam
```

### OGM 生成（无需训练）

```
LiDAR 点云
    ↓
 lidar_to_ogm() / lidar_to_ogm_probabilistic() / lidar_to_height_map()
    ↓
 可选：BEVFusion 网络输出作 learned OGM
    ↓
 visualize_ogm() → PNG
```

---

## 四、运行方式

```bash
conda activate bevfusion
cd bevfusion

# 生成 OGM（立即可用，无需训练）
python run_ogm.py --dataroot v1.0-mini --range 30 --resolution 0.4

# 完整 BEVFusion 推理
python run_inference.py --dataroot v1.0-mini

# 训练（需 nuScenes 数据）
python train.py --dataroot v1.0-mini --epochs 10
```

---

## 五、配置参数（config.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| bev_x_range | (-30, 30) m | BEV 范围 |
| bev_size | (120, 120) | 网格 H×W |
| bev_resolution | 0.5 m | 每格代表实际距离 |
| depth_bins | 39 | LSS 深度散散数 |
| cam_channels | 64 | 相机 BEV 特征维数 |
| lidar_channels | 64 | LiDAR BEV 特征维数 |
| fused_channels | 128 | 融合后特征维数 |
| num_classes | 2 | 0=空闲，1=占用 |

---

## 六、环境

- **硬件**：Mac Apple Silicon（无 CUDA）
- **Python**：3.10，PyTorch 2.0+，支持 MPS 加速
- **数据**：nuScenes v1.0-mini（10 场景，404 样本）