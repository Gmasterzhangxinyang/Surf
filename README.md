# Active BEVFusion Agent - 主动感知BEV融合系统

基于 BEVFusion 的主动感知 Agent 系统，通过图像优化工具提升雨夜、低光照等恶劣条件下的 BEV 分割质量。

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [核心模块详解](#核心模块详解)
- [实验流程](#实验流程)
- [工具箱](#工具箱)
- [可视化输出](#可视化输出)
- [命令行工具](#命令行工具)
- [配置说明](#配置说明)

---

## 项目概述

### 目标

在雨夜、强反光、低光照等恶劣视觉条件下，通过主动感知 Agent 选择合适的图像优化工具，提升 BEVFusion 的 BEV 分割质量（mIoU 和 Pixel Accuracy）。

### 核心流程

```
输入: 6路相机图像 + LiDAR 点云
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 1: Baseline BEVFusion        │
│  生成初始 BEV 分割图                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 2: BEV Quality Evaluation    │
│  评估 edge_density, integrity,      │
│  problem_coords                     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 3: Agent Decision            │
│  规则匹配 / VisionLLM 分析           │
│  选择合适的图像优化工具               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 4: Image Refinement          │
│  对选中相机应用图像优化工具            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 5: Re-run BEVFusion          │
│  重新生成优化后的 BEV 分割图          │
└─────────────────────────────────────┘
    │
    ▼
输出: 优化后的 BEV 分割图 + 评估报告
```

---

## 系统架构

### 目录结构

```
bevfusion/
├── agent/                      # Agent 模块
│   ├── core.py                 # ReAct 引擎，Agent 主控逻辑
│   ├── bev_evaluator.py        # BEV 质量评估 + 几何映射
│   ├── refiner.py              # 图像优化工具执行器
│   ├── vision_llm.py           # Qwen2.5-VL 视觉分析接口
│   ├── functions.py            # Function Calling 定义
│   ├── prompts.py              # 提示词模板
│   └── data_logger.py          # 训练数据记录
│
├── models/                     # BEVFusion 模型
│   ├── bevfusion.py           # 主模型 (融合 + 分割)
│   ├── camera_encoder.py      # 相机分支 (ResNet50 + FPN + LSS)
│   ├── lidar_encoder.py       # LiDAR 分支 (PointPillars)
│   ├── fusion.py              # 融合模块 (ConvFuser)
│   └── heads.py               # 分割头
│
├── data/                       # 数据处理
│   ├── nuscenes_loader.py     # nuScenes 数据加载
│   └── bev_gt.py              # BEV Ground Truth 生成
│
├── utils/                      # 工具函数
│   ├── geometry.py           # 几何变换
│   ├── ogm.py                 # OGM (Occupancy Grid Map) 生成
│   └── visualize.py           # 可视化工具
│
├── experiments/                # 实验模块
│   ├── tool_configs.py        # 工具配置 (TOOL_CONFIGS, COMBO_CONFIGS)
│   ├── image_quality.py       # 图像质量评估
│   └── visualize_ablation.py  # Ablation 可视化
│
├── config.py                  # BEV 配置 (BEVConfig)
├── train.py                   # 训练脚本
├── tool_ablation.py           # 工具消融实验
├── summarize_ablation.py       # 消融结果汇总
├── oracle_policy.py            # Oracle 上界实验
├── agent_policy_eval.py        # Agent 策略评估
├── bev_comparison.py           # Baseline vs Agent 对比
├── run_inference.py           # 基础推理入口
└── README.md                  # 本文档
```

### 数据流

```
                    nuScenes Dataset
                         │
                         ▼
              ┌─────────────────────┐
              │  NuScenesLoader     │
              │  6x相机图像 + LiDAR  │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    BEVFusion        │
              │  Camera Encoder     │
              │  LiDAR Encoder      │
              │  Fusion Module      │
              │  Segmentation Head  │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   BEV Segmentation   │
              │      (120x120)       │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   BEVEvaluator      │
              │  - mIoU / Accuracy  │
              │  - edge_density     │
              │  - integrity        │
              │  - problem_coords   │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    ImageRefiner     │
              │  - enhance_image    │
              │  - remove_rain      │
              │  - dehaze           │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Re-run BEVFusion │
              │   (迭代 1-3 次)     │
              └─────────────────────┘
```

---

## 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 安装依赖
pip install torch torchvision
pip install numpy opencv-python pyquaternion scikit-image matplotlib
pip install ollama  # 仅标准模式需要
```

### 2. 数据准备

下载 nuScenes v1.0-mini 数据集：
```bash
# 从 https://www.nuscenes.org/nuscenes#download 下载
# 解压到项目根目录的 v1.0-mini 文件夹
```

### 3. 运行基础推理

```bash
# 使用 CPU
python run_inference.py --dataroot ./v1.0-mini --sample_idx 0

# 使用 MPS (Apple Silicon)
python run_inference.py --dataroot ./v1.0-mini --sample_idx 0 --device mps
```

### 4. 运行工具消融实验 (Tool Ablation)

```bash
# 雨夜场景实验 (scene-1094)
python tool_ablation.py \
  --dataroot ./v1.0-mini \
  --scenes scene-1094 \
  --tools all \
  --num_samples 40 \
  --output results/scene1094_tool_ablation.jsonl \
  --save_vis \
  --model_path ../best_model\(2\).pth

# 指定工具测试
python tool_ablation.py \
  --scenes scene-1094 \
  --tools gamma_0.8,contrast_1.2,clahe \
  --num_samples 10 \
  --output results/debug_tool_ablation.jsonl \
  --save_vis

# 夜间辅助场景
python tool_ablation.py \
  --scenes scene-1077,scene-1100 \
  --tools all \
  --num_samples 80 \
  --output results/night_tool_ablation.jsonl

# 晴天对照场景
python tool_ablation.py \
  --scenes scene-0061,scene-0103 \
  --tools all \
  --num_samples 80 \
  --output results/sunny_tool_ablation.jsonl
```

### 5. 汇总消融实验结果

```bash
python summarize_ablation.py \
  --input results/scene1094_tool_ablation.jsonl \
  --output_csv results/scene1094_tool_ablation_summary.csv \
  --output_md results/scene1094_tool_ablation_summary.md
```

### 6. 运行 Oracle 上界实验

```bash
python oracle_policy.py \
  --input results/scene1094_tool_ablation.jsonl \
  --output_csv results/scene1094_oracle_summary.csv \
  --output_md results/scene1094_oracle_summary.md \
  --output_jsonl results/scene1094_oracle_selection.jsonl
```

### 7. 运行 Agent 策略评估

```bash
# Fast 模式 (纯规则，无需 Ollama)
python agent_policy_eval.py \
  --agent rule \
  --scenes scene-1094 \
  --num_samples 40 \
  --output results/scene1094_rule_agent.jsonl \
  --save_vis

# VisionLLM 模式 (需要 Ollama + Qwen2.5-VL)
ollama serve
ollama pull qwen2.5-vl:7b

python agent_policy_eval.py \
  --agent vision_llm \
  --scenes scene-1094 \
  --num_samples 40 \
  --output results/scene1094_vision_agent.jsonl \
  --save_vis
```

---

## 核心模块详解

### 1. BEVFusion 模型 (`models/bevfusion.py`)

融合相机和 LiDAR 信息的 BEV 分割模型。

**输入:**
- `images`: (B, 6, 3, 128, 352) - 6个相机的RGB图像
- `intrinsics`: (B, 6, 3, 3) - 相机内参
- `extrinsics`: (B, 6, 4, 4) - 相机外参 (cam2ego)
- `lidar_points`: (B, N, 5) - LiDAR 点云 (x,y,z,intensity,ring)
- `lidar_mask`: (B, N) - 有效点云掩码

**输出:**
- `logits`: (B, num_classes, 120, 120) - 分割 logits
- `bev_seg`: (B, 120, 120) - 分割结果 (类别ID)

### 2. BEV 评估器 (`agent/bev_evaluator.py`)

评估 BEV 分割质量，包含与 Ground Truth 的比较和无需 GT 的质量评估。

**评估指标:**

| 指标 | 说明 | 阈值 |
|------|------|------|
| `iou` | mIoU (二值: 占用vs空闲) | 越高越好 |
| `accuracy` | Pixel Accuracy | 越高越好 |
| `edge_density` | 边缘密度 | < 0.05 表示碎片化 |
| `integrity` | 物体完整性 | < 0.5 表示需要优化 |
| `problem_coords` | 问题区域中心 | 用于相机映射 |

**几何映射:**
`bev_to_camera_mapping()` 将 BEV 问题区域映射到对应的相机 ID。

### 3. 图像优化器 (`agent/refiner.py`)

执行各种图像优化工具。

**工具类型:**

| 方法 | 功能 | 适用场景 |
|------|------|----------|
| `enhance_image` | 对比度/锐化/Gamma/降噪 | 低光照、模糊 |
| `remove_rain` | 去雨 (CLAHE/高斯/导向滤波) | 雨天 |
| `dehaze` | 去雾 (CLAHE/直方图均衡) | 雾霾 |

### 4. Agent 核心 (`agent/core.py`)

ReAct 循环引擎。

```python
# 伪代码
for iteration in range(max_iterations):
    1. Evaluate BEV quality
    2. If quality OK → finalize
    3. Analyze visual issues (VisionLLM or rules)
    4. Select action (tool + cameras)
    5. Execute refinement
    6. Re-run BEVFusion
```

---

## 实验流程

### 阶段 1: Baseline

运行原始 BEVFusion，建立每个场景的基线指标。

```bash
python tool_ablation.py --scenes scene-1094 --num_samples 40 --output baseline.jsonl
```

### 阶段 2: 单工具消融

测试每个工具的独立效果。

```bash
python tool_ablation.py --scenes scene-1094 --tools all --num_samples 40 --save_vis
```

**评估标准:**
- mean ΔmIoU > 0: 工具有效
- improved_ratio > 55%: 工具稳定有效
- 晴天场景下降明显: 工具破坏分布

### 阶段 3: 工具组合消融

测试多步工具组合。

```bash
# 组合已在 tool_configs.py 的 COMBO_CONFIGS 中定义
# 会自动测试所有组合
```

### 阶段 4: Oracle 上界

对每个 sample 选择最优工具，计算理论上限。

```bash
python oracle_policy.py --input results/scene1094_tool_ablation.jsonl
```

### 阶段 5: Rule Agent

基于规则的 Agent，不使用 LLM。

```bash
python agent_policy_eval.py --agent rule --scenes scene-1094 --save_vis
```

**规则示例:**
```python
if scene == "scene-1094":
    if highlight_ratio > 0.12:
        return "gamma_1.2_clahe"  # 强反光
    elif brightness < 60:
        return "gamma_0.8_clahe"  # 低亮度
```

### 阶段 6: VisionLLM Agent

使用 Qwen2.5-VL 分析视觉问题，规则选择工具。

```bash
python agent_policy_eval.py --agent vision_llm --scenes scene-1094 --save_vis
```

---

## 工具箱

### 单工具配置 (TOOL_CONFIGS)

| 方法名 | 工具 | 参数 | 说明 |
|--------|------|------|------|
| `contrast_1.1` | enhance_image | mode=contrast, factor=1.1 | 轻度对比度增强 |
| `contrast_1.2` | enhance_image | mode=contrast, factor=1.2 | 中度对比度增强 |
| `contrast_1.3` | enhance_image | mode=contrast, factor=1.3 | 强度对比度增强 |
| `gamma_0.6` | enhance_image | mode=gamma, gamma=0.6 | 强力亮度提升 |
| `gamma_0.8` | enhance_image | mode=gamma, gamma=0.8 | 中度亮度提升 |
| `gamma_1.2` | enhance_image | mode=gamma, gamma=1.2 | 轻度亮度压制 |
| `gamma_1.4` | enhance_image | mode=gamma, gamma=1.4 | 强力亮度压制 |
| `sharpen_0.3` | enhance_image | mode=sharpen, strength=0.3 | 轻度锐化 |
| `sharpen_0.5` | enhance_image | mode=sharpen, strength=0.5 | 中度锐化 |
| `denoise_3` | enhance_image | mode=denoise, strength=3 | 轻度降噪 |
| `denoise_5` | enhance_image | mode=denoise, strength=5 | 中度降噪 |
| `denoise_7` | enhance_image | mode=denoise, strength=7 | 强度降噪 |
| `clahe` | remove_rain | mode=clahe | CLAHE 对比度增强 |
| `gaussian_refine` | remove_rain | mode=gaussian | 高斯模糊 |
| `hist_equalization` | dehaze | mode=hist_equalization | 直方图均衡化 |
| `dehaze_clahe` | dehaze | mode=clahe | CLAHE 去雾 |
| `low_light_gamma_0.5` | enhance_image | mode=gamma, gamma=0.5 | 低光增强 |
| `low_light_clahe` | enhance_image | mode=low_light | 低光增强组合 |
| `derain_guided` | remove_rain | mode=guided | 导向滤波去雨 |
| `derain_bilateral` | remove_rain | mode=bilateral | 双边滤波去雨 |
| `dehaze_adaptive` | dehaze | mode=adaptive | 自适应去雾 |
| `dehaze_retinex` | dehaze | mode=retinex | Retinex 去雾 |

### 工具组合 (COMBO_CONFIGS)

| 方法名 | 步骤 | 说明 |
|--------|------|------|
| `gamma_0.8_contrast_1.1` | gamma(0.8) → contrast(1.1) | 亮度+对比度组合 |
| `gamma_0.8_clahe` | gamma(0.8) → clahe | 亮度+CLAHE组合 |
| `gamma_1.2_clahe` | gamma(1.2) → clahe | 压制+CLAHE组合 |
| `denoise_5_gamma_0.8` | denoise(5) → gamma(0.8) | 降噪+亮度组合 |
| `denoise_5_clahe` | denoise(5) → clahe | 降噪+CLAHE组合 |
| `clahe_sharpen_0.3` | clahe → sharpen(0.3) | CLAHE+锐化组合 |

---

## 可视化输出

### 14图大图布局 (tool_ablation 可视化)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Tool: gamma_0.8_clahe  │  Baseline mIoU: 0.032  │  Refined mIoU: 0.041  │
│  Enhanced Cameras: ALL  │  ΔmIoU: +0.009         │  Improved: ✓          │
├─────────────────┬─────────────────┬─────────────────┬────────────────────┤
│  CAM_FRONT      │ CAM_FRONT_RIGHT │ CAM_FRONT_LEFT  │ CAM_BACK           │
│  (Original)     │ (Original)      │ (Original)      │ (Original)         │
│                 │                 │                 │                    │
│    [Image]      │    [Image]      │    [Image]      │    [Image]         │
│                 │                 │                 │                    │
├─────────────────┼─────────────────┼─────────────────┼────────────────────┤
│  CAM_FRONT      │ CAM_FRONT_RIGHT │ CAM_FRONT_LEFT  │ CAM_BACK           │
│  (Refined)      │ (Refined)       │ (Refined)       │ (Refined)          │
│                 │                 │                 │                    │
│    [Image]      │    [Image]      │    [Image]      │    [Image]         │
│                 │                 │                 │                    │
├─────────────────┴─────────────────┴─────────────────┴────────────────────┤
│  CAM_BACK_LEFT (Original)          │  CAM_BACK_RIGHT (Original)          │
│                [Image]            │                [Image]              │
├────────────────────────────────────┼─────────────────────────────────────┤
│  CAM_BACK_LEFT (Refined)           │  CAM_BACK_RIGHT (Refined)          │
│                [Image]            │                [Image]              │
├────────────────────────────────────┴─────────────────────────────────────┤
│          Baseline BEV                    │          Refined BEV            │
│                                        │                                 │
│            [BEV Image]                 │          [BEV Image]            │
│                                        │                                 │
└────────────────────────────────────────┴─────────────────────────────────┘
```

**说明:**
- 顶部: 工具名称、IoU 变化、是否改进
- 第一行: 6个相机的原始图像 (Original)
- 第二行: 6个相机的优化后图像 (Refined)
- 第三行: Baseline BEV 分割图 vs Refined BEV 分割图

---

## 命令行工具

### tool_ablation.py - 工具消融实验

```bash
python tool_ablation.py \
  --dataroot ./v1.0-mini              # 数据集路径
  --version v1.0-mini                  # 数据集版本
  --scenes scene-1094                   # 场景 (逗号分隔或 "all")
  --tools all                          # 工具 (逗号分隔或 "all")
  --num_samples 40                      # 每场景样本数
  --output results/ablation.jsonl      # 输出 JSONL 路径
  --save_vis                           # 保存可视化
  --model_path ../best_model\(2\).pth   # 模型权重路径
  --seed 42                            # 随机种子
```

### summarize_ablation.py - 结果汇总

```bash
python summarize_ablation.py \
  --input results/scene1094_tool_ablation.jsonl \
  --output_csv results/summary.csv \
  --output_md results/summary.md
```

### oracle_policy.py - Oracle 上界

```bash
python oracle_policy.py \
  --input results/scene1094_tool_ablation.jsonl \
  --output_csv results/oracle_summary.csv \
  --output_md results/oracle_summary.md \
  --output_jsonl results/oracle_selection.jsonl
```

### agent_policy_eval.py - Agent 评估

```bash
python agent_policy_eval.py \
  --agent rule|vision_llm             # Agent 类型
  --scenes scene-1094                  # 场景
  --num_samples 40                     # 样本数
  --output results/agent.jsonl        # 输出路径
  --save_vis                          # 保存可视化
```

---

## 配置说明

### BEVConfig (`config.py`)

```python
@dataclass
class BEVConfig:
    # BEV 网格
    bev_x_range: Tuple[float, float] = (-30.0, 30.0)  # 米
    bev_y_range: Tuple[float, float] = (-30.0, 30.0)
    bev_resolution: float = 0.5  # 米/像素
    bev_size: Tuple[int, int] = (120, 120)  # H, W

    # 深度范围
    depth_min: float = 1.0
    depth_max: float = 40.0
    depth_bins: int = 59

    # 相机配置
    image_size: Tuple[int, int] = (128, 352)  # H, W
    cam_channels: int = 64
    num_cameras: int = 6

    # LiDAR 配置
    point_cloud_range: List[float] = [-30.0, -30.0, -5.0, 30.0, 30.0, 3.0]
    pillar_size: List[float] = [0.5, 0.5, 8.0]
    max_pillars: int = 20000
    max_points_per_pillar: int = 32

    # 模型配置
    fused_channels: int = 128
    num_classes: int = 6  # background, vehicle, pedestrian, biological, building, other
```

### 场景配置 (`experiments/tool_configs.py`)

```python
SCENE_CONDITIONS = {
    "scene-0061": "sunny",
    "scene-0103": "sunny",
    "scene-1077": "night",
    "scene-1094": "night_after_rain",  # 主实验场景
    "scene-1100": "night",
}
```

---

## 结果分析

### 判断标准

**可以继续做 Agent 的条件 (满足任意两个):**
1. scene-1094 上至少一个工具 mean ΔmIoU > 0
2. scene-1094 上至少一个工具 improved_ratio > 55%
3. Oracle ΔmIoU 明显高于 baseline
4. Rule Agent 能达到 Oracle 提升的 50% 以上
5. Sunny scene 上增强不会造成明显性能下降

**不建议继续的情况:**
1. 所有工具在 scene-1094 上都使 mIoU 下降
2. Oracle 相比 baseline 几乎没有提升
3. 工具提升完全随机，没有稳定 pattern
4. 晴天和雨夜都被增强伤害

### 输出结论格式

```
1. 在 scene-1094 上，表现最好的单工具是：XXX
2. mean ΔmIoU = XXX
3. mean Δaccuracy = XXX
4. improved ratio = XXX%
5. 表现最好的组合工具是：XXX
6. Oracle mean ΔmIoU = XXX
7. Rule Agent mean ΔmIoU = XXX
8. Rule Agent 达到 Oracle 提升的 XXX%
9. Sunny scenes 上是否出现明显下降：是/否
10. 是否值得继续做 VisionLLM Agent：是/否
```

---

## 常见问题

### Q: MPS 不可用
```bash
# 检查 MPS 支持
python -c "import torch; print(torch.backends.mps.is_available())"

# 降级到 CPU
python tool_ablation.py ... --device cpu
```

### Q: Ollama 连接失败
```bash
# 启动 Ollama
ollama serve

# 拉取模型
ollama pull qwen2.5-vl:7b

# 检查模型列表
ollama list
```

### Q: 内存不足
```bash
# 减少样本数
python tool_ablation.py ... --num_samples 20

# 使用 CPU
python tool_ablation.py ... --device cpu
```
