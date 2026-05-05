# Active LLM Agent for BEVFusion 

基于 BEVFusion 的主动感知 Agent 系统，通过视觉语言模型和 ReAct 循环优化 BEV 分割质量。


## 功能特点

- **ReAct 决策循环**: 评估 → 视觉分析 → 决策 → 执行
- **视觉语言模型**: Qwen2.5-VL 分析相机图像，检测天气/光照问题
- **BEV→相机几何映射**: 精确定位问题区域对应的相机
- **图像优化工具**: 去雨、去雾、对比度增强、裁剪放大

## 项目结构

```
bevfusion/
├── agent/                      # Agent 模块
│   ├── core.py                # ReAct 引擎，Agent 主控逻辑
│   ├── bev_evaluator.py       # BEV 质量评估 + 几何映射
│   ├── refiner.py             # 图像优化工具
│   ├── vision_llm.py          # Qwen2.5-VL 视觉分析接口
│   ├── functions.py           # Function Calling 定义
│   ├── prompts.py             # 提示词模板
│   ├── data_logger.py         # 微调数据记录
│   └── __init__.py
│
├── models/                     # BEVFusion 模型
│   ├── bevfusion.py           # 主模型
│   ├── camera_encoder.py       # 相机分支 (ResNet50 + FPN + LSS)
│   ├── lidar_encoder.py        # LiDAR 分支 (PointPillars)
│   ├── fusion.py              # 融合模块 (ConvFuser)
│   └── heads.py               # 分割头
│
├── data/                       # 数据处理
│   ├── nuscenes_loader.py     # nuScenes 数据加载
│   └── bev_gt.py              # BEV GT 生成
│
├── utils/                      # 工具函数
│   ├── geometry.py            # 几何变换
│   ├── ogm.py                # OGM 生成
│   └── visualize.py          # 可视化
│
├── config.py                  # 配置文件
├── train.py                   # 训练脚本
├── run_agent_inference.py     # Agent 推理入口
├── run_inference.py          # 原始推理入口
├── run_ogm.py                # OGM 生成
├── ARCHITECTURE.md           # 系统架构文档
└── README.md                  # 本文档
```

## 核心流程

```
输入: 6路相机图像 + LiDAR 点云
    │
    ▼
BEVFusion 生成 BEV 分割图
    │
    ▼
Agent 评估 BEV 质量
(edge_density, integrity, problem_coords)
    │
    ▼
几何映射: BEV 问题区域 → 对应相机
    │
    ▼
VisionLLM 分析: Qwen2.5-VL 检测天气/光照问题
    │
    ▼
规则匹配: 根据 VisionLLM 结果选择工具
    │
    ▼
执行优化 → 重新生成 BEV
    │
    ▼
迭代最多3次或 finalize
    │
    ▼
输出: 优化后的 BEV 分割图
```

### 未来扩展: LLM 决策层

当前使用**规则匹配**选择工具，未来可以升级到 **Qwen3-8B LLM 决策**：

```
VisionLLM 分析结果 → Qwen3-8B → "根据这些信息，思考应该用什么工具"
                              ↓
                    thought + action (真正的LLM决策)
```

**为什么暂时用规则匹配**:
- VisionLLM 已返回结构化 suggested_tools，减少 LLM 调用
- 降低延迟，规则系统在简单场景下足够有效
- 未来积累足够微调数据后可升级

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision numpy opencv-python pyquaternion scikit-image
pip install ollama  # 用于本地 LLM 服务
```

### 2. 启动 Ollama

```bash
ollama serve
ollama pull qwen2.5-vl:7b  # 视觉语言模型
```

### 3. 运行 Agent 推理

```bash
# 加载训练好的权重
python run_agent_inference.py \
    --dataroot ./v1.0-mini \
    --sample 0 \
    --num_samples 10
```

## Agent 工具

| 工具 | 参数 | 功能 |
|------|------|------|
| `enhance_image` | camera_ids, type, factor | 对比度/锐化/降噪 |
| `remove_rain` | camera_ids, method, regions | 去雨 (CLAHE/高斯) |
| `dehaze` | camera_ids, method, regions | 去雾 (CLAHE/直方图均衡) |
| `crop_and_zoom` | camera_ids, bbox, zoom | 裁剪放大 |
| `finalize` | - | 确认输出 |

## 数据记录

Agent 决策会自动记录到 `agent_training_data.jsonl`，用于未来微调：

```json
{
  "session_id": "uuid",
  "iteration": 1,
  "bev_quality": {"iou": 0.72, "accuracy": 0.91},
  "agent_output": {
    "thought": "检测到相机0有雨",
    "action": {"name": "remove_rain", "parameters": {...}}
  },
  "result": {"iou_improvement": 0.03, "improved": true}
}
```

## 配置

Agent 使用 `config.py` 中的 BEV 配置：

```python
BEVConfig:
    bev_x_range = (-30, 30)      # BEV 范围(米)
    bev_y_range = (-30, 30)
    bev_size = (120, 120)        # 分辨率
```

Agent 配置：

```python
AgentCore:
    llm_url = "http://localhost:11434"
    max_iterations = 3
    model_name = "qwen2.5-vl:7b"
```

## 评估阈值

| 指标 | 阈值 | 说明 |
|------|------|------|
| edge_density | < 0.3 | 需要优化 |
| integrity | < 0.5 | 需要优化 |

达到阈值 → finalize → 输出当前 BEV
