# Parking Slot Agent v2

这是与旧 `parking_slot_hybrid_3d/`、`parking_slot_part2/` 隔离的新一代实现。
旧目录和旧产物保持不变；v2 只复用已经验证过的15帧 LiDAR、车位几何和媒体渲染基础模块。

## 数据流

```text
t0 之前15帧校正后 LiDAR + 1397车位几何库
                    │
                    ▼
Part 1：t0 周围30m排查
  ├─ nearby_map：30m内全部车位几何
  │    ├─ 已评估：free / occupied / unknown
  │    └─ 未被LiDAR覆盖：unobserved（不猜状态）
  └─ candidates：仅已评估的 free + unknown
       ├─ Free优先
       └─ Unknown其次
                    │
                    ▼
Part 2：每个候选一个可更新 SlotCase
  ├─ 强制粗FOV检查（内参包络约101°、可靠区80°；在原Part1地图叠加扇区）
  ├─ Camera可用：干净语义地图 + 原始全图 → Agent主动提出定位假设
  │    └─ 用 crop / causal sequence 支持、反证或保留歧义
  ├─ Camera不可用：目标局部多帧 LiDAR 细查
  ├─ 最多3次证据工具调用；没有合法工具时可提前结束为Unknown
  └─ Free或Occupied置信度达到0.90时结束当前车位
                    │
                    ▼
Free >= 0.90：停止全队列并输出
Occupied >= 0.90 / Unknown：继续下一个候选
```

## 关键合同

- `SceneSnapshot`：同一 (t_0) 下的30米语义地图、ego pose、15帧传感器清单。
- `SlotCase`：单车位的可更新档案，保留 Part 1 初态、当前态、置信度、Unknown 原因、FOV、证据和每轮推理。
- `Part1Output`：一个 `SceneSnapshot` 加按 Free→Unknown 排序的 `SlotCase` 集合。
- `FovResult`：只使用地图方位和保守角度范围；不声称知道车位在图像中的精确像素。
- `EvidenceRecord`：FOV、Camera、crop、LiDAR 的结构化观察及媒体引用。

Part 1 现有 `free_evidence.strength` / `occupied_evidence.strength` 是工程 gate 分数，
不是校准概率。v2 会保留它们并明确标记 `calibrated=false`。Part 2 的0.90阈值是当前
策略阈值；在完成真实标签标定前同样不能解释为统计学概率保证。

## Camera 语义对应

Camera context 给模型按顺序发送两张独立图片：第一张是以车辆朝向归一化的30米干净语义
地图，第二张是未叠加车位投影的原始 Camera 全图。地图规定前方朝上、地图左等于相机左、
正方位角等于相机左。Agent必须根据车位排列、道路边缘、路沿、墙体和车辆等共同参照物
提出可证伪的目标区域，再用Crop或因果序列将其标记为supported/refuted/ambiguous。

Camera 结论分别保存：

- `localization_confidence`：找对目标车位的把握；
- `occupancy_confidence`：在该区域判断空闲/占用的把握。

定位把握不足时，不允许仅凭某个看似空闲的图像区域给目标车位下结论。

## 当前数据入口

```text
outputs/frame_map_dataset_pose_corrected_final/frames.csv
outputs/frame_map_dataset_pose_corrected_final/map_points/
outputs/full_icpark_allframes_vehicle_cluster/slot_database.json
/home/ParkingAgent/dataset/dataset/dataset/image/
/home/ParkingAgent/dataset/dataset/dataset/velodyne/
```

frame 9277 的验收窗口为 LiDAR 9263–9277，最近 Camera 为
`left027800.png`，Camera-LiDAR 时间差约 8.4 ms。

## 本地 VLM：Qwen3-VL-4B-Instruct-FP8

v2 默认验收模型选用官方 `Qwen/Qwen3-VL-4B-Instruct-FP8`，固定 revision
`fefbb44cbcce8d1bb7e20b920b94f77432b3446d`。模型与运行参数记录在
`configs/qwen3_vl_4b_fp8.json`，权重位于
`models/Qwen3-VL-4B-Instruct-FP8/`。独立推理环境固定依赖记录在
`configs/qwen3_vl_4b_fp8_requirements.txt`。

启动本地、仅 loopback 可访问的 vLLM 服务：

```bash
scripts/serve_parking_slot_vlm.sh
```

启动脚本默认要求至少 14000 MiB 空闲显存，避免和已有训练任务争抢 GPU；默认只申请
40% 总显存、单并发、16K 上下文，并允许每个请求最多8张图片。服务就绪后运行 frame
9277 的真实候选队列：

```bash
scripts/run_parking_slot_agent_v2_qwen.sh
```

也可以显式传入 Part1 文件和输出目录：

```bash
scripts/run_parking_slot_agent_v2_qwen.sh \
  path/to/part1_output.json \
  path/to/output_dir
```

Agent 通过 `http://127.0.0.1:8010/v1/chat/completions` 发送多图请求，并要求
`json_object` 响应；返回内容仍会经过 v2 严格 action schema 和终止条件校验。

## OpenAI 多模态 Agent

OpenAI 版本使用官方 Responses API、严格结构化输出以及权限为 `600` 的独立密钥文件。
固定模型与推理参数记录在 `configs/openai_gpt_5_6_terra.json`，当前验证过的 SDK
依赖记录在 `configs/openai_vlm_requirements.txt`。密钥内容不会写入运行结果或审计文件。

安装项目内隔离依赖并运行 frame 9277：

```bash
python3 -m pip install --target .openai-runtime \
  -r configs/openai_vlm_requirements.txt
scripts/run_parking_slot_agent_v2_openai.sh
```

也可以将 Part1 输入和输出目录作为前两个位置参数传入启动脚本。
启动脚本会先验证固定 SDK 版本、Part1 合同、15 帧媒体、Key 权限、剩余磁盘和输出
目录隔离，并通过原子 lock 目录拒绝并发写入。同一输出目录已有结果时默认拒绝覆盖；
如确实需要显式覆盖，设置 `PARKING_ALLOW_OVERWRITE=1`。

## 目录职责

```text
contracts.py   # SceneSnapshot / SlotCase / 证据与轮次合同
part1.py       # 30m、15帧 Part 1 runner 和 v2 输出适配
fov.py         # 不依赖像素投影的粗 FOV 工具
tools.py       # Camera context/crop 与 LiDAR detail
model.py       # 严格动作合同、Replay 与 loopback Local VLM
agent.py       # 单 SlotCase、最多3轮的 ReAct 执行器
pipeline.py    # Free优先队列与找到可信Free后的全局早停
extended_lidar.py # 保持Part1 15帧合同的Part2严格因果长窗口证据构建
cli.py         # `python -m parking_slot_agent_v2 ...`
reporting.py   # Part2 基线/实验指标、汇总图与逐车位 HTML 审计报告
reporting_localization_trace.py # 单车位Map↔Camera主动定位轨迹可视化
lidar_geometry.py # 多帧LiDAR数值证据卡与Free/Occupied确定性硬门
reporting_optimization_zh.py # v3优化前后、单Case与全候选硬门报告
reporting_extended75_zh.py # 旧75帧Unknown消歧率、代表Case与逐车位中文报告
reporting_nature_zh.py # 60帧消融、跨场景验证与Nature风格中文研究稿
tests/         # v2 独立单元与集成测试
```

## Part 2 可视化对比报告

完整运行后，可将本地小模型基线与 OpenAI VLM 运行整理为一个可复用报告：

```bash
python3 scripts/build_parking_slot_agent_v2_report.py \
  --baseline-dir outputs/parking_slot_agent_v2_frame_9277/qwen3_5_0_8b \
  --experiment-dir outputs/parking_slot_agent_v2_frame_9277/openai_gpt_5_6_terra \
  --output-dir outputs/parking_slot_agent_v2_frame_9277/openai_gpt_5_6_terra/report
```

报告同时输出机器可读指标 JSON、静态汇总 PNG，以及按队列顺序展示每个候选
FOV/Camera/LiDAR 证据的 HTML。没有独立 GT 时，报告只声明工作流可靠性变化，
不会把未校准置信度解释为分类精度。

需要完整中文研究报告（工作流、定量图表、API 审计、24 个候选全部可视化）时：

```bash
python3 scripts/build_parking_slot_agent_v2_report_zh.py
```

默认输出到 `outputs/parking_slot_agent_v2_frame_9277/完整中文报告/`，并生成 HTML、
Markdown 摘要、机器可读 manifest 与全部汇总图表。

离线稳定性验收使用 `scripts/verify_parking_slot_agent_v2_stability.py`：它连续执行
两次全队列重放，验证同目录结果字节哈希一致，并将跨目录重放的决策投影与线上结果比较。

面向人工阅读、先讲结果再展开证据的直观版本：

```bash
python3 scripts/build_parking_slot_agent_v2_story_report_zh.py
```

默认输出到 `outputs/parking_slot_agent_v2_frame_9277/直观中文报告/`。完整技术报告仍保留，
直观版只把关键变化、三个代表案例和24车位点击式证据放在主阅读路径上。

如果需要检查一个车位从 Part1 到 Agent 终态的完整决策链：

```bash
python3 scripts/build_parking_slot_agent_v2_case_report.py --slot-id slot_1012
```

v3 优化验收报告使用已验证的 `openai_optimized_v3` 运行结果：

```bash
python3 scripts/build_parking_slot_agent_v2_optimization_report.py
```

默认输出到 `outputs/parking_slot_agent_v2_frame_9277/Part2_v3优化报告/`。报告首页
同时展示旧流程总体结果、新流程可信 Free 早停结果、`slot_1038` 的 FOV/LiDAR
证据链，以及全部24个候选的机器可读几何硬门审计。

单Case报告将确定性工作流路由、工具Observation和模型融合分开记录，同时输出机器可读
decision trace；不保存或展示不可验证的隐藏思维链。

## Part2 60 帧严格因果扩展

该扩展不修改 Part1 的 15 帧合同。它只为 Part2 的 `lidar_detail` 增加一个截至同一
`t0` 的 60 帧身份绑定证据包，并保留默认“第一个可信 Free 立即停止”的工作流。

```bash
python3 scripts/build_extended_part2_input.py \
  --part1 outputs/parking_slot_agent_v2_frame_9277/part1_optimized_v1/part1_output.json \
  --frames-csv outputs/frame_map_dataset_pose_corrected_final/frames.csv \
  --slot-db outputs/full_icpark_allframes_vehicle_cluster/slot_database.json \
  --map-points-dir outputs/frame_map_dataset_pose_corrected_final/map_points \
  --output-dir outputs/parking_slot_agent_v2_frame_9277/part1_optimized_v3_extended60 \
  --window 60
```

全候选统计必须显式传 `--evaluation-exhaustive`；正式运行不要传该开关。API 中断后可在
同一输出目录传 `--resume` 从原子 checkpoint 续跑。旧版 75 帧 frame 9277 报告用以下命令重建：

```bash
python3 scripts/build_parking_slot_agent_v2_extended75_report.py
```

报告输出在 `outputs/parking_slot_agent_v2_frame_9277/Part2_v4_75帧优化报告/`。当前
22 个原始 Unknown 中有 9 个通过严格几何门并由 Agent 确认（2 Free、7 Occupied）；
没有人工 GT，因此这是 40.9% 的严格消歧率，不是分类准确率。

## Nature 风格研究稿与复现审计

60 帧是当前消融实验选出的 Pareto 工作点：在 frame 9277 的 22 个 Part1 Unknown 上，
60 帧与 75 帧均严格消歧 9 个，而几何构建耗时由 135.68 秒降至 105.86 秒。独立于该
开发锚点的 6 个系统采样锚点上，匹配的 15 帧基线为 0/208，60 帧方案为 34/208。
这些数字衡量严格门控下的消歧能力，不是无 GT 条件下的准确率。

重建主文、PDF、全部主图、扩展数据、CSV 表和复现清单，并执行跨产物一致性审计：

```bash
python3 scripts/build_parking_slot_agent_v2_nature_report.py
python3 scripts/validate_parking_slot_agent_v2_nature_report.py
```

报告输出到
`outputs/parking_slot_agent_v2_nature_paper/Nature风格中文研究稿/`。审计器会逐项检查
消融计数、跨锚点匹配、22 个 OpenAI 终态与确定性硬门的一致性、60 帧因果性、HTML
资源和 PDF，并将结果写入 `validation_audit.json`。真实投稿前仍须补齐冻结 GT、独立
测试集、统计区间和外部基线；当前稿件明确标注为内部研究稿。

需要更完整的双语论文包（详细中文主文、英文 CVPR 草稿、匿名 LaTeX、补充材料、
21 条核验参考文献、全部 8 张图和 5 张数据表）时：

```bash
python3 scripts/build_parking_slot_agent_v2_full_paper.py
python3 scripts/validate_parking_slot_agent_v2_full_paper.py
```

输出位于 `outputs/parking_slot_agent_v2_nature_paper/CVPR完整论文包/`。验证器对章节完整性、
关键数字、参考文献唯一性、图像引用、CVPR 匿名 review 模式、PDF 和全部生成文件哈希执行
失败关闭审计。
