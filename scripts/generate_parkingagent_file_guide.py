#!/usr/bin/env python3
"""Generate a Chinese, exhaustive file-purpose catalog for ParkingAgent."""

from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "PARKINGAGENT_FILE_GUIDE.txt"
GUIDE_RELATIVE_PATH = "PARKINGAGENT_FILE_GUIDE.txt"


DIRECTORY_ROLES = {
    ".ipynb_checkpoints": "编辑器自动保存的副本；不是正式源文件或正式输出。",
    ".pytest_cache": "pytest 自动生成的测试发现与上次运行缓存，可删除后重建。",
    ".superpowers": "早期 SDD brief、合同和阶段报告，属于历史开发记录。",
    "configs": "运行配置和结构示例；placeholder 配置不能用于真实运行。",
    "docs": "当前算法/数据合同文档以及历史设计规格和实施计划。",
    "external": "随项目保存的第三方源码；不属于 ParkingAgent 自研业务逻辑。",
    "fast_lio_kitti": "面向当前 KITTI 风格数据集的 FAST-LIO ROS 启动与传感器配置。",
    "outputs": "数据集索引、逐帧点云、当前结果、历史回放和调试可视化等派生产物。",
    "parking_pose_correction": "LiDAR/map 配准、车辆 pose 漂移修正、插值平滑与审计。",
    "parking_slot_box_scoring": "车辆框假设、车位几何和旧版评分；部分几何能力被当前 Part1 复用。",
    "parking_slot_hybrid_3d": "当前 Part1：局部多帧 LiDAR scope、证据融合、三态判断与输出。",
    "parking_slot_part2": "Part2：unknown 队列、证据工具、保守决策、Camera gate 与 Replay/VLM 适配。",
    "parking_slot_validation": "人工核验车位状态所需的数据模型、证据、存储、指标与 Web 服务。",
    "protected_artifacts": "受保护的人工标签、预测和核验证据，普通实验不得覆盖。",
    "reports": "数据完整性、占用准备度与工作流的人读审计报告，不是运行时输入。",
    "scripts": "命令行入口、数据构建、审计、报告、可视化与历史实验脚本。",
    "tests": "单元测试、合同测试、回归测试、合成场景与固定测试夹具。",
}


EXACT_DESCRIPTIONS = {
    "README.md": "项目主说明：记录当前 Part1/Part2 主线、输入坐标、运行命令、输出、目录状态和已知边界。",
    "2026-07-11-active-parking-mvp-design.md": "早期主动泊车感知 MVP 的总体架构、数据合同、评分、状态机、Agent 和验收设计。",
    "azimuth_time_odometry_full_with_lidar_index.csv": "把 LiDAR 文件编号/时间戳与里程计时间及车辆 xyz+四元数 pose 对齐的原始复现输入表。",
    "file.gltf": "ICpark 停车场 glTF 全局结构地图，含墙、电梯、挡车器和车位等几何，供配准与可视化。",
    "file.obj": "同一停车场的 OBJ 网格结构地图，供结构解析、定位和静态几何脚本使用。",
    "微信图片_20260702201133_1359_2502.png": "微信导入的项目参考截图；正式算法和接口未引用。",
    "微信图片_20260702201311_1366_2502.png": "微信导入的另一张项目参考截图；正式算法和接口未引用。",
    "-0.55": "误生成的零字节临时文件，不是配置、源码或算法输入。",
    "{z_summary[suggested_first_pass_non_ground_threshold]}": "误生成的零字节模板占位文件，不是配置、源码或算法输入。",
    GUIDE_RELATIVE_PATH: "本文件：ParkingAgent 全目录的人读摘要与逐文件用途索引。",
    "configs/camera_front.placeholder.json": "Camera 标定 schema 示例；其中数值全部是 placeholder，严格加载器默认拒绝真实运行。",
    "docs/camera_observability_tool.md": "说明 Part2 Camera 预调用决策的输入、FOV 质量、LOS 遮挡、阈值、reason 和输出合同。",
    "docs/camera_observability_upstream.md": "说明 Camera gate 所需真实标定、map pose、墙柱层、坐标变换、可视化和数据缺口。",
    "docs/camera_projection_audit.md": "记录 Camera 投影 audit v2 的独立像素参考、中央/边缘误差政策和构建命令。",
    "docs/slot_aligned_mainline_algorithm_context.md": "详述车位对齐点云主线的坐标、累积、cluster ownership、状态、Camera 复核与局限。",
    "fast_lio_kitti/mapping_velodyne_dataset.launch": "为当前数据集加载 Velodyne 配置并启动 FAST-LIO mapping，可选启动 RViz。",
    "fast_lio_kitti/velodyne_dataset.yaml": "配置点云/IMU topic、64线10Hz LiDAR、180° FOV、量程、外参和 PCD 输出。",
    "parking_pose_correction/__init__.py": "地图位姿漂移诊断和校正工具包的公开入口。",
    "parking_pose_correction/dataset.py": "读取同步帧记录，并按车辆位姿把原始 LiDAR 转换到 map 坐标。",
    "parking_pose_correction/registration.py": "执行点集刚体配准/局部修正估计并计算修正前后对齐指标。",
    "parking_pose_correction/reporting.py": "生成位姿漂移曲线和校正诊断 HTML 报告。",
    "parking_pose_correction/se2.py": "处理角度归一化、关键帧、SE(2) 修正插值/平滑/稳定化及应用。",
    "parking_pose_correction/timestamps.py": "读取并同步 LiDAR、里程计和 Camera 时间轴，建立最近帧关联。",
    "parking_slot_box_scoring/__init__.py": "旧版受车位约束的车辆框评分实验包入口。",
    "parking_slot_box_scoring/box_hypotheses.py": "在车位局部坐标中枚举不同尺寸、偏移和角度的候选车辆框。",
    "parking_slot_box_scoring/config.py": "定义旧 Box Scoring 的 ROI、车辆尺寸、高度和评分阈值。",
    "parking_slot_box_scoring/frame_selection.py": "从基线或 pose 对齐轨迹选择每个车位的锚点帧和前后窗口。",
    "parking_slot_box_scoring/geometry.py": "提供车位坐标、旋转框、点在多边形内、凸裁剪和重叠率等几何函数。",
    "parking_slot_box_scoring/point_filtering.py": "估计局部地面并按 ROI/高度清理用于车辆框评分的点云。",
    "parking_slot_box_scoring/reporting.py": "输出 Box Scoring 的 JSON/CSV、调试图和 HTML 汇总报告。",
    "parking_slot_box_scoring/scoring.py": "计算候选框覆盖、高度、线性和邻位冲突分数并给出旧三态分类。",
    "parking_slot_hybrid_3d/__init__.py": "当前 Part1 保守 Hybrid3D 停车位证据包的公开入口。",
    "parking_slot_hybrid_3d/accumulation.py": "为单车位选择多帧、逐帧地面归一化，并保留射线/点来源地累计证据。",
    "parking_slot_hybrid_3d/camera.py": "Part1→Part2 的 Camera 几何、标定加载、车位投影和投影审计信任绑定。",
    "parking_slot_hybrid_3d/config.py": "集中定义并校验 Hybrid3D 的范围、地面、占用、自由空间和稳定性阈值。",
    "parking_slot_hybrid_3d/contracts.py": "定义帧、车位、scope、累计、3D/free/stability 证据和最终决策数据结构。",
    "parking_slot_hybrid_3d/decision.py": "按硬门限把 occupied/free/weak/stability 证据保守路由为三态。",
    "parking_slot_hybrid_3d/evaluation.py": "提供离线 GT provider 与含 abstention/risk-coverage 的诚实评估，不参与在线决策。",
    "parking_slot_hybrid_3d/evidence_3d.py": "从候选车辆框提取高度层、体素、BEV、形状、邻位重叠和时序一致性特征。",
    "parking_slot_hybrid_3d/free_space.py": "以多帧 LiDAR 射线体素化计算 free/hit/occluded/unobserved 并形成自由空间证据。",
    "parking_slot_hybrid_3d/geometry.py": "在 map 坐标与以车位为中心的米制局部坐标之间转换车位和点。",
    "parking_slot_hybrid_3d/ground.py": "对单帧车位邻域拟合局部地面、提供保守回退并归一化点高。",
    "parking_slot_hybrid_3d/io.py": "严格读取帧/车位/点云，并确定性、原子地写 JSON、JSONL、CSV 和文本。",
    "parking_slot_hybrid_3d/known_slot_scope.py": "依据距离、真实射线穿越、回波命中和核心覆盖判定车位观测范围。",
    "parking_slot_hybrid_3d/local_map.py": "选择因果局部帧窗，生成真实 LiDAR footprint，限制局部车位并评估邻排/通道候选。",
    "parking_slot_hybrid_3d/local_map_visualization.py": "绘制当前局部 LiDAR 覆盖、自车、三态车位、候选和可选 Camera overlay 的 BEV。",
    "parking_slot_hybrid_3d/occupied.py": "枚举车辆框，执行数量、高度、体素、邻位和静态结构硬门限以形成占用证据。",
    "parking_slot_hybrid_3d/part2_evidence.py": "把 unknown 的多帧点、射线、车位几何和邻接写为内容寻址 NPZ 证据包。",
    "parking_slot_hybrid_3d/pipeline.py": "串联 scope、累计、occupied/free、位姿稳定性和三态决策的 Part1 主流水线。",
    "parking_slot_hybrid_3d/projection_audit.py": "依据独立像素标注构建 fail-closed Camera 重投影审计和中央/边缘误差。",
    "parking_slot_hybrid_3d/raycasting.py": "提供线段与 AABB/凸棱柱裁剪、射线覆盖栅格等底层三维几何。",
    "parking_slot_hybrid_3d/reporting.py": "写出 Part1 scope、decisions、trace、local map、Part2 queue/evidence 和报告。",
    "parking_slot_hybrid_3d/shadow.py": "无 GT 条件下比较 Hybrid3D 与旧 202 车位基线的状态迁移。",
    "parking_slot_hybrid_3d/stability.py": "对点云/射线施加平移与 yaw 扰动，复算终态并检验结论稳定性。",
    "parking_slot_part2/__init__.py": "导出 provider-neutral 的 Part2 queue、模型、决策、编排和报告接口。",
    "parking_slot_part2/camera_calibration.py": "严格读取/校验 fisheye 或 pinhole 内外参、统一方向并执行畸变投影。",
    "parking_slot_part2/camera_observability.py": "在调用 Camera 语义模型前，以 FOV 质量、多点 LOS 和遮挡输出使用决策。",
    "parking_slot_part2/camera_observability_map.py": "把 Part1 local_map、场景几何和严格 Camera gate 合成地图审计报告。",
    "parking_slot_part2/camera_pose.py": "在 Camera 时刻插值车辆 SE(3)，并计算带协方差的 T_map_camera。",
    "parking_slot_part2/contracts.py": "定义 Part2 队列资源、视觉帧、encounter、关系、item 和 envelope 合同。",
    "parking_slot_part2/decision.py": "校验结构化语义提案并在不改 Part1 基线证据时生成 Part2 resolution。",
    "parking_slot_part2/grouping.py": "按空间/证据冲突关系确定性地把 unknown 队列项分组。",
    "parking_slot_part2/local_vlm.py": "调用用户管理的 loopback OpenAI-compatible 本地 VLM，并限制媒体与失败边界。",
    "parking_slot_part2/map_only_camera_precheck.py": "仅用 Part1 局部地图和 anchor pose 代理，标记候选是否值得后续调用 Camera。",
    "parking_slot_part2/map_only_camera_precheck_visualization.py": "绘制 map-only Camera 预检的机器 BEV 和带 FOV/射线/reason 的审计图。",
    "parking_slot_part2/media.py": "严格加载 LiDAR/RGB 证据并生成不泄露本地路径的受限媒体表示。",
    "parking_slot_part2/model.py": "定义 provider-neutral 模型动作和可确定性复现的 ReplayModel。",
    "parking_slot_part2/orchestrator.py": "按工具白名单、回合/预算/停止策略编排一个 Part2 冲突组。",
    "parking_slot_part2/preflight.py": "建立 EvidenceCatalog，并在不做语义判断时检查 Camera 证据能力和几何先决条件。",
    "parking_slot_part2/queueing.py": "校验 unknown queue 的身份哈希、schema、资源关联、文件哈希和唯一性。",
    "parking_slot_part2/reporting.py": "派生 run id、校验 trace/attempt/result，并原子写 Part2 JSON/HTML。",
    "parking_slot_part2/shadow.py": "构造与正式决策隔离、可外部审计的 Part2 shadow 样本子集。",
    "parking_slot_part2/shadow_replay.py": "把盲审 shadow judgement 编译为有界、确定性的 Replay 动作。",
    "parking_slot_part2/static_obstacle_map.py": "读取墙柱静态层，建立局部索引并查询视线走廊完整性和遮挡。",
    "parking_slot_part2/tools.py": "实现 Part2 唯一允许的受界证据检查工具及精确 allowlist。",
    "parking_slot_part2/trace.py": "写入确定性、只追加的 Part2 JSONL 证据与工具调用账本。",
    "parking_slot_validation/__init__.py": "人工停车位占用复核模块的公开入口。",
    "parking_slot_validation/evidence.py": "为人工复核选择 Camera 可见、时间匹配且多帧的车位证据。",
    "parking_slot_validation/metrics.py": "仅对人工已解决标签计算准确率、覆盖率和混淆等指标。",
    "parking_slot_validation/models.py": "规范化预测三态并生成稳定 dataset/manifest/sample/evidence 身份。",
    "parking_slot_validation/static/index.html": "浏览器端人工标注界面，展示证据并提交三态标签。",
    "parking_slot_validation/storage.py": "带 dataset/evidence 防串用校验地原子保存人工标签和审计数据。",
    "parking_slot_validation/web.py": "提供本地 HTTP API 和静态页面以运行人工车位标注流程。",
}


SCRIPT_DESCRIPTIONS = {
    "apply_lidar_map_registration.py": "把 LiDAR—地图配准变换应用到完整里程计轨迹，输出 map 轨迹与检查结果。",
    "audit_camera_extrinsic.py": "利用 CARLA 深度图审计 LiDAR→Camera 的坐标轴、方向和位移外参。",
    "build_camera_projection_audit.py": "依据真实像素参考生成哈希绑定的 Camera 投影审计，检查中央/边缘误差。",
    "build_frame_map_dataset.py": "把图像、LiDAR、同步 map pose 和 map 点云整理为逐帧数据集。",
    "build_front_camera_multiframe_review.py": "为 LiDAR 车位候选寻找 anchor 前的前置 Camera 多帧图并生成复核材料。",
    "build_highest_score_map_review.py": "为旧版最高分车位复核案例补充全局地图叠加图。",
    "build_hybrid_3d_global_map_report.py": "把旧 Hybrid3D 全路线结果画成历史审计报告；不是当前局部图或全局真值。",
    "build_hybrid_3d_shadow_report.py": "无 GT 对比旧 202 车位基线与 Hybrid3D shadow 的状态迁移。",
    "build_multiframe_temporal_consensus.py": "汇总多锚点多帧累积实验并计算跨时间共识状态。",
    "build_occupied_chain_analysis.py": "分析相邻 occupied 链，辅助区分车辆链、边界、墙体或车道侧结构。",
    "build_paired_dataset_subset.py": "从完整数据制作可独立使用的小型 Camera 图像与点云配对子集。",
    "build_part2_case_review_pack.py": "把 Part1 待复核案例组织成 Part2 案例、分组和调试资源包。",
    "build_part2_shadow_global_map_report.py": "只读叠加 Part2 shadow 并生成无 GT 全局地图审计报告。",
    "build_pose_assisted_review.py": "为 Part2 案例增加由车辆 pose 推导的诊断信号，不修改 Part1 状态。",
    "build_pose_corrected_frame_dataset.py": "使用已审计校正轨迹重新生成逐帧 map pose 和 map 点云。",
    "build_pose_correction_review_report.py": "生成 pose 校正前后 Camera、LiDAR 和全局地图同步对比报告。",
    "build_selected_consensus_camera_review.py": "为多帧共识筛选出的车位制作 Camera 图像复核包。",
    "build_slot_aligned_camera_correspondence_report.py": "生成车位对齐多帧证据的 Camera、LiDAR 与地图对应报告。",
    "build_slot_aligned_global_map_report.py": "把车位中心对齐累积实验绘制为地图视角 HTML。",
    "build_slot_box_correspondence_report.py": "生成车位约束车辆框的点云、照片和全局地图三方对应报告。",
    "build_slot_box_global_map_report.py": "把旧车位约束车辆框评分绘制成全局地图 HTML。",
    "build_slot_decision_table.py": "合并旧状态、车辆聚类、pose 辅助和 occupied 链诊断为统一审计表。",
    "build_static_obstacle_layer_from_lidar.py": "从多帧 map LiDAR 生成墙柱候选；只有动态过滤与覆盖通过才标完整。",
    "build_top5_camera_review.py": "为 possible-free 前五目标生成 Camera 叠加图和局部裁剪。",
    "dataset_integrity_linkage.py": "审计图像、LiDAR、pose、时间戳与 map 点云的数据完整性和关联。",
    "fast_slot_scoring.py": "使用 pose 和单帧稀疏 LiDAR 对 glTF 地图可见车位做早期快速评分。",
    "fit_pose_to_map_correspondences.py": "根据人工帧号—地图坐标对应点拟合 pose→map 二维相似变换。",
    "generate_part1_15frame_summary_pdf.py": "根据当前 Part1 输出生成一页中文十五帧算法简明 PDF。",
    "generate_parkingagent_file_guide.py": "扫描整个项目并生成本文件用途清单，同时校验每个普通文件均被列出。",
    "generate_world_to_map_report_pdf.py": "生成 world/pose→map 外参缺失与地图匹配问题的中文说明 PDF。",
    "gltf_lidar_ndt.py": "用语义 glTF 地图做 LiDAR 二维 NDT 定位的早期原型。",
    "interpolate_map_keyframe_corrections.py": "把 NDT 关键帧 map pose 修正插值到整条 LiDAR 轨迹。",
    "kitti_to_fastlio_bag.py": "把 KITTI 风格 LiDAR/OXTS 转换成 FAST-LIO 可用 ROS1 bag。",
    "lidar_map_ndt_optimized.py": "结合里程计、glTF 与 NDT 执行优化版整段定位和轻量平滑。",
    "lidar_map_tracking.py": "以里程计相对运动预测、NDT 地图修正，在 glTF 地图跟踪轨迹。",
    "lidar_only_odometry.py": "对 KITTI Velodyne 点云运行纯 LiDAR 二维 scan-to-submap 里程计。",
    "multiframe_pointcloud_accumulation_probe.py": "诊断多帧 LiDAR 累积能否形成车辆证据，不更新 Part1 状态。",
    "ndt_localization.py": "把停车场 LiDAR 扫描与 OBJ 地图做平面 NDT 定位的基础实验。",
    "occupancy_readiness_audit.py": "检查逐帧数据、地图对齐、点高和旧评分是否具备车位占用建图条件。",
    "part1_slot_scoring.py": "较早的 Part1 保守评分实现；基于 map LiDAR 与射线，未观测保持 unknown。",
    "pose_gltf_route.py": "把 pose-truth 全路线绘制到 glTF 停车场并输出交互路线。",
    "pose_truth_html_player.py": "导出结合 Camera、pose 和累积 LiDAR 的二维动态定位 HTML。",
    "pose_truth_report.py": "生成 pose 约束的车辆轨迹、Camera 和累积 LiDAR 静态诊断报告。",
    "pose_truth_video.py": "把 pose、Camera 和累积 LiDAR 地图渲染为视频或 GIF。",
    "refine_lidar_map_registration_seed.py": "在已有 LiDAR—地图初始配准附近优化尺度、旋转和平移。",
    "register_lidar_map_to_gltf.py": "把多帧累积 LiDAR 局部地图整体配准到 glTF 语义地图。",
    "register_lidar_obstacles_to_structural_map.py": "以高于地面的 LiDAR 障碍点对齐 glTF 墙、电梯和挡车器结构层。",
    "render_alignment_debug_hd.py": "高分辨率渲染地图、轨迹和现有对齐变换以检查配准。",
    "render_map_only_camera_precheck.py": "读取 Part1 局部图执行 map-only Camera 预检并画审计图，不调用语义模型。",
    "render_part1_camera_observability_map.py": "在 Part1 局部图上运行严格 Camera gate 并可视化；缺数据时 fail-closed。",
    "render_pose_lidar_world_hd.py": "不使用 ICpark 地图，仅在 world/pose 坐标超高清绘制轨迹与 LiDAR。",
    "run_hybrid_3d_slot_evidence.py": "当前正式 Part1 CLI：在以当前帧结尾的 3–15 帧因果窗口运行 Hybrid3D。",
    "run_parking_slot_validation.py": "准备并启动人工核验 occupied/free 的本地 Web 界面。",
    "run_part2_agent.py": "校验/Replay/本地 VLM 运行 Part2 unknown-agent，并支持 shadow 与媒体编译。",
    "run_pose_drift_correction.py": "审计 map pose 漂移、拟合校正并重建校正后的逐帧数据。",
    "run_slot_constrained_box_scoring.py": "在车位 polygon 约束下执行旧车辆 3D box 评分并对比基线。",
    "slot_aligned_multiframe_accumulation_probe.py": "围绕指定车位按 pose 选 anchor，做车位坐标多帧累积诊断。",
    "slot_structure_map.py": "从 OBJ 结构地图和车位配置渲染清晰的停车场车位平面图。",
    "visualize_camera_observability.py": "把严格 Camera gate 的采样点、投影、遮挡射线和质量区画到 Camera 图。",
}


TEST_DESCRIPTIONS = {
    "hybrid_3d_fixtures.py": "Hybrid3D 合成几何、车辆点、射线、scope/decision 场景的共享构造器；不是独立测试入口。",
    "test_build_paired_dataset_subset.py": "验证成对数据子集的采样、质量/同步过滤和自包含归档。",
    "test_build_static_obstacle_layer_from_lidar.py": "验证 LiDAR 墙柱候选生成、动态过滤和 incomplete fail-closed。",
    "test_camera_correspondence_report.py": "验证 Camera/LiDAR 时间戳匹配和 anchor 前多帧窗口选择。",
    "test_dataset_integrity_linkage.py": "验证 image/LiDAR/map points/pose/slot 的完整性和链接审计。",
    "test_front_camera_multiframe_review.py": "验证前视多帧 Camera 预筛、投影质量、排序、abstain、overlay 和 HTML。",
    "test_hybrid_3d_accumulation.py": "验证逐车位多帧累计、anchor、点/射线来源、坏帧审计和 split。",
    "test_hybrid_3d_camera.py": "验证旧 Hybrid Camera 标定、投影审计绑定、capability 和候选排序。",
    "test_hybrid_3d_cli.py": "验证当前 Hybrid CLI help 和 scope-only 输出边界。",
    "test_hybrid_3d_contracts.py": "验证 map→metric 坐标、配置约束、box 搜索空间和冻结合同。",
    "test_hybrid_3d_decision.py": "验证 Part1 三态优先级、scope/质量/冲突/稳定性/弱证据规则。",
    "test_hybrid_3d_evaluation.py": "验证无 GT 不输出伪准确率，有 GT 时 denominator、ignore 和身份正确。",
    "test_hybrid_3d_evidence.py": "验证 3D 体素、归属、高度层、多帧支持、静态风险和时间特征。",
    "test_hybrid_3d_free_space.py": "验证射线自由空间、前景遮挡、弱障碍 veto、未观测区和时序冲突。",
    "test_hybrid_3d_global_map_report.py": "验证旧全局报告必须明确标注且 scope 缺失不可伪装成 unknown。",
    "test_hybrid_3d_ground.py": "验证斜坡地面拟合、退化回退和坏点处理。",
    "test_hybrid_3d_io.py": "验证 frame/slot/NPZ 加载、坏数据拒绝、缓存和原子规范写出。",
    "test_hybrid_3d_local_map.py": "验证 3–15 帧因果局部图、远处隐藏、三态、A/B 和候选硬约束。",
    "test_hybrid_3d_occupied.py": "验证车辆框 occupied gate、失败码、矮物/柱体 veto 和鲁棒性。",
    "test_hybrid_3d_ownership.py": "验证目标核心、外部和相邻弱障碍的归属与保守 free veto。",
    "test_hybrid_3d_pipeline.py": "验证 Part1 全阶段、错误降级、规范产物/queue 和 replay hash 稳定。",
    "test_hybrid_3d_projection_audit.py": "验证 Camera 投影审计、中央/边缘误差、provenance/hash 和 fail-closed。",
    "test_hybrid_3d_raycasting.py": "验证射线/车位 prism 裁剪、first hit 和三维体素遍历边界。",
    "test_hybrid_3d_scope.py": "验证 in/partial/out scope、真实穿越、覆盖、坏帧和角度索引。",
    "test_hybrid_3d_shadow.py": "验证旧 box baseline 与 Hybrid 的无 GT 状态迁移和 shadow 输出。",
    "test_hybrid_3d_stability.py": "验证七种 pose 扰动、点/原点/射线变换和 6/7 稳定门限。",
    "test_hybrid_3d_synthetic_scenarios.py": "验证批准的 Hybrid3D 合成场景矩阵、reason 和终态不变量。",
    "test_hybrid_3d_terminal_stability.py": "验证 pose 变体下同目标关联、受约束重拟合和冲突复算。",
    "test_hybrid_3d_weak_assessment.py": "验证弱证据 taxonomy、归属、静态/时间 disposition 和 unknown 审计。",
    "test_map_only_camera_precheck.py": "验证直接读 Part1 map 的 ±80° Camera 预标记、遮挡、尺度和不调用 Camera。",
    "test_map_only_camera_precheck_visualization.py": "验证 map-only machine BEV/annotated 图颜色、FOV、射线和调用声明。",
    "test_parking_slot_validation_evidence.py": "验证人工核验的证据帧质量、抽样、overlay、manifest 和禁止覆盖。",
    "test_parking_slot_validation_metrics.py": "验证人工评估中 unknown abstention、unobservable 排除和二分类 confusion。",
    "test_parking_slot_validation_models.py": "验证预测/标签模型、身份、canonical ID 和内容指纹。",
    "test_parking_slot_validation_storage.py": "验证人工标签保存/修订、身份、防串用、导出删除和备份恢复。",
    "test_part2_camera_calibration.py": "验证严格 pinhole/fisheye 标定、外参求逆、placeholder 拒绝和鱼眼投影。",
    "test_part2_camera_observability.py": "验证 Camera use/do-not-use/insufficient 的 FOV、边缘、遮挡和缺输入规则。",
    "test_part2_camera_observability_map.py": "验证严格 Camera gate debug trace 叠加局部 map 且不污染纯 Part1 图。",
    "test_part2_camera_observability_upstream.py": "验证墙柱完整性、高度、out-of-route 遮挡、pose 质量和鱼眼边界。",
    "test_part2_camera_pose.py": "验证 T_map_camera 组合、SE(3)/covariance 插值、过期与 frame mismatch。",
    "test_part2_decision_cli.py": "验证 Part2 确定性 gate、组归并、ledger/replay/hash、报告和 CLI。",
    "test_part2_e2e_fixture.py": "用固定 fixture 运行完整、确定性的 Part2 CLI 和 shadow media 合同。",
    "test_part2_grouping_tools.py": "验证冲突/共享证据分组、preflight、EvidenceCatalog 和工具 allowlist。",
    "test_part2_lidar_media.py": "验证内容寻址 LiDAR evidence pack、身份/hash、三联图和失败边界。",
    "test_part2_local_vlm.py": "验证仅 loopback 的本地多模态 adapter、严格 action、媒体与隐私限制。",
    "test_part2_local_vlm_cli.py": "验证本地 VLM CLI 的 RGB 回合、可回放报告和不安全输入拒绝。",
    "test_part2_orchestrator.py": "验证 Replay、action 修复、证据归属、工具预算、fallback unknown 和 trace。",
    "test_part2_queue_contracts.py": "验证 Part1→Part2 queue 的版本、身份/hash、资源、encounter 和引用。",
    "test_part2_rgb_media.py": "验证 RGB 解码、标注、contact sheet、内容寻址、投影边界和防篡改。",
    "test_part2_shadow.py": "验证无 GT shadow 分层抽样、group closure、资源解析和 fail-closed。",
    "test_part2_shadow_global_map_report.py": "验证 shadow before/after 地图、媒体身份、blind records 和 Part1 不变。",
    "test_part2_shadow_replay.py": "验证盲判记录编译为工具/Replay 动作以及缺 RGB 时降级。",
    "test_part2_static_obstacle_map.py": "验证局部墙柱完整性、索引、高度、动态排除、polygon 和坐标一致性。",
    "test_pose_correction_review_report.py": "验证 pose review HTML 的资源嵌入。",
    "test_pose_drift_correction.py": "验证 pose 漂移修正的同步、关键帧、配准限制、插值和逐帧平滑。",
    "test_slot_aligned_accumulation.py": "验证旧 slot-aligned 对称窗口、stride、边界去重和 anchor。",
    "test_slot_box_geometry.py": "验证车位局部坐标往返和车辆框 polygon 面积。",
    "test_slot_box_scoring_synthetic.py": "验证旧 box scoring 在居中车辆、矮残留和边界场景的分数方向。",
    "test_visualize_camera_observability.py": "验证 Camera 调试图的 LiDAR/射线/FOV/墙柱及分区重投影误差。",
}


FIXTURE_DESCRIPTIONS = {
    "tests/fixtures/part2/valid_queue.json": "可通过严格 Part2 queue 合同的示例 unknown 队列。",
    "tests/fixtures/part2/replay_actions.json": "Part2 测试使用的确定性 Replay 模型动作。",
    "tests/fixtures/part2/artifacts/slot_database.json": "Part2 测试用的小型车位几何资源。",
    "tests/fixtures/part2/artifacts/lidar_map.pcd": "Part2 测试用的小型 LiDAR 地图资源。",
    "tests/fixtures/part2/artifacts/map_points/manifest.json": "Part2 测试用 map-points 目录身份清单。",
    "tests/fixtures/part2/artifacts/slot_decisions.json": "Part2 测试用小型 Part1 base 决策。",
    "tests/fixtures/part2/artifacts/corrected_frames.csv": "Part2 测试用校正 pose/frame 索引。",
    "tests/fixtures/part2/artifacts/camera_calibration.json": "Part2 测试专用 Camera 标定。",
    "tests/fixtures/part2/artifacts/rgb_frame.ppm": "无需额外图片编解码依赖的微型 RGB fixture。",
}


OUTPUT_DIRECTORY_ROLES = {
    "current_mainline": "当前人读主线索引，以及 2026-07-16 清理时的历史机器快照。",
    "frame_map_dataset": "pose v3 上游逐帧索引和原始 map-point 重建输入。",
    "frame_map_dataset_pose_corrected_final": "应用 pose v3 后的逐帧索引和 map 点云；当前 Part1 直接输入。",
    "full_icpark_allframes_vehicle_cluster": "1397 个既有车位的完整几何库；不提供实时占用真值。",
    "part1_local_lidar_frame_9277": "当前 15 帧局部 Part1、Part2 交接和 Camera 旁路验收样例。",
    "part1_local_lidar_frame_11338": "另一 anchor 的较早 15 帧局部 Part1 样例。",
    "pose_drift_correction_v3": "当前 canonical pose 修正参数、关键帧配准、审计和 review。",
    "slot_constrained_box_scoring_pose_corrected_final": "旧 Box Scoring 迁移基线，不是执行主线或 GT。",
    "slot_hybrid_3d_pose_corrected_v2_replay": "历史整路线 Hybrid v2 回放；不是当前局部感知图。",
    "slot_part2_clean_agent_shadow_v1": "Part2 盲判、Replay、媒体和 SHADOW/NO-GT 报告。",
}


OUTPUT_BASENAME_DESCRIPTIONS = {
    "metadata.json": "逐帧 map-point 数据集的 schema、源文件、坐标和统计元数据。",
    "frames.csv": "逐帧 pose、点云路径、时间戳与 Camera 同步索引。",
    "local_frame_manifest.json": "当前局部 Part1 窗口内帧的 pose、点云、时间戳和同步元数据。",
    "local_slot_database.json": "从完整几何库裁出的当前局部车位数据库。",
    "known_slot_scope.csv": "每个局部车位的 near/crossing/hit 帧、覆盖率和 scope 原因。",
    "slot_decisions.csv": "便于查看的车位三态、主原因、证据强度和稳定性摘要表。",
    "slot_decisions.json": "完整车位决策、occupied/free/unknown 证据、失败门和详细原因。",
    "decision_trace.jsonl": "逐车位各阶段只追加审计轨迹；含实际 selected/valid/excluded 帧。",
    "summary.json": "对应一次流水线运行的数量、原因、资源和耗时汇总。",
    "report.html": "对应运行结果的人读 HTML 报告。",
    "unknown_agent_queue.json": "Part1→Part2 的正式 unknown 队列和资源引用。",
    "local_map.json": "Part1 正式局部、不完整 LiDAR map；只含被当前窗口真实覆盖的车位。",
    "local_map.png": "与 local_map.json 一致的纯 Part1 局部 BEV 可视化。",
    "camera_capability.json": "在投影审计约束下 Camera 证据是否可供 Part2 使用的能力摘要。",
    "camera_observability_map.json": "严格 Camera gate 在 Part1 局部地图上的结构化判断报告。",
    "local_map_camera_observability.png": "叠加严格 Camera FOV、射线和 reason 的局部地图审计图。",
    "map_only_camera_precheck.json": "仅用 Part1 map 的 Camera 候选预标记结果；不改变车位状态。",
    "map_only_camera_precheck.png": "带 pose proxy、±80° FOV、射线和 reason 的 map-only 审计图。",
    "camera_gate_bev.png": "map-only 预检使用的固定 RGB 机器 BEV；颜色不编码最终决策。",
    "part1_15frame_state_algorithm_summary.pdf": "一页中文 PDF，简述当前十五帧 Part1 三态算法和 frame 9277 结果。",
    "slot_database.json": "完整停车位 polygon、中心、方向和邻接几何数据库；不是占用真值。",
    "slot_box_scores.csv": "旧车位约束车辆框评分结果表，保留为迁移基线。",
    "artifact_manifest.json": "历史清理时的产物路径、大小和哈希快照，不随当前主线自动重写。",
    "alignment_transform.json": "LiDAR/pose 坐标到停车场 map 坐标的已拟合对齐变换。",
    "alignment_parameters.csv": "对齐变换的尺度、旋转和平移参数表。",
    "aligned_trajectory_final.csv": "应用全局对齐后的车辆轨迹。",
    "azimuth_time_odometry_compatible.csv": "整理成项目兼容字段的 LiDAR—里程计同步 pose 表。",
    "corrected_trajectory.csv": "应用漂移修正后的完整车辆 map 轨迹。",
    "pose_corrections.csv": "按关键帧/位置记录的 pose 修正量。",
    "keyframe_registration.csv": "各关键帧局部 LiDAR—结构地图配准结果与 gate。",
    "drift_before_after.json": "pose 漂移修正前后残差和统计对比。",
    "drift_report.html": "pose 漂移校正的人读 HTML 审计报告。",
    "pose_correction_correspondence_report.html": "pose 校正前后 Camera/LiDAR/map 对应报告。",
    "pose_correction_correspondence_report_embedded.html": "将图片嵌入单文件的 pose 校正对应报告。",
    "pose_correction_over_distance.png": "沿路线距离展示 pose 修正量的曲线图。",
    "structural_residual_before_after.png": "结构地图残差在校正前后的对比图。",
    "trajectory_before_after_on_map.png": "校正前后轨迹叠加停车场地图的对比图。",
    "review_index.json": "pose review 图片、帧和来源的索引。",
    "part2_resolutions.json": "Part2 对 queued unknown 给出的结构化 resolution；不覆盖原始 Part1 文件。",
    "final_route_states.json": "把 Part2 resolution 合并到 Part1 基线后得到的路线状态快照。",
    "run_manifest.json": "Part2 运行身份、输入哈希、模型/Replay 和输出资源清单。",
    "selection_manifest.json": "Part2 shadow/复核样本选择规则、身份和选中项清单。",
    "media_manifest.json": "Part2 发布媒体的内容哈希、类型、尺寸和证据身份。",
    "blind_records.json": "与 GT 隔离的 Part2 人工/模型盲判记录。",
    "replay_actions.json": "从盲判或固定输入编译出的确定性 Part2 Replay 动作。",
    "global_map_part2_shadow_full.png": "Part2 shadow provenance 叠加完整停车场几何的审计图。",
    "global_map_part2_shadow_route_zoom.png": "Part2 shadow 相关路线区域的放大图。",
    "global_map_part2_evaluated_before_after.png": "仅比较实际有 resolution 车位的 Part1 before / Part2 after。",
    "global_map_part2_shadow_report.html": "明确标注 SHADOW/NO GT 的 Part2 全局地图报告。",
    "common_baseline_transition.csv": "旧基线与 Hybrid 共同车位的状态迁移表。",
    "newly_evaluated_route_slots.csv": "Hybrid 相对旧基线新增实际评估的路线车位表。",
    "shadow_summary.json": "无 GT shadow 对比的覆盖和状态迁移汇总。",
}


HISTORICAL_OUTPUT_OVERRIDES = {
    "local_frame_manifest.json": "该历史 anchor 的 Part1 窗口帧 pose、点云、时间戳和同步元数据。",
    "local_slot_database.json": "从完整几何库裁出的该历史 anchor 局部车位数据库。",
    "local_map.json": "该历史运行生成的局部、不完整 LiDAR map；只含当时窗口覆盖的车位。",
    "local_map.png": "与该历史运行 local_map.json 对应的局部 BEV 可视化。",
}


REPORT_DESCRIPTIONS = {
    "reports/workflow_explanation/slot_occupancy_workflow_report.md": "旧点云占用工作流、Part1/Part2 卡片和旧统一状态的讲解稿。",
    "reports/dataset_integrity_linkage/integrity_summary.json": "数据链路检查的聚合计数。",
    "reports/dataset_integrity_linkage/dataset_integrity_linkage_report.md": "image/LiDAR/pose/map-points/slot 链路完整性叙述报告。",
    "reports/dataset_integrity_linkage/frame_linkage_audit.csv": "逐帧检查 image/LiDAR/map points/pose、点数和问题码。",
    "reports/dataset_integrity_linkage/slot_linkage_audit.csv": "逐车位检查 geometry、旧 scores、旧 decision table 的链接问题。",
    "reports/observed_slot_occupancy/observed_slot_occupancy_summary.txt": "较早 observed occupancy 实验的简要统计。",
    "reports/observed_slot_occupancy/workflow_summary_and_issues.md": "较早两段轨迹占用流程、问题和后续建议。",
    "reports/observed_slot_occupancy/workflow_summary_and_issues.pdf": "对应旧工作流 Markdown 的 PDF 版。",
    "reports/observed_slot_occupancy/observed_slot_occupancy_map.csv": "较早逐车位 observed state/probability/visibility/support/conflict/reason 表。",
    "reports/occupancy_readiness/occupancy_readiness_report.md": "较早数据是否足以做占用图的准备度审计。",
    "reports/occupancy_readiness/dataset_summary.json": "较早输入数据库、帧和点云摘要。",
    "reports/occupancy_readiness/slot_database_summary.json": "较早车位几何库结构统计。",
    "reports/occupancy_readiness/slot_geometry_summary.csv": "逐车位 polygon、中心、面积、长宽和邻接统计。",
    "reports/occupancy_readiness/slot_scores_summary.json": "旧车位 scoring 结果的聚合分布。",
    "reports/occupancy_readiness/slot_score_summary.csv": "旧 score/probability 列的 count/mean/std/quantile。",
    "reports/occupancy_readiness/frame_slot_evidence_summary.json": "旧 frame-slot evidence 聚合计数。",
    "reports/occupancy_readiness/fused_belief_summary.json": "旧融合 belief/probability 汇总。",
    "reports/occupancy_readiness/figures/z_distribution.png": "LiDAR Z 高度分布诊断图。",
}


PROTECTED_DESCRIPTIONS = {
    "protected_artifacts/README.md": "说明此目录保存人工/手工审阅数据，不能随普通 outputs 清理。",
    "predictions.json": "旧 box-scoring baseline 的 202 个待人工验证 prediction/case。",
    "evidence_manifest.json": "人工验证 case 的证据帧、几何、source、asset 和 hash 审计清单。",
    "human_labels.json": "当前人工车位标签、reviewer、reason、时间和证据引用；是主标签文件。",
    "human_labels.json.bak": "较早人工标签备份，用于恢复/历史审计；不是当前权威标签源。",
}


EXTERNAL_EXACT = {
    "1": "上游 FAST-LIO PCD 目录内仅含数字 1 的占位文件；不是实际 PCD 点云。",
    "CMakeLists.txt": "配置第三方 FAST-LIO 的 ROS/catkin 编译、依赖、消息和可执行文件。",
    "LICENSE": "第三方 FAST-LIO 源码许可证。",
    "README.md": "第三方 FAST-LIO/FAST-LIO2 原理、安装、运行和传感器说明。",
    "package.xml": "第三方 FAST-LIO ROS 包名、版本、维护者、许可证和依赖元数据。",
    "Pose6D.msg": "FAST-LIO 自定义六自由度位姿/协方差 ROS 消息。",
    "laserMapping.cpp": "FAST-LIO ROS 主节点：同步、滤波、scan-to-map、ikd-Tree 维护和发布。",
    "preprocess.h": "声明不同 LiDAR 消息的盲区、时间/线束和特征预处理器。",
    "preprocess.cpp": "实现 Avia/Velodyne/Ouster 等点云预处理和可选特征提取。",
    "IMU_Processing.hpp": "实现 FAST-LIO IMU 初始化、去畸变、传播和 LiDAR-IMU 时段处理。",
    "common_lib.h": "定义 FAST-LIO 公共点类型、测量组、状态别名、日志宏和通用函数。",
    "Exp_mat.h": "FAST-LIO 旋转/李代数矩阵指数辅助函数。",
    "so3_math.h": "FAST-LIO SO(3) 指数、对数、帽算子和导数数学函数。",
    "use-ikfom.hpp": "定义 FAST-LIO 使用的 IKFoM 状态、输入、过程和观测模型接口。",
    "matplotlibcpp.h": "第三方 matplotlib-cpp 单头文件，使 C++ 可调用 Python Matplotlib。",
    "ikd_Tree.h": "声明支持增删、重建、范围和近邻查询的增量 k-d tree。",
    "ikd_Tree.cpp": "实现 ikd-Tree 并发重建、增删、范围和 KNN 查询。",
    "esekfom.hpp": "IKFoM 流形误差状态迭代扩展卡尔曼滤波器核心模板。",
    "Fast_LIO_2.pdf": "上游 FAST-LIO2 论文 PDF。",
    "overview_fastlio2.svg": "上游 FAST-LIO2 算法流程矢量图。",
}


EXTERNAL_PATH_DESCRIPTIONS = {
    "external/FAST_LIO/include/ikd-Tree/README.md": (
        "第三方 ikd-Tree 子模块的编译、API、增量更新和性能说明；不是 FAST-LIO 主 README。"
    ),
}


HISTORICAL_OUTPUT_DIRS = {
    "part1_local_lidar_frame_11338",
    "slot_constrained_box_scoring_pose_corrected_final",
    "slot_hybrid_3d_pose_corrected_v2_replay",
    "slot_part2_clean_agent_shadow_v1",
}


CURRENT_OUTPUT_DIRS = {
    "frame_map_dataset",
    "frame_map_dataset_pose_corrected_final",
    "full_icpark_allframes_vehicle_cluster",
    "part1_local_lidar_frame_9277",
    "pose_drift_correction_v3",
}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _is_cache(parts: tuple[str, ...], suffix: str) -> bool:
    return bool(
        "__pycache__" in parts
        or ".pytest_cache" in parts
        or ".ipynb_checkpoints" in parts
        or suffix == ".pyc"
    )


def _is_git_metadata(parts: tuple[str, ...]) -> bool:
    return ".git" in parts or bool(
        parts and parts[-1] in {".git", ".gitignore", ".gitmodules", ".gitattributes"}
    )


def category_for(relative_path: str) -> str:
    path = Path(relative_path)
    parts = path.parts
    suffix = path.suffix.lower()
    if _is_cache(parts, suffix):
        return "缓存/自动副本"
    if _is_git_metadata(parts):
        return "版本控制元数据"
    if suffix in {".orig", ".rej", ".bak"}:
        return "历史备份"
    if relative_path in {"-0.55", "{z_summary[suggested_first_pass_non_ground_threshold]}"}:
        return "异常临时文件"
    if not parts:
        return "其他"
    top = parts[0]
    if top == "external":
        return "第三方依赖"
    if top == "protected_artifacts":
        return "受保护人工数据"
    if top == "tests":
        return "测试/夹具"
    if top == "reports":
        return "历史审计报告"
    if top == "outputs":
        output_group = parts[1] if len(parts) > 1 else ""
        if output_group in HISTORICAL_OUTPUT_DIRS:
            return "历史/实验输出"
        if output_group in CURRENT_OUTPUT_DIRS:
            return "当前输入/输出"
        return "派生输出"
    if top == "scripts":
        return "CLI/构建/审计脚本"
    if top in {
        "parking_pose_correction",
        "parking_slot_box_scoring",
        "parking_slot_hybrid_3d",
        "parking_slot_part2",
        "parking_slot_validation",
    }:
        return "Python 源码"
    if top in {"docs", ".superpowers"}:
        return "历史设计记录" if top == ".superpowers" or "superpowers" in parts else "当前文档"
    if top in {"configs", "fast_lio_kitti"}:
        return "配置/适配"
    if len(parts) == 1 and suffix in {".gltf", ".obj", ".csv"}:
        return "上游输入/地图资产"
    if len(parts) == 1 and suffix in {".png", ".jpg", ".jpeg"}:
        return "参考图片"
    if suffix in {".md", ".txt", ".pdf"}:
        return "文档"
    return "其他"


def _module_docstring(path: Path) -> str | None:
    if path.suffix.lower() != ".py" or path.stat().st_size > 2_000_000:
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        text = ast.get_docstring(tree)
    except (OSError, SyntaxError, UnicodeError):
        return None
    if not text:
        return None
    first = " ".join(text.strip().splitlines())
    return first[:240] + ("…" if len(first) > 240 else "")


def _checkpoint_description(path: Path) -> str:
    name = path.name.replace("-checkpoint", "")
    return f"Jupyter 自动保存的 {name} 副本；可能落后于正式文件，不应作为运行输入。"


def _cache_description(path: Path) -> str:
    if ".ipynb_checkpoints" in path.parts:
        return _checkpoint_description(path)
    if path.suffix.lower() == ".pyc":
        source = path.name.split(".cpython-")[0].split(".pytest-")[0] + ".py"
        return f"Python/pytest 为 {source} 自动生成的字节码缓存；可删除后重建，不是源代码。"
    if ".pytest_cache" in path.parts:
        return "pytest 自动生成的测试发现、失败记录或缓存元数据；可删除后重建。"
    return "工具自动生成的缓存文件，不是正式源代码或正式结果。"


def _git_description(path: Path) -> str:
    if path.name == ".gitignore":
        return "Git 忽略规则，避免第三方 FAST-LIO 的构建物或临时文件进入版本历史。"
    if path.name == ".gitmodules":
        return "第三方 FAST-LIO 上游仓库的 Git submodule 声明。"
    if path.name == ".git":
        return "第三方源码内嵌依赖的 Git submodule 指针元数据，不参与算法运行。"
    suffix = path.suffix.lower()
    if suffix == ".sample":
        return "vendored 第三方仓库的 Git hook 示例，不参与 ParkingAgent 运行。"
    if suffix in {".pack", ".idx"}:
        return "vendored 第三方仓库的 Git 对象压缩包/索引，仅用于版本历史。"
    return "vendored 第三方仓库的 Git 版本控制元数据，不参与算法运行。"


def _describe_output(path: Path) -> str:
    parts = path.parts
    name = path.name
    group = parts[1] if len(parts) > 1 else "outputs"
    if len(parts) >= 4 and parts[2] == "map_points" and path.suffix.lower() == ".npz":
        frame = path.stem
        if group == "frame_map_dataset_pose_corrected_final":
            return f"第 {frame} 帧应用 pose v3 修正后的 map 坐标点云（points_map_xyzi）；当前 Part1 直接输入。"
        return f"第 {frame} 帧按原始 pose v3 构建的 map 坐标点云；保留用于校正前后对照和重建。"
    if "part2_lidar_evidence" in parts and path.suffix.lower() == ".npz":
        return "按 SHA-256 内容寻址的 unknown 车位多帧 LiDAR 证据包，供 Part2 受界工具读取。"
    if "tool_artifacts" in parts and name.startswith("attempt_") and path.suffix.lower() == ".json":
        return "一次实际 Part2 证据工具调用的内容寻址结构化结果。"
    if "media" in parts and re.fullmatch(r"[0-9a-f]{64}\.png", name):
        return "Part2 私有、内容寻址的 LiDAR 三联证据图；公开 trace 只引用不透明 evidence id。"
    if "review" in parts and re.match(r"\d+_lidar_\d+_camera_\d+\.png", name):
        return "pose 校正 review 中一个关键帧的 LiDAR/Camera 对应复核图。"
    if "review" in parts and "pointcloud_map_compare" in name:
        return "pose 校正 review 中一个关键帧的点云与结构地图配准对比图。"
    if name in OUTPUT_BASENAME_DESCRIPTIONS:
        if group in HISTORICAL_OUTPUT_DIRS:
            base = HISTORICAL_OUTPUT_OVERRIDES.get(name, OUTPUT_BASENAME_DESCRIPTIONS[name])
            return base + " 本文件属于历史/SHADOW 运行，不代表当前局部主线或 GT。"
        return OUTPUT_BASENAME_DESCRIPTIONS[name]
    if name == "README.md":
        role = OUTPUT_DIRECTORY_ROLES.get(group, "该派生产物目录")
        return role + " 此文件提供人读使用说明。"
    if re.fullmatch(r"[0-9a-f]{64}\.png", name):
        return "由内容哈希命名的 Part2/审计可视化资源；具体身份由同目录 manifest 绑定。"
    if path.suffix.lower() == ".npz":
        return "该输出流程生成的压缩 NumPy 点云/证据数组；字段含义由同目录 manifest 或 JSON 引用定义。"
    return _generic_description(path, context=f"{group} 派生流程")


def _describe_protected(path: Path) -> str:
    name = path.name
    relative = path.as_posix()
    if relative in PROTECTED_DESCRIPTIONS:
        return PROTECTED_DESCRIPTIONS[relative]
    if name in PROTECTED_DESCRIPTIONS:
        return PROTECTED_DESCRIPTIONS[name]
    if re.search(r"_slot_\d+_map\.png$", name):
        match = re.search(r"(\d+)_slot_(\d+)_map\.png$", name)
        detail = f"case {match.group(1)} / slot_{match.group(2)}" if match else "对应 case"
        return f"受保护人工核验的 {detail} 局部车位地图上下文图。"
    if "_lidar_" in name and "_camera_" in name and path.suffix.lower() == ".png":
        match = re.search(r"(\d+)_slot_(\d+)_([0-9]+)_lidar_(\d+)_camera_(\d+)\.png$", name)
        if match:
            return (
                f"受保护人工核验 case {match.group(1)} / slot_{match.group(2)} 的第 {match.group(3)} 张"
                f" Camera-LiDAR overlay（LiDAR {match.group(4)}，Camera {match.group(5)}）。"
            )
        return "受保护人工核验的 Camera-LiDAR overlay 证据图。"
    return _generic_description(path, context="受保护人工核验")


def _describe_external(path: Path) -> str:
    relative = path.as_posix()
    if relative in EXTERNAL_PATH_DESCRIPTIONS:
        return EXTERNAL_PATH_DESCRIPTIONS[relative]
    name = path.name
    if name in EXTERNAL_EXACT:
        return EXTERNAL_EXACT[name]
    if path.suffix.lower() in {".yaml", ".yml"} and "config" in path.parts:
        return f"第三方 FAST-LIO 针对 {path.stem} 传感器/数据源的 topic、噪声、外参和 mapping 参数。"
    if path.suffix.lower() == ".launch":
        return f"第三方 FAST-LIO 的 {path.stem} ROS 启动文件，负责加载配置并启动 mapping/RViz。"
    if "doc" in path.parts and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
        return "第三方 FAST-LIO README/论文使用的实验平台、流程或建图结果图片。"
    if "include" in path.parts and path.suffix.lower() in {".h", ".hpp", ".cpp"}:
        return "第三方 FAST-LIO/IKFoM/MTK 的数学、滤波、流形或增量树实现文件。"
    if path.suffix.lower() in {".m", ".py"} and "Log" in path.parts:
        return "第三方 FAST-LIO 运行日志的绘图或耗时分析工具。"
    return _generic_description(path, context="vendored FAST-LIO 第三方工程")


def _generic_description(path: Path, *, context: str = "项目") -> str:
    suffix = path.suffix.lower()
    descriptions = {
        ".json": f"{context}使用的结构化 JSON 配置、索引、结果或 manifest；具体字段见文件内容/生产脚本。",
        ".jsonl": f"{context}生成的逐行 JSON 事件或审计轨迹。",
        ".csv": f"{context}使用的表格输入、逐帧索引、指标或审计结果。",
        ".npz": f"{context}生成的压缩 NumPy 点云、数组或证据缓存。",
        ".png": f"{context}生成或保存的 PNG 地图、诊断、证据或参考图片。",
        ".jpg": f"{context}保存的 JPEG 参考或实验图片。",
        ".jpeg": f"{context}保存的 JPEG 参考或实验图片。",
        ".gif": f"{context}保存的动画演示资源。",
        ".svg": f"{context}保存的矢量流程图或可视化资源。",
        ".pdf": f"{context}的人读 PDF 文档或报告。",
        ".html": f"{context}的人读 HTML 页面、报告或本地界面。",
        ".md": f"{context}的 Markdown 说明、设计、计划或报告。",
        ".txt": f"{context}的纯文本输入、摘要或说明。",
        ".py": f"{context}的 Python 源文件；模块用途可结合文件名和入口参数查看。",
        ".cpp": f"{context}的 C++ 实现文件。",
        ".hpp": f"{context}的 C++ 模板/头文件。",
        ".h": f"{context}的 C/C++ 头文件。",
        ".yaml": f"{context}的 YAML 配置。",
        ".yml": f"{context}的 YAML 配置。",
        ".launch": f"{context}的 ROS launch 启动配置。",
        ".rviz": f"{context}的 RViz 显示布局。",
        ".xml": f"{context}的 XML/ROS 包配置。",
        ".pcd": f"{context}的 PCD 点云资源。",
        ".ppm": f"{context}的无压缩 RGB 测试图片。",
        ".obj": f"{context}的 OBJ 三维网格地图资产。",
        ".gltf": f"{context}的 glTF 三维结构地图资产。",
    }
    return descriptions.get(suffix, f"{context}中的辅助、元数据或二进制文件；用途由所在目录和生产工具决定。")


def describe_file(relative_path: str) -> str:
    if relative_path in EXACT_DESCRIPTIONS:
        return EXACT_DESCRIPTIONS[relative_path]
    if relative_path in FIXTURE_DESCRIPTIONS:
        return FIXTURE_DESCRIPTIONS[relative_path]
    if relative_path in REPORT_DESCRIPTIONS:
        return REPORT_DESCRIPTIONS[relative_path]
    path = Path(relative_path)
    parts = path.parts
    suffix = path.suffix.lower()
    if _is_cache(parts, suffix):
        return _cache_description(path)
    if _is_git_metadata(parts):
        return _git_description(path)
    if suffix == ".orig":
        source = path.with_suffix("").name
        return f"{source} 的编辑/补丁历史备份；正式运行和 unittest discovery 不读取。"
    if suffix == ".rej":
        return "补丁未能自动应用时留下的 reject 片段；不是正式源码。"
    if suffix == ".bak":
        return "对应主文件的历史备份；非权威输入，但可能用于恢复。"
    if not parts:
        return "项目辅助文件。"
    top = parts[0]
    if top == "scripts" and suffix == ".py":
        description = SCRIPT_DESCRIPTIONS.get(path.name)
        if description:
            return description
        docstring = _module_docstring(ROOT / path)
        return docstring or _generic_description(path, context="ParkingAgent 脚本")
    if top == "tests" and suffix == ".py":
        description = TEST_DESCRIPTIONS.get(path.name)
        if description:
            return description
        return "验证对应模块/脚本的行为、错误边界和回归合同。"
    if top == "outputs":
        return _describe_output(path)
    if top == "protected_artifacts":
        return _describe_protected(path)
    if top == "external":
        return _describe_external(path)
    if top == "reports":
        return _generic_description(path, context="较早离线审计")
    if top == ".superpowers" or (top == "docs" and "superpowers" in parts):
        kind = "实现计划" if "plans" in parts else "设计规格/历史任务记录"
        return f"早期 SDD {kind}：{path.stem.replace('-', ' ')}；用于追溯，不是运行时输入。"
    if top in {
        "parking_pose_correction",
        "parking_slot_box_scoring",
        "parking_slot_hybrid_3d",
        "parking_slot_part2",
        "parking_slot_validation",
    } and suffix == ".py":
        docstring = _module_docstring(ROOT / path)
        return docstring or _generic_description(path, context="ParkingAgent Python 包")
    return _generic_description(path)


def _iter_regular_files(excluded_relative: str) -> Iterable[Path]:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = _relative(path)
        if relative == excluded_relative:
            continue
        yield path


def _inventory_paths(output: Path) -> list[str]:
    try:
        output_relative = _relative(output)
    except ValueError as exc:
        raise ValueError("output must be inside the ParkingAgent project root") from exc
    paths = [_relative(path) for path in _iter_regular_files(output_relative)]
    paths.append(output_relative)
    if len(paths) != len(set(paths)):
        raise AssertionError("duplicate file path while building the inventory")
    if any("\t" in path or "\n" in path for path in paths):
        raise ValueError("file paths containing tab/newline are not supported")
    return sorted(paths, key=lambda value: (value.casefold(), value))


def _build_header(paths: list[str], output_relative: str) -> list[str]:
    top_counts = Counter(Path(path).parts[0] if Path(path).parts else "." for path in paths)
    category_counts = Counter(category_for(path) for path in paths)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "ParkingAgent 每个文件是做什么的（中文全量清单）",
        "=" * 78,
        f"生成时间：{generated}",
        f"项目根目录：{ROOT}",
        f"清单文件：{output_relative}",
        f"覆盖普通文件：{len(paths)} 个（包含本清单自身）",
        "",
        "一、怎样阅读",
        "- 前半部分给出目录职责、当前/历史边界和已知异常，适合快速理解项目。",
        "- 最后一部分按相对路径逐个列出所有普通文件；可直接搜索文件名。",
        "- 2.34 万余个逐帧 NPZ 和 2000 余张人工证据图也逐个列名，但同模式文件共享用途规则。",
        "- 对二进制/生成物的说明来自目录、文件名、manifest 和生产脚本，不把它们误称为人工真值。",
        "- .pyc、.pytest_cache、.ipynb_checkpoints、.orig/.bak 和 vendored .git 元数据会明确标为非正式文件。",
        "",
        "二、顶层目录职责",
    ]
    for name, role in DIRECTORY_ROLES.items():
        count = top_counts.get(name, 0)
        lines.append(f"- {name}/：{role}（{count} 个文件）")
    root_file_count = sum(1 for path in paths if len(Path(path).parts) == 1)
    lines.extend(
        [
            f"- 根目录普通文件：地图资产、里程计输入、主 README、参考截图和本清单（{root_file_count} 个）。",
            "",
            "三、outputs/ 一级目录边界",
        ]
    )
    for name, role in OUTPUT_DIRECTORY_ROLES.items():
        count = sum(1 for path in paths if path.startswith(f"outputs/{name}/"))
        lines.append(f"- outputs/{name}/：{role}（{count} 个文件）")
    lines.extend(
        [
            "",
            "四、重点提醒",
            "- 当前正式 Part1 是 outputs/part1_local_lidar_frame_9277/ 的因果 15 帧局部地图；旧整路线 replay 不是当前全场真值。",
            "- outputs/frame_map_dataset_pose_corrected_final/ 是当前 Part1 直接读取的逐帧 map 点云；未修正目录保留作 before/after。",
            "- protected_artifacts/ 是受保护人工数据；human_labels.json.bak 与未被当前 manifest 引用的旧图片也不能直接清理。",
            "- reports/ 内容大多早于当前 Hybrid 局部主线，其中旧状态词汇和数字只能作历史审计。",
            "- outputs/**/.ipynb_checkpoints、tests/__pycache__ 和 scripts/*.orig 都不是正式接口。",
            "- outputs/frame_map_dataset/metadata.json 含一条已不存在的旧 aligned_trajectory provenance 路径；数据本体仍在，但从零复现前应修复来源记录。",
            "- 根目录 -0.55 与 {z_summary[suggested_first_pass_non_ground_threshold]} 均为零字节误生成文件。",
            "",
            "五、按类别计数",
        ]
    )
    for name, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}：{count}")
    lines.extend(
        [
            "",
            "六、全量逐文件用途索引",
            "格式：FILE<TAB>序号<TAB>类别<TAB>相对路径<TAB>用途",
            "",
        ]
    )
    return lines


def build_guide(output: Path) -> tuple[int, int]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output_relative = _relative(output)
    paths = _inventory_paths(output)
    lines = _build_header(paths, output_relative)
    for index, relative_path in enumerate(paths, start=1):
        lines.append(
            "\t".join(
                [
                    "FILE",
                    f"{index:06d}",
                    category_for(relative_path),
                    relative_path,
                    describe_file(relative_path),
                ]
            )
        )
    lines.extend(
        [
            "",
            "七、覆盖校验",
            f"- 逐文件索引条目数：{len(paths)}",
            "- 校验规则：项目内每个普通文件恰好出现一次；目录本身不算文件。",
            "- 重新生成：python3 scripts/generate_parkingagent_file_guide.py",
            "",
        ]
    )
    text = "\n".join(lines)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)

    actual = {_relative(path) for path in ROOT.rglob("*") if path.is_file()}
    listed = set(paths)
    if actual != listed:
        missing = sorted(actual - listed)
        extra = sorted(listed - actual)
        raise AssertionError(f"inventory mismatch: missing={missing[:5]} extra={extra[:5]}")
    parsed = {
        line.split("\t", 4)[3]
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.startswith("FILE\t")
    }
    if parsed != actual or len(parsed) != len(paths):
        raise AssertionError("written guide does not contain every file exactly once")
    return len(paths), output.stat().st_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    count, size = build_guide(args.output)
    print(f"{args.output.resolve()}\tfiles={count}\tbytes={size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
