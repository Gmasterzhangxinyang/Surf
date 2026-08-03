"""Build a detailed bilingual paper package from audited Part2 artifacts.

The generated manuscript is deliberately explicit about the lack of human
ground truth.  It reports selective resolution coverage, not classification
accuracy, and keeps every numerical claim tied to a checked JSON artifact.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from PIL import Image

from .reporting_cvpr_figures import build_cvpr_figure_suite
from .reporting_story_zh import FONT_CJK, FONT_FALLBACK


TITLE_ZH = "严格因果时序几何与硬门控多模态智能体用于停车位占用不确定性消解"
TITLE_EN = "Resolving Parking-Slot Occupancy Uncertainty with Causal Temporal Geometry and Hard-Gated Multimodal Agents"


REFERENCES: tuple[dict[str, str], ...] = (
    {"key": "hornung2013octomap", "text": "Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C. & Burgard, W. OctoMap: an efficient probabilistic 3D mapping framework based on octrees. Autonomous Robots 34, 189–206 (2013).", "url": "https://doi.org/10.1007/s10514-012-9321-0"},
    {"key": "xu2022fastlio2", "text": "Xu, W. et al. FAST-LIO2: Fast direct LiDAR-inertial odometry. IEEE Transactions on Robotics 38, 2053–2073 (2022).", "url": "https://doi.org/10.1109/TRO.2022.3141876"},
    {"key": "lang2019pointpillars", "text": "Lang, A. H. et al. PointPillars: Fast encoders for object detection from point clouds. CVPR, 12697–12705 (2019).", "url": "https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html"},
    {"key": "caesar2020nuscenes", "text": "Caesar, H. et al. nuScenes: A multimodal dataset for autonomous driving. CVPR, 11621–11631 (2020).", "url": "https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html"},
    {"key": "li2022bevformer", "text": "Li, Z. et al. BEVFormer: Learning bird's-eye-view representation from multi-camera images via spatiotemporal transformers. ECCV, 1–18 (2022).", "url": "https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/694_ECCV_2022_paper.php"},
    {"key": "man2023bevguide", "text": "Man, Y., Gui, L.-Y. & Wang, Y.-X. BEV-guided multi-modality fusion for driving perception. CVPR, 21960–21969 (2023).", "url": "https://openaccess.thecvf.com/content/CVPR2023/html/Man_BEV-Guided_Multi-Modality_Fusion_for_Driving_Perception_CVPR_2023_paper.html"},
    {"key": "wei2023surroundocc", "text": "Wei, Y. et al. SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving. ICCV, 21729–21740 (2023).", "url": "https://openaccess.thecvf.com/content/ICCV2023/html/Wei_SurroundOcc_Multi-camera_3D_Occupancy_Prediction_for_Autonomous_Driving_ICCV_2023_paper.html"},
    {"key": "tian2023occ3d", "text": "Tian, X. et al. Occ3D: A large-scale 3D occupancy prediction benchmark for autonomous driving. NeurIPS 36 (2023).", "url": "https://papers.nips.cc/paper/2023/hash/cabfaeecaae7d6540ee797a66f0130b0-Abstract-Datasets_and_Benchmarks.html"},
    {"key": "huang2024selfocc", "text": "Huang, Y., Zheng, W., Zhang, B., Zhou, J. & Lu, J. SelfOcc: Self-supervised vision-based 3D occupancy prediction. CVPR, 19946–19956 (2024).", "url": "https://openaccess.thecvf.com/content/CVPR2024/html/Huang_SelfOcc_Self-Supervised_Vision-Based_3D_Occupancy_Prediction_CVPR_2024_paper.html"},
    {"key": "shi2024streamingflow", "text": "Shi, Y. et al. StreamingFlow: Streaming occupancy forecasting with asynchronous multi-modal data streams via neural ordinary differential equation. CVPR, 14833–14842 (2024).", "url": "https://openaccess.thecvf.com/content/CVPR2024/html/Shi_StreamingFlow_Streaming_Occupancy_Forecasting_with_Asynchronous_Multi-modal_Data_Streams_via_CVPR_2024_paper.html"},
    {"key": "zhang2024occfusion", "text": "Zhang, J., Ding, Y. & Liu, Z. OccFusion: Depth estimation free multi-sensor fusion for 3D occupancy prediction. ACCV, 3587–3604 (2024).", "url": "https://openaccess.thecvf.com/content/ACCV2024/html/Zhang_OccFusion_Depth_Estimation_Free_Multi-sensor_Fusion_for_3D_Occupancy_Prediction_ACCV_2024_paper.html"},
    {"key": "duan2025sdgocc", "text": "Duan, Z. et al. SDGOCC: Semantic and depth-guided bird's-eye view transformation for 3D multimodal occupancy prediction. CVPR, 6751–6760 (2025).", "url": "https://openaccess.thecvf.com/content/CVPR2025/html/Duan_SDGOCC_Semantic_and_Depth-Guided_Birds-Eye_View_Transformation_for_3D_Multimodal_CVPR_2025_paper.html"},
    {"key": "leng2025stocc", "text": "Leng, Z., Yang, J., Yi, W. & Zhou, B. Occupancy learning with spatiotemporal memory. ICCV, 26569–26578 (2025).", "url": "https://openaccess.thecvf.com/content/ICCV2025/html/Leng_Occupancy_Learning_with_Spatiotemporal_Memory_ICCV_2025_paper.html"},
    {"key": "ahrnbom2016parking", "text": "Ahrnbom, M., Åström, K. & Nilsson, M. Fast classification of empty and occupied parking spaces using integral channel features. CVPR Workshops, 9–15 (2016).", "url": "https://openaccess.thecvf.com/content_cvpr_2016_workshops/w25/html/Ahrnbom_Fast_Classification_of_CVPR_2016_paper.html"},
    {"key": "suhr2020parking", "text": "Suhr, J. K. & Jung, H. G. End-to-end trainable one-stage parking slot detection integrating global and local information. arXiv:2003.02445 (2020).", "url": "https://arxiv.org/abs/2003.02445"},
    {"key": "grbic2023parking", "text": "Grbić, R. & Koch, B. Automatic vision-based parking slot detection and occupancy classification. Expert Systems with Applications 225, 120147 (2023).", "url": "https://arxiv.org/abs/2308.08192"},
    {"key": "yao2023react", "text": "Yao, S. et al. ReAct: Synergizing reasoning and acting in language models. ICLR (2023).", "url": "https://openreview.net/forum?id=WE_vluYUL-X"},
    {"key": "schick2023toolformer", "text": "Schick, T. et al. Toolformer: Language models can teach themselves to use tools. NeurIPS 36 (2023).", "url": "https://proceedings.neurips.cc/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html"},
    {"key": "yang2023gpt4tools", "text": "Yang, R. et al. GPT4Tools: Teaching large language model to use tools via self-instruction. NeurIPS 36 (2023).", "url": "https://proceedings.neurips.cc/paper_files/paper/2023/hash/e393677793767624f2821cec8bdd02f1-Abstract-Conference.html"},
    {"key": "geifman2017selective", "text": "Geifman, Y. & El-Yaniv, R. Selective classification for deep neural networks. NeurIPS 30 (2017).", "url": "https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html"},
    {"key": "geifman2019selectivenet", "text": "Geifman, Y. & El-Yaniv, R. SelectiveNet: A deep neural network with an integrated reject option. ICML, 2151–2159 (2019).", "url": "https://proceedings.mlr.press/v97/geifman19a.html"},
)


def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _reference_markdown() -> str:
    return "\n".join(
        f"{index}. {row['text']} {row['url']}"
        for index, row in enumerate(REFERENCES, 1)
    )


def _bibtex() -> str:
    entries = []
    for row in REFERENCES:
        safe_title = row["text"].replace("&", "and")
        entries.append(
            "@misc{" + row["key"] + ",\n"
            f"  title = {{{safe_title}}},\n"
            f"  howpublished = {{\\url{{{row['url']}}}}},\n"
            "  note = {Bibliographic string verified against the publisher or proceedings page}\n}"
        )
    return "\n\n".join(entries) + "\n"


def _threshold_rows() -> list[dict[str, Any]]:
    return [
        {"group": "Occupied", "criterion": "vehicle height interval", "operator": "within", "threshold": "0.30–2.20 m", "interpretation": "remove ground and implausibly high returns"},
        {"group": "Occupied", "criterion": "candidate points", "operator": ">=", "threshold": 40, "interpretation": "minimum spatial support"},
        {"group": "Occupied", "criterion": "support frames", "operator": ">=", "threshold": 3, "interpretation": "minimum temporal support"},
        {"group": "Occupied", "criterion": "temporal support", "operator": ">=", "threshold": 0.20, "interpretation": "avoid one-frame obstacles"},
        {"group": "Occupied", "criterion": "supported height layers", "operator": ">=", "threshold": 2, "interpretation": "require vertical vehicle-like structure"},
        {"group": "Occupied", "criterion": "z95", "operator": ">=", "threshold": "0.60 m", "interpretation": "reject low curbs or ground clutter"},
        {"group": "Occupied", "criterion": "height span", "operator": ">=", "threshold": "0.35 m", "interpretation": "require non-flat structure"},
        {"group": "Occupied", "criterion": "core overlap", "operator": ">=", "threshold": 0.45, "interpretation": "bind evidence to target slot interior"},
        {"group": "Occupied", "criterion": "adjacent overlap", "operator": "<=", "threshold": 0.35, "interpretation": "limit ownership ambiguity"},
        {"group": "Occupied", "criterion": "boundary ratio", "operator": "<", "threshold": 0.50, "interpretation": "veto walls and slot borders"},
        {"group": "Occupied", "criterion": "linearity risk", "operator": "<", "threshold": 0.60, "interpretation": "veto linear static structures"},
        {"group": "Free", "criterion": "ray frames", "operator": ">=", "threshold": 5, "interpretation": "multi-frame traversal evidence"},
        {"group": "Free", "criterion": "viewpoints", "operator": ">=", "threshold": 2, "interpretation": "avoid a single viewing ray"},
        {"group": "Free", "criterion": "viewpoint separation", "operator": ">=", "threshold": "10 degrees", "interpretation": "geometric baseline"},
        {"group": "Free", "criterion": "observed volume", "operator": ">=", "threshold": 0.70, "interpretation": "positive free-space coverage"},
        {"group": "Free", "criterion": "near-ground BEV coverage", "operator": ">=", "threshold": 0.70, "interpretation": "cover the parking footprint"},
        {"group": "Free", "criterion": "unobserved component", "operator": "<=", "threshold": 0.20, "interpretation": "limit unseen subvolume"},
        {"group": "Free", "criterion": "occlusion ratio", "operator": "<=", "threshold": 0.20, "interpretation": "limit hidden core"},
        {"group": "Robustness", "criterion": "pose variants", "operator": "=", "threshold": "7/7", "interpretation": "original, ±0.20 m x/y, ±0.50 degrees yaw"},
        {"group": "Agent", "criterion": "terminal confidence", "operator": ">=", "threshold": 0.90, "interpretation": "necessary but not sufficient; hard gate also required"},
    ]


def _metrics(window: Mapping[str, Any], baseline: Mapping[str, Any], extended: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    by_window = {int(row["window_frames"]): row for row in window["experiments"]}
    ext_rows = list(extended["experiments"])
    base_rows = list(baseline["experiments"])
    unknown_rows = [row for row in result["slot_results"] if row["case"]["part1_state"] == "unknown"]
    states = Counter(row["case"]["final_state"] for row in unknown_rows)
    pooled_total = sum(int(row["part1_unknown_count"]) for row in ext_rows)
    pooled_resolved = sum(int(row["resolved_count"]) for row in ext_rows)
    return {
        "by_window": by_window,
        "window_counts": [int(by_window[size]["resolved_count"]) for size in (15, 30, 45, 60, 75)],
        "saved_percent": 100.0 * (float(by_window[75]["processing_seconds"]) - float(by_window[60]["processing_seconds"])) / float(by_window[75]["processing_seconds"]),
        "pooled_total": pooled_total,
        "pooled_resolved": pooled_resolved,
        "pooled_rate": pooled_resolved / pooled_total,
        "pooled_free": sum(int(row["gate_state_counts"].get("free", 0)) for row in ext_rows),
        "pooled_occupied": sum(int(row["gate_state_counts"].get("occupied", 0)) for row in ext_rows),
        "base_total": sum(int(row["part1_unknown_count"]) for row in base_rows),
        "base_resolved": sum(int(row["resolved_count"]) for row in base_rows),
        "states": states,
        "model_calls": sum(int(row["model_turns"]) for row in result["slot_results"]),
        "validation_errors": sum(len(row["validation_errors"]) for row in result["slot_results"]),
        "unknown_rows": unknown_rows,
    }


def _zh_sections(m: Mapping[str, Any]) -> list[tuple[str, str]]:
    w = m["by_window"]
    refs = _reference_markdown()
    return [
        ("摘要", f"自动泊车需要在有限时延内区分可用车位、被车辆占据的车位以及证据不足的车位。短时局部点云容易受到稀疏采样、路沿和墙面回波、遮挡、相邻车位归属以及位姿误差影响。本文提出一个保持既有 Free 优先队列和 ReAct 工具工作流不变的选择性占用判断系统。Part1 继续使用 15 帧局部 LiDAR 建立候选；Part2 仅使用决策时刻 t0 及以前的 60 帧，构造与车位身份绑定的目标局部几何证据。多模态智能体可以阅读 Camera、FOV 和 LiDAR 工具结果，但其终态必须同时满足 0.90 分数合同、Free/Occupied 互斥硬门以及 7/7 位姿扰动稳定性。窗口消融在开发锚点 frame 9277 的 22 个原始 Unknown 上得到 15/30/45/60/75 帧分别解决 {m['window_counts'][0]}/{m['window_counts'][1]}/{m['window_counts'][2]}/{m['window_counts'][3]}/{m['window_counts'][4]} 个；60 帧与 75 帧同为 9/22，但节省 {m['saved_percent']:.1f}% 几何处理时间。在排除开发锚点的六个系统采样路线位置上，匹配 15 帧基线解决 {m['base_resolved']}/{m['base_total']}，60 帧解决 {m['pooled_resolved']}/{m['pooled_total']}（{100*m['pooled_rate']:.1f}%；{m['pooled_free']} Free、{m['pooled_occupied']} Occupied）。真实 OpenAI Agent 在开发锚点确认 2 Free、7 Occupied、13 Unknown，结构化校验错误为 0。本文没有人工占用真值，因此报告的是严格证据消歧覆盖率而不是分类准确率。"),
        ("1 引言", "停车位占用判断看似是二分类，部署时却必须保留第三种结果 Unknown。相机可以提供外观和语义，但视角、光照、遮挡以及未完成的车位到像素标定会使目标定位不可靠；LiDAR 提供直接距离和高度，却在远距离、掠射角和结构边界处稀疏。把任意弱信号强行压成 Free 或 Occupied 会提高表面覆盖率，同时增加安全风险。\n\n本文聚焦一个受约束问题：地图已经给出车位多边形，Part1 已在当前 30 m 局部区域产生 Free 与 Unknown 候选，Part2 的任务是按 Free 优先顺序找到一个可信 Free，或安全地保留 Unknown。该问题与全场景稠密占用预测不同，也不同于固定监控相机上的停车位分类。核心挑战是如何在不改变系统控制逻辑的前提下，让历史传感器证据增加可观测性，同时防止语言模型把不完整证据解释成高置信终态。\n\n本文的贡献有四点。第一，提出严格因果的长窗口目标局部证据包：Part1 的 15 帧合同不变，Part2 额外使用截至同一 t0 的历史帧，且每个包绑定车位、帧文件、配置与地图哈希。第二，定义 Free 与 Occupied 的互斥几何门，将核心点、高度层、边界比例、线性风险、自由空间覆盖和遮挡显式暴露给智能体。第三，使用 7 个位姿扰动进行终态重算，并在 Agent 分数之外执行不可绕过的代码门。第四，提供窗口消融、跨路线配对实验、真实 API 回放、成功案例、保守失败案例和机器可读复现清单。"),
        ("2 相关工作", "2.1 停车位检测与占用分类。固定摄像头方法通常在已知或自动发现的图像区域内进行二分类。Ahrnbom 等使用通道特征处理大规模停车图像[14]；Suhr 与 Jung 将车位入口、类型、占用和局部角点统一到单阶段网络[15]；Grbić 与 Koch 在鸟瞰聚类后使用 ResNet 进行占用分类[16]。这些工作说明视觉分类在标定良好的固定视角上有效，但本文面对移动平台、地图车位、视野变化与不完整跨传感器标定。\n\n2.2 三维占用与 BEV。OctoMap 显式建模 free、occupied 和 unknown[1]，其三态语义与本文的拒判设计一致。PointPillars 展示了柱状点云表示的效率[3]。nuScenes 推动了 Camera、LiDAR 与 Radar 的联合研究[4]；BEVFormer 使用时空 Transformer 形成 BEV 表示[5]；BEVGuide 在 BEV 中统一多传感器[6]；SurroundOcc、Occ3D 与 SelfOcc 将任务扩展至稠密三维占用[7–9]。本文不训练稠密网络，而是在少量已知车位内构建可解释的目标级证据。\n\n2.3 时序和异步融合。StreamingFlow 研究异步数据流中的持续占用预测[10]，OccFusion 与 SDGOCC 讨论多传感器融合中的深度和几何对齐[11,12]，ST-Occ 使用时空记忆提高多帧一致性[13]。这些研究共同支持历史观测改善空间覆盖的动机；本文进一步强调严格 t≤t0、窗口消融和位姿扰动重算。\n\n2.4 工具型智能体与选择性预测。ReAct 将推理和动作交错[17]，Toolformer 与 GPT4Tools 展示了模型调用外部工具的能力[18,19]。然而工具调用不能提供安全保证。选择性分类通过拒判交换覆盖率与风险[20,21]。本文据此把 VLM 定位为证据解释器，把 Unknown 视为合法安全输出，并将终态权力保留在确定性状态机。\n\n2.5 研究空白。已有停车研究主要优化标签准确率，三维占用研究主要优化全场景 mIoU，智能体研究主要优化任务成功率。三者较少共同回答：当地图车位已经给定、传感器证据稀疏且没有充分标定时，如何在工具工作流中安全地减少 Unknown，并让每个状态变化可审计。"),
        ("3 问题定义", "设当前决策时刻为 t0，局部车位集合为 S={s_i}。每个车位具有地图多边形 P_i、Part1 状态 y_i^(1)∈{Free, Occupied, Unknown}、未校准证据分数以及传感器资源。正式候选队列只包含 Part1 Free 和 Unknown，并按 Free 优先、同类保持确定性顺序排列。\n\nPart2 输出 y_i^(2)∈{Free, Occupied, Unknown}。本文不优化把所有样本强制分类的准确率，而优化选择性消歧覆盖率 R=N_resolved/N_unknown，其中 N_resolved 是原始 Unknown 中通过终态硬门的数量。没有人工 GT 时，R 不能解释为 accuracy。\n\n系统满足四个约束：(1) 因果性：所有证据帧 f 必须满足 f≤t0；(2) 身份绑定：证据包必须匹配 slot_id、地图哈希、配置哈希和源文件哈希；(3) 互斥性：Free 与 Occupied 不能同时成为合法终态；(4) 控制一致性：默认运行在第一个 Free 分数≥0.90且通过硬门时停止，其余情况继续队列。"),
        ("4 系统总览", "系统由 Part1、FOV 路由、单车位 Agent 和全局队列四层组成。Part1 在 30 m 区域内使用 15 帧局部 LiDAR，输出地图上下文和按 Free→Unknown 排列的 SlotCase。每个 SlotCase 保留初态、未解决原因、工具证据、每轮可审计摘要、最终分数和终止原因。\n\nAgent 必须首先执行 Camera FOV 检查并写回 SlotCase。若 Camera 不可用或目标不在可用视域，Camera 工具被代码禁用；若 Camera 可用，模型可请求上下文、序列或裁剪。LiDAR 细查从身份绑定的 60 帧资源加载数值卡和可视化。单车位最多三次证据工具调用，模型动作必须符合严格 JSON schema。\n\n模型提出 Final 后，状态机检查分数阈值和几何门。Free 合法时全局立即早停；Occupied 合法时记录并继续；不满足门、证据冲突或轮次耗尽时保留 Unknown。评测模式必须显式启用 evaluation-exhaustive，避免把为统计而处理完整队列的行为误写成线上工作流。"),
        ("5 严格因果时序证据", "5.1 帧窗构造。对于窗口长度 W，从 pose-corrected 帧索引中选择不晚于 t0 的最后 W 帧。若不足 W 帧、末帧不是 t0 或任一源点云缺失，则构建失败，不以 Unknown 掩盖系统错误。本研究比较 W∈{15,30,45,60,75}。\n\n5.2 坐标统一。每帧点云通过校正位姿变换到地图坐标，再以目标车位长短轴构建局部坐标。x/y 表示沿车位和横跨车位方向，z 相对局部地面。证据包同时保存每点 frame_id，使空间结构可以追溯到时间支持。\n\n5.3 资源身份。每个 NPZ 包绑定 task_id、encounter_id、dataset_id、slot_id、anchor frame、配置哈希、车位地图哈希和所有源文件哈希。工具加载时重新验证 slot_id 和合同，防止相邻车位或旧实验媒体被错误复用。\n\n5.4 为什么长窗口有效。短窗口常只看见路沿或车体一侧，无法同时获得自由空间穿越和立体高度。历史轨迹改变观察原点，增加视角分离、核心体积覆盖和被遮挡区域的可见机会。但窗口过长会增加计算和动态场景不一致，所以必须通过消融选择，而不是默认越长越好。"),
        ("6 Occupied 几何门", "Occupied 证据只使用车位局部车辆高度区间 0.30–2.20 m。候选必须至少有 40 个点、3 个支持帧、0.20 时序支持和两个高度层；z95 不低于 0.60 m，高度跨度不低于 0.35 m。目标核心重叠必须≥0.45，相邻车位重叠≤0.35。\n\n边界比例 b=N_boundary/N_candidate 必须小于 0.50，避免把墙、路沿和标线边缘当作车。二维 PCA 线性风险 l 必须小于 0.60；同时使用裁剪 5%–95% 后的 robust pillar 检查，抑制少量离群点。终态门还要求核心内部至少一个障碍点，Free 强门不得同时成立。所有阈值均来自固定配置并写入 Table 4，而不是由 VLM 临时解释。"),
        ("7 Free 几何门", "Free 不是“没有点”，而是射线明确穿过本应容纳车辆的体积。系统体素化每个观测原点到回波终点的射线，计算目标核心的观测体积比例、近地 BEV 覆盖、未观测连通体和遮挡比例。\n\n强 Free 要求至少 5 个射线帧、2 个视点且视角分离≥10°；观测体积和近地覆盖均≥0.70；未观测连通体和遮挡均≤0.20；不存在未解决核心命中。Occupied 强门必须被边界、线性或其他失败条件否决。这个定义防止把稀疏、被遮挡或仅看到车位入口的区域错误标为空闲。"),
        ("8 位姿稳健性与互斥终态", "定位误差会把边界点推入车位核心，也会让射线覆盖虚增。对每个候选终态，系统重新计算七个变体：原始位姿、x±0.20 m、y±0.20 m、yaw±0.50°。上游 stability 字段允许用于诊断的 0.80 比例，但 Part2 终态硬门更严格，要求 7/7 全部通过。\n\n最终 Free_eligible = FreeStrong ∧ ¬CoreHit ∧ OccupiedVeto ∧ Stable7/7；Occupied_eligible = OccupiedStrong ∧ b<0.5 ∧ l<0.6 ∧ CorePoints>0 ∧ ¬FreeStrong ∧ Stable7/7。任一门失败即 Unknown。slot_1251 在中心几何上偏向 Occupied，但只通过 6/7 扰动，因此被拒绝，是硬门避免高分错误的关键案例。"),
        ("9 多模态 Agent 与状态机", "VLM 接收的是受限 SlotCase，而不是整条路线的自由文本。请求包含目标车位、Part1 小型数值摘要、FOV、可用工具、已有 Observation 和合法动作 schema。模型可以请求 camera_context、camera_sequence、camera_crop 或 lidar_detail；不可见时只允许 LiDAR。\n\n每个工具返回结构化 EvidenceRecord，包括工具名、轮次、状态、摘要、reason codes、元数据、审计媒体和模型实际看见的图片路径。模型给出 Final 时必须返回 Free/Occupied/Unknown 三分数、引用证据和简洁理由。代码不保存或展示不可验证的隐藏思维链；论文只使用工具 Observation、动作、硬门和终态理由。\n\n0.90 是工作流合同而非校准概率。分数达到阈值仍必须通过几何门；未通过时状态机保留 Unknown 并记录 blocker。API 失败、schema 错误和媒体缺失是系统错误，不能被静默改写成语义 Unknown。"),
        ("10 实验设置", "数据来自本地停车路线、pose-corrected-final 帧索引、逐帧地图点云和 1,397 个已知车位几何库。开发锚点为 frame 9277，Part1 在 30 m 内产生 24 个候选，其中 22 个为原始 Unknown。\n\n窗口消融固定候选、地图、几何门和代码，只改变 W。跨路线实验从既有 30 锚点 manifest 按索引 1、7、13、19、25、30 系统采样 frame 234、3160、5283、7605、9443、11678，并排除开发锚点。每个锚点的 15 帧与 60 帧使用完全相同的 Part1 Unknown 身份。\n\n真实 Agent 验收使用固定 OpenAI 配置和结构化输出。exhaustive 运行用于统计全部候选；operational 运行验证第一个可信 Free 早停。主要指标是 resolved count/rate、Free/Occupied 组成、模型调用、schema 错误和几何耗时。没有 GT，因此不计算 accuracy、precision、recall、F1、AUROC 或 p 值。"),
        ("10.1 数据链、样本身份与防泄漏", "实验的最小数据链从 pose-corrected-final/frames.csv 开始。该索引为每个 frame_id 提供校正后位姿和点云路径；slot_database.json 提供固定的车位 polygon、相邻关系与地图尺度。Part1 输出不是通过搜索最终结果重新生成的样本表，而是保存 SceneSnapshot、candidate queue 和每个 SlotCase 初态的不可变交接文件。Part2 只能在该队列上增加资源和证据，不能重排或删除不利样本。\n\n开发锚点 frame 9277 用于观察窗口长度的边际收益。跨路线锚点从此前已经存在的 30-anchor manifest 通过固定索引系统采样，而不是根据 60 帧结果选择。开发锚点被明确排除，避免同一位置同时承担方法开发和路线外验证。15 帧基线与 60 帧扩展在每个锚点共享完全相同的 slot_id 集合；审计器逐锚点比较集合相等性，因此 resolved 数量差异不能来自候选筛除。\n\n因果泄漏由三层防护处理。第一层在 frames.csv 上只截取 frame_id≤t0 的记录；第二层要求窗口末帧恰好等于 t0 且数量恰好等于 W；第三层在完成后的 24 个 NPZ 包中重新读取 selected_frames，验证 max(frame_id)≤t0。任何一层失败都会终止构建。未来帧、全路线合并地图和旧 shadow 结果均不能作为当前终态证据。\n\n语义泄漏同样受限。旧 30 锚点 Codex 运行只用于说明早期工作流覆盖，使用的是 15 帧证据和不同 Agent，不能与当前 60 帧 OpenAI 结果合并。模型请求不包含人工标签，因为不存在 GT；也不把扩展几何的最终 gate state直接写成要求模型照抄的标签。模型看到数值卡、工具 Observation 和允许动作，代码随后独立重算终态资格。"),
        ("10.2 实现细节与运行协议", "Part1 与 Part2 使用同一 anchor 和同一地图坐标系。扩展构建器把 Hybrid3DConfig 的 frame_stride 固定为 1，并把 window_before/window_after 扩展到足以覆盖请求跨度；实际选择仍严格截断在 t0，不使用 window_after 的未来部分。FramePointProvider 使用受限缓存读取逐帧 NPZ，避免把完整路线点云一次性混入局部证据。\n\n点云首先估计局部地面。车辆候选高度范围为 0.30–2.20 m；低于该范围的近地点用于 free-space traversal，高于范围的结构不进入车辆主体门。BEV 体素分辨率为 0.25 m，垂直体素为 0.20 m。Occupied 候选先执行低成本 gate，再对有限 shortlist 计算形状、PCA、相邻重叠和稳健裁剪，从而控制长窗口成本。\n\nFree 证据通过射线体素遍历得到。每条射线保留观测原点、终点和 frame provenance；只有实际被射线穿越的核心体素才算可观测，不能用 ROI 中“没有回波”替代。遇到强 Occupied、弱障碍未解决、视点不足或遮挡过高时，Free 失败关闭。\n\n每个 Agent case 先强制执行 FOV。FOV 使用地图方位和相机参数近似可视扇区，只做工具路由，不声称像素级精确投影。若 camera-projection audit 未通过，Camera 图像可以帮助描述环境，但不能单独成为 Free/Occupied 终态依据。LiDAR 工具从 pack 中生成三视图和数值证据卡，同时在 metadata 中写入 assess_terminal_geometry 的代码结果。\n\n模型适配器要求结构化动作；非法 tool 名、重复无效工具、非有限分数、字段缺失和 provider 错误都有独立异常路径。结果使用临时文件、fsync 和原子 replace 写入 checkpoint；resume 时核验 snapshot、queue order 和 exhaustive/operational 模式，防止跨实验续跑。"),
        ("10.3 评价指标与统计单位", "主要评价量为严格消歧数 N_resolved，即原始 Part1 Unknown 中满足 Free_eligible 或 Occupied_eligible 的样本数。消歧率 R=N_resolved/N_unknown。状态组成分别记录 N_free、N_occupied 和 N_remaining_unknown，并满足三者之和等于原始 Unknown 总数。模型运行质量记录 validation error 数量、model turns、tool rounds 和 stop reason。\n\n窗口效率比较使用完整几何构建的 wall-clock 时间 T_W。本文报告的 22.0% 节省按 (T_75−T_60)/T_75 计算。它只说明相同容器和相同候选集合上的相对耗时；没有多次独立重复和硬件隔离，因此不报告标准差或吞吐量置信区间。\n\n跨路线汇总保留两级单位。车位级合并数便于描述工程覆盖，但同一 anchor 内的车位共享轨迹、点云和环境结构，统计上相关。只有六个锚点时，基于锚点的非参数区间也会很宽。因此本文列出每个锚点的分子和分母，不进行把 208 个车位当作独立 Bernoulli 试验的显著性检验。\n\n在具有冻结 GT 后，评价应扩展为两组互补指标。第一组是条件分类风险：在系统选择作答的样本上计算 Free 误报、Occupied 漏报、precision、recall 与混淆矩阵；第二组是选择性覆盖：绘制阈值变化下 risk–coverage curve，报告在目标风险上限下的最大覆盖。还应采用 anchor/route cluster bootstrap，而不是普通车位 bootstrap。"),
        ("10.4 算法流程与失败关闭伪代码", "算法输入为 Part1Output、严格因果帧索引、车位数据库和工具集合。首先构建队列 Q=Free candidates + Unknown candidates。对 Q 中每个 SlotCase，执行 FOV 并写回可见性。随后加载目标绑定的 60 帧 LiDAR 包，验证哈希和 t≤t0；验证失败则抛出运行错误。\n\n在单车位循环中，Agent 根据已有 Observation 选择合法工具。工具结果追加到 SlotCase，代码更新 geometry card 和 terminal gate。模型提出 Free 时，只有 score_free≥0.90、score_occupied<0.90 且 free_eligible=true 才能 mark resolved；提出 Occupied 时对应检查 score_occupied、反向分数和 occupied_eligible；其他情况为 Unknown。若还存在合法工具且未满三轮，允许继续收集证据，否则终止当前 case。\n\n全局控制在可信 Free 后立刻返回 slot_id。可信 Occupied 和 Unknown 都进入下一候选。exhaustive 模式只在离线评测中关闭早停，并在结果合同中显式标记 evaluation_exhaustive=true。该标记避免离线“发现多个 Free”的报告被误解为线上同时规划多个目标。\n\n以伪代码表示：BuildQueue；for case in Q：ForceFOV；ValidateExtendedPack；for round≤3：Observe/Act；gate←AssessGeometry；proposal←ModelFinal；if proposal∧gate then Resolve else ContinueOrUnknown；if ResolveFree∧operational then Return；循环结束仍未找到可信 Free 时返回 not_found。任何 provider、schema、媒体或身份错误走 Error 分支，而不是 Unknown 分支。"),
        ("11 结果：窗口消融", f"15、30、45、60、75 帧分别解决 {m['window_counts'][0]}、{m['window_counts'][1]}、{m['window_counts'][2]}、{m['window_counts'][3]}、{m['window_counts'][4]} 个原始 Unknown。30 帧首先产生 2 Free 和 4 Occupied；45 帧增加到 2 Free 和 6 Occupied；60 帧达到 2 Free 和 7 Occupied；75 帧没有新增终态。\n\n几何处理时间从 60 帧的 {float(w[60]['processing_seconds']):.2f} s 增加到 75 帧的 {float(w[75]['processing_seconds']):.2f} s。60 帧与最大窗口具有相同严格消歧数，但节省 {m['saved_percent']:.1f}% 时间，因此是当前数据上的 Pareto 工作点。该结论是单服务器描述性测量，不等于跨硬件基准。"),
        ("12 结果：跨路线复核", f"六个独立路线锚点共有 {m['pooled_total']} 个 Part1 Unknown。匹配 15 帧基线在每个锚点均为 0，合计 {m['base_resolved']}/{m['base_total']}。60 帧在 frame 234、3160、5283、7605、9443、11678 分别解决 5/33、2/21、7/45、12/37、1/21、7/51，合计 {m['pooled_resolved']}/{m['pooled_total']}（{100*m['pooled_rate']:.2f}%），包括 {m['pooled_free']} Free 与 {m['pooled_occupied']} Occupied。\n\n所有六处均为正提升，说明长窗口作用不只存在于 frame 9277。然而开发锚点为 40.9%，跨路线合并仅 16.3%，表明增益依赖轨迹、视角基线和局部结构。车位嵌套于锚点，不能把 208 个候选当作完全独立样本做简单显著性检验。"),
        ("13 结果：真实 Agent 与运营早停", f"60 帧 exhaustive 运行中，22 个原始 Unknown 最终为 {m['states']['free']} Free、{m['states']['occupied']} Occupied、{m['states']['unknown']} Unknown。所有终态与确定性 hard gate 一致，结构化输出错误为 {m['validation_errors']}。完整 24 候选运行共 {m['model_calls']} 次模型调用。\n\n默认运营模式没有启用 exhaustive。队列第一个候选 slot_1038 经一次 LiDAR 工具和一次模型调用达到 Free=0.98，随后立即停止。这证明实验统计开关没有破坏原始“可信 Free 立即停止”控制语义。"),
        ("14 案例研究", "成功 Free：slot_1012 的短窗口受到边界回波干扰；60 帧提供充分体积与近地覆盖，Occupied 候选被边界/线性门否决，最终为 Free=0.98。成功 Occupied：slot_1258 在核心内部具有跨帧、跨高度层的稳定立体回波，边界比例和线性风险均通过，最终 Occupied=0.96。\n\n保守失败：slot_1251 为 6/7 位姿稳定，保持 Unknown；slot_1010 同时存在 Free 与 Occupied 方向证据，互斥门阻止终态；slot_1248 只有部分路线覆盖且 Camera 没有可靠像素投影，仍为 Unknown。这些案例说明剩余 Unknown 不是统一故障，而是稳健性、冲突、遮挡、部分覆盖和传感器路由等不同 blocker。"),
        ("15 讨论", "实验支持一个有限但清楚的主张：在地图车位和短窗口 Unknown 的条件下，严格因果的历史几何能够提高可观测覆盖；明确的互斥与位姿硬门可以把 VLM 限制为证据解释器。提升主要来自传感器几何，而不是让模型自由猜测。\n\n相比端到端占用网络，本方法不需要训练数据，能输出逐门审计，并适合少量高价值候选；代价是覆盖率较低、阈值需要跨数据标定、动态目标可能使长窗口过时。VLM 的价值目前体现在统一读取工具卡、组织理由和遵守队列协议，而不是证明其优于确定性融合。未来带 GT 的无 VLM/有 VLM对比是必要实验。\n\n60 帧不是普遍最优常数。它是本路线、当前帧率和计算环境下的 Pareto 点。更合理的下一步是基于视点增益或覆盖增益动态停止历史累积，而不是固定拉长窗口。"),
        ("15.1 阈值选择的解释与敏感性", "当前阈值承担的是工程安全门而不是由训练集学习的统计决策边界。Occupied 的 point/frame/height 门组合用于排除单帧噪声和扁平结构；boundary<0.50 和 linearity<0.60 直接针对停车场中最常见的墙、路沿和标线边缘；Free 的 0.70 覆盖和 0.20 遮挡限制要求正向穿越证据。阈值组合采用 conjunction，意味着任一薄弱环节都会保留 Unknown。\n\n这种设计具有可解释和失败关闭的优点，但也可能造成保守。跨路线 16.3% 低于开发锚点 40.9%，可能部分来自统一阈值对不同视角和密度不自适应。不能简单降低阈值来提高 resolved 数；在没有 GT 时，降低门只会产生更多未经验证的终态。\n\n正式敏感性实验应在冻结 GT 上对每个门做 one-at-a-time 和联合网格分析，报告 Free false-positive 对 boundary、coverage、occlusion 和 stability 的响应。位姿扰动幅度也应根据独立定位误差分布设定，而不是永远固定 ±0.20 m/±0.50°。本文保留全部配置是为了让该校准可以复现。"),
        ("15.2 与可能基线的关系", "15 帧 deterministic gate 是本文唯一严格匹配的基线，因为它与 60 帧方案共享样本、几何实现和阈值，只改变历史长度。旧 Codex 30-anchor 运行不满足这种匹配条件，因而只放在 Extended Data。OpenAI Agent 结果用于确认结构化工具工作流和终态语义，也不能替代无 Agent 几何基线。\n\n正式 CVPR 比较至少应包括：Part1 15 帧；60 帧但无位姿稳健性；60 帧但无互斥门；60 帧 deterministic-only；60 帧本地轻量 VLM；60 帧 OpenAI VLM；基于 Camera crop 的视觉分类器；点云学习或占用网络基线。所有方法必须使用同一 GT、同一候选集合和同一 early-stop 定义。\n\n本文与 SurroundOcc、Occ3D 等稠密占用工作不处于相同任务设置，不能直接比较 mIoU。更公平的接口是把稠密模型输出投影到已知车位体积，再在相同 Free/Occupied/Unknown 拒判合同下比较风险—覆盖。与停车视觉分类器比较时，也需要把移动相机目标定位失败作为 Unknown，而不是只在成功 crop 上统计准确率。"),
        ("16 局限性与威胁", "内部有效性：阈值来自工程配置，尚未在冻结 GT 上调优；处理时间只有单机重复，不包含 API 网络波动。外部有效性：跨路线只含六个锚点，缺少天气、夜间、动态车辆和不同传感器。构念有效性：resolved 表示通过当前门，不代表真实标签正确。\n\nCamera 限制尤其重要：现有 FOV 可以在 occupancy map 上标注可见方向，但没有经独立像素 reference 验收的 slot-to-image polygon 投影。因此 Camera 不得单独支撑终态。OpenAI 分数未经概率校准，0.90 仅是策略阈值。\n\n没有人工 GT 是当前最大缺口。正式 CVPR 论文必须增加冻结标签、混淆矩阵、Free 误报、Occupied 漏报、风险—覆盖曲线、锚点聚类置信区间，以及学习式和非 Agent 基线。"),
        ("17 安全、伦理与隐私", "系统用于停车位建议，不应直接控制车辆运动。Unknown 必须触发继续搜索、减速或人工/其他感知复核，而不是默认 Free。报告不包含 API Key；密钥独立保存且权限为 600，产物进行泄漏扫描。原始 Camera 可能包含行人和车牌，公开数据前需要脱敏、授权和数据治理。\n\n语言模型供应商失败、输出格式错误和服务不可用必须在运行层显式报告。模型生成的文字理由不应被当作安全证明，真正可执行的保证来自数据合同、因果帧约束、身份哈希、确定性门和失败关闭状态机。"),
        ("18 可复现性与代码可用性", "核心代码模块为 extended_lidar.py、lidar_geometry.py、tools.py、agent.py 与 pipeline.py。run_extended_lidar_ablation.py 重建窗口和跨锚点硬门实验；build_parking_slot_agent_v2_full_paper.py 生成本文；validate_parking_slot_agent_v2_full_paper.py 检查数字、引用、媒体和 PDF。\n\n报告目录提供三张原始统计 CSV、硬门阈值表、全部主图与 Extended Data、OpenAI 逐车位终态、SHA-256 manifest 和审计 JSON。原始传感器数据因体积与授权未随稿公开；正式投稿前应提供脱敏最小复现集或明确的数据访问流程。"),
        ("19 结论", "本文在不改变 Free 优先 ReAct 队列设计的前提下，通过 60 帧严格因果目标几何和不可绕过硬门减少停车位 Unknown。开发锚点从 0/22 提升至 9/22，六个系统采样锚点从 0/208 提升至 34/208；60 帧达到 75 帧相同终态数并减少 22.0% 几何时间。该结果证明了工作流层面的可观测性提升与安全拒判能力，但由于缺少 GT，尚不能声明分类准确率或生产安全性。"),
        ("20 投稿前实验清单", "为了把当前完整研究稿升级为正式 CVPR 实证论文，下一轮应先冻结标注协议：由至少两名标注者独立查看时间同步 Camera/LiDAR 和车位地图，对 Free、Occupied、不可标注进行判定，并报告一致性及仲裁过程。测试集必须在阈值冻结后保持不可见。\n\n随后补齐四组核心实验：统一 GT 下的全部基线；Free 误报优先的风险—覆盖曲线；按路线聚类的置信区间；夜间、遮挡、动态车辆、稀疏点云和定位扰动分层结果。消融必须分别移除长窗口、互斥门、7/7 稳健性和 VLM，才能判断每个组件对准确性、覆盖率和成本的独立贡献。\n\n最后完成匿名化、作者冲突声明、数据授权、计算资源说明和局限披露。论文主张应保持为“在既定拒判风险下提高覆盖”，只有 GT 结果支持时才能进一步声称准确率或安全收益。"),
        ("参考文献", refs),
    ]


def _en_sections(m: Mapping[str, Any]) -> list[tuple[str, str]]:
    w = m["by_window"]
    refs = _reference_markdown()
    return [
        ("Abstract", f"Parking-slot occupancy decisions on a moving platform are frequently underdetermined by sparse observations, boundary returns, occlusion, and pose error. We present a selective perception workflow that preserves a Free-first ReAct queue while extending only the Part2 LiDAR detail tool with a strictly causal, identity-bound 60-frame history. A multimodal agent may interpret field-of-view, camera, and LiDAR evidence, but a terminal proposal is accepted only when a 0.90 score contract, mutually exclusive Free/Occupied geometry gates, and seven-of-seven pose perturbation tests all pass. On 22 Part1-Unknown slots at the development anchor, 15/30/45/60/75 frames resolve 0/6/8/9/9 cases; 60 frames match 75 frames while reducing geometry time by {m['saved_percent']:.1f}%. Across six systematically selected route anchors excluding development, a matched 15-frame baseline resolves {m['base_resolved']}/{m['base_total']} and the 60-frame method resolves {m['pooled_resolved']}/{m['pooled_total']} ({100*m['pooled_rate']:.1f}%; {m['pooled_free']} Free and {m['pooled_occupied']} Occupied). A real OpenAI-agent replay confirms 2 Free, 7 Occupied, and 13 Unknown outcomes with zero schema violations. Because no human occupancy ground truth is available, these measurements quantify strict evidence resolution coverage, not classification accuracy."),
        ("1. Introduction", "Occupancy classification is operationally ternary even when the semantic target is binary. A deployed system must retain an Unknown state whenever evidence is insufficient. Cameras provide semantics but are vulnerable to viewpoint, lighting, occlusion, and calibration error. LiDAR provides metric geometry but is sparse at grazing angles and around slot boundaries. Collapsing weak evidence into Free or Occupied improves nominal coverage at the cost of unsafe errors.\n\nWe study a constrained setting in which a map supplies parking-slot polygons and Part1 has already produced local Free and Unknown candidates. Part2 must find a trustworthy Free slot in a deterministic queue or abstain. Our contributions are: (i) a strictly causal, slot-bound temporal evidence package that leaves the 15-frame Part1 contract unchanged; (ii) auditable and mutually exclusive Free/Occupied geometry gates; (iii) seven-variant pose robustness recomputation outside the language model; and (iv) matched window and cross-route evaluations with real API replay and fail-closed negative cases."),
        ("2. Related Work", "Parking occupancy systems commonly classify predefined or automatically discovered image regions [14–16]. These methods perform well under calibrated fixed views, whereas our moving-platform setting must address changing visibility and incomplete image projection. Probabilistic maps explicitly retain free, occupied, and unknown space [1]. Modern 3D perception uses point encoders [3], multimodal datasets [4], BEV representations [5,6], and dense occupancy objectives [7–9]. Temporal and asynchronous fusion further improves coverage and consistency [10–13].\n\nReAct, Toolformer, and GPT4Tools demonstrate that language models can interleave decisions and tool calls [17–19], but tool competence is not a safety guarantee. Selective prediction instead treats abstention as a first-class output and studies risk--coverage trade-offs [20,21]. Our system combines these ideas at the workflow level: the agent organizes evidence, while deterministic code owns terminal authority."),
        ("3. Problem Formulation", "Let t0 be the decision time and S the mapped slots in a 30 m local scene. Each slot has polygon P_i, a Part1 state, uncalibrated evidence scores, and sensor resources. The candidate queue contains Part1 Free followed by Unknown cases. Part2 outputs Free, Occupied, or Unknown. We measure selective resolution R=N_resolved/N_unknown rather than forced-classification accuracy. Every evidence frame must satisfy t<=t0; packages must match slot, map, configuration, and source hashes; Free and Occupied eligibility are mutually exclusive; operational inference stops at the first Free score >=0.90 that also passes the hard gate."),
        ("4. System Architecture", "Part1 uses 15 local LiDAR frames and preserves the existing interface. Every Part2 SlotCase first receives a mandatory FOV observation. Camera tools are disabled when visibility is unavailable. The LiDAR detail tool loads a 60-frame identity-bound package and exposes a compact geometry card. The agent is limited to three evidence-tool rounds and strict JSON actions. A deterministic state machine validates terminal scores and gates. Exhaustive evaluation is an explicit mode and is separated from operational early stopping."),
        ("5. Causal Temporal Evidence", "For window W, we select the last W pose-corrected frames no later than t0. Missing frames, a final frame different from t0, or missing source point clouds are run failures. Each scan is transformed to map coordinates and then into target-slot coordinates. Point provenance retains frame identity. The NPZ package binds task, encounter, dataset, slot, anchor, configuration hash, slot-map hash, and source-file hashes. This prevents stale or adjacent-slot evidence from being silently reused."),
        ("6. Occupied and Free Gates", "Occupied evidence is evaluated in a 0.30--2.20 m vehicle-height interval. It requires at least 40 points, three support frames, 0.20 temporal support, two height layers, z95>=0.60 m, height span>=0.35 m, core overlap>=0.45, adjacent overlap<=0.35, boundary ratio<0.50, and linearity<0.60. Free is positive ray evidence, not the absence of returns. It requires at least five ray frames, two viewpoints separated by at least 10 degrees, observed-volume and near-ground coverage>=0.70, unobserved-component and occlusion ratios<=0.20, and no unresolved core hit. The opposite terminal must be vetoed."),
        ("7. Robustness and Agent Control", "Each tentative terminal is recomputed under the original pose, x/y translations of +/-0.20 m, and yaw perturbations of +/-0.50 degrees. Although the upstream diagnostic uses an 0.80 stability ratio, the Part2 terminal gate requires all seven variants. The VLM receives bounded observations, legal tools, and structured evidence. Its 0.90 score is necessary but insufficient. API errors and schema failures are run errors rather than semantic Unknowns. Hidden chain-of-thought is neither required nor reported; the audit retains observations, actions, gate outcomes, and concise terminal rationales."),
        ("8. Experimental Protocol", "The development anchor is frame 9277, with 24 candidates and 22 Part1 Unknown cases. Window ablation holds candidates, maps, gates, and code constant while varying W in {15,30,45,60,75}. Cross-route anchors 234, 3160, 5283, 7605, 9443, and 11678 are systematically selected from a pre-existing 30-anchor manifest and exclude development. Matched 15- and 60-frame runs use identical Unknown identities. Real-agent evaluation uses fixed structured settings. We report resolution coverage, state composition, model calls, validation errors, and geometry time. No accuracy or significance claim is made without ground truth."),
        ("9. Results", f"The development ablation resolves 0, 6, 8, 9, and 9 of 22 cases for 15, 30, 45, 60, and 75 frames. Sixty frames require {float(w[60]['processing_seconds']):.2f} s versus {float(w[75]['processing_seconds']):.2f} s for 75 frames, with no loss in terminal count. Across six held-out route positions, the matched baseline resolves {m['base_resolved']}/{m['base_total']}; the 60-frame method resolves {m['pooled_resolved']}/{m['pooled_total']} ({100*m['pooled_rate']:.2f}%). Every selected anchor improves, but rates vary substantially. The OpenAI replay produces 2 Free, 7 Occupied, and 13 Unknown outcomes with zero validation errors. Operational mode processes slot_1038 once and stops at Free=0.98."),
        ("10. Qualitative Analysis", "slot_1012 changes from Unknown to Free because extended ray evidence covers the slot volume while boundary-like occupied returns are vetoed. slot_1258 changes to Occupied because stable multi-frame, multi-height core returns pass ownership and shape gates. slot_1251 remains Unknown because one pose perturbation fails; slot_1010 retains conflicting Free and Occupied evidence; slot_1248 has partial route coverage and no validated pixel projection. These negative cases demonstrate that the method does not achieve coverage by lowering thresholds."),
        ("11. Discussion and Limitations", "The evidence supports a narrow claim: causal temporal geometry improves selective resolution under the current gate contract. It does not establish occupancy accuracy. The method is training-free and auditable but lower-coverage than an aggressively forced classifier. Thresholds remain engineering choices, cross-route evaluation contains six anchors, camera projection is not independently validated, model scores are uncalibrated, and runtime is measured in one server container. Ground-truth labels, clustered confidence intervals, risk--coverage analysis, multimodal baselines, and no-agent ablations are mandatory before a defensible CVPR accuracy claim."),
        ("12. Safety, Ethics, and Reproducibility", "Unknown must trigger continued search or another safety mechanism, never default Free. The system must not directly control vehicle motion. Camera data may contain people and license plates and requires authorization and de-identification before release. API secrets are excluded from artifacts. Reproducibility assets include source JSON, CSV tables, figures, code entry points, hashes, and an automated causal/gate audit. Raw sensors require a documented access process because of licensing and size."),
        ("13. Conclusion", "A 60-frame causal evidence extension and hard-gated multimodal agent reduce unresolved parking-slot cases without changing the Free-first workflow. The development anchor improves from 0/22 to 9/22 and six systematically selected anchors improve from 0/208 to 34/208. Sixty frames match the 75-frame terminal count at 22.0% lower geometry time. These are workflow-resolution results, not ground-truth accuracy results."),
        ("References", refs),
    ]


def _markdown(title: str, sections: Sequence[tuple[str, str]], status: str) -> str:
    lines = [f"# {title}", "", "Authors: [to be completed]", "", f"> {status}", ""]
    for heading, content in sections:
        lines.extend([f"## {heading}", "", content, ""])
    return "\n".join(lines)


def _html_document(title: str, sections: Sequence[tuple[str, str]], figures: Sequence[tuple[str, str]], status: str) -> str:
    body = []
    for heading, content in sections:
        paragraphs = "".join(f"<p>{html.escape(part)}</p>" for part in content.split("\n\n"))
        body.append(f"<section><h2>{html.escape(heading)}</h2>{paragraphs}</section>")
    figs = "".join(f"<figure><img src='{html.escape(path)}'><figcaption>{html.escape(caption)}</figcaption></figure>" for path, caption in figures)
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>
body{{margin:0;background:#e9eef2;color:#182b3e;font-family:Georgia,'Noto Serif CJK SC','Microsoft YaHei',serif;line-height:1.88}}main{{max-width:1180px;margin:auto;background:#fff;padding:58px 82px}}h1{{font-size:40px;line-height:1.25}}h2{{font-size:25px;margin-top:42px;border-bottom:1px solid #aab9c7;padding-bottom:7px}}p{{font-size:17px;text-align:justify}}.status{{padding:16px 20px;background:#fff6df;border-left:5px solid #d79a24}}figure{{margin:44px 0}}img{{width:100%;border:1px solid #cbd5df}}figcaption{{color:#526779;font-size:14px}}@media(max-width:760px){{main{{padding:24px}}h1{{font-size:29px}}}}</style></head><body><main><h1>{html.escape(title)}</h1><p>Authors: [to be completed]</p><p class='status'>{html.escape(status)}</p>{''.join(body)}<section><h2>完整图集 / Complete figure set</h2>{figs}</section></main></body></html>"""


def _pdf(path: Path, title: str, sections: Sequence[tuple[str, str]], figures: Sequence[tuple[Path, str]], status: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    font_path = FONT_CJK if FONT_CJK.is_file() else FONT_FALLBACK
    pdfmetrics.registerFont(TTFont("FullPaperCJK", str(font_path)))
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("FullTitle", parent=styles["Title"], fontName="FullPaperCJK", fontSize=19, leading=28, alignment=TA_CENTER, textColor=colors.HexColor("#14283d"), spaceAfter=18)
    heading_style = ParagraphStyle("FullHeading", parent=styles["Heading1"], fontName="FullPaperCJK", fontSize=13, leading=19, textColor=colors.HexColor("#2463a8"), spaceBefore=12, spaceAfter=7)
    body_style = ParagraphStyle("FullBody", parent=styles["BodyText"], fontName="FullPaperCJK", fontSize=9.3, leading=15.2, textColor=colors.HexColor("#20364a"), spaceAfter=8)
    note_style = ParagraphStyle("FullNote", parent=body_style, fontSize=8.5, leading=12.5, textColor=colors.HexColor("#735c20"), backColor=colors.HexColor("#fff6df"), borderPadding=8)
    caption_style = ParagraphStyle("FullCaption", parent=body_style, fontSize=8.1, leading=12, textColor=colors.HexColor("#526779"), spaceAfter=12)
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=1.65*cm, leftMargin=1.65*cm, topMargin=1.55*cm, bottomMargin=1.55*cm, title=title)
    story: list[Any] = [Paragraph(html.escape(title), title_style), Paragraph("Authors: [to be completed]", caption_style), Paragraph(html.escape(status), note_style), Spacer(1, 10)]
    for heading, content in sections:
        story.append(Paragraph(html.escape(heading), heading_style))
        for paragraph in content.split("\n\n"):
            story.append(Paragraph(html.escape(paragraph).replace("\n", "<br/>"), body_style))
    story.extend([PageBreak(), Paragraph("完整图集与图注", heading_style)])
    max_width = A4[0] - 3.3*cm
    for figure_path, caption in figures:
        with Image.open(figure_path) as image:
            width, height = image.size
        draw_width = max_width
        draw_height = draw_width * height / width
        if draw_height > 18.3*cm:
            draw_height = 18.3*cm
            draw_width = draw_height * width / height
        story.extend([RLImage(str(figure_path), width=draw_width, height=draw_height), Paragraph(html.escape(caption), caption_style), Spacer(1, 8)])
    doc.build(story)


def _english_tex(sections: Sequence[tuple[str, str]]) -> str:
    # This source intentionally targets the official CVPR package.  It is kept
    # separate from the reportlab review PDF because cvpr.sty is venue-owned.
    abstract = sections[0][1].replace("%", "\\%")
    chunks = []
    for heading, content in sections[1:-1]:
        safe = (content.replace("&", "\\&").replace("%", "\\%").replace("≥", "$\\ge$").replace("≤", "$\\le$")
                .replace("±", "$\\pm$").replace("∈", "$\\in$").replace("→", "$\\rightarrow$"))
        chunks.append(f"\\section{{{heading.split('. ', 1)[-1]}}}\n\n{safe}\n")
    keys = ",".join(row["key"] for row in REFERENCES)
    return f"""\\documentclass[10pt,twocolumn,letterpaper]{{article}}
\\usepackage[review]{{cvpr}}
\\usepackage{{times}}
\\usepackage{{epsfig}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{booktabs}}
\\usepackage[pagebackref,breaklinks,colorlinks]{{hyperref}}
\\def\\cvprPaperID{{****}}
\\def\\confName{{CVPR}}
\\def\\confYear{{2027}}
\\begin{{document}}
\\title{{{TITLE_EN}}}
\\author{{Anonymous CVPR submission}}
\\maketitle
\\begin{{abstract}}
{abstract}
\\end{{abstract}}

{''.join(chunks)}

\\nocite{{{keys}}}
\\bibliographystyle{{ieeenat_fullname}}
\\bibliography{{references}}
\\end{{document}}
"""


def _supplement_zh(m: Mapping[str, Any]) -> str:
    anchor_lines = []
    # Values are verified by the builder from the extended artifact.
    for anchor, total, resolved, free, occupied in (
        (234, 33, 5, 0, 5), (3160, 21, 2, 0, 2), (5283, 45, 7, 7, 0),
        (7605, 37, 12, 3, 9), (9443, 21, 1, 0, 1), (11678, 51, 7, 0, 7),
    ):
        anchor_lines.append(f"- frame {anchor}: {resolved}/{total}，Free={free}，Occupied={occupied}。")
    return f"""# 补充材料

## S1 合同和模式

Part1 永远保持 15 帧。Part2 60 帧资源是独立附加项，不回写 Part1 初态。默认 operational 模式遇到首个可信 Free 早停；论文统计只能使用显式 exhaustive 模式。

## S2 严格因果审计

24/24 扩展证据包恰好包含 60 帧，且最大 frame_id 不超过 anchor t0。包加载时校验 slot_id、配置、地图和源文件身份。

## S3 全部跨锚点结果

{chr(10).join(anchor_lines)}

合计：15 帧 {m['base_resolved']}/{m['base_total']}；60 帧 {m['pooled_resolved']}/{m['pooled_total']}。

## S4 全部开发锚点终态

逐车位状态、三分数、模型轮次、工具轮次、stop reason 和 final reason 位于 tables/Table_3_openai_case_results.csv。Extended Data Figure 3 将全部 22 个原始 Unknown 按终态和终态分排列。

## S5 失败类型

- robustness_not_all_variants_stable：七个扰动没有全部通过。
- opposing geometry not vetoed：反向状态仍有强证据。
- partial coverage/high occlusion：没有形成充分正向 Free 观测。
- boundary/linearity risk：回波更可能来自静态结构。
- Camera projection unavailable：FOV 只能路由，不能完成像素级目标身份确认。

## S6 统计说明

车位嵌套于路线锚点，且没有 GT。因此本稿不提供把 208 个车位视为独立样本的 p 值，也不计算 accuracy、precision、recall 或 F1。正式实验应以锚点或路线为聚类单元进行 bootstrap，并同时报告选择性覆盖与条件风险。

## S7 复现命令

```bash
python3 scripts/run_extended_lidar_ablation.py --help
python3 scripts/build_parking_slot_agent_v2_full_paper.py
python3 scripts/validate_parking_slot_agent_v2_full_paper.py
python3 -m unittest discover -s parking_slot_agent_v2/tests -p 'test*.py'
python3 -m unittest discover -s tests -p 'test*.py'
```
"""


def _supplement_en(m: Mapping[str, Any]) -> str:
    anchor_lines = []
    for anchor, total, resolved, free, occupied in (
        (234, 33, 5, 0, 5), (3160, 21, 2, 0, 2), (5283, 45, 7, 7, 0),
        (7605, 37, 12, 3, 9), (9443, 21, 1, 0, 1), (11678, 51, 7, 0, 7),
    ):
        anchor_lines.append(f"- frame {anchor}: {resolved}/{total}; Free={free}; Occupied={occupied}.")
    return f"""# Supplementary Material

## S1. Contracts and execution modes

Part1 always retains its original 15-frame contract. The 60-frame Part2 resource is an explicit, separate attachment and never rewrites the Part1 state. Operational mode stops at the first trustworthy Free slot. Full-queue processing is legal only when `evaluation_exhaustive=true`; the mode is persisted in checkpoints so an operational run cannot be resumed as an evaluation run or vice versa.

Each SlotCase preserves the mapped polygon, Part1 state, unresolved reason codes, FOV result, structured evidence records, tool actions, final scores, terminal reason, and audit resources. Hidden chain-of-thought is neither requested nor stored. The reproducible decision trace consists of observations, legal actions, concise model rationales, deterministic gates, and state transitions.

## S2. Strict-causality and identity audit

All 24/24 extended evidence packages contain exactly 60 frames. Every selected frame satisfies `frame_id <= t0`, and the final selected frame is the Part1 anchor. The package loader validates slot identity, dataset identity, configuration hash, slot-map hash, source-file hashes, task identity, and encounter identity. A missing source, wrong slot, stale package, or post-t0 frame is a run-level failure rather than a semantic Unknown outcome.

The development and cross-route experiments use pose-corrected per-frame point clouds. They do not use a future global map, a route-wide occupancy label, or the old Codex shadow output as terminal evidence. The legacy 30-anchor visualization is included only as workflow background and is explicitly excluded from the current 60-frame/OpenAI statistics.

## S3. Complete matched cross-anchor results

{chr(10).join(anchor_lines)}

Pooled matched result: 15 frames resolve {m['base_resolved']}/{m['base_total']}; 60 frames resolve {m['pooled_resolved']}/{m['pooled_total']} ({100*m['pooled_rate']:.2f}%), with {m['pooled_free']} Free and {m['pooled_occupied']} Occupied. Candidate identities are identical between the 15- and 60-frame runs at every anchor. The development anchor 9277 is not included in this set.

## S4. Development-window ablation

The same 22 Part1-Unknown identities are evaluated at 15, 30, 45, 60, and 75 frames. The respective resolved counts are 0, 6, 8, 9, and 9. Sixty frames are selected because this is the first window to reach the maximum terminal count; 75 frames add geometry time but no terminal case. Runtime is a single-container descriptive measurement and should not be interpreted as a hardware-independent benchmark.

## S5. Full development-anchor outcomes

`tables/Table_3_openai_case_results.csv` contains every original Unknown slot, final three-way state, Free/Occupied/Unknown scores, model turns, tool rounds, stop reason, validation-error count, and final rationale. Extended Data Figure 3 visualizes all 22 terminal outcomes. The exhaustive OpenAI run yields 2 Free, 7 Occupied, and 13 Unknown cases with zero structured-output validation errors.

The operational replay is separate: it processes the first candidate, slot_1038, invokes LiDAR once and the model once, reaches Free=0.98, and stops. It is not pooled with the 22-Unknown resolution analysis because slot_1038 entered the queue as a Part1 Free candidate.

## S6. Hard-gate taxonomy

The complete threshold registry is `tables/Table_4_hard_gate_thresholds.csv`. Important failure codes include:

- `robustness_not_all_variants_stable`: fewer than seven of seven pose variants pass.
- `no_strong_free_geometry_candidate`: positive free traversal is incomplete.
- `unresolved_core_hit`: an obstacle remains inside the slot core.
- `opposing_occupied_geometry_not_vetoed`: Free cannot be accepted while credible Occupied evidence remains.
- `boundary_ratio_not_below_0_5`: candidate returns are dominated by slot borders or static boundaries.
- `linearity_risk_not_below_0_6`: candidate geometry is too line-like to support a vehicle interpretation.
- `opposing_free_strong_gate`: Occupied cannot be accepted when strong Free evidence also passes.
- partial coverage or high occlusion: the sensor trajectory does not observe enough of the target volume.
- unavailable camera projection: map-level FOV may route tools but cannot establish pixel-level target identity.

## S7. Pose perturbations

Terminal geometry is recomputed for seven variants: original pose, x translation +0.20 m and -0.20 m, y translation +0.20 m and -0.20 m, and yaw +0.50 degrees and -0.50 degrees. Part2 requires 7/7, even though the upstream diagnostic stability field uses a less strict 0.80 pass ratio. slot_1251 demonstrates the distinction: the central estimate supports Occupied, but only 6/7 variants pass, so the terminal state remains Unknown.

## S8. Statistical interpretation

Slots are nested within route anchors and share trajectories, maps, and environmental structure. The paper therefore reports both anchor-level numerators/denominators and a descriptive pooled count. It does not treat all 208 slots as independent Bernoulli samples, and it does not report a p-value. Without human ground truth, accuracy, precision, recall, F1, sensitivity, specificity, and AUROC are undefined for this experiment.

A future labeled evaluation should report selective risk and coverage jointly. Confidence intervals should use route- or anchor-cluster bootstrap. Threshold sensitivity should be evaluated after freezing the test set, with particular attention to Free false positives as functions of observed volume, near-ground coverage, occlusion, boundary ratio, linearity, and pose-stability requirements.

## S9. Reproduction commands

```bash
python3 scripts/run_extended_lidar_ablation.py --help
python3 scripts/build_parking_slot_agent_v2_nature_report.py
python3 scripts/validate_parking_slot_agent_v2_nature_report.py
python3 scripts/build_parking_slot_agent_v2_full_paper.py
python3 scripts/validate_parking_slot_agent_v2_full_paper.py
python3 -m unittest discover -s parking_slot_agent_v2/tests -p 'test*.py'
python3 -m unittest discover -s tests -p 'test*.py'
```

The full-paper validator checks required sections, audited numerical tokens, bibliography keys, local figure references, the 20-row hard-gate table, anonymous CVPR review mode, PDF validity, and SHA-256 hashes for generated files.

## S10. Required work before formal submission

Formal evaluation requires a frozen human-labeled test set, independent annotators and adjudication, confusion matrices, Free false-positive and Occupied false-negative rates, risk--coverage curves, route-cluster confidence intervals, and comparisons against deterministic-only, no-long-window, no-robustness, no-mutual-veto, local-VLM, camera-classifier, and learned occupancy baselines. Camera-to-slot pixel projection must receive an independent calibration audit. Data authorization, anonymization, author identities, conflicts, compute reporting, and a venue-compliant template must also be completed.
"""


def build_full_paper_package(*, paper_root: str | Path, source_report_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root = Path(paper_root).resolve()
    source = Path(source_report_dir).resolve()
    destination = Path(output_dir).resolve()
    # Keep the redesigned, paper-facing suite isolated from legacy diagnostic
    # plots that may already exist in an older generated package.
    figures_dir = destination / "figures_cvpr"
    tables_dir = destination / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    window_path = root / "window_ablation_frame_9277.json"
    baseline_path = root / "cross_anchor_baseline15_systematic6.json"
    extended_path = root / "cross_anchor_extended60_systematic6.json"
    result_path = root.parent / "parking_slot_agent_v2_frame_9277/openai_extended60_exhaustive_v1/part2_result.json"
    window, baseline, extended, result = map(_load, (window_path, baseline_path, extended_path, result_path))
    m = _metrics(window, baseline, extended, result)

    figures = build_cvpr_figure_suite(
        window=window,
        baseline=baseline,
        extended=extended,
        result=result,
        output_dir=figures_dir,
    )

    for csv_path in sorted((source / "tables").glob("*.csv")):
        shutil.copy2(csv_path, tables_dir / csv_path.name)
    threshold_path = tables_dir / "Table_4_hard_gate_thresholds.csv"
    _write_csv(threshold_path, _threshold_rows())
    protocol_rows = [
        {"experiment": "development window ablation", "anchors": "9277", "unknown_slots": 22, "windows": "15/30/45/60/75", "agent": "deterministic geometry gates", "purpose": "select causal history length"},
        {"experiment": "matched cross-route baseline", "anchors": "234/3160/5283/7605/9443/11678", "unknown_slots": 208, "windows": "15", "agent": "deterministic geometry gates", "purpose": "matched short-window control"},
        {"experiment": "matched cross-route extension", "anchors": "234/3160/5283/7605/9443/11678", "unknown_slots": 208, "windows": "60", "agent": "deterministic geometry gates", "purpose": "route generalization audit"},
        {"experiment": "OpenAI exhaustive replay", "anchors": "9277", "unknown_slots": 22, "windows": "60", "agent": "OpenAI structured agent", "purpose": "semantic/state-machine confirmation"},
        {"experiment": "OpenAI operational replay", "anchors": "9277", "unknown_slots": "queue", "windows": "60", "agent": "OpenAI structured agent", "purpose": "first trustworthy Free early stop"},
    ]
    _write_csv(tables_dir / "Table_5_experiment_protocol.csv", protocol_rows)

    zh = _zh_sections(m)
    en = _en_sections(m)
    status_zh = "完整研究稿；数字已经过自动审计；无人工 GT，因此不是准确率论文，也尚非可直接提交的 CVPR 终稿。"
    status_en = "Complete research draft with audited numbers. No human ground truth is available; this is not yet a submission-ready CVPR accuracy paper."
    zh_md = destination / "paper_zh_detailed.md"
    en_md = destination / "paper_en_cvpr_draft.md"
    zh_html = destination / "paper_zh_detailed.html"
    zh_pdf = destination / "paper_zh_detailed.pdf"
    tex_path = destination / "paper_en_cvpr_draft.tex"
    bib_path = destination / "references.bib"
    supp_zh = destination / "supplementary_zh.md"
    supp_en = destination / "supplementary_en.md"
    ref_json = destination / "references_verified.json"
    visual_report = destination / "visual_results_report.html"
    zh_md.write_text(_markdown(TITLE_ZH, zh, status_zh), encoding="utf-8")
    en_md.write_text(_markdown(TITLE_EN, en, status_en), encoding="utf-8")
    html_figures = [(f"figures_cvpr/{path.name}", caption) for path, caption in figures]
    zh_html.write_text(_html_document(TITLE_ZH, zh, html_figures, status_zh), encoding="utf-8")
    _pdf(zh_pdf, TITLE_ZH, zh, figures, status_zh)
    tex_path.write_text(_english_tex(en), encoding="utf-8")
    bib_path.write_text(_bibtex(), encoding="utf-8")
    supp_zh.write_text(_supplement_zh(m), encoding="utf-8")
    supp_en.write_text(_supplement_en(m), encoding="utf-8")
    ref_json.write_text(json.dumps({"schema_version": "parking-slot-paper-references/1.0", "count": len(REFERENCES), "references": list(REFERENCES)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visual_sections = [
        ("先看结论", "60 帧是第一个达到最大消歧数的窗口：开发锚点 0→9/22，六个独立锚点 0→34/208；75 帧没有新增终态。"),
        ("再看为什么", "成功案例只展示真正决定终态的局部几何和硬门，不再缩放整张调试卡。失败案例明确指出是哪一个门阻止了高分结论。"),
        ("最后看边界", "没有人工 GT，因此这些图证明的是严格证据消歧覆盖提升，不是分类准确率。"),
    ]
    visual_report.write_text(_html_document("ParkingSlotAgent v2｜直观实验结果", visual_sections, html_figures, "CVPR 图表重制版：每张图只有一个可验证结论。"), encoding="utf-8")

    readme = destination / "README.md"
    readme.write_text(f"""# 完整论文包

推荐阅读顺序：

1. `visual_results_report.html`：先看这个；重制后的直观结果与完整证据链。
2. `paper_zh_detailed.html`：完整主文与全部图。
3. `paper_zh_detailed.pdf`：可离线阅读的详细中文论文。
4. `paper_en_cvpr_draft.md`：完整英文内容草稿。
5. `paper_en_cvpr_draft.tex`：面向官方 CVPR 模板的匿名 LaTeX 源。
6. `supplementary_zh.md`：合同、逐锚点结果、失败类型和复现命令。
7. `references.bib` 与 `references_verified.json`：{len(REFERENCES)} 条已核验参考文献。

当前论文只使用 `figures_cvpr/` 中的 7 张重制图。若目录中仍有 `figures/`，它是旧版诊断图归档，不属于当前论文结果。

当前稿件已经完整覆盖摘要、引言、相关工作、问题定义、系统、方法、实验、结果、案例、讨论、局限、安全伦理、复现、结论和参考文献。没有人工 GT，因此正式投稿前仍需补齐标签和准确率实验。
""", encoding="utf-8")

    inputs = [window_path, baseline_path, extended_path, result_path]
    generated = [zh_md, en_md, zh_html, zh_pdf, tex_path, bib_path, supp_zh, supp_en, ref_json, visual_report, readme, *[path for path, _ in figures], *sorted(tables_dir.glob("*.csv"))]
    manifest_path = destination / "paper_manifest.json"
    manifest = {
        "schema_version": "parking-slot-agent-v2-full-paper/1.0",
        "status": "complete_research_draft_not_submission_ready_without_gt",
        "titles": {"zh": TITLE_ZH, "en": TITLE_EN},
        "claims": {"development": "9/22", "cross_route_baseline": f"{m['base_resolved']}/{m['base_total']}", "cross_route_extended": f"{m['pooled_resolved']}/{m['pooled_total']}", "selected_window": 60, "reference_count": len(REFERENCES)},
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in inputs],
        "files": [{"path": str(path), "sha256": _sha256(path)} for path in generated],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(destination), "visual_report": str(visual_report), "zh_html": str(zh_html), "zh_pdf": str(zh_pdf), "zh_markdown": str(zh_md), "en_markdown": str(en_md), "en_tex": str(tex_path), "supplement": str(supp_zh), "references": len(REFERENCES), "manifest": str(manifest_path)}


__all__ = ["REFERENCES", "build_full_paper_package"]
