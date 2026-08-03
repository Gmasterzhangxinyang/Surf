#!/usr/bin/env python3
"""Build the concise frame6241 Camera-first Part1-to-Part2 audit homepage."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import shutil
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "Nature_ParkingAgent_实验报告_20260728"
FRAME = REPORT / "frame6241_blind"
RUN = FRAME / "pose_tuned_run/camera_first_final_full11_w45k3_gt_scope"
CROP_RUN = FRAME / "pose_tuned_run/camera_crop_trial_slot0951_v3"
INPUT = FRAME / "pose_tuned_run/w45k3_agent_input"
GT_PATH = FRAME / "gt_annotation_pack/frame6241_gt_locked.csv"
GT_REVISION = FRAME / "gt_annotation_pack/frame6241_gt_revision_20260801.json"
METRICS_PATH = FRAME / "locked_gt_experiments/final_camera_first_agent_metrics.json"
ABLATION_PATH = FRAME / "locked_gt_experiments/part1_ablation/ablation_summary.json"
QWEN_ABLATION_PATH = FRAME / "qwen_ablation/qwen_vlm_ablation_comparison.json"
QWEN_REPORT = REPORT / "frame6241_qwen_vlm_ablation.html"
PAPER_DIR = REPORT / "paper_parkingagent_20260801"
PAPER_PDF = PAPER_DIR / "DASP-Park_Decision-Aware_Active_Semantic_Perception.pdf"
PAPER_TEX = PAPER_DIR / "parkingagent.tex"
PAPER_BIB = PAPER_DIR / "references.bib"
PAPER_ARCH = PAPER_DIR / "figures/fig1_dasp_park_method.svg"
PAPER_CASES = PAPER_DIR / "figures/fig3_multicase_evidence.png"
FRONT_MAP = REPORT / "frame6241_final_assets/frame6241_front180_w45k3.png"
LOCATION_DIR = FRAME / "gt_annotation_pack/review_pose_tuned_yaw_minus1/location_maps"
CAMERA_DIR = FRAME / "gt_annotation_pack/camera_views_pose_tuned_yaw_minus1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPORT.resolve()).as_posix()


def state(value: str) -> str:
    value = str(value).lower()
    return f"<span class='state {esc(value)}'>{esc(value.title())}</span>"


def short(values: list[str], limit: int = 5) -> str:
    if not values:
        return "—"
    text = ", ".join(values[:limit])
    return text + (f" …(+{len(values)-limit})" if len(values) > limit else "")


def main() -> None:
    required = [
        RUN / "live/part2_result.json",
        RUN / "agent_plan_execute_observe.json",
        RUN / "replay_verification.json",
        CROP_RUN / "live/part2_result.json",
        CROP_RUN / "agent_plan_execute_observe.json",
        CROP_RUN / "replay_verification.json",
        METRICS_PATH,
        ABLATION_PATH,
        INPUT / "part1_all_local.json",
        INPUT / "part1_front180.json",
        GT_PATH,
        GT_REVISION,
        QWEN_ABLATION_PATH,
        QWEN_REPORT,
        PAPER_PDF,
        PAPER_TEX,
        PAPER_BIB,
        PAPER_ARCH,
        PAPER_CASES,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing report inputs: " + "; ".join(missing))

    metrics = load(METRICS_PATH)
    final_metrics = metrics["methods"]["best_part1_w45k3_plus_camera_first_agent"]
    baseline_metrics = metrics["methods"]["best_part1_w45k3"]
    visible_metrics = metrics["subsets"]["camera_visible_processed"]
    occluded_metrics = metrics["subsets"]["occluded_processed"]
    determinate_metrics = metrics["subsets"]["all_determinate_gt"]
    high_conf_metrics = metrics["subsets"]["high_confidence_gt_0_90"]
    agent_result = load(RUN / "live/part2_result.json")
    trace = load(RUN / "agent_plan_execute_observe.json")
    part1 = load(INPUT / "part1_all_local.json")
    front = load(INPUT / "part1_front180.json")
    ablation = load(ABLATION_PATH)
    replay = load(RUN / "replay_verification.json")
    input_manifest = load(INPUT / "input_manifest.json")
    crop_result = load(CROP_RUN / "live/part2_result.json")
    crop_case = crop_result["slot_results"][0]["case"]
    crop_evidence = next(
        item for item in crop_case["evidence"] if item["tool_name"] == "camera_crop"
    )
    crop_round = next(
        item for item in crop_case["rounds"] if item["tool_name"] == "camera_crop"
    )
    crop_replay = load(CROP_RUN / "replay_verification.json")
    qwen_ablation = load(QWEN_ABLATION_PATH)

    with GT_PATH.open(newline="", encoding="utf-8-sig") as handle:
        gt_rows = list(csv.DictReader(handle))
    gt = {row["slot_id"]: row for row in gt_rows}
    final_rows = {
        row["slot_id"]: row
        for row in final_metrics["rows"]
    }
    details = metrics["agent_details"]
    trace_by_id = {row["slot_id"]: row for row in trace["slots"]}

    part1_table = []
    for case in sorted(part1["slot_cases"], key=lambda row: row["slot"]["slot_id"]):
        sid = case["slot"]["slot_id"]
        part1_table.append(
            "<tr>"
            f"<td>{esc(sid)}</td>"
            f"<td>{state(case['part1_state'])}</td>"
            f"<td>{esc(case['decision_reason'])}</td>"
            f"<td>{esc(short(case.get('unknown_reasons', [])))}</td>"
            f"<td>{case['slot']['distance_to_anchor_m']:.2f} m</td>"
            "</tr>"
        )

    front_table = []
    for case in front["slot_cases"]:
        sid = case["slot"]["slot_id"]
        audit = case["slot"].get("metadata", {}).get("front180", {})
        if not audit:
            audit = case.get("resources", {})
        fov = trace_by_id.get(sid, {}).get("initial_camera_gate", {})
        front_table.append(
            "<tr>"
            f"<td>{esc(sid)}</td>"
            f"<td>{state(case['part1_state'])}</td>"
            f"<td>{case['slot']['distance_to_anchor_m']:.2f} m</td>"
            f"<td>{esc(fov.get('visibility', 'not evaluated in GT subset'))}</td>"
            f"<td>{'正式11-case评测' if sid in trace_by_id else '前向候选，未在锁定GT子集'}</td>"
            "</tr>"
        )

    gt_table = []
    for row in gt_rows:
        sid = row["slot_id"]
        result = final_rows[sid]
        detail = details.get(sid)
        tools = " → ".join(detail.get("tools", [])) if detail else "Part1 retained"
        confidence = ""
        if detail:
            predicted = detail["state"]
            confidence = detail["scores"].get(f"{predicted}_confidence", "")
        gt_table.append(
            "<tr>"
            f"<td>{esc(sid)}</td><td>{state(row['gt_state'])}</td>"
            f"<td>{state(result['prediction'])}</td>"
            f"<td class='{'ok' if result['correct'] else 'bad'}'>{'✓' if result['correct'] else '✗'}</td>"
            f"<td>{esc(row['gt_observability'])}</td>"
            f"<td>{esc(row['state_confidence'])}</td>"
            f"<td>{esc(tools)}</td><td>{esc(confidence)}</td>"
            "</tr>"
        )

    case_cards = []
    for slot_trace in trace["slots"]:
        sid = slot_trace["slot_id"]
        row = final_rows[sid]
        detail = details[sid]
        fov = slot_trace["initial_camera_gate"]
        step_cards = []
        for step in slot_trace["steps"]:
            if step["phase"] == "plan_execute_observe":
                observe = step.get("observe") or {}
                step_cards.append(
                    "<div class='peo'>"
                    f"<b>Turn {step['turn']} · Plan</b><p>{esc(step['plan'])}</p>"
                    f"<b>Execute</b><code>{esc(step['execute']['tool'])}</code>"
                    f"<b>Observe</b><p>{esc(observe.get('summary', '—'))}</p>"
                    "</div>"
                )
            elif step["phase"] == "preliminary_final_recheck":
                obs = step["observe"]
                step_cards.append(
                    "<div class='peo review'>"
                    f"<b>Turn {step['turn']} · Preliminary Final / Self-check</b>"
                    f"<p>提议 {state(obs['state'])}，运行时未接受并要求一次有界复核。</p>"
                    f"<p>{esc(obs['reason'])}</p>"
                    "</div>"
                )
            else:
                obs = step["observe"]
                step_cards.append(
                    "<div class='peo final'>"
                    f"<b>Turn {step['turn']} · Accepted Final</b>"
                    f"<p>{state(obs['state'])} · Free={obs['free_confidence']:.2f} · "
                    f"Occupied={obs['occupied_confidence']:.2f}</p>"
                    f"<p>{esc(obs['reason'])}</p>"
                    f"<small>{esc(', '.join(obs['reason_codes']))}</small>"
                    "</div>"
                )

        images = []
        location = LOCATION_DIR / f"{sid}.png"
        camera = CAMERA_DIR / f"{sid}.jpg"
        fov_map = RUN / f"live/media/{sid}/fov/{sid}_fov_occupancy_map.png"
        for path, caption in (
            (location, "大位置Map：人工检查目标位置"),
            (fov_map, "check_fov审计图：只做工具门控"),
            (camera, "Agent Camera工具：青色目标多帧时序"),
        ):
            if path.is_file():
                images.append(
                    f"<figure><img loading='lazy' src='{esc(rel(path))}'>"
                    f"<figcaption>{esc(caption)}</figcaption></figure>"
                )
        lidar_candidates = sorted(
            (RUN / f"live/media/{sid}/lidar_detail").glob("*_explained.png")
        )
        if lidar_candidates:
            path = lidar_candidates[-1]
            images.append(
                f"<figure><img loading='lazy' src='{esc(rel(path))}'>"
                "<figcaption>Agent LiDAR回退工具证据</figcaption></figure>"
            )
        case_cards.append(
            "<article class='case'>"
            f"<h3>{esc(sid)} · GT {state(row['gt_state'])} → 系统 {state(row['prediction'])}</h3>"
            f"<p><b>Pre-Agent check_fov：</b>{esc(fov['visibility'])}, "
            f"confidence={float(fov['confidence']):.2f}。它只生成Camera工具可用性，不判断占用。</p>"
            f"<p><b>实际工具：</b>{esc(' → '.join(detail['tools']))}；"
            f"模型轮次={detail['model_turns']}；工具轮次={detail['tool_rounds']}；"
            f"结构化错误={len(detail['validation_errors'])}。</p>"
            "<div class='gallery'>" + "".join(images) + "</div>"
            "<div class='peogrid'>" + "".join(step_cards) + "</div>"
            "</article>"
        )

    ablation_rows = []
    for row in ablation["all"]:
        best = (
            row["history_span"] == 45
            and row["history_sample_count"] == 3
        )
        ablation_rows.append(
            f"<tr class='{'best' if best else ''}'>"
            f"<td>{row['history_span']}</td><td>{row['history_sample_count']}</td>"
            f"<td>{pct(row['three_class_accuracy'])}</td>"
            f"<td>{pct(row['terminal_coverage'])}</td>"
            f"<td>{pct(row['terminal_selective_accuracy'])}</td>"
            f"<td>{pct(row['unknown_false_resolution_rate'])}</td>"
            f"<td>{esc(row['prediction_counts'])}</td>"
            f"<td>{'采用' if best else ''}</td></tr>"
        )

    qwen_rows = []
    for row in qwen_ablation["models"]:
        qwen_rows.append(
            "<tr>"
            f"<td><b>{esc(row['label'])}</b><br><small>{esc(row['size'])}</small></td>"
            f"<td>{pct(row['all14_accuracy'])}</td>"
            f"<td>{pct(row['agent11_accuracy'])}</td>"
            f"<td>{pct(row['camera_visible_accuracy'])}</td>"
            f"<td>{pct(row['all14_terminal_coverage'])}</td>"
            f"<td>{row['first_tool_camera_count']}/11</td>"
            f"<td>{esc(row['accepted_tool_counts'])}</td>"
            f"<td>{row['validation_error_count']}</td>"
            f"<td>{str(row['replay_verified']).lower()}</td>"
            "</tr>"
        )
    archive = REPORT / "index_depth_v4_archive_20260801.html"
    index = REPORT / "index.html"
    if index.is_file() and not archive.exists():
        shutil.copy2(index, archive)

    html_text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>DASP-Park Frame6241 · Camera-first Agent完整实验</title>
<style>
:root{{--bg:#f5f7fb;--ink:#172033;--muted:#637086;--line:#dce3ed;--blue:#2563eb;--green:#12865f;--red:#d74747;--amber:#c47a09}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,system-ui,"Microsoft YaHei",sans-serif}}
nav{{position:sticky;top:0;z-index:8;background:#122039;padding:11px 4vw;display:flex;gap:18px;overflow:auto}}nav a{{color:#dce8fb;text-decoration:none;white-space:nowrap}}
header,main{{max-width:1500px;margin:auto}}header{{padding:45px 4vw 28px}}h1{{font-size:clamp(34px,5vw,60px);line-height:1.08;margin:8px 0}}h2{{font-size:28px;margin:0 0 8px}}h3{{margin:0 0 10px}}
.lead,.muted{{color:var(--muted)}}.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:9px;margin-top:22px}}.metric,.section,.case{{background:white;border:1px solid var(--line);border-radius:14px}}.metric{{padding:14px}}.metric b{{display:block;font-size:25px}}
main{{padding:0 4vw 70px}}.section{{padding:24px;margin:18px 0}}.note{{padding:12px 15px;background:#eef4ff;border-left:4px solid var(--blue);margin:12px 0}}.warn{{background:#fff5e5;border-left-color:var(--amber)}}
.flow{{display:grid;grid-template-columns:repeat(6,1fr);gap:18px;margin:15px 0}}.flow div{{padding:13px;border:1px solid var(--line);border-radius:10px;position:relative}}.flow div:not(:last-child):after{{content:"→";position:absolute;right:-15px;top:40%;color:var(--blue)}}
.scroll{{overflow:auto;max-height:680px;border:1px solid var(--line);border-radius:10px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}th{{position:sticky;top:0;background:#eaf0f8;z-index:2}}.best{{background:#e9f8f1}}
.state{{font-weight:800}}.state.free,.ok{{color:var(--green)}}.state.occupied,.bad{{color:var(--red)}}.state.unknown{{color:#778396}}
.case{{padding:18px;margin:14px 0}}.gallery{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}}figure{{margin:0;border:1px solid var(--line);border-radius:9px;overflow:hidden}}figure img{{width:100%;height:270px;object-fit:contain;background:#102039;display:block}}figcaption{{padding:7px;color:var(--muted);font-size:12px}}
.peogrid{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:10px}}.peo{{background:#122039;color:#e8effa;border-radius:10px;padding:12px}}.peo b{{display:block;color:#74b8ff;margin-top:5px}}.peo p{{font-size:12px}}.peo code{{display:block;color:#ffd17a}}.peo.final{{background:#173d34}}.links{{display:flex;gap:8px;flex-wrap:wrap}}.links a{{padding:8px 11px;border:1px solid var(--line);border-radius:8px;text-decoration:none}}
@media(max-width:1050px){{.metrics{{grid-template-columns:repeat(3,1fr)}}.flow,.gallery,.peogrid{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:650px){{.metrics,.flow,.gallery,.peogrid{{grid-template-columns:1fr}}}}
</style></head><body>
<nav><a href='#paper'>论文</a><a href='#result'>结果</a><a href='#input'>输入</a><a href='#part1'>Part1</a><a href='#front'>180°</a><a href='#fov'>check_fov</a><a href='#agent'>Agent PEO</a><a href='#crop'>自主Crop</a><a href='#gt'>GT</a><a href='#ablation'>帧消融</a><a href='#qwen'>VLM消融</a><a href='#audit'>审计</a></nav>
<header><div style='color:#2563eb;font-weight:800'>FRAME 6241 · LOCKED HUMAN GT · CAMERA-FIRST FINAL</div>
<h1>Part1 → Part2<br>结构化自主Agent完整实验</h1>
<p class='lead'>系统check_fov只负责Camera工具门控；Agent读取结构化状态后自主Plan、调用Camera/Crop/LiDAR、观察Observation并输出三态Final。青色投影重叠不是直接Occupied，Camera占用候选在LiDAR可用时必须复核。</p>
<div class='metrics'>
<div class='metric'><b>{pct(final_metrics['three_class_accuracy'])}</b>14个锁定GT三分类准确率</div>
<div class='metric'><b>{pct(final_metrics['terminal_coverage'])}</b>终态覆盖</div>
<div class='metric'><b>{pct(final_metrics['terminal_selective_accuracy'])}</b>终态选择性准确率</div>
<div class='metric'><b>{final_metrics['false_occupied_count']} / {final_metrics['false_free_count']}</b>错误Occupied / Free</div>
<div class='metric'><b>{pct(visible_metrics['three_class_accuracy'])}</b>Camera-visible 3例准确率</div>
<div class='metric'><b>{metrics['first_tool_camera_count']}/11</b>首工具Camera</div>
</div></header><main>
<section class='section' id='paper'><h2>论文稿：DASP-Park</h2>
<p><b>DASP-Park: Decision-Aware Active Semantic Perception for Reliable Parking-Slot Occupancy under Occlusion</b></p>
<p>已整理为6页IEEE双栏科研论文，包含Introduction、Related Work、Methodology、Experiments、Ablation Study，以及讨论与局限。论文中的结果全部绑定本页最新锁定GT和可复现实验文件。</p>
<img style='width:100%;max-height:650px;object-fit:contain;background:#f8fafc' src='{esc(rel(PAPER_ARCH))}' alt='DASP-Park system architecture'>
<figure style='margin-top:12px'><img loading='lazy' style='height:auto;max-height:760px' src='{esc(rel(PAPER_CASES))}'><figcaption>真实Experiment多case：Free、Occupied、正确Unknown与失败case；全部来自锁定实验产物。</figcaption></figure>
<div class='links'><a href='{esc(rel(PAPER_PDF))}'>打开论文PDF</a><a href='{esc(rel(PAPER_TEX))}'>LaTeX源码</a><a href='{esc(rel(PAPER_BIB))}'>参考文献BibTeX</a></div>
<p class='muted'>当前作者栏为Anonymous Author(s)，投稿前替换作者与单位即可。核心图采用可编辑的科研矢量双子图：左侧呈现Part 1到Part 2的完整决策流，右侧单独展开Plan–Execute–Observe与证据验证闭环。</p></section>

<section class='section' id='result'><h2>0. 正式结果</h2>
<div class='note'><b>与W45/K3 Part1比较：</b>准确率 {pct(baseline_metrics['three_class_accuracy'])} → {pct(final_metrics['three_class_accuracy'])}；终态覆盖 {pct(baseline_metrics['terminal_coverage'])} → {pct(final_metrics['terminal_coverage'])}。人工确定Free/Occupied为 {sum(r['correct'] for r in determinate_metrics['rows'])}/{determinate_metrics['evaluated_gt_slots']} 正确；GT置信度≥0.90为 {sum(r['correct'] for r in high_conf_metrics['rows'])}/{high_conf_metrics['evaluated_gt_slots']} 正确。</div>
<div class='note warn'><b>评价口径：</b>只在冻结的14个GT ID上统计；Unknown是正式类别。11个Part2 case输入不含GT状态。提示词开发使用过该anchor，结果属于单anchor开发性实验，不是独立外部测试。</div>
<div class='note'><b>GT修订：</b>Agent预测完成后，用户于2026-08-01确认slot_0918与slot_0953为Free；仅修订状态，原occluded和0.5置信度保持不变，修订记录可审计。</div></section>

<section class='section' id='input'><h2>1. 输入数据</h2>
<div class='flow'><div><b>Anchor</b><br>LiDAR 6241<br>Camera 18700</div><div><b>历史池</b><br>W=45<br>只读t≤6241</div><div><b>采样</b><br>K=3<br>{esc(input_manifest['history_frame_ids'])}</div><div><b>定位</b><br>纵向−0.10m<br>yaw−1.0°</div><div><b>车位</b><br>18m局部范围<br>地图多边形</div><div><b>GT</b><br>14 slots<br>{metrics['gt_counts'].get('free', 0)}F/{metrics['gt_counts'].get('occupied', 0)}O/{metrics['gt_counts'].get('unknown', 0)}U</div></div>
<p>输入包含原始LiDAR、左相机RGB、地图位姿、map-frame点云、1397车位数据库、车位多边形及静态地图。GT仅在运行后用于评价。</p></section>

<section class='section' id='part1'><h2>2. Part1逐车位输出</h2>
<p>W45/K3把历史点云对齐到地图系；Occupied分支使用核心点、车辆尺度、高度层、跨帧支持及柱墙/边界否决；Free分支使用射线穿越、体积和近地覆盖、遮挡及核心hit冲突。仅一侧强证据成立才输出终态。</p>
<div class='scroll'><table><thead><tr><th>slot</th><th>Part1</th><th>主原因</th><th>Unknown原因</th><th>距离</th></tr></thead><tbody>{''.join(part1_table)}</tbody></table></div></section>

<section class='section' id='front'><h2>3. Part2前方180°候选</h2>
<p>只保留车辆前向半平面中的Free/Unknown候选。全量17个；正式准确率实验再按冻结GT ID取11个，标签值不传给Agent。</p>
<img style='width:100%;max-height:760px;object-fit:contain' src='{esc(rel(FRONT_MAP))}'>
<div class='scroll'><table><thead><tr><th>slot</th><th>Part1</th><th>距离</th><th>check_fov</th><th>范围</th></tr></thead><tbody>{''.join(front_table)}</tbody></table></div></section>

<section class='section' id='fov'><h2>4. Pre-Agent check_fov</h2>
<div class='flow'><div><b>输入</b><br>车辆位姿、车位多边形、Camera内参</div><div><b>计算</b><br>水平投影与鲁棒采样</div><div><b>输出</b><br>visible / partial / uncertain / outside</div><div><b>作用</b><br>生成available_tools</div><div><b>不做</b><br>不看Camera像素</div><div><b>不输出</b><br>不判断F/O/U</div></div>
<p>因此check_fov是确定性环境门控，不伪装成Agent自主工具。大位置图保留给人审计，但不作为占用分类图输入。</p></section>

<section class='section' id='agent'><h2>5. 11个Agent case：Plan → Execute → Observe → Final</h2>
<p>Camera可用时始终优先；目标在最佳tile中过小、模糊或边缘压缩时，Agent可自主选择camera_crop；Camera明确Free可直接终态；Camera占用候选在LiDAR可用时需要复核；Camera归属ambiguous时LiDAR障碍点不能单独升级Occupied。下面逐case展示大位置图、Camera、可选LiDAR以及真实结构化轨迹。</p>
{''.join(case_cards)}</section>

<section class='section' id='crop'><h2>5A. Agent自主Camera Crop真实试跑</h2>
<div class='note'><b>不是脚本指定：</b>slot_0951试跑中，Agent先观察camera_context，然后自己选择camera_crop，再决定是否调用lidar_detail。GT未传入模型。</div>
<div class='flow'><div><b>Plan</b><br>{esc(crop_round['reasoning_summary'])}</div><div><b>Execute</b><br>camera_crop<br>{esc(crop_round['tool_arguments'])}</div><div><b>选中tile</b><br>{esc(crop_evidence['metadata']['selected_tile_indices'])}</div><div><b>选中帧</b><br>{esc(crop_evidence['metadata']['selected_frame_ids'])}</div><div><b>Observe</b><br>放大青色目标与车辆边界</div><div><b>Final</b><br>{state(crop_case['final_state'])}<br>归属仍不充分</div></div>
<div class='gallery'>
<figure><img loading='lazy' src='{esc(rel(CAMERA_DIR / "slot_0951.jpg"))}'><figcaption>camera_context：8帧青色目标总览</figcaption></figure>
<figure><img loading='lazy' src='{esc(rel(Path(crop_evidence['artifact_paths'][0])))}'><figcaption>Agent自己选择的bbox在contact sheet中的位置</figcaption></figure>
<figure><img loading='lazy' src='{esc(rel(Path(crop_evidence['artifact_paths'][1])))}'><figcaption>camera_crop放大结果，enhancement={esc(crop_evidence['metadata']['enhancement'])}</figcaption></figure>
<figure><img loading='lazy' src='{esc(rel(Path(next(item for item in crop_case['evidence'] if item['tool_name'] == "lidar_detail")['metadata']['model_image_paths'][0])))}'><figcaption>Crop仍无法确认归属后，Agent自主调用LiDAR</figcaption></figure>
</div>
<p>{esc(crop_case['final_reason'])}</p>
<div class='links'><a href='{esc(rel(CROP_RUN / "agent_plan_execute_observe.json"))}'>Crop试跑PEO</a><a href='{esc(rel(CROP_RUN / "system_prompt.txt"))}'>Crop版提示词</a><a href='{esc(rel(CROP_RUN / "replay_verification.json"))}'>Crop Replay</a></div>
<p>replay_verified={str(bool(crop_replay.get('verified'))).lower()}。裁剪只放大原始像素，不产生占用标签；最终仍可选择Unknown。</p></section>

<section class='section' id='gt'><h2>6. 与锁定GT逐车位比较</h2>
<p>Camera-visible子集：{visible_metrics['evaluated_gt_slots']}例，准确率{pct(visible_metrics['three_class_accuracy'])}；Agent处理的occluded子集：{occluded_metrics['evaluated_gt_slots']}例，三分类准确率{pct(occluded_metrics['three_class_accuracy'])}。</p>
<div class='scroll'><table><thead><tr><th>slot</th><th>GT</th><th>最终系统</th><th>正确</th><th>GT可观测性</th><th>GT置信度</th><th>工具</th><th>预测置信度</th></tr></thead><tbody>{''.join(gt_table)}</tbody></table></div></section>

<section class='section' id='ablation'><h2>7. Part1历史帧消融</h2>
<p>扫描W∈{{15,30,45,60}}与K∈{{3,5,10,15}}。先看终态选择性准确率和Unknown误解析，再比较三分类准确率、覆盖与采样开销；本anchor选择W45/K3。</p>
<div class='scroll'><table><thead><tr><th>W</th><th>K</th><th>三分类准确率</th><th>终态覆盖</th><th>选择性准确率</th><th>Unknown误解析</th><th>分布</th><th>选择</th></tr></thead><tbody>{''.join(ablation_rows)}</tbody></table></div></section>

<section class='section' id='qwen'><h2>8. 中心VLM模型大小消融：2B vs 0.8B</h2>
<p>本地自部署Qwen 2B与0.8B使用完全相同的Part1输入、11-case范围、Camera/Crop/LiDAR工具、严格JSON Schema、系统提示和0.6终态门；GT不传入模型。0.8B是“0.7B档”的官方近似规格。</p>
<div class='scroll'><table><thead><tr><th>模型</th><th>14-GT准确率</th><th>Agent 11准确率</th><th>Camera-visible</th><th>终态覆盖</th><th>首工具Camera</th><th>接受工具</th><th>契约/运行时错误</th><th>Replay</th></tr></thead><tbody>{''.join(qwen_rows)}</tbody></table></div>
<div class='note'><b>行为差异：</b>2B在11/11 case先成功调用camera_context，但随后46次Crop提议因没有建立合规的Camera定位假设而被拒；0.8B首工具Camera为0/11，60次有效原始动作均是直接Final，且没有读取图片。两者最终均回退Unknown，因此合并14-GT准确率为50.00%、Camera-visible为0%。</div>
<div class='note warn'><b>不能靠降阈值修复：</b>小模型主要失败在工具规划、证据引用和定位合同，不是0.6阈值本身。放宽终态门会把“未读Camera的Free/Occupied提议”作为答案，增加假终态风险。</div>
<p>Qwen pair prompt_identical={str(bool(qwen_ablation['experimental_control']['qwen_pair_prompt_identical'])).lower()}。既有OpenAI 92.86%列只作上下文参考，其提示词早于camera_crop增补，不能用于严格模型大小因果归因。<a href='{esc(rel(QWEN_REPORT))}'>打开完整VLM消融与逐车位表</a>。</p></section>

<section class='section' id='audit'><h2>9. 复现与审计文件</h2>
<div class='links'>
<a href='{esc(rel(METRICS_PATH))}'>最终指标JSON</a>
<a href='{esc(rel(FRAME / "locked_gt_experiments/final_camera_first_agent_rows.csv"))}'>逐车位CSV</a>
<a href='{esc(rel(RUN / "agent_plan_execute_observe.json"))}'>PEO轨迹JSON</a>
<a href='{esc(rel(RUN / "system_prompt.txt"))}'>正式提示词</a>
<a href='{esc(rel(RUN / "replay_verification.json"))}'>Replay校验</a>
<a href='{esc(rel(GT_PATH))}'>锁定GT</a>
<a href='{esc(rel(GT_REVISION))}'>GT修订记录</a>
<a href='{esc(rel(ABLATION_PATH))}'>16组消融JSON</a>
<a href='{esc(rel(QWEN_ABLATION_PATH))}'>VLM消融JSON</a>
<a href='{esc(rel(QWEN_REPORT))}'>VLM消融HTML</a>
<a href='{esc(rel(FRAME / "qwen_ablation/deployment_manifest.json"))}'>Qwen部署清单</a>
</div>
<p>replay_verified={str(bool(replay.get('verified'))).lower()}；Camera-first policy compliance={pct(metrics['camera_first_policy_compliance_rate'])}。</p></section>
</main></body></html>"""

    report_path = REPORT / "frame6241_camera_first_report.html"
    report_path.write_text(html_text, encoding="utf-8")
    index.write_text(html_text, encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "index": str(index),
        "bytes": len(html_text.encode("utf-8")),
        "replay_verified": replay.get("verified"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
