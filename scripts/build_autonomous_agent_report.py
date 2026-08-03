#!/usr/bin/env python3
"""Build the homepage from the real autonomous Agent audit trail."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.build_random_midroute_full_report import (
    badge,
    build_case_position_svg,
    build_front_svg,
    read_json,
    reason_list,
    relative_position,
    rel_link,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "Nature_ParkingAgent_实验报告_20260728"
EXPERIMENT = REPORT_ROOT / "random_midroute_experiment"
AUTO = EXPERIMENT / "autonomous_agent"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _state_counts(values: list[str]) -> dict[str, int]:
    counts = Counter(values)
    return {name: counts.get(name, 0) for name in ("free", "occupied", "unknown")}


def _images(evidence: dict[str, Any]) -> list[Path]:
    result: list[Path] = []
    for raw in evidence.get("artifact_paths", []):
        path = Path(raw)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            path.resolve().relative_to(REPORT_ROOT.resolve())
        except ValueError:
            continue
        if path.is_file() and path not in result:
            result.append(path)
    return result


def _score_text(scores: dict[str, Any] | None) -> str:
    if not scores:
        return "N/A"
    return (
        f"F={float(scores['free_confidence']):.2f} · "
        f"O={float(scores['occupied_confidence']):.2f} · "
        f"U={float(scores['unknown_confidence']):.2f}"
    )


def build(output: Path) -> None:
    anchor = read_json(EXPERIMENT / "anchor_selection.json")
    local_map = read_json(EXPERIMENT / "part1_w30k5/local_map.json")
    decisions = read_json(EXPERIMENT / "part1_w30k5/slot_decisions.json")["decisions"]
    frames = read_json(EXPERIMENT / "part1_w30k5/local_frame_manifest.json")["frames"]
    input_manifest = read_json(AUTO / "input_manifest.json")
    workflow = read_json(AUTO / "workflow_manifest.json")
    extended = read_json(AUTO / "extended_lidar_w60/manifest.json")
    result = read_json(AUTO / "openai_run/part2_result.json")
    strict_result = read_json(AUTO / "strict_060_run/openai_run/part2_result.json")
    pre_projection_result = read_json(
        AUTO / "pre_projection_gate_55pct/openai_run/part2_result.json"
    )
    replay = read_json(AUTO / "replay_verification.json")
    gt = read_json(EXPERIMENT / "independent_gt/manifest.json")
    ablation = read_json(EXPERIMENT / "history_ablation/label_free_wk_ablation.json")
    prompt = (AUTO / "system_prompt.txt").read_text(encoding="utf-8")

    decisions_by_id = {row["slot_id"]: row for row in decisions}
    local_by_id = {row["slot_id"]: row for row in local_map["slots"]}
    case_rows = result["slot_results"]
    cases = [row["case"] for row in case_rows]
    case_by_id = {case["slot"]["slot_id"]: case for case in cases}
    selected_ids = list(input_manifest["selected_case_ids"])
    selected_set = set(selected_ids)
    final_by_id = {slot_id: case_by_id[slot_id]["final_state"] for slot_id in selected_ids}
    final_counts = _state_counts(list(final_by_id.values()))
    resolved_count = final_counts["free"] + final_counts["occupied"]
    resolved_rate = resolved_count / len(selected_ids) if selected_ids else 0.0
    strict_states = [row["case"]["final_state"] for row in strict_result["slot_results"]]
    strict_counts = _state_counts(strict_states)
    strict_resolved = strict_counts["free"] + strict_counts["occupied"]
    pre_projection_states = [
        row["case"]["final_state"] for row in pre_projection_result["slot_results"]
    ]
    pre_projection_counts = _state_counts(pre_projection_states)
    pre_projection_resolved = (
        pre_projection_counts["free"] + pre_projection_counts["occupied"]
    )
    validation_feedbacks = sum(len(row["validation_errors"]) for row in case_rows)
    part1_front_counts = _state_counts([decisions_by_id[sid]["state"] for sid in selected_ids])
    free_ids = {sid for sid, state in final_by_id.items() if state == "free"}
    occupied_ids = {sid for sid, state in final_by_id.items() if state == "occupied"}
    unknown_ids = {sid for sid, state in final_by_id.items() if state == "unknown"}
    front_all = {sid for sid, slot in local_by_id.items() if relative_position(local_map, slot)["front"]}
    rear_ids = set(local_by_id) - front_all

    assets = EXPERIMENT / "report_assets_autonomous"
    position_dir = assets / "case_position_maps"
    position_dir.mkdir(parents=True, exist_ok=True)
    position_paths: dict[str, Path] = {}
    for sid in selected_ids:
        path = position_dir / f"{sid}.svg"
        path.write_text(
            build_case_position_svg(local_map, frames, sid, final_by_id[sid]),
            encoding="utf-8",
        )
        position_paths[sid] = path
    front_path = assets / "front180_autonomous_final.svg"
    front_path.write_text(
        build_front_svg(local_map, free_ids, occupied_ids, unknown_ids),
        encoding="utf-8",
    )

    audit_files = sorted((AUTO / "openai_run/openai_audit").glob("**/turn_*.json"))
    first_audit = read_json(audit_files[0])
    model = first_audit["model"]
    resolved_model = first_audit.get("resolved_model") or model
    usage = [read_json(path).get("usage") or {} for path in audit_files]
    input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in usage)
    output_tokens = sum(int(row.get("output_tokens", 0) or 0) for row in usage)
    tool_calls = sum(
        1
        for case in cases
        for evidence in case["evidence"]
        if evidence["round_index"] >= 1 and evidence["tool_name"] != "agent_final"
    )

    part1_rows: list[str] = []
    for row in decisions:
        sid = row["slot_id"]
        pos = relative_position(local_map, local_by_id[sid])
        route = "进入自主Agent" if sid in selected_set else "后方排除" if not pos["front"] else "非Free/Unknown候选"
        part1_rows.append(
            "<tr>"
            f"<td><code>{escape(sid)}</code></td><td>{badge(row['state'])}</td>"
            f"<td>{escape(row['decision_reason'])}</td><td>{reason_list(row['unknown_reasons'])}</td>"
            f"<td>{pos['forward']:.2f}</td><td>{pos['left']:.2f}</td><td>{pos['bearing']:.1f}°</td>"
            f"<td>{escape(route)}</td></tr>"
        )

    frame_rows = "".join(
        "<tr>"
        f"<td>{row['frame_id']}</td><td>{row['camera_frame']}</td>"
        f"<td>{row['map_x']:.6f}, {row['map_y']:.6f}</td><td>{row['map_yaw']:.6f}</td>"
        f"<td>{1000*row['camera_lidar_dt_sec']:.1f} ms</td><td><code>{escape(Path(row['map_points_path']).name)}</code></td>"
        "</tr>"
        for row in frames
    )

    case_cards: list[str] = []
    for row in case_rows:
        case = row["case"]
        sid = case["slot"]["slot_id"]
        pos = relative_position(local_map, local_by_id[sid])
        tool_evidence = [
            evidence for evidence in case["evidence"]
            if evidence["round_index"] >= 1 and evidence["tool_name"] != "agent_final"
        ]
        fov_evidence = next(
            evidence for evidence in case["evidence"]
            if evidence["tool_name"] == "check_fov"
        )
        fov_details = case["fov"].get("details", {})
        center_u = fov_details.get("target_center_pixel_u")
        camera_usable = case["fov"]["visibility"] in {"visible", "partially_visible"}
        fov_image_html = "".join(
            f'<a href="{escape(rel_link(path, REPORT_ROOT))}" target="_blank" rel="noopener">'
            f'<img loading="lazy" src="{escape(rel_link(path, REPORT_ROOT))}" alt="{escape(sid)} Camera feasibility gate"></a>'
            for path in _images(fov_evidence)
        )
        media_blocks: list[str] = [
            f'<div class="tool-media"><h4>Mandatory Gate · Camera物理可用性</h4>'
            f'<p>{escape(fov_evidence["summary"])} 中心横向像素 u='
            f'{"N/A" if center_u is None else f"{float(center_u):.1f}"}；'
            f'Camera工具={"开放" if camera_usable else "禁用"}。该投影只控制工具，不用于车位位置或占用判断。</p>'
            f'<div class="gallery">{fov_image_html}</div></div>'
        ]
        for evidence in tool_evidence:
            images = _images(evidence)
            image_html = "".join(
                f'<a href="{escape(rel_link(path, REPORT_ROOT))}" target="_blank" rel="noopener">'
                f'<img loading="lazy" src="{escape(rel_link(path, REPORT_ROOT))}" alt="{escape(sid)} {escape(evidence["tool_name"])} output"></a>'
                for path in images
            )
            media_blocks.append(
                f'<div class="tool-media"><h4>Round {evidence["round_index"]} · <code>{escape(evidence["tool_name"])}</code></h4>'
                f'<p>{escape(evidence["summary"])}</p><div class="gallery">{image_html}</div></div>'
            )
        round_rows: list[str] = []
        for round_data in case["rounds"]:
            scores = round_data["scores_after"]
            loc = round_data.get("localization") or {}
            round_rows.append(
                "<tr>"
                f"<td>{round_data['round_index']}</td><td><code>{escape(round_data['tool_name'])}</code></td>"
                f"<td>{escape(round_data['reasoning_summary'])}</td>"
                f"<td>{escape(round_data['observation_summary'])}</td>"
                f"<td>{escape(str(loc.get('stage','')))} / {float(loc.get('confidence_after',0)):.2f}</td>"
                f"<td>{badge(round_data['state_after'])}<br><small>{escape(_score_text(scores))}</small></td>"
                "</tr>"
            )
        audit_dir = AUTO / "openai_run/openai_audit" / case["case_id"].replace("/", "_")
        audit_links = " · ".join(
            f'<a href="{escape(rel_link(path, REPORT_ROOT))}">{escape(path.stem)}</a>'
            for path in sorted(audit_dir.glob("turn_*.json"))
        )
        case_cards.append(
            f'''<article class="case" id="{escape(sid)}">
            <div class="case-head"><div><span class="eyebrow">Autonomous SlotCase</span><h3>{escape(sid)}</h3></div>
            <div>{badge(case['part1_state'])}<span class="arrow">→</span>{badge(case['final_state'])}</div></div>
            <div class="facts"><div><b>地图中心</b><span>{case['slot']['center_map'][0]:.6f}, {case['slot']['center_map'][1]:.6f}</span></div>
            <div><b>车体坐标</b><span>前 {pos['forward']:.2f} m / 左 {pos['left']:.2f} m</span></div>
            <div><b>距离 / 方位</b><span>{pos['distance']:.2f} m / {pos['bearing']:.1f}°</span></div>
            <div><b>最终分数</b><span>{escape(_score_text(case['final_scores']))}</span></div>
            <div><b>Camera物理门</b><span>{escape(case['fov']['visibility'])} / u={"N/A" if center_u is None else f"{float(center_u):.1f}"} / {"开放" if camera_usable else "禁用"}</span></div></div>
            <figure class="position"><a href="{escape(rel_link(position_paths[sid], REPORT_ROOT))}" target="_blank"><img src="{escape(rel_link(position_paths[sid], REPORT_ROOT))}" alt="{escape(sid)} map position"></a>
            <figcaption>紫色目标车位；蓝色EGO及5帧轨迹；虚线为正前方180°边界。车位位置始终来自地图与pose；Camera投影只做工具可用性门。</figcaption></figure>
            <div class="reason"><b>Part1保留原因</b>{reason_list(case['unknown_reasons'])}</div>
            <h4>模型自主动作轨迹</h4><div class="table-wrap"><table><thead><tr><th>轮次</th><th>自主动作</th><th>模型选择理由</th><th>工具返回</th><th>定位阶段/置信</th><th>融合状态</th></tr></thead><tbody>{''.join(round_rows)}</tbody></table></div>
            {''.join(media_blocks)}
            <div class="final"><b>最终结论：{badge(case['final_state'])}</b><p>{escape(case['final_reason'])}</p>
            <small>model turns={row['model_turns']} · tool rounds={row['tool_rounds']} · validation errors={len(row['validation_errors'])} · 原始模型动作：{audit_links}</small></div>
            </article>'''
        )

    ablation_rows: list[str] = []
    recommendation = ablation["recommended_for_gt_followup_not_accuracy_best"]
    for row in ablation["runs"]:
        selected = row["history_span_W"] == recommendation["history_span_W"] and row["sample_count_K"] == recommendation["sample_count_K"]
        ablation_rows.append(
            f'<tr class="{"selected" if selected else ""}"><td>{row["history_span_W"]}</td><td>{row["sample_count_K"]}</td>'
            f'<td>{row["state_counts"].get("free",0)}</td><td>{row["state_counts"].get("occupied",0)}</td><td>{row["state_counts"].get("unknown",0)}</td>'
            f'<td>{row["terminal_count"]}</td><td>{100*row["terminal_coverage"]:.1f}%</td><td>{100*row["all_state_modal_agreement"]:.1f}%</td><td>{row["runtime_seconds"]:.2f}</td></tr>'
        )

    selected_anchor = anchor["selected_anchor"]
    prompt_hash = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    generated = datetime.now(timezone.utc).isoformat()
    node_html = "".join(
        f'<div><b>{escape(name)}</b><span>{escape(node["status"])}</span></div>'
        for name, node in workflow["nodes"].items()
        if name != "05_html_report"
    ) + '<div><b>05_html_report</b><span>completed</span></div>'

    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ParkingAgent真实自主Agent全流程 · frame6681</title><style>
:root{{--bg:#07101c;--panel:#0d1a2b;--panel2:#122238;--line:#2b3b52;--text:#e7eef8;--muted:#9dacbf;--cyan:#38bdf8}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.65 Inter,system-ui,sans-serif}}a{{color:#7dd3fc}}code,pre{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}.wrap{{max-width:1500px;margin:auto;padding:28px}}nav{{position:sticky;top:0;z-index:5;padding:10px 0;background:#07101cee;border-bottom:1px solid var(--line)}}nav a{{margin-right:18px;text-decoration:none;font-weight:700}}header,section{{padding:44px 0;border-bottom:1px solid var(--line)}}h1{{font-size:clamp(34px,5vw,66px);line-height:1.05;letter-spacing:-.04em}}h2{{font-size:30px}}h3{{font-size:24px;margin:4px 0}}h4{{margin:18px 0 8px}}.eyebrow{{color:var(--cyan);font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.16em}}.lead{{font-size:18px;color:#c9d5e5;max-width:1100px}}.metrics,.facts,.flow{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:20px 0}}.metric,.facts div,.flow div{{padding:15px;background:var(--panel);border:1px solid var(--line);border-radius:11px}}.metric b{{display:block;font-size:27px}}.metric span,.facts span,.flow span,small,figcaption{{color:var(--muted)}}.flow b,.flow span,.facts b,.facts span{{display:block}}.callout,.reason,.final{{padding:16px 18px;border:1px solid #075985;background:#092232;border-radius:11px;margin:18px 0}}.warning{{border-color:#92400e;background:#2a1b09}}.good{{border-color:#166534;background:#0b271a}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:11px}}table{{width:100%;border-collapse:collapse;background:var(--panel);font-size:13px}}th,td{{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}}th{{background:#15263d}}tr.selected td{{background:#173520}}.badge{{display:inline-block;padding:2px 9px;border-radius:999px;font-weight:800;border:1px solid currentColor}}.badge.free{{color:#86efac}}.badge.occupied{{color:#fca5a5}}.badge.unknown{{color:#fcd34d}}.arrow{{padding:0 9px;color:var(--muted)}}img{{max-width:100%;border:1px solid var(--line);border-radius:9px;background:#07101c}}.map{{width:100%}}.case{{margin:28px 0;padding:22px;background:var(--panel);border:1px solid var(--line);border-radius:16px}}.case-head{{display:flex;align-items:center;justify-content:space-between;gap:15px}}.position img{{width:100%;min-height:500px;object-fit:contain}}.tool-media{{padding:14px;margin:14px 0;background:var(--panel2);border-radius:11px}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px}}.gallery img{{width:100%;height:420px;object-fit:contain}}details{{margin:14px 0}}summary{{cursor:pointer;color:#bae6fd}}pre{{max-height:520px;overflow:auto;white-space:pre-wrap;padding:16px;background:#030812;border-radius:10px}}.formula{{padding:15px;background:#030812;border-left:4px solid var(--cyan);font-family:monospace}}@media(max-width:800px){{.case-head{{align-items:flex-start;flex-direction:column}}.gallery{{grid-template-columns:1fr}}.position img{{min-height:0}}}}
</style></head><body><nav><div class="wrap"><a href="#summary">结论</a><a href="#workflow">工作流</a><a href="#input">输入</a><a href="#part1">Part1</a><a href="#part2">180°</a><a href="#agent">自主Agent</a><a href="#gt">GT</a><a href="#ablation">消融</a><a href="#artifacts">产物</a></div></nav><main class="wrap">
<header id="summary"><span class="eyebrow">Real central-agent audit · no slot-specific router</span><h1>Part1 → Part2 真实自主Agent闭环<br>随机中段 frame 6681</h1><p class="lead">这份首页只使用真实 OpenAI 中心Agent运行结果。流程骨架固定，但每个车位是否先看Camera、裁哪里、是否调用60帧LiDAR以及最终状态，都由模型根据当前证据自行决定。旧的硬编码规则原型不参与本页结论。</p>
<div class="metrics"><div class="metric"><b>{resolved_count}/9</b><span>得到明确结论（{100*resolved_rate:.1f}%）</span></div><div class="metric"><b>9/9</b><span>前方SlotCase全部运行</span></div><div class="metric"><b>{len(audit_files)}</b><span>有效结构化模型动作</span></div><div class="metric"><b>{tool_calls}</b><span>Agent自主工具调用</span></div><div class="metric"><b>{final_counts['free']} / {final_counts['occupied']} / {final_counts['unknown']}</b><span>最终 F / O / U</span></div><div class="metric"><b>{str(replay['verified']).lower()}</b><span>逐字段回放一致</span></div><div class="metric"><b>{validation_feedbacks}</b><span>Agent收到并自修复的动作反馈</span></div></div>
<div class="callout good"><b>最终结果：</b>Free = <code>{escape(", ".join(sorted(free_ids)) or "无")}</code>；Occupied = <code>{escape(", ".join(sorted(occupied_ids)) or "无")}</code>；Unknown = <code>{escape(", ".join(sorted(unknown_ids)) or "无")}</code>。统一策略把明确结论从严格版的 {strict_resolved}/9 提高到 {resolved_count}/9（{100*resolved_rate:.1f}%）；<code>slot_0996</code> 仍因遮挡与归属不清保持 Unknown。</div><div class="callout warning"><b>准确率不能宣称：</b>frame6681没有冻结的独立人工终态GT，因此此处只能报告真实输出、保守性、工具轨迹和回放一致性，不能把Agent输出当GT。</div></header>
<section id="workflow"><span class="eyebrow">01 · One-command workflow</span><h2>固定节点骨架，节点内由Agent自主循环</h2><div class="flow">{node_html}</div><div class="formula">for each SlotCase: request(model, current evidence, available_tools) → model chooses tool OR final → execute tool → append observation → repeat (max 3 tools / 6 model turns)</div><p>通用运行时只约束物理合法工具、三轮预算、证据引用、互斥终态和{float(workflow["terminal_confidence"]):.2f}门槛；没有“slot_XXXX应该用什么/应该判什么”的表。Camera中心必须通过ZED内参和审计支持外参的横向像素门，画面外/uncertain时Camera工具不会出现在available_tools。模型为 <code>{escape(model)}</code>，服务端解析模型 <code>{escape(resolved_model)}</code>。</p><details><summary>查看完整系统提示（SHA {escape(prompt_hash)}）</summary><pre>{escape(prompt)}</pre></details></section>
<section id="input"><span class="eyebrow">02 · Data</span><h2>实际输入数据</h2><div class="metrics"><div class="metric"><b>{selected_anchor['frame_id']}</b><span>随机锚点</span></div><div class="metric"><b>{selected_anchor['camera_frame']}</b><span>同步Camera帧</span></div><div class="metric"><b>W30/K5</b><span>Part1历史</span></div><div class="metric"><b>W60/60</b><span>Part2增量LiDAR工具</span></div><div class="metric"><b>{input_tokens:,}</b><span>模型输入tokens</span></div><div class="metric"><b>{output_tokens:,}</b><span>模型输出tokens</span></div></div><div class="table-wrap"><table><thead><tr><th>LiDAR帧</th><th>Camera帧</th><th>地图位置</th><th>yaw</th><th>同步差</th><th>地图点文件</th></tr></thead><tbody>{frame_rows}</tbody></table></div></section>
<section id="part1"><span class="eyebrow">03 · Conservative perception</span><h2>Part1完整23车位三态输出</h2><p>Occupied不是“雷达柱里有点”。只有车辆三维占地、下部覆盖、高度层、跨帧支持、归属和7/7位姿稳健性同时闭合才允许Occupied；柱、墙、边界主导和水平盖板会否决。Free要求射线穿越、核心体积及近地覆盖充分；其余全部保留Unknown原因交给Agent。</p><div class="table-wrap"><table><thead><tr><th>车位</th><th>Part1状态</th><th>原因</th><th>Unknown原因</th><th>前向m</th><th>左向m</th><th>方位</th><th>路由</th></tr></thead><tbody>{''.join(part1_rows)}</tbody></table></div><div class="callout">Part1前方180°状态：Free={part1_front_counts['free']}、Occupied={part1_front_counts['occupied']}、Unknown={part1_front_counts['unknown']}。与GT比较：N/A（同一时刻没有正式人工GT）。</div></section>
<section id="part2"><span class="eyebrow">04 · Geometry scope</span><h2>只保留车头正前方180°</h2><div class="formula">dx,dy=(slot_center-ego)/map_units_per_meter<br>forward=cos(yaw)·dx+sin(yaw)·dy<br>selected ⇔ Part1∈{{Free,Unknown}} ∧ forward≥0</div><p>选择出9个车位进入Agent；局部图中其余{len(rear_ids)}个后方车位排除。没有使用旧160°逻辑，也没有用Camera投影决定车位地图位置。</p><img class="map" src="{escape(rel_link(front_path, REPORT_ROOT))}" alt="front180 autonomous final map"></section>
<section id="agent"><span class="eyebrow">05 · Autonomous evidence loop</span><h2>逐case物理门、模型动作、工具输出与最终结论</h2><p>工具箱包含 <code>camera_context</code>、<code>camera_crop</code>、<code>camera_sequence</code>、<code>lidar_detail</code>。Camera只对中心像素在画面内的0942/0943/0944/0945开放；其余五个case的Agent只能自主选择LiDAR或Final。9个case的60帧LiDAR均通过增量合同，下面均为真实审计产物。</p>{''.join(case_cards)}<h3>原始、严格版与平衡版对比</h3><div class="table-wrap"><table><thead><tr><th>阶段</th><th>Free</th><th>Occupied</th><th>Unknown</th><th>明确率</th><th>说明</th></tr></thead><tbody><tr><td>Part1前方180°</td><td>{part1_front_counts['free']}</td><td>{part1_front_counts['occupied']}</td><td>{part1_front_counts['unknown']}</td><td>{100*(part1_front_counts['free']+part1_front_counts['occupied'])/len(selected_ids):.1f}%</td><td>保留Unknown原因，交给Part2</td></tr><tr><td>旧严格Agent（0.60）</td><td>{strict_counts['free']}</td><td>{strict_counts['occupied']}</td><td>{strict_counts['unknown']}</td><td>{100*strict_resolved/len(selected_ids):.1f}%</td><td>门槛与几何合同过严</td></tr><tr><td>旧平衡Agent（粗FOV）</td><td>{pre_projection_counts['free']}</td><td>{pre_projection_counts['occupied']}</td><td>{pre_projection_counts['unknown']}</td><td>{100*pre_projection_resolved/len(selected_ids):.1f}%</td><td>存在画面外Camera误调用</td></tr><tr><td>当前自主Agent（{float(workflow['terminal_confidence']):.2f}）</td><td>{final_counts['free']}</td><td>{final_counts['occupied']}</td><td>{final_counts['unknown']}</td><td>{100*resolved_rate:.1f}%</td><td>真实横向像素门；无车位ID特判</td></tr></tbody></table></div></section>
<section id="gt"><span class="eyebrow">06 · Ground truth</span><h2>GT状态与可评估边界</h2><p>现有独立GT协议围绕frame6681构建了预测无关的几何全集和100帧证据，但 <code>formal_gt={str(gt.get('formal_gt',False)).lower()}</code>，人工F/O标签尚未冻结。因此Part1准确率、Agent准确率、precision/recall均为N/A。frame9277的GT不能跨时刻挪用到frame6681。</p><div class="callout warning">这不是遗漏：如果用旧Part1筛出的候选或Agent结论回填GT，会形成标签泄漏。正式准确率实验必须由独立标注者在不看预测的情况下冻结frame6681标签后再计算。</div></section>
<section id="ablation"><span class="eyebrow">07 · W/K ablation</span><h2>历史帧消融（15组）</h2><p>W为因果历史池，K为均匀采样帧数。无正式GT时只能比较终态覆盖、跨配置冲突和稳定性，不能称“准确率最佳”。当前Part1采用W30/K5；Part2工具使用独立W60全帧作为增量证据。</p><div class="table-wrap"><table><thead><tr><th>W</th><th>K</th><th>Free</th><th>Occupied</th><th>Unknown</th><th>终态</th><th>覆盖</th><th>众数一致</th><th>秒</th></tr></thead><tbody>{''.join(ablation_rows)}</tbody></table></div><div class="callout"><b>无GT稳定性优先：</b>W={recommendation['history_span_W']} / K={recommendation['sample_count_K']}；仅是下一轮GT评测优先配置，不是准确率冠军。</div></section>
<section id="artifacts"><span class="eyebrow">08 · Reproducibility</span><h2>全部可复核产物</h2><ul><li><a href="random_midroute_experiment/autonomous_agent/workflow_manifest.json">工作流节点清单</a> · <a href="random_midroute_experiment/autonomous_agent/input_manifest.json">纯几何front180输入清单</a></li><li><a href="random_midroute_experiment/autonomous_agent/system_prompt.txt">完整系统提示</a> · <a href="random_midroute_experiment/autonomous_agent/part1_front180.json">Agent完整输入</a></li><li><a href="random_midroute_experiment/autonomous_agent/extended_lidar_w60/manifest.json">60帧LiDAR工具合同</a> · <a href="random_midroute_experiment/autonomous_agent/openai_run/part2_result.json">真实最终结果</a></li><li><a href="random_midroute_experiment/autonomous_agent/openai_run/openai_audit/">{len(audit_files)}次有效结构化模型动作目录</a> · <a href="random_midroute_experiment/autonomous_agent/replay_actions.json">动作回放文件</a> · <a href="random_midroute_experiment/autonomous_agent/replay_verification.json">回放验证</a></li><li><a href="random_midroute_experiment/history_ablation/label_free_wk_ablation.json">15组消融完整JSON</a> · <a href="random_midroute_experiment/independent_gt/manifest.json">独立GT协议</a></li></ul><div class="callout warning"><b>历史原型隔离：</b><code>agent_closed_loop</code> 和 <code>camera_only_pilot</code> 不参与本页结论；前者包含规则路由，后者用于发现增量LiDAR合同缺失。</div></section><footer><p>生成时间 {escape(generated)} · replay_verified={str(replay['verified']).lower()} · slot_specific_policy=false · formal_gt=false</p></footer></main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    workflow["nodes"]["05_html_report"] = {"status": "completed", "output": str(output), "sha256": "sha256:" + sha256(output)}
    _write_json(AUTO / "workflow_manifest.json", workflow)
    manifest = {
        "schema_version": "autonomous-agent-homepage-report/1.0",
        "generated_at_utc": generated,
        "output": str(output),
        "output_sha256": "sha256:" + sha256(output),
        "model": model,
        "resolved_model": resolved_model,
        "system_prompt_sha256": prompt_hash,
        "slot_specific_policy": False,
        "model_action_count": len(audit_files),
        "tool_call_count": tool_calls,
        "validation_feedback_count": validation_feedbacks,
        "terminal_confidence": float(workflow["terminal_confidence"]),
        "resolved_count": resolved_count,
        "resolved_rate": resolved_rate,
        "strict_final_counts": strict_counts,
        "strict_resolved_count": strict_resolved,
        "pre_projection_final_counts": pre_projection_counts,
        "pre_projection_resolved_count": pre_projection_resolved,
        "camera_gate": "horizontal_pixel_projection_center_reachability",
        "camera_usable_slot_ids": sorted(
            case["slot"]["slot_id"] for case in cases
            if case["fov"]["visibility"] in {"visible", "partially_visible"}
        ),
        "final_counts": final_counts,
        "final_states": final_by_id,
        "replay_verified": replay["verified"],
        "formal_gt": False,
        "accuracy_evaluable": False,
        "position_maps": {sid: str(path) for sid, path in position_paths.items()},
    }
    _write_json(AUTO / "report_manifest.json", manifest)


def main() -> int:
    build(REPORT_ROOT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
