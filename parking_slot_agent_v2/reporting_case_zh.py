"""Single-case before/after report with an auditable decision trace."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

from .reporting_story_zh import _draw_wrapped, _fit_image, _font, _rounded_box


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _rel(path: str | Path, base: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), base.resolve())).as_posix()


def _find_case(result: Mapping[str, Any], slot_id: str) -> Mapping[str, Any]:
    for row in _list(result.get("slot_results")):
        item = _dict(row)
        slot = _dict(_dict(item.get("case")).get("slot"))
        if slot.get("slot_id") == slot_id:
            return item
    raise ValueError(f"slot not found: {slot_id}")


def _find_evidence(case: Mapping[str, Any], tool: str) -> Mapping[str, Any]:
    for raw in _list(case.get("evidence")):
        evidence = _dict(raw)
        if evidence.get("tool_name") == tool:
            return evidence
    raise ValueError(f"evidence not found: {tool}")


def _annotate_original_map(source: Path, target: Path, slot_id: str) -> None:
    with Image.open(source) as raw:
        image = raw.convert("RGB")
    # Fixed frame-9277 local-map axes: the slot_1012 center is approximately
    # (4.14 m, -8.43 m) in ego coordinates, mapping to this pixel location.
    draw = ImageDraw.Draw(image)
    center = (841, 753)
    radius = 43
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        outline="#dc2626",
        width=8,
    )
    label = f"{slot_id}: Part1 Unknown"
    font = _font(25)
    label_box = (center[0] - 170, center[1] + 51, center[0] + 170, center[1] + 96)
    draw.rounded_rectangle(label_box, radius=9, fill="#fff1f2", outline="#dc2626", width=3)
    draw.text((label_box[0] + 12, label_box[1] + 7), label, font=font, fill="#b91c1c")
    image.save(target)


def _build_transition_visual(
    path: Path,
    *,
    original_map: Path,
    fov_map: Path,
    lidar_image: Path,
) -> None:
    canvas = Image.new("RGB", (2100, 1320), "#f4f7fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((64, 43), "slot_1012：从 Part1 Unknown 到有证据的 Occupied", font=_font(51), fill="#102a43")
    draw.text((66, 112), "重点不是“状态变了”，而是每一步为什么变、用了什么证据，都能回查。", font=_font(27), fill="#40566d")

    sources = [
        ("① Part1 原始地图", original_map),
        ("② FOV 路由结果", fov_map),
        ("③ 15帧 LiDAR 细节", lidar_image),
    ]
    panel_width = 620
    for index, (label, source) in enumerate(sources):
        x = 64 + index * 680
        y = 190
        fitted = _fit_image(source, (panel_width, 650))
        canvas.paste(fitted, (x, y))
        draw.rectangle((x, y, x + panel_width, y + 650), outline="#94a3b8", width=3)
        draw.text((x, y + 668), label, font=_font(28), fill="#27384a")

    steps = [
        ("Part1", "Unknown", "Free=0.667 与 Occupied=0.929 冲突，且都不是校准概率。", "#64748b"),
        ("FOV", "Camera 禁用", "目标方位 −113°，远在可靠半视场 ±40° 之外。", "#d97706"),
        ("工具", "自动 LiDAR", "系统调用 9263–9277 共15帧目标局部点云。", "#2563eb"),
        ("模型融合", "Occupied 0.92", "引用 FOV+LiDAR；状态机一次验收通过。", "#be123c"),
    ]
    y1 = 955
    box_width = 465
    for index, (stage, state, detail, color) in enumerate(steps):
        x1 = 64 + index * 505
        _rounded_box(draw, (x1, y1, x1 + box_width, y1 + 255), fill="#ffffff", outline=color, radius=18, width=3)
        draw.text((x1 + 22, y1 + 18), stage, font=_font(25), fill=color)
        draw.text((x1 + 22, y1 + 66), state, font=_font(37), fill="#102a43")
        _draw_wrapped(draw, (x1 + 22, y1 + 126), detail, font=_font(22), fill="#40566d", width=415)
        if index < len(steps) - 1:
            draw.polygon(
                [(x1 + box_width + 10, y1 + 116), (x1 + box_width + 34, y1 + 132), (x1 + box_width + 10, y1 + 148)],
                fill="#64748b",
            )
    canvas.save(path)


def build_single_case_report(
    *,
    slot_id: str,
    part1_map: str | Path,
    baseline_dir: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    baseline_root = Path(baseline_dir).resolve()
    experiment_root = Path(experiment_dir).resolve()
    destination = Path(output_dir).resolve()
    assets = destination / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    baseline_row = _find_case(_load(baseline_root / "part2_result.json"), slot_id)
    experiment_row = _find_case(_load(experiment_root / "part2_result.json"), slot_id)
    baseline_case = _dict(baseline_row.get("case"))
    case = _dict(experiment_row.get("case"))
    part1_scores = _dict(case.get("part1_scores"))
    final_scores = _dict(case.get("final_scores"))
    fov = _dict(case.get("fov"))
    fov_details = _dict(fov.get("details"))
    fov_evidence = _find_evidence(case, "check_fov")
    lidar_evidence = _find_evidence(case, "lidar_detail")
    final_evidence = _find_evidence(case, "agent_final")
    fov_map = Path(str(_list(fov_evidence.get("artifact_paths"))[0])).resolve()
    lidar_image = Path(str(_list(lidar_evidence.get("artifact_paths"))[0])).resolve()
    annotated_map = assets / f"{slot_id}_part1原始地图标注.png"
    _annotate_original_map(Path(part1_map).resolve(), annotated_map, slot_id)
    transition = assets / f"{slot_id}_状态变化与决策链.png"
    _build_transition_visual(
        transition,
        original_map=annotated_map,
        fov_map=fov_map,
        lidar_image=lidar_image,
    )

    audit_files = sorted(
        (experiment_root / "openai_audit").glob(f"*{slot_id}/turn_*.json")
    )
    audit = _load(audit_files[0]) if audit_files else {}
    round_row = _dict(_list(case.get("rounds"))[0])
    trace = {
        "schema_version": "parking-slot-agent-v2-case-decision-trace/1.0",
        "slot_id": slot_id,
        "hidden_chain_of_thought_recorded": False,
        "auditable_decision_summary_recorded": True,
        "before": {
            "state": case.get("part1_state"),
            "scores": part1_scores,
            "decision_reason": case.get("decision_reason"),
            "unknown_reasons": case.get("unknown_reasons"),
        },
        "events": [
            {
                "index": 0,
                "actor": "part1",
                "event": "candidate_state",
                "state": case.get("part1_state"),
                "reason": "weak_obstacle_evidence with conflicting uncalibrated gate scores",
            },
            {
                "index": 1,
                "actor": "deterministic_workflow",
                "event": "mandatory_fov",
                "result": fov,
                "reason": "all target samples are outside the conservative camera FOV",
            },
            {
                "index": 2,
                "actor": "deterministic_workflow",
                "event": "automatic_tool_route",
                "tool": "lidar_detail",
                "reason": round_row.get("reasoning_summary"),
            },
            {
                "index": 3,
                "actor": "lidar_detail_tool",
                "event": "observation",
                "status": lidar_evidence.get("status"),
                "summary": lidar_evidence.get("summary"),
                "metadata": lidar_evidence.get("metadata"),
            },
            {
                "index": 4,
                "actor": "gpt-5.6-terra",
                "event": "final_evidence_fusion",
                "action": audit.get("action"),
                "reason": final_evidence.get("summary"),
                "cited_evidence_ids": _dict(final_evidence.get("metadata")).get("evidence_ids"),
            },
            {
                "index": 5,
                "actor": "state_machine",
                "event": "contract_validation",
                "result": "accepted",
                "reason": "occupied_confidence >= 0.90 and successful lidar_detail evidence is cited",
            },
        ],
        "after": {
            "state": case.get("final_state"),
            "scores": final_scores,
            "stop_reason": experiment_row.get("stop_reason"),
            "validation_errors": experiment_row.get("validation_errors"),
        },
        "baseline": {
            "state": baseline_case.get("final_state"),
            "model_turns": baseline_row.get("model_turns"),
            "tool_rounds": baseline_row.get("tool_rounds"),
            "validation_errors": baseline_row.get("validation_errors"),
        },
        "openai_audit_file": str(audit_files[0]) if audit_files else None,
    }
    trace_path = destination / f"{slot_id}_decision_trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    errors = _list(baseline_row.get("validation_errors"))
    report_path = destination / f"{slot_id}_完整Case报告.html"
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{_esc(slot_id)}完整Case报告</title><style>
body{{margin:0;background:#f4f7fb;color:#172033;font-family:system-ui,-apple-system,"Microsoft YaHei","PingFang SC",sans-serif;line-height:1.7}}main{{max-width:1500px;margin:auto;padding:26px}}h1{{font-size:34px}}h2{{margin-top:40px;border-left:7px solid #2563eb;padding-left:12px}}.hero,.panel{{background:white;border:1px solid #cbd5e1;border-radius:14px;padding:20px;margin:18px 0}}.hero img,.wide{{width:100%;object-fit:contain}}.state-row{{display:grid;grid-template-columns:1fr 80px 1fr;gap:12px;align-items:center}}.state{{padding:18px;border-radius:12px;background:#f8fafc;border:1px solid #cbd5e1}}.state.after{{background:#fee2e2;border-color:#f87171}}.arrow{{font-size:42px;text-align:center}}.timeline{{border-left:5px solid #93c5fd;margin-left:22px;padding-left:30px}}.event{{position:relative;background:white;border:1px solid #cbd5e1;border-radius:10px;padding:15px;margin:16px 0}}.event:before{{content:'';position:absolute;left:-43px;top:25px;width:19px;height:19px;border-radius:50%;background:#2563eb}}.actor{{display:inline-block;padding:2px 9px;border-radius:999px;background:#dbeafe;font-size:12px}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #cbd5e1;padding:8px;vertical-align:top}}th{{background:#e8eef6}}.note{{background:#fff7ed;border:1px solid #fdba74;padding:14px;border-radius:10px}}.good{{background:#ecfdf5;border:1px solid #6ee7b7;padding:14px;border-radius:10px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0f172a;color:#e2e8f0;padding:13px;border-radius:8px}}code{{overflow-wrap:anywhere}}@media(max-width:800px){{.state-row{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg)}}}}</style></head><body><main>
<section class='hero'><h1>{_esc(slot_id)}：原始状态 → Agent取证 → 最终状态</h1><p>这个页面只讲一个车位，不用整体Unknown数量掩盖过程。</p><img src='{_esc(_rel(transition,destination))}'></section>
<section><h2>1. 状态到底怎么变的</h2><div class='state-row'><div class='state'><h3>Part1 原始状态：Unknown</h3><p>Free={float(part1_scores.get('free_confidence',0)):.3f}<br>Occupied={float(part1_scores.get('occupied_confidence',0)):.3f}<br>Unknown={float(part1_scores.get('unknown_confidence',0)):.3f}</p><p>原因：弱障碍证据、边界占主导、车辆形态证据弱。这些分数是未校准gate分数，Free和Occupied同时偏高表示冲突，不是概率和。</p></div><div class='arrow'>→</div><div class='state after'><h3>Agent 最终状态：Occupied</h3><p>Free={float(final_scores.get('free_confidence',0)):.2f}<br>Occupied={float(final_scores.get('occupied_confidence',0)):.2f}</p><p>状态机结果：terminal_confidence；验证错误：0。</p></div></div></section>
<section><h2>2. Agent具体做了什么，为什么这么做</h2><div class='timeline'>
<article class='event'><span class='actor'>Part1</span><h3>发现冲突，建立 SlotCase</h3><p>车位在原始地图上为Unknown。Part1建议继续检查车辆形状和遮挡，不允许直接把0.929解释为确定Occupied。</p></article>
<article class='event'><span class='actor'>确定性工作流</span><h3>强制检查 FOV</h3><p>目标中心方位 {float(fov_details.get('target_bearing_deg',0)):.1f}°，可靠半视场 ±{float(fov_details.get('half_fov_deg',0)):.1f}°；五个目标采样点全部远离可靠视区，因此结果为 not_visible，Camera被禁止。</p></article>
<article class='event'><span class='actor'>确定性工作流</span><h3>自动路由到 LiDAR</h3><p>这里不是模型自由选择工具。规则是：FOV不可见时，先自动获取一次目标局部LiDAR，避免模型反复请求无效Camera。</p></article>
<article class='event'><span class='actor'>LiDAR工具</span><h3>生成可供模型观察的细节图</h3><p>使用 frame 9263–9277 共15帧、{int(_dict(lidar_evidence.get('metadata')).get('point_count',0)):,}个目标局部点，生成BEV、高度纵剖面和归一化最大高度三联图。</p><img class='wide' src='{_esc(_rel(lidar_image,destination))}'></article>
<article class='event'><span class='actor'>gpt-5.6-terra</span><h3>一轮完成证据融合</h3><p>记录的决策摘要：目标局部多帧LiDAR显示持续的高起障碍结构占据目标车位区域；Camera因目标位于粗FOV之外而不可用。</p><p>输出：Occupied=0.92、Free=0.04；引用FOV和LiDAR两条证据。</p></article>
<article class='event'><span class='actor'>状态机</span><h3>合同验收通过</h3><p>Occupied≥0.90，且引用了成功的Part2 LiDAR证据，没有相反的Free终态置信度，因此接受并停止该车位。</p></article></div></section>
<section><h2>3. 改进在哪里</h2><table><tr><th></th><th>旧 Qwen 0.8B</th><th>新工作流</th></tr><tr><td>模型轮次</td><td>{int(baseline_row.get('model_turns',0))}</td><td>{int(experiment_row.get('model_turns',0))}</td></tr><tr><td>细节工具</td><td>0</td><td>1次自动LiDAR</td></tr><tr><td>模型行为</td><td>连续6次想报Occupied，但都没有细节证据</td><td>看到LiDAR后一次输出Occupied</td></tr><tr><td>合同错误</td><td>{len(errors)}次：均为“Occupied缺少成功Part2证据”</td><td>0次</td></tr><tr><td>最终状态</td><td>Unknown</td><td>Occupied 0.92</td></tr></table><p class='good'><b>真正的改进：</b>不是简单把Unknown改成Occupied，而是补齐了“FOV路由 → LiDAR取证 → 模型融合 → 合同验收”这条证据链。</p></section>
<section><h2>4. Agent思考过程记录在哪里</h2><div class='note'><b>不会记录或展示隐藏思维链。</b>那类内部token既不可稳定验证，也不应被当作实验依据。当前记录的是可复核的决策摘要：谁做了决定、为什么选择该路由、工具返回什么、分数如何变化、引用了哪些证据、状态机为什么接受。</div><p><b>本Case要特别区分：</b>LiDAR不是模型临时“想到”后调用的，而是FOV不可见触发的确定性工作流动作；模型真正负责的是读取FOV与LiDAR证据并输出最终判断。</p><table><tr><th>记录内容</th><th>本Case是否存在</th><th>位置</th></tr><tr><td>Part1状态和Unknown原因</td><td>有</td><td>SlotCase / decision trace</td></tr><tr><td>FOV路由原因</td><td>有</td><td>check_fov evidence</td></tr><tr><td>工具调用原因</td><td>有（系统规则）</td><td>decision trace event 2 / ReasoningRound.reasoning_summary</td></tr><tr><td>工具Observation</td><td>有</td><td>EvidenceRecord.summary + metadata +图片</td></tr><tr><td>模型最终理由和分数</td><td>有</td><td>OpenAI audit action.reason</td></tr><tr><td>引用证据ID</td><td>有</td><td>agent_final metadata</td></tr><tr><td>隐藏Chain-of-Thought</td><td>没有</td><td>不记录</td></tr></table><p>机器可读记录：<a href='{_esc(_rel(trace_path,destination))}'>{_esc(trace_path.name)}</a></p></section>
<section><h2>5. 原始证据文件</h2><p>Part1原始地图：</p><img class='wide' src='{_esc(_rel(annotated_map,destination))}'><p>FOV覆盖地图：</p><img class='wide' src='{_esc(_rel(fov_map,destination))}'><p>OpenAI调用审计：<code>{_esc(audit_files[0] if audit_files else 'missing')}</code></p></section>
<section><h2>6. 结论边界</h2><p class='note'>这个Case证明：新流程能把原来的冲突Unknown补充为“有证据、合同合法”的Occupied结论。但没有人工GT，因此仍不能说slot_1012在现实中一定被占用。下一步需要你对原始图和LiDAR可视化进行人工确认，形成GT。</p></section>
</main></body></html>"""
    report_path.write_text(page, encoding="utf-8")
    return {
        "html_report": str(report_path),
        "decision_trace": str(trace_path),
        "transition_visual": str(transition),
        "annotated_part1_map": str(annotated_map),
        "slot_id": slot_id,
    }


__all__ = ["build_single_case_report"]
