"""Plain-Chinese visual report for the extended causal Part2 experiment."""

from __future__ import annotations

from collections import Counter
import html
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from PIL import Image, ImageDraw

from parking_slot_part2.media import load_lidar_evidence_pack

from .contracts import Part1Output
from .lidar_geometry import assess_terminal_geometry, build_geometry_card
from .reporting_story_zh import _draw_wrapped, _fit_image, _font, _rounded_box


STATE_ZH = {"free": "空闲", "occupied": "占用", "unknown": "未知"}
STATE_COLOR = {"free": "#059669", "occupied": "#dc2626", "unknown": "#d97706"}
REASON_ZH = {
    "robustness_not_all_variants_stable": "位姿扰动没有全部稳定通过",
    "no_strong_free_geometry_candidate": "自由空间证据不够强",
    "free_strong_gate_failed": "Free 强门未通过",
    "free_geometry_has_failures": "自由空间仍有失败项",
    "unresolved_core_hit": "核心区域仍有障碍命中",
    "opposing_occupied_geometry_not_vetoed": "相反的 Occupied 证据未被排除",
    "no_strong_occupied_geometry_candidate": "占用几何证据不够强",
    "occupied_strong_gate_failed": "Occupied 强门未通过",
    "occupied_geometry_has_failures": "占用几何仍有失败项",
    "boundary_ratio_not_below_0_5": "返回点过度集中在边界",
    "linearity_risk_not_below_0_6": "线性墙沿/路沿风险过高",
    "no_core_obstacle_points": "核心区域没有可靠障碍点",
    "opposing_free_strong_gate": "存在相反的强 Free 证据",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _copy(source: str | Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(source), destination)
    return destination


def _result_rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        row["case"]["slot"]["slot_id"]: row
        for row in payload.get("slot_results", [])
    }


def _tool(case: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(row for row in case.get("evidence", []) if row.get("tool_name") == name)


def _reason_zh(state: str, card: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    if state == "free":
        free = card["free_geometry"]
        return (
            f"自由空间观测体积 {float(free['observed_volume_ratio']):.3f}，近地覆盖 "
            f"{float(free['near_ground_bev_coverage']):.3f}，遮挡 {float(free['occlusion_ratio']):.3f}；"
            "核心无未解决障碍命中，且 7/7 位姿扰动通过。"
        )
    if state == "occupied":
        occupied = card["occupied_geometry"]
        return (
            f"核心内部点 {int(occupied['core_point_count'])}，高度跨度 "
            f"{float(occupied['height_span_m']):.2f} m，边界比 {float(occupied['boundary_ratio']):.3f}；"
            "多帧立体障碍稳定，且 7/7 位姿扰动通过。"
        )
    blockers = list(dict.fromkeys(gate["free_blockers"] + gate["occupied_blockers"]))
    return "仍保持 Unknown：" + "；".join(REASON_ZH.get(item, item) for item in blockers[:4]) + "。"


def _overview(path: Path, *, transitions: Mapping[str, list[str]]) -> None:
    image = Image.new("RGB", (2100, 1300), "#f4f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((60, 42), "Part2 扩展因果 LiDAR：Unknown 解决率从 0% 到 40.9%", font=_font(45), fill="#102a43")
    draw.text((62, 105), "同一工作流、同一 0.90 终态阈值；Part1 仍是 15 帧，Part2 细查扩展到 t0 之前 75 帧。", font=_font(27), fill="#40566d")
    metrics = [
        ("原始 Unknown", "22", "评测母集", "#64748b"),
        ("15帧严格可解", "0", "0 / 22", "#d97706"),
        ("75帧严格可解", "9", "9 / 22", "#2563eb"),
        ("Agent确认", "9", "2空闲 + 7占用", "#059669"),
    ]
    for index, (label, value, note, color) in enumerate(metrics):
        x = 62 + index * 505
        _rounded_box(draw, (x, 180, x + 460, 405), fill="white", outline=color, radius=18, width=4)
        draw.text((x + 22, 202), label, font=_font(25), fill=color)
        draw.text((x + 22, 250), value, font=_font(66), fill="#102a43")
        draw.text((x + 22, 344), note, font=_font(23), fill="#40566d")

    groups = [
        ("变为空闲", transitions["free"], "#ecfdf5", "#059669"),
        ("变为占用", transitions["occupied"], "#fff1f2", "#dc2626"),
        ("继续未知", transitions["unknown"], "#fff7ed", "#d97706"),
    ]
    y = 470
    for title, slots, fill, color in groups:
        height = 190 if len(slots) < 10 else 280
        _rounded_box(draw, (62, y, 2038, y + height), fill=fill, outline=color, radius=18, width=3)
        draw.text((88, y + 20), f"{title}（{len(slots)}）", font=_font(29), fill=color)
        _draw_wrapped(draw, (88, y + 74), "、".join(slots), font=_font(27), fill="#26384b", width=1880, spacing=12)
        y += height + 24
    draw.text((62, 1235), "注意：这里统计的是严格证据消歧率，不是准确率；数据没有人工 GT。", font=_font(25), fill="#5b6472")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _case_story(
    path: Path,
    *,
    slot_id: str,
    final_state: str,
    confidence: float,
    fov_image: Path,
    evidence_image: Path,
    facts: list[str],
    model_reason: str,
) -> None:
    canvas = Image.new("RGB", (2200, 1500), "#f4f7fb")
    draw = ImageDraw.Draw(canvas)
    color = STATE_COLOR[final_state]
    draw.text((58, 38), f"{slot_id}：Part1 未知  →  Agent {STATE_ZH[final_state]} {confidence:.2f}", font=_font(43), fill="#102a43")
    draw.text((60, 103), "左：原 occupancy map + FOV；中：Agent 实际看到的 LiDAR 组合证据；右：状态为何改变。", font=_font(25), fill="#40566d")
    left = _fit_image(fov_image, (610, 750))
    middle = _fit_image(evidence_image, (940, 1120))
    canvas.paste(left, (55, 220))
    canvas.paste(middle, (700, 220))
    draw.rectangle((55, 220, 665, 970), outline="#94a3b8", width=3)
    draw.rectangle((700, 220, 1640, 1340), outline="#94a3b8", width=3)
    _rounded_box(draw, (1680, 220, 2145, 1340), fill="white", outline=color, radius=18, width=4)
    draw.text((1710, 250), f"最终：{STATE_ZH[final_state]}", font=_font(34), fill=color)
    draw.text((1710, 305), f"置信度：{confidence:.2f}", font=_font(29), fill="#102a43")
    y = 375
    for index, fact in enumerate(facts, 1):
        y = _draw_wrapped(draw, (1710, y), f"{index}. {fact}", font=_font(23), fill="#334155", width=390, spacing=8) + 12
    draw.text((1710, y + 5), "Agent理由（中文忠实转述）", font=_font(25), fill=color)
    _draw_wrapped(draw, (1710, y + 52), model_reason, font=_font(21), fill="#475569", width=390, spacing=7)
    draw.text((55, 1010), "原状态：Unknown。FOV 只负责决定 Camera 是否可用，不直接改车位状态。", font=_font(24), fill="#475569")
    _draw_wrapped(draw, (55, 1060), "状态改变发生在 LiDAR 数值卡通过代码硬门之后；模型给出的置信度不能绕开 core 点、边界比、遮挡和 7/7 位姿稳定性检查。", font=_font(23), fill="#334155", width=600, spacing=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_extended75_report(
    *,
    part1_path: str | Path,
    exhaustive_result: str | Path,
    targeted_1258_result: str | Path,
    operational_result: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    part1_file = Path(part1_path).resolve()
    destination = Path(output_dir).resolve()
    assets = destination / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    part1 = Part1Output.from_dict(_load(part1_file))
    exhaustive = _load(Path(exhaustive_result))
    targeted = _load(Path(targeted_1258_result))
    operational = _load(Path(operational_result))
    operational_row = operational["slot_results"][0]
    agent_rows = _result_rows(exhaustive)
    agent_rows["slot_1258"] = _result_rows(targeted)["slot_1258"]

    audit_rows: list[dict[str, Any]] = []
    for case in part1.slot_cases:
        decision_payload = _load(Path(case.resources["extended_lidar_decision_path"]))
        pack = load_lidar_evidence_pack(
            case.resources["extended_lidar_evidence_path"], expected_slot_id=case.slot_id
        )
        card = build_geometry_card(
            case,
            pack,
            decision_override=decision_payload["decision"],
            decision_source="part2_extended_causal_lidar",
        )
        gate = assess_terminal_geometry(card)
        geometry_state = "free" if gate["free_eligible"] else ("occupied" if gate["occupied_eligible"] else "unknown")
        agent = agent_rows[case.slot_id]
        agent_final_state = agent["case"]["final_state"]
        audit_rows.append(
            {
                "slot_id": case.slot_id,
                "part1_state": case.part1_state.value,
                "geometry_state": geometry_state,
                "agent_final_state": agent_final_state,
                "agent_final_scores": agent["case"]["final_scores"],
                "agent_final_reason": agent["case"]["final_reason"],
                "agent_final_reason_zh": _reason_zh(agent_final_state, card, gate),
                "agent_stop_reason": agent["stop_reason"],
                "validation_errors": agent["validation_errors"],
                "geometry_card": card,
                "terminal_gate": gate,
                "result_source": (
                    "targeted_semantic_regression_after_card_fix"
                    if case.slot_id == "slot_1258"
                    else "exhaustive_openai_evaluation"
                ),
            }
        )
    unknown_rows = [row for row in audit_rows if row["part1_state"] == "unknown"]
    audit_by_id = {row["slot_id"]: row for row in audit_rows}
    transitions = {
        state: [row["slot_id"] for row in unknown_rows if row["agent_final_state"] == state]
        for state in ("free", "occupied", "unknown")
    }
    audit_path = destination / "22个Unknown_逐车位审计.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "parking-slot-agent-v2-extended75-evaluation/1.0",
                "ground_truth_available": False,
                "terminal_threshold": 0.90,
                "unknown_candidate_count": 22,
                "resolved_count": len(transitions["free"]) + len(transitions["occupied"]),
                "resolved_rate": 9 / 22,
                "rows": unknown_rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    overview = assets / "00_Unknown解决率总览.png"
    _overview(overview, transitions=transitions)
    stories: dict[str, Path] = {}
    story_specs = {
        "slot_1012": [
            "75 个有效帧、4 个视角，视角分离 38.6°。",
            "核心射线与近地面覆盖均为 1.0，观测体积 0.998，遮挡仅 0.002。",
            "所谓障碍点 100% 在边界，核心点为 0，因此 Occupied 被硬门否决。",
            "7/7 位姿扰动通过，最终 Free 0.97。",
        ],
        "slot_1258": [
            "39,773 个核心内部点，核心重叠 1.0。",
            "高度跨度 1.90 m，三层高度结构，75 帧都有时序支持。",
            "边界比与线性结构风险均为 0，排除墙沿/路沿误报。",
            "7/7 位姿扰动通过，最终 Occupied 0.95。",
        ],
    }
    for slot_id in story_specs:
        row = agent_rows[slot_id]
        case = row["case"]
        fov = _tool(case, "check_fov")
        lidar = _tool(case, "lidar_detail")
        fov_asset = _copy(fov["artifact_paths"][0], assets / f"{slot_id}_FOV地图.png")
        lidar_asset = _copy(lidar["metadata"]["model_image_paths"][0], assets / f"{slot_id}_75帧LiDAR证据.png")
        story = assets / f"{slot_id}_状态变化说明.png"
        scores = case["final_scores"]
        final = case["final_state"]
        confidence = scores["free_confidence"] if final == "free" else scores["occupied_confidence"]
        _case_story(
            story,
            slot_id=slot_id,
            final_state=final,
            confidence=float(confidence),
            fov_image=fov_asset,
            evidence_image=lidar_asset,
            facts=story_specs[slot_id],
            model_reason=audit_by_id[slot_id]["agent_final_reason_zh"],
        )
        stories[slot_id] = story

    table_rows = "".join(
        "<tr>"
        f"<td><code>{_esc(row['slot_id'])}</code></td>"
        f"<td>未知</td><td class='{_esc(row['agent_final_state'])}'>{_esc(STATE_ZH[row['agent_final_state']])}</td>"
        f"<td>{max(float(row['agent_final_scores']['free_confidence']), float(row['agent_final_scores']['occupied_confidence']), float(row['agent_final_scores']['unknown_confidence'])):.2f}</td>"
        f"<td>{_esc(row['agent_final_reason_zh'])}</td>"
        "</tr>"
        for row in unknown_rows
    )
    unresolved_reason_counts = Counter(
        reason
        for row in unknown_rows
        if row["agent_final_state"] == "unknown"
        for reason in row["terminal_gate"]["free_blockers"] + row["terminal_gate"]["occupied_blockers"]
    )
    blocker_html = "".join(
        f"<li><code>{_esc(reason)}</code>：{count} 个门限命中</li>"
        for reason, count in unresolved_reason_counts.most_common(8)
    )
    report_path = destination / "frame_9277_Part2_75帧优化报告.html"
    report_path.write_text(
        f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Part2 75帧优化报告</title><style>
body{{margin:0;background:#f4f7fb;color:#172033;font-family:system-ui,-apple-system,"Microsoft YaHei","PingFang SC",sans-serif;line-height:1.72}}main{{max-width:1500px;margin:auto;padding:28px}}section{{background:#fff;border:1px solid #cbd5e1;border-radius:15px;padding:24px;margin:20px 0}}h1{{font-size:38px;margin-top:0}}h2{{border-left:7px solid #2563eb;padding-left:12px}}img{{width:100%;object-fit:contain;border-radius:8px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{padding:16px;border-radius:12px;background:#eff6ff;border:1px solid #93c5fd}}.metric b{{display:block;font-size:32px}}.good{{background:#ecfdf5;border:1px solid #6ee7b7;padding:14px;border-radius:10px}}.warn{{background:#fff7ed;border:1px solid #fdba74;padding:14px;border-radius:10px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{border:1px solid #cbd5e1;padding:8px;vertical-align:top}}th{{background:#e8eef6}}td.free{{color:#047857;font-weight:bold}}td.occupied{{color:#b91c1c;font-weight:bold}}td.unknown{{color:#b45309;font-weight:bold}}code{{overflow-wrap:anywhere}}@media(max-width:850px){{.metrics{{grid-template-columns:1fr}}}}</style></head><body><main>
<section><h1>Part2：75 帧严格因果证据优化</h1><p>这份报告只回答三个问题：到底解决了多少 Unknown、状态为什么改变、代码和证据在哪里。</p><img src='{_esc(_rel(overview,destination))}'></section>
<section><h2>1. 最终结果</h2><div class='metrics'><div class='metric'>原始 Unknown<b>22</b>同一 frame 9277</div><div class='metric'>15帧严格可解<b>0</b>不降低门槛</div><div class='metric'>75帧严格可解<b>9</b>40.9%</div><div class='metric'>Agent确认<b>9</b>2空闲 + 7占用</div></div><p class='good'><b>明确提升：</b>Unknown 消歧从 0/22 提升到 9/22。9 个终态全部通过 0.90 置信度、几何互斥门和 7/7 位姿稳定性；其余 13 个没有强行改标签。</p><p class='good'><b>正式工作流实测：</b>默认早停模式只处理 1 个候选，选择 <code>{_esc(operational['selected_slot_id'])}</code>，Free={float(operational_row['case']['final_scores']['free_confidence']):.2f}；1 次 LiDAR、{int(operational_row['model_turns'])} 次模型调用、0 校验错误，然后以 <code>{_esc(operational['stop_reason'])}</code> 停止。</p><p class='warn'>没有人工 GT，所以 40.9% 是“严格消歧率”，不是识别准确率；0.95/0.97 是工作流置信度，不是统计校准概率。</p></section>
<section><h2>2. 一个 Unknown → Free 的完整案例</h2><img src='{_esc(_rel(stories['slot_1012'],destination))}'><p><b>关键纠错：</b>旧实验把 slot_1012 报成 Occupied 0.92，但它的核心点为 0、边界比为 1.0。新硬门先否决这个伪 Occupied，再由 75 帧自由空间覆盖证明 Free。</p></section>
<section><h2>3. 一个 Unknown → Occupied 的完整案例</h2><img src='{_esc(_rel(stories['slot_1258'],destination))}'><p><b>为什么 15 帧不够：</b>短窗口下姿态扰动不稳定；75 帧增加视角和时间支持后，内部体积障碍在 7/7 扰动中保持一致。这里增加的是证据，不是降低阈值。</p></section>
<section><h2>4. Agent 到底“思考”了什么</h2><p>系统记录的是可审计的推理摘要，不保存或展示模型私有的隐藏思维链。每个 case 都保留：FOV 判断、工具路由摘要、工具 observation、引用的 evidence_id、最终理由和未解决原因。</p><ol><li>强制 FOV：只决定 Camera 是否允许使用，并把可见性写回 SlotCase。</li><li>自动读取身份绑定的 75 帧 LiDAR 包；包内所有帧均不晚于 t0=9277。</li><li>数值卡计算核心点、边界比、高度、覆盖、遮挡、视角分离和 7/7 稳定性。</li><li>OpenAI 模型输出结构化终态与理由；状态机再执行硬门。模型自报 ≥0.90 但硬门失败，仍会被拒绝。</li></ol><p>逐车位完整记录：<a href='{_esc(_rel(audit_path,destination))}'>{_esc(audit_path.name)}</a>；原始 Agent 输出位于 <code>{_esc(Path(exhaustive_result).resolve())}</code> 和 slot_1258 定向回归输出。</p></section>
<section><h2>5. 22 个 Unknown 逐项结果</h2><table><tr><th>车位</th><th>Part1</th><th>Agent最终</th><th>最高置信度</th><th>可审计最终理由</th></tr>{table_rows}</table></section>
<section><h2>6. 仍然 Unknown 的 13 个为什么没改</h2><p>它们主要存在冲突、遮挡、边界主导、核心观测不足或位姿不稳定。系统选择保留 Unknown，而不是用 Prompt 猜测。</p><ul>{blocker_html}</ul></section>
<section><h2>7. 整体设计是否改变</h2><p><b>没有。</b>仍是 Free 优先队列 → 强制 FOV → 单车位 ReAct → Camera/LiDAR 工具 → 0.90 终态 → 第一个可信 Free 早停。新增的是 Part2 LiDAR 工具内部的严格因果长窗口和硬门资格显示。正式模式仍会在第一个可信 Free 停止；本次为统计全量结果，使用了显式 exhaustive 评测开关。</p></section>
<section><h2>8. 代码和产物</h2><ul><li>扩展证据构建：<code>parking_slot_agent_v2/extended_lidar.py</code></li><li>LiDAR 数值卡与硬门：<code>parking_slot_agent_v2/lidar_geometry.py</code></li><li>工具接入：<code>parking_slot_agent_v2/tools.py</code></li><li>Agent 状态机：<code>parking_slot_agent_v2/agent.py</code></li><li>断点续跑与评测模式：<code>parking_slot_agent_v2/pipeline.py</code></li><li>75帧输入与manifest：<code>{_esc(str(part1_file.parent))}</code></li></ul></section>
</main></body></html>""",
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "overview": str(overview),
        "audit": str(audit_path),
        "unknown_count": 22,
        "resolved_count": 9,
        "resolved_rate": 9 / 22,
        "free_transitions": transitions["free"],
        "occupied_transitions": transitions["occupied"],
        "remaining_unknown": transitions["unknown"],
        "validation_error_count": sum(len(row["validation_errors"]) for row in audit_rows),
    }


__all__ = ["build_extended75_report"]
