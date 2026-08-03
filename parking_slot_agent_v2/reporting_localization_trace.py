"""Visualize one audited Map-to-Camera localization trajectory.

This report makes the Agent the subject of the figure.  It uses only structured
``rationale``, ``localization`` and final-action fields saved by the OpenAI audit;
it never claims to expose private hidden chain-of-thought.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .contracts import Part1Output
from .tools import V2ToolSuite


STATE_ZH = {"free": "空闲", "occupied": "占用", "unknown": "未知"}
STAGE_ZH = {
    "not_attempted": "尚未定位",
    "hypothesis": "定位假设",
    "supported": "假设获支持",
    "refuted": "假设被反证",
    "ambiguous": "验证后仍有歧义",
    "not_visible": "目标不可见",
}
SIDE_ZH = {"left": "左侧", "center": "中央", "right": "右侧", "unknown": "未知"}
TOOL_ZH = {
    "lidar_detail": "LiDAR细查",
    "camera_context": "地图+相机全景",
    "camera_crop": "候选区域Crop",
    "camera_sequence": "因果相机序列",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/home/ParkingAgent/DASP/simulation_carla/CARLA_0.9.16/Engine/Content/Slate/Fonts/DroidSansFallback.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int, max_lines: int) -> list[str]:
    words = str(text).replace("\n", " ").split()
    if not words:
        return ["—"]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if current and draw.textlength(candidate, font=font) > width:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                break
        else:
            current = candidate
    if len(lines) < max_lines and current:
        lines.append(current)
    if len(lines) == max_lines and sum(len(line.split()) for line in lines) < len(words):
        lines[-1] = lines[-1].rstrip("…") + "…"
    return lines


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: Sequence[str],
    font: ImageFont.ImageFont,
    fill: str,
    spacing: int = 8,
) -> int:
    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag示", font=font)[3] + spacing
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _copy(source: Path, output: Path, name: str) -> Path:
    target = output / "assets" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _audits(run_dir: Path, case_id: str) -> list[dict[str, Any]]:
    directory = run_dir / "openai_audit" / case_id
    return [_read(path) for path in sorted(directory.glob("turn_*.json"))]


def _regenerate_semantic_map(
    part1_path: Path,
    output_dir: Path,
    slot_id: str,
) -> Path:
    part1 = Part1Output.from_dict(_read(part1_path))
    case = next(item for item in part1.slot_cases if item.slot_id == slot_id)
    tools = V2ToolSuite(map_radius_m=part1.scene.radius_m)
    tools.check_fov(part1.scene, case, output_dir / "generated")
    evidence = tools.inspect_camera_context(
        part1.scene,
        case,
        output_dir / "generated",
        round_index=2,
    )
    model_paths = evidence.metadata.get("model_image_paths", [])
    if evidence.status != "ok" or not model_paths:
        raise ValueError("semantic Camera context could not be regenerated")
    return Path(str(model_paths[0]))


def _trace_payload(
    run_dir: Path,
    part1_path: Path,
    output_dir: Path,
    slot_id: str,
) -> dict[str, Any]:
    case = _read(run_dir / "slot_cases" / f"{slot_id}.json")
    audits = _audits(run_dir, str(case["case_id"]))
    actions = [dict(row.get("action", {})) for row in audits]
    if not actions:
        raise ValueError("no audited Agent actions found")

    semantic_map = _regenerate_semantic_map(part1_path, output_dir, slot_id)
    crop_dir = run_dir / "media" / slot_id / "camera_crop"
    overlay = next(crop_dir.glob("*_hypothesis_on_full_camera.png"))
    crop = next(path for path in crop_dir.glob("*.png") if "_hypothesis_" not in path.name)
    fov = next((run_dir / "media" / slot_id / "fov").glob("*.png"))
    assets = {
        "fov": _copy(fov, output_dir, f"{slot_id}_01_fov.png"),
        "semantic_map": _copy(semantic_map, output_dir, f"{slot_id}_02_semantic_map.png"),
        "hypothesis_overlay": _copy(overlay, output_dir, f"{slot_id}_03_hypothesis_overlay.png"),
        "crop": _copy(crop, output_dir, f"{slot_id}_04_crop.png"),
    }

    turns: list[dict[str, Any]] = []
    for index, action in enumerate(actions, start=1):
        localization = dict(action.get("localization", {}))
        if action.get("type") == "tool":
            decision = f"调用 {TOOL_ZH.get(str(action.get('tool')), str(action.get('tool')))}"
            rationale = str(action.get("rationale", ""))
        else:
            decision = f"提交 {STATE_ZH.get(str(action.get('state')), str(action.get('state')))}"
            rationale = str(action.get("reason", ""))
        turns.append({
            "turn": index,
            "decision": decision,
            "rationale": rationale,
            "state": str((action.get("belief") or {}).get("state", action.get("state", "unknown"))),
            "localization": localization,
            "reason_codes": list(action.get("reason_codes", [])),
            "raw_action": action,
        })

    hypotheses = [
        dict(action.get("localization", {}))
        for action in actions
        if action.get("localization", {}).get("stage") != "not_attempted"
    ]
    initial_loc = hypotheses[0] if hypotheses else dict(actions[0].get("localization", {}))
    final = actions[-1]
    final_loc = dict(final.get("localization", {}))
    initial_side = SIDE_ZH.get(str(initial_loc.get("target_side", "unknown")), "未知")
    final_side = SIDE_ZH.get(str(final_loc.get("target_side", "unknown")), "未知")
    initial_confidence = float(initial_loc.get("confidence_after", 0.0))
    final_confidence = float(final_loc.get("confidence_after", 0.0))
    final_stage = STAGE_ZH.get(str(final_loc.get("stage", "not_attempted")), "尚未定位")
    matched = [str(value).replace("_", " ") for value in final_loc.get("matched_landmarks", [])]
    missing = [str(value).replace("_", " ") for value in final_loc.get("missing_landmarks", [])]
    direction_change = (
        f"{initial_side} → {final_side}"
        if initial_side != final_side else f"保持{final_side}"
    )
    final_state = str(case.get("final_state", case.get("current_state", "unknown")))
    return {
        "schema_version": "parking-agent-localization-trace/1.0",
        "slot_id": slot_id,
        "case_id": case["case_id"],
        "part1_state": case.get("part1_state", "unknown"),
        "final_state": final_state,
        "part1_scores": case.get("part1_scores", {}),
        "final_scores": case.get("final_scores", case.get("current_scores", {})),
        "fov": case.get("fov", {}),
        "turns": turns,
        "initial_localization": initial_loc,
        "final_localization": final_loc,
        "summary": {
            "direction_change": direction_change,
            "initial_confidence": initial_confidence,
            "final_confidence": final_confidence,
            "final_stage": final_stage,
            "matched_landmarks": matched,
            "missing_landmarks": missing,
        },
        "assets": {key: f"assets/{path.name}" for key, path in assets.items()},
        "findings": [
            "FOV只确认目标在保守视野内或边缘，不声称给出像素位置。",
            f"Agent的定位方位变化为{direction_change}，定位置信度从{initial_confidence:.2f}变为{final_confidence:.2f}。",
            "Agent用结构化地标构造可证伪假设：" + ("、".join(matched) if matched else "没有形成可靠地标匹配" ) + "。",
            f"验证后定位阶段为“{final_stage}”；" + ("仍缺少" + "、".join(missing[:3]) + "，" if missing else "") + f"最终状态为{STATE_ZH.get(final_state, final_state)}。",
        ],
    }


def _card_summary(turn: Mapping[str, Any]) -> str:
    action = turn.get("raw_action", {})
    localization = turn.get("localization", {})
    tool = action.get("tool")
    if tool == "lidar_detail":
        return "Agent先获取目标局部多帧几何，检查Free/Occupied是否已有直接证据。"
    if tool == "camera_context":
        return "几何仍不足，Agent主动请求语义地图和完整相机，准备建立可核查的像素区域假设。"
    if tool == "camera_crop":
        side = SIDE_ZH.get(str(localization.get("target_side", "unknown")), "未知")
        return f"Agent根据地图与地标提出{side}候选区域，并调用同一帧Crop尝试支持或反证。"
    if tool == "camera_sequence":
        return "Agent调用严格早于t0的相机序列，检查同一车位行与地标关系能否随运动保持一致。"
    stage = STAGE_ZH.get(str(localization.get("stage", "not_attempted")), "尚未定位")
    return f"工具预算结束；Agent把定位验证记为“{stage}”，再提交可审计的最终状态。"


def _fit(path: Path, box: tuple[int, int, int, int]) -> tuple[Image.Image, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = box
    image.thumbnail((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
    return image, (x1 + (x2 - x1 - image.width) // 2, y1 + (y2 - y1 - image.height) // 2)


def _render_board(payload: Mapping[str, Any], output_dir: Path) -> Path:
    width, height = 2400, 1760
    canvas = Image.new("RGB", (width, height), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    title, h2, body, small, tiny = _font(44, bold=True), _font(27, bold=True), _font(20), _font(17), _font(14)
    ink, muted, blue, purple, amber, green = "#132238", "#63758b", "#2472d8", "#7a53c6", "#e6a018", "#15936f"

    draw.text((48, 32), f"{payload['slot_id']} · Agent主动定位与验证轨迹", font=title, fill=ink)
    draw.text((48, 91), "真实OpenAI结构化审计 · 不是隐藏思维链 · frame 9277", font=small, fill=muted)
    draw.rounded_rectangle((1910, 37, 2350, 102), radius=30, fill=amber)
    transition = f"{str(payload['part1_state']).upper()} → {str(payload['final_state']).upper()}"
    draw.text((1950, 52), transition, font=h2, fill="white")

    # Outcome strip.
    draw.rounded_rectangle((48, 130, 2350, 260), radius=18, fill="white", outline="#d8e1ec", width=2)
    draw.text((74, 153), "本次工具带来的真实变化", font=h2, fill=ink)
    summary = payload["summary"]
    draw.text((74, 199), f"左右方位：{summary['direction_change']}", font=body, fill=purple)
    draw.text((760, 199), f"定位置信度：{summary['initial_confidence']:.2f} → {summary['final_confidence']:.2f}", font=body, fill=blue)
    draw.text((1320, 199), f"最终状态：{STATE_ZH.get(payload['final_state'], payload['final_state'])} · {summary['final_stage']}", font=body, fill=amber)

    # Four Agent turns.
    card_y1, card_y2, gap = 300, 870, 22
    card_w = (2302 - 3 * gap) // 4
    for idx, turn in enumerate(payload["turns"]):
        x1 = 48 + idx * (card_w + gap); x2 = x1 + card_w
        loc = turn.get("localization", {})
        draw.rounded_rectangle((x1, card_y1, x2, card_y2), radius=18, fill="#111f33", outline="#29415e", width=2)
        draw.rounded_rectangle((x1, card_y1, x2, card_y1 + 64), radius=18, fill="#1d3452")
        draw.rectangle((x1, card_y1 + 44, x2, card_y1 + 64), fill="#1d3452")
        draw.text((x1 + 22, card_y1 + 18), f"OPENAI TURN {turn['turn']}", font=small, fill="white")
        draw.text((x2 - 155, card_y1 + 18), f"belief: {STATE_ZH.get(turn['state'], turn['state'])}", font=tiny, fill="#f4b228")
        y = card_y1 + 88
        draw.text((x1 + 22, y), "可审计决策摘要", font=tiny, fill="#55a8ff"); y += 27
        y = _draw_lines(draw, (x1 + 22, y), _wrap(draw, _card_summary(turn), body, card_w - 44, 4), body, "#e8eef6") + 18
        draw.text((x1 + 22, y), "结构化定位状态", font=tiny, fill="#55a8ff"); y += 28
        stage = STAGE_ZH.get(str(loc.get("stage", "not_attempted")), str(loc.get("stage", "—")))
        side = SIDE_ZH.get(str(loc.get("target_side", "unknown")), str(loc.get("target_side", "—")))
        conf = float(loc.get("confidence_after", 0.0))
        details = [
            f"阶段：{stage}", f"方位：{side} · 深度：{loc.get('depth_band', '—')}",
            f"定位置信度：{conf:.2f}", f"假设：{loc.get('hypothesis_id') or '—'}",
        ]
        y = _draw_lines(draw, (x1 + 22, y), details, small, "#c9d5e4", 7) + 16
        draw.text((x1 + 22, y), "Agent Action", font=tiny, fill="#55a8ff"); y += 28
        draw.rounded_rectangle((x1 + 20, y, x2 - 20, y + 70), radius=12, fill="#f4ad23")
        draw.text((x1 + 42, y + 20), f"→ {turn['decision']}", font=h2, fill="#101b2c")
        y += 92
        matched = [str(item).replace("_", " ") for item in loc.get("matched_landmarks", [])]
        if matched:
            draw.text((x1 + 22, y), "依据的可见锚点", font=tiny, fill="#55a8ff"); y += 25
            y = _draw_lines(draw, (x1 + 22, y), _wrap(draw, " · ".join(matched), tiny, card_w - 44, 5), tiny, "#aabbd0", 5)

    # Evidence row.
    draw.text((48, 910), "工具执行后到底看到了什么", font=h2, fill=ink)
    evidence = [
        ("semantic_map", "① 语义地图", f"目标{str(payload['slot_id']).replace('slot_', '')}的地图方位；正方位角=相机左侧。"),
        ("hypothesis_overlay", "② Agent提出Crop", "紫框是Agent假设，不是Ground Truth投影。"),
        ("crop", "③ Crop验证", "能看到左侧柱/墙走廊，但看不清目标车位边界。"),
    ]
    boxes = [(48, 970, 800, 1550), (824, 970, 1576, 1550), (1600, 970, 2350, 1550)]
    for (key, label, caption), box in zip(evidence, boxes):
        x1, y1, x2, y2 = box
        draw.rounded_rectangle(box, radius=18, fill="white", outline="#d8e1ec", width=2)
        draw.text((x1 + 20, y1 + 18), label, font=h2, fill=ink)
        fitted, position = _fit(output_dir / str(payload["assets"][key]), (x1 + 18, y1 + 66, x2 - 18, y2 - 88))
        canvas.paste(fitted, position)
        _draw_lines(draw, (x1 + 20, y2 - 70), _wrap(draw, caption, small, x2 - x1 - 40, 2), small, muted, 4)

    draw.rounded_rectangle((48, 1580, 2350, 1718), radius=18, fill="#fff8e8", outline="#e9c76c", width=2)
    draw.text((74, 1602), "最终结论", font=h2, fill=ink)
    final_sentence = f"定位变化：{summary['direction_change']}；验证阶段：{summary['final_stage']}；最终状态：{STATE_ZH.get(payload['final_state'], payload['final_state'])}。"
    draw.text((74, 1647), final_sentence, font=body, fill=ink)
    draw.text((74, 1682), "定位信息增益与占用状态消歧是两件事；报告不会把未达到硬门的定位置信度包装成终态提升。", font=small, fill=amber)
    output = output_dir / "agent_localization_trace.png"
    canvas.save(output, optimize=True)
    return output


def _document(payload: Mapping[str, Any]) -> str:
    turns = "".join(
        f"<article><h3>Turn {turn['turn']} · {html.escape(turn['decision'])}</h3>"
        f"<p>{html.escape(_card_summary(turn))}</p>"
        f"<pre>{html.escape(json.dumps(turn['raw_action'], ensure_ascii=False, indent=2))}</pre></article>"
        for turn in payload["turns"]
    )
    findings = "".join(f"<li>{html.escape(item)}</li>" for item in payload["findings"])
    assets = payload["assets"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(payload['slot_id'])} 主动定位报告</title><style>
body{{margin:0;background:#f3f6fa;color:#132238;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin-bottom:4px}}.lead{{color:#63758b}}.hero,.panel,article{{background:#fff;border:1px solid #d8e1ec;border-radius:16px;padding:18px;margin:15px 0}}.hero img{{width:100%;display:block}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric{{background:#fff;border:1px solid #d8e1ec;border-radius:14px;padding:16px}}.metric b{{display:block;font-size:24px}}.evidence{{display:grid;grid-template-columns:1fr 1.4fr 1fr;gap:12px}}.evidence img{{width:100%;height:390px;object-fit:contain;background:#0f1b2b}}.turns{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}pre{{white-space:pre-wrap;max-height:360px;overflow:auto;background:#101b2c;color:#d8e5f3;padding:12px;border-radius:9px;font-size:12px}}.warning{{border-left:5px solid #e6a018;background:#fff8e8;padding:13px}}@media(max-width:900px){{.metrics,.evidence,.turns{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>{html.escape(payload['slot_id'])}：Agent如何主动定位并验证车位</h1><p class="lead">真实OpenAI结构化审计。展示的是可审计rationale、定位假设和工具动作，不是模型隐藏思维链。</p>
<section class="hero"><img src="agent_localization_trace.png" alt="Agent主动定位轨迹"></section>
<section class="metrics"><div class="metric"><span>Part1 → Part2</span><b>{STATE_ZH.get(payload['part1_state'], payload['part1_state'])} → {STATE_ZH.get(payload['final_state'], payload['final_state'])}</b></div><div class="metric"><span>定位置信度</span><b>{payload['summary']['initial_confidence']:.2f} → {payload['summary']['final_confidence']:.2f}</b></div><div class="metric"><span>方位假设</span><b>{html.escape(payload['summary']['direction_change'])}</b></div></section>
<section class="panel"><h2>四个问题的直接答案</h2><ol>{findings}</ol><p class="warning"><b>没有伪造提升：</b>本例的提升是“方位假设被纠正且有审计轨迹”，不是占用状态被解决。Crop后目标车位线仍不可辨，所以Unknown是正确的安全输出。</p></section>
<section class="evidence"><figure><img src="{assets['semantic_map']}"><figcaption>语义地图：目标在左侧远端</figcaption></figure><figure><img src="{assets['hypothesis_overlay']}"><figcaption>紫框：Agent提出的像素区域假设，不是Ground Truth</figcaption></figure><figure><img src="{assets['crop']}"><figcaption>Crop：同一帧重新检查候选区域</figcaption></figure></section>
<h2>逐轮结构化审计</h2><section class="turns">{turns}</section>
</main></body></html>"""


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload['slot_id']} 主动定位验证报告", "",
        "## 结论", "",
        f"Agent按结构化协议完成定位假设与工具验证。方位变化为{payload['summary']['direction_change']}，验证阶段为{payload['summary']['final_stage']}。", "",
        f"- 定位置信度：{payload['summary']['initial_confidence']:.2f} → {payload['summary']['final_confidence']:.2f}。", f"- 状态：{payload['part1_state']} → {payload['final_state']}。", "- 解释：定位信息增益与占用状态消歧分开报告。", "",
        "## 四个问题", "",
    ]
    lines.extend(f"{index}. {item}" for index, item in enumerate(payload["findings"], start=1))
    lines.extend(["", "## Agent轨迹", ""])
    for turn in payload["turns"]:
        loc = turn.get("localization", {})
        lines.extend([
            f"### Turn {turn['turn']}：{turn['decision']}", "",
            _card_summary(turn), "",
            f"- 定位阶段：{STAGE_ZH.get(str(loc.get('stage')), str(loc.get('stage')))}",
            f"- 方位：{SIDE_ZH.get(str(loc.get('target_side')), str(loc.get('target_side')))}",
            f"- 定位置信度：{float(loc.get('confidence_after', 0)):.2f}",
            f"- 可审计原始理由：{turn['rationale']}", "",
        ])
    lines.extend([
        "## 文件", "",
        "- `agent_localization_trace.png`：一张图看完整决策链。",
        "- `index.html`：逐轮原始结构化action。",
        "- `trace.json`：机器可读轨迹。",
        "- `assets/`：FOV、语义地图、Agent假设框和Crop。", "",
        "说明：报告不展示或伪造不可审计的隐藏思维链。", "",
    ])
    return "\n".join(lines)


def build_localization_trace_report(
    *,
    run_dir: str | Path,
    part1_path: str | Path,
    output_dir: str | Path,
    slot_id: str,
) -> dict[str, str]:
    run = Path(run_dir).resolve()
    part1 = Path(part1_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = _trace_payload(run, part1, output, slot_id)
    board = _render_board(payload, output)
    html_path = output / "index.html"
    html_path.write_text(_document(payload), encoding="utf-8")
    json_path = output / "trace.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = output / "REPORT.md"
    report_path.write_text(_markdown(payload), encoding="utf-8")
    return {
        "board": str(board),
        "html": str(html_path),
        "report": str(report_path),
        "trace": str(json_path),
    }


__all__ = ["build_localization_trace_report"]
