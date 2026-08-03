#!/usr/bin/env python3
"""Generate a one-page Chinese summary of the current Part1 15-frame algorithm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PART1_DIR = PROJECT_ROOT / "outputs" / "part1_local_lidar_frame_9277"
DEFAULT_OUTPUT_NAME = "part1_15frame_state_algorithm_summary.pdf"
FONT_NAME = "STSong-Light"
EMBEDDED_FONT_CANDIDATES = (
    Path(
        "/home/ParkingAgent/DASP/simulation_carla/CARLA_0.9.16/"
        "Engine/Content/Slate/Fonts/DroidSansFallback.ttf"
    ),
    Path(
        "/home/ParkingAgent/simulation_carla/CARLA_0.9.16/"
        "Engine/Content/Slate/Fonts/DroidSansFallback.ttf"
    ),
)


def _register_font() -> None:
    """Prefer an embedded CJK font; retain the built-in CID fallback."""

    global FONT_NAME
    for path in EMBEDDED_FONT_CANDIDATES:
        if path.is_file():
            FONT_NAME = "DroidSansFallback"
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(path)))
            return
    FONT_NAME = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fit_text(text: str, font_size: float, max_width: float) -> str:
    if pdfmetrics.stringWidth(text, FONT_NAME, font_size) <= max_width:
        return text
    suffix = "…"
    clipped = text
    while clipped and pdfmetrics.stringWidth(clipped + suffix, FONT_NAME, font_size) > max_width:
        clipped = clipped[:-1]
    return clipped + suffix


def _wrap_text(text: str, font_size: float, max_width: float) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        current = ""
        for character in paragraph:
            candidate = current + character
            if current and pdfmetrics.stringWidth(candidate, FONT_NAME, font_size) > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        lines.append(current)
    return lines


def _draw_wrapped(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font_size: float = 8.2,
    leading: float = 11.0,
    color: colors.Color = colors.HexColor("#334155"),
    max_lines: int | None = None,
) -> float:
    lines = _wrap_text(text, font_size, width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _fit_text(lines[-1] + "…", font_size, width)
    pdf.setFont(FONT_NAME, font_size)
    pdf.setFillColor(color)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def _rounded_panel(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = "#FFFFFF",
    stroke: str = "#CBD5E1",
    radius: float = 7.0,
) -> None:
    pdf.setFillColor(colors.HexColor(fill))
    pdf.setStrokeColor(colors.HexColor(stroke))
    pdf.setLineWidth(0.8)
    pdf.roundRect(x, y, width, height, radius, fill=1, stroke=1)


def _draw_badge(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    label: str,
    value: str,
    *,
    fill: str,
    value_color: str = "#0F172A",
) -> None:
    _rounded_panel(pdf, x, y, width, 22, fill=fill, stroke=fill, radius=6)
    pdf.setFont(FONT_NAME, 7.3)
    pdf.setFillColor(colors.HexColor("#475569"))
    pdf.drawString(x + 8, y + 8, label)
    pdf.setFont(FONT_NAME, 10.4)
    pdf.setFillColor(colors.HexColor(value_color))
    pdf.drawRightString(x + width - 8, y + 7, value)


def _draw_flow(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    total_width: float,
) -> None:
    stages: Sequence[tuple[str, tuple[str, ...], str]] = (
        ("1  因果尾窗", ("当前帧", "+ 过去14帧"), "#DBEAFE"),
        ("2  局部覆盖", ("真实射线穿过", "或回波命中"), "#E0E7FF"),
        ("3  车位内融合", ("逐帧地面归一", "保留点/射线来源"), "#F3E8FF"),
        ("4  双证据", ("3D障碍形状", "+ 射线自由空间"), "#FCE7F3"),
        ("5  稳定性复核", ("位置 ±0.20m", "朝向 ±0.50°"), "#FEF3C7"),
        ("6  三态输出", ("证据充分才终态", "否则 unknown"), "#DCFCE7"),
    )
    gap = 13.0
    box_width = (total_width - gap * (len(stages) - 1)) / len(stages)
    box_height = 58.0
    for index, (title, details, fill) in enumerate(stages):
        left = x + index * (box_width + gap)
        _rounded_panel(pdf, left, y, box_width, box_height, fill=fill, stroke=fill, radius=7)
        pdf.setFont(FONT_NAME, 9.0)
        pdf.setFillColor(colors.HexColor("#0F172A"))
        pdf.drawCentredString(left + box_width / 2, y + 40, title)
        pdf.setFont(FONT_NAME, 7.2)
        pdf.setFillColor(colors.HexColor("#475569"))
        pdf.drawCentredString(left + box_width / 2, y + 25, details[0])
        pdf.drawCentredString(left + box_width / 2, y + 14, details[1])
        if index < len(stages) - 1:
            arrow_start = left + box_width + 2.5
            arrow_end = arrow_start + gap - 5.0
            arrow_y = y + box_height / 2
            pdf.setStrokeColor(colors.HexColor("#64748B"))
            pdf.setFillColor(colors.HexColor("#64748B"))
            pdf.setLineWidth(1.2)
            pdf.line(arrow_start, arrow_y, arrow_end, arrow_y)
            pdf.line(arrow_end, arrow_y, arrow_end - 4, arrow_y + 3)
            pdf.line(arrow_end, arrow_y, arrow_end - 4, arrow_y - 3)


def _decision_by_state(
    decisions: Sequence[Mapping[str, Any]],
    state: str,
    *,
    reason: str | None = None,
) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in decisions
            if str(row.get("state")) == state
            and (reason is None or str(row.get("decision_reason")) == reason)
        ),
        None,
    )


def _example_lines(decisions: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    free = _decision_by_state(decisions, "free")
    occupied = _decision_by_state(decisions, "occupied")
    unstable = _decision_by_state(decisions, "unknown", reason="pose_unstable")
    if free:
        free_evidence = free.get("free_evidence", {})
        stability = free.get("stability", {})
        result.append(
            (
                str(free.get("slot_id", "free example")),
                "free",
                "自由体积覆盖 %.1f%%，遮挡 %.1f%%，稳定性 %.0f%%"
                % (
                    100.0 * _safe_float(free_evidence.get("observed_volume_ratio")),
                    100.0 * _safe_float(free_evidence.get("occlusion_ratio")),
                    100.0 * _safe_float(stability.get("pass_ratio")),
                ),
            )
        )
    if occupied:
        occupied_evidence = occupied.get("occupied_evidence", {})
        stability = occupied.get("stability", {})
        result.append(
            (
                str(occupied.get("slot_id", "occupied example")),
                "occupied",
                "3D障碍强度 %.1f%%，支持 %d 帧，稳定性 %.0f%%"
                % (
                    100.0 * _safe_float(occupied_evidence.get("strength")),
                    int(occupied_evidence.get("support_frame_count", 0) or 0),
                    100.0 * _safe_float(stability.get("pass_ratio")),
                ),
            )
        )
    if unstable:
        occupied_evidence = unstable.get("occupied_evidence", {})
        stability = unstable.get("stability", {})
        result.append(
            (
                str(unstable.get("slot_id", "unknown example")),
                "unknown",
                "名义障碍强度 %.1f%%，但扰动仅通过 %d/%d，因此保守保留 unknown"
                % (
                    100.0 * _safe_float(occupied_evidence.get("strength")),
                    int(stability.get("passing_variants", 0) or 0),
                    int(stability.get("total_variants", 0) or 0),
                ),
            )
        )
    return result


def build_pdf(part1_dir: Path, output_path: Path) -> None:
    summary = _load_json(part1_dir / "summary.json")
    local_map = _load_json(part1_dir / "local_map.json")
    decision_payload = _load_json(part1_dir / "slot_decisions.json")
    decisions_raw = decision_payload.get("decisions", [])
    if not isinstance(decisions_raw, list):
        raise ValueError("slot_decisions.json decisions must be a list")
    decisions = [row for row in decisions_raw if isinstance(row, Mapping)]

    window = local_map.get("lidar_window", {})
    counts = local_map.get("counts", {})
    frame_ids = list(window.get("frame_ids", []) or [])
    first_frame = frame_ids[0] if frame_ids else window.get("first_frame_id", "?")
    last_frame = frame_ids[-1] if frame_ids else window.get("last_frame_id", "?")
    first_timestamp = window.get("first_timestamp")
    last_timestamp = window.get("last_timestamp")
    duration = None
    if first_timestamp is not None and last_timestamp is not None:
        duration = _safe_float(last_timestamp) - _safe_float(first_timestamp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _register_font()
    page_width, page_height = landscape(A4)
    pdf = canvas.Canvas(str(output_path), pagesize=(page_width, page_height), pageCompression=1)
    pdf.setTitle("Part1 十五帧车位状态算法简明说明")
    pdf.setAuthor("ParkingAgent")
    pdf.setSubject("Causal 15-frame LiDAR parking-slot state algorithm")

    navy = colors.HexColor("#0F172A")
    slate = colors.HexColor("#475569")
    light = colors.HexColor("#F8FAFC")
    pdf.setFillColor(light)
    pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)

    margin = 12 * mm
    usable_width = page_width - 2 * margin

    pdf.setFont(FONT_NAME, 20)
    pdf.setFillColor(navy)
    pdf.drawString(margin, page_height - 24 * mm, "Part1：15 帧如何得到车位状态")
    pdf.setFont(FONT_NAME, 9.2)
    pdf.setFillColor(slate)
    pdf.drawString(
        margin,
        page_height - 32 * mm,
        "因果局部 LiDAR ｜ 不使用未来帧 ｜ 不使用 Camera ｜ 关键证据不足就输出 unknown",
    )

    badge_y = page_height - 43 * mm
    badge_gap = 7.0
    badge_width = (usable_width - badge_gap * 4) / 5
    duration_text = "%.2f 秒" % duration if duration is not None else "未知"
    badge_values = (
        ("输入窗口", f"{len(frame_ids)} 帧", "#E0F2FE", "#075985"),
        ("帧范围", f"{first_frame}–{last_frame}", "#EEF2FF", "#3730A3"),
        ("时间跨度", duration_text, "#F3E8FF", "#6B21A8"),
        ("局部车位", str(counts.get("local_slot_count", 0)), "#ECFDF5", "#047857"),
        ("远处车位", f"隐藏 {counts.get('unobserved_slots_omitted', 0)}", "#F1F5F9", "#334155"),
    )
    for index, (label, value, fill, value_color) in enumerate(badge_values):
        _draw_badge(
            pdf,
            margin + index * (badge_width + badge_gap),
            badge_y,
            badge_width,
            label,
            value,
            fill=fill,
            value_color=value_color,
        )

    pdf.setFont(FONT_NAME, 11.5)
    pdf.setFillColor(navy)
    pdf.drawString(margin, page_height - 54 * mm, "一眼看懂算法流程")
    _draw_flow(pdf, margin, page_height - 80 * mm, usable_width)
    pdf.setFont(FONT_NAME, 7.5)
    pdf.setFillColor(slate)
    pdf.drawString(
        margin,
        page_height - 84 * mm,
        "注意：这不是 15 次分类后的多数投票；算法融合每个点、每条射线、跨帧支持、遮挡与位姿鲁棒性。",
    )

    panel_y = 18 * mm
    panel_height = page_height - 106 * mm
    panel_gap = 4 * mm
    left_width = (usable_width - panel_gap) * 0.51
    right_width = usable_width - panel_gap - left_width
    left_x = margin
    right_x = margin + left_width + panel_gap
    _rounded_panel(pdf, left_x, panel_y, left_width, panel_height, fill="#FFFFFF")
    _rounded_panel(pdf, right_x, panel_y, right_width, panel_height, fill="#FFFFFF")

    inner = 7 * mm
    pdf.setFont(FONT_NAME, 12)
    pdf.setFillColor(navy)
    pdf.drawString(left_x + inner, panel_y + panel_height - 10 * mm, "什么时候能输出 occupied / free？")

    rules = (
        (
            "occupied：确实看到稳定的车辆形状",
            "至少 40 个车辆高度点、3 个支持帧、Z95≥0.60m、高度跨度≥0.35m、至少 8 个3D体素和2个高度层；同时排除低矮路沿、细长墙柱、相邻车位串点。",
            "#FEE2E2",
            "#B91C1C",
        ),
        (
            "free：射线真正穿透了足够空间",
            "至少 5 个射线帧、2 个相差≥10°的观察方向；自由体积和近地覆盖均≥70%，最大未观测区和遮挡均≤20%；不能存在核心障碍点或 hit/free 冲突。",
            "#DCFCE7",
            "#15803D",
        ),
        (
            "unknown：不是第三种物理状态，而是证据未达标",
            "覆盖不足、遮挡、弱障碍、证据冲突、地面/数据质量问题，或者结论对轻微位姿误差敏感，都会 fail-closed 保留 unknown。",
            "#FEF3C7",
            "#A16207",
        ),
    )
    rule_top = panel_y + panel_height - 21 * mm
    rule_height = 20.5 * mm
    rule_gap = 3.0 * mm
    for index, (title, description, fill, accent) in enumerate(rules):
        bottom = rule_top - (index + 1) * rule_height - index * rule_gap
        _rounded_panel(
            pdf,
            left_x + inner,
            bottom,
            left_width - 2 * inner,
            rule_height,
            fill=fill,
            stroke=fill,
            radius=6,
        )
        pdf.setFont(FONT_NAME, 9.2)
        pdf.setFillColor(colors.HexColor(accent))
        pdf.drawString(left_x + inner + 8, bottom + rule_height - 13, title)
        _draw_wrapped(
            pdf,
            description,
            left_x + inner + 8,
            bottom + rule_height - 26,
            left_width - 2 * inner - 16,
            font_size=7.2,
            leading=9.2,
            color=slate,
            max_lines=3,
        )

    pdf.setFont(FONT_NAME, 7.4)
    pdf.setFillColor(slate)
    pdf.drawString(
        left_x + inner,
        panel_y + 6 * mm,
        "阈值是当前工程默认配置，不代表传感器的物理真值或人工标注真值。",
    )

    pdf.setFont(FONT_NAME, 12)
    pdf.setFillColor(navy)
    pdf.drawString(right_x + inner, panel_y + panel_height - 10 * mm, "最后一道保险与当前实跑结果")
    stability_bottom = panel_y + panel_height - 33 * mm
    _rounded_panel(
        pdf,
        right_x + inner,
        stability_bottom,
        right_width - 2 * inner,
        17 * mm,
        fill="#EFF6FF",
        stroke="#BFDBFE",
        radius=6,
    )
    pdf.setFont(FONT_NAME, 8.8)
    pdf.setFillColor(colors.HexColor("#1D4ED8"))
    pdf.drawString(right_x + inner + 8, stability_bottom + 34, "位姿扰动复核：共 7 个版本")
    pdf.setFont(FONT_NAME, 7.4)
    pdf.setFillColor(slate)
    pdf.drawString(right_x + inner + 8, stability_bottom + 21, "原始 + X/Y 各 ±0.20m + Yaw ±0.50°")
    pdf.drawString(right_x + inner + 8, stability_bottom + 10, "至少通过 6/7 才允许输出终态；否则 reason = pose_unstable。")

    count_y = stability_bottom - 13 * mm
    count_gap = 6.0
    count_width = (right_width - 2 * inner - count_gap * 2) / 3
    current_counts = (
        ("free", int(summary.get("free", 0) or 0), "#DCFCE7", "#15803D"),
        ("occupied", int(summary.get("occupied", 0) or 0), "#FEE2E2", "#B91C1C"),
        ("unknown", int(summary.get("unknown", 0) or 0), "#FEF3C7", "#A16207"),
    )
    for index, (label, value, fill, accent) in enumerate(current_counts):
        left = right_x + inner + index * (count_width + count_gap)
        _rounded_panel(pdf, left, count_y, count_width, 11 * mm, fill=fill, stroke=fill, radius=5)
        pdf.setFont(FONT_NAME, 8)
        pdf.setFillColor(colors.HexColor(accent))
        pdf.drawString(left + 7, count_y + 12, label)
        pdf.setFont(FONT_NAME, 14)
        pdf.drawRightString(left + count_width - 7, count_y + 9, str(value))

    examples = _example_lines(decisions)
    examples_y = count_y - 8 * mm
    pdf.setFont(FONT_NAME, 9.3)
    pdf.setFillColor(navy)
    pdf.drawString(right_x + inner, examples_y, f"frame {summary.get('anchor_frame', '?')} 的三个直观例子")
    cursor_y = examples_y - 15
    state_colors = {"free": "#15803D", "occupied": "#B91C1C", "unknown": "#A16207"}
    for slot_id, state, explanation in examples:
        pdf.setFillColor(colors.HexColor(state_colors.get(state, "#475569")))
        pdf.circle(right_x + inner + 3, cursor_y + 3, 2.3, fill=1, stroke=0)
        pdf.setFont(FONT_NAME, 8.2)
        pdf.drawString(right_x + inner + 10, cursor_y, f"{slot_id}  →  {state}")
        cursor_y = _draw_wrapped(
            pdf,
            explanation,
            right_x + inner + 10,
            cursor_y - 11,
            right_width - 2 * inner - 12,
            font_size=7.2,
            leading=9.0,
            color=slate,
            max_lines=2,
        ) - 6

    reason_counts = summary.get("decision_reason_counts", {})
    unknown_breakdown = (
        f"本次 unknown：weak obstacle {reason_counts.get('weak_obstacle_evidence', 0)}，"
        f"pose unstable {reason_counts.get('pose_unstable', 0)}，"
        f"insufficient evidence {reason_counts.get('insufficient_terminal_evidence', 0)}。"
    )
    _draw_wrapped(
        pdf,
        unknown_breakdown,
        right_x + inner,
        panel_y + 13 * mm,
        right_width - 2 * inner,
        font_size=7.2,
        leading=9.0,
        color=slate,
        max_lines=2,
    )
    pdf.setFont(FONT_NAME, 7.2)
    pdf.setFillColor(colors.HexColor("#64748B"))
    pdf.drawString(
        right_x + inner,
        panel_y + 6 * mm,
        "完整证据：slot_decisions.json ｜ 实际有效帧：decision_trace.jsonl",
    )

    pdf.setFont(FONT_NAME, 7.4)
    pdf.setFillColor(colors.HexColor("#64748B"))
    pdf.drawString(
        margin,
        7 * mm,
        "结论：只有“看到了足够清晰、无遮挡且对 pose 误差稳定的证据”才输出 free/occupied；远处隐藏车位不等于 unknown。",
    )
    pdf.drawRightString(
        page_width - margin,
        7 * mm,
        "ParkingAgent · Part1 local LiDAR snapshot",
    )
    pdf.showPage()
    pdf.save()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1-output-dir", type=Path, default=DEFAULT_PART1_DIR)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    part1_dir = args.part1_output_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else part1_dir / DEFAULT_OUTPUT_NAME
    )
    build_pdf(part1_dir, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
