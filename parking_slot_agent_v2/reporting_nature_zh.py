"""Nature-style Chinese research manuscript and high-resolution figure suite."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image, ImageDraw

from .reporting_story_zh import FONT_CJK, FONT_FALLBACK, _draw_wrapped, _fit_image, _font, _rounded_box


STATE_ZH = {"free": "空闲", "occupied": "占用", "unknown": "未知"}
STATE_COLOR = {"free": "#159E72", "occupied": "#D34A4A", "unknown": "#D79A24"}
BLUE = "#2463A8"
INK = "#14283D"
MUTED = "#60758A"


def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _rel(path: str | Path, root: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), root.resolve())).as_posix()


def _copy(source: str | Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(source), destination)
    return destination


def _result_rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        row["case"]["slot"]["slot_id"]: row
        for row in payload.get("slot_results", ())
    }


def _tool(case: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(row for row in case.get("evidence", ()) if row.get("tool_name") == name)


def _mpl_font(size: int) -> FontProperties:
    source = FONT_CJK if FONT_CJK.is_file() else FONT_FALLBACK
    return FontProperties(fname=str(source), size=size)


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _workflow_figure(path: Path) -> None:
    canvas = Image.new("RGB", (3000, 1700), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 65), "Figure 1 | 保持原工作流的时序证据扩展与安全硬门", font=_font(50), fill=INK)
    draw.text((92, 135), "Part1 合同不变；只扩展 Part2 的严格因果 LiDAR 细查。", font=_font(29), fill=MUTED)
    stages = [
        ("Part1", "15 帧 / 30 m", "输出 Free + Unknown\nFree 优先排序", "#EAF2FC", BLUE),
        ("强制 FOV", "Camera 路由门", "写回可见性\n不可见时禁用 Camera", "#FFF5E6", "#C77C11"),
        ("Part2 LiDAR", "60 帧严格因果", "t ≤ t0；目标局部坐标\n覆盖 / 遮挡 / 核心点 / 高度", "#EDF8F4", "#14845F"),
        ("VLM + 状态机", "结构化证据融合", "OpenAI 提议终态\n代码执行互斥硬门", "#F2EEFB", "#7550A3"),
        ("队列控制", "0.90 + 早停", "可信 Free 立即停止\n否则处理下一车位", "#EAF7EE", "#27834B"),
    ]
    y0, y1 = 370, 900
    for index, (title, subtitle, detail, fill, color) in enumerate(stages):
        x0 = 90 + index * 575
        x1 = x0 + 490
        _rounded_box(draw, (x0, y0, x1, y1), fill=fill, outline=color, radius=28, width=5)
        draw.text((x0 + 28, y0 + 35), title, font=_font(37), fill=color)
        draw.text((x0 + 28, y0 + 105), subtitle, font=_font(31), fill=INK)
        draw.multiline_text((x0 + 28, y0 + 205), detail, font=_font(25), fill="#354A5F", spacing=15)
        if index < len(stages) - 1:
            draw.line((x1 + 18, 635, x1 + 76, 635), fill="#708399", width=8)
            draw.polygon([(x1 + 76, 615), (x1 + 112, 635), (x1 + 76, 655)], fill="#708399")

    _rounded_box(draw, (650, 1060, 2350, 1580), fill="#F8FAFC", outline="#64748B", radius=26, width=4)
    draw.text((705, 1100), "终态硬门（模型置信度不能绕过）", font=_font(38), fill=INK)
    gates = [
        ("Free", "充分体积/近地覆盖；低遮挡；无 core hit；反向 Occupied 被否决；7/7 稳定", STATE_COLOR["free"]),
        ("Occupied", "核心内部立体点；边界比 <0.5；线性风险 <0.6；反向 Free 不强；7/7 稳定", STATE_COLOR["occupied"]),
        ("Unknown", "任一关键门失败、证据冲突、部分路线覆盖或 Camera 无法可靠定位", STATE_COLOR["unknown"]),
    ]
    for index, (name, text, color) in enumerate(gates):
        y = 1190 + index * 115
        draw.ellipse((710, y, 752, y + 42), fill=color)
        draw.text((775, y - 2), name, font=_font(29), fill=color)
        _draw_wrapped(draw, (960, y - 2), text, font=_font(25), fill="#364A5F", width=1300, spacing=8)
    canvas.save(path)


def _window_ablation_figure(path: Path, experiments: Sequence[Mapping[str, Any]]) -> None:
    rows = sorted(experiments, key=lambda row: int(row["window_frames"]))
    windows = [int(row["window_frames"]) for row in rows]
    resolved = [int(row["resolved_count"]) for row in rows]
    seconds = [float(row["processing_seconds"]) for row in rows]
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    bars = ax.bar(windows, resolved, width=8.5, color=["#B8C5D3", "#84AED6", "#4F8CC5", "#159E72", "#8DB7A7"], edgecolor="white")
    ax.set_xlabel("严格因果 LiDAR 窗口（帧）", fontproperties=_mpl_font(13))
    ax.set_ylabel("通过终态硬门的原始 Unknown 数量（n=22）", fontproperties=_mpl_font(13))
    ax.set_ylim(0, 11)
    ax.set_xticks(windows)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(_mpl_font(11))
    for bar, count in zip(bars, resolved):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.25, f"{count}/22", ha="center", fontproperties=_mpl_font(11), color=INK)
    cost = ax.twinx()
    cost.plot(windows, seconds, color="#D04A3A", marker="o", linewidth=2.8, markersize=8)
    cost.set_ylabel("完整几何处理时间（秒）", fontproperties=_mpl_font(13), color="#A63B2D")
    cost.tick_params(axis="y", colors="#A63B2D")
    for label in cost.get_yticklabels():
        label.set_fontproperties(_mpl_font(11))
    ax.axvline(60, color="#159E72", linestyle="--", linewidth=1.8)
    ax.text(60.8, 1.0, "Pareto 工作点\n60 与 75 同为 9/22", fontproperties=_mpl_font(11), color="#087A55")
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("Figure 2 | 消歧随时序窗口增长而提升，并在 60 帧进入平台期", fontproperties=_mpl_font(17), color=INK, pad=16)
    _save_figure(fig, path)


def _cross_anchor_figure(
    path: Path,
    baseline: Sequence[Mapping[str, Any]],
    extended: Sequence[Mapping[str, Any]],
) -> None:
    base = {int(row["anchor_frame"]): row for row in baseline}
    ext = {int(row["anchor_frame"]): row for row in extended}
    anchors = sorted(ext)
    baseline_rates = [100 * float(base[a]["resolved_rate"]) for a in anchors]
    extended_rates = [100 * float(ext[a]["resolved_rate"]) for a in anchors]
    totals = [int(ext[a]["part1_unknown_count"]) for a in anchors]
    pooled_resolved = sum(int(ext[a]["resolved_count"]) for a in anchors)
    pooled_total = sum(totals)
    labels = [str(a) for a in anchors] + ["合并"]
    baseline_rates.append(0.0)
    extended_rates.append(100 * pooled_resolved / pooled_total)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.bar(x - width / 2, baseline_rates, width, label="15 帧严格基线", color="#C8D1DB")
    bars = ax.bar(x + width / 2, extended_rates, width, label="60 帧严格因果", color="#2463A8")
    for index, bar in enumerate(bars):
        if index < len(totals):
            note = f"{int(ext[anchors[index]]['resolved_count'])}/{totals[index]}"
        else:
            note = f"{pooled_resolved}/{pooled_total}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, note, ha="center", fontproperties=_mpl_font(10), color=INK)
    ax.set_xticks(x, labels)
    ax.set_xlabel("系统采样锚帧（开发锚点 9277 未包含）", fontproperties=_mpl_font(13))
    ax.set_ylabel("原始 Unknown 严格消歧率（%）", fontproperties=_mpl_font(13))
    ax.set_ylim(0, max(extended_rates) + 8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(_mpl_font(11))
    ax.legend(prop=_mpl_font(11), frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("Figure 3 | 六个预先固定路线锚点的配对复核均显示正提升", fontproperties=_mpl_font(17), color=INK, pad=16)
    _save_figure(fig, path)


def _case_gallery(path: Path, result_rows: Mapping[str, Mapping[str, Any]]) -> None:
    canvas = Image.new("RGB", (3400, 2600), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 45), "Figure 4 | 两类成功消歧具有互补的几何证据", font=_font(48), fill=INK)
    specs = [
        ("slot_1012", "Unknown → Free 0.98", "边界点不等于车位内部障碍；自由空间覆盖充分。", STATE_COLOR["free"]),
        ("slot_1258", "Unknown → Occupied 0.96", "核心内部存在跨帧、跨高度层的稳定立体返回。", STATE_COLOR["occupied"]),
    ]
    for index, (slot_id, title, summary, color) in enumerate(specs):
        x = 70 + index * 1660
        row = result_rows[slot_id]
        case = row["case"]
        fov = Path(_tool(case, "check_fov")["artifact_paths"][0])
        evidence = Path(_tool(case, "lidar_detail")["metadata"]["model_image_paths"][0])
        _rounded_box(draw, (x, 165, x + 1580, 2500), fill="#F8FAFC", outline=color, radius=24, width=5)
        draw.text((x + 40, 205), title, font=_font(39), fill=color)
        _draw_wrapped(draw, (x + 40, 275), summary, font=_font(27), fill="#40566D", width=1480, spacing=9)
        fov_fit = _fit_image(fov, (660, 760))
        evidence_fit = _fit_image(evidence, (1480, 1280))
        canvas.paste(fov_fit, (x + 40, 440))
        canvas.paste(evidence_fit, (x + 40, 1180))
        draw.rectangle((x + 40, 440, x + 700, 1200), outline="#A8B5C4", width=3)
        draw.rectangle((x + 40, 1180, x + 1520, 2460), outline="#A8B5C4", width=3)
        draw.text((x + 745, 520), "FOV 只负责路由 Camera；\n状态改变来自 LiDAR 数值硬门。", font=_font(26), fill="#53687C")
    canvas.save(path)


def _safety_gallery(path: Path, result_rows: Mapping[str, Mapping[str, Any]]) -> None:
    canvas = Image.new("RGB", (3600, 2100), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 45), "Figure 5 | 负结果与安全反例：强证据外观仍不能绕过硬门", font=_font(47), fill=INK)
    specs = [
        ("slot_1251", "位姿不稳定", "上游倾向 Occupied，但仅 6/7 扰动通过；最终 Unknown。"),
        ("slot_1010", "Free/Occupied 冲突", "双方证据同时存在，互斥条件未满足；最终 Unknown。"),
        ("slot_1248", "部分路线覆盖", "核心射线不足且 Camera 无像素投影；三轮后仍 Unknown。"),
    ]
    for index, (slot_id, title, summary) in enumerate(specs):
        x = 70 + index * 1170
        row = result_rows[slot_id]
        case = row["case"]
        evidence = Path(_tool(case, "lidar_detail")["metadata"]["model_image_paths"][0])
        _rounded_box(draw, (x, 170, x + 1080, 1980), fill="#FFF9EF", outline=STATE_COLOR["unknown"], radius=22, width=4)
        draw.text((x + 35, 205), f"{slot_id} | {title}", font=_font(32), fill="#A66A08")
        _draw_wrapped(draw, (x + 35, 270), summary, font=_font(25), fill="#4B5E70", width=1005, spacing=9)
        fitted = _fit_image(evidence, (1010, 1390))
        canvas.paste(fitted, (x + 35, 500))
        draw.rectangle((x + 35, 500, x + 1045, 1890), outline="#B7A57A", width=3)
    canvas.save(path)


def _outcome_and_blocker_figure(path: Path, result_rows: Mapping[str, Mapping[str, Any]]) -> None:
    unknown_rows = [row for row in result_rows.values() if row["case"]["part1_state"] == "unknown"]
    states = Counter(row["case"]["final_state"] for row in unknown_rows)
    blockers: Counter[str] = Counter()
    for row in unknown_rows:
        if row["case"]["final_state"] != "unknown":
            continue
        lidar = _tool(row["case"], "lidar_detail")
        gate = lidar.get("metadata", {}).get("geometry_card", {}).get("terminal_geometry_gate", {})
        blockers.update(set(gate.get("free_blockers", ())) | set(gate.get("occupied_blockers", ())))
    blocker_names = [name for name, _ in blockers.most_common(8)]
    blocker_values = [blockers[name] for name in blocker_names]
    short = {
        "robustness_not_all_variants_stable": "位姿未全稳",
        "no_strong_free_geometry_candidate": "Free 不够强",
        "free_strong_gate_failed": "Free 强门失败",
        "free_geometry_has_failures": "Free 有失败项",
        "unresolved_core_hit": "核心仍命中",
        "opposing_occupied_geometry_not_vetoed": "Occupied 未排除",
        "no_strong_occupied_geometry_candidate": "Occupied 不够强",
        "occupied_strong_gate_failed": "Occupied 强门失败",
        "occupied_geometry_has_failures": "Occupied 有失败项",
        "boundary_ratio_not_below_0_5": "边界比过高",
        "linearity_risk_not_below_0_6": "线性风险过高",
        "no_core_obstacle_points": "无核心障碍点",
        "opposing_free_strong_gate": "Free 反证过强",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), gridspec_kw={"width_ratios": [0.8, 1.5]})
    labels = ["Free", "Occupied", "Unknown"]
    values = [states["free"], states["occupied"], states["unknown"]]
    axes[0].bar(labels, values, color=[STATE_COLOR["free"], STATE_COLOR["occupied"], STATE_COLOR["unknown"]])
    axes[0].set_ylim(0, 15)
    axes[0].set_title("a  OpenAI Agent 终态", fontproperties=_mpl_font(14), loc="left")
    axes[0].set_ylabel("车位数（n=22）", fontproperties=_mpl_font(12))
    for index, value in enumerate(values):
        axes[0].text(index, value + 0.35, str(value), ha="center", fontproperties=_mpl_font(12))
    for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
        label.set_fontproperties(_mpl_font(11))
    y = np.arange(len(blocker_names))
    axes[1].barh(y, blocker_values, color="#D79A24")
    axes[1].set_yticks(y, [short.get(name, name) for name in blocker_names])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("13 个剩余 Unknown 中命中数量（非互斥）", fontproperties=_mpl_font(12))
    axes[1].set_title("b  硬门阻断原因", fontproperties=_mpl_font(14), loc="left")
    for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
        label.set_fontproperties(_mpl_font(10.5))
    for index, value in enumerate(blocker_values):
        axes[1].text(value + 0.15, index, str(value), va="center", fontproperties=_mpl_font(10))
    fig.suptitle("Extended Data Figure 1 | 终态组成与失败模式完整披露", fontproperties=_mpl_font(17), color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, path)


def _legacy_30_anchor_figure(path: Path, manifest: Mapping[str, Any]) -> None:
    anchors = manifest["anchors"]
    frames = [int(row["anchor_frame"]) for row in anchors]
    candidates = [int(row["candidate_count"]) for row in anchors]
    processed = [int(row["part2_processed_count"]) for row in anchors]
    selected = [row.get("part2_selected_slot_id") is not None for row in anchors]
    x = np.arange(len(frames))
    fig, ax = plt.subplots(figsize=(15, 6.8))
    ax.bar(x, candidates, color="#CCD6E0", label="候选数")
    ax.bar(x, processed, color="#4F8CC5", label="实际处理数")
    for index, flag in enumerate(selected):
        if flag:
            ax.scatter(index, candidates[index] + 3, marker="*", s=140, color="#159E72", zorder=3)
    ax.set_xticks(x, [str(frame) for frame in frames], rotation=55, ha="right")
    ax.set_ylabel("车位数", fontproperties=_mpl_font(12))
    ax.set_xlabel("30 个系统采样锚帧", fontproperties=_mpl_font(12))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(_mpl_font(8.5))
    ax.legend(prop=_mpl_font(10), frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("Extended Data Figure 2 | 独立 30 锚点旧工作流审计（星号：找到可信 Free）", fontproperties=_mpl_font(16), color=INK, pad=14)
    _save_figure(fig, path)


def _state_matrix_figure(path: Path, result_rows: Mapping[str, Mapping[str, Any]]) -> None:
    rows = [row for row in result_rows.values() if row["case"]["part1_state"] == "unknown"]
    rows.sort(key=lambda row: row["case"]["slot"]["slot_id"])
    canvas = Image.new("RGB", (2600, 1300), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((65, 45), "Extended Data Figure 3 | 22 个原始 Unknown 的完整终态矩阵", font=_font(43), fill=INK)
    for index, row in enumerate(rows):
        col = index % 6
        line = index // 6
        x = 65 + col * 415
        y = 170 + line * 260
        case = row["case"]
        state = case["final_state"]
        scores = case["final_scores"]
        confidence = max(float(scores["free_confidence"]), float(scores["occupied_confidence"]), float(scores["unknown_confidence"]))
        color = STATE_COLOR[state]
        _rounded_box(draw, (x, y, x + 370, y + 210), fill="#F8FAFC", outline=color, radius=18, width=4)
        draw.text((x + 20, y + 23), case["slot"]["slot_id"], font=_font(27), fill=INK)
        draw.text((x + 20, y + 83), f"Unknown → {STATE_ZH[state]}", font=_font(25), fill=color)
        draw.text((x + 20, y + 143), f"终态分 {confidence:.2f}", font=_font(22), fill="#53687C")
    canvas.save(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _build_pdf(
    path: Path,
    *,
    title: str,
    sections: Sequence[tuple[str, str]],
    figures: Sequence[tuple[Path, str]],
) -> None:
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
    pdfmetrics.registerFont(TTFont("PaperCJK", str(font_path)))
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("PaperTitle", parent=styles["Title"], fontName="PaperCJK", fontSize=20, leading=28, alignment=TA_CENTER, textColor=colors.HexColor(INK), spaceAfter=18)
    heading = ParagraphStyle("PaperHeading", parent=styles["Heading1"], fontName="PaperCJK", fontSize=14, leading=20, textColor=colors.HexColor(BLUE), spaceBefore=12, spaceAfter=8)
    body = ParagraphStyle("PaperBody", parent=styles["BodyText"], fontName="PaperCJK", fontSize=9.5, leading=15, textColor=colors.HexColor("#23384C"), spaceAfter=8)
    caption = ParagraphStyle("PaperCaption", parent=body, fontSize=8.5, leading=12, textColor=colors.HexColor("#52677A"), spaceAfter=14)
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=1.7 * cm, leftMargin=1.7 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm, title=title)
    story: list[Any] = [Paragraph(title, title_style), Paragraph("研究级内部稿｜非正式投稿版本｜无人工 GT", caption), Spacer(1, 8)]
    for heading_text, content in sections:
        story.append(Paragraph(heading_text, heading))
        for paragraph in content.split("\n\n"):
            story.append(Paragraph(html.escape(paragraph).replace("\n", "<br/>"), body))
    story.append(PageBreak())
    story.append(Paragraph("主要图与图注", heading))
    max_width = A4[0] - 3.4 * cm
    for figure_path, legend in figures:
        with Image.open(figure_path) as image:
            width, height = image.size
        draw_width = max_width
        draw_height = draw_width * height / width
        if draw_height > 18.5 * cm:
            draw_height = 18.5 * cm
            draw_width = draw_height * width / height
        story.extend([RLImage(str(figure_path), width=draw_width, height=draw_height), Paragraph(html.escape(legend), caption), Spacer(1, 8)])
    doc.build(story)


def build_nature_style_report(
    *,
    window_ablation_path: str | Path,
    cross_anchor_baseline_path: str | Path,
    cross_anchor_extended_path: str | Path,
    openai_result_path: str | Path,
    operational_result_path: str | Path,
    legacy_manifest_path: str | Path,
    legacy_summary_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    figures_dir = destination / "figures"
    extended_dir = destination / "extended_data"
    tables_dir = destination / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    extended_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    window = _load(window_ablation_path)
    baseline = _load(cross_anchor_baseline_path)
    extended = _load(cross_anchor_extended_path)
    openai_result = _load(openai_result_path)
    operational = _load(operational_result_path)
    legacy_manifest = _load(legacy_manifest_path)
    legacy_summary = _load(legacy_summary_path)
    result_rows = _result_rows(openai_result)

    fig1 = figures_dir / "Figure_1_workflow.png"
    fig2 = figures_dir / "Figure_2_window_ablation.png"
    fig3 = figures_dir / "Figure_3_cross_anchor.png"
    fig4 = figures_dir / "Figure_4_success_cases.png"
    fig5 = figures_dir / "Figure_5_safety_failures.png"
    ext1 = extended_dir / "Extended_Data_Figure_1_outcomes_blockers.png"
    ext2 = extended_dir / "Extended_Data_Figure_2_legacy30.png"
    ext3 = extended_dir / "Extended_Data_Figure_3_all_unknowns.png"
    _workflow_figure(fig1)
    _window_ablation_figure(fig2, window["experiments"])
    _cross_anchor_figure(fig3, baseline["experiments"], extended["experiments"])
    _case_gallery(fig4, result_rows)
    _safety_gallery(fig5, result_rows)
    _outcome_and_blocker_figure(ext1, result_rows)
    _legacy_30_anchor_figure(ext2, legacy_manifest)
    _state_matrix_figure(ext3, result_rows)

    window_rows = [
        {
            "window_frames": row["window_frames"],
            "unknown_count": row["part1_unknown_count"],
            "resolved_count": row["resolved_count"],
            "resolved_rate": row["resolved_rate"],
            "free_count": row["gate_state_counts"].get("free", 0),
            "occupied_count": row["gate_state_counts"].get("occupied", 0),
            "processing_seconds": row["processing_seconds"],
        }
        for row in window["experiments"]
    ]
    _write_csv(tables_dir / "Table_1_window_ablation.csv", window_rows, tuple(window_rows[0]))
    cross_rows: list[dict[str, Any]] = []
    base_by_anchor = {int(row["anchor_frame"]): row for row in baseline["experiments"]}
    for row in extended["experiments"]:
        anchor = int(row["anchor_frame"])
        cross_rows.append(
            {
                "anchor_frame": anchor,
                "part1_unknown_count": row["part1_unknown_count"],
                "baseline15_resolved": base_by_anchor[anchor]["resolved_count"],
                "extended60_resolved": row["resolved_count"],
                "extended60_rate": row["resolved_rate"],
                "free_count": row["gate_state_counts"].get("free", 0),
                "occupied_count": row["gate_state_counts"].get("occupied", 0),
                "processing_seconds": row["processing_seconds"],
            }
        )
    _write_csv(tables_dir / "Table_2_cross_anchor.csv", cross_rows, tuple(cross_rows[0]))
    case_rows = []
    for slot_id, row in sorted(result_rows.items()):
        case = row["case"]
        if case["part1_state"] != "unknown":
            continue
        scores = case["final_scores"]
        case_rows.append(
            {
                "slot_id": slot_id,
                "part1_state": case["part1_state"],
                "final_state": case["final_state"],
                "free_confidence": scores["free_confidence"],
                "occupied_confidence": scores["occupied_confidence"],
                "unknown_confidence": scores["unknown_confidence"],
                "model_turns": row["model_turns"],
                "tool_rounds": row["tool_rounds"],
                "stop_reason": row["stop_reason"],
                "validation_error_count": len(row["validation_errors"]),
                "final_reason": case["final_reason"],
            }
        )
    _write_csv(tables_dir / "Table_3_openai_case_results.csv", case_rows, tuple(case_rows[0]))

    window_by_size = {int(row["window_frames"]): row for row in window["experiments"]}
    time60 = float(window_by_size[60]["processing_seconds"])
    time75 = float(window_by_size[75]["processing_seconds"])
    saved = 100 * (time75 - time60) / time75
    pooled_total = sum(int(row["part1_unknown_count"]) for row in extended["experiments"])
    pooled_resolved = sum(int(row["resolved_count"]) for row in extended["experiments"])
    pooled_free = sum(int(row["gate_state_counts"].get("free", 0)) for row in extended["experiments"])
    pooled_occupied = sum(int(row["gate_state_counts"].get("occupied", 0)) for row in extended["experiments"])
    final_states = Counter(
        row["case"]["final_state"]
        for row in result_rows.values()
        if row["case"]["part1_state"] == "unknown"
    )
    model_turns = sum(int(row["model_turns"]) for row in result_rows.values())
    validation_errors = sum(len(row["validation_errors"]) for row in result_rows.values())

    title = "严格因果时序证据与硬门控多模态智能体降低停车位占用判断的不确定性"
    abstract = (
        "自动泊车中的车位占用判断常因稀疏观测、遮挡、边界结构和位姿敏感性而停留在 Unknown。"
        "本文提出一种保持既有 Free 优先 ReAct 工作流不变的证据扩展方法：Part1 继续使用 15 帧局部地图，"
        "Part2 在同一决策时刻之前积累 60 帧目标对齐 LiDAR，并通过核心点、边界比、线性风险、自由空间覆盖、遮挡与 7/7 位姿扰动硬门约束多模态模型。"
        "在开发锚点 frame 9277 的 22 个原始 Unknown 上，严格消歧从 0/22 提升至 9/22；OpenAI Agent 独立确认 2 Free、7 Occupied，零校验错误。"
        f"窗口消融显示 15/30/45/60/75 帧分别解决 0/6/8/9/9 个，60 帧相较 75 帧节省 {saved:.1f}% 几何处理时间。"
        f"在排除开发锚点的六个系统采样路线位置上，匹配 15 帧基线为 0/{pooled_total}，60 帧解决 {pooled_resolved}/{pooled_total}（{100*pooled_resolved/pooled_total:.1f}%；{pooled_free} Free、{pooled_occupied} Occupied）。"
        "由于缺少人工真值，这些结果度量严格证据消歧率而非分类准确率。"
    )
    sections = [
        ("摘要", abstract),
        ("引言", "停车位占用判断不是简单的图像分类问题。已知车位多边形可以把问题约束到局部几何，但稀疏 LiDAR、长条边界结构、遮挡和位姿漂移会令单帧或短窗口证据产生冲突。传统停车位视觉方法侧重检测与分类，而自动驾驶占用研究表明，多帧对齐能够显著增加空间覆盖。与此同时，ReAct 式语言模型智能体能够在推理与工具调用之间迭代，但其自报置信度不能替代可验证的几何约束。\n\n我们把 VLM 定位为证据解释器，而不是最终安全裁判。方法保留强制 FOV、单车位最多三轮、Free 优先队列和 0.90 终态合同，仅将 Part2 LiDAR 细查从 15 帧扩展到严格因果长窗口，并在状态机中加入不可绕过的互斥硬门。"),
        ("结果一｜窗口长度控制消歧与计算成本", f"在 frame 9277 的同一批 22 个 Part1 Unknown 上，只改变历史窗口长度。15、30、45、60 和 75 帧分别解决 0、6、8、9 和 9 个。该单调趋势支持“更多有效视角减少未知性”的机制解释，而非 Prompt 偶然波动。60 帧首次达到最大消歧数，处理时间 {time60:.1f} 秒；75 帧为 {time75:.1f} 秒且没有新增终态，因此选取 60 帧为 Pareto 工作点。"),
        ("结果二｜跨路线系统采样复核", f"从既有 30 锚点 manifest 中按索引 1、7、13、19、25、30 系统取样，并排除开发锚点 9277。六处共 {pooled_total} 个原始 Unknown；配对 15 帧严格基线全部保持 Unknown，60 帧严格门解决 {pooled_resolved} 个（{100*pooled_resolved/pooled_total:.1f}%），其中 {pooled_free} Free、{pooled_occupied} Occupied。六个锚点均为正提升，但单锚点解决率存在明显差异，提示增益依赖车辆轨迹与可见基线。"),
        ("结果三｜真实 OpenAI Agent 验收", f"在 60 帧正式证据包上进行全候选 exhaustive 评测。22 个原始 Unknown 最终为 {final_states['free']} Free、{final_states['occupied']} Occupied 和 {final_states['unknown']} Unknown；共 {model_turns} 次模型调用，校验错误 {validation_errors}。默认运营模式不启用 exhaustive：只处理第一个 Free 候选 slot_1038，一次 LiDAR、一次模型调用，Free={float(operational['slot_results'][0]['case']['final_scores']['free_confidence']):.2f} 后立即停止。"),
        ("结果四｜硬门阻止高置信错误", "slot_1251 是关键安全反例。60 帧上游几何强烈偏向 Occupied，但仅 6/7 位姿扰动通过；模型因此保留 Unknown。slot_1010 同时存在 Free 与 Occupied 证据，slot_1248 则受部分路线覆盖和 Camera 无像素投影限制。三类失败均未通过调低阈值强行解决。"),
        ("讨论", "本研究的主要发现是：在已知车位几何条件下，Unknown 的减少主要由因果时序覆盖和显式几何互斥产生，VLM 的作用是阅读、组织和引用证据。60 帧工作点在开发锚点达到与 75 帧相同的严格终态数，并减少约五分之一计算时间。跨锚点结果证明机制具有路线外延性，但 16.3% 的合并解决率低于开发锚点的 40.9%，说明不能用单一锚点结果概括全路线。\n\n局限性包括：没有人工 GT，无法报告准确率、灵敏度或特异度；跨锚点实验只覆盖六个系统采样位置；Camera 没有经过真实车位到像素投影标定；OpenAI 分数未经统计校准；处理时间来自单一服务器容器。当前稿件应视为方法学与工作流证据，而不是生产安全认证。"),
        ("方法", "数据与采样：使用 pose-corrected-final 帧索引、地图点云和已知车位数据库。所有扩展窗口均满足 frame_id≤t0，无未来帧。开发锚点为 9277；跨路线锚点按预先固定 manifest 索引系统采样。\n\n几何证据：点云被变换到车位局部坐标。Occupied 证据考察核心点、三维高度层、时序支持、边界比、线性和相邻车位重叠；Free 证据考察核心射线覆盖、近地 BEV 覆盖、观测体积、遮挡和未解决核心命中。\n\n稳健性：每个终态要求 7/7 位姿扰动变体通过。Free 与 Occupied 还必须相互否决，任何冲突均保留 Unknown。\n\nAgent：每个 SlotCase 先执行 FOV；扩展 LiDAR 资源存在时预先加载；OpenAI 返回严格 JSON 动作。最终状态需要置信度≥0.90且通过代码硬门。全量统计使用显式 evaluation-exhaustive；默认运营模式保持第一个可信 Free 早停。"),
        ("数据和代码可用性", "本研究输入位于本地数据挂载与 outputs/frame_map_dataset_pose_corrected_final。出于数据授权和体积限制，本稿未将原始传感器数据公开。最小机器审计、窗口消融、跨锚点结果、逐车位表格、生成脚本和哈希 manifest 随报告目录提供。代码入口包括 parking_slot_agent_v2/extended_lidar.py、lidar_geometry.py、tools.py、agent.py、pipeline.py 与 scripts/run_extended_lidar_ablation.py。"),
        ("参考文献", "1. Yao et al. ReAct: Synergizing Reasoning and Acting in Language Models. ICLR (2023). https://openreview.net/pdf?id=WE_vluYUL-X\n2. Wei et al. SurroundOcc: Multi-camera 3D Occupancy Prediction for Autonomous Driving. ICCV (2023). https://openaccess.thecvf.com/content/ICCV2023/html/Wei_SurroundOcc_Multi-camera_3D_Occupancy_Prediction_for_Autonomous_Driving_ICCV_2023_paper.html\n3. Man et al. BEV-Guided Multi-Modality Fusion for Driving Perception. CVPR (2023). https://openaccess.thecvf.com/content/CVPR2023/html/Man_BEV-Guided_Multi-Modality_Fusion_for_Driving_Perception_CVPR_2023_paper.html\n4. Shi et al. StreamingFlow: Streaming Occupancy Forecasting with Asynchronous Multi-modal Data Streams. CVPR (2024). https://openaccess.thecvf.com/content/CVPR2024/html/Shi_StreamingFlow_Streaming_Occupancy_Forecasting_with_Asynchronous_Multi-modal_Data_Streams_via_CVPR_2024_paper.html\n5. Grbić & Koch. Automatic Vision-Based Parking Slot Detection and Occupancy Classification (2023). https://arxiv.org/abs/2308.08192"),
    ]
    legends = [
        (fig1, "Figure 1｜系统结构。Part1 保持 15 帧合同；Part2 使用截至同一 t0 的 60 帧目标对齐 LiDAR；VLM 输出必须通过代码硬门。"),
        (fig2, "Figure 2｜frame 9277 窗口消融。柱为严格解决数量，折线为完整几何处理时间。60 帧首次达到 9/22 的平台。"),
        (fig3, "Figure 3｜六锚点配对复核。每个锚点的 15 帧基线均为 0；60 帧在六处均解决至少一个 Unknown。"),
        (fig4, "Figure 4｜代表性成功案例。slot_1012 由边界伪障碍纠正为 Free；slot_1258 由稳定核心体积证据判为 Occupied。"),
        (fig5, "Figure 5｜安全失败案例。位姿不稳、证据冲突与部分覆盖均保持 Unknown。"),
    ]

    md_path = destination / "manuscript_zh.md"
    md_lines = [f"# {title}", "", "作者：待填写", "", "> 研究级内部稿；非正式投稿版本；无人工 GT。", ""]
    for heading, content in sections:
        md_lines.extend([f"## {heading}", "", content, ""])
        if heading == "方法":
            md_lines.extend(["## 图", ""] + [f"![{legend}]({_rel(path, destination)})\n\n*{legend}*\n" for path, legend in legends])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    html_path = destination / "manuscript.html"
    section_html = "".join(
        f"<section><h2>{html.escape(heading)}</h2>" + "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in content.split("\n\n")) + "</section>"
        for heading, content in sections
    )
    figure_html = "".join(
        f"<figure><img src='{html.escape(_rel(path,destination))}'><figcaption>{html.escape(legend)}</figcaption></figure>"
        for path, legend in legends
    )
    extended_html = "".join(
        f"<figure><img src='{html.escape(_rel(path,destination))}'></figure>"
        for path in (ext1, ext2, ext3)
    )
    html_path.write_text(
        f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>
body{{margin:0;background:#eef2f5;color:#1d2d3d;font-family:Georgia,"Noto Serif CJK SC","Microsoft YaHei",serif;line-height:1.82}}main{{max-width:1120px;margin:auto;background:white;padding:54px 72px}}h1{{font-size:42px;line-height:1.25;color:#14283d}}h2{{font-size:25px;margin-top:42px;border-bottom:1px solid #b8c6d3;padding-bottom:8px}}p{{font-size:17px;text-align:justify}}.meta{{color:#66798b}}.summary{{font-size:19px;border-left:5px solid #2463a8;padding:18px 24px;background:#f2f7fb}}figure{{margin:38px 0}}img{{width:100%;height:auto;border:1px solid #d5dde5}}figcaption{{font-size:14px;color:#52677a;margin-top:9px}}nav{{background:#f4f7fa;padding:16px 22px;border-radius:8px}}code{{overflow-wrap:anywhere}}@media(max-width:760px){{main{{padding:24px}}h1{{font-size:30px}}}}</style></head><body><main><h1>{html.escape(title)}</h1><p class='meta'>作者：待填写｜研究级内部稿｜生成日期：2026-07-21｜无人工 GT</p><p class='summary'>{html.escape(abstract)}</p><nav><b>交付内容：</b>主文、5 张主图、3 张 Extended Data 图、3 张 CSV 表、机器可读复现 manifest 和 PDF。</nav>{section_html}<section><h2>主要图</h2>{figure_html}</section><section><h2>Extended Data</h2>{extended_html}</section></main></body></html>""",
        encoding="utf-8",
    )

    pdf_path = destination / "manuscript.pdf"
    _build_pdf(pdf_path, title=title, sections=sections, figures=legends)

    checklist_path = destination / "reporting_summary_zh.md"
    checklist_path.write_text(
        f"""# Nature Portfolio 风格 Reporting Summary（项目适配版）

## 研究设计

- 研究类型：确定性方法学与工作流评测，不是监督分类准确率研究。
- 开发锚点：frame 9277，原始 Part1 Unknown `n=22` 个车位。
- 窗口消融：同一批 22 个车位重复测量 15、30、45、60、75 帧。
- 跨路线复核：manifest 索引 1、7、13、19、25、30，对应 6 个预先固定锚点；开发锚点 9277 排除。
- 跨路线分析单位：车位候选 `n={pooled_total}`，嵌套于 6 个锚点；报告描述性合并率和逐锚点率，不把车位当作完全独立随机样本做显著性检验。
- 旧工作流背景：独立 30 锚点审计，980 个候选、754 个实际处理；该运行使用旧 15 帧 Codex agent，与 60 帧 OpenAI 主实验严格分开。

## 样本选择与排除

- 锚点选择：按已有、不可变 manifest 的系统索引选择，不根据实验结果回选。
- 车位纳入：每个锚点 Part1 合同中状态为 Unknown 的全部候选。
- 排除：无事后排除；管线缺失的候选将按 Unknown 失败关闭。
- 随机化与盲法：不适用；算法为确定性几何门，OpenAI 评测使用固定结构化提示与同一模型配置。

## 统计报告

- 开发锚点：15/30/45/60/75 帧严格解决数为 0/6/8/9/9（分母均为 22）。
- 系统六锚点：15 帧为 0/{pooled_total}；60 帧为 {pooled_resolved}/{pooled_total}（{100*pooled_resolved/pooled_total:.2f}%）。
- 60 帧状态组成：{pooled_free} Free、{pooled_occupied} Occupied。
- OpenAI 60 帧：2 Free、7 Occupied、13 Unknown；状态机校验错误 {validation_errors}。
- 未计算准确率、置信区间、p 值、灵敏度或特异度，因为没有人工 GT，且车位在锚点内聚类。

## 可复现性

- 所有扩展窗口严格满足 `frame_id <= t0`。
- 终态要求模型分数 ≥0.90，并通过 Free/Occupied 互斥几何门与 7/7 位姿扰动。
- API 提供方失败属于运行级异常，不写成语义 Unknown；输出支持原子 checkpoint 与 `--resume`。
- 输入、结果、Figures、CSV 与稿件均在 `reproducibility_manifest.json` 中记录 SHA-256。

## 数据与代码可用性

- 本地最小审计数据、消融 JSON、逐车位 CSV、生成脚本和 Figures 随本目录提供。
- 原始传感器数据受体积与授权约束，正式投稿前需要给出可访问的数据声明或脱敏最小数据集。
- 正式投稿前仍需：人工 GT、更多路线/天气/车辆场景、独立 Camera 像素标定、统计校准和外部复现。
""",
        encoding="utf-8",
    )
    gap_path = destination / "submission_gap_analysis_zh.md"
    gap_path.write_text(
        """# 投稿差距分析

当前产物是完整的 Nature 风格研究稿，不等于已经满足 Nature 的编辑标准。

## 已具备

- 清楚的单一机制主张：因果时序几何证据与不可绕过硬门减少 Unknown。
- 同锚点窗口消融、跨锚点配对、真实 OpenAI Agent 验收和负结果披露。
- 主图、Extended Data、逐车位表格、方法、数据/代码声明与哈希复现 manifest。

## 正式投稿前必须补齐

1. 人工标注 GT，并报告准确率、Free/Occupied 混淆矩阵、误报和漏报。
2. 预注册或固定更大规模独立测试集，覆盖不同路线、天气、光照和车辆动态。
3. 车辆级/锚点级聚类统计与置信区间，避免把相关车位当作独立样本。
4. 与学习式视觉、LiDAR 基线和无 VLM/无硬门/无长窗口方案做统一 GT 对比。
5. 独立 Camera 像素投影标定与跨传感器时序误差评估。
6. 外部团队复现、数据授权、作者/单位、利益冲突、贡献声明和真实投稿期刊选择。

在这些缺口补齐前，最准确的定位是“研究级方法学原型与内部论文稿”。
""",
        encoding="utf-8",
    )

    source_paths = [
        Path(window_ablation_path), Path(cross_anchor_baseline_path), Path(cross_anchor_extended_path),
        Path(openai_result_path), Path(operational_result_path), Path(legacy_manifest_path), Path(legacy_summary_path),
    ]
    generated_paths = [fig1, fig2, fig3, fig4, fig5, ext1, ext2, ext3, *tables_dir.glob("*.csv"), md_path, html_path, pdf_path, checklist_path, gap_path]
    manifest_path = destination / "reproducibility_manifest.json"
    manifest_payload = {
        "schema_version": "parking-slot-agent-v2-nature-style-report/1.0",
        "title": title,
        "report_status": "research_internal_not_submission_ready",
        "ground_truth_available": False,
        "claims": {
            "development_anchor_resolution": "9/22",
            "systematic_six_anchor_resolution": f"{pooled_resolved}/{pooled_total}",
            "openai_validation_errors": validation_errors,
            "selected_window_frames": 60,
            "window75_compute_saving_percent": saved,
        },
        "source_files": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in source_paths],
        "generated_files": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in generated_paths],
        "legacy_30_anchor_scope": legacy_summary["completion"],
        "nature_reporting_basis": [
            "https://www.nature.com/nature-portfolio/for-authors/write",
            "https://www.nature.com/nature/for-authors/initial-submission",
            "https://www.nature.com/documents/nr-reporting-summary-flat.pdf",
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "html": str(html_path),
        "pdf": str(pdf_path),
        "markdown": str(md_path),
        "figures": [str(path) for path, _ in legends],
        "extended_data": [str(ext1), str(ext2), str(ext3)],
        "tables": [str(path) for path in sorted(tables_dir.glob("*.csv"))],
        "manifest": str(manifest_path),
        "reporting_summary": str(checklist_path),
        "submission_gap_analysis": str(gap_path),
        "development_resolved": 9,
        "development_unknown": 22,
        "cross_anchor_resolved": pooled_resolved,
        "cross_anchor_unknown": pooled_total,
        "openai_validation_errors": validation_errors,
    }


__all__ = ["build_nature_style_report"]
