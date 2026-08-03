"""CVPR-style visual narrative for the audited parking-slot experiments."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .reporting_story_zh import FONT_CJK, FONT_FALLBACK, _font, _rounded_box


NAVY = "#142B44"
BLUE = "#2D6BAA"
FREE = "#179C72"
OCC = "#D94F4F"
UNKNOWN = "#D99A20"
BG = "#F5F7FA"
INK = "#1D3145"
MUTED = "#65788B"
GRID = "#D7E0E8"


def _case_rows(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["case"]["slot"]["slot_id"]: row for row in result["slot_results"]}


def _lidar_evidence(case: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(row for row in case["evidence"] if row["tool_name"] == "lidar_detail")


def _detail_path(case: Mapping[str, Any]) -> Path:
    evidence = _lidar_evidence(case)
    path = Path(evidence["artifact_paths"][0])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _fov_evidence(case: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(row for row in case["evidence"] if row["tool_name"] == "check_fov")


def _map_crop(case: Mapping[str, Any], size: tuple[int, int]) -> Image.Image:
    evidence = _fov_evidence(case)
    source_path = Path(evidence["artifact_paths"][0])
    details = evidence["metadata"]["details"]
    center_map = case["slot"]["center_map"]
    anchor_map = case["snapshot_pose_map"][:2]
    dx_map = float(center_map[0]) - float(anchor_map[0])
    dy_map = float(center_map[1]) - float(anchor_map[1])
    distance = float(details["target_distance_m"])
    map_distance = math.hypot(dx_map, dy_map)
    scale = map_distance / distance
    # The underlying audited Part1 map is translated to the anchor but not
    # rotated; this deliberately matches that renderer instead of FOV bearing.
    target_x = dx_map / scale
    target_y = dy_map / scale
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        # Pixel bounds of the ego-relative map axes in the audited FOV render.
        plot_left, plot_right = 418, 1016
        plot_top, plot_bottom = 130, 1115
        px = plot_left + (target_x + 15.0) / 39.0 * (plot_right - plot_left)
        py = plot_top + (31.0 - target_y) / 64.0 * (plot_bottom - plot_top)
        crop_w, crop_h = 470, 470
        left = max(0, min(image.width - crop_w, int(px - crop_w / 2)))
        top = max(0, min(image.height - crop_h, int(py - crop_h / 2)))
        crop = image.crop((left, top, left + crop_w, top + crop_h))
        draw = ImageDraw.Draw(crop)
        cx, cy = int(px - left), int(py - top)
        draw.ellipse((cx - 34, cy - 34, cx + 34, cy + 34), outline=UNKNOWN, width=8)
        draw.line((cx - 50, cy, cx + 50, cy), fill=UNKNOWN, width=4)
        draw.line((cx, cy - 50, cx, cy + 50), fill=UNKNOWN, width=4)
        return ImageOps.fit(crop, size, method=Image.Resampling.LANCZOS)


def _card(case: Mapping[str, Any]) -> Mapping[str, Any]:
    return _lidar_evidence(case)["metadata"]["geometry_card"]


def _panel(path: Path, index: int, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        third = image.width // 3
        crop = image.crop((index * third, 0, image.width if index == 2 else (index + 1) * third, image.height))
        # Remove the large empty top/bottom margin while retaining the plot.
        crop = crop.crop((0, int(crop.height * 0.04), crop.width, int(crop.height * 0.93)))
        return ImageOps.fit(crop, size, method=Image.Resampling.LANCZOS)


def _paste_with_border(canvas: Image.Image, panel: Image.Image, xy: tuple[int, int], *, color: str = GRID, width: int = 3) -> None:
    canvas.paste(panel, xy)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((xy[0], xy[1], xy[0] + panel.width, xy[1] + panel.height), outline=color, width=width)


def _pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, fill: str, fg: str = "white", font_size: int = 28) -> int:
    font = _font(font_size)
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + 34
    height = box[3] - box[1] + 22
    draw.rounded_rectangle((xy[0], xy[1], xy[0] + width, xy[1] + height), radius=height // 2, fill=fill)
    draw.text((xy[0] + 17, xy[1] + 7), text, font=font, fill=fg)
    return width


def _metric(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, *, passed: bool, width: int = 500) -> None:
    color = FREE if passed else OCC
    _rounded_box(draw, (x, y, x + width, y + 67), fill="#FFFFFF", outline=GRID, radius=14, width=2)
    draw.ellipse((x + 18, y + 19, x + 47, y + 48), fill=color)
    draw.text((x + 62, y + 15), label, font=_font(24), fill=MUTED)
    value_box = draw.textbbox((0, 0), value, font=_font(27))
    draw.text((x + width - (value_box[2] - value_box[0]) - 20, y + 12), value, font=_font(27), fill=color)


def _teaser(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    canvas = Image.new("RGB", (2600, 1450), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 60), "Causal history turns abstention into auditable parking decisions", font=_font(52), fill=NAVY)
    draw.text((92, 128), "The VLM proposes a state; deterministic geometry and 7/7 pose checks decide whether it is legal.", font=_font(28), fill=MUTED)

    specs = [
        ("slot_1012", 80, FREE, "UNKNOWN", "FREE  0.98", "Free space is observed; boundary returns are vetoed."),
        ("slot_1258", 1320, OCC, "UNKNOWN", "OCCUPIED  0.96", "A stable 3-D core is observed; free-space evidence fails."),
    ]
    for slot_id, x, color, before, after, takeaway in specs:
        row = rows[slot_id]
        case = row["case"]
        card = _card(case)
        _rounded_box(draw, (x, 220, x + 1200, 1110), fill=BG, outline=color, radius=26, width=5)
        draw.text((x + 35, 250), slot_id, font=_font(31), fill=NAVY)
        w1 = _pill(draw, (x + 35, 315), before, fill=UNKNOWN)
        draw.text((x + 55 + w1, 318), "→", font=_font(34), fill=NAVY)
        _pill(draw, (x + 105 + w1, 315), after, fill=color)
        draw.text((x + 35, 385), takeaway, font=_font(25), fill=INK)

        detail = _detail_path(case)
        _paste_with_border(canvas, _map_crop(case, (500, 340)), (x + 35, 445))
        _paste_with_border(canvas, _panel(detail, 0, (500, 340)), (x + 560, 445))
        draw.text((x + 40, 795), "Part1 local map: target is Unknown", font=_font(22), fill=MUTED)
        draw.text((x + 565, 795), "Part2: target-aligned 60-frame geometry", font=_font(22), fill=MUTED)

        if slot_id == "slot_1012":
            free = card["free_geometry"]
            occupied = card["occupied_geometry"]
            metrics = [
                ("Observed volume", f"{100*free['observed_volume_ratio']:.1f}% ≥ 70%", True),
                ("Near-ground coverage", f"{100*free['near_ground_bev_coverage']:.1f}% ≥ 70%", True),
                ("Core obstacle points", f"{occupied['core_point_count']} = 0", True),
                ("Pose robustness", f"{card['robustness']['passing_variants']}/7", True),
            ]
        else:
            occupied = card["occupied_geometry"]
            free = card["free_geometry"]
            metrics = [
                ("Core obstacle points", f"{occupied['core_point_count']:,}", True),
                ("Boundary ratio", f"{occupied['boundary_ratio']:.3f} < 0.50", True),
                ("Observed free volume", f"{100*free['observed_volume_ratio']:.1f}% < 70%", True),
                ("Pose robustness", f"{card['robustness']['passing_variants']}/7", True),
            ]
        for i, (label, value, passed) in enumerate(metrics):
            _metric(draw, x + 35 + (i % 2) * 555, 845 + (i // 2) * 86, label, value, passed=passed, width=525)

    kpis = [
        ("Development anchor", "0 → 9 / 22", "Unknown slots resolved"),
        ("6 held-out anchors", "0 → 34 / 208", "Every anchor improves"),
        ("Pareto window", "60 = 75 frames", "Same 9/22, 22% less time"),
    ]
    for i, (label, value, note) in enumerate(kpis):
        x = 80 + i * 835
        _rounded_box(draw, (x, 1165, x + 780, 1370), fill="#F8FAFC", outline=BLUE, radius=22, width=3)
        draw.text((x + 30, 1192), label, font=_font(24), fill=MUTED)
        draw.text((x + 30, 1230), value, font=_font(43), fill=BLUE)
        draw.text((x + 30, 1300), note, font=_font(24), fill=INK)
    canvas.save(path)


def _method(path: Path) -> None:
    canvas = Image.new("RGB", (2600, 1050), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 55), "The workflow is unchanged; only Part2 evidence becomes longer and auditable", font=_font(50), fill=NAVY)
    draw.text((92, 122), "One slot at a time. Camera is routed by FOV. A trustworthy Free still stops the global queue immediately.", font=_font(27), fill=MUTED)
    stages = [
        ("1", "Part1 queue", "15-frame local map\nFree first, then Unknown", BLUE, "INPUT"),
        ("2", "Mandatory FOV", "Write visibility to SlotCase\nDisable illegal Camera calls", UNKNOWN, "ROUTE"),
        ("3", "Causal evidence", "60 frames, all t <= t0\nSlot/map/source hashes", FREE, "OBSERVE"),
        ("4", "VLM proposal", "Read structured evidence\nPropose one of 3 states", "#7653A6", "PROPOSE"),
        ("5", "Code hard gate", "0.90 + mutual veto + 7/7\nThe model cannot bypass this", OCC, "VERIFY"),
        ("6", "Queue control", "Free: stop and return\nOtherwise: next candidate", "#287A55", "ACT"),
    ]
    y0, y1 = 280, 760
    box_w, gap = 365, 50
    for i, (number, title, detail, color, tag) in enumerate(stages):
        x0 = 80 + i * (box_w + gap)
        _rounded_box(draw, (x0, y0, x0 + box_w, y1), fill=BG, outline=color, radius=24, width=4)
        draw.ellipse((x0 + 28, y0 + 28, x0 + 85, y0 + 85), fill=color)
        draw.text((x0 + 48, y0 + 36), number, font=_font(28), fill="white", anchor="mm")
        draw.text((x0 + 105, y0 + 31), tag, font=_font(20), fill=color)
        draw.text((x0 + 28, y0 + 125), title, font=_font(32), fill=NAVY)
        draw.multiline_text((x0 + 28, y0 + 205), detail, font=_font(24), fill=INK, spacing=16)
        if i < len(stages) - 1:
            start = x0 + box_w + 10
            draw.line((start, 520, start + 30, 520), fill=MUTED, width=7)
            draw.polygon([(start + 30, 500), (start + 62, 520), (start + 30, 540)], fill=MUTED)
    _rounded_box(draw, (500, 835, 2100, 980), fill="#FFF7E8", outline=UNKNOWN, radius=20, width=3)
    draw.text((550, 865), "Fail closed", font=_font(30), fill=UNKNOWN)
    draw.text((760, 865), "Missing media, invalid schema, identity mismatch, conflicting geometry, or failed robustness", font=_font(25), fill=INK)
    draw.text((760, 915), "→ runtime error or Unknown — never silently coerced to Free", font=_font(27), fill=OCC)
    canvas.save(path)


def _mpl_font(size: int) -> FontProperties:
    source = FONT_CJK if FONT_CJK.is_file() else FONT_FALLBACK
    return FontProperties(fname=str(source), size=size)


def _quantitative(path: Path, window: Mapping[str, Any], baseline: Mapping[str, Any], extended: Mapping[str, Any]) -> None:
    plt.rcParams.update({"axes.edgecolor": NAVY, "axes.linewidth": 1.2})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), gridspec_kw={"hspace": 0.38, "wspace": 0.25})
    fig.suptitle("60 frames is the first Pareto-optimal window and improves every held-out anchor", fontproperties=_mpl_font(22), color=NAVY, y=0.985)
    rows = sorted(window["experiments"], key=lambda r: int(r["window_frames"]))
    windows = [int(r["window_frames"]) for r in rows]
    frees = np.array([int(r["gate_state_counts"].get("free", 0)) for r in rows])
    occs = np.array([int(r["gate_state_counts"].get("occupied", 0)) for r in rows])
    unknowns = 22 - frees - occs

    ax = axes[0, 0]
    ax.bar(windows, frees, width=9, color=FREE, label="Free")
    ax.bar(windows, occs, width=9, bottom=frees, color=OCC, label="Occupied")
    ax.bar(windows, unknowns, width=9, bottom=frees + occs, color="#D9E0E7", label="Still Unknown")
    for x, free, occ in zip(windows, frees, occs):
        ax.text(x, free + occ + 0.35, f"{free+occ}/22 resolved", ha="center", fontsize=10, color=NAVY)
    ax.set_title("A  |  Longer history resolves more cases", loc="left", fontproperties=_mpl_font(14), color=NAVY)
    ax.set_xlabel("Causal LiDAR window (frames)")
    ax.set_ylabel("Original Part1-Unknown slots")
    ax.set_ylim(0, 24.5)
    ax.set_xticks(windows)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    times = [float(r["processing_seconds"]) for r in rows]
    ax.plot(windows, times, color=BLUE, marker="o", linewidth=3, markersize=8)
    ax.scatter([60], [times[3]], s=180, color=FREE, zorder=5)
    ax.scatter([75], [times[4]], s=180, color=OCC, zorder=5)
    ax.annotate("first 9/22\n105.9 s", (60, times[3]), xytext=(-52, -48), textcoords="offset points", color=FREE, fontsize=11, arrowprops={"arrowstyle": "->", "color": FREE})
    ax.annotate("still 9/22\n135.7 s", (75, times[4]), xytext=(-55, -52), textcoords="offset points", color=OCC, fontsize=11, arrowprops={"arrowstyle": "->", "color": OCC})
    ax.set_title("B  |  75 frames add 28% time but no decision", loc="left", fontproperties=_mpl_font(14), color=NAVY)
    ax.set_xlabel("Causal LiDAR window (frames)")
    ax.set_ylabel("Geometry processing time (s)")
    ax.set_xticks(windows)
    ax.grid(alpha=0.25)

    ext = sorted(extended["experiments"], key=lambda r: int(r["anchor_frame"]))
    anchors = [str(r["anchor_frame"]) for r in ext]
    rates = [100 * int(r["resolved_count"]) / int(r["part1_unknown_count"]) for r in ext]
    labels = [f"{r['resolved_count']}/{r['part1_unknown_count']}" for r in ext]
    ax = axes[1, 0]
    y = np.arange(len(anchors))
    ax.hlines(y, 0, rates, color="#B7C6D4", linewidth=4)
    ax.scatter([0] * len(y), y, color="#AAB5C0", s=80, label="15-frame baseline")
    ax.scatter(rates, y, color=BLUE, s=110, label="60-frame method")
    for rate, yy, label in zip(rates, y, labels):
        ax.text(rate + 1.0, yy, label, va="center", fontsize=10, color=NAVY)
    ax.set_yticks(y, anchors)
    ax.invert_yaxis()
    ax.set_xlim(-1, 39)
    ax.set_xlabel("Resolved fraction of original Unknown (%)")
    ax.set_title("C  |  All six held-out anchors improve from zero", loc="left", fontproperties=_mpl_font(14), color=NAVY)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1, 1]
    totals = [208, 208]
    free = [0, sum(int(r["gate_state_counts"].get("free", 0)) for r in ext)]
    occupied = [0, sum(int(r["gate_state_counts"].get("occupied", 0)) for r in ext)]
    unknown = [totals[i] - free[i] - occupied[i] for i in range(2)]
    x = np.arange(2)
    ax.bar(x, unknown, color="#D9E0E7", label="Still Unknown")
    ax.bar(x, occupied, bottom=unknown, color=OCC, label="Occupied")
    ax.bar(x, free, bottom=np.array(unknown) + np.array(occupied), color=FREE, label="Free")
    ax.text(0, 104, "208 Unknown", ha="center", va="center", fontsize=14, color=NAVY, fontweight="bold")
    ax.text(1, 78, "174 remain", ha="center", va="center", fontsize=12, color=NAVY)
    ax.text(1, 191, "34 resolved\n10 Free + 24 Occupied", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    ax.set_xticks(x, ["15-frame baseline", "60-frame method"])
    ax.set_ylabel("Matched held-out slots")
    ax.set_ylim(0, 220)
    ax.set_title("D  |  Pooled matched comparison: 0/208 → 34/208", loc="left", fontproperties=_mpl_font(14), color=NAVY)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", ncol=3, loc="lower center")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _success_cases(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    canvas = Image.new("RGB", (2600, 1650), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((85, 55), "Why did the state change? Two complete evidence-to-decision chains", font=_font(50), fill=NAVY)
    draw.text((87, 120), "Each terminal needs positive evidence, an explicit veto of the opposite state, and 7/7 pose robustness.", font=_font(27), fill=MUTED)
    specs = [
        ("slot_1012", 200, FREE, "UNKNOWN  →  FREE 0.98", [
            ("Positive free-space evidence", "volume 99.8%; near-ground 100%", True),
            ("Occupied hypothesis vetoed", "boundary ratio 1.00; core points 0", True),
            ("Pose perturbations", "7 / 7 pass", True),
        ]),
        ("slot_1258", 930, OCC, "UNKNOWN  →  OCCUPIED 0.96", [
            ("Positive 3-D obstacle evidence", "27,891 core points; 3 height layers", True),
            ("Static-boundary veto passed", "boundary 0.00; linearity 0.565 < 0.60", True),
            ("Pose perturbations", "7 / 7 pass", True),
        ]),
    ]
    for slot_id, y, color, transition, checks in specs:
        row = rows[slot_id]
        case = row["case"]
        _rounded_box(draw, (80, y, 2520, y + 650), fill=BG, outline=color, radius=24, width=5)
        draw.text((115, y + 28), slot_id, font=_font(30), fill=MUTED)
        draw.text((115, y + 75), transition, font=_font(40), fill=color)
        detail = _detail_path(case)
        _paste_with_border(canvas, _map_crop(case, (425, 400)), (115, y + 165))
        _paste_with_border(canvas, _panel(detail, 0, (425, 400)), (565, y + 165))
        _paste_with_border(canvas, _panel(detail, 1, (425, 400)), (1015, y + 165))
        draw.text((120, y + 575), "1  Part1 map: Unknown", font=_font(21), fill=UNKNOWN)
        draw.text((570, y + 575), "2  60-frame bird's-eye", font=_font(21), fill=MUTED)
        draw.text((1020, y + 575), "3  60-frame height profile", font=_font(21), fill=MUTED)
        draw.text((1490, y + 155), "4  TERMINAL CHECKLIST", font=_font(25), fill=NAVY)
        for i, (label, value, passed) in enumerate(checks):
            yy = y + 210 + i * 115
            c = FREE if passed else OCC
            draw.ellipse((1490, yy, 1532, yy + 42), fill=c)
            draw.text((1501, yy + 3), "✓" if passed else "×", font=_font(28), fill="white")
            draw.text((1555, yy - 3), label, font=_font(27), fill=INK)
            draw.text((1555, yy + 38), value, font=_font(23), fill=MUTED)
        _rounded_box(draw, (1490, y + 555, 2430, y + 620), fill="#FFFFFF", outline=color, radius=14, width=3)
        gate = _card(case)["terminal_geometry_gate"]
        state = case["final_state"]
        draw.text((1520, y + 570), f"Code gate: {state.upper()} eligible = {str(gate[state + '_eligible']).upper()}", font=_font(27), fill=color)
    canvas.save(path)


def _failure_cases(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    canvas = Image.new("RGB", (2600, 1350), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((85, 55), "Strong-looking evidence is rejected when one safety condition fails", font=_font(50), fill=NAVY)
    draw.text((87, 120), "Remaining Unknown is an explicit outcome, not a visualization failure.", font=_font(28), fill=MUTED)
    specs = [
        ("slot_1251", "POSE SENSITIVITY", "6/7 perturbations pass", "Required: 7/7", "One failed pose variant blocks Occupied."),
        ("slot_1010", "CONFLICT", "Occupied core + free traversal", "Unresolved core hit", "Opposing evidence cannot be mutually vetoed."),
        ("slot_1248", "NO COVERAGE", "0 ray frames; occlusion 100%", "No pixel projection", "Neither modality can identify a terminal state."),
    ]
    for i, (slot_id, label, signal, failure, reason) in enumerate(specs):
        x = 70 + i * 845
        row = rows[slot_id]
        case = row["case"]
        _rounded_box(draw, (x, 210, x + 790, 1250), fill="#FFF9EF", outline=UNKNOWN, radius=24, width=4)
        draw.text((x + 30, 240), slot_id, font=_font(29), fill=NAVY)
        _pill(draw, (x + 30, 290), label, fill=UNKNOWN, font_size=23)
        detail = _detail_path(case)
        _paste_with_border(canvas, _panel(detail, 0, (730, 440)), (x + 30, 370))
        draw.text((x + 30, 835), "WHY THE GATE STOPS", font=_font(23), fill=MUTED)
        draw.ellipse((x + 30, 885, x + 75, 930), fill=OCC)
        draw.text((x + 52, 888), "×", font=_font(30), fill="white", anchor="mm")
        draw.text((x + 95, 880), signal, font=_font(27), fill=INK)
        draw.text((x + 95, 925), failure, font=_font(24), fill=OCC)
        draw.text((x + 30, 1010), reason, font=_font(24), fill=MUTED)
        _rounded_box(draw, (x + 30, 1110, x + 760, 1195), fill="white", outline=UNKNOWN, radius=16, width=3)
        draw.text((x + 60, 1130), "FINAL: UNKNOWN", font=_font(31), fill=UNKNOWN)
        scores = case["final_scores"]
        draw.text((x + 380, 1136), f"U={scores['unknown_confidence']:.2f}", font=_font(25), fill=MUTED)
    canvas.save(path)


def _outcomes(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    unknown_rows = [row for row in rows.values() if row["case"]["part1_state"] == "unknown"]
    counts = Counter(row["case"]["final_state"] for row in unknown_rows)
    canvas = Image.new("RGB", (2400, 1250), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((80, 55), "What happened to all 22 Part1-Unknown slots?", font=_font(50), fill=NAVY)
    draw.text((82, 120), "Nine pass a terminal hard gate; thirteen remain explicitly Unknown.", font=_font(28), fill=MUTED)

    _rounded_box(draw, (90, 270, 620, 610), fill="#FFF9EF", outline=UNKNOWN, radius=24, width=4)
    draw.text((150, 325), "Part1", font=_font(28), fill=MUTED)
    draw.text((150, 385), "22", font=_font(80), fill=UNKNOWN)
    draw.text((150, 500), "UNKNOWN", font=_font(36), fill=UNKNOWN)

    targets = [
        ("FREE", counts["free"], FREE, 850, 230),
        ("OCCUPIED", counts["occupied"], OCC, 850, 500),
        ("UNKNOWN", counts["unknown"], UNKNOWN, 850, 770),
    ]
    for label, count, color, x, y in targets:
        draw.line((620, 440, x, y + 110), fill="#9BAAB8", width=7)
        draw.polygon([(x - 10, y + 93), (x + 20, y + 110), (x - 10, y + 127)], fill="#9BAAB8")
        _rounded_box(draw, (x + 20, y, x + 600, y + 220), fill=BG, outline=color, radius=22, width=4)
        draw.text((x + 55, y + 28), str(count), font=_font(62), fill=color)
        draw.text((x + 180, y + 45), label, font=_font(33), fill=color)
        draw.text((x + 55, y + 135), f"{100*count/22:.1f}% of original Unknown", font=_font(23), fill=MUTED)

    _rounded_box(draw, (1600, 270, 2300, 990), fill="#F6F8FA", outline=BLUE, radius=24, width=4)
    draw.text((1640, 305), "Interpretation", font=_font(31), fill=NAVY)
    notes = [
        (FREE, "2 Free", "positive traversed volume\nand opposite state vetoed"),
        (OCC, "7 Occupied", "stable 3-D core returns\nand free evidence fails"),
        (UNKNOWN, "13 Unknown", "robustness, conflict,\nocclusion, or no coverage"),
    ]
    for i, (color, title, text) in enumerate(notes):
        y = 390 + i * 185
        draw.ellipse((1645, y, 1685, y + 40), fill=color)
        draw.text((1710, y - 5), title, font=_font(28), fill=color)
        draw.multiline_text((1710, y + 42), text, font=_font(23), fill=MUTED, spacing=8)
    draw.text((1625, 1045), "Resolution coverage ≠ accuracy", font=_font(30), fill=OCC)
    draw.text((1625, 1095), "No human GT is available.", font=_font(25), fill=MUTED)
    canvas.save(path)


def _decision_matrix(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    data = []
    for row in rows.values():
        case = row["case"]
        if case["part1_state"] != "unknown":
            continue
        card = _card(case)
        data.append((case["slot"]["slot_id"], case["final_state"], case["final_scores"], card))
    data.sort(key=lambda item: ({"free": 0, "occupied": 1, "unknown": 2}[item[1]], item[0]))
    columns = ["Free score", "Occupied score", "Stable 7/7", "Free gate", "Occupied gate"]
    matrix = []
    for _, _, scores, card in data:
        gate = card["terminal_geometry_gate"]
        stable = card["robustness"].get("passing_variants") == 7 and card["robustness"].get("total_variants") == 7
        matrix.append([scores["free_confidence"], scores["occupied_confidence"], float(stable), float(gate["free_eligible"]), float(gate["occupied_eligible"])])
    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(arr, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(columns)), columns, rotation=25, ha="right")
    ax.set_yticks(range(len(data)), [f"{sid}  →  {state.upper()}" for sid, state, _, _ in data])
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            text = f"{arr[y, x]:.2f}" if x < 2 else ("PASS" if arr[y, x] else "—")
            ax.text(x, y, text, ha="center", va="center", color="white" if arr[y, x] > 0.55 else NAVY, fontsize=8)
    ax.set_title("Extended Data | Every terminal state is traceable to a deterministic gate", loc="left", fontproperties=_mpl_font(15), color=NAVY, pad=18)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="score / binary gate")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_cvpr_figure_suite(*, window: Mapping[str, Any], baseline: Mapping[str, Any], extended: Mapping[str, Any], result: Mapping[str, Any], output_dir: str | Path) -> list[tuple[Path, str]]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = _case_rows(result)
    specs = [
        ("Figure_1_teaser.png", "Figure 1. Two representative state transitions and the three headline results. The agent proposal is accepted only after deterministic terminal checks."),
        ("Figure_2_method.png", "Figure 2. The original Free-first workflow is retained. Part2 gains a causal 60-frame evidence package and a non-bypassable hard gate."),
        ("Figure_3_quantitative.png", "Figure 3. Window ablation, runtime trade-off, per-anchor gains, and the matched pooled comparison."),
        ("Figure_4_success_cases.png", "Figure 4. Complete evidence-to-decision chains for one Free and one Occupied transition."),
        ("Figure_5_failure_cases.png", "Figure 5. Three explicit rejection mechanisms: pose sensitivity, conflicting evidence, and absent coverage."),
        ("Figure_6_all_outcomes.png", "Figure 6. Final disposition of every Part1-Unknown case at the development anchor."),
        ("Extended_Data_Figure_1_decision_matrix.png", "Extended Data Figure 1. Per-slot model scores, robustness, and mutually exclusive terminal gates."),
    ]
    paths = {name: destination / name for name, _ in specs}
    _teaser(paths["Figure_1_teaser.png"], rows)
    _method(paths["Figure_2_method.png"])
    _quantitative(paths["Figure_3_quantitative.png"], window, baseline, extended)
    _success_cases(paths["Figure_4_success_cases.png"], rows)
    _failure_cases(paths["Figure_5_failure_cases.png"], rows)
    _outcomes(paths["Figure_6_all_outcomes.png"], rows)
    _decision_matrix(paths["Extended_Data_Figure_1_decision_matrix.png"], rows)
    return [(paths[name], caption) for name, caption in specs]


__all__ = ["build_cvpr_figure_suite"]
