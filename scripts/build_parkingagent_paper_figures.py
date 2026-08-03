#!/usr/bin/env python3
"""Generate publication figures for the ParkingAgent paper."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_parkingagent_architecture_v2 import build as build_architecture_v2


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "Nature_ParkingAgent_实验报告_20260728"
PAPER = REPORT / "paper_parkingagent_20260801"
FIG = PAPER / "figures"
FRAME = REPORT / "frame6241_blind"

FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_B = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_B if bold else FONT), size=size)


def rr(draw: ImageDraw.ImageDraw, box, radius=24, fill="white", outline="#CBD5E1", width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw: ImageDraw.ImageDraw, xy, text: str, fnt, fill="#0F172A"):
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def wrap(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill="#334155", spacing=9):
    x1, y1, x2, _ = box
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = word if not line else line + " " + word
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= x2 - x1:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    y = y1
    for line in lines:
        draw.text((x1, y), line, font=fnt, fill=fill)
        y += fnt.size + spacing
    return y


def arrow(draw: ImageDraw.ImageDraw, a, b, color="#2563EB", width=7):
    draw.line([a, b], fill=color, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    length = 25
    for delta in (2.55, -2.55):
        p = (b[0] + length * math.cos(ang + delta), b[1] + length * math.sin(ang + delta))
        draw.line([b, p], fill=color, width=width)


def build_architecture() -> None:
    im = Image.new("RGB", (2800, 1420), "#F8FAFC")
    d = ImageDraw.Draw(im)
    centered(d, (1400, 64), "ParkingAgent: Evidence-Gated Multimodal Parking-Slot Reasoning", font(47, True))
    centered(d, (1400, 116), "Causal geometry proposes uncertainty; an autonomous agent resolves only evidence-supported cases", font(24), "#475569")

    xs = [(70, 205, 600, 1110), (690, 205, 1280, 1110), (1370, 205, 1870, 1110), (1960, 205, 2730, 1110)]
    colors = ["#E0F2FE", "#ECFDF5", "#FFF7ED", "#F3E8FF"]
    outlines = ["#0284C7", "#059669", "#EA580C", "#7C3AED"]
    titles = ["Causal Scene Memory", "Part 1: Geometric Triage", "Candidate Contract", "Part 2: Autonomous Agent"]
    tags = ["INPUT", "MAP-ALIGNED LIDAR", "FORWARD 180°", "PLAN · EXECUTE · OBSERVE"]
    for i, box in enumerate(xs):
        rr(d, box, fill="white", outline=outlines[i], width=4)
        d.rounded_rectangle((box[0] + 20, box[1] + 20, box[2] - 20, box[1] + 78), 15, fill=colors[i])
        d.text((box[0] + 38, box[1] + 34), tags[i], font=font(21, True), fill=outlines[i])
        centered(d, ((box[0] + box[2]) // 2, box[1] + 126), titles[i], font(30, True))
    for i in range(3):
        arrow(d, (xs[i][2] + 15, 650), (xs[i + 1][0] - 15, 650))

    # Panel 1: sensor memory.
    b = xs[0]
    rr(d, (b[0] + 35, 385, b[2] - 35, 640), 18, "#F0F9FF", "#7DD3FC", 3)
    d.text((b[0] + 58, 405), "LiDAR history  H_t(W,K)", font=font(25, True), fill="#0369A1")
    origin = (b[0] + 270, 575)
    for ang in np.linspace(-2.7, -0.45, 17):
        r = 120 + 30 * math.sin(ang * 4)
        end = (origin[0] + r * math.cos(ang), origin[1] + r * math.sin(ang))
        d.line([origin, end], fill="#38BDF8", width=3)
        d.ellipse((end[0] - 5, end[1] - 5, end[0] + 5, end[1] + 5), fill="#0EA5E9")
    d.polygon([(origin[0] - 20, origin[1] + 5), (origin[0] + 20, origin[1] + 5), (origin[0] + 14, origin[1] + 45), (origin[0] - 14, origin[1] + 45)], fill="#0F172A")
    rr(d, (b[0] + 35, 680, b[2] - 35, 920), 18, "#F8FAFC", "#CBD5E1", 3)
    d.text((b[0] + 58, 700), "Map + left-camera sequence", font=font(25, True), fill="#334155")
    for k in range(4):
        x = b[0] + 62 + k * 112
        d.rounded_rectangle((x, 760, x + 92, 865), 9, fill="#1E293B")
        d.polygon([(x + 12, 844), (x + 45, 790), (x + 82, 844)], outline="#22D3EE", width=5)
    d.text((b[0] + 58, 950), "Strictly causal: all evidence precedes t₀", font=font(22), fill="#475569")

    # Panel 2: Part1.
    b = xs[1]
    cards = [
        ("Free-space rays", "volume / ground coverage\nviewpoint diversity", "#DCFCE7", "#15803D"),
        ("Obstacle ownership", "core support / height\nvehicle-scale geometry", "#FEE2E2", "#B91C1C"),
        ("Static veto", "pillar · wall · boundary\npose perturbation", "#FEF3C7", "#B45309"),
    ]
    y = 380
    for title, body, fc, oc in cards:
        rr(d, (b[0] + 42, y, b[2] - 42, y + 170), 18, fc, oc, 3)
        d.text((b[0] + 65, y + 24), title, font=font(26, True), fill=oc)
        d.multiline_text((b[0] + 65, y + 75), body, font=font(21), fill="#334155", spacing=8)
        y += 195
    d.text((b[0] + 72, 985), "Free", font=font(26, True), fill="#15803D")
    d.text((b[0] + 230, 985), "Occupied", font=font(26, True), fill="#B91C1C")
    d.text((b[0] + 435, 985), "Unknown", font=font(26, True), fill="#64748B")

    # Panel 3: forward candidate contract.
    b = xs[2]
    center = ((b[0] + b[2]) // 2, 665)
    d.arc((center[0] - 190, center[1] - 190, center[0] + 190, center[1] + 190), 180, 360, fill="#F97316", width=13)
    d.line([(center[0] - 190, center[1]), (center[0] + 190, center[1])], fill="#F97316", width=6)
    d.polygon([(center[0] - 25, center[1] + 30), (center[0] + 25, center[1] + 30), (center[0] + 18, center[1] - 45), (center[0] - 18, center[1] - 45)], fill="#0F172A")
    for ang, col in [(-70, "#64748B"), (-30, "#2563EB"), (10, "#2563EB"), (48, "#2563EB"), (120, "#CBD5E1")]:
        rad = math.radians(ang - 90)
        x, y = center[0] + 145 * math.cos(rad), center[1] + 145 * math.sin(rad)
        d.rounded_rectangle((x - 34, y - 18, x + 34, y + 18), 6, fill=col)
    centered(d, (center[0], 405), "Unknown ∧ observed", font(25, True), "#C2410C")
    centered(d, (center[0], 455), "distance ≤ 18 m", font(23), "#475569")
    centered(d, (center[0], 500), "|relative bearing| ≤ 90°", font(23), "#475569")
    centered(d, (center[0], 930), "No reverse search", font(25, True), "#C2410C")
    centered(d, (center[0], 980), "GT never enters the queue", font(22), "#475569")

    # Panel 4: autonomous PEO and validator.
    b = xs[3]
    rr(d, (b[0] + 40, 365, b[2] - 40, 480), 18, "#EDE9FE", "#7C3AED", 3)
    centered(d, ((b[0] + b[2]) // 2, 408), "Pre-Agent check_fov", font(27, True), "#6D28D9")
    centered(d, ((b[0] + b[2]) // 2, 452), "tool availability only", font(21), "#475569")
    y = 535
    labels = [("PLAN", "select next evidence"), ("EXECUTE", "Camera · Crop · LiDAR"), ("OBSERVE", "structured evidence + IDs"), ("FINAL", "Free / Occupied / Unknown")]
    for idx, (head, body) in enumerate(labels):
        fill = ["#F5F3FF", "#EFF6FF", "#ECFDF5", "#FFF7ED"][idx]
        oc = ["#7C3AED", "#2563EB", "#059669", "#EA580C"][idx]
        rr(d, (b[0] + 55, y, b[2] - 55, y + 105), 16, fill, oc, 3)
        d.text((b[0] + 80, y + 19), head, font=font(24, True), fill=oc)
        d.text((b[0] + 230, y + 23), body, font=font(21), fill="#334155")
        if idx < len(labels) - 1:
            arrow(d, ((b[0] + b[2]) // 2, y + 106), ((b[0] + b[2]) // 2, y + 130), oc, 5)
        y += 135
    rr(d, (b[0] + 55, 1080, b[2] - 55, 1205), 16, "#0F172A", "#0F172A", 3)
    centered(d, ((b[0] + b[2]) // 2, 1123), "Deterministic evidence validator", font(25, True), "white")
    centered(d, ((b[0] + b[2]) // 2, 1168), "τ = 0.60 · binding · modality gates", font(20), "#CBD5E1")

    rr(d, (150, 1260, 2650, 1365), 24, "#0F172A", "#0F172A", 3)
    centered(d, (1400, 1302), "Auditable output: slot state, uncertainty reasons, tool trace, evidence binding, replay verification", font(28, True), "#F8FAFC")
    centered(d, (1400, 1342), "Safety invariant: invalid, occluded, contradictory, or weak evidence remains Unknown", font(22), "#93C5FD")
    FIG.mkdir(parents=True, exist_ok=True)
    im.save(FIG / "fig1_system_architecture.png", quality=95)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#F8FAFC")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def build_qualitative() -> None:
    paths = [
        FRAME / "gt_annotation_pack/review_pose_tuned_yaw_minus1/location_maps/slot_0951.png",
        FRAME / "gt_annotation_pack/camera_views_pose_tuned_yaw_minus1/slot_0951.jpg",
        FRAME / "pose_tuned_run/camera_first_final_full11_w45k3_gt_scope/live/media/slot_0951/lidar_detail/slot_0951_round_02_explained.png",
    ]
    im = Image.new("RGB", (2700, 980), "white")
    d = ImageDraw.Draw(im)
    titles = ["(a) Ego-centric map audit", "(b) Causal Camera contact sheet", "(c) Target-local LiDAR fallback"]
    for i, (path, title) in enumerate(zip(paths, titles)):
        x = 45 + i * 875
        rr(d, (x, 70, x + 825, 870), 18, "#F8FAFC", "#CBD5E1", 3)
        panel = fit_image(path, (785, 690))
        im.paste(panel, (x + 20, 125))
        centered(d, (x + 412, 98), title, font(25, True))
        caps = [
            "Target identity and forward geometry",
            "Cyan polygon marks the mapped target bay",
            "Free-space, boundary, and obstacle ownership",
        ]
        centered(d, (x + 412, 840), caps[i], font(20), "#475569")
    centered(d, (1350, 930), "slot_0951 · GT Occupied · final system Occupied · Camera-first → LiDAR verification", font(27, True), "#7C2D12")
    im.save(FIG / "fig2_qualitative_evidence.png", quality=95)


def build_ablation() -> None:
    payload = json.loads((FRAME / "locked_gt_experiments/part1_ablation/ablation_summary.json").read_text())
    ws, ks = [15, 30, 45, 60], [3, 5, 10, 15]
    lookup = {(r["history_span"], r["history_sample_count"]): r for r in payload["all"]}
    matrices = []
    for key in ("three_class_accuracy", "terminal_coverage", "unknown_false_resolution_rate"):
        matrices.append(np.array([[100 * lookup[(w, k)][key] for k in ks] for w in ws]))
    titles = ["Three-class accuracy (%)", "Terminal coverage (%)", "False resolution of Unknown (%)"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), constrained_layout=True)
    cmaps = ["Blues", "YlGn", "OrRd"]
    for ax, matrix, title, cmap in zip(axes, matrices, titles, cmaps):
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(50, float(matrix.max())))
        ax.set_xticks(range(len(ks)), ks)
        ax.set_yticks(range(len(ws)), ws)
        ax.set_xlabel("sample count K")
        ax.set_ylabel("history span W")
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i in range(len(ws)):
            for j in range(len(ks)):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > 0.6 * max(1, matrix.max()) else "#0F172A")
        ax.scatter([0], [2], marker="*", s=180, facecolor="#FACC15", edgecolor="#0F172A", linewidth=0.8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("History ablation on the revised locked GT (★: selected W45/K3)", fontsize=13, fontweight="bold")
    fig.savefig(FIG / "fig3_history_ablation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_model_ablation() -> None:
    data = json.loads((FRAME / "qwen_ablation/qwen_vlm_ablation_comparison.json").read_text())
    rows = data["models"]
    labels = ["OpenAI\n(reference)", "Qwen3-VL\n2B", "Qwen3.5\n0.8B"]
    accuracy = [100 * row["all14_accuracy"] for row in rows]
    visible = [100 * row["camera_visible_accuracy"] for row in rows]
    compliance = [100 * (row["camera_first_compliance"] or 0) for row in rows]
    x = np.arange(3)
    width = 0.23
    fig, ax = plt.subplots(figsize=(8.7, 4.2), constrained_layout=True)
    ax.bar(x - width, accuracy, width, label="14-GT accuracy", color="#2563EB")
    ax.bar(x, visible, width, label="Camera-visible accuracy", color="#10B981")
    ax.bar(x + width, compliance, width, label="Camera-first compliance", color="#F59E0B")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    ax.set_title("VLM capacity changes protocol adherence before final accuracy", fontweight="bold")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=8)
    fig.savefig(FIG / "fig4_vlm_ablation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    build_architecture_v2(FIG / "fig1_system_architecture.png")
    build_qualitative()
    build_ablation()
    build_model_ablation()
    print(json.dumps({"figures": [str(p) for p in sorted(FIG.glob("fig*.png"))]}, indent=2))


if __name__ == "__main__":
    main()
