#!/usr/bin/env python3
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "vibe_demo"


def load_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = load_font(42, bold=True)
FONT_H = load_font(28, bold=True)
FONT = load_font(22)
FONT_SMALL = load_font(18)
FONT_MONO = load_font(20)


def fit_image(path: Path, box: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(box, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", box, "white")
    x = (box[0] - img.width) // 2
    y = (box[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font, fill, width_px: int, line_gap: int = 6):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width_px:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def card(draw, xywh, title, body, accent="#2563eb"):
    x, y, w, h = xywh
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill="#f8fafc", outline="#94a3b8", width=2)
    draw.rectangle((x, y, x + 10, y + h), fill=accent)
    draw.text((x + 24, y + 18), title, font=FONT_H, fill="#111827")
    draw_wrapped(draw, body, (x + 24, y + 60), FONT, "#334155", w - 48)


def make_storyboard():
    trace = json.loads((OUT / "agent_trace.json").read_text())
    metrics = json.loads((OUT / "metrics_before_after.json").read_text())["metrics"]
    history = trace["target_slot_history"]
    initial = history[0]["slot_id"]
    final = history[-1]["slot_id"]

    canvas = Image.new("RGB", (1800, 1220), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 38), "DASP-Park Demo In One Slide", font=FONT_TITLE, fill="#111827")
    draw.text(
        (60, 94),
        "Key story: sparse LiDAR first picks a risky slot; AI selects action/tool, updates belief, then checks ranking stability.",
        font=FONT,
        fill="#475569",
    )

    img_box = (500, 315)
    panels = [
        (
            "1. Initial input",
            OUT / "02_initial_observed_lidar.png",
            "Sparse LiDAR is incomplete. Hidden obstacle is not visible enough at first.",
        ),
        (
            "2. Initial choice",
            OUT / "00_slot_selection_scores.png",
            f"Known slot map + sparse belief selects {initial}. This is only an initial guess.",
        ),
        (
            "3. AI-advised inspection",
            OUT / "10_selected_regions_on_priority.png",
            "Rules generate legal actions/tools; OpenAI selects one plan under the sensing budget.",
        ),
        (
            "4. Evidence changes belief",
            OUT / "13_occupancy_after.png",
            f"Local tools update belief. Top3 changes, so target moves from {initial} toward {final}.",
        ),
        (
            "5. Before/after errors",
            OUT / "15_effectiveness_panel.png",
            "False-free errors fall and occupied IoU improves after local active perception.",
        ),
        (
            "6. AI briefing",
            OUT / "17_ai_agent_briefing.png",
            "AI explains: continue with the better target, but remain cautious because uncertainty remains.",
        ),
    ]
    coords = [(60, 150), (650, 150), (1240, 150), (60, 665), (650, 665), (1240, 665)]
    for idx, ((title, image_path, body), (x, y)) in enumerate(zip(panels, coords), start=1):
        draw.rounded_rectangle((x, y, x + 520, y + 455), radius=16, fill="#f8fafc", outline="#cbd5e1", width=2)
        draw.text((x + 18, y + 16), title, font=FONT_H, fill="#0f172a")
        thumb = fit_image(image_path, img_box)
        canvas.paste(thumb, (x + 10, y + 62))
        draw_wrapped(draw, body, (x + 20, y + 385), FONT_SMALL, "#334155", 480)
        if idx in {1, 2, 3, 4, 5}:
            ax = x + 535
            ay = y + 220
            draw.line((ax, ay, ax + 50, ay), fill="#2563eb", width=6)
            draw.polygon([(ax + 50, ay), (ax + 32, ay - 12), (ax + 32, ay + 12)], fill="#2563eb")

    metric_text = (
        f"Result: target {initial} -> {final} | "
        f"false-free {metrics['false_free_before']} -> {metrics['false_free_after']} | "
        f"IoU {metrics['occupied_iou_before']:.3f} -> {metrics['occupied_iou_after']:.3f} | "
        f"target unknown {metrics['target_slot_unknown_ratio_before']:.3f} -> {metrics['target_slot_unknown_ratio_after']:.3f}"
    )
    draw.rounded_rectangle((60, 1135, 1740, 1195), radius=16, fill="#eff6ff", outline="#60a5fa", width=2)
    draw.text((85, 1152), metric_text, font=FONT, fill="#1e3a8a")
    canvas.save(OUT / "18_quick_storyboard.png")


def make_frame(title: str, image_path: Path, bullets: list[str], footer: str) -> Image.Image:
    frame = Image.new("RGB", (1280, 720), "#ffffff")
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 1280, 86), fill="#0f172a")
    draw.text((40, 22), title, font=FONT_TITLE, fill="white")
    img = fit_image(image_path, (660, 520))
    frame.paste(img, (40, 120))
    x = 750
    y = 130
    for bullet in bullets:
        draw.ellipse((x, y + 8, x + 12, y + 20), fill="#2563eb")
        y = draw_wrapped(draw, bullet, (x + 24, y), FONT, "#111827", 455, line_gap=8) + 18
    draw.rounded_rectangle((740, 585, 1225, 660), radius=14, fill="#eff6ff", outline="#93c5fd", width=2)
    draw_wrapped(draw, footer, (760, 604), FONT_SMALL, "#1e3a8a", 440, line_gap=4)
    return frame


def make_gif():
    trace = json.loads((OUT / "agent_trace.json").read_text())
    metrics = json.loads((OUT / "metrics_before_after.json").read_text())["metrics"]
    history = trace["target_slot_history"]
    initial = history[0]["slot_id"]
    final = history[-1]["slot_id"]

    frame_specs = [
        (
            "1/6  Sparse LiDAR Input",
            OUT / "02_initial_observed_lidar.png",
            [
                "The car only sees sparse, incomplete LiDAR.",
                "Some slot evidence is missing at the start.",
                "Unknown space is not treated as free.",
            ],
            "Input: observed LiDAR only.",
        ),
        (
            "2/6  Initial Slot Choice",
            OUT / "00_slot_selection_scores.png",
            [
                f"Known slot map is available.",
                f"Initial belief selects {initial}.",
                "This is a hypothesis, not a commitment.",
            ],
            "Output: initial target slot.",
        ),
        (
            "3/6  AI-Advised Active Check",
            OUT / "10_selected_regions_on_priority.png",
            [
                "Rules generate legal candidate actions.",
                "OpenAI advisor selects action + tool.",
                "Validator allows only legal action/tool IDs.",
            ],
            "AI participates in action and tool selection.",
        ),
        (
            "4/6  Local Evidence Updates Belief",
            OUT / "13_occupancy_after.png",
            [
                "Tools inspect only selected regions.",
                "Local LiDAR evidence updates occupancy cells.",
                f"Target switches: {initial} -> {final}.",
            ],
            "No ground truth is used for updates.",
        ),
        (
            "5/6  Before / After",
            OUT / "15_effectiveness_panel.png",
            [
                f"False-free: {metrics['false_free_before']} -> {metrics['false_free_after']}.",
                f"Occupied IoU: {metrics['occupied_iou_before']:.3f} -> {metrics['occupied_iou_after']:.3f}.",
                f"Target unknown: {metrics['target_slot_unknown_ratio_before']:.3f} -> {metrics['target_slot_unknown_ratio_after']:.3f}.",
            ],
            "Output: measurable improvement.",
        ),
        (
            "6/6  AI Agent Briefing",
            OUT / "17_ai_agent_briefing.png",
            [
                "AI explains the trace and recommendation.",
                f"Use {final} as current best candidate.",
                "Remain cautious because uncertainty remains.",
            ],
            "Demo takeaway: inspect, update, switch.",
        ),
    ]
    frames = [make_frame(*spec) for spec in frame_specs]
    gif_path = OUT / "19_agent_flow.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=[1800, 1800, 2200, 2200, 2200, 2400],
        loop=0,
        optimize=True,
    )


def main():
    make_storyboard()
    make_gif()
    print(f"Saved {OUT / '18_quick_storyboard.png'}")
    print(f"Saved {OUT / '19_agent_flow.gif'}")


if __name__ == "__main__":
    main()
