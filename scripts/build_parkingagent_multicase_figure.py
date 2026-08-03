#!/usr/bin/env python3
"""Build a four-case qualitative comparison from real frame-6241 artifacts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "Nature_ParkingAgent_实验报告_20260728"
FRAME = REPORT / "frame6241_blind"
FIG = REPORT / "paper_parkingagent_20260801/figures"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_B = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def ft(size: int, bold: bool = False):
    return ImageFont.truetype(str(FONT_B if bold else FONT), size=size)


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    source = Image.open(path).convert("RGB")
    source.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(source, ((size[0] - source.width) // 2, (size[1] - source.height) // 2))
    return canvas


def centered(draw, xy, text, size, color="#0F172A", bold=False):
    draw.text(xy, text, font=ft(size, bold), fill=color, anchor="mm")


def build() -> Path:
    cases = [
        {
            "slot": "slot_0952",
            "panel": "(a)",
            "tag": "Visible Free",
            "gt": "GT Free",
            "pred": "System Free",
            "reason": "Camera-direct: clear target pavement",
            "edge": "#15803D",
            "fill": "#DCFCE7",
        },
        {
            "slot": "slot_0951",
            "panel": "(b)",
            "tag": "Visible Occupied",
            "gt": "GT Occupied",
            "pred": "System Occupied",
            "reason": "Camera candidate + LiDAR ownership check",
            "edge": "#B91C1C",
            "fill": "#FEE2E2",
        },
        {
            "slot": "slot_0954",
            "panel": "(c)",
            "tag": "Occluded Unknown",
            "gt": "GT Unknown",
            "pred": "System Unknown",
            "reason": "Occlusion remains after legal evidence",
            "edge": "#64748B",
            "fill": "#E2E8F0",
        },
        {
            "slot": "slot_0950",
            "panel": "(d)",
            "tag": "Failure Case",
            "gt": "GT Unknown",
            "pred": "System Occupied",
            "reason": "False resolution under occlusion",
            "edge": "#DC2626",
            "fill": "#FEF2F2",
        },
    ]
    # A 2x2 scientific plate keeps the real evidence large enough to audit.
    # No infographic title, card fill, legend, or decorative icon is added.
    width, height = 3300, 1760
    im = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(im)

    margin_x, margin_y, gap_x, gap_y = 42, 28, 36, 42
    panel_w = (width - 2 * margin_x - gap_x) // 2
    panel_h = (height - 2 * margin_y - gap_y) // 2
    image_gap = 20
    evidence_w = (panel_w - image_gap) // 2
    map_dir = FRAME / "gt_annotation_pack/review_pose_tuned_yaw_minus1/location_maps"
    cam_dir = FRAME / "gt_annotation_pack/camera_views_pose_tuned_yaw_minus1"

    for index, case in enumerate(cases):
        row, col = divmod(index, 2)
        x1 = margin_x + col * (panel_w + gap_x)
        y1 = margin_y + row * (panel_h + gap_y)
        x2 = x1 + panel_w

        result = f'{case["gt"].replace("GT ", "GT: ")}   |   {case["pred"].replace("System ", "Pred: ")}'
        result_color = "#B42318" if index == 3 else "#111111"
        d.text((x1, y1), f'{case["panel"]}  {case["slot"]}', font=ft(31, True), fill="#111111")
        d.text((x1 + 330, y1 + 3), result, font=ft(27, True), fill=result_color)
        d.text((x1, y1 + 48), case["reason"], font=ft(22), fill="#555555")
        d.line((x1, y1 + 82, x2, y1 + 82), fill="#8A8A8A", width=2)

        map_x, cam_x = x1, x1 + evidence_w + image_gap
        image_y = y1 + 126
        d.text((map_x, y1 + 93), "Target map", font=ft(23, True), fill="#222222")
        d.text((cam_x, y1 + 93), "Causal Camera evidence", font=ft(23, True), fill="#222222")

        map_panel = fit(map_dir / f'{case["slot"]}.png', (evidence_w, 620))
        cam_panel = fit(cam_dir / f'{case["slot"]}.jpg', (evidence_w, 620))
        im.paste(map_panel, (map_x, image_y))
        im.paste(cam_panel, (cam_x, image_y))
        d.rectangle((map_x, image_y, map_x + evidence_w, image_y + 620), outline="#C8C8C8", width=2)
        d.rectangle((cam_x, image_y, cam_x + evidence_w, image_y + 620), outline="#C8C8C8", width=2)

    destination = FIG / "fig3_multicase_evidence.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    im.save(destination, quality=96)
    return destination


if __name__ == "__main__":
    print(build())
