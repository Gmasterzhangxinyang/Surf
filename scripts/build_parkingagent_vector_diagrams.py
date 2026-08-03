#!/usr/bin/env python3
"""Build deterministic, publication-ready vector diagrams for ParkingAgent."""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Group, Line, Path as RPath, Polygon, Rect, String
from reportlab.lib import colors


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "Nature_ParkingAgent_实验报告_20260728/paper_parkingagent_20260801"
FIG = PAPER / "figures"

NAVY = colors.HexColor("#132A44")
BLUE = colors.HexColor("#2B6CB0")
BLUE_BG = colors.HexColor("#EAF3FB")
PURPLE = colors.HexColor("#7251A6")
PURPLE_BG = colors.HexColor("#F1ECF8")
TEAL = colors.HexColor("#147D79")
TEAL_BG = colors.HexColor("#E7F5F3")
GREEN = colors.HexColor("#2F855A")
GREEN_BG = colors.HexColor("#ECF7F0")
RED = colors.HexColor("#B94A48")
RED_BG = colors.HexColor("#FBEDEC")
AMBER = colors.HexColor("#A56700")
AMBER_BG = colors.HexColor("#FFF5DD")
GREY = colors.HexColor("#607080")
LIGHT = colors.HexColor("#F7F9FB")
LINE = colors.HexColor("#91A0AE")
WHITE = colors.white


def _rect(d: Drawing | Group, x: float, y: float, w: float, h: float, *,
          fill=WHITE, stroke=LINE, radius: float = 7, sw: float = 1.15) -> None:
    d.add(Rect(x, y, w, h, rx=radius, ry=radius, fillColor=fill,
               strokeColor=stroke, strokeWidth=sw))


def _text(d: Drawing | Group, x: float, y: float, value: str, *, size: float = 10,
          color=NAVY, bold: bool = False, anchor: str = "start") -> None:
    d.add(String(x, y, value, fontName="Helvetica-Bold" if bold else "Helvetica",
                 fontSize=size, fillColor=color, textAnchor=anchor))


def _multiline(d: Drawing | Group, x: float, y: float, lines: list[str], *,
               size: float = 9, leading: float = 12, color=NAVY,
               bold_first: bool = False, anchor: str = "middle") -> None:
    for n, value in enumerate(lines):
        _text(d, x, y - n * leading, value, size=size, color=color,
              bold=(bold_first and n == 0), anchor=anchor)


def _arrow(d: Drawing | Group, x1: float, y1: float, x2: float, y2: float, *,
           color=NAVY, sw: float = 1.35, dashed: bool = False) -> None:
    line = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=sw)
    if dashed:
        line.strokeDashArray = [4, 3]
    d.add(line)
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    length, half = 7, 3.2
    bx, by = x2 - length * math.cos(angle), y2 - length * math.sin(angle)
    px, py = half * -math.sin(angle), half * math.cos(angle)
    d.add(Polygon([x2, y2, bx + px, by + py, bx - px, by - py],
                  fillColor=color, strokeColor=color))


def _elbow(d: Drawing | Group, points: list[tuple[float, float]], *, color=NAVY,
           sw: float = 1.35, dashed: bool = False) -> None:
    path = RPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.strokeColor = color
    path.strokeWidth = sw
    path.fillColor = None
    if dashed:
        path.strokeDashArray = [4, 3]
    d.add(path)
    _arrow(d, *points[-2], *points[-1], color=color, sw=0)


def _pill(d: Drawing | Group, x: float, y: float, w: float, label: str, *,
          fill=LIGHT, stroke=LINE, color=NAVY) -> None:
    _rect(d, x, y, w, 22, fill=fill, stroke=stroke, radius=11, sw=0.8)
    _text(d, x + w / 2, y + 7.3, label, size=8.2, color=color, bold=True, anchor="middle")


def system_drawing() -> Drawing:
    """Two-stage system diagram matching the implemented causal data flow."""
    d = Drawing(1000, 480)
    d.add(Rect(0, 0, 1000, 480, fillColor=WHITE, strokeColor=None))
    _text(d, 26, 449, "DASP-Park: causal occupancy estimation and active verification",
          size=18, bold=True)
    _text(d, 26, 429, "Only unresolved forward slots enter Part 2; camera legality is checked before observation.",
          size=9.5, color=GREY)

    # Stage bands.
    _rect(d, 20, 247, 960, 164, fill=colors.HexColor("#F5F9FD"), stroke=BLUE, radius=10, sw=1.4)
    _pill(d, 38, 376, 192, "PART 1 · GLOBAL ESTIMATE", fill=BLUE_BG, stroke=BLUE, color=BLUE)
    _rect(d, 20, 33, 960, 192, fill=colors.HexColor("#FAF8FC"), stroke=PURPLE, radius=10, sw=1.4)
    _pill(d, 38, 190, 242, "PART 2 · ACTIVE VERIFICATION", fill=PURPLE_BG, stroke=PURPLE, color=PURPLE)

    # Part 1 blocks.
    blocks = [
        (42, 284, 166, 72, BLUE_BG, BLUE, ["Causal input", "W=45, K=3 frames", "pose-corrected LiDAR"]),
        (246, 284, 166, 72, WHITE, BLUE, ["Map alignment", "transform returns", "into slot polygons"]),
        (450, 284, 180, 72, WHITE, BLUE, ["Evidence gates", "exclusive Free / Occupied", "otherwise Unknown + reason"]),
        (688, 284, 260, 72, GREEN_BG, GREEN, ["Part 1 output", "per-slot state + confidence", "evidence IDs + reason code"]),
    ]
    for x, y, w, h, fill, stroke, lines in blocks:
        _rect(d, x, y, w, h, fill=fill, stroke=stroke)
        _multiline(d, x + w / 2, y + 49, lines, size=9.2, leading=14, bold_first=True)
    for x1, x2 in ((208, 246), (412, 450), (630, 688)):
        _arrow(d, x1, 320, x2, 320, color=BLUE)

    # Part 1 branches.
    _pill(d, 687, 255, 126, "Free / Occupied", fill=GREEN_BG, stroke=GREEN, color=GREEN)
    _pill(d, 827, 255, 121, "Unknown only", fill=AMBER_BG, stroke=AMBER, color=AMBER)
    _arrow(d, 751, 284, 751, 277, color=GREEN)
    _arrow(d, 888, 284, 888, 277, color=AMBER)
    _elbow(d, [(751, 255), (751, 235), (946, 235), (946, 124)], color=GREEN)
    _text(d, 820, 237, "terminal system output", size=8.2, color=GREEN, bold=True, anchor="middle")

    # Part 2 blocks.
    p2 = [
        (44, 84, 184, 76, PURPLE_BG, PURPLE, ["Candidate gate", "Unknown ∧ d≤18 m", "forward heading ≤90°"]),
        (272, 84, 174, 76, WHITE, PURPLE, ["check_fov", "geometry only", "returns legal cameras"]),
        (490, 84, 190, 76, WHITE, PURPLE, ["Active-perception Agent", "plan → execute → observe", "camera first; LiDAR fallback"]),
        (724, 84, 170, 76, WHITE, PURPLE, ["Deterministic validator", "evidence citation + modality", "confidence ≥ threshold"]),
    ]
    for x, y, w, h, fill, stroke, lines in p2:
        _rect(d, x, y, w, h, fill=fill, stroke=stroke)
        _multiline(d, x + w / 2, y + 52, lines, size=9.0, leading=14, bold_first=True)
    _elbow(d, [(888, 255), (888, 210), (136, 210), (136, 160)], color=AMBER)
    for x1, x2 in ((228, 272), (446, 490), (680, 724)):
        _arrow(d, x1, 122, x2, 122, color=PURPLE)
    _arrow(d, 894, 122, 948, 122, color=PURPLE)
    _rect(d, 922, 63, 54, 118, fill=GREEN_BG, stroke=GREEN, radius=8)
    _multiline(d, 949, 145, ["FINAL", "F / O / U", "+ audit"], size=8.4, leading=16,
               color=GREEN, bold_first=True)

    # Explicit semantic notes.
    _pill(d, 43, 47, 183, "No reverse search", fill=LIGHT, stroke=LINE, color=GREY)
    _pill(d, 271, 47, 176, "FOV is not occupancy", fill=LIGHT, stroke=LINE, color=GREY)
    _pill(d, 490, 47, 190, "VLM chooses legal tools", fill=LIGHT, stroke=LINE, color=GREY)
    _pill(d, 724, 47, 170, "Fail closed to Unknown", fill=LIGHT, stroke=LINE, color=GREY)
    return d


def agent_drawing() -> Drawing:
    """Plan–execute–observe agent with legal tools and deterministic validation."""
    d = Drawing(1000, 480)
    d.add(Rect(0, 0, 1000, 480, fillColor=WHITE, strokeColor=None))
    _text(d, 26, 449, "Active-perception Agent: evidence acquisition under visibility constraints",
          size=18, bold=True)
    _text(d, 26, 429, "The center VLM chooses among legal tools; observations are structured and independently validated.",
          size=9.5, color=GREY)

    # Input and center reasoning state.
    _rect(d, 31, 264, 168, 104, fill=BLUE_BG, stroke=BLUE)
    _multiline(d, 115, 338, ["SlotCase", "Part 1 state + reason", "pose / slot geometry", "legal camera IDs"],
               size=9.2, leading=16, bold_first=True)
    _arrow(d, 199, 316, 299, 316, color=BLUE)

    _rect(d, 299, 239, 264, 154, fill=PURPLE_BG, stroke=PURPLE, radius=13, sw=1.6)
    _text(d, 431, 363, "CENTER VLM", size=13, color=PURPLE, bold=True, anchor="middle")
    _text(d, 431, 344, "PLAN", size=10, color=PURPLE, bold=True, anchor="middle")
    _multiline(d, 431, 321, ["Inspect current belief and available evidence", "Choose one legal observation tool", "Propose label, confidence, and evidence IDs"],
               size=9.0, leading=18, color=NAVY)
    _pill(d, 339, 253, 184, "re-plan after each observation", fill=WHITE, stroke=PURPLE, color=PURPLE)

    # Tool rail.
    _text(d, 699, 395, "EXECUTE · legal tool set", size=10.5, color=TEAL, bold=True, anchor="middle")
    tools = [
        (618, 329, "camera_context", "first when visible"),
        (618, 259, "camera_crop", "optional detail"),
        (618, 189, "lidar_detail", "fallback / corroboration"),
    ]
    for x, y, title, subtitle in tools:
        _rect(d, x, y, 192, 49, fill=TEAL_BG, stroke=TEAL)
        _text(d, x + 96, y + 28, title, size=9.5, color=TEAL, bold=True, anchor="middle")
        _text(d, x + 96, y + 12, subtitle, size=8.0, color=GREY, anchor="middle")
    _arrow(d, 563, 331, 618, 353, color=TEAL)
    _arrow(d, 563, 316, 618, 283, color=TEAL)
    _arrow(d, 563, 300, 618, 213, color=TEAL)

    # Observation record and feedback.
    _rect(d, 834, 214, 140, 142, fill=LIGHT, stroke=TEAL)
    _multiline(d, 904, 329, ["OBSERVE", "EvidenceRecord", "source + frame", "crop / returns", "visibility + quality"],
               size=8.8, leading=18, color=NAVY, bold_first=True)
    for y in (353, 283, 213):
        _arrow(d, 810, y, 834, 285, color=TEAL)
    _elbow(d, [(904, 214), (904, 165), (431, 165), (431, 239)], color=TEAL, dashed=True)
    _text(d, 672, 170, "structured observation feedback", size=8.3, color=TEAL, bold=True, anchor="middle")

    # Proposal and validator.
    _arrow(d, 431, 239, 431, 126, color=PURPLE)
    _pill(d, 339, 113, 184, "label + confidence + citations", fill=PURPLE_BG, stroke=PURPLE, color=PURPLE)
    _arrow(d, 523, 124, 617, 124, color=PURPLE)
    _rect(d, 617, 82, 192, 84, fill=WHITE, stroke=NAVY)
    _multiline(d, 713, 142, ["DETERMINISTIC VALIDATOR", "cited evidence exists", "modality is legal · p ≥ τ"],
               size=8.8, leading=17, bold_first=True)

    # Accept/reject terminals.
    _arrow(d, 809, 140, 892, 164, color=GREEN)
    _rect(d, 892, 141, 82, 49, fill=GREEN_BG, stroke=GREEN)
    _multiline(d, 933, 171, ["ACCEPT", "terminal F/O/U"], size=8.0, leading=14,
               color=GREEN, bold_first=True)
    _arrow(d, 809, 103, 892, 97, color=RED)
    _rect(d, 892, 74, 82, 46, fill=RED_BG, stroke=RED)
    _multiline(d, 933, 103, ["REJECT", "re-plan"], size=7.9, leading=14,
               color=RED, bold_first=True)
    _elbow(d, [(892, 97), (842, 97), (842, 68), (540, 68), (540, 239)], color=RED, dashed=True)
    _text(d, 680, 72, "tools remain", size=7.6, color=RED, bold=True, anchor="middle")
    _arrow(d, 933, 74, 933, 55, color=RED)
    _rect(d, 892, 12, 82, 43, fill=LIGHT, stroke=RED)
    _multiline(d, 933, 40, ["TURN LIMIT", "Unknown"], size=7.5, leading=14, color=RED, bold_first=True)

    # Audit strip.
    _rect(d, 31, 24, 778, 35, fill=colors.HexColor("#F1F5F9"), stroke=LINE, radius=5, sw=0.8)
    _text(d, 420, 37, "AUDIT TRAIL  ·  plan actions  ·  tool arguments  ·  EvidenceRecords  ·  validator decision",
          size=8.3, color=GREY, bold=True, anchor="middle")
    return d


def export_all() -> dict[str, str]:
    FIG.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for stem, drawing in (
        ("fig1_system_overview_vector", system_drawing()),
        ("fig2_active_perception_agent_vector", agent_drawing()),
    ):
        svg = FIG / f"{stem}.svg"
        pdf = FIG / f"{stem}.pdf"
        renderSVG.drawToFile(drawing, str(svg))
        renderPDF.drawToFile(drawing, str(pdf))
        outputs[f"{stem}_svg"] = str(svg)
        outputs[f"{stem}_pdf"] = str(pdf)
    return outputs


if __name__ == "__main__":
    import json
    print(json.dumps(export_all(), ensure_ascii=False, indent=2))
