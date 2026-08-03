#!/usr/bin/env python3
"""Render restrained, journal-style ParkingAgent method diagrams."""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Line, Path as RPath, Polygon, Rect, String
from reportlab.lib import colors


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "Nature_ParkingAgent_实验报告_20260728/paper_parkingagent_20260801"
FIG = PAPER / "figures"

INK = colors.HexColor("#17202A")
MID = colors.HexColor("#56616C")
RULE = colors.HexColor("#9AA4AE")
PALE = colors.HexColor("#F5F7F8")
BLUE = colors.HexColor("#245B82")
BLUE_PALE = colors.HexColor("#EEF3F7")
ORANGE = colors.HexColor("#A55B20")
GREEN = colors.HexColor("#2F684C")
RED = colors.HexColor("#963D3D")
WHITE = colors.white


def text(d: Drawing, x: float, y: float, value: str, *, size: float = 9,
         bold: bool = False, color=INK, anchor: str = "start") -> None:
    d.add(String(x, y, value, fontName="Helvetica-Bold" if bold else "Helvetica",
                 fontSize=size, fillColor=color, textAnchor=anchor))


def multiline(d: Drawing, x: float, y: float, values: list[str], *, size: float = 8.6,
              leading: float = 13, first_bold: bool = True, color=INK,
              anchor: str = "middle") -> None:
    for i, value in enumerate(values):
        text(d, x, y - i * leading, value, size=size,
             bold=first_bold and i == 0, color=color, anchor=anchor)


def box(d: Drawing, x: float, y: float, w: float, h: float, *, fill=WHITE,
        stroke=RULE, width: float = 0.9) -> None:
    d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=stroke, strokeWidth=width))


def module(d: Drawing, x: float, y: float, w: float, h: float, title: str,
           body: list[str], *, accent=BLUE, fill=WHITE) -> None:
    box(d, x, y, w, h, fill=fill, stroke=RULE)
    d.add(Rect(x, y + h - 5, w, 5, fillColor=accent, strokeColor=None))
    text(d, x + 10, y + h - 23, title, size=9.2, bold=True)
    for i, line in enumerate(body):
        text(d, x + 10, y + h - 40 - 13 * i, line, size=8.1, color=MID)


def arrow(d: Drawing, x1: float, y1: float, x2: float, y2: float, *,
          color=INK, width: float = 1.0, dashed: bool = False) -> None:
    line = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width)
    if dashed:
        line.strokeDashArray = [4, 3]
    d.add(line)
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    length, half = 6.2, 2.8
    bx = x2 - length * math.cos(angle)
    by = y2 - length * math.sin(angle)
    px = half * -math.sin(angle)
    py = half * math.cos(angle)
    d.add(Polygon([x2, y2, bx + px, by + py, bx - px, by - py],
                  fillColor=color, strokeColor=color))


def elbow(d: Drawing, points: list[tuple[float, float]], *, color=INK,
          width: float = 1.0, dashed: bool = False) -> None:
    path = RPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.strokeColor = color
    path.strokeWidth = width
    path.fillColor = None
    if dashed:
        path.strokeDashArray = [4, 3]
    d.add(path)
    arrow(d, *points[-2], *points[-1], color=color, width=0)


def stage_label(d: Drawing, y: float, marker: str, title: str, subtitle: str) -> None:
    text(d, 28, y, marker, size=12.5, bold=True, color=INK)
    text(d, 58, y, title, size=11.5, bold=True, color=INK)
    text(d, 58, y - 15, subtitle, size=8.2, color=MID)


def system_drawing() -> Drawing:
    """Two-stage causal pipeline without decorative infographic elements."""
    d = Drawing(1000, 410)
    d.add(Rect(0, 0, 1000, 410, fillColor=WHITE, strokeColor=None))

    stage_label(d, 385, "a", "Causal map-aligned occupancy estimation",
                "Part 1 operates on historical sensor evidence; GT is excluded from inference.")
    y, h = 252, 82
    specs = [
        (30, 145, "Input", ["LiDAR W=45, K=3", "corrected poses + map"]),
        (215, 145, "Registration", ["map-frame transform", "slot polygon association"]),
        (400, 155, "Evidence", ["ray traversal / obstacle", "ownership + structure veto"]),
        (595, 155, "Exclusive gates", ["G_F ∧ ¬G_O  |  G_O ∧ ¬G_F", "conflict → Unknown"]),
        (790, 180, "Per-slot state", ["Free / Occupied / Unknown", "confidence + reason + IDs"]),
    ]
    for x, w, title, body in specs:
        module(d, x, y, w, h, title, body, accent=BLUE,
               fill=BLUE_PALE if title == "Input" else WHITE)
    for a, b in ((175, 215), (360, 400), (555, 595), (750, 790)):
        arrow(d, a, y + h / 2, b, y + h / 2, color=INK)

    d.add(Line(28, 216, 972, 216, strokeColor=RULE, strokeWidth=0.55))
    stage_label(d, 193, "b", "Forward active verification",
                "Only unresolved slots in the forward 180° half-plane and within 18 m are eligible.")

    y2, h2 = 72, 78
    lower = [
        (70, 150, "Candidate", ["Unknown ∧ d ≤ 18 m", "|Δheading| ≤ 90°"]),
        (260, 145, "check_fov", ["geometry-only test", "returns legal cameras"]),
        (445, 160, "Agent policy", ["plan–execute–observe", "camera first; LiDAR fallback"]),
        (645, 155, "Validator", ["evidence + modality", "confidence ≥ τ"]),
        (840, 130, "System output", ["F / O / U", "complete audit trace"]),
    ]
    for x, w, title, body in lower:
        module(d, x, y2, w, h2, title, body,
               accent=GREEN if title == "System output" else BLUE,
               fill=PALE if title == "System output" else WHITE)
    for a, b in ((220, 260), (405, 445), (605, 645), (800, 840)):
        arrow(d, a, y2 + h2 / 2, b, y2 + h2 / 2, color=INK)

    # Unknown is the only Part 1 branch entering Part 2.
    elbow(d, [(880, 252), (880, 229), (145, 229), (145, 150)], color=ORANGE)
    text(d, 505, 232, "Unknown only", size=8.0, bold=True, color=ORANGE, anchor="middle")
    # Strong Part 1 decisions bypass active verification.
    elbow(d, [(940, 252), (940, 166), (905, 166), (905, 150)], color=GREEN)
    text(d, 951, 207, "F/O", size=7.8, bold=True, color=GREEN, anchor="middle")

    text(d, 260, 50, "FOV determines tool legality, not occupancy.", size=7.8, color=MID)
    text(d, 645, 50, "Rejected proposals remain Unknown unless further legal evidence is acquired.",
         size=7.8, color=MID)
    return d


def agent_drawing() -> Drawing:
    """Plan–execute–observe loop rendered as a compact methods schematic."""
    d = Drawing(1000, 410)
    d.add(Rect(0, 0, 1000, 410, fillColor=WHITE, strokeColor=None))
    stage_label(d, 385, "c", "Evidence-seeking controller and deterministic admission",
                "The VLM selects observations; the validator controls terminal state transitions.")

    module(d, 35, 205, 155, 95, "SlotCase", ["Part 1 state + reason", "geometry + legal tools"],
           accent=BLUE, fill=BLUE_PALE)
    module(d, 245, 195, 190, 115, "Belief / policy  πθ", ["PLAN one legal action", "or propose label + IDs", "update after observation"],
           accent=BLUE)

    # Tool interface uses one container with three non-decorative rows.
    box(d, 500, 185, 175, 135, fill=WHITE, stroke=RULE)
    d.add(Rect(500, 315, 175, 5, fillColor=BLUE, strokeColor=None))
    text(d, 510, 294, "Tool interface", size=9.2, bold=True)
    for y, name, note in (
        (266, "camera_context", "first if visible"),
        (236, "camera_crop", "optional detail"),
        (206, "lidar_detail", "fallback / support"),
    ):
        d.add(Line(510, y - 7, 665, y - 7, strokeColor=RULE, strokeWidth=0.45))
        text(d, 510, y, name, size=8.3, bold=True, color=INK)
        text(d, 665, y, note, size=7.6, color=MID, anchor="end")

    module(d, 740, 205, 220, 95, "EvidenceRecord", ["source / frame / crop or returns", "visibility + quality + evidence ID"],
           accent=BLUE, fill=PALE)

    arrow(d, 190, 252, 245, 252)
    arrow(d, 435, 252, 500, 252, color=BLUE)
    arrow(d, 675, 252, 740, 252, color=BLUE)
    text(d, 468, 260, "EXECUTE", size=7.5, bold=True, color=BLUE, anchor="middle")
    text(d, 708, 260, "OBSERVE", size=7.5, bold=True, color=BLUE, anchor="middle")

    # Observation feedback is visually separated above the main flow.
    elbow(d, [(850, 300), (850, 342), (340, 342), (340, 310)], color=BLUE, dashed=True)
    text(d, 595, 346, "structured observation updates belief", size=7.8,
         color=BLUE, anchor="middle")

    # Proposal and deterministic validator below the policy.
    arrow(d, 340, 195, 340, 136, color=INK)
    text(d, 350, 158, "label, confidence, evidence IDs", size=7.7, color=MID)
    module(d, 245, 45, 235, 82, "Deterministic validator",
           ["citation ⊆ observed evidence", "legal modality  ∧  p ≥ τ"], accent=INK)

    module(d, 565, 72, 135, 55, "ACCEPT", ["terminal F / O / U"],
           accent=GREEN, fill=PALE)
    arrow(d, 480, 99, 565, 99, color=GREEN)

    module(d, 565, 10, 135, 46, "REJECT", ["return validation error"],
           accent=RED, fill=WHITE)
    arrow(d, 480, 66, 565, 35, color=RED)
    elbow(d, [(565, 30), (215, 30), (215, 175), (285, 195)], color=RED, dashed=True)
    text(d, 365, 33, "legal tools and turns remain", size=7.5, color=RED, anchor="middle")

    module(d, 790, 10, 170, 46, "TURN LIMIT", ["fail closed → Unknown"],
           accent=RED, fill=PALE)
    arrow(d, 700, 33, 790, 33, color=RED)

    text(d, 790, 99, "accepted state is immutable", size=7.6, color=GREEN)
    text(d, 35, 13, "Audit: plan actions · tool arguments · EvidenceRecords · validator decisions",
         size=7.7, color=MID)
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
