#!/usr/bin/env python3
"""Draw the paper's implementation-faithful ParkingAgent architecture."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_B = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def ft(size: int, bold: bool = False):
    return ImageFont.truetype(str(FONT_B if bold else FONT), size=size)


def box(draw, xy, *, fill="#FFFFFF", outline="#CBD5E1", width=4, radius=22):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def label(draw, xy, text, size=32, color="#0F172A", bold=False, anchor="la"):
    draw.text(xy, text, font=ft(size, bold), fill=color, anchor=anchor)


def multiline(draw, xy, text, size=28, color="#334155", bold=False, spacing=12, anchor="la", align="left"):
    draw.multiline_text(xy, text, font=ft(size, bold), fill=color, spacing=spacing, anchor=anchor, align=align)


def arrow(draw, points, *, color="#2563EB", width=9, head=28):
    draw.line(points, fill=color, width=width, joint="curve")
    a, b = points[-2], points[-1]
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    left = (b[0] - head * math.cos(angle - 0.55), b[1] - head * math.sin(angle - 0.55))
    right = (b[0] - head * math.cos(angle + 0.55), b[1] - head * math.sin(angle + 0.55))
    draw.polygon([b, left, right], fill=color)


def pill(draw, xy, text, *, fill, color, size=25):
    draw.rounded_rectangle(xy, radius=18, fill=fill)
    label(draw, ((xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2), text, size, color, True, "mm")


def build(destination: Path) -> None:
    im = Image.new("RGB", (3200, 1700), "#F8FAFC")
    d = ImageDraw.Draw(im)

    label(d, (1600, 62), "ParkingAgent: From Causal Geometry to Evidence-Gated Agent Decisions", 50, "#0F172A", True, "ma")
    label(d, (1600, 118), "Part 1 preserves uncertainty; Part 2 autonomously acquires evidence under a deterministic safety contract", 29, "#475569", False, "ma")

    # Top pipeline: data -> Part 1 -> candidate gate -> structured case.
    top_y1, top_y2 = 190, 710
    stages = [
        (70, 610, "#E0F2FE", "#0284C7", "1  CAUSAL SCENE INPUT"),
        (690, 1350, "#ECFDF5", "#059669", "2  PART 1 · GEOMETRIC TRIAGE"),
        (1430, 2060, "#FFF7ED", "#EA580C", "3  FORWARD CANDIDATE GATE"),
        (2140, 3130, "#EDE9FE", "#7C3AED", "4  STRUCTURED SLOTCASE"),
    ]
    for x1, x2, fill, edge, title in stages:
        box(d, (x1, top_y1, x2, top_y2), fill="#FFFFFF", outline=edge, width=5)
        d.rounded_rectangle((x1 + 18, top_y1 + 18, x2 - 18, top_y1 + 82), radius=15, fill=fill)
        label(d, (x1 + 38, top_y1 + 50), title, 29, edge, True, "lm")
    for left, right in zip(stages, stages[1:]):
        arrow(d, [(left[1] + 12, 450), (right[0] - 14, 450)])

    # Stage 1.
    label(d, (115, 320), "Synchronized evidence", 34, "#0F172A", True)
    multiline(d, (115, 375), "• corrected ego pose\n• mapped slot polygons\n• LiDAR history  Hₜ(W,K)\n• causal left-camera frames", 30, spacing=19)
    pill(d, (115, 595, 550, 653), "GT excluded from inference", fill="#E2E8F0", color="#334155", size=25)

    # Stage 2.
    cols = [
        (315, 410, "FREE", "ray traversal · volume / ground coverage", "#DCFCE7", "#15803D"),
        (425, 520, "OCCUPIED", "target-core · height · temporal persistence", "#FEE2E2", "#B91C1C"),
        (535, 630, "VETO", "pillar / wall / boundary · adjacent ownership", "#FEF3C7", "#B45309"),
    ]
    for y1, y2, head, body, fill, edge in cols:
        box(d, (730, y1, 1310, y2), fill=fill, outline=edge, width=3, radius=15)
        label(d, (755, (y1 + y2) / 2), head, 23, edge, True, "lm")
        label(d, (950, (y1 + y2) / 2), body, 20, "#334155", False, "lm")
    label(d, (1020, 652), "Per-slot state + reason codes", 23, "#334155", True, "ma")
    pill(d, (745, 670, 910, 706), "Free", fill="#DCFCE7", color="#15803D", size=20)
    pill(d, (930, 670, 1110, 706), "Occupied", fill="#FEE2E2", color="#B91C1C", size=20)
    pill(d, (1130, 670, 1295, 706), "Unknown", fill="#E2E8F0", color="#475569", size=20)

    # Stage 3.
    label(d, (1745, 335), "Only Part1 Unknown continues", 31, "#C2410C", True, "ma")
    multiline(d, (1745, 405), "distance ≤ 18 m\n|relative bearing| ≤ 90°\nforward driving region only", 31, "#334155", spacing=16, anchor="ma", align="center")
    pill(d, (1500, 590, 1990, 652), "17 forward Unknown slots", fill="#FFEDD5", color="#C2410C", size=27)
    label(d, (1745, 678), "No reverse search · candidate geometry ≠ occupancy", 23, "#64748B", False, "ma")

    # Stage 4: check_fov is a deterministic pre-agent gate, not a model action.
    box(d, (2190, 315, 2485, 610), fill="#F5F3FF", outline="#7C3AED", width=3, radius=18)
    label(d, (2337, 360), "check_fov", 34, "#6D28D9", True, "ma")
    multiline(d, (2337, 420), "geometry only\nno pixel reading\nno F/O/U decision", 27, "#475569", spacing=14, anchor="ma", align="center")
    arrow(d, [(2500, 462), (2600, 462)], color="#7C3AED")
    box(d, (2615, 305, 3080, 620), fill="#FAF5FF", outline="#A78BFA", width=3, radius=18)
    label(d, (2847, 350), "Agent request envelope", 31, "#6D28D9", True, "ma")
    multiline(d, (2660, 405), "target slot + Part1 reason\navailable_tools\nobserved evidence IDs\nlegal JSON actions\nturn budget ≤ 6", 27, "#334155", spacing=13)
    pill(d, (2210, 640, 3055, 694), "Camera-visible ⇒ camera_context is the mandatory first tool", fill="#EDE9FE", color="#6D28D9", size=24)

    # Direct Part1 terminal branch.
    arrow(d, [(1030, 712), (1030, 760), (410, 760), (410, 835)], color="#0F766E", width=7, head=24)
    pill(d, (115, 815, 690, 875), "Strong Part1 Free / Occupied → system output", fill="#CCFBF1", color="#0F766E", size=25)

    # Bottom: the actual agent loop.
    box(d, (70, 905, 3130, 1545), fill="#FFFFFF", outline="#334155", width=5, radius=28)
    pill(d, (110, 930, 660, 992), "PART 2 · AUTONOMOUS PLAN–EXECUTE–OBSERVE", fill="#E2E8F0", color="#0F172A", size=27)

    # State / planner.
    box(d, (125, 1040, 610, 1365), fill="#F8FAFC", outline="#64748B", width=3)
    label(d, (367, 1090), "Agent state  zₖ", 34, "#0F172A", True, "ma")
    multiline(d, (175, 1150), "Part1 uncertainty reason\nlegal tools\nall prior observations\nlocalization hypothesis", 27, "#475569", spacing=16)
    box(d, (735, 1040, 1235, 1365), fill="#F5F3FF", outline="#7C3AED", width=5)
    label(d, (985, 1100), "CENTER VLM", 38, "#6D28D9", True, "ma")
    multiline(d, (985, 1170), "PLAN next evidence\nor propose Final(F/O/U)", 31, "#334155", spacing=18, anchor="ma", align="center")
    pill(d, (800, 1280, 1170, 1335), "autonomous choice", fill="#EDE9FE", color="#6D28D9", size=25)
    arrow(d, [(610, 1200), (720, 1200)], color="#7C3AED")

    # Tools selected by VLM.
    label(d, (1640, 1015), "EXECUTE one legal tool", 31, "#1D4ED8", True, "ma")
    tool_cards = [
        (1335, 1060, 1640, 1255, "camera_context", "primary pixels\ncyan target sheet", "#EFF6FF", "#2563EB"),
        (1660, 1060, 1950, 1255, "camera_crop", "optional zoom\nafter Camera", "#ECFEFF", "#0891B2"),
        (1970, 1060, 2265, 1255, "lidar_detail", "target-local\nfallback / ownership", "#ECFDF5", "#059669"),
    ]
    for x1, y1, x2, y2, head, body, fill, edge in tool_cards:
        box(d, (x1, y1, x2, y2), fill=fill, outline=edge, width=3, radius=18)
        label(d, ((x1 + x2) / 2, y1 + 50), head, 27, edge, True, "mm")
        multiline(d, ((x1 + x2) / 2, y1 + 105), body, 23, "#475569", spacing=10, anchor="ma", align="center")
    arrow(d, [(1235, 1140), (1318, 1140)], color="#2563EB")

    # Structured observation and return loop.
    box(d, (1390, 1320, 2210, 1480), fill="#F1F5F9", outline="#64748B", width=3)
    label(d, (1800, 1360), "OBSERVE · EvidenceRecord", 31, "#334155", True, "ma")
    label(d, (1800, 1422), "evidence_id · summary · metadata · audit image", 25, "#475569", False, "ma")
    arrow(d, [(2115, 1260), (2115, 1305)], color="#64748B", width=7, head=22)
    arrow(d, [(1390, 1400), (650, 1400), (650, 1280), (610, 1280)], color="#64748B", width=7, head=22)

    # Final validator and feedback.
    box(d, (2390, 1035, 3045, 1375), fill="#FFF7ED", outline="#EA580C", width=5)
    label(d, (2717, 1080), "DETERMINISTIC VALIDATOR", 31, "#C2410C", True, "ma")
    multiline(d, (2440, 1140), "observed evidence binding\nclass-specific modality gates\nconfidence threshold  τ=0.60\ncontradiction + identity checks", 27, "#334155", spacing=14)
    arrow(d, [(1235, 1280), (2370, 1280)], color="#EA580C", width=8, head=26)
    label(d, (1800, 1255), "Final proposal", 24, "#C2410C", True, "ma")
    arrow(d, [(2715, 1378), (2715, 1445)], color="#0F766E", width=8, head=26)
    pill(d, (2385, 1450, 3045, 1510), "ACCEPTED → Free / Occupied / Unknown", fill="#CCFBF1", color="#0F766E", size=27)
    arrow(d, [(2390, 1165), (2290, 1165), (2290, 1515), (985, 1515), (985, 1378)], color="#DC2626", width=7, head=23)
    label(d, (1640, 1490), "Rejected → validation error → re-plan; turn limit → Unknown", 23, "#B91C1C", True, "ma")

    # Audit footer.
    box(d, (210, 1580, 2990, 1660), fill="#0F172A", outline="#0F172A", width=2, radius=20)
    label(d, (1600, 1620), "AUDIT TRACE  ·  plans  ·  tool calls  ·  observations  ·  evidence bindings  ·  validation errors  ·  replay", 27, "#F8FAFC", True, "mm")

    destination.parent.mkdir(parents=True, exist_ok=True)
    im.save(destination, quality=96)


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "Nature_ParkingAgent_实验报告_20260728/paper_parkingagent_20260801/figures/fig1_system_architecture.png")
