#!/usr/bin/env python3
"""Render a map-only report for slot-aligned accumulation outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import base64
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon as MplPolygon


DEFAULT_INPUT = Path("outputs/slot_aligned_accumulation_stride4_w52_eps025_ms8")
DEFAULT_SLOT_DATABASE = Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json")


STATE_STYLE = {
    "accumulated_vehicle_core_supported": ("#dc2626", "#7f1d1d", 0.78, "core-supported"),
    "accumulated_adjacent_conflict": ("#f97316", "#9a3412", 0.70, "adjacent conflict"),
    "accumulated_boundary_conflict": ("#a855f7", "#581c87", 0.68, "boundary conflict"),
    "accumulated_static_like": ("#64748b", "#334155", 0.45, "static-like"),
    "accumulated_no_vehicle_evidence": ("#d1d5db", "#94a3b8", 0.22, "no vehicle evidence"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build map-only report for slot-aligned accumulation")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--slot-database", type=Path, default=DEFAULT_SLOT_DATABASE)
    parser.add_argument("--output-name", default="slot_aligned_global_map_report.html")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def i(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def draw_map(path: Path, slots: dict[str, dict[str, Any]], rows: list[dict[str, str]], zoom: bool) -> None:
    row_by_slot = {str(row["slot_id"]): row for row in rows}
    all_pts = np.vstack([np.asarray(slot["polygon_map"], dtype=np.float64) for slot in slots.values()])
    fig, ax = plt.subplots(figsize=(13, 10), dpi=190)
    for slot_id, slot in slots.items():
        row = row_by_slot.get(slot_id)
        state = row.get("state_by_slot_aligned_accumulation", "") if row else ""
        face, edge, alpha, label = STATE_STYLE.get(state, ("#e5e7eb", "#cbd5e1", 0.12, "not selected"))
        lw = 0.24
        zorder = 1
        if state == "accumulated_vehicle_core_supported":
            lw = 1.35
            zorder = 5
        elif state in {"accumulated_adjacent_conflict", "accumulated_boundary_conflict"}:
            lw = 0.95
            zorder = 4
        poly = np.asarray(slot["polygon_map"], dtype=np.float64)
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=zorder))
    for row in rows:
        state = row.get("state_by_slot_aligned_accumulation", "")
        if state not in {"accumulated_vehicle_core_supported", "accumulated_adjacent_conflict", "accumulated_boundary_conflict"}:
            continue
        slot = slots.get(str(row["slot_id"]))
        if not slot:
            continue
        center = np.asarray(slot["center_map"], dtype=np.float64)
        ax.text(
            center[0],
            center[1],
            str(row["slot_id"]).replace("slot_", ""),
            fontsize=5.0,
            ha="center",
            va="center",
            color="#111827",
            zorder=8,
        )
    if zoom:
        highlighted = [
            str(row["slot_id"])
            for row in rows
            if row.get("state_by_slot_aligned_accumulation")
            in {"accumulated_vehicle_core_supported", "accumulated_adjacent_conflict", "accumulated_boundary_conflict"}
        ]
        if highlighted:
            pts = np.vstack([np.asarray(slots[slot_id]["polygon_map"], dtype=np.float64) for slot_id in highlighted if slot_id in slots])
        else:
            pts = all_pts
        margin = 0.55
        bmin = pts.min(axis=0) - margin
        bmax = pts.max(axis=0) + margin
    else:
        bmin = all_pts.min(axis=0) - 0.5
        bmax = all_pts.max(axis=0) + 0.5
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    ax.set_title("Slot-aligned accumulation, ±28 frames, stride 4")
    handles = [
        Patch(facecolor=face, edgecolor=edge, alpha=alpha, label=label)
        for face, edge, alpha, label in STATE_STYLE.values()
    ]
    handles.append(Patch(facecolor="#e5e7eb", edgecolor="#cbd5e1", alpha=0.12, label="not selected"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def esc(value: Any) -> str:
    return html.escape(str(value))


def image_src(path: Path, embed: bool) -> str:
    if not embed:
        return path.name
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def write_html_report(path: Path, full_path: Path, zoom_path: Path, summary: dict[str, Any], counts: Counter[str], embed: bool) -> None:
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Slot-aligned Global Map</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; background: white; }}
    .meta {{ color: #475569; }}
  </style>
</head>
<body>
  <h1>Slot-aligned Global Map</h1>
  <p class="meta">Window: ±{esc(summary.get("input", {}).get("window_before", ""))} frames, stride {esc(summary.get("input", {}).get("window_frame_stride", ""))}; counts: {esc(dict(counts))}</p>
  <h2>Full map</h2>
  <img src="{esc(image_src(full_path, embed))}">
  <h2>Candidate zoom</h2>
  <img src="{esc(image_src(zoom_path, embed))}">
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    rows = read_csv(input_dir / "slot_aligned_evidence.csv")
    slot_db = load_json(args.slot_database)
    slots = {str(slot["slot_id"]): slot for slot in slot_db["slots"]}
    summary_path = input_dir / "slot_aligned_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    counts = Counter(row.get("state_by_slot_aligned_accumulation", "") for row in rows)
    full_name = "slot_aligned_global_map_full.png"
    zoom_name = "slot_aligned_global_map_zoom.png"
    full_path = input_dir / full_name
    zoom_path = input_dir / zoom_name
    draw_map(full_path, slots, rows, zoom=False)
    draw_map(zoom_path, slots, rows, zoom=True)
    normal_path = input_dir / args.output_name
    embedded_path = input_dir / args.output_name.replace(".html", "_embedded.html")
    write_html_report(normal_path, full_path, zoom_path, summary, counts, embed=False)
    write_html_report(embedded_path, full_path, zoom_path, summary, counts, embed=True)
    print(json.dumps({"output": str(normal_path), "embedded_output": str(embedded_path), "counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
