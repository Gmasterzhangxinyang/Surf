#!/usr/bin/env python3
"""Render a global map report for slot-constrained box scoring outputs."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon as MplPolygon


DEFAULT_INPUT = Path("outputs/slot_constrained_box_scoring_v1")
DEFAULT_SLOT_DATABASE = Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json")


STATE_STYLE = {
    "box_vehicle_core_supported": ("#dc2626", "#7f1d1d", 0.78, "box vehicle core supported"),
    "box_adjacent_conflict": ("#f97316", "#9a3412", 0.72, "box adjacent conflict"),
    "box_boundary_conflict": ("#a855f7", "#581c87", 0.68, "box boundary conflict"),
    "box_low_height_residual": ("#64748b", "#334155", 0.46, "box low-height residual"),
    "box_wall_like_or_static_suspect": ("#0f766e", "#134e4a", 0.52, "wall/static suspect"),
    "box_no_vehicle_evidence": ("#d1d5db", "#94a3b8", 0.28, "box no vehicle evidence"),
    "box_unknown_insufficient_visibility": ("#facc15", "#a16207", 0.42, "insufficient visibility"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build global map for slot-constrained box scoring")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--slot-database", type=Path, default=DEFAULT_SLOT_DATABASE)
    parser.add_argument("--output-name", default="slot_box_global_map_report.html")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def draw_map(path: Path, slots: dict[str, dict[str, Any]], rows: list[dict[str, str]], zoom: bool) -> None:
    row_by_slot = {str(row["slot_id"]): row for row in rows}
    all_pts = np.vstack([np.asarray(slot["polygon_map"], dtype=np.float64) for slot in slots.values()])
    fig, ax = plt.subplots(figsize=(13, 10), dpi=190)

    for slot_id, slot in slots.items():
        row = row_by_slot.get(slot_id)
        state = row.get("state", "") if row else ""
        face, edge, alpha, _ = STATE_STYLE.get(state, ("#e5e7eb", "#cbd5e1", 0.10, "not selected"))
        lw = 0.25
        zorder = 1
        if state == "box_vehicle_core_supported":
            lw = 1.15
            zorder = 6
        elif state in {"box_adjacent_conflict", "box_boundary_conflict"}:
            lw = 0.95
            zorder = 5
        elif state:
            lw = 0.55
            zorder = 4
        poly = np.asarray(slot["polygon_map"], dtype=np.float64)
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=zorder))

    for row in rows:
        state = row.get("state", "")
        if state not in {"box_vehicle_core_supported", "box_adjacent_conflict", "box_boundary_conflict"}:
            continue
        slot = slots.get(str(row["slot_id"]))
        if not slot:
            continue
        center = np.asarray(slot["center_map"], dtype=np.float64)
        ax.text(
            center[0],
            center[1],
            str(row["slot_id"]).replace("slot_", ""),
            fontsize=4.8,
            ha="center",
            va="center",
            color="#111827",
            zorder=10,
        )

    if zoom:
        highlighted = [str(row["slot_id"]) for row in rows if row.get("state") != "box_no_vehicle_evidence"]
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
    ax.set_title("Slot-constrained box scoring v1 global map")
    handles = [Patch(facecolor=face, edgecolor=edge, alpha=alpha, label=label) for face, edge, alpha, label in STATE_STYLE.values()]
    handles.append(Patch(facecolor="#e5e7eb", edgecolor="#cbd5e1", alpha=0.10, label="not selected"))
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


def write_html_report(path: Path, full_path: Path, zoom_path: Path, counts: Counter[str], summary: dict[str, Any], embed: bool) -> None:
    count_items = "".join(f"<li><b>{esc(k)}</b>: {esc(v)}</li>" for k, v in sorted(counts.items()))
    config = summary.get("config", {}) if isinstance(summary, dict) else {}
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Slot Box Global Map</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; background: white; }}
    .meta {{ color: #475569; }}
  </style>
</head>
<body>
  <h1>Slot-constrained Box Scoring V1 Global Map</h1>
  <p class="meta">Window: ±{esc(config.get("window_before", ""))} / ±{esc(config.get("window_after", ""))} frames, stride {esc(config.get("frame_stride", ""))}. These are evidence states, not final occupied/free decisions.</p>
  <h2>Counts</h2>
  <ul>{count_items}</ul>
  <h2>Full Map</h2>
  <img src="{esc(image_src(full_path, embed))}">
  <h2>Candidate Zoom</h2>
  <img src="{esc(image_src(zoom_path, embed))}">
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    rows = read_csv(input_dir / "slot_box_scores.csv")
    slot_db = load_json(args.slot_database)
    slots = {str(slot["slot_id"]): slot for slot in slot_db["slots"]}
    summary_path = input_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    counts = Counter(row.get("state", "") for row in rows)

    full_path = input_dir / "slot_box_global_map_full.png"
    zoom_path = input_dir / "slot_box_global_map_zoom.png"
    draw_map(full_path, slots, rows, zoom=False)
    draw_map(zoom_path, slots, rows, zoom=True)

    normal_path = input_dir / args.output_name
    embedded_path = input_dir / args.output_name.replace(".html", "_embedded.html")
    write_html_report(normal_path, full_path, zoom_path, counts, summary, embed=False)
    write_html_report(embedded_path, full_path, zoom_path, counts, summary, embed=True)
    print(json.dumps({"output": str(normal_path), "embedded_output": str(embedded_path), "counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
