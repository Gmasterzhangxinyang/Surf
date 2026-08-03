from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import box_polygon
from .point_filtering import estimate_local_ground_z
from .scoring import BoxScore, low_height_mask, vehicle_height_mask
from .config import BoxScoringConfig


def write_json(path: str | Path, data: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2)


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def state_counts(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    return dict(Counter(str(row.get("state", "")) for row in rows))


def draw_debug_slot(
    output_path: str | Path,
    slot: dict,
    adjacent_slots: list[dict],
    score: BoxScore,
    accumulated_points: np.ndarray,
    config: BoxScoringConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    if len(accumulated_points):
        ground_z = estimate_local_ground_z(accumulated_points[:, :3], config.ground_quantile)
        z_rel = accumulated_points[:, 2] - ground_z
        low = low_height_mask(z_rel, config)
        vehicle = vehicle_height_mask(z_rel, config)
        background = ~(low | vehicle)
        ax.scatter(accumulated_points[background, 0], accumulated_points[background, 1], s=2, c="#9ca3af", alpha=0.18, label="other")
        ax.scatter(accumulated_points[low, 0], accumulated_points[low, 1], s=4, c="#f59e0b", alpha=0.45, label="low-height")
        ax.scatter(accumulated_points[vehicle, 0], accumulated_points[vehicle, 1], s=4, c="#ef4444", alpha=0.45, label="vehicle-height")

    for adjacent in adjacent_slots:
        poly = np.asarray(adjacent.get("polygon_np", adjacent.get("polygon_map")), dtype=np.float64)
        closed = np.vstack([poly, poly[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#94a3b8", linewidth=1.0, alpha=0.7)

    polygon = np.asarray(slot.get("polygon_np", slot.get("polygon_map")), dtype=np.float64)
    core = np.asarray(slot.get("core_np", slot.get("core_polygon_map", polygon)), dtype=np.float64)
    closed = np.vstack([polygon, polygon[0]])
    core_closed = np.vstack([core, core[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="#111827", linewidth=2.0, label="slot")
    ax.plot(core_closed[:, 0], core_closed[:, 1], color="#2563eb", linewidth=2.0, label="core")

    best_box = box_polygon(np.asarray([score.center_x, score.center_y]), score.yaw, score.length, score.width)
    box_closed = np.vstack([best_box, best_box[0]])
    ax.plot(box_closed[:, 0], box_closed[:, 1], color="#22c55e", linewidth=2.5, label="best box")
    ax.scatter([score.center_x], [score.center_y], c="#16a34a", s=35)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_title(
        f"{score.slot_id} | {score.state}\n"
        f"score={score.score:.3f}, z95={score.z95_above_ground:.2f}, span={score.height_span:.2f}, "
        f"temporal={score.temporal_support:.2f}, adjacent={score.adjacent_overlap:.2f}"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_html_report(
    output_path: str | Path,
    summary: dict[str, object],
    rows: list[dict[str, object]],
    debug_images: dict[str, str],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = summary.get("state_counts", {})
    top_rows = sorted(rows, key=lambda r: float(r.get("score") or 0.0), reverse=True)[:80]

    def e(value: object) -> str:
        return html.escape(str(value))

    state_items = "".join(f"<li><b>{e(k)}</b>: {e(v)}</li>" for k, v in sorted(counts.items()))
    table_rows = []
    for row in top_rows:
        sid = str(row.get("slot_id", ""))
        debug = debug_images.get(sid, "")
        debug_cell = f'<a href="{e(debug)}"><img src="{e(debug)}" alt="{e(sid)}" /></a>' if debug else ""
        table_rows.append(
            "<tr>"
            f"<td>{e(sid)}</td>"
            f"<td>{e(row.get('state', ''))}</td>"
            f"<td>{float(row.get('score') or 0):.3f}</td>"
            f"<td>{e(row.get('baseline_state', ''))}</td>"
            f"<td>{e(row.get('anchor_frame', ''))}</td>"
            f"<td>{e(row.get('supported_frame_count', ''))}/{e(row.get('selected_frame_count', ''))}</td>"
            f"<td>{float(row.get('z95_above_ground') or 0):.2f}</td>"
            f"<td>{float(row.get('height_span') or 0):.2f}</td>"
            f"<td>{float(row.get('slot_core_overlap') or 0):.2f}</td>"
            f"<td>{float(row.get('adjacent_overlap') or 0):.2f}</td>"
            f"<td>{e(row.get('reason', ''))}</td>"
            f"<td>{debug_cell}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Slot Constrained Box Scoring V1</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    img {{ max-width: 220px; max-height: 220px; }}
    .note {{ color: #4b5563; max-width: 960px; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Slot Constrained Box Scoring V1</h1>
  <p class="note">This is an experimental vehicle-hypothesis pipeline. It does not output final free/occupied decisions and does not replace the DBSCAN baseline.</p>
  <h2>Summary</h2>
  <ul>{state_items}</ul>
  <p><b>Processed slots:</b> {e(summary.get('processed_slot_count', ''))}</p>
  <p><b>Baseline:</b> <code>{e(summary.get('baseline_dir', ''))}</code></p>
  <h2>Top Scored Slots</h2>
  <table>
    <thead>
      <tr>
        <th>slot</th><th>state</th><th>score</th><th>DBSCAN state</th><th>anchor</th><th>temporal</th>
        <th>z95</th><th>height span</th><th>core overlap</th><th>adjacent overlap</th><th>reason</th><th>debug</th>
      </tr>
    </thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html_text)
