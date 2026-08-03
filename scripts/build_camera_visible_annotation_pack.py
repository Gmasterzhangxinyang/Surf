#!/usr/bin/env python3
"""Build a prediction-blind, camera-visible parking-slot annotation pack."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def _ego_xy(
    points_map: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_yaw: float,
    scale: float,
) -> np.ndarray:
    delta = (points_map - anchor_xy) / scale
    rotation = np.asarray(
        [
            [math.cos(anchor_yaw), math.sin(anchor_yaw)],
            [-math.sin(anchor_yaw), math.cos(anchor_yaw)],
        ],
        dtype=np.float64,
    )
    return delta @ rotation.T


def _render_location(
    path: Path,
    slots: list[dict],
    visible_ids: set[str],
    target_id: str,
    route_map: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_yaw: float,
    scale: float,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 10), dpi=150)
    route = _ego_xy(route_map, anchor_xy, anchor_yaw, scale)
    all_points = [route]
    for slot in slots:
        slot_id = str(slot["slot_id"])
        polygon = _ego_xy(
            np.asarray(slot["polygon_map"], dtype=np.float64),
            anchor_xy,
            anchor_yaw,
            scale,
        )
        all_points.append(polygon)
        if slot_id == target_id:
            face, edge, width, alpha, zorder = "#f0abfc", "#a21caf", 3.5, 0.95, 4
        elif slot_id in visible_ids:
            face, edge, width, alpha, zorder = "#dbeafe", "#2563eb", 1.4, 0.72, 3
        else:
            face, edge, width, alpha, zorder = "#e5e7eb", "#9ca3af", 0.7, 0.45, 1
        axis.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=face,
                edgecolor=edge,
                linewidth=width,
                alpha=alpha,
                zorder=zorder,
            )
        )
        center = polygon.mean(axis=0)
        if slot_id == target_id or slot_id in visible_ids:
            axis.text(
                center[0],
                center[1],
                slot_id.replace("slot_", ""),
                ha="center",
                va="center",
                fontsize=8 if slot_id == target_id else 6,
                fontweight="bold" if slot_id == target_id else "normal",
                color="#701a75" if slot_id == target_id else "#17345f",
                zorder=5,
            )
    axis.plot(route[:, 0], route[:, 1], color="#111827", linewidth=2.0, zorder=2)
    axis.scatter([0], [0], marker="x", s=100, color="#ef4444", zorder=6)
    axis.arrow(
        0,
        0,
        3,
        0,
        width=0.10,
        head_width=0.65,
        color="#ef4444",
        length_includes_head=True,
        zorder=6,
    )
    points = np.concatenate(all_points)
    margin = 3.5
    axis.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
    axis.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
    axis.set_aspect("equal")
    axis.grid(color="#d7dee8", linewidth=0.5)
    axis.set_xlabel("anchor forward (m)")
    axis.set_ylabel("anchor left (m)")
    axis.set_title(
        f"{target_id} location — magenta target; blue camera-visible GT set",
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--camera-manifest", type=Path, required=True)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    camera = json.loads(args.camera_manifest.read_text(encoding="utf-8"))
    visible_rows = [row for row in camera["rows"] if int(row["view_count"]) > 0]
    visible_ids = {str(row["slot_id"]) for row in visible_rows}
    slots = universe["slots"]
    slot_by_id = {str(slot["slot_id"]): slot for slot in slots}
    unknown = visible_ids - set(slot_by_id)
    if unknown:
        raise ValueError(f"camera manifest contains slots outside universe: {sorted(unknown)}")

    history = set(int(value) for value in universe["history_frames"])
    with args.frames_csv.open("r", encoding="utf-8", newline="") as handle:
        frame_rows = [
            row for row in csv.DictReader(handle) if int(row["frame"]) in history
        ]
    if len(frame_rows) != len(history):
        raise ValueError("frames CSV does not contain the complete GT history")
    route_map = np.asarray(
        [[float(row["map_x"]), float(row["map_y"])] for row in frame_rows],
        dtype=np.float64,
    )
    anchor_row = frame_rows[-1]
    anchor_xy = route_map[-1]
    anchor_yaw = float(anchor_row["map_yaw"])
    scale = float(universe["map_units_per_meter"])

    root = args.output_dir
    locations = root / "location_maps"
    locations.mkdir(parents=True)
    template_rows = []
    case_blocks = []
    camera_root = args.camera_manifest.parent
    for index, row in enumerate(visible_rows, start=1):
        slot_id = str(row["slot_id"])
        location_name = f"{slot_id}.png"
        _render_location(
            locations / location_name,
            slots,
            visible_ids,
            slot_id,
            route_map,
            anchor_xy,
            anchor_yaw,
            scale,
        )
        camera_path = Path(str(row["image"]))
        if not camera_path.is_absolute():
            workspace_candidate = Path.cwd() / camera_path
            camera_path = (
                workspace_candidate
                if workspace_candidate.exists()
                else camera_root / camera_path.name
            )
        camera_rel = Path("..") / "camera_views" / camera_path.name
        template_rows.append(
            {
                "slot_id": slot_id,
                "gt_state": "",
                "gt_observability": "",
                "identity_verified": "",
                "state_confidence": "",
                "annotator": "",
                "evidence_frames": ";".join(str(value) for value in row["frames"]),
                "notes": "",
            }
        )
        case_blocks.append(
            f"""
<section>
  <h2>{index:02d}. {html.escape(slot_id)}</h2>
  <div class="grid">
    <figure><img src="location_maps/{location_name}"><figcaption>位置图：紫色为本车位</figcaption></figure>
    <figure><img src="{camera_rel.as_posix()}"><figcaption>多帧相机图：青色多边形为投影车位</figcaption></figure>
  </div>
  <p>请填写：Free / Occupied / Unknown；被遮挡或无法确认必须填 Unknown。</p>
</section>"""
        )

    with (root / "camera_visible_gt_template.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(template_rows[0]))
        writer.writeheader()
        writer.writerows(template_rows)
    (root / "annotation_index.html").write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Frame {int(universe['anchor_frame'])} camera-visible GT annotation</title>
<style>
body{{margin:0;background:#edf2f7;color:#14213d;font-family:Arial,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}}
section{{background:white;border-radius:14px;padding:20px;margin:20px 0;box-shadow:0 4px 18px #22334d18}}
.grid{{display:grid;grid-template-columns:1fr 1.25fr;gap:18px;align-items:start}}
figure{{margin:0}}img{{width:100%;height:auto;border:1px solid #ccd6e3;border-radius:8px}}
figcaption{{padding:7px;color:#52647b}}code{{background:#e8edf4;padding:2px 5px}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Frame {int(universe['anchor_frame'])} 相机可见车位人工GT</h1>
<p>共 {len(visible_rows)} 个车位。该集合只由地图几何、历史60帧路线和相机投影生成，
未使用Part1状态或Agent预测。请在 <code>camera_visible_gt_template.csv</code> 填写结果。</p>
<p><b>标注规则：</b>只有目标车位身份明确且车位内部充分可见时填 Free/Occupied；
被车、柱子或视角遮挡时填 Unknown。</p>
{''.join(case_blocks)}
</main></body></html>
""",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "camera-visible-gt-annotation-pack/1.0",
                "anchor_frame": int(universe["anchor_frame"]),
                "prediction_blind": True,
                "selection_rule": "independent route universe and camera view_count > 0",
                "camera_extrinsic_status": camera.get("extrinsic_status"),
                "slot_count": len(visible_rows),
                "slot_ids": [row["slot_id"] for row in visible_rows],
                "formal_gt": False,
                "required_user_output": "camera_visible_gt_template.csv",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(root),
                "slot_count": len(visible_rows),
                "index": str(root / "annotation_index.html"),
                "template": str(root / "camera_visible_gt_template.csv"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
