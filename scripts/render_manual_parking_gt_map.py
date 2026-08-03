#!/usr/bin/env python3
"""Render a user-supplied slot-level GT map for one ParkingAgent snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Polygon, Wedge


_FONT_PATH = Path(
    "/home/ParkingAgent/DASP/simulation_carla/CARLA_0.9.16/"
    "Engine/Content/Slate/Fonts/DroidSansFallback.ttf"
)
if _FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(_FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=str(_FONT_PATH)
    ).get_name()
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--occupied-slot-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    payload = json.loads(args.part1.read_text(encoding="utf-8"))
    scene = payload["scene"]
    cases = payload["slot_cases"]
    candidate_ids = [str(case["slot"]["slot_id"]) for case in cases]
    candidate_set = set(candidate_ids)
    occupied = {
        value if str(value).startswith("slot_") else f"slot_{value}"
        for value in args.occupied_slot_id
    }
    unknown_ids = occupied - candidate_set
    if unknown_ids:
        raise ValueError(f"occupied IDs are not candidate slots: {sorted(unknown_ids)}")
    free = candidate_set - occupied

    ego_x, ego_y, ego_yaw = (float(value) for value in scene["anchor_pose_map"])
    scale = float(scene["map_units_per_meter"])
    radius = float(scene["radius_m"])
    if scale <= 0:
        raise ValueError("map_units_per_meter must be positive")

    def local_m(point: list[float]) -> tuple[float, float]:
        return (
            (float(point[0]) - ego_x) / scale,
            (float(point[1]) - ego_y) / scale,
        )

    figure, axis = plt.subplots(figsize=(12, 12), dpi=170)
    axis.add_patch(
        Wedge(
            (0.0, 0.0),
            radius,
            math.degrees(ego_yaw) - 40.0,
            math.degrees(ego_yaw) + 40.0,
            facecolor="#dbeafe55",
            edgecolor="#3b82f6",
            linewidth=1.2,
            linestyle="--",
            zorder=0,
        )
    )
    for slot in scene["slots"]:
        slot_id = str(slot["slot_id"])
        polygon = [local_m(point) for point in slot["polygon_map"]]
        if slot_id in occupied:
            face, edge, width, text_color = "#fecaca", "#dc2626", 3.2, "#7f1d1d"
        elif slot_id in free:
            face, edge, width, text_color = "#bbf7d0", "#16a34a", 2.4, "#14532d"
        else:
            face, edge, width, text_color = "#f1f5f9", "#94a3b8", 0.8, "#64748b"
        axis.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=face,
                edgecolor=edge,
                linewidth=width,
                zorder=2 if slot_id in candidate_set else 1,
            )
        )
        if slot_id in candidate_set:
            x, y = local_m(slot["center_map"])
            short_id = slot_id.replace("slot_", "")
            state_mark = "O" if slot_id in occupied else "F"
            axis.text(
                x,
                y,
                f"{short_id}\n{state_mark}",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                color=text_color,
                zorder=4,
            )
    axis.arrow(
        0.0,
        0.0,
        3.0 * math.cos(ego_yaw),
        3.0 * math.sin(ego_yaw),
        width=0.12,
        head_width=0.75,
        color="#111827",
        length_includes_head=True,
        zorder=5,
    )
    axis.scatter([0.0], [0.0], marker="x", s=95, color="#111827", zorder=6)
    axis.set_xlim(-radius, radius)
    axis.set_ylim(-radius, radius)
    axis.set_aspect("equal")
    axis.grid(color="#dbe3ed", linewidth=0.55)
    axis.set_xlabel("以锚点为原点的 map x (m)")
    axis.set_ylabel("以锚点为原点的 map y (m)")
    axis.set_title(
        (
            f"Frame {int(scene['anchor_frame_id'])} 人工GT｜"
            f"Occupied {len(occupied)}｜Free {len(free)}｜候选车位 {len(candidate_ids)}"
        ),
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    axis.legend(
        handles=[
            Patch(facecolor="#fecaca", edgecolor="#dc2626", label="Occupied（O）"),
            Patch(facecolor="#bbf7d0", edgecolor="#16a34a", label="Free（F）"),
            Patch(facecolor="#f1f5f9", edgecolor="#94a3b8", label="非候选背景车位"),
            Patch(facecolor="#dbeafe55", edgecolor="#3b82f6", label="Camera ±40°路由FOV"),
        ],
        loc="upper left",
        frameon=True,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.015,
        "GT来源：用户人工观察；红色=Occupied，绿色=Free；模型预测未参与标注",
        ha="center",
        fontsize=10,
        color="#475569",
    )

    args.output_dir.mkdir(parents=True)
    image_path = args.output_dir / "frame9277_manual_gt_map.png"
    figure.savefig(image_path, bbox_inches="tight")
    plt.close(figure)

    rows = []
    for slot_id in candidate_ids:
        rows.append(
            {
                "slot_id": slot_id,
                "anchor_lidar_frame": int(scene["anchor_frame_id"]),
                "anchor_camera_frame": int(scene["frames"][-1]["camera_frame_id"]),
                "gt_state": "occupied" if slot_id in occupied else "free",
                "gt_camera_observability": "",
                "gt_confidence": "",
                "annotator": "user_manual_observation",
                "notes": "manual GT supplied by user on 2026-07-26",
            }
        )
    csv_path = args.output_dir / "gt_candidate_slots_24.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": "parking-agent-manual-gt/1.0",
        "snapshot_id": scene["snapshot_id"],
        "anchor_lidar_frame": int(scene["anchor_frame_id"]),
        "anchor_camera_frame": int(scene["frames"][-1]["camera_frame_id"]),
        "label_scope": "24 Part1 candidate slots only",
        "label_source": "user_manual_observation",
        "label_date": "2026-07-26",
        "occupied_slot_ids": sorted(occupied),
        "free_slot_ids": sorted(free),
        "occupied_count": len(occupied),
        "free_count": len(free),
        "unknown_count": 0,
        "non_candidate_scene_slots_are_unlabeled": True,
        "model_predictions_used": False,
        "gt_map": image_path.name,
        "gt_csv": csv_path.name,
    }
    (args.output_dir / "gt_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
