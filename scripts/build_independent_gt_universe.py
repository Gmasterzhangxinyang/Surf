#!/usr/bin/env python3
"""Build a prediction-independent parking-slot GT annotation universe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_frames(path: Path, anchor: int, history_count: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["frame"]) <= anchor
        ]
    if not rows or int(rows[-1]["frame"]) != anchor:
        raise ValueError(f"anchor frame {anchor} not found as a causal record")
    selected = rows[-history_count:]
    if len(selected) != history_count:
        raise ValueError(
            f"requested {history_count} causal frames, found {len(selected)}"
        )
    return selected


def _load_legacy_gt(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["slot_id"]): str(row["gt_state"]).lower()
            for row in csv.DictReader(handle)
        }


def _load_current_states(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = _load_json(path)
    return {
        str(row["slot_id"]): str(row["state"]).lower()
        for row in payload.get("decisions", [])
    }


def _ego_xy(
    points_map: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_yaw: float,
    scale: float,
) -> np.ndarray:
    delta_m = (np.asarray(points_map, dtype=np.float64) - anchor_xy) / scale
    cosine = math.cos(anchor_yaw)
    sine = math.sin(anchor_yaw)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]])
    return delta_m @ rotation.T


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_map(
    path: Path,
    universe: list[dict[str, Any]],
    route_map: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_yaw: float,
    scale: float,
) -> None:
    route_ego = _ego_xy(route_map, anchor_xy, anchor_yaw, scale)
    figure, axis = plt.subplots(figsize=(15, 13), dpi=160)
    all_points: list[np.ndarray] = [route_ego]
    for item in universe:
        polygon = _ego_xy(
            np.asarray(item["polygon_map"], dtype=np.float64),
            anchor_xy,
            anchor_yaw,
            scale,
        )
        all_points.append(polygon)
        axis.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor="#dbeafe",
                edgecolor="#246bdb",
                linewidth=1.2,
                alpha=0.75,
            )
        )
        center = polygon.mean(axis=0)
        axis.text(
            center[0],
            center[1],
            item["slot_id"].replace("slot_", ""),
            ha="center",
            va="center",
            fontsize=6,
            color="#10233f",
        )
    axis.plot(
        route_ego[:, 0],
        route_ego[:, 1],
        "-",
        color="#111827",
        linewidth=2.0,
        label="60-frame causal route",
    )
    axis.scatter([0.0], [0.0], marker="x", s=110, color="#ef4444", label="anchor")
    axis.arrow(
        0.0,
        0.0,
        3.0,
        0.0,
        width=0.10,
        head_width=0.65,
        color="#ef4444",
        length_includes_head=True,
    )
    points = np.concatenate(all_points, axis=0)
    margin = 4.0
    axis.set_xlim(float(points[:, 0].min() - margin), float(points[:, 0].max() + margin))
    axis.set_ylim(float(points[:, 1].min() - margin), float(points[:, 1].max() + margin))
    axis.set_aspect("equal")
    axis.grid(color="#d7dee8", linewidth=0.5)
    axis.set_xlabel("anchor forward (m)")
    axis.set_ylabel("anchor left (m)")
    axis.set_title(
        "Independent GT universe — geometry-only route-distance selection\n"
        "No Part1 state or candidate output used",
        fontsize=14,
        fontweight="bold",
    )
    axis.legend(frameon=False)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--slot-db", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, default=9277)
    parser.add_argument("--history-count", type=int, default=60)
    parser.add_argument("--route-radius-m", type=float, default=18.0)
    parser.add_argument("--legacy-gt-csv", type=Path)
    parser.add_argument("--current-decisions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.route_radius_m <= 0.0:
        raise ValueError("route radius must be positive")

    frames = _load_frames(args.frames_csv, args.anchor_frame, args.history_count)
    slot_db = _load_json(args.slot_db)
    scale = float(slot_db["map_units_per_meter"])
    route_map = np.asarray(
        [[float(row["map_x"]), float(row["map_y"])] for row in frames],
        dtype=np.float64,
    )
    anchor_row = frames[-1]
    anchor_xy = route_map[-1]
    anchor_yaw = float(anchor_row["map_yaw"])
    legacy_gt = _load_legacy_gt(args.legacy_gt_csv)
    current_states = _load_current_states(args.current_decisions)

    universe: list[dict[str, Any]] = []
    for slot in slot_db["slots"]:
        center = np.asarray(slot["center_map"], dtype=np.float64)
        route_distances = np.linalg.norm((route_map - center) / scale, axis=1)
        min_index = int(np.argmin(route_distances))
        min_distance = float(route_distances[min_index])
        if min_distance > args.route_radius_m:
            continue
        center_ego = _ego_xy(
            center.reshape(1, 2), anchor_xy, anchor_yaw, scale
        )[0]
        universe.append(
            {
                "slot_id": str(slot["slot_id"]),
                "polygon_map": slot["polygon_map"],
                "core_polygon_map": slot["core_polygon_map"],
                "center_map": slot["center_map"],
                "min_route_distance_m": min_distance,
                "closest_lidar_frame": int(frames[min_index]["frame"]),
                "anchor_distance_m": float(np.linalg.norm(center_ego)),
                "anchor_forward_m": float(center_ego[0]),
                "anchor_left_m": float(center_ego[1]),
                "anchor_bearing_deg": float(
                    math.degrees(math.atan2(center_ego[1], center_ego[0]))
                ),
                "legacy_gt_member": str(slot["slot_id"]) in legacy_gt,
                "current_part1_member": str(slot["slot_id"]) in current_states,
            }
        )
    universe.sort(key=lambda row: (row["min_route_distance_m"], row["slot_id"]))

    root = args.output_dir
    root.mkdir(parents=True)
    (root / "geometry_universe.json").write_text(
        json.dumps(
            {
                "schema_version": "parking-slot-independent-gt-universe/1.0",
                "selection_rule": (
                    f"slot center min distance to causal {args.history_count}-frame "
                    f"route <= {args.route_radius_m:.3f} m"
                ),
                "prediction_fields_used_for_selection": [],
                "anchor_frame": args.anchor_frame,
                "history_frames": [int(row["frame"]) for row in frames],
                "map_units_per_meter": scale,
                "slots": universe,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    template_rows: list[dict[str, Any]] = []
    for item in universe:
        slot_id = item["slot_id"]
        template_rows.append(
            {
                "slot_id": slot_id,
                "anchor_lidar_frame": args.anchor_frame,
                "closest_lidar_frame": item["closest_lidar_frame"],
                "min_route_distance_m": f"{item['min_route_distance_m']:.6f}",
                "anchor_distance_m": f"{item['anchor_distance_m']:.6f}",
                "gt_state": "",
                "gt_observability": "",
                "identity_verified": "",
                "state_confidence": "",
                "annotator": "",
                "evidence_frames": "",
                "legacy_gt_state_unverified": legacy_gt.get(slot_id, ""),
                "legacy_gt_member": int(item["legacy_gt_member"]),
                "current_part1_member_audit_only": int(item["current_part1_member"]),
                "notes": "",
            }
        )
    _write_csv(root / "gt_annotation_template.csv", template_rows)

    provisional_rows = [
        {
            "slot_id": slot_id,
            "legacy_gt_state": state,
            "status": "unverified_not_formal_gt",
            "reason": "legacy universe was selected by old Part1 candidate output",
        }
        for slot_id, state in sorted(legacy_gt.items())
    ]
    if "slot_1252" in {row["slot_id"] for row in universe}:
        provisional_rows.append(
            {
                "slot_id": "slot_1252",
                "legacy_gt_state": "free",
                "status": "user_supplied_requires_identity_recheck",
                "reason": "supplemental user observation on 2026-07-29",
            }
        )
    if provisional_rows:
        _write_csv(root / "legacy_labels_for_recheck.csv", provisional_rows)

    _render_map(
        root / "geometry_universe_neutral_map.png",
        universe,
        route_map,
        anchor_xy,
        anchor_yaw,
        scale,
    )
    manifest = {
        "schema_version": "parking-slot-independent-gt-pack/1.0",
        "formal_gt": False,
        "annotation_status": "identity_and_state_review_required",
        "anchor_frame": args.anchor_frame,
        "history_count": args.history_count,
        "route_radius_m": args.route_radius_m,
        "universe_slot_count": len(universe),
        "selection_is_prediction_independent": True,
        "selection_inputs": [str(args.frames_csv), str(args.slot_db)],
        "legacy_gt_used_for_selection": False,
        "current_part1_used_for_selection": False,
        "legacy_and_current_membership_are_audit_columns_only": True,
        "camera_projection_status": "unavailable_not_allowed_for_identity_claim",
        "required_formal_label_fields": [
            "gt_state",
            "gt_observability",
            "identity_verified",
            "state_confidence",
            "annotator",
            "evidence_frames",
        ],
        "source_sha256": {
            str(args.frames_csv): _sha256(args.frames_csv),
            str(args.slot_db): _sha256(args.slot_db),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        """# Frame 9277 独立GT标注包

本包的车位全集只由60帧因果路线与全量车位几何的距离确定，未使用任何
Part1状态、Unknown原因或candidate输出。

正式标注顺序：

1. 先确认车位ID与真实车位一一对应，填写 `identity_verified=yes`。
2. 再填写可观测性：`visible / partial / occluded / out_of_view`。
3. 只有充分可见时填写 `free` 或 `occupied`；否则必须为 `unknown`。
4. `legacy_gt_state_unverified` 只用于复核，不得自动复制为正式标签。
5. 没有通过审计的map→pixel投影时，不允许仅凭相机大致方位绑定slot ID。

只有完成双人独立标注、冲突仲裁并锁定哈希后，`formal_gt` 才能改为true。
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(root),
                "universe_slot_count": len(universe),
                "legacy_gt_members": sum(
                    bool(row["legacy_gt_member"]) for row in universe
                ),
                "current_part1_members": sum(
                    bool(row["current_part1_member"]) for row in universe
                ),
                "selection_is_prediction_independent": True,
                "formal_gt": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
