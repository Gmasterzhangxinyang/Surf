#!/usr/bin/env python3
"""Fast parking-slot scoring from pose + sparse LiDAR on the GLTF map."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}") from exc

from gltf_lidar_ndt import (
    accessor_array,
    decode_data_uri,
    draw_gltf_map,
    load_gltf_map,
    node_matrix,
    normalize_angle,
    transform_positions,
    yaw_to_rot,
)


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
MATCH_FILE = Path("/home/ParkingAgent/dataset/dataset/matched_frames.txt")
DEFAULT_ALIGNMENT = Path("outputs/pose_gltf_route_all/alignment.json")


@dataclass
class Slot:
    slot_id: str
    polygon: np.ndarray
    center: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast score visible GLTF parking slots with pose + LiDAR")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--match-file", type=Path, default=MATCH_FILE)
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--frame-id", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fast_slot_scoring"))
    parser.add_argument("--map-sample-step", type=float, default=0.06)
    parser.add_argument("--lidar-range-m", type=float, default=35.0)
    parser.add_argument("--z-min", type=float, default=-1.2)
    parser.add_argument("--z-max", type=float, default=2.2)
    parser.add_argument("--obstacle-z-min", type=float, default=0.20)
    parser.add_argument("--obstacle-z-max", type=float, default=2.0)
    parser.add_argument("--camera-fov-deg", type=float, default=95.0)
    parser.add_argument("--visible-range-map", type=float, default=4.5)
    parser.add_argument("--occupied-min-points", type=int, default=4)
    parser.add_argument("--free-max-points", type=int, default=1)
    parser.add_argument("--require-observed", action="store_true", help="Only mark free slots as candidates when at least one LiDAR point falls in the slot polygon")
    parser.add_argument("--no-render", action="store_true", help="Skip PNG rendering for low-latency scoring")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pose_rows(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    with path.open("r") as handle:
        for line in handle:
            vals = np.fromstring(line, sep=" ")
            if vals.size != 12:
                continue
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = vals.reshape(3, 4)
            poses.append(mat)
    return poses


def load_match_pose_index(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with path.open("r") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = [int(x) for x in line.split()]
            if len(fields) >= 19:
                valid = [idx for idx in fields[14:19] if idx >= 0]
                if valid:
                    mapping[fields[0]] = valid[0]
    return mapping


def pose_to_xyyaw(mat: np.ndarray) -> np.ndarray:
    return np.array([mat[0, 3], mat[1, 3], math.atan2(float(mat[1, 0]), float(mat[0, 0]))], dtype=np.float64)


def map_pose_from_world(world_pose: np.ndarray, alignment: dict) -> np.ndarray:
    scale = float(alignment["scale"])
    yaw = float(alignment["yaw"])
    trans = np.asarray(alignment["translation"], dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    out[:2] = scale * (world_pose[:2] @ yaw_to_rot(yaw).T) + trans
    out[2] = normalize_angle(float(world_pose[2] + yaw))
    return out


def lidar_points_to_map(raw_xyz: np.ndarray, world_pose: np.ndarray, alignment: dict) -> np.ndarray:
    rot_world = yaw_to_rot(float(world_pose[2]))
    world_xy = raw_xyz[:, :2] @ rot_world.T + world_pose[:2]
    scale = float(alignment["scale"])
    yaw = float(alignment["yaw"])
    trans = np.asarray(alignment["translation"], dtype=np.float64)
    map_xy = scale * (world_xy @ yaw_to_rot(yaw).T) + trans
    return np.column_stack([map_xy, raw_xyz[:, 2]])


def load_lidar_xyz(path: Path, z_min: float, z_max: float, range_m: float) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Invalid LiDAR bin: {path}")
    pts = raw.reshape(-1, 4)[:, :3].astype(np.float64)
    mask = np.isfinite(pts).all(axis=1)
    mask &= (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    mask &= np.linalg.norm(pts[:, :2], axis=1) <= range_m
    return pts[mask]


def order_polygon(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def extract_parking_slots(gltf_path: Path) -> list[Slot]:
    data = json.loads(gltf_path.read_text(encoding="utf-8", errors="ignore"))
    buffer_bytes = decode_data_uri(data["buffers"][0]["uri"])
    slots: list[Slot] = []

    for node in data.get("nodes", []):
        if (node.get("extras") or {}).get("type") != "parkingSpace" or "mesh" not in node:
            continue
        mesh = data["meshes"][int(node["mesh"])]
        mat = node_matrix(node)
        for primitive in mesh.get("primitives", []):
            pos_accessor = primitive["attributes"].get("POSITION")
            if pos_accessor is None or "indices" not in primitive:
                continue
            positions = accessor_array(data, buffer_bytes, int(pos_accessor)).astype(np.float64)
            positions = transform_positions(positions, mat)[:, :2]
            indices = accessor_array(data, buffer_bytes, int(primitive["indices"])).astype(np.int64).reshape(-1)
            tris = indices.reshape(-1, 3)

            vert_to_tri: dict[int, list[int]] = defaultdict(list)
            for tri_idx, tri in enumerate(tris):
                for vertex in tri:
                    vert_to_tri[int(vertex)].append(tri_idx)

            seen = np.zeros(len(tris), dtype=bool)
            for tri_idx in range(len(tris)):
                if seen[tri_idx]:
                    continue
                stack = [tri_idx]
                seen[tri_idx] = True
                component: list[int] = []
                while stack:
                    current = stack.pop()
                    component.append(current)
                    for vertex in tris[current]:
                        for neighbor in vert_to_tri[int(vertex)]:
                            if not seen[neighbor]:
                                seen[neighbor] = True
                                stack.append(neighbor)

                unique_vertices = np.unique(tris[component].reshape(-1))
                poly_points = np.unique(positions[unique_vertices], axis=0)
                if len(poly_points) < 4:
                    continue
                polygon = order_polygon(poly_points)
                bbox_min = polygon.min(axis=0)
                bbox_max = polygon.max(axis=0)
                area = float(np.prod(bbox_max - bbox_min))
                if area <= 1e-5:
                    continue
                slot_id = f"slot_{len(slots):04d}"
                slots.append(Slot(slot_id=slot_id, polygon=polygon, center=polygon.mean(axis=0), bbox_min=bbox_min, bbox_max=bbox_max))

    return slots


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= intersect
        j = i
    return inside


def slot_in_camera_view(slot: Slot, ego_map_pose: np.ndarray, fov_deg: float, max_range: float) -> tuple[bool, float, float]:
    rel = (slot.center - ego_map_pose[:2]) @ yaw_to_rot(float(ego_map_pose[2]))
    forward = float(rel[0])
    lateral = float(rel[1])
    dist = float(np.linalg.norm(rel))
    angle = math.atan2(lateral, max(forward, 1e-9))
    visible = forward > 0.0 and dist <= max_range and abs(angle) <= math.radians(fov_deg) * 0.5
    return visible, dist, angle


def score_slots(
    slots: list[Slot],
    map_points_xyz: np.ndarray,
    ego_map_pose: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    xy = map_points_xyz[:, :2]
    z = map_points_xyz[:, 2]
    obstacle_mask_all = (z >= args.obstacle_z_min) & (z <= args.obstacle_z_max)

    for slot in slots:
        visible, dist, angle = slot_in_camera_view(slot, ego_map_pose, args.camera_fov_deg, args.visible_range_map)
        if not visible:
            status = "unknown"
            total = 0
            obstacle = 0
            free_score = 0.0
        else:
            bbox_mask = np.all((xy >= slot.bbox_min) & (xy <= slot.bbox_max), axis=1)
            candidates = xy[bbox_mask]
            candidate_obstacle = obstacle_mask_all[bbox_mask]
            inside = points_in_polygon(candidates, slot.polygon)
            total = int(inside.sum())
            obstacle = int((candidate_obstacle & inside).sum())
            if obstacle >= args.occupied_min_points:
                status = "occupied"
            elif args.require_observed and total <= 0:
                status = "unknown"
            elif obstacle <= args.free_max_points:
                status = "candidate"
            else:
                status = "unknown"
            distance_term = max(0.0, 1.0 - dist / max(args.visible_range_map, 1e-9))
            obstacle_penalty = min(1.0, obstacle / max(args.occupied_min_points, 1))
            free_score = float(np.clip(0.75 * (1.0 - obstacle_penalty) + 0.25 * distance_term, 0.0, 1.0))

        rows.append(
            {
                "slot_id": slot.slot_id,
                "center_x": float(slot.center[0]),
                "center_y": float(slot.center[1]),
                "visible": bool(visible),
                "distance_map": float(dist),
                "angle_deg": float(math.degrees(angle)),
                "total_points": total,
                "obstacle_points": obstacle,
                "status": status,
                "free_score": free_score,
            }
        )
    return rows


def draw_result(output: Path, gltf_map, slots: list[Slot], rows: list[dict[str, object]], ego_pose: np.ndarray, frame_id: int) -> None:
    status_by_id = {str(row["slot_id"]): row for row in rows}
    colors = {
        "candidate": ("#16a34a", 0.62),
        "occupied": ("#dc2626", 0.68),
        "unknown": ("#94a3b8", 0.12),
    }
    fig, ax = plt.subplots(figsize=(11, 13), dpi=220)
    draw_gltf_map(ax, gltf_map)
    for slot in slots:
        row = status_by_id[slot.slot_id]
        if not row["visible"] and row["status"] == "unknown":
            continue
        color, alpha = colors[str(row["status"])]
        patch = Polygon(slot.polygon, closed=True, facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.8, zorder=8)
        ax.add_patch(patch)
    heading = np.array([math.cos(ego_pose[2]), math.sin(ego_pose[2])])
    ax.scatter([ego_pose[0]], [ego_pose[1]], c="#111827", s=38, zorder=12, label="ego pose")
    ax.arrow(ego_pose[0], ego_pose[1], heading[0] * 0.85, heading[1] * 0.85, color="#111827", width=0.025, zorder=12)
    counts = {k: sum(1 for r in rows if r["status"] == k and r["visible"]) for k in ("candidate", "occupied", "unknown")}
    ax.set_title(f"Fast slot scoring | frame {frame_id:06d} | candidate={counts['candidate']} occupied={counts['occupied']} unknown={counts['unknown']}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    ensure_dir(output.parent)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    gltf_map = load_gltf_map(args.gltf, args.map_sample_step)
    slots = extract_parking_slots(args.gltf)
    if not slots:
        raise SystemExit("No parkingSpace components found in GLTF")
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    poses = load_pose_rows(args.dataset_root / "pose" / "poses.txt")
    pose_index = load_match_pose_index(args.match_file)
    pose_idx = pose_index.get(args.frame_id, args.frame_id if args.frame_id < len(poses) else -1)
    if pose_idx < 0 or pose_idx >= len(poses):
        raise SystemExit(f"No pose for LiDAR frame {args.frame_id}")

    world_pose = pose_to_xyyaw(poses[pose_idx])
    ego_map_pose = map_pose_from_world(world_pose, alignment)
    scan_path = args.dataset_root / "velodyne" / f"{args.frame_id:06d}.bin"
    lidar_xyz = load_lidar_xyz(scan_path, args.z_min, args.z_max, args.lidar_range_m)
    map_points_xyz = lidar_points_to_map(lidar_xyz, world_pose, alignment)
    rows = score_slots(slots, map_points_xyz, ego_map_pose, args)
    rows_sorted = sorted(rows, key=lambda row: (row["status"] != "candidate", -float(row["free_score"]), float(row["distance_map"])))

    csv_path = args.output_dir / f"slot_scores_{args.frame_id:06d}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    json_path = args.output_dir / f"slot_scores_{args.frame_id:06d}.json"
    summary = {
        "frame_id": args.frame_id,
        "pose_index": pose_idx,
        "ego_map_pose": ego_map_pose.tolist(),
        "slot_count": len(slots),
        "visible_slots": int(sum(1 for row in rows if row["visible"])),
        "candidate_slots": int(sum(1 for row in rows if row["status"] == "candidate" and row["visible"])),
        "occupied_slots": int(sum(1 for row in rows if row["status"] == "occupied" and row["visible"])),
        "top_candidates": [row for row in rows_sorted if row["status"] == "candidate"][: args.top_k],
        "warning": alignment.get("warning", ""),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    map_path = args.output_dir / f"slot_map_{args.frame_id:06d}.png"
    if not args.no_render:
        draw_result(map_path, gltf_map, slots, rows, ego_map_pose, args.frame_id)
    print(f"[slots] {len(slots)} total, {summary['visible_slots']} visible")
    print(f"[candidate] {summary['candidate_slots']} [occupied] {summary['occupied_slots']}")
    print(f"[csv] {csv_path}")
    print(f"[json] {json_path}")
    if not args.no_render:
        print(f"[map] {map_path}")


if __name__ == "__main__":
    main()
