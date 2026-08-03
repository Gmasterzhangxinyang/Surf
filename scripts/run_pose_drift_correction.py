#!/usr/bin/env python3
"""Audit map-pose drift, fit conservative corrections, and rebuild a synchronized dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from gltf_lidar_ndt import load_gltf_map  # noqa: E402
from parking_pose_correction.dataset import (  # noqa: E402
    lidar_to_map_xyzi,
    load_timestamp_file,
    synchronize_frame_rows,
)
from parking_pose_correction.registration import evaluate_alignment, estimate_local_correction  # noqa: E402
from parking_pose_correction.reporting import write_drift_plots, write_drift_report  # noqa: E402
from parking_pose_correction.se2 import (  # noqa: E402
    apply_pose_corrections,
    interpolate_corrections,
    select_keyframes,
    stabilize_dense_corrections,
    smooth_keyframe_corrections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, default=Path("outputs/frame_map_dataset/frames.csv"))
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ParkingAgent/dataset/dataset/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pose_drift_correction"))
    parser.add_argument("--corrected-dataset-dir", type=Path, default=Path("outputs/frame_map_dataset_pose_corrected"))
    parser.add_argument("--layers", default="wall,elevator,arrester")
    parser.add_argument("--keyframe-distance-m", type=float, default=2.0)
    parser.add_argument("--keyframe-yaw-deg", type=float, default=10.0)
    parser.add_argument("--max-keyframes", type=int, default=0)
    parser.add_argument("--target-radius-m", type=float, default=28.0)
    parser.add_argument("--scan-range-min-m", type=float, default=2.0)
    parser.add_argument("--scan-range-max-m", type=float, default=25.0)
    parser.add_argument("--scan-z-min-m", type=float, default=0.30)
    parser.add_argument("--scan-z-max-m", type=float, default=2.20)
    parser.add_argument("--scan-max-points", type=int, default=2200)
    parser.add_argument("--max-translation-m", type=float, default=1.5)
    parser.add_argument("--max-yaw-deg", type=float, default=5.0)
    parser.add_argument("--max-correspondence-m", type=float, default=0.75)
    parser.add_argument("--trim-fraction", type=float, default=0.35)
    parser.add_argument("--min-matched-points", type=int, default=150)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.15)
    parser.add_argument("--min-improvement-ratio", type=float, default=0.20)
    parser.add_argument("--max-linearity", type=float, default=0.995)
    parser.add_argument("--max-camera-delta-sec", type=float, default=0.04)
    parser.add_argument("--temporal-window-frames", type=int, default=101)
    parser.add_argument("--max-translation-step-m", type=float, default=0.02)
    parser.add_argument("--max-yaw-step-deg", type=float, default=0.05)
    parser.add_argument("--skip-map-points", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--reuse-keyframe-registration", type=Path, default=None)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cumulative_distance_m(poses_map: np.ndarray, map_scale: float) -> np.ndarray:
    result = np.zeros(len(poses_map), dtype=np.float64)
    if len(poses_map) > 1:
        result[1:] = np.cumsum(np.linalg.norm(np.diff(poses_map[:, :2], axis=0), axis=1) / map_scale)
    return result


def scan_linearity(points: np.ndarray) -> float:
    if len(points) < 3:
        return 1.0
    values = np.linalg.eigvalsh(np.cov(points.T))
    return float(values[-1] / max(1e-12, values.sum()))


def load_map_scale(rows: list[dict[str, str]]) -> float:
    for row in rows:
        raw_path = row.get("map_points_path", "")
        if raw_path and resolve(raw_path).exists():
            with np.load(resolve(raw_path)) as data:
                return float(data["map_scale"][0])
    raise RuntimeError("no map point NPZ available to read map scale")


def select_evenly(indices: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum, dtype=int)
    return indices[np.unique(positions)]


def rejection_reason(result: Any, linearity: float, args: argparse.Namespace) -> str:
    reasons = []
    if not result.converged:
        reasons.append("not_converged")
    if result.matched_points < args.min_matched_points:
        reasons.append("insufficient_matches")
    if result.inlier_ratio < args.min_inlier_ratio:
        reasons.append("low_inlier_ratio")
    if result.improvement_ratio < args.min_improvement_ratio:
        reasons.append("insufficient_improvement")
    if linearity > args.max_linearity:
        reasons.append("linear_geometry_ambiguous")
    if math.hypot(result.dx_m, result.dy_m) >= 0.98 * args.max_translation_m:
        reasons.append("translation_at_search_bound")
    if abs(math.degrees(result.dyaw_rad)) >= 0.98 * args.max_yaw_deg:
        reasons.append("yaw_at_search_bound")
    return ";".join(reasons)


def local_register_keyframes(
    rows: list[dict[str, str]],
    poses: np.ndarray,
    key_indices: np.ndarray,
    travel: np.ndarray,
    map_scale: float,
    target_points: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    target_tree = cKDTree(target_points)
    records: list[dict[str, Any]] = []
    observed = np.full((len(key_indices), 3), np.nan, dtype=np.float64)
    accepted = np.zeros(len(key_indices), dtype=bool)
    confidence = np.zeros(len(key_indices), dtype=np.float64)
    radius_map = args.target_radius_m * map_scale

    for ordinal, row_index in enumerate(key_indices):
        row = rows[int(row_index)]
        pose = poses[int(row_index)]
        with np.load(resolve(row["map_points_path"])) as data:
            points = data["points_map_xyzi"].astype(np.float64)
        relative_m = (points[:, :2] - pose[:2]) / map_scale
        ranges = np.linalg.norm(relative_m, axis=1)
        mask = (
            (ranges >= args.scan_range_min_m)
            & (ranges <= args.scan_range_max_m)
            & (points[:, 2] >= args.scan_z_min_m)
            & (points[:, 2] <= args.scan_z_max_m)
        )
        scan = points[mask, :2]
        if len(scan) > args.scan_max_points:
            sample_indices = np.linspace(0, len(scan) - 1, args.scan_max_points, dtype=int)
            scan = scan[sample_indices]
        target_indices = target_tree.query_ball_point(pose[:2], radius_map)
        local_target = target_points[np.asarray(target_indices, dtype=np.int64)]
        centered_scan = scan - pose[:2]
        centered_target = local_target - pose[:2]
        linearity = scan_linearity(centered_scan)
        result = estimate_local_correction(
            centered_scan,
            centered_target,
            map_units_per_meter=map_scale,
            max_translation_m=args.max_translation_m,
            max_yaw_deg=args.max_yaw_deg,
            max_correspondence_m=args.max_correspondence_m,
            trim_fraction=args.trim_fraction,
        )
        reason = rejection_reason(result, linearity, args)
        is_accepted = not reason
        conf = min(
            1.0,
            result.matched_points / max(1.0, args.min_matched_points * 2.0),
            result.inlier_ratio / max(args.min_inlier_ratio * 2.0, 1e-9),
            result.improvement_ratio / max(args.min_improvement_ratio * 2.0, 1e-9),
        )
        if is_accepted:
            observed[ordinal] = [result.dx_m * map_scale, result.dy_m * map_scale, result.dyaw_rad]
            accepted[ordinal] = True
            confidence[ordinal] = conf
        records.append(
            {
                "keyframe_ordinal": ordinal,
                "row_index": int(row_index),
                "frame": int(row["frame"]),
                "travel_distance_m": float(travel[int(row_index)]),
                "map_x": float(pose[0]),
                "map_y": float(pose[1]),
                "map_yaw": float(pose[2]),
                "scan_points": int(len(scan)),
                "target_points": int(len(local_target)),
                "scan_linearity": linearity,
                "residual_before_m": result.residual_before_m,
                "residual_after_m": result.residual_after_m,
                "inlier_ratio": result.inlier_ratio,
                "matched_points": result.matched_points,
                "improvement_ratio": result.improvement_ratio,
                "raw_dx_m": result.dx_m,
                "raw_dy_m": result.dy_m,
                "raw_dyaw_deg": math.degrees(result.dyaw_rad),
                "accepted": int(is_accepted),
                "confidence": conf if is_accepted else 0.0,
                "rejection_reason": reason,
            }
        )
        if (ordinal + 1) % 20 == 0 or ordinal + 1 == len(key_indices):
            print(f"[registration] {ordinal + 1}/{len(key_indices)} accepted={int(accepted.sum())}", flush=True)
    return records, observed, accepted, confidence


def load_registration_records(path: Path, map_scale: float) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_rows = read_csv(path)
    integer_fields = {"keyframe_ordinal", "row_index", "frame", "scan_points", "target_points", "matched_points", "accepted"}
    text_fields = {"rejection_reason"}
    records: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if key in integer_fields:
                row[key] = int(value)
            elif key in text_fields:
                row[key] = value
            elif value != "":
                row[key] = float(value)
        records.append(row)
    key_indices = np.asarray([int(row["row_index"]) for row in records], dtype=np.int64)
    accepted = np.asarray([bool(row["accepted"]) for row in records], dtype=bool)
    confidence = np.asarray([float(row["confidence"]) for row in records], dtype=np.float64)
    observed = np.full((len(records), 3), np.nan, dtype=np.float64)
    for index in np.flatnonzero(accepted):
        row = records[int(index)]
        observed[index] = [
            float(row["raw_dx_m"]) * map_scale,
            float(row["raw_dy_m"]) * map_scale,
            math.radians(float(row["raw_dyaw_deg"])),
        ]
    return records, observed, accepted, confidence, key_indices


def evaluate_smoothed_keyframes(
    rows: list[dict[str, str]],
    poses: np.ndarray,
    key_indices: np.ndarray,
    smoothed_key: np.ndarray,
    map_scale: float,
    target_points: np.ndarray,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    target_tree = cKDTree(target_points)
    radius_map = args.target_radius_m * map_scale
    for ordinal, (row_index, correction) in enumerate(zip(key_indices, smoothed_key)):
        row = rows[int(row_index)]
        pose = poses[int(row_index)]
        with np.load(resolve(row["map_points_path"])) as data:
            points = data["points_map_xyzi"].astype(np.float64)
        centered = points[:, :2] - pose[:2]
        ranges = np.linalg.norm(centered / map_scale, axis=1)
        mask = (
            (ranges >= args.scan_range_min_m)
            & (ranges <= args.scan_range_max_m)
            & (points[:, 2] >= args.scan_z_min_m)
            & (points[:, 2] <= args.scan_z_max_m)
        )
        scan = centered[mask]
        if len(scan) > args.scan_max_points:
            sample_indices = np.linspace(0, len(scan) - 1, args.scan_max_points, dtype=int)
            scan = scan[sample_indices]
        target_indices = target_tree.query_ball_point(pose[:2], radius_map)
        local_target = target_points[np.asarray(target_indices, dtype=np.int64)] - pose[:2]
        c, s = math.cos(float(correction[2])), math.sin(float(correction[2]))
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        corrected_scan = scan @ rotation.T + correction[:2]
        metrics = evaluate_alignment(
            corrected_scan,
            local_target,
            map_units_per_meter=map_scale,
            max_correspondence_m=args.max_correspondence_m,
            trim_fraction=args.trim_fraction,
        )
        records[ordinal]["smoothed_residual_m"] = metrics.residual_m
        records[ordinal]["smoothed_inlier_ratio"] = metrics.inlier_ratio
        records[ordinal]["smoothed_matched_points"] = metrics.matched_points


def rebuild_dataset(
    rows: list[dict[str, Any]],
    original_poses: np.ndarray,
    corrected_poses: np.ndarray,
    map_scale: float,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    points_dir = output_dir / "map_points"
    points_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    for index, (source, pose, original_pose) in enumerate(zip(rows, corrected_poses, original_poses)):
        row = dict(source)
        frame = int(row["frame"])
        lidar_path = resolve(str(row["lidar_path"]))
        raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        projected = lidar_to_map_xyzi(raw, pose, map_scale)
        points_path = points_dir / f"{frame:06d}.npz"
        writer = np.savez_compressed if args.compress else np.savez
        writer(
            points_path,
            points_map_xyzi=projected,
            ego_map_pose=pose.astype(np.float32),
            original_ego_map_pose=original_pose.astype(np.float32),
            frame=np.array([frame], dtype=np.int32),
            map_scale=np.array([map_scale], dtype=np.float32),
        )
        row.update(
            {
                "original_image_path": row.get("image_path", ""),
                "image_path": row["camera_image_path"],
                "map_points_path": str(points_path),
                "original_map_x": float(original_pose[0]),
                "original_map_y": float(original_pose[1]),
                "original_map_yaw": float(original_pose[2]),
                "map_x": float(pose[0]),
                "map_y": float(pose[1]),
                "map_yaw": float(pose[2]),
                "pose_correction_dx_m": float((pose[0] - original_pose[0]) / map_scale),
                "pose_correction_dy_m": float((pose[1] - original_pose[1]) / map_scale),
                "pose_correction_dyaw_deg": math.degrees(float(pose[2] - original_pose[2])),
            }
        )
        output_rows.append(row)
        if (index + 1) % 250 == 0:
            print(f"[dataset] {index + 1}/{len(rows)}", flush=True)
    return output_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.corrected_dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.frames_csv)
    map_scale = load_map_scale(rows)
    poses = np.asarray([[float(r["map_x"]), float(r["map_y"]), float(r["map_yaw"])] for r in rows])
    frames = np.asarray([int(r["frame"]) for r in rows], dtype=np.int64)
    travel = cumulative_distance_m(poses, map_scale)

    camera_frames, camera_timestamps = load_timestamp_file(args.dataset_root / "image" / "timestamps.txt")
    synced_rows = synchronize_frame_rows(
        rows,
        camera_frames,
        camera_timestamps,
        image_dir=args.dataset_root / "image",
        max_delta_sec=args.max_camera_delta_sec,
    )
    camera_deltas = np.abs(np.asarray([float(row["camera_lidar_dt_sec"]) for row in synced_rows]))

    gltf = load_gltf_map(resolve(args.gltf), 0.025)
    layer_names = [name.strip() for name in args.layers.split(",") if name.strip()]
    map_layers = {name: gltf.layers[name].points for name in layer_names if name in gltf.layers}
    if not map_layers:
        raise RuntimeError(f"no GLTF layers found for {layer_names}")
    target_points = np.vstack(list(map_layers.values())).astype(np.float64)

    key_indices = select_keyframes(
        frames,
        poses,
        distance_m=args.keyframe_distance_m,
        yaw_deg=args.keyframe_yaw_deg,
        map_units_per_meter=map_scale,
    )
    key_indices = select_evenly(key_indices, args.max_keyframes)
    if args.reuse_keyframe_registration is not None:
        records, observed, accepted, confidence, key_indices = load_registration_records(
            args.reuse_keyframe_registration, map_scale
        )
        print(f"[registration] reused {len(records)} records from {args.reuse_keyframe_registration}", flush=True)
    else:
        records, observed, accepted, confidence = local_register_keyframes(
            rows, poses, key_indices, travel, map_scale, target_points, args
        )
    smoothed_key = smooth_keyframe_corrections(
        observed,
        accepted,
        confidence,
        positions=travel[key_indices],
    )
    key_translation_norms = np.linalg.norm(smoothed_key[:, :2], axis=1) / map_scale
    over_limit = key_translation_norms > args.max_translation_m
    if np.any(over_limit):
        smoothed_key[over_limit, :2] *= (
            args.max_translation_m / key_translation_norms[over_limit]
        )[:, None]
    smoothed_key[:, 2] = np.clip(
        smoothed_key[:, 2], -math.radians(args.max_yaw_deg), math.radians(args.max_yaw_deg)
    )
    dense_corrections = interpolate_corrections(frames, frames[key_indices], smoothed_key)
    dense_corrections = stabilize_dense_corrections(
        dense_corrections,
        map_units_per_meter=map_scale,
        window_frames=args.temporal_window_frames,
        max_translation_step_m=args.max_translation_step_m,
        max_yaw_step_deg=args.max_yaw_step_deg,
    )
    actual_key_corrections = dense_corrections[key_indices]
    evaluate_smoothed_keyframes(
        rows, poses, key_indices, actual_key_corrections, map_scale, target_points, records, args
    )
    corrected_poses = apply_pose_corrections(poses, dense_corrections)
    for record, smooth in zip(records, actual_key_corrections):
        record["smoothed_dx_m"] = float(smooth[0] / map_scale)
        record["smoothed_dy_m"] = float(smooth[1] / map_scale)
        record["smoothed_dyaw_deg"] = math.degrees(float(smooth[2]))

    correction_rows = []
    for frame, original, corrected, correction, distance in zip(frames, poses, corrected_poses, dense_corrections, travel):
        correction_rows.append(
            {
                "frame": int(frame),
                "travel_distance_m": float(distance),
                "original_map_x": float(original[0]),
                "original_map_y": float(original[1]),
                "original_map_yaw": float(original[2]),
                "corrected_map_x": float(corrected[0]),
                "corrected_map_y": float(corrected[1]),
                "corrected_map_yaw": float(corrected[2]),
                "correction_dx_m": float(correction[0] / map_scale),
                "correction_dy_m": float(correction[1] / map_scale),
                "correction_dyaw_deg": math.degrees(float(correction[2])),
            }
        )

    write_csv(args.output_dir / "keyframe_registration.csv", records)
    write_csv(args.output_dir / "pose_corrections.csv", correction_rows)
    write_csv(args.output_dir / "corrected_trajectory.csv", [
        {"frame": int(f), "x": float(p[0]), "y": float(p[1]), "yaw": float(p[2])}
        for f, p in zip(frames, corrected_poses)
    ])

    if args.skip_map_points:
        manifest_rows = []
    else:
        manifest_rows = rebuild_dataset(
            synced_rows, poses, corrected_poses, map_scale, args.corrected_dataset_dir, args
        )
        write_csv(args.corrected_dataset_dir / "frames.csv", manifest_rows)

    accepted_records = [row for row in records if row["accepted"]]
    before_values = np.asarray([row["residual_before_m"] for row in accepted_records], dtype=float)
    after_values = np.asarray([row["residual_after_m"] for row in accepted_records], dtype=float)
    smoothed_accepted_values = np.asarray([row["smoothed_residual_m"] for row in accepted_records], dtype=float)
    all_before_values = np.asarray([row["residual_before_m"] for row in records], dtype=float)
    all_smoothed_values = np.asarray([row["smoothed_residual_m"] for row in records], dtype=float)
    summary = {
        "pipeline": "pose_drift_correction_v1",
        "frame_count": len(rows),
        "keyframe_count": len(records),
        "accepted_keyframes": int(accepted.sum()),
        "rejected_keyframes": int(len(records) - accepted.sum()),
        "map_scale_units_per_meter": map_scale,
        "odometry_path_length_m": float(travel[-1]),
        "camera_sync": {
            "valid_count": int(sum(int(row["camera_match_valid"]) for row in synced_rows)),
            "abs_median_sec": float(np.median(camera_deltas)),
            "abs_p95_sec": float(np.quantile(camera_deltas, 0.95)),
            "abs_max_sec": float(camera_deltas.max()),
        },
        "accepted_structural_residual": {
            "before_median_m": float(np.median(before_values)) if len(before_values) else None,
            "local_icp_median_m": float(np.median(after_values)) if len(after_values) else None,
            "smoothed_trajectory_median_m": float(np.median(smoothed_accepted_values)) if len(smoothed_accepted_values) else None,
            "smoothed_relative_median_improvement": (
                float(1.0 - np.median(smoothed_accepted_values) / np.median(before_values))
                if len(before_values) and np.median(before_values) > 0
                else None
            ),
        },
        "all_keyframe_structural_residual": {
            "before_median_m": float(np.median(all_before_values)),
            "smoothed_trajectory_median_m": float(np.median(all_smoothed_values)),
            "smoothed_relative_median_improvement": float(
                1.0 - np.median(all_smoothed_values) / np.median(all_before_values)
            ),
        },
        "correction_extent": {
            "max_translation_m": float(np.max(np.linalg.norm(dense_corrections[:, :2], axis=1)) / map_scale),
            "max_abs_yaw_deg": float(np.max(np.abs(np.degrees(dense_corrections[:, 2])))),
        },
        "config": vars(args) | {"frames_csv": str(args.frames_csv), "gltf": str(args.gltf), "dataset_root": str(args.dataset_root), "output_dir": str(args.output_dir), "corrected_dataset_dir": str(args.corrected_dataset_dir)},
        "outputs": {
            "keyframe_registration": str(args.output_dir / "keyframe_registration.csv"),
            "pose_corrections": str(args.output_dir / "pose_corrections.csv"),
            "corrected_trajectory": str(args.output_dir / "corrected_trajectory.csv"),
            "corrected_frames": str(args.corrected_dataset_dir / "frames.csv") if manifest_rows else None,
        },
    }
    summary["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in summary["config"].items()}
    (args.output_dir / "drift_before_after.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if manifest_rows:
        metadata = dict(summary)
        metadata["description"] = "Timestamp-synchronized images and map points rebuilt from smoothed map-pose corrections."
        (args.corrected_dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plots = write_drift_plots(args.output_dir, records, poses, corrected_poses, map_layers)
    write_drift_report(args.output_dir / "drift_report.html", summary, records, plots)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
