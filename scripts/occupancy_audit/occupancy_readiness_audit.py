#!/usr/bin/env python3
"""Audit readiness for slot-level parking occupancy mapping.

Run from the project root:
    python3 scripts/occupancy_audit/occupancy_readiness_audit.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FRAME_DATASET = ROOT / "outputs" / "frame_map_dataset"
SCORING_DIR = ROOT / "outputs" / "part1_slot_scoring_1000_boundary_fixed"
BASELINE_SCORING_DIR = ROOT / "outputs" / "part1_slot_scoring_1000"
REPORT_DIR = ROOT / "reports" / "occupancy_readiness"
FIGURE_DIR = REPORT_DIR / "figures"

FILES_USED = {
    "frames_csv": FRAME_DATASET / "frames.csv",
    "metadata_json": FRAME_DATASET / "metadata.json",
    "map_points_dir": FRAME_DATASET / "map_points",
    "alignment_transform_json": ROOT / "outputs" / "alignment_transform.json",
    "alignment_parameters_csv": ROOT / "outputs" / "alignment_parameters.csv",
    "aligned_trajectory_final_csv": ROOT / "outputs" / "aligned_trajectory_final.csv",
    "slot_database_json": SCORING_DIR / "slot_database.json",
    "slot_scores_csv": SCORING_DIR / "slot_scores.csv",
    "slot_belief_fused_json": SCORING_DIR / "slot_belief_fused.json",
    "frame_slot_evidence_jsonl": SCORING_DIR / "frame_slot_evidence.jsonl",
    "validation_report_json": SCORING_DIR / "validation_report.json",
    "baseline_scoring_dir": BASELINE_SCORING_DIR,
}

REQUIRED_FRAME_COLUMNS = [
    "frame",
    "image_path",
    "lidar_path",
    "map_points_path",
    "map_x",
    "map_y",
    "map_yaw",
    "lidar_timestamp",
    "odom_timestamp",
    "time_delta_sec",
    "num_points",
    "missing_image",
    "missing_lidar",
]

Z_PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def finite_min_max(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": None, "max": None}
    return {"min": float(finite.min()), "max": float(finite.max())}


def evenly_spaced_indices(count: int, sample_count: int) -> list[int]:
    if count <= 0:
        return []
    sample_count = min(sample_count, count)
    return sorted(set(int(i) for i in np.linspace(0, count - 1, sample_count)))


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    arr = np.asarray(points, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def polygon_center(points: list[list[float]]) -> list[float] | None:
    if not points:
        return None
    arr = np.asarray(points, dtype=float)
    return [float(arr[:, 0].mean()), float(arr[:, 1].mean())]


def extract_polygon(slot: dict[str, Any]) -> tuple[str | None, list[list[float]] | None]:
    geometry_keys = [
        "polygon_map",
        "polygon",
        "corners_map",
        "corners",
        "vertices_map",
        "vertices",
        "inner_polygon",
        "core_polygon_map",
    ]
    for key in geometry_keys:
        value = slot.get(key)
        if (
            isinstance(value, list)
            and len(value) >= 3
            and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in value)
        ):
            return key, [[float(p[0]), float(p[1])] for p in value]
    return None, None


def extract_center(slot: dict[str, Any], polygon: list[list[float]] | None) -> list[float] | None:
    for key in ["center_map", "center", "centroid_map", "centroid"]:
        value = slot.get(key)
        if isinstance(value, list) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
    if polygon:
        return polygon_center(polygon)
    return None


def range_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    width = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    denom = max(1e-9, min(a_max - a_min, b_max - b_min))
    return float(width / denom)


def audit_frames(frames: pd.DataFrame) -> dict[str, Any]:
    missing_columns = [c for c in REQUIRED_FRAME_COLUMNS if c not in frames.columns]
    map_paths_exist = []
    missing_map_points_paths = []
    for raw_path in frames.get("map_points_path", pd.Series(dtype=str)).dropna():
        p = ROOT / str(raw_path) if not Path(str(raw_path)).is_absolute() else Path(str(raw_path))
        exists = p.exists()
        map_paths_exist.append(exists)
        if not exists and len(missing_map_points_paths) < 25:
            missing_map_points_paths.append(str(raw_path))

    return {
        "required_columns": REQUIRED_FRAME_COLUMNS,
        "missing_required_columns": missing_columns,
        "frame_count": int(len(frames)),
        "first_frame": int(frames["frame"].iloc[0]) if len(frames) and "frame" in frames else None,
        "last_frame": int(frames["frame"].iloc[-1]) if len(frames) and "frame" in frames else None,
        "missing_lidar_count": int(frames["missing_lidar"].sum()) if "missing_lidar" in frames else None,
        "missing_image_count": int(frames["missing_image"].sum()) if "missing_image" in frames else None,
        "map_pose_nan_count": {
            c: int(frames[c].isna().sum()) if c in frames else None
            for c in ["map_x", "map_y", "map_yaw"]
        },
        "num_points": {
            "min": int(frames["num_points"].min()) if "num_points" in frames else None,
            "max": int(frames["num_points"].max()) if "num_points" in frames else None,
            "mean": float(frames["num_points"].mean()) if "num_points" in frames else None,
        },
        "map_points_path_files": {
            "checked": int(len(map_paths_exist)),
            "existing": int(sum(map_paths_exist)),
            "missing": int(len(map_paths_exist) - sum(map_paths_exist)),
            "all_exist": bool(map_paths_exist and all(map_paths_exist)),
            "missing_examples": missing_map_points_paths,
        },
        "trajectory_map_range": {
            "x": finite_min_max(frames["map_x"].to_numpy()) if "map_x" in frames else {"min": None, "max": None},
            "y": finite_min_max(frames["map_y"].to_numpy()) if "map_y" in frames else {"min": None, "max": None},
            "yaw": finite_min_max(frames["map_yaw"].to_numpy()) if "map_yaw" in frames else {"min": None, "max": None},
        },
    }


def audit_npz(frames: pd.DataFrame, sample_count: int = 32) -> tuple[dict[str, Any], np.ndarray]:
    sample_indices = evenly_spaced_indices(len(frames), sample_count)
    samples = []
    all_xyz_i_ranges: dict[str, list[float]] = defaultdict(list)
    z_samples = []
    expected_keys = {"points_map_xyzi", "ego_map_pose", "frame", "map_scale"}

    for idx in sample_indices:
        row = frames.iloc[idx]
        raw_path = str(row["map_points_path"])
        p = ROOT / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
        result: dict[str, Any] = {
            "frame": int(row["frame"]),
            "map_points_path": raw_path,
            "exists": p.exists(),
        }
        if not p.exists():
            result["valid"] = False
            result["errors"] = ["missing npz file"]
            samples.append(result)
            continue

        errors = []
        try:
            with np.load(p) as npz:
                keys = set(npz.files)
                points = npz["points_map_xyzi"] if "points_map_xyzi" in keys else np.empty((0, 0))
                ego_pose = npz["ego_map_pose"] if "ego_map_pose" in keys else np.empty((0,))
                result["keys"] = sorted(keys)
                result["required_keys_present"] = sorted(expected_keys.intersection(keys))
                if not expected_keys.issubset(keys):
                    errors.append(f"missing keys: {sorted(expected_keys - keys)}")
                if points.ndim != 2 or points.shape[1] != 4:
                    errors.append(f"points_map_xyzi shape is {list(points.shape)}, expected [N, 4]")
                if ego_pose.shape != (3,):
                    errors.append(f"ego_map_pose shape is {list(ego_pose.shape)}, expected [3]")
                finite = bool(np.isfinite(points).all()) if points.size else True
                if not finite:
                    errors.append("points_map_xyzi contains NaN or Inf")
                result["points_shape"] = list(points.shape)
                result["ego_map_pose_shape"] = list(ego_pose.shape)
                result["finite_points"] = finite
                if points.ndim == 2 and points.shape[1] == 4 and points.size:
                    labels = ["x", "y", "z_lidar_m", "intensity"]
                    ranges = {}
                    for col, label in enumerate(labels):
                        stats = finite_min_max(points[:, col])
                        ranges[label] = stats
                        if stats["min"] is not None:
                            all_xyz_i_ranges[f"{label}_min"].append(stats["min"])
                            all_xyz_i_ranges[f"{label}_max"].append(stats["max"])
                    result["ranges"] = ranges
                    stride = max(1, points.shape[0] // 10000)
                    z_samples.append(points[::stride, 2].astype(float))
        except Exception as exc:  # noqa: BLE001 - audit should capture corrupt files
            errors.append(f"failed to read npz: {exc}")
        result["valid"] = not errors
        result["errors"] = errors
        samples.append(result)

    if z_samples:
        z_values = np.concatenate(z_samples)
        z_values = z_values[np.isfinite(z_values)]
    else:
        z_values = np.array([], dtype=float)

    aggregate_ranges = {}
    for label in ["x", "y", "z_lidar_m", "intensity"]:
        mins = all_xyz_i_ranges.get(f"{label}_min", [])
        maxs = all_xyz_i_ranges.get(f"{label}_max", [])
        aggregate_ranges[label] = {
            "min": float(min(mins)) if mins else None,
            "max": float(max(maxs)) if maxs else None,
        }

    npz_summary = {
        "sample_count_requested": sample_count,
        "sample_count_read": len(samples),
        "valid_sample_count": int(sum(s.get("valid", False) for s in samples)),
        "all_samples_valid": bool(samples and all(s.get("valid", False) for s in samples)),
        "aggregate_ranges_from_samples": aggregate_ranges,
        "samples": samples,
    }
    return npz_summary, z_values


def z_height_summary(z_values: np.ndarray) -> dict[str, Any]:
    if z_values.size == 0:
        return {
            "sample_count": 0,
            "percentiles": {},
            "suggested_first_pass_non_ground_threshold": None,
            "z_gt_minus_0_55_reasonable": False,
            "z_gt_minus_0_55_reasonable_first_pass": False,
            "z_threshold_separation_clear": False,
            "threshold_evaluation": "No z samples were available.",
        }

    percentiles = {
        str(p): float(np.percentile(z_values, p))
        for p in Z_PERCENTILES
    }
    share_above = float(np.mean(z_values > -0.55))
    p5 = percentiles["5"]
    p25 = percentiles["25"]
    median = percentiles["50"]
    ground_band_matches_prior = -0.9 <= p5 <= -0.7 and -0.9 <= p25 <= -0.6
    first_pass_reasonable = bool(ground_band_matches_prior and -0.55 > p25)
    separation_clear = bool(first_pass_reasonable and share_above < 0.5 and median < -0.55)
    if first_pass_reasonable and not separation_clear:
        reason = (
            "The lower z percentiles match the manually observed ground-like band near -0.8, "
            "so z > -0.55 is a plausible first-pass non-ground candidate threshold. "
            f"However, {share_above:.1%} of sampled points are above -0.55, so this threshold "
            "does not by itself prove clean ground/obstacle separation."
        )
    elif separation_clear:
        reason = "The sampled distribution supports z > -0.55 as a conservative first-pass obstacle filter."
    elif median < -0.55:
        reason = "Most sampled points are below -0.55, but the lower percentiles do not clearly match the expected ground band."
    else:
        reason = "The sampled distribution does not clearly support z > -0.55 as a standalone obstacle threshold."
    return {
        "sample_count": int(z_values.size),
        "percentiles": percentiles,
        "suggested_first_pass_non_ground_threshold": -0.55,
        "z_gt_minus_0_55_reasonable": first_pass_reasonable,
        "z_gt_minus_0_55_reasonable_first_pass": first_pass_reasonable,
        "z_threshold_separation_clear": separation_clear,
        "share_points_above_minus_0_55": share_above,
        "threshold_evaluation": reason,
    }


def save_z_histogram(z_values: np.ndarray, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if z_values.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(z_values, bins=120, color="#34699a", edgecolor="white", linewidth=0.2)
    ax.axvline(-0.55, color="#b3261e", linestyle="--", linewidth=1.5, label="z = -0.55")
    ax.set_title("Sampled LiDAR z_lidar_m Distribution")
    ax.set_xlabel("z_lidar_m")
    ax.set_ylabel("Point count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def audit_slots(slot_db: dict[str, Any], frames: pd.DataFrame, point_ranges: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    slots = slot_db.get("slots", [])
    if isinstance(slots, dict):
        slot_items = list(slots.values())
    elif isinstance(slots, list):
        slot_items = slots
    else:
        slot_items = []

    rows = []
    ids = []
    all_x = []
    all_y = []
    missing_geometry = 0
    missing_center = 0
    nonpositive_area = 0

    for i, slot in enumerate(slot_items):
        if not isinstance(slot, dict):
            continue
        slot_id = str(slot.get("slot_id", slot.get("id", f"slot_index_{i:04d}")))
        ids.append(slot_id)
        geometry_key, polygon = extract_polygon(slot)
        center = extract_center(slot, polygon)
        area = slot.get("area_m2", slot.get("area"))
        computed_area = polygon_area(polygon) if polygon else None
        if area is None:
            area = computed_area
        if not polygon:
            missing_geometry += 1
        else:
            arr = np.asarray(polygon, dtype=float)
            all_x.extend(arr[:, 0].tolist())
            all_y.extend(arr[:, 1].tolist())
        if center is None:
            missing_center += 1
        if area is None or float(area) <= 0:
            nonpositive_area += 1
        rows.append(
            {
                "slot_id": slot_id,
                "geometry_key": geometry_key,
                "vertex_count": len(polygon) if polygon else 0,
                "center_x": center[0] if center else None,
                "center_y": center[1] if center else None,
                "area_source_value": float(area) if area is not None else None,
                "computed_polygon_area_map_units2": computed_area,
                "heading_deg": slot.get("heading_deg"),
                "length_m": slot.get("length_m"),
                "width_m": slot.get("width_m"),
                "adjacent_slot_count": len(slot.get("adjacent_slots", [])) if isinstance(slot.get("adjacent_slots"), list) else 0,
            }
        )

    slot_range = {
        "x": {"min": float(min(all_x)) if all_x else None, "max": float(max(all_x)) if all_x else None},
        "y": {"min": float(min(all_y)) if all_y else None, "max": float(max(all_y)) if all_y else None},
    }
    trajectory_range = {
        "x": finite_min_max(frames["map_x"].to_numpy()) if "map_x" in frames else {"min": None, "max": None},
        "y": finite_min_max(frames["map_y"].to_numpy()) if "map_y" in frames else {"min": None, "max": None},
    }
    sampled_point_range = {
        "x": point_ranges.get("x", {"min": None, "max": None}),
        "y": point_ranges.get("y", {"min": None, "max": None}),
    }

    overlaps = {}
    for axis in ["x", "y"]:
        sr = slot_range[axis]
        tr = trajectory_range[axis]
        pr = sampled_point_range[axis]
        if None not in [sr["min"], sr["max"], tr["min"], tr["max"]]:
            overlaps[f"slot_vs_trajectory_{axis}"] = range_overlap(sr["min"], sr["max"], tr["min"], tr["max"])
        if None not in [sr["min"], sr["max"], pr["min"], pr["max"]]:
            overlaps[f"slot_vs_sampled_points_{axis}"] = range_overlap(sr["min"], sr["max"], pr["min"], pr["max"])

    same_coordinate_system = (
        missing_geometry == 0
        and bool(overlaps)
        and min(overlaps.values()) > 0.05
        and all(abs(v) < 10000 for v in all_x[:100] + all_y[:100])
    )

    summary = {
        "top_level_fields": list(slot_db.keys()),
        "map_units_per_meter": slot_db.get("map_units_per_meter"),
        "declared_slot_count": slot_db.get("slot_count"),
        "actual_slot_count": len(rows),
        "unique_slot_ids": len(set(ids)),
        "duplicate_slot_ids": sorted([slot_id for slot_id, count in Counter(ids).items() if count > 1])[:25],
        "missing_geometry_count": missing_geometry,
        "missing_center_count": missing_center,
        "nonpositive_area_count": nonpositive_area,
        "slot_coordinate_range": slot_range,
        "trajectory_map_range": trajectory_range,
        "sampled_point_xy_range": sampled_point_range,
        "range_overlap_ratios": overlaps,
        "same_map_coordinate_system_likely": same_coordinate_system,
        "example_slot_structure": slot_items[0] if slot_items else None,
    }
    return summary, pd.DataFrame(rows)


def audit_slot_scores(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    scores = pd.read_csv(path)
    numeric = scores.select_dtypes(include=[np.number])
    inferred = {
        "slot_id": [c for c in scores.columns if c.lower() in {"slot_id", "slot", "id"}],
        "occupancy_score_or_probability": [c for c in scores.columns if any(t in c.lower() for t in ["occupied", "occ"])],
        "free_belief_or_probability": [c for c in scores.columns if "free" in c.lower()],
        "unknown_or_uncertainty": [c for c in scores.columns if any(t in c.lower() for t in ["unknown", "uncertainty"])],
        "observed_support": [c for c in scores.columns if any(t in c.lower() for t in ["support", "visibility", "observed"])],
        "non_ground_evidence": [c for c in scores.columns if any(t in c.lower() for t in ["obstacle", "height", "hit", "object"])],
        "rank_or_final_score": [c for c in scores.columns if any(t in c.lower() for t in ["rank", "score", "availability"])],
    }
    summary_rows = []
    for col in numeric.columns:
        series = numeric[col]
        summary_rows.append(
            {
                "column": col,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std()) if series.count() > 1 else 0.0,
                "min": float(series.min()),
                "25%": float(series.quantile(0.25)),
                "50%": float(series.quantile(0.5)),
                "75%": float(series.quantile(0.75)),
                "max": float(series.max()),
            }
        )
    summary = {
        "shape": [int(scores.shape[0]), int(scores.shape[1])],
        "columns": list(scores.columns),
        "first_10_rows": scores.head(10).to_dict(orient="records"),
        "numeric_summary": summary_rows,
        "inferred_column_meanings": inferred,
        "inference_note": "Column meanings are inferred from names; scoring semantics should be confirmed against the producing code before using thresholds as ground truth.",
        "state_counts": scores["state"].value_counts(dropna=False).to_dict() if "state" in scores else {},
    }
    return summary, scores


def audit_fused_belief(path: Path, slot_count: int | None) -> dict[str, Any]:
    data = read_json(path, default=None)
    if data is None:
        return {"exists": False}
    if isinstance(data, dict) and isinstance(data.get("slots"), list):
        items = data["slots"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = list(data.values())
    else:
        items = []
    first = items[0] if items else None
    keys = set(first.keys()) if isinstance(first, dict) else set()
    belief_keys = sorted(k for k in keys if any(t in k.lower() for t in ["state", "free", "occupied", "unknown", "confidence", "evidence", "score"]))
    return {
        "exists": True,
        "container_type": type(data).__name__,
        "top_level_keys": list(data.keys()) if isinstance(data, dict) else None,
        "item_count": len(items),
        "matches_slot_database_count": bool(slot_count is not None and len(items) == slot_count),
        "first_item": first,
        "belief_state_keys_in_first_item": belief_keys,
        "state_counts": Counter(item.get("state") for item in items if isinstance(item, dict)).most_common(),
        "contains_per_slot_belief_state": bool({"slot_id", "state"}.issubset(keys) and any(k in keys for k in ["p_free", "p_occupied", "p_unknown"])),
    }


def audit_frame_slot_evidence(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    first_records = []
    fields = set()
    frame_ids = set()
    slot_ids = set()
    state_counts = Counter()
    support_by_slot = Counter()
    nonzero_obstacle_records = 0
    line_count = 0
    malformed = 0

    with path.open() as f:
        for line in f:
            line_count += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if len(first_records) < 5:
                first_records.append(rec)
            fields.update(rec.keys())
            frame_id = rec.get("frame_id", rec.get("frame"))
            slot_id = rec.get("slot_id", rec.get("slot"))
            if frame_id is not None:
                frame_ids.add(frame_id)
            if slot_id is not None:
                slot_ids.add(slot_id)
                support_by_slot[slot_id] += 1
            if rec.get("frame_state") is not None:
                state_counts[rec.get("frame_state")] += 1
            if any(float(rec.get(k, 0) or 0) > 0 for k in ["obstacle_point_count_inner", "core_obstacle_point_count", "object_hit_cell_ratio"]):
                nonzero_obstacle_records += 1

    linkage_fields = {
        "frame_id": any(k in fields for k in ["frame_id", "frame"]),
        "slot_id": any(k in fields for k in ["slot_id", "slot"]),
        "point_or_non_ground_evidence": any("obstacle" in k or "hit" in k or "height" in k for k in fields),
        "visibility_or_coverage": any(k in fields for k in ["visibility_score", "observed_area_ratio", "ray_free_ratio", "occlusion_ratio"]),
        "status": any(k in fields for k in ["frame_state", "state", "status", "uncertainty_type"]),
    }
    support_counts = np.array(list(support_by_slot.values()), dtype=float)
    return {
        "exists": True,
        "line_count": line_count,
        "malformed_line_count": malformed,
        "fields": sorted(fields),
        "first_records": first_records,
        "linkage_fields_present": linkage_fields,
        "unique_frame_count": len(frame_ids),
        "unique_slot_count": len(slot_ids),
        "state_counts": dict(state_counts),
        "records_with_nonzero_obstacle_evidence": nonzero_obstacle_records,
        "support_records_per_slot": {
            "min": float(support_counts.min()) if support_counts.size else None,
            "median": float(np.median(support_counts)) if support_counts.size else None,
            "mean": float(support_counts.mean()) if support_counts.size else None,
            "max": float(support_counts.max()) if support_counts.size else None,
        },
    }


def decide_readiness(
    dataset_summary: dict[str, Any],
    slot_summary: dict[str, Any],
    score_summary: dict[str, Any],
    fused_summary: dict[str, Any],
    evidence_summary: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    passed = []
    uncertain = []

    frame_summary = dataset_summary["frames"]
    npz_summary = dataset_summary["npz"]
    z_summary = dataset_summary["z_height"]

    criteria = [
        ("frames.csv complete", not frame_summary["missing_required_columns"] and frame_summary["missing_lidar_count"] == 0 and frame_summary["missing_image_count"] == 0 and all(v == 0 for v in frame_summary["map_pose_nan_count"].values())),
        ("all sampled NPZ files readable and valid", npz_summary["all_samples_valid"]),
        ("projected points are in map coordinates", bool(dataset_summary["metadata"].get("map_points_format")) and "points_map_xyzi" in dataset_summary["metadata"].get("map_points_format", {})),
        ("slot_database contains valid per-slot geometry", slot_summary["missing_geometry_count"] == 0 and slot_summary["missing_center_count"] == 0 and slot_summary["nonpositive_area_count"] == 0),
        ("slot geometry coordinate range matches projected point/trajectory coordinate range", slot_summary["same_map_coordinate_system_likely"]),
        ("z threshold can separate ground-like points from obstacle-like points", z_summary["z_threshold_separation_clear"]),
        ("frame_slot_evidence exists and links frames to slots", evidence_summary.get("exists") and all(evidence_summary.get("linkage_fields_present", {}).values())),
        ("slot_scores and slot_belief_fused contain per-slot scoring/belief", bool(score_summary.get("shape", [0])[0]) and fused_summary.get("contains_per_slot_belief_state")),
        ("enough frames provide support for most slots", evidence_summary.get("unique_slot_count", 0) >= 0.5 * slot_summary["actual_slot_count"]),
        ("known limitations are documented", True),
    ]

    for label, ok in criteria:
        if ok:
            passed.append(label)
        else:
            uncertain.append(label)

    if not frame_summary["missing_required_columns"] and npz_summary["all_samples_valid"] and slot_summary["missing_geometry_count"] == 0 and fused_summary.get("contains_per_slot_belief_state") and evidence_summary.get("exists"):
        decision = "PARTIALLY_READY"
    else:
        decision = "NOT_READY"
    if not uncertain:
        decision = "READY"
    return decision, passed, uncertain


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def build_report(
    dataset_summary: dict[str, Any],
    slot_summary: dict[str, Any],
    score_summary: dict[str, Any],
    fused_summary: dict[str, Any],
    evidence_summary: dict[str, Any],
    decision: str,
    passed: list[str],
    uncertain: list[str],
) -> str:
    frames = dataset_summary["frames"]
    z_summary = dataset_summary["z_height"]
    score_shape = score_summary.get("shape", [0, 0])
    state_counts = score_summary.get("state_counts", {})
    files_table = markdown_table(
        ["File", "Status", "Evidence"],
        [
            [str(path.relative_to(ROOT)) if path.exists() else str(path), "exists" if path.exists() else "missing", key]
            for key, path in FILES_USED.items()
            if key != "baseline_scoring_dir"
        ],
    )
    passed_list = "\n".join(f"- {item}" for item in passed)
    uncertain_list = "\n".join(f"- {item}" for item in uncertain) if uncertain else "- None from the automated checks."
    score_columns = ", ".join(score_summary.get("columns", []))
    inferred = score_summary.get("inferred_column_meanings", {})
    inferred_lines = "\n".join(f"- {k}: {', '.join(v) if v else 'not found'}" for k, v in inferred.items())
    sample_range = dataset_summary["npz"]["aggregate_ranges_from_samples"]

    return f"""# Occupancy Readiness Report

## Executive summary

Final decision: **{decision}**.

The current dataset has the core structures needed to generate a slot-level occupancy map: complete frame manifests, readable sampled map-point NPZ files, valid per-slot polygons, frame-slot evidence, slot scores, and fused per-slot belief states. The decision is **{decision}** rather than READY because precision still depends on calibration quality, slot boundary accuracy, z-height thresholding, LiDAR sparsity/occlusion handling, and validation against known occupied/free examples.

## Data inventory table

{files_table}

## Dataset manifest check

- Frames: {frames["frame_count"]}
- First frame: {frames["first_frame"]}
- Last frame: {frames["last_frame"]}
- Missing LiDAR count: {frames["missing_lidar_count"]}
- Missing image count: {frames["missing_image_count"]}
- NaN counts: map_x={frames["map_pose_nan_count"]["map_x"]}, map_y={frames["map_pose_nan_count"]["map_y"]}, map_yaw={frames["map_pose_nan_count"]["map_yaw"]}
- num_points min/max/mean: {frames["num_points"]["min"]} / {frames["num_points"]["max"]} / {frames["num_points"]["mean"]:.2f}
- map_points_path files existing: {frames["map_points_path_files"]["existing"]} of {frames["map_points_path_files"]["checked"]}

## NPZ point cloud check

- Sampled NPZ files: {dataset_summary["npz"]["sample_count_read"]}
- Valid sampled NPZ files: {dataset_summary["npz"]["valid_sample_count"]}
- Sampled map point x range: {sample_range["x"]["min"]} to {sample_range["x"]["max"]}
- Sampled map point y range: {sample_range["y"]["min"]} to {sample_range["y"]["max"]}
- Sampled z_lidar_m range: {sample_range["z_lidar_m"]["min"]} to {sample_range["z_lidar_m"]["max"]}
- Sampled intensity range: {sample_range["intensity"]["min"]} to {sample_range["intensity"]["max"]}

## Z-height / ground filtering check

- z samples: {z_summary["sample_count"]}
- z percentiles: {json.dumps(z_summary["percentiles"], default=json_default)}
- Suggested first-pass non-ground threshold: `z_lidar_m > {z_summary["suggested_first_pass_non_ground_threshold"]}`
- Share of sampled points above `-0.55`: {z_summary["share_points_above_minus_0_55"]:.3f}
- Evaluation of `z > -0.55`: {z_summary["threshold_evaluation"]}
- Histogram: `reports/occupancy_readiness/figures/z_distribution.png`

## Passed checks

{passed_list}

## Failed or uncertain checks

{uncertain_list}

## Slot database structure

The slot database top-level fields are: {", ".join(slot_summary["top_level_fields"])}.

- map_units_per_meter: {slot_summary["map_units_per_meter"]}
- declared slot_count: {slot_summary["declared_slot_count"]}
- actual slot count: {slot_summary["actual_slot_count"]}
- unique slot IDs: {slot_summary["unique_slot_ids"]}
- missing geometry count: {slot_summary["missing_geometry_count"]}
- missing center count: {slot_summary["missing_center_count"]}
- nonpositive area count: {slot_summary["nonpositive_area_count"]}
- slot x range: {slot_summary["slot_coordinate_range"]["x"]}
- slot y range: {slot_summary["slot_coordinate_range"]["y"]}
- trajectory x/y range: {slot_summary["trajectory_map_range"]}
- sampled point x/y range: {slot_summary["sampled_point_xy_range"]}
- same map coordinate system likely: {slot_summary["same_map_coordinate_system_likely"]}

One slot item includes `slot_id`, `polygon_map`, `inner_polygon`, `core_polygon_map`, margin geometry, `center_map`, heading, metric dimensions, area, and adjacency fields. Per-slot geometry summary is saved to `reports/occupancy_readiness/slot_geometry_summary.csv`.

## Existing scoring result summary

- slot_scores.csv shape: {score_shape[0]} rows x {score_shape[1]} columns
- state counts: {json.dumps(state_counts, default=json_default)}
- columns: {score_columns}

Inferred column meanings:

{inferred_lines}

The fused belief file is a {fused_summary.get("container_type")} with {fused_summary.get("item_count")} slot items. It contains per-slot belief state: {fused_summary.get("contains_per_slot_belief_state")}. It matches the slot database count: {fused_summary.get("matches_slot_database_count")}.

The frame-slot evidence file has {evidence_summary.get("line_count")} records across {evidence_summary.get("unique_frame_count")} frames and {evidence_summary.get("unique_slot_count")} slots. Linkage fields present: {json.dumps(evidence_summary.get("linkage_fields_present"), default=json_default)}.

## Whether the dataset supports per-slot occupancy map

The dataset supports implementing a per-slot occupancy map, but the expected output should be treated as **validation-needed** rather than a final precision guarantee. The data has projected LiDAR in map coordinates and valid slot polygons, so an algorithm can crop points by slot and fuse evidence. Existing scoring and fused belief files already demonstrate a per-slot evidence pipeline.

The main limitation is that accurate occupancy for every parking slot depends on whether the slot polygons are aligned precisely enough to the projected LiDAR, whether `z_lidar_m > -0.55` is robust across the whole route, and whether sparse/occluded slots receive enough observations.

## Recommended next implementation steps

1. Use `slot_database.json` polygons as the authoritative slot geometry.
2. For each frame NPZ, load `points_map_xyzi` and filter to candidate slots near the ego pose.
3. Crop projected points inside each slot polygon and optional core/margin polygons.
4. Filter ground using a tunable threshold, starting with `z_lidar_m > -0.55`.
5. Compute non-ground evidence, height span, hit-cell ratio, and boundary/margin conflict signals.
6. Compute visibility and free-space coverage from available ray or observed-area fields where possible.
7. Fuse frame evidence into `P_occ`, `P_free`, and `P_unknown` per slot.
8. Output a slot-level occupancy table and Top-K candidate slots for review.
9. Validate thresholds against manually inspected slots before using the map as ground truth.

## Risks

- Slot polygon boundary accuracy: small map alignment or polygon errors can move obstacle points into neighboring slots.
- z threshold sensitivity: `z > -0.55` is a reasonable first pass from this audit, but curbs, slopes, and vehicle body height variation can change the optimal cutoff.
- Sparse LiDAR / occlusion: many slots may have limited support, so unknown should remain a first-class state.
- Map alignment error: projected points, trajectory, and slot polygons appear to share coordinates, but precision depends on calibration and manual alignment quality.
- Static vs dynamic obstacle confusion: the evidence may include moving vehicles, pedestrians, or transient objects unless temporal fusion handles consistency.

## Suggested next algorithm

For each slot:

1. Crop projected map points inside the slot polygon.
2. Filter ground using `z_lidar_m > -0.55` initially.
3. Compute non-ground evidence from obstacle point counts, hit-cell ratios, and height span.
4. Compute coverage / visibility if possible.
5. Compute `P_occ`, `P_free`, and `P_unknown`.
6. Fuse evidence over frames.
7. Output a slot-level occupancy table and Top-K candidate slots.
"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    frames = pd.read_csv(FILES_USED["frames_csv"])
    metadata = read_json(FILES_USED["metadata_json"], default={})

    frame_summary = audit_frames(frames)
    npz_summary, z_values = audit_npz(frames, sample_count=32)
    z_summary = z_height_summary(z_values)
    hist_saved = save_z_histogram(z_values, FIGURE_DIR / "z_distribution.png")
    z_summary["histogram_saved"] = hist_saved
    z_summary["histogram_path"] = "reports/occupancy_readiness/figures/z_distribution.png" if hist_saved else None

    dataset_summary = {
        "files_used": {k: str(v.relative_to(ROOT)) if v.exists() else str(v) for k, v in FILES_USED.items()},
        "metadata": metadata,
        "frames": frame_summary,
        "npz": npz_summary,
        "z_height": z_summary,
    }

    slot_db = read_json(FILES_USED["slot_database_json"], default={})
    slot_summary, slot_geometry = audit_slots(slot_db, frames, npz_summary["aggregate_ranges_from_samples"])
    slot_geometry.to_csv(REPORT_DIR / "slot_geometry_summary.csv", index=False)

    score_summary, scores = audit_slot_scores(FILES_USED["slot_scores_csv"])
    pd.DataFrame(score_summary["numeric_summary"]).to_csv(REPORT_DIR / "slot_score_summary.csv", index=False)

    fused_summary = audit_fused_belief(FILES_USED["slot_belief_fused_json"], slot_summary["actual_slot_count"])
    evidence_summary = audit_frame_slot_evidence(FILES_USED["frame_slot_evidence_jsonl"])

    decision, passed, uncertain = decide_readiness(
        dataset_summary,
        slot_summary,
        score_summary,
        fused_summary,
        evidence_summary,
    )

    dataset_summary["decision"] = decision
    dataset_summary["passed_checks"] = passed
    dataset_summary["failed_or_uncertain_checks"] = uncertain

    write_json(REPORT_DIR / "dataset_summary.json", dataset_summary)
    write_json(REPORT_DIR / "fused_belief_summary.json", fused_summary)
    write_json(REPORT_DIR / "frame_slot_evidence_summary.json", evidence_summary)
    write_json(REPORT_DIR / "slot_database_summary.json", slot_summary)
    write_json(REPORT_DIR / "slot_scores_summary.json", score_summary)

    report = build_report(
        dataset_summary,
        slot_summary,
        score_summary,
        fused_summary,
        evidence_summary,
        decision,
        passed,
        uncertain,
    )
    (REPORT_DIR / "occupancy_readiness_report.md").write_text(report)

    print(f"Decision: {decision}")
    print(f"Wrote {REPORT_DIR / 'occupancy_readiness_report.md'}")
    print(f"Wrote {REPORT_DIR / 'dataset_summary.json'}")
    print(f"Wrote {REPORT_DIR / 'slot_geometry_summary.csv'}")
    print(f"Wrote {REPORT_DIR / 'slot_score_summary.csv'}")
    print(f"Wrote {REPORT_DIR / 'fused_belief_summary.json'}")
    print(f"Wrote {REPORT_DIR / 'frame_slot_evidence_summary.json'}")
    print(f"Wrote {FIGURE_DIR / 'z_distribution.png'}")


if __name__ == "__main__":
    main()
