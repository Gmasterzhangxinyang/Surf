#!/usr/bin/env python3
"""Audit ParkingAgent dataset completeness across linked data types."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PROJECT_ROOT = Path("/home/ParkingAgent/ParkingAgent")
DEFAULT_OUTPUT_DIR = Path("reports/dataset_integrity_linkage")
REQUIRED_NPZ_KEYS = {"points_map_xyzi", "ego_map_pose", "frame", "map_scale"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit linked ParkingAgent dataset completeness")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--npz-samples", type=int, default=25)
    parser.add_argument("--check-all-npz", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def to_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def frame_set(rows: list[dict[str, str]]) -> set[int]:
    frames: set[int] = set()
    for row in rows:
        frame = to_int(row.get("frame"))
        if frame is not None:
            frames.add(frame)
    return frames


def choose_npz_rows(rows: list[dict[str, str]], sample_count: int, check_all: bool) -> list[dict[str, str]]:
    if check_all or sample_count <= 0 or sample_count >= len(rows):
        return rows
    if not rows:
        return []
    indexes = {0, len(rows) - 1, len(rows) // 2}
    if sample_count > 3:
        denominator = max(sample_count - 1, 1)
        for idx in range(sample_count):
            indexes.add(round(idx * (len(rows) - 1) / denominator))
    return [rows[idx] for idx in sorted(indexes)]


def parse_frame_from_name(path: Path, prefix: str = "", suffix: str = "") -> int | None:
    name = path.name
    if prefix and name.startswith(prefix):
        name = name[len(prefix) :]
    if suffix and name.endswith(suffix):
        name = name[: -len(suffix)]
    match = re.search(r"(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def audit_frames(project_root: Path, frames_rows: list[dict[str, str]], pose_frames: set[int], aligned_frames: set[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    frames: list[int] = []
    num_points_sum = 0
    missing_flag_images = 0
    missing_flag_lidars = 0

    for row in frames_rows:
        issues: list[str] = []
        frame = to_int(row.get("frame"))
        if frame is None:
            issues.append("bad_frame_id")
            frame = -1
        else:
            frames.append(frame)

        image_path = resolve_path(project_root, row.get("image_path", ""))
        lidar_path = resolve_path(project_root, row.get("lidar_path", ""))
        map_points_path = resolve_path(project_root, row.get("map_points_path", ""))
        image_exists = image_path.exists()
        lidar_exists = lidar_path.exists()
        map_points_exists = map_points_path.exists()
        pose_exists = frame in pose_frames
        aligned_pose_exists = frame in aligned_frames

        missing_image_flag = to_int(row.get("missing_image"), 1)
        missing_lidar_flag = to_int(row.get("missing_lidar"), 1)
        num_points = to_int(row.get("num_points"), -1)
        if missing_image_flag:
            missing_flag_images += 1
            issues.append("csv_missing_image_flag")
        if missing_lidar_flag:
            missing_flag_lidars += 1
            issues.append("csv_missing_lidar_flag")
        if not image_exists:
            issues.append("image_missing")
        if not lidar_exists:
            issues.append("lidar_missing")
        if not map_points_exists:
            issues.append("map_points_missing")
        if not pose_exists:
            issues.append("pose_missing")
        if not aligned_pose_exists:
            issues.append("aligned_pose_missing")
        if num_points is None or num_points < 0:
            issues.append("bad_num_points")
            num_points = 0
        else:
            num_points_sum += num_points
        for key in ("map_x", "map_y", "map_yaw"):
            if to_float(row.get(key)) is None:
                issues.append(f"bad_{key}")

        audit_rows.append(
            {
                "frame": frame,
                "image_exists": int(image_exists),
                "lidar_exists": int(lidar_exists),
                "map_points_exists": int(map_points_exists),
                "pose_exists": int(pose_exists),
                "aligned_pose_exists": int(aligned_pose_exists),
                "csv_missing_image_flag": missing_image_flag,
                "csv_missing_lidar_flag": missing_lidar_flag,
                "num_points": num_points,
                "status": "ok" if not issues else "bad",
                "issues": "|".join(issues),
            }
        )

    duplicate_frames = sorted(frame for frame, count in Counter(frames).items() if count > 1)
    missing_frame_numbers: list[int] = []
    if frames:
        expected = set(range(min(frames), max(frames) + 1))
        missing_frame_numbers = sorted(expected - set(frames))
    bad_rows = sum(1 for row in audit_rows if row["status"] != "ok")
    summary = {
        "rows": len(frames_rows),
        "first_frame": min(frames) if frames else None,
        "last_frame": max(frames) if frames else None,
        "continuous_frame_range": not missing_frame_numbers,
        "missing_frame_number_count": len(missing_frame_numbers),
        "missing_frame_numbers_first20": missing_frame_numbers[:20],
        "duplicate_frame_count": len(duplicate_frames),
        "duplicate_frames_first20": duplicate_frames[:20],
        "bad_rows": bad_rows,
        "bad_rows_first20": [row["frame"] for row in audit_rows if row["status"] != "ok"][:20],
        "missing_image_flags": missing_flag_images,
        "missing_lidar_flags": missing_flag_lidars,
        "num_points_sum": num_points_sum,
        "status": "PASS" if bad_rows == 0 and not missing_frame_numbers and not duplicate_frames else "FAIL",
    }
    return audit_rows, summary


def validate_npz(project_root: Path, frames_rows: list[dict[str, str]], sample_count: int, check_all: bool) -> dict[str, Any]:
    samples = choose_npz_rows(frames_rows, sample_count, check_all)
    errors: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for row in samples:
        frame = to_int(row.get("frame"), -1)
        expected_points = to_int(row.get("num_points"), -1)
        path = resolve_path(project_root, row.get("map_points_path", ""))
        issue_list: list[str] = []
        details: dict[str, Any] = {"frame": frame, "path": str(path)}
        try:
            with np.load(path) as data:
                keys = set(data.files)
                missing_keys = sorted(REQUIRED_NPZ_KEYS - keys)
                if missing_keys:
                    issue_list.append("missing_keys:" + ",".join(missing_keys))
                points = data["points_map_xyzi"] if "points_map_xyzi" in data else None
                pose = data["ego_map_pose"] if "ego_map_pose" in data else None
                npz_frame = None
                if "frame" in data:
                    npz_frame = int(np.asarray(data["frame"]).reshape(-1)[0])
                    if npz_frame != frame:
                        issue_list.append("npz_frame_mismatch")
                if points is None or points.ndim != 2 or points.shape[1] != 4:
                    issue_list.append("bad_points_shape")
                else:
                    details["points_shape"] = list(points.shape)
                    details["points_dtype"] = str(points.dtype)
                    if expected_points is not None and expected_points >= 0 and int(points.shape[0]) != expected_points:
                        issue_list.append("num_points_mismatch")
                    if not np.isfinite(points[: min(len(points), 1000)]).all():
                        issue_list.append("non_finite_points_sample")
                if pose is None or tuple(pose.shape) != (3,):
                    issue_list.append("bad_ego_pose_shape")
                else:
                    details["ego_pose_shape"] = list(pose.shape)
                details["npz_frame"] = npz_frame
                details["keys"] = sorted(keys)
        except Exception as exc:  # pragma: no cover - exercised by real corrupt files
            issue_list.append(f"npz_read_error:{type(exc).__name__}:{exc}")

        details["status"] = "ok" if not issue_list else "bad"
        details["issues"] = "|".join(issue_list)
        sample_rows.append(details)
        if issue_list:
            errors.append(details)

    return {
        "mode": "all" if check_all else "sample",
        "requested_samples": sample_count,
        "checked_files": len(samples),
        "error_count": len(errors),
        "errors_first10": errors[:10],
        "samples_first5": sample_rows[:5],
        "status": "PASS" if not errors else "FAIL",
    }


def audit_slots(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = project_root / "outputs" / "full_icpark_allframes_vehicle_cluster"
    slot_db_path = base / "slot_database.json"
    scores_path = base / "slot_scores.csv"
    decision_path = base / "slot_decision_table" / "slot_decision_table.csv"

    slot_db = load_json(slot_db_path, {"slots": []})
    slot_ids = {str(slot.get("slot_id")) for slot in slot_db.get("slots", []) if slot.get("slot_id")}
    scores = {row.get("slot_id", ""): row for row in read_csv(scores_path) if row.get("slot_id")}
    decisions = {row.get("slot_id", ""): row for row in read_csv(decision_path) if row.get("slot_id")}

    all_slot_ids = sorted(slot_ids | set(scores) | set(decisions))
    audit_rows: list[dict[str, Any]] = []
    for slot_id in all_slot_ids:
        issues: list[str] = []
        in_slot_database = slot_id in slot_ids
        in_part1_scores = slot_id in scores
        in_decision_table = slot_id in decisions
        if not in_slot_database:
            issues.append("missing_slot_database")
        if not in_part1_scores:
            issues.append("missing_part1_score")
        if not in_decision_table:
            issues.append("missing_decision_table")
        audit_rows.append(
            {
                "slot_id": slot_id,
                "in_slot_database": int(in_slot_database),
                "in_part1_scores": int(in_part1_scores),
                "in_decision_table": int(in_decision_table),
                "part1_state": scores.get(slot_id, {}).get("state", ""),
                "final_decision": decisions.get(slot_id, {}).get("final_decision", ""),
                "status": "ok" if not issues else "bad",
                "issues": "|".join(issues),
            }
        )

    bad_rows = sum(1 for row in audit_rows if row["status"] != "ok")
    summary = {
        "slot_database_exists": slot_db_path.exists(),
        "part1_scores_exists": scores_path.exists(),
        "decision_table_exists": decision_path.exists(),
        "slot_count": len(slot_ids),
        "slot_unique_ids": len(slot_ids),
        "part1_score_rows": len(scores),
        "decision_table_rows": len(decisions),
        "bad_rows": bad_rows,
        "bad_slots_first20": [row["slot_id"] for row in audit_rows if row["status"] != "ok"][:20],
        "part1_state_counts": dict(Counter(row.get("state", "") for row in scores.values())),
        "final_decision_counts": dict(Counter(row.get("final_decision", "") for row in decisions.values())),
        "status": "PASS" if bad_rows == 0 and slot_ids and len(slot_ids) == len(scores) == len(decisions) else "FAIL",
    }
    return audit_rows, summary


def raw_dataset_context(project_root: Path, metadata: dict[str, Any], frame_summary: dict[str, Any]) -> dict[str, Any]:
    dataset_root = Path(str(metadata.get("dataset_root", "")))
    image_dir = dataset_root / "image"
    lidar_dir = dataset_root / "velodyne"
    image_files = sorted(image_dir.glob("left*.png")) if image_dir.exists() else []
    lidar_files = sorted(lidar_dir.glob("*.bin")) if lidar_dir.exists() else []
    first_frame = frame_summary.get("first_frame")
    image_frames = [value for value in (parse_frame_from_name(path, prefix="left", suffix=".png") for path in image_files) if value is not None]
    lidar_frames = [value for value in (parse_frame_from_name(path, suffix=".bin") for path in lidar_files) if value is not None]

    def excluded_before(frames: list[int]) -> list[int]:
        if first_frame is None:
            return []
        return sorted(frame for frame in frames if frame < int(first_frame))

    return {
        "dataset_root": str(dataset_root),
        "image_dir_exists": image_dir.exists(),
        "lidar_dir_exists": lidar_dir.exists(),
        "raw_image_count": len(image_files),
        "raw_lidar_count": len(lidar_files),
        "raw_image_frame_min": min(image_frames) if image_frames else None,
        "raw_image_frame_max": max(image_frames) if image_frames else None,
        "raw_lidar_frame_min": min(lidar_frames) if lidar_frames else None,
        "raw_lidar_frame_max": max(lidar_frames) if lidar_frames else None,
        "excluded_raw_image_frames_before_dataset_first": excluded_before(image_frames)[:20],
        "excluded_raw_lidar_frames_before_dataset_first": excluded_before(lidar_frames)[:20],
        "note": "Raw frames before the aligned frame range are excluded, not missing, when pose/alignment tables also start later.",
    }


def metadata_checks(metadata: dict[str, Any], frame_summary: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "selected_rows_matches_frames": metadata.get("selected_rows") == frame_summary.get("rows"),
        "processed_lidar_frames_matches_frames": metadata.get("processed_lidar_frames") == frame_summary.get("rows"),
        "missing_lidar_is_zero": metadata.get("missing_lidar") == 0,
        "missing_image_is_zero": metadata.get("missing_image") == 0,
        "total_projected_points_matches_sum": metadata.get("total_projected_points") == frame_summary.get("num_points_sum"),
    }
    checks["status"] = "PASS" if all(checks.values()) else "FAIL"
    return checks


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    frame_checks = summary["frame_checks"]
    slot_checks = summary["slot_checks"]
    npz_checks = summary["npz_sample_validation"]
    raw_context = summary["raw_dataset_context"]
    metadata = summary["metadata_checks"]
    status_text = "完整" if summary["overall_status"] == "PASS" else "存在问题"
    lines = [
        "# ParkingAgent 数据完整性与链路检查报告",
        "",
        "## 1. 当前结论",
        "",
        f"结论：**{summary['overall_status']}**。按当前对齐后的 ParkingAgent 数据集口径，数据链路{status_text}。",
        "",
        "本报告检查的是当前流水线实际使用的数据：`frames.csv` 中的每一帧必须能连接到图像、LiDAR、map 坐标点云、pose、aligned trajectory；每个 slot 必须能连接到 slot database、Part 1 score 和 final decision。",
        "",
        "## 2. 数据链路如何串起来",
        "",
        "```text",
        "frames.csv.frame",
        "  -> image_path",
        "  -> lidar_path",
        "  -> map_points_path / points_map_xyzi",
        "  -> azimuth_time_odometry_compatible.csv.frame",
        "  -> aligned_trajectory_final.csv.frame",
        "",
        "slot_database.slot_id",
        "  -> slot_scores.csv.slot_id",
        "  -> slot_decision_table.csv.slot_id",
        "```",
        "",
        "## 3. Frame 级检查结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| frame 行数 | {frame_checks['rows']} |",
        f"| frame 范围 | {frame_checks['first_frame']} - {frame_checks['last_frame']} |",
        f"| frame 是否连续 | {frame_checks['continuous_frame_range']} |",
        f"| 缺失 frame 编号数 | {frame_checks['missing_frame_number_count']} |",
        f"| 重复 frame 数 | {frame_checks['duplicate_frame_count']} |",
        f"| 异常 frame 行数 | {frame_checks['bad_rows']} |",
        f"| num_points 总和 | {frame_checks['num_points_sum']} |",
        "",
        "## 4. Metadata 一致性",
        "",
        "| 检查项 | 结果 |",
        "| --- | --- |",
        f"| selected_rows 匹配 frame 行数 | {metadata['selected_rows_matches_frames']} |",
        f"| processed_lidar_frames 匹配 frame 行数 | {metadata['processed_lidar_frames_matches_frames']} |",
        f"| missing_lidar 为 0 | {metadata['missing_lidar_is_zero']} |",
        f"| missing_image 为 0 | {metadata['missing_image_is_zero']} |",
        f"| total_projected_points 匹配 num_points 总和 | {metadata['total_projected_points_matches_sum']} |",
        "",
        "## 5. NPZ 检查结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 检查模式 | {npz_checks['mode']} |",
        f"| 检查 NPZ 文件数 | {npz_checks['checked_files']} |",
        f"| NPZ 错误数 | {npz_checks['error_count']} |",
        "",
        "NPZ 必须包含：`points_map_xyzi`、`ego_map_pose`、`frame`、`map_scale`。`points_map_xyzi` 必须是 `[N, 4]`，并且 N 要和 `frames.csv.num_points` 对齐。",
        "",
        "## 6. Slot 级检查结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| slot database 数量 | {slot_checks['slot_count']} |",
        f"| Part 1 score 行数 | {slot_checks['part1_score_rows']} |",
        f"| final decision 行数 | {slot_checks['decision_table_rows']} |",
        f"| 异常 slot 行数 | {slot_checks['bad_rows']} |",
        "",
        "Final decision 分布：",
        "",
        "```json",
        json.dumps(slot_checks["final_decision_counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 7. 原始数据边界说明",
        "",
        f"- 原始 image 文件数：{raw_context['raw_image_count']}",
        f"- 原始 LiDAR 文件数：{raw_context['raw_lidar_count']}",
        f"- 当前对齐数据集首帧：{frame_checks['first_frame']}",
        f"- 首帧之前的原始 image frame 示例：{raw_context['excluded_raw_image_frames_before_dataset_first']}",
        f"- 首帧之前的原始 LiDAR frame 示例：{raw_context['excluded_raw_lidar_frames_before_dataset_first']}",
        "",
        "这些首帧之前的原始文件不计为当前流水线缺失，因为 pose/alignment 表也是从当前对齐首帧开始。",
        "",
        "## 8. 问题示例",
        "",
    ]
    if summary["overall_status"] == "PASS":
        lines.extend(["未发现会打断当前数据链路的问题。", ""])
    else:
        lines.extend(
            [
                f"- 异常 frame 示例：{frame_checks['bad_rows_first20']}",
                f"- 异常 slot 示例：{slot_checks['bad_slots_first20']}",
                f"- NPZ 错误示例：{npz_checks['errors_first10']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(project_root: Path, output_dir: Path, npz_samples: int = 25, check_all_npz: bool = False) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_path = project_root / "outputs" / "frame_map_dataset" / "frames.csv"
    metadata_path = project_root / "outputs" / "frame_map_dataset" / "metadata.json"
    pose_path = project_root / "outputs" / "azimuth_time_odometry_compatible.csv"
    aligned_path = project_root / "outputs" / "aligned_trajectory_final.csv"

    frames_rows = read_csv(frames_path)
    metadata = load_json(metadata_path, {})
    pose_rows = read_csv(pose_path)
    aligned_rows = read_csv(aligned_path)
    pose_frames = frame_set(pose_rows)
    aligned_frames = frame_set(aligned_rows)

    frame_audit_rows, frame_summary = audit_frames(project_root, frames_rows, pose_frames, aligned_frames)
    npz_summary = validate_npz(project_root, frames_rows, npz_samples, check_all_npz)
    slot_audit_rows, slot_summary = audit_slots(project_root)
    metadata_summary = metadata_checks(metadata, frame_summary)
    raw_context = raw_dataset_context(project_root, metadata, frame_summary)

    overall_pass = all(
        item.get("status") == "PASS"
        for item in [frame_summary, npz_summary, slot_summary, metadata_summary]
    )
    summary: dict[str, Any] = {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "project_root": str(project_root),
        "outputs": {
            "integrity_summary_json": str(output_dir / "integrity_summary.json"),
            "frame_linkage_audit_csv": str(output_dir / "frame_linkage_audit.csv"),
            "slot_linkage_audit_csv": str(output_dir / "slot_linkage_audit.csv"),
            "dataset_integrity_linkage_report_md": str(output_dir / "dataset_integrity_linkage_report.md"),
        },
        "frame_checks": frame_summary,
        "metadata_checks": metadata_summary,
        "npz_sample_validation": npz_summary,
        "slot_checks": slot_summary,
        "raw_dataset_context": raw_context,
    }

    write_csv(
        output_dir / "frame_linkage_audit.csv",
        frame_audit_rows,
        [
            "frame",
            "image_exists",
            "lidar_exists",
            "map_points_exists",
            "pose_exists",
            "aligned_pose_exists",
            "csv_missing_image_flag",
            "csv_missing_lidar_flag",
            "num_points",
            "status",
            "issues",
        ],
    )
    write_csv(
        output_dir / "slot_linkage_audit.csv",
        slot_audit_rows,
        [
            "slot_id",
            "in_slot_database",
            "in_part1_scores",
            "in_decision_table",
            "part1_state",
            "final_decision",
            "status",
            "issues",
        ],
    )
    (output_dir / "integrity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(output_dir / "dataset_integrity_linkage_report.md", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = run_audit(args.project_root, args.output_dir, args.npz_samples, args.check_all_npz)
    print(f"overall_status: {summary['overall_status']}")
    print(f"frame rows: {summary['frame_checks']['rows']}")
    print(f"bad frame rows: {summary['frame_checks']['bad_rows']}")
    print(f"slot rows: {summary['slot_checks']['slot_count']}")
    print(f"bad slot rows: {summary['slot_checks']['bad_rows']}")
    print(f"report: {summary['outputs']['dataset_integrity_linkage_report_md']}")
    return 0 if summary["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
