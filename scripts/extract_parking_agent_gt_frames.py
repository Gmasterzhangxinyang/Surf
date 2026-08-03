#!/usr/bin/env python3
"""Extract the exact frame-9277 evidence window into a blind GT annotation pack."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge
from PIL import Image, ImageDraw, ImageFont


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_timestamps(path: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.strip().split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number} must have frame and timestamp")
            frame = int(fields[0])
            timestamp = float(fields[1])
            if frame in result:
                raise ValueError(f"duplicate timestamp frame {frame}")
            result[frame] = timestamp
    return result


def _load_lidar_timestamps(path: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[int(row["lidar_file_index"])] = float(row["lidar_timestamp"])
    return result


def _nearest(timestamps: dict[int, float], target: float) -> tuple[int, float]:
    ordered = sorted((timestamp, frame) for frame, timestamp in timestamps.items())
    values = [item[0] for item in ordered]
    index = bisect_left(values, target)
    candidates = ordered[max(0, index - 1): min(len(ordered), index + 1)]
    timestamp, frame = min(candidates, key=lambda item: abs(item[0] - target))
    return frame, timestamp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _contact_sheets(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[str] = []
    columns, rows_per_sheet = 5, 3
    thumb_size = (384, 216)
    cell_size = (400, 260)
    per_sheet = columns * rows_per_sheet
    for sheet_index, start in enumerate(range(0, len(rows), per_sheet), start=1):
        batch = rows[start:start + per_sheet]
        canvas = Image.new(
            "RGB",
            (columns * cell_size[0], rows_per_sheet * cell_size[1] + 60),
            "#eef3f8",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (20, 15),
            f"Frame 9277 GT review | causal window {batch[0]['lidar_frame']:06d}–{batch[-1]['lidar_frame']:06d}",
            font=_font(24, bold=True),
            fill="#132238",
        )
        for index, row in enumerate(batch):
            image = Image.open(row["camera_copy"]).convert("RGB")
            image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            x = (index % columns) * cell_size[0] + 8
            y = (index // columns) * cell_size[1] + 65
            canvas.paste(image, (x, y))
            draw.text(
                (x, y + 220),
                f"L {row['lidar_frame']:06d} → C {row['camera_frame']:06d}  dt={row['camera_lidar_dt_sec']*1000:+.1f}ms",
                font=_font(16),
                fill="#25364d",
            )
        path = output_dir / f"camera_contact_sheet_{sheet_index:02d}.jpg"
        canvas.save(path, quality=91)
        result.append(str(path))
    return result


def _neutral_map(scene: dict[str, Any], candidate_ids: set[str], path: Path) -> None:
    ego_x, ego_y, ego_yaw = (float(value) for value in scene["anchor_pose_map"])
    scale = float(scene["map_units_per_meter"])
    if scale <= 0:
        raise ValueError("scene.map_units_per_meter must be positive")
    def local_m(point: list[float]) -> tuple[float, float]:
        return (
            (float(point[0]) - ego_x) / scale,
            (float(point[1]) - ego_y) / scale,
        )
    figure, axis = plt.subplots(figsize=(12, 12), dpi=150)
    for slot in scene["slots"]:
        polygon = [local_m(point) for point in slot["polygon_map"]]
        candidate = slot["slot_id"] in candidate_ids
        axis.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor="#dbeafe" if candidate else "#f1f5f9",
                edgecolor="#246bdb" if candidate else "#94a3b8",
                linewidth=1.8 if candidate else 0.7,
            )
        )
        if candidate:
            x, y = local_m(slot["center_map"])
            axis.text(x, y, slot["slot_id"].replace("slot_", ""), fontsize=7,
                      ha="center", va="center", color="#132238")
    radius = float(scene["radius_m"])
    axis.add_patch(
        Wedge(
            (0.0, 0.0),
            radius,
            math.degrees(ego_yaw) - 40.0,
            math.degrees(ego_yaw) + 40.0,
            facecolor="#22c55e18",
            edgecolor="#16a34a",
            linewidth=1.3,
            linestyle="--",
        )
    )
    axis.arrow(
        0.0, 0.0, 3.0 * math.cos(ego_yaw), 3.0 * math.sin(ego_yaw),
        width=0.12, head_width=0.7, color="#ef4444", length_includes_head=True,
    )
    axis.scatter([0.0], [0.0], marker="x", s=90, color="#132238", zorder=5)
    axis.set_xlim(-radius, radius)
    axis.set_ylim(-radius, radius)
    axis.set_aspect("equal")
    axis.grid(color="#dbe3ed", linewidth=0.5)
    axis.set_title(
        "Blind GT map — blue: 24 candidate slots; green: ±40° routing FOV; red: ego heading",
        fontsize=13,
    )
    axis.set_xlabel("anchor-centered map x (m)")
    axis.set_ylabel("anchor-centered map y (m)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--lidar-pose-csv", type=Path, required=True)
    parser.add_argument("--geometry-regression", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")

    part1 = _load_json(args.part1)
    scene = part1["scene"]
    anchor = int(scene["anchor_frame_id"])
    candidate_cases = part1["slot_cases"]
    candidate_ids = [str(case["slot"]["slot_id"]) for case in candidate_cases]
    decisions = [
        _load_json(Path(case["resources"]["extended_lidar_decision_path"]))
        for case in candidate_cases
    ]
    selected_sets = {
        tuple(int(value) for value in decision["selected_frames"])
        for decision in decisions
    }
    if len(selected_sets) != 1:
        raise ValueError("candidate cases do not share one extended frame window")
    extended_frames = next(iter(selected_sets))
    if extended_frames[-1] != anchor or any(value > anchor for value in extended_frames):
        raise ValueError("extended frame window violates the t0 causal contract")
    part1_frames = tuple(int(frame["frame_id"]) for frame in scene["frames"])

    camera_timestamps = _load_timestamps(args.dataset_root / "image" / "timestamps.txt")
    lidar_timestamps = _load_lidar_timestamps(args.lidar_pose_csv)
    root = args.output_dir
    root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    copied_files: list[dict[str, Any]] = []
    for lidar_frame in extended_frames:
        lidar_timestamp = lidar_timestamps[lidar_frame]
        camera_frame, camera_timestamp = _nearest(camera_timestamps, lidar_timestamp)
        camera_source = args.dataset_root / "image" / f"left{camera_frame:06d}.png"
        lidar_source = args.dataset_root / "velodyne" / f"{lidar_frame:06d}.bin"
        camera_destination = (
            root / "03_camera_timestamp_matched_60"
            / f"lidar_{lidar_frame:06d}__left{camera_frame:06d}.png"
        )
        lidar_destination = root / "04_lidar_raw_60" / f"{lidar_frame:06d}.bin"
        copied_files.append(_copy(camera_source, camera_destination))
        copied_files.append(_copy(lidar_source, lidar_destination))
        rows.append(
            {
                "lidar_frame": lidar_frame,
                "lidar_timestamp": lidar_timestamp,
                "camera_frame": camera_frame,
                "camera_timestamp": camera_timestamp,
                "camera_lidar_dt_sec": camera_timestamp - lidar_timestamp,
                "is_part1_15frame": lidar_frame in set(part1_frames),
                "is_anchor_t0": lidar_frame == anchor,
                "camera_copy": str(camera_destination),
                "lidar_copy": str(lidar_destination),
            }
        )

    anchor_row = next(row for row in rows if row["is_anchor_t0"])
    copied_files.append(
        _copy(
            Path(anchor_row["camera_copy"]),
            root / "01_anchor_t0" / f"left{anchor_row['camera_frame']:06d}.png",
        )
    )
    copied_files.append(
        _copy(
            Path(anchor_row["lidar_copy"]),
            root / "01_anchor_t0" / f"{anchor:06d}.bin",
        )
    )
    for row in rows:
        if not row["is_part1_15frame"]:
            continue
        copied_files.append(
            _copy(
                Path(row["camera_copy"]),
                root / "02_part1_camera_15"
                / f"lidar_{row['lidar_frame']:06d}__left{row['camera_frame']:06d}.png",
            )
        )
    contact_sheets = _contact_sheets(rows, root / "05_contact_sheets")
    neutral_map = root / "06_annotation" / "candidate_slots_neutral_map.png"
    _neutral_map(scene, set(candidate_ids), neutral_map)

    frame_csv_rows = [
        {
            **{key: value for key, value in row.items() if key not in {"camera_copy", "lidar_copy"}},
            "camera_file": str(Path(row["camera_copy"]).relative_to(root)),
            "lidar_file": str(Path(row["lidar_copy"]).relative_to(root)),
        }
        for row in rows
    ]
    _write_csv(
        root / "frame_manifest.csv",
        [
            "lidar_frame", "lidar_timestamp", "camera_frame", "camera_timestamp",
            "camera_lidar_dt_sec", "is_part1_15frame", "is_anchor_t0",
            "camera_file", "lidar_file",
        ],
        frame_csv_rows,
    )
    blank_rows = [
        {
            "slot_id": slot_id,
            "anchor_lidar_frame": anchor,
            "anchor_camera_frame": anchor_row["camera_frame"],
            "gt_state": "",
            "gt_camera_observability": "",
            "gt_confidence": "",
            "annotator": "",
            "notes": "",
        }
        for slot_id in candidate_ids
    ]
    _write_csv(
        root / "06_annotation" / "gt_candidate_slots_24.csv",
        [
            "slot_id", "anchor_lidar_frame", "anchor_camera_frame", "gt_state",
            "gt_camera_observability", "gt_confidence", "annotator", "notes",
        ],
        blank_rows,
    )
    slot_geometry = {
        "anchor_lidar_frame": anchor,
        "anchor_camera_frame": anchor_row["camera_frame"],
        "coordinate_frame": scene["coordinate_frame"],
        "slots": [
            {
                "slot_id": slot["slot_id"],
                "polygon_map": slot["polygon_map"],
                "core_polygon_map": slot.get("core_polygon_map"),
                "center_map": slot["center_map"],
            }
            for slot in scene["slots"]
            if slot["slot_id"] in set(candidate_ids)
        ],
    }
    (root / "06_annotation" / "candidate_slot_geometry.json").write_text(
        json.dumps(slot_geometry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reference_rows: list[dict[str, Any]] = []
    geometry_by_slot: dict[str, Any] = {}
    if args.geometry_regression:
        geometry = _load_json(args.geometry_regression)
        geometry_by_slot = {row["slot_id"]: row for row in geometry["cases"]}
    for case in candidate_cases:
        slot_id = case["slot"]["slot_id"]
        reference_rows.append(
            {
                "slot_id": slot_id,
                "part1_state": case["part1_state"],
                "geometry_gate_state": geometry_by_slot.get(slot_id, {}).get("state", ""),
            }
        )
    _write_csv(
        root / "07_predictions_open_only_after_gt" / "model_predictions_reference.csv",
        ["slot_id", "part1_state", "geometry_gate_state"],
        reference_rows,
    )

    readme = f"""# Frame 9277 人工GT标注帧包

## 项目实际使用的帧

- 状态判定时刻（t0）：LiDAR `{anchor:06d}`，时间戳 `{scene['anchor_timestamp']}`。
- t0最近邻Camera：`left{anchor_row['camera_frame']:06d}.png`，
  Camera-LiDAR差 `{anchor_row['camera_lidar_dt_sec'] * 1000:+.3f} ms`。
- Part1窗口：LiDAR `{part1_frames[0]:06d}–{part1_frames[-1]:06d}`，共 `{len(part1_frames)}` 帧。
- 扩展LiDAR工具窗口：LiDAR `{extended_frames[0]:06d}–{extended_frames[-1]:06d}`，
  共 `{len(extended_frames)}` 帧，全部不晚于t0。
- 当前修复后的Agent实跑中，Camera Context/Crop使用的是t0原图
  `left{anchor_row['camera_frame']:06d}.png`；扩展60帧主要用于LiDAR几何。

## 建议标注顺序

1. 先打开 `06_annotation/candidate_slots_neutral_map.png` 确认24个目标车位位置。
2. 运行 `scripts/build_gt_neutral_lidar_views.py` 后，查看生成的
   `05_contact_sheets/lidar_neutral/` 和
   `06_annotation/lidar_neutral_60/`。这些图不包含模型预测。
3. 依次查看Camera `05_contact_sheets/camera_contact_sheet_*.jpg`，必要时打开
   `03_camera_timestamp_matched_60/` 中的原分辨率图片。
4. 在 `06_annotation/gt_candidate_slots_24.csv` 填写：
   - `gt_state`: `free` / `occupied` / `unknown`
   - `gt_camera_observability`: `visible` / `partial` / `not_visible`
   - `gt_confidence`: 建议0–1
5. 完成并锁定GT后，才打开
   `07_predictions_open_only_after_gt/model_predictions_reference.csv`，避免模型结果影响标注。

`unknown` 应用于无法从现有Camera/点云可靠确认的车位，不要为了完整率强行二分类。

## GT口径

- `occupied`：车辆或其他实体障碍明确进入目标车位核心区域。
- `free`：目标车位核心区域被充分观察，且没有实体障碍。
- `unknown`：目标核心未被充分观察、只看到边界/墙柱，或Camera与LiDAR证据冲突。
- 车位线、墙边和相邻车位回波本身不能标成`occupied`。
"""
    (root / "README_人工标注说明.md").write_text(readme, encoding="utf-8")
    manifest = {
        "schema_version": "parking-agent-gt-frame-pack/1.0",
        "source_part1": str(args.part1.resolve()),
        "snapshot_id": scene["snapshot_id"],
        "anchor_lidar_frame": anchor,
        "anchor_camera_frame": anchor_row["camera_frame"],
        "part1_lidar_frames": list(part1_frames),
        "extended_lidar_frames": list(extended_frames),
        "frame_count": len(rows),
        "candidate_slot_count": len(candidate_ids),
        "candidate_slot_ids": candidate_ids,
        "timestamp_matching": "nearest_camera_timestamp",
        "max_abs_camera_lidar_dt_sec": max(abs(row["camera_lidar_dt_sec"]) for row in rows),
        "contact_sheets": [str(Path(path).relative_to(root)) for path in contact_sheets],
        "copied_file_count": len(copied_files),
        "copied_bytes": sum(item["bytes"] for item in copied_files),
        "files": copied_files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "files"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
