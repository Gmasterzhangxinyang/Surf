"""Camera-visible evidence selection for parking-slot human review."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .models import Prediction, make_manifest_id, make_sample_id


@dataclass(frozen=True)
class EvidenceSettings:
    search_before: int = 400
    search_after: int = 400
    frame_stride: int = 5
    half_fov_deg: float = 40.0
    pose_prefilter_limit: int = 30
    max_frames: int = 7
    min_frames: int = 3
    max_camera_dt_sec: float = 0.05
    min_finite_vertices: int = 3
    min_projection_score: float = 0.5
    min_projected_area_px: float = 200.0
    ground_quantile: float = 0.08

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameAssessment:
    lidar_frame: int
    lidar_timestamp: float
    camera_frame: int
    camera_timestamp: float
    camera_lidar_dt_sec: float
    camera_image_path: str
    camera_image_exists: bool
    bearing_deg: float
    distance_m: float
    projection_score: float
    projected_area_px: float
    finite_vertices: int
    target_polygon_uv: tuple[tuple[float, float], ...]
    adjacent_polygons_uv: dict[str, tuple[tuple[float, float], ...]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def relative_bearing_deg(ego_pose: np.ndarray, slot_center: np.ndarray) -> float:
    delta = np.asarray(slot_center, dtype=np.float64) - np.asarray(ego_pose[:2], dtype=np.float64)
    yaw = float(ego_pose[2])
    forward = math.cos(yaw) * float(delta[0]) + math.sin(yaw) * float(delta[1])
    left = -math.sin(yaw) * float(delta[0]) + math.cos(yaw) * float(delta[1])
    return math.degrees(math.atan2(left, forward))


def _camera_match_valid(row: Mapping[str, str]) -> bool:
    raw = str(row.get("camera_match_valid", "1")).strip().lower()
    return raw in {"1", "true", "yes"}


def pose_prefilter(
    frame_rows: Mapping[int, Mapping[str, str]],
    slot_center: np.ndarray,
    *,
    anchor_frame: int,
    search_before: int,
    search_after: int,
    frame_stride: int,
    half_fov_deg: float,
    limit: int,
) -> list[int]:
    """Return closest pose-visible frames on both sides of an anchor."""
    stride = max(1, int(frame_stride))
    anchor = int(anchor_frame)
    lower = anchor - max(0, int(search_before))
    upper = anchor + max(0, int(search_after))
    center = np.asarray(slot_center, dtype=np.float64)
    candidates: list[tuple[float, float, int]] = []
    for frame_id, row in frame_rows.items():
        frame = int(frame_id)
        if frame < lower or frame > upper or (frame - anchor) % stride != 0:
            continue
        if not _camera_match_valid(row):
            continue
        try:
            pose = np.asarray(
                [float(row["map_x"]), float(row["map_y"]), float(row["map_yaw"])],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue
        bearing = relative_bearing_deg(pose, center)
        if abs(bearing) > float(half_fov_deg):
            continue
        distance = float(np.linalg.norm(center - pose[:2]))
        candidates.append((distance, abs(bearing), frame))
    candidates.sort()
    selected = [frame for _, _, frame in candidates[: max(0, int(limit))]]
    return sorted(selected)


def quality_reasons(
    row: FrameAssessment,
    *,
    half_fov_deg: float = 40.0,
    max_camera_dt_sec: float = 0.05,
    min_finite_vertices: int = 3,
    min_projection_score: float = 0.5,
    min_projected_area_px: float = 200.0,
) -> list[str]:
    reasons: list[str] = []
    if abs(float(row.bearing_deg)) > float(half_fov_deg):
        reasons.append("outside_safe_fov")
    if abs(float(row.camera_lidar_dt_sec)) > float(max_camera_dt_sec):
        reasons.append("camera_timestamp_delta_too_large")
    if int(row.finite_vertices) < int(min_finite_vertices):
        reasons.append("insufficient_projected_corners")
    if float(row.projection_score) < float(min_projection_score):
        reasons.append("projection_score_too_low")
    if float(row.projected_area_px) < float(min_projected_area_px):
        reasons.append("projected_area_too_small")
    if not bool(row.camera_image_exists):
        reasons.append("missing_camera_image")
    return reasons


def select_evidence_frames(
    assessments: Sequence[FrameAssessment],
    *,
    max_frames: int = 7,
    min_frames: int = 3,
    quality_options: Mapping[str, Any] | None = None,
) -> list[FrameAssessment]:
    valid = [row for row in assessments if not quality_reasons(row, **dict(quality_options or {}))]
    valid.sort(key=lambda row: row.lidar_frame)
    if len(valid) < max(1, int(min_frames)):
        return []
    limit = max(1, int(max_frames))
    center = max(
        valid,
        key=lambda row: (
            row.projection_score,
            row.projected_area_px,
            row.finite_vertices,
            -abs(row.bearing_deg),
            -abs(row.camera_lidar_dt_sec),
        ),
    )
    side_limit = max(0, (limit - 1) // 2)
    earlier = [row for row in valid if row.lidar_frame < center.lidar_frame]
    later = [row for row in valid if row.lidar_frame > center.lidar_frame]
    selected = earlier[-side_limit:] + [center] + later[:side_limit]
    if len(selected) < min(limit, len(valid)):
        chosen = {row.lidar_frame for row in selected}
        remaining = [row for row in valid if row.lidar_frame not in chosen]
        remaining.sort(
            key=lambda row: (
                row.projection_score,
                row.projected_area_px,
                -abs(row.bearing_deg),
            ),
            reverse=True,
        )
        selected.extend(remaining[: limit - len(selected)])
    return sorted(selected[:limit], key=lambda row: row.lidar_frame)


def _finite_polygon(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    return [
        (float(point[0]), float(point[1]))
        for point in points
        if len(point) >= 2 and np.isfinite(np.asarray(point[:2], dtype=np.float64)).all()
    ]


def draw_overlay(
    source_path: Path,
    output_path: Path,
    assessment: FrameAssessment,
    slot_id: str,
) -> None:
    image = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for adjacent_id, polygon in assessment.adjacent_polygons_uv.items():
        points = _finite_polygon(polygon)
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=(30, 144, 255), width=4)
            draw.text(points[0], adjacent_id, fill=(120, 200, 255))
    target = _finite_polygon(assessment.target_polygon_uv)
    if len(target) >= 3:
        draw.line(target + [target[0]], fill=(0, 255, 80), width=5)
        draw.text(target[0], slot_id, fill=(0, 255, 80))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else base_dir / path


def _finite_uv(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple(
        (float(point[0]), float(point[1]))
        for point in np.asarray(points, dtype=np.float64)
        if np.isfinite(point[:2]).all()
    )


def assess_frame(
    frame_id: int,
    row: Mapping[str, str],
    slot: Mapping[str, Any],
    slots: Mapping[str, Mapping[str, Any]],
    map_units_per_meter: float,
    settings: EvidenceSettings,
    base_dir: Path,
) -> FrameAssessment:
    """Project one target slot and its direct neighbors into a synchronized image."""
    from scripts import build_top5_camera_review as camera_projection

    points_path = _resolve_path(str(row.get("map_points_path", "")), base_dir)
    with np.load(points_path) as data:
        points = np.asarray(data["points_map_xyzi"], dtype=np.float64)
    finite_z = points[np.isfinite(points).all(axis=1), 2]
    ground_z = float(np.quantile(finite_z, settings.ground_quantile)) if len(finite_z) else 0.0
    pose = np.asarray(
        [float(row["map_x"]), float(row["map_y"]), float(row["map_yaw"])],
        dtype=np.float64,
    )
    camera_path = _resolve_path(
        str(row.get("camera_image_path") or row.get("image_path") or ""),
        base_dir,
    )
    image_width, image_height = (1280, 720)
    if camera_path.exists():
        with Image.open(camera_path) as source:
            image_width, image_height = source.size
    target_polygon = np.asarray(slot["polygon_map"], dtype=np.float64)
    target_uv, _ = camera_projection.polygon_to_uv(
        target_polygon,
        pose,
        float(map_units_per_meter),
        ground_z,
    )
    quality = camera_projection.projection_quality([target_uv], image_width, image_height)
    adjacent_uv: dict[str, tuple[tuple[float, float], ...]] = {}
    for adjacent_id in slot.get("adjacent_slots", []):
        adjacent = slots.get(str(adjacent_id))
        if adjacent is None:
            continue
        projected, _ = camera_projection.polygon_to_uv(
            np.asarray(adjacent["polygon_map"], dtype=np.float64),
            pose,
            float(map_units_per_meter),
            ground_z,
        )
        finite = _finite_uv(projected)
        if len(finite) >= 3:
            adjacent_uv[str(adjacent_id)] = finite
    center = np.asarray(slot["center_map"], dtype=np.float64)
    return FrameAssessment(
        lidar_frame=int(frame_id),
        lidar_timestamp=float(row["lidar_timestamp"]),
        camera_frame=int(float(row["camera_frame"])),
        camera_timestamp=float(row["camera_timestamp"]),
        camera_lidar_dt_sec=float(row["camera_lidar_dt_sec"]),
        camera_image_path=str(camera_path),
        camera_image_exists=camera_path.exists(),
        bearing_deg=relative_bearing_deg(pose, center),
        distance_m=float(np.linalg.norm(center - pose[:2]) / max(float(map_units_per_meter), 1e-12)),
        projection_score=float(quality["score"]),
        projected_area_px=float(quality["area"]),
        finite_vertices=int(quality["finite_points"]),
        target_polygon_uv=_finite_uv(target_uv),
        adjacent_polygons_uv=adjacent_uv,
    )


def draw_map_context(
    output_path: Path,
    slot: Mapping[str, Any],
    slots: Mapping[str, Mapping[str, Any]],
    *,
    width: int = 560,
    height: int = 420,
) -> None:
    polygons: list[tuple[str, np.ndarray]] = [
        ("target", np.asarray(slot["polygon_map"], dtype=np.float64))
    ]
    for adjacent_id in slot.get("adjacent_slots", []):
        adjacent = slots.get(str(adjacent_id))
        if adjacent is not None:
            polygons.append(("adjacent", np.asarray(adjacent["polygon_map"], dtype=np.float64)))
    all_points = np.vstack([polygon for _, polygon in polygons])
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    padding = 36.0

    def project(polygon: np.ndarray) -> list[tuple[float, float]]:
        normalized = (polygon - minimum) / span
        return [
            (
                padding + float(point[0]) * (width - 2 * padding),
                height - padding - float(point[1]) * (height - 2 * padding),
            )
            for point in normalized
        ]

    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    for kind, polygon in polygons:
        points = project(polygon)
        color = (0, 180, 70) if kind == "target" else (30, 120, 230)
        draw.line(points + [points[0]], fill=color, width=5 if kind == "target" else 3)
    draw.text((16, 12), str(slot.get("slot_id", "target slot")), fill=(15, 23, 42))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _labels_exist(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return bool(payload.get("labels"))


def _quality_options(settings: EvidenceSettings) -> dict[str, Any]:
    return {
        "half_fov_deg": settings.half_fov_deg,
        "max_camera_dt_sec": settings.max_camera_dt_sec,
        "min_finite_vertices": settings.min_finite_vertices,
        "min_projection_score": settings.min_projection_score,
        "min_projected_area_px": settings.min_projected_area_px,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_manifest(
    predictions: Sequence[Prediction],
    frames_csv: Path,
    slot_database_path: Path,
    output_dir: Path,
    settings: EvidenceSettings | None = None,
    *,
    assessor: Callable[..., FrameAssessment] | None = None,
    rebuild: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    settings = settings or EvidenceSettings()
    output_dir = Path(output_dir)
    manifest_path = output_dir / "evidence_manifest.json"
    labels_path = output_dir / "human_labels.json"
    if rebuild and _labels_exist(labels_path):
        raise ValueError("cannot rebuild evidence because labels already exist")
    dataset_ids = {prediction.dataset_id for prediction in predictions}
    if len(dataset_ids) > 1:
        raise ValueError("all predictions must belong to one dataset")
    if manifest_path.exists() and not rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if dataset_ids and manifest.get("dataset_id") not in dataset_ids:
            raise ValueError("existing evidence dataset does not match predictions")
        expected = make_manifest_id(str(manifest.get("dataset_id", "")), settings.to_dict())
        if manifest.get("manifest_id") != expected:
            raise ValueError("existing evidence settings do not match requested settings")
        return manifest
    if not predictions:
        raise ValueError("at least one prediction is required to build evidence")
    dataset_id = next(iter(dataset_ids))
    manifest_id = make_manifest_id(dataset_id, settings.to_dict())
    frame_rows = {int(row["frame"]): row for row in _read_csv(Path(frames_csv))}
    slot_database = json.loads(Path(slot_database_path).read_text(encoding="utf-8"))
    slots = {str(slot["slot_id"]): slot for slot in slot_database["slots"]}
    scale = float(slot_database.get("map_units_per_meter", 1.0))
    assessor_fn = assessor or assess_frame
    assets_dir = output_dir / "evidence" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    total = len(predictions)
    quality_options = _quality_options(settings)
    base_dir = Path.cwd()
    for index, prediction in enumerate(predictions, start=1):
        slot = slots.get(prediction.slot_id)
        assessments: list[FrameAssessment] = []
        errors: list[str] = []
        if slot is not None:
            candidates = pose_prefilter(
                frame_rows,
                np.asarray(slot["center_map"], dtype=np.float64),
                anchor_frame=prediction.anchor_frame,
                search_before=settings.search_before,
                search_after=settings.search_after,
                frame_stride=settings.frame_stride,
                half_fov_deg=settings.half_fov_deg,
                limit=settings.pose_prefilter_limit,
            )
            for frame_id in candidates:
                try:
                    assessments.append(
                        assessor_fn(
                            frame_id,
                            frame_rows[frame_id],
                            slot,
                            slots,
                            scale,
                            settings,
                            base_dir,
                        )
                    )
                except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
                    errors.append(f"frame {frame_id}: {exc}")
        selected = select_evidence_frames(
            assessments,
            max_frames=settings.max_frames,
            min_frames=settings.min_frames,
            quality_options=quality_options,
        )
        anchor_row = frame_rows.get(prediction.anchor_frame, {})
        fallback_timestamp = float(
            anchor_row.get("camera_timestamp")
            or anchor_row.get("lidar_timestamp")
            or prediction.anchor_frame
        )
        encounter_timestamp = (
            max(selected, key=lambda row: (row.projection_score, row.projected_area_px)).camera_timestamp
            if selected
            else fallback_timestamp
        )
        sample_id = make_sample_id(dataset_id, prediction.slot_id, encounter_timestamp)
        evidence_frames: list[dict[str, Any]] = []
        for position, row in enumerate(selected, start=1):
            relative = Path("evidence") / "assets" / (
                f"{index:03d}_{prediction.slot_id}_{position:02d}_"
                f"lidar_{row.lidar_frame:06d}_camera_{row.camera_frame:06d}.png"
            )
            draw_overlay(Path(row.camera_image_path), output_dir / relative, row, prediction.slot_id)
            frame_payload = row.to_dict()
            frame_payload["asset_path"] = relative.as_posix()
            evidence_frames.append(frame_payload)
        map_asset = ""
        if slot is not None:
            relative_map = Path("evidence") / "assets" / f"{index:03d}_{prediction.slot_id}_map.png"
            draw_map_context(output_dir / relative_map, slot, slots)
            map_asset = relative_map.as_posix()
        reason_counts: Counter[str] = Counter()
        for row in assessments:
            reason_counts.update(quality_reasons(row, **quality_options))
        if slot is None:
            automatic_reasons = ["missing_slot_geometry"]
        elif not selected and reason_counts:
            automatic_reasons = [name for name, _ in reason_counts.most_common()]
        elif not selected and errors:
            automatic_reasons = ["assessment_error"]
        elif not selected:
            automatic_reasons = ["outside_fov_or_no_candidate"]
        else:
            automatic_reasons = []
        cases.append(
            {
                "sample_id": sample_id,
                "dataset_id": dataset_id,
                "slot_id": prediction.slot_id,
                "source_anchor_frame": prediction.anchor_frame,
                "encounter_timestamp": encounter_timestamp,
                "review_status": "reviewable" if selected else "unobservable",
                "automatic_reasons": automatic_reasons,
                "assessment_errors": errors,
                "map_asset": map_asset,
                "evidence_frames": evidence_frames,
            }
        )
        if progress is not None:
            progress(index, total, prediction.slot_id)
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "manifest_id": manifest_id,
        "settings": settings.to_dict(),
        "summary": {
            "case_count": len(cases),
            "reviewable_count": sum(case["review_status"] == "reviewable" for case in cases),
            "unobservable_count": sum(case["review_status"] == "unobservable" for case in cases),
        },
        "cases": cases,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest
