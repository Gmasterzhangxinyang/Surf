"""Deterministic, path-free multi-frame LiDAR evidence packs for Part2.

The pack is an NPZ container, but it is written with fixed ZIP metadata and a
fixed member order.  Rebuilding from identical semantic inputs therefore
produces identical bytes.  Source paths are used only to verify and hash the
selected frame artifacts; paths are never stored in the pack.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np

from .contracts import MetricSlot, SlotAccumulation


LIDAR_EVIDENCE_PACK_SCHEMA_VERSION = "part2-lidar-evidence-pack/1.0"
LIDAR_EVIDENCE_PACK_BUILDER_VERSION = "part2-lidar-evidence-builder/1.0"
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

PACK_ARRAY_KEYS = (
    "metadata_json",
    "points_local_xyzi",
    "point_frame_ids",
    "selected_frames",
    "valid_frames",
    "observation_origins_local_xyz",
    "polygon_local_m",
    "core_polygon_local_m",
    "margin_polygon_local_m",
    "adjacent_polygon_points_local_m",
    "adjacent_polygon_offsets",
)


@dataclass(frozen=True, slots=True)
class LidarEvidencePackArtifact:
    """Identity and basic counts for one written evidence pack."""

    path: Path
    sha256: str
    size_bytes: int
    slot_id: str
    selected_frames: tuple[int, ...]
    valid_frames: tuple[int, ...]
    point_count: int


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_identity(value: str, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical sha256 identity")
    return value


def _float32_array(value: Any, *, columns: int, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.shape[1] != columns:
        raise ValueError(f"{name} must have shape [N, {columns}]")
    if source.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must be numeric")
    result = np.ascontiguousarray(source, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite or float32-overflow values")
    return result


def _canonicalize_points(
    points: np.ndarray,
    point_frame_ids: np.ndarray,
    valid_frames: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int]]]:
    point_array = _float32_array(points, columns=4, name="points_local_xyzi")
    raw_ids = np.asarray(point_frame_ids)
    if raw_ids.ndim != 1 or len(raw_ids) != len(point_array):
        raise ValueError("point_frame_ids must have one value per point")
    if raw_ids.dtype.kind not in {"i", "u"}:
        raise ValueError("point_frame_ids must be integer-valued")
    ids64 = np.asarray(raw_ids, dtype=np.int64)
    if np.any(ids64 < 0) or np.any(ids64 > np.iinfo(np.int32).max):
        raise ValueError("point_frame_ids exceed the supported int32 range")
    valid_set = set(valid_frames)
    unknown = sorted(set(int(value) for value in ids64) - valid_set)
    if unknown:
        raise ValueError(f"point_frame_ids contain non-valid frame(s): {unknown}")

    if len(point_array):
        # Frame first, then XYZI.  This makes point ordering independent of the
        # provider's incidental order while preserving every point exactly at
        # the pack's declared float32 precision.
        order = np.lexsort(
            (
                point_array[:, 3],
                point_array[:, 2],
                point_array[:, 1],
                point_array[:, 0],
                ids64,
            )
        )
        point_array = np.ascontiguousarray(point_array[order], dtype=np.float32)
        ids64 = np.ascontiguousarray(ids64[order], dtype=np.int64)
    ids = np.ascontiguousarray(ids64, dtype=np.int32)

    spans: dict[int, tuple[int, int]] = {}
    offset = 0
    for frame_id in valid_frames:
        count = int(np.count_nonzero(ids == frame_id))
        spans[frame_id] = (offset, count)
        offset += count
    if offset != len(ids):
        raise ValueError("point frame spans do not cover the packed points")
    return point_array, ids, spans


def _canonicalize_adjacency(
    adjacent_polygons_local_m: Mapping[str, np.ndarray] | None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    values = dict(adjacent_polygons_local_m or {})
    slot_ids = tuple(sorted(values))
    batches: list[np.ndarray] = []
    offsets = [0]
    for slot_id in slot_ids:
        if not isinstance(slot_id, str) or not slot_id:
            raise ValueError("adjacent slot IDs must be non-empty strings")
        polygon = _float32_array(
            values[slot_id], columns=2, name=f"adjacent polygon {slot_id}"
        )
        if len(polygon) < 3:
            raise ValueError(f"adjacent polygon {slot_id} must have at least three vertices")
        batches.append(polygon)
        offsets.append(offsets[-1] + len(polygon))
    points = (
        np.ascontiguousarray(np.vstack(batches), dtype=np.float32)
        if batches
        else np.empty((0, 2), dtype=np.float32)
    )
    return slot_ids, points, np.asarray(offsets, dtype=np.int32)


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(
        output,
        np.ascontiguousarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return output.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if tuple(arrays) != PACK_ARRAY_KEYS:
        raise ValueError("evidence-pack arrays are not in the canonical member order")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
        with zipfile.ZipFile(
            temporary_name,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for name in PACK_ARRAY_KEYS:
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(arrays[name]), compresslevel=9)
        with Path(temporary_name).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def build_lidar_evidence_pack(
    output_path: str | Path,
    *,
    accumulation: SlotAccumulation,
    slot: MetricSlot,
    source_paths: Mapping[int, str | Path],
    task_id: str,
    encounter_id: str,
    dataset_id: str,
    config_hash: str,
    slot_map_hash: str,
    adjacent_polygons_local_m: Mapping[str, np.ndarray] | None = None,
) -> LidarEvidencePackArtifact:
    """Build one deterministic multi-frame, slot-local Part2 evidence pack.

    Every selected source frame must resolve to a concrete file.  A partially
    identified input set is rejected rather than silently producing evidence
    with unverifiable provenance.
    """

    if accumulation.slot_id != slot.slot_id:
        raise ValueError("accumulation and metric slot IDs do not match")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty string")
    if not isinstance(encounter_id, str) or not encounter_id.strip():
        raise ValueError("encounter_id must be a non-empty string")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")
    _require_identity(config_hash, "config_hash")
    _require_identity(slot_map_hash, "slot_map_hash")

    selected_frames = tuple(int(value) for value in accumulation.selected_frames)
    if not selected_frames or selected_frames != tuple(sorted(set(selected_frames))):
        raise ValueError("selected_frames must be non-empty, unique, and sorted")
    if any(value < 0 or value > np.iinfo(np.int32).max for value in selected_frames):
        raise ValueError("selected_frames exceed the supported int32 range")
    anchor_frame = int(accumulation.anchor_frame)
    if anchor_frame not in selected_frames:
        raise ValueError("anchor_frame must be present in selected_frames")

    observation_by_frame: dict[int, Any] = {}
    for observation in accumulation.observations:
        frame_id = int(observation.frame_id)
        if frame_id in observation_by_frame:
            raise ValueError(f"duplicate observation for frame {frame_id}")
        observation_by_frame[frame_id] = observation
    valid_frames = tuple(sorted(observation_by_frame))
    if not set(valid_frames).issubset(selected_frames):
        raise ValueError("valid observation frames must be selected frames")
    if not valid_frames:
        raise ValueError("an evidence pack requires at least one valid observation frame")

    excluded: dict[int, tuple[str, ...]] = {}
    for frame_id, reasons in accumulation.excluded_frames:
        normalized_id = int(frame_id)
        if normalized_id in excluded:
            raise ValueError(f"duplicate exclusion for frame {normalized_id}")
        normalized_reasons = tuple(sorted(set(str(reason) for reason in reasons)))
        if not normalized_reasons or any(not reason for reason in normalized_reasons):
            raise ValueError("excluded frames must have non-empty reason codes")
        excluded[normalized_id] = normalized_reasons
    if not set(excluded).issubset(selected_frames):
        raise ValueError("excluded frames must be selected frames")
    if set(excluded) & set(valid_frames):
        raise ValueError("a frame cannot be both valid and excluded")
    accounted = set(valid_frames) | set(excluded)
    if accounted != set(selected_frames):
        missing = sorted(set(selected_frames) - accounted)
        raise ValueError(f"selected frame(s) lack valid/excluded status: {missing}")

    points, point_frame_ids, spans = _canonicalize_points(
        accumulation.points_local_xyzi,
        accumulation.point_frame_ids,
        valid_frames,
    )
    origins: list[np.ndarray] = []
    for frame_id in valid_frames:
        origin = np.asarray(observation_by_frame[frame_id].origin_local_xyz)
        if origin.shape != (3,) or origin.dtype.kind not in {"f", "i", "u"}:
            raise ValueError(f"observation origin for frame {frame_id} must have shape [3]")
        converted = np.asarray(origin, dtype=np.float32)
        if not np.isfinite(converted).all():
            raise ValueError(f"observation origin for frame {frame_id} is non-finite")
        origins.append(converted)
    origins_array = (
        np.ascontiguousarray(np.vstack(origins), dtype=np.float32)
        if origins
        else np.empty((0, 3), dtype=np.float32)
    )

    polygon = _float32_array(slot.polygon_local_m, columns=2, name="polygon_local_m")
    core = _float32_array(slot.core_polygon_local_m, columns=2, name="core_polygon_local_m")
    margin = _float32_array(slot.margin_polygon_local_m, columns=2, name="margin_polygon_local_m")
    for name, value in (("polygon", polygon), ("core polygon", core), ("margin polygon", margin)):
        if len(value) < 3:
            raise ValueError(f"{name} must have at least three vertices")
    adjacent_ids, adjacent_points, adjacent_offsets = _canonicalize_adjacency(
        adjacent_polygons_local_m
    )

    source_rows: list[dict[str, Any]] = []
    normalized_sources = {int(frame_id): Path(path) for frame_id, path in source_paths.items()}
    for frame_id in selected_frames:
        source = normalized_sources.get(frame_id)
        if source is None or not source.is_file():
            raise ValueError(f"selected frame {frame_id} has no concrete source artifact")
        try:
            source_size = source.stat().st_size
            source_sha256 = _file_sha256(source)
        except OSError as exc:
            raise ValueError(f"selected frame {frame_id} source artifact is unreadable") from exc
        offset, count = spans.get(frame_id, (0, 0))
        source_rows.append(
            {
                "exclusion_reasons": list(excluded.get(frame_id, ())),
                "frame_id": frame_id,
                "point_count": count,
                "point_offset": offset,
                "source_sha256": source_sha256,
                "source_size_bytes": source_size,
                "status": "valid" if frame_id in observation_by_frame else "excluded",
            }
        )

    metadata = {
        "adjacent_slot_ids": list(adjacent_ids),
        "anchor_frame": anchor_frame,
        "array_keys": list(PACK_ARRAY_KEYS),
        "builder_version": LIDAR_EVIDENCE_PACK_BUILDER_VERSION,
        "config_hash": config_hash,
        "dataset_id": dataset_id,
        "encounter_id": encounter_id,
        "frames": source_rows,
        "schema_version": LIDAR_EVIDENCE_PACK_SCHEMA_VERSION,
        "selected_frames": list(selected_frames),
        "slot_id": slot.slot_id,
        "slot_map_hash": slot_map_hash,
        "task_id": task_id,
        "valid_frames": list(valid_frames),
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    arrays = {
        "metadata_json": np.frombuffer(metadata_bytes, dtype=np.uint8).copy(),
        "points_local_xyzi": points,
        "point_frame_ids": point_frame_ids,
        "selected_frames": np.asarray(selected_frames, dtype=np.int32),
        "valid_frames": np.asarray(valid_frames, dtype=np.int32),
        "observation_origins_local_xyz": origins_array,
        "polygon_local_m": polygon,
        "core_polygon_local_m": core,
        "margin_polygon_local_m": margin,
        "adjacent_polygon_points_local_m": adjacent_points,
        "adjacent_polygon_offsets": adjacent_offsets,
    }

    destination = Path(output_path)
    _write_deterministic_npz(destination, arrays)
    try:
        size_bytes = destination.stat().st_size
        sha256 = _file_sha256(destination)
    except OSError as exc:
        raise RuntimeError("written LiDAR evidence pack is not readable") from exc
    if not math.isfinite(float(size_bytes)):
        raise RuntimeError("written LiDAR evidence pack has invalid size")
    return LidarEvidencePackArtifact(
        path=destination,
        sha256=sha256,
        size_bytes=size_bytes,
        slot_id=slot.slot_id,
        selected_frames=selected_frames,
        valid_frames=valid_frames,
        point_count=len(points),
    )


__all__ = [
    "LIDAR_EVIDENCE_PACK_BUILDER_VERSION",
    "LIDAR_EVIDENCE_PACK_SCHEMA_VERSION",
    "PACK_ARRAY_KEYS",
    "LidarEvidencePackArtifact",
    "build_lidar_evidence_pack",
]
