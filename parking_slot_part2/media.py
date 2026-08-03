"""Strict evidence loading and path-private media rendering.

This module is an internal provider bridge.  Public tool results continue to
contain only evidence metadata; a provider that is handed the store can resolve
newly rendered LiDAR or annotated RGB media after the corresponding opaque tool
call succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from parking_slot_hybrid_3d.part2_evidence import (
    LIDAR_EVIDENCE_PACK_BUILDER_VERSION,
    LIDAR_EVIDENCE_PACK_SCHEMA_VERSION,
    PACK_ARRAY_KEYS,
)

from .contracts import freeze_json


LIDAR_MEDIA_RENDERER_VERSION = "part2-lidar-triptych/1.0"
LIDAR_MEDIA_WIDTH = 1536
LIDAR_MEDIA_HEIGHT = 512
RGB_FRAME_RENDERER_VERSION = "part2-rgb-target-overlay/2.0"
RGB_SEQUENCE_RENDERER_VERSION = "part2-rgb-contact-sheet/2.0"
RGB_SEQUENCE_MAX_FRAMES = 5
RGB_SEQUENCE_COLUMNS = 2
RGB_SEQUENCE_TILE_WIDTH = 640
RGB_SEQUENCE_TILE_HEIGHT = 400
RGB_SEQUENCE_TILE_GAP = 8
RGB_SEQUENCE_HEADER_HEIGHT = 28
MAX_RGB_FILE_BYTES = 128 * 1024 * 1024
MAX_RGB_WIDTH = 8192
MAX_RGB_HEIGHT = 8192
MAX_RGB_PIXELS = 50_000_000
MAX_PACK_FILE_BYTES = 512 * 1024 * 1024
MAX_PACK_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_PACK_POINTS = 5_000_000
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
EVIDENCE_ID_PATTERN = re.compile(r"^ev_[0-9a-f]{64}$")

_METADATA_FIELDS = frozenset(
    {
        "adjacent_slot_ids",
        "anchor_frame",
        "array_keys",
        "builder_version",
        "config_hash",
        "dataset_id",
        "encounter_id",
        "frames",
        "schema_version",
        "selected_frames",
        "slot_id",
        "slot_map_hash",
        "task_id",
        "valid_frames",
    }
)
_FRAME_FIELDS = frozenset(
    {
        "exclusion_reasons",
        "frame_id",
        "point_count",
        "point_offset",
        "source_sha256",
        "source_size_bytes",
        "status",
    }
)


class EvidenceMediaError(ValueError):
    """An evidence artifact cannot be safely exposed through the media bridge."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class EvidencePackError(EvidenceMediaError):
    """A LiDAR evidence pack failed strict structural or identity validation."""

    def __init__(self, message: str) -> None:
        super().__init__("evidence_media_unavailable", message)


@dataclass(frozen=True, slots=True)
class LidarEvidencePack:
    """Validated in-memory evidence with immutable array views."""

    source_sha256: str
    metadata: Mapping[str, Any]
    points_local_xyzi: np.ndarray
    point_frame_ids: np.ndarray
    selected_frames: np.ndarray
    valid_frames: np.ndarray
    observation_origins_local_xyz: np.ndarray
    polygon_local_m: np.ndarray
    core_polygon_local_m: np.ndarray
    margin_polygon_local_m: np.ndarray
    adjacent_polygon_points_local_m: np.ndarray
    adjacent_polygon_offsets: np.ndarray


@dataclass(frozen=True, slots=True)
class RgbFrameMediaInput:
    """Private, identity-bound input for one annotated RGB rendering."""

    source_path: Path
    source_sha256: str
    resource_id: str
    task_id: str
    slot_id: str
    encounter_id: str
    visual_frame_id: str
    lidar_frame: int
    camera_frame: int
    camera_timestamp: float
    polygon_uv: Any
    adjacent_polygons_uv: Any
    expected_width: int | None = None
    expected_height: int | None = None


@dataclass(frozen=True, slots=True)
class MediaArtifact:
    """Internal content-addressed media binding for one evidence ID."""

    evidence_id: str
    kind: str
    path: Path
    sha256: str
    size_bytes: int
    width: int
    height: int
    source_sha256: str
    renderer_version: str

    def manifest_record(self) -> dict[str, Any]:
        """Return a path-free record suitable for a generation manifest."""

        return {
            "evidence_id": self.evidence_id,
            "height": self.height,
            "kind": self.kind,
            "renderer_version": self.renderer_version,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_sha256": self.source_sha256,
            "width": self.width,
        }


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _bytes_sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_exact_dtype(array: np.ndarray, dtype: np.dtype[Any], name: str) -> np.ndarray:
    expected = np.dtype(dtype)
    if array.dtype != expected:
        raise EvidencePackError(f"{name} must use exact dtype {expected}")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _require_matrix(
    array: np.ndarray,
    *,
    columns: int,
    dtype: np.dtype[Any],
    name: str,
) -> np.ndarray:
    result = _require_exact_dtype(array, dtype, name)
    if result.ndim != 2 or result.shape[1] != columns:
        raise EvidencePackError(f"{name} must have shape [N, {columns}]")
    if result.dtype.kind == "f" and not np.isfinite(result).all():
        raise EvidencePackError(f"{name} contains non-finite values")
    return result


def _require_vector(
    array: np.ndarray,
    *,
    dtype: np.dtype[Any],
    name: str,
) -> np.ndarray:
    result = _require_exact_dtype(array, dtype, name)
    if result.ndim != 1:
        raise EvidencePackError(f"{name} must be one-dimensional")
    return result


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidencePackError(f"{name} must be an integer >= {minimum}")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidencePackError(f"{name} must be a non-empty string")
    return value


def _require_hash(value: Any, name: str) -> str:
    result = _require_string(value, name)
    if SHA256_PATTERN.fullmatch(result) is None:
        raise EvidencePackError(f"{name} must be a canonical sha256 identity")
    return result


def _strict_archive_members(path: Path) -> None:
    expected = tuple(f"{name}.npy" for name in PACK_ARRAY_KEYS)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            names = tuple(item.filename for item in members)
            if names != expected:
                raise EvidencePackError("evidence pack has unsupported or reordered members")
            if any(item.flag_bits & 0x1 for item in members):
                raise EvidencePackError("encrypted evidence-pack members are not supported")
            if sum(item.file_size for item in members) > MAX_PACK_UNCOMPRESSED_BYTES:
                raise EvidencePackError("evidence pack exceeds the uncompressed size limit")
    except EvidencePackError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise EvidencePackError("evidence pack is not a readable NPZ archive") from exc


def is_lidar_evidence_pack(path: str | Path) -> bool:
    """Return whether an NPZ explicitly advertises the v1 pack metadata member.

    Legacy ``lidar_map`` and anchor-frame ``pointcloud_artifact`` resources may
    still occur in old queues, but they are not successful visual evidence.
    The bridge must not reinterpret them as v1 packs merely because their
    filename ends in ``.npz``.
    """

    candidate = Path(path)
    if candidate.suffix.lower() != ".npz" or not candidate.is_file():
        return False
    try:
        with zipfile.ZipFile(candidate, "r") as archive:
            return "metadata_json.npy" in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as payload:
            if tuple(payload.files) != PACK_ARRAY_KEYS:
                raise EvidencePackError("evidence pack has unsupported array keys")
            for name in PACK_ARRAY_KEYS:
                arrays[name] = np.array(payload[name], copy=True)
    except EvidencePackError:
        raise
    except Exception as exc:
        raise EvidencePackError("evidence pack contains an unreadable array") from exc
    return arrays


def load_lidar_evidence_pack(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_slot_id: str | None = None,
    expected_task_id: str | None = None,
    expected_encounter_id: str | None = None,
    expected_dataset_id: str | None = None,
    expected_config_hash: str | None = None,
    expected_slot_map_hash: str | None = None,
) -> LidarEvidencePack:
    """Load a pack using strict schema, dtype, provenance, and span checks."""

    source = Path(path)
    try:
        if not source.is_file():
            raise EvidencePackError("evidence pack is not a concrete file")
        if source.stat().st_size > MAX_PACK_FILE_BYTES:
            raise EvidencePackError("evidence pack exceeds the file size limit")
        actual_sha256 = _file_sha256(source)
    except EvidencePackError:
        raise
    except OSError as exc:
        raise EvidencePackError("evidence pack cannot be read") from exc
    if expected_sha256 is not None:
        _require_hash(expected_sha256, "expected_sha256")
        if actual_sha256 != expected_sha256:
            raise EvidencePackError("evidence pack sha256 does not match its resource")

    _strict_archive_members(source)
    arrays = _load_arrays(source)
    metadata_bytes = _require_vector(
        arrays["metadata_json"], dtype=np.uint8, name="metadata_json"
    ).tobytes()
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidencePackError("metadata_json is not canonical UTF-8 JSON") from exc
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
        raise EvidencePackError("metadata_json fields do not match the evidence-pack schema")
    if _canonical_json_bytes(metadata) != metadata_bytes:
        raise EvidencePackError("metadata_json is not canonically encoded")
    if metadata["schema_version"] != LIDAR_EVIDENCE_PACK_SCHEMA_VERSION:
        raise EvidencePackError("unsupported evidence-pack schema_version")
    if metadata["builder_version"] != LIDAR_EVIDENCE_PACK_BUILDER_VERSION:
        raise EvidencePackError("unsupported evidence-pack builder_version")
    if metadata["array_keys"] != list(PACK_ARRAY_KEYS):
        raise EvidencePackError("metadata array_keys do not match the archive")
    slot_id = _require_string(metadata["slot_id"], "slot_id")
    if expected_slot_id is not None and slot_id != expected_slot_id:
        raise EvidencePackError("evidence pack slot_id does not match its task")
    task_id = _require_string(metadata["task_id"], "task_id")
    encounter_id = _require_string(metadata["encounter_id"], "encounter_id")
    if expected_task_id is not None and task_id != expected_task_id:
        raise EvidencePackError("evidence pack task_id does not match its task")
    if expected_encounter_id is not None and encounter_id != expected_encounter_id:
        raise EvidencePackError("evidence pack encounter_id does not match its task")
    dataset_id = _require_string(metadata["dataset_id"], "dataset_id")
    config_hash = _require_hash(metadata["config_hash"], "config_hash")
    slot_map_hash = _require_hash(metadata["slot_map_hash"], "slot_map_hash")
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise EvidencePackError("evidence pack dataset_id does not match its resource")
    if expected_config_hash is not None:
        _require_hash(expected_config_hash, "expected_config_hash")
        if config_hash != expected_config_hash:
            raise EvidencePackError("evidence pack config_hash does not match its resource")
    if expected_slot_map_hash is not None:
        _require_hash(expected_slot_map_hash, "expected_slot_map_hash")
        if slot_map_hash != expected_slot_map_hash:
            raise EvidencePackError("evidence pack slot_map_hash does not match its resource")

    points = _require_matrix(
        arrays["points_local_xyzi"],
        columns=4,
        dtype=np.float32,
        name="points_local_xyzi",
    )
    if len(points) > MAX_PACK_POINTS:
        raise EvidencePackError("evidence pack exceeds the point count limit")
    point_frame_ids = _require_vector(
        arrays["point_frame_ids"], dtype=np.int32, name="point_frame_ids"
    )
    if len(point_frame_ids) != len(points):
        raise EvidencePackError("point_frame_ids length does not match the point array")
    selected = _require_vector(
        arrays["selected_frames"], dtype=np.int32, name="selected_frames"
    )
    valid = _require_vector(arrays["valid_frames"], dtype=np.int32, name="valid_frames")
    origins = _require_matrix(
        arrays["observation_origins_local_xyz"],
        columns=3,
        dtype=np.float32,
        name="observation_origins_local_xyz",
    )
    polygon = _require_matrix(
        arrays["polygon_local_m"], columns=2, dtype=np.float32, name="polygon_local_m"
    )
    core = _require_matrix(
        arrays["core_polygon_local_m"],
        columns=2,
        dtype=np.float32,
        name="core_polygon_local_m",
    )
    margin = _require_matrix(
        arrays["margin_polygon_local_m"],
        columns=2,
        dtype=np.float32,
        name="margin_polygon_local_m",
    )
    adjacent_points = _require_matrix(
        arrays["adjacent_polygon_points_local_m"],
        columns=2,
        dtype=np.float32,
        name="adjacent_polygon_points_local_m",
    )
    adjacent_offsets = _require_vector(
        arrays["adjacent_polygon_offsets"],
        dtype=np.int32,
        name="adjacent_polygon_offsets",
    )
    if any(len(value) < 3 for value in (polygon, core, margin)):
        raise EvidencePackError("target slot polygons must each have at least three vertices")

    selected_tuple = tuple(int(value) for value in selected)
    valid_tuple = tuple(int(value) for value in valid)
    if not selected_tuple or selected_tuple != tuple(sorted(set(selected_tuple))):
        raise EvidencePackError("selected_frames must be non-empty, unique, and sorted")
    if valid_tuple != tuple(sorted(set(valid_tuple))):
        raise EvidencePackError("valid_frames must be unique and sorted")
    if not valid_tuple:
        raise EvidencePackError("an evidence pack requires at least one valid frame")
    if not set(valid_tuple).issubset(selected_tuple):
        raise EvidencePackError("valid_frames must be a subset of selected_frames")
    if metadata["selected_frames"] != list(selected_tuple):
        raise EvidencePackError("metadata selected_frames do not match the array")
    if metadata["valid_frames"] != list(valid_tuple):
        raise EvidencePackError("metadata valid_frames do not match the array")
    if len(origins) != len(valid_tuple):
        raise EvidencePackError("observation origins do not match valid_frames")
    anchor_frame = _require_int(metadata["anchor_frame"], "anchor_frame")
    if anchor_frame not in selected_tuple:
        raise EvidencePackError("anchor_frame is not selected")

    frame_rows = metadata["frames"]
    if not isinstance(frame_rows, list) or len(frame_rows) != len(selected_tuple):
        raise EvidencePackError("metadata frames do not match selected_frames")
    expected_offset = 0
    seen_frames: list[int] = []
    for row in frame_rows:
        if not isinstance(row, dict) or set(row) != _FRAME_FIELDS:
            raise EvidencePackError("metadata frame fields do not match the schema")
        frame_id = _require_int(row["frame_id"], "frames.frame_id")
        seen_frames.append(frame_id)
        _require_hash(row["source_sha256"], "frames.source_sha256")
        _require_int(row["source_size_bytes"], "frames.source_size_bytes")
        offset = _require_int(row["point_offset"], "frames.point_offset")
        count = _require_int(row["point_count"], "frames.point_count")
        status = row["status"]
        reasons = row["exclusion_reasons"]
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason for reason in reasons
        ):
            raise EvidencePackError("frames.exclusion_reasons must be string reason codes")
        if reasons != sorted(set(reasons)):
            raise EvidencePackError("frames.exclusion_reasons must be unique and sorted")
        if status == "valid":
            if frame_id not in valid_tuple or reasons:
                raise EvidencePackError("valid frame metadata is inconsistent")
            if offset != expected_offset:
                raise EvidencePackError("valid frame point spans are not contiguous")
            if not np.all(point_frame_ids[offset : offset + count] == frame_id):
                raise EvidencePackError("valid frame span does not match point_frame_ids")
            expected_offset += count
        elif status == "excluded":
            if frame_id in valid_tuple or not reasons or offset != 0 or count != 0:
                raise EvidencePackError("excluded frame metadata is inconsistent")
        else:
            raise EvidencePackError("frames.status must be valid or excluded")
    if tuple(seen_frames) != selected_tuple or expected_offset != len(points):
        raise EvidencePackError("metadata frame order or point coverage is inconsistent")
    if len(point_frame_ids) and not set(int(value) for value in point_frame_ids).issubset(valid_tuple):
        raise EvidencePackError("point_frame_ids contain a non-valid frame")

    # Builder output is canonical by frame and then XYZI.  Reject a semantically
    # equivalent but reordered payload so its byte identity remains meaningful.
    if len(points):
        order = np.lexsort(
            (points[:, 3], points[:, 2], points[:, 1], points[:, 0], point_frame_ids)
        )
        if not np.array_equal(order, np.arange(len(points))):
            raise EvidencePackError("points are not in canonical frame/XYZI order")

    adjacent_ids = metadata["adjacent_slot_ids"]
    if (
        not isinstance(adjacent_ids, list)
        or any(not isinstance(value, str) or not value for value in adjacent_ids)
        or adjacent_ids != sorted(set(adjacent_ids))
    ):
        raise EvidencePackError("adjacent_slot_ids must be unique and sorted")
    if len(adjacent_offsets) != len(adjacent_ids) + 1:
        raise EvidencePackError("adjacent polygon offsets do not match adjacent_slot_ids")
    if (
        len(adjacent_offsets) == 0
        or adjacent_offsets[0] != 0
        or adjacent_offsets[-1] != len(adjacent_points)
        or np.any(np.diff(adjacent_offsets) < 3)
    ):
        raise EvidencePackError("adjacent polygon offsets are invalid")

    return LidarEvidencePack(
        source_sha256=actual_sha256,
        metadata=freeze_json(metadata),
        points_local_xyzi=points,
        point_frame_ids=point_frame_ids,
        selected_frames=selected,
        valid_frames=valid,
        observation_origins_local_xyz=origins,
        polygon_local_m=polygon,
        core_polygon_local_m=core,
        margin_polygon_local_m=margin,
        adjacent_polygon_points_local_m=adjacent_points,
        adjacent_polygon_offsets=adjacent_offsets,
    )


def _height_color(z: float) -> tuple[int, int, int]:
    if z <= 0.20:
        return (148, 163, 184)
    if z <= 0.60:
        return (245, 158, 11)
    if z <= 1.40:
        return (234, 88, 12)
    return (220, 38, 38)


def _xy_bounds(pack: LidarEvidencePack) -> tuple[float, float, float, float]:
    batches = [pack.margin_polygon_local_m]
    if len(pack.adjacent_polygon_points_local_m):
        batches.append(pack.adjacent_polygon_points_local_m)
    geometry = np.vstack(batches)
    x0, y0 = geometry.min(axis=0)
    x1, y1 = geometry.max(axis=0)
    pad = 0.35
    if x1 - x0 < 0.5:
        x0, x1 = x0 - 0.25, x1 + 0.25
    if y1 - y0 < 0.5:
        y0, y1 = y0 - 0.25, y1 + 0.25
    return float(x0 - pad), float(x1 + pad), float(y0 - pad), float(y1 + pad)


def _panel_transform(
    bounds: tuple[float, float, float, float],
    panel_index: int,
    *,
    margin: int = 34,
):
    x0, x1, y0, y1 = bounds
    left = panel_index * 512 + margin
    right = (panel_index + 1) * 512 - margin
    top = margin
    bottom = LIDAR_MEDIA_HEIGHT - margin

    def convert(x: float, y: float) -> tuple[int, int]:
        px = left + (float(x) - x0) / max(x1 - x0, 1e-9) * (right - left)
        py = bottom - (float(y) - y0) / max(y1 - y0, 1e-9) * (bottom - top)
        return int(round(px)), int(round(py))

    return convert


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    transform: Any,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    coordinates = [transform(float(point[0]), float(point[1])) for point in polygon]
    if coordinates:
        draw.line(coordinates + [coordinates[0]], fill=color, width=width)


def _sample_indices(count: int, limit: int = 45_000) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=limit, dtype=np.int64)


def render_lidar_triptych(pack: LidarEvidencePack) -> bytes:
    """Render deterministic BEV, longitudinal, and max-height panels."""

    image = Image.new("RGB", (LIDAR_MEDIA_WIDTH, LIDAR_MEDIA_HEIGHT), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.line([(512, 0), (512, 511)], fill=(203, 213, 225), width=2)
    draw.line([(1024, 0), (1024, 511)], fill=(203, 213, 225), width=2)

    bounds = _xy_bounds(pack)
    bev = _panel_transform(bounds, 0)
    sampled = _sample_indices(len(pack.points_local_xyzi))
    for index in sampled:
        x, y, z = pack.points_local_xyzi[index, :3]
        px, py = bev(float(x), float(y))
        if 0 <= px < 512 and 0 <= py < 512:
            draw.point((px, py), fill=_height_color(float(z)))
    for offset in range(len(pack.adjacent_polygon_offsets) - 1):
        start = int(pack.adjacent_polygon_offsets[offset])
        end = int(pack.adjacent_polygon_offsets[offset + 1])
        _draw_polygon(draw, bev, pack.adjacent_polygon_points_local_m[start:end], (100, 116, 139), width=1)
    _draw_polygon(draw, bev, pack.margin_polygon_local_m, (245, 158, 11), width=2)
    _draw_polygon(draw, bev, pack.polygon_local_m, (37, 99, 235), width=2)
    _draw_polygon(draw, bev, pack.core_polygon_local_m, (22, 163, 74), width=2)
    draw.text((12, 10), "BEV / height-coded", fill=(15, 23, 42))

    x0, x1, _, _ = bounds
    side = _panel_transform((x0, x1, -0.30, 3.00), 1)
    for index in sampled:
        x, _, z = pack.points_local_xyzi[index, :3]
        px, py = side(float(x), float(z))
        if 512 <= px < 1024 and 0 <= py < 512:
            draw.point((px, py), fill=_height_color(float(z)))
    ground_start = side(x0, 0.0)
    ground_end = side(x1, 0.0)
    draw.line([ground_start, ground_end], fill=(100, 116, 139), width=1)
    for polygon, color in (
        (pack.margin_polygon_local_m, (245, 158, 11)),
        (pack.polygon_local_m, (37, 99, 235)),
        (pack.core_polygon_local_m, (22, 163, 74)),
    ):
        lower, upper = float(polygon[:, 0].min()), float(polygon[:, 0].max())
        draw.line([side(lower, -0.05), side(lower, 2.8)], fill=color, width=1)
        draw.line([side(upper, -0.05), side(upper, 2.8)], fill=color, width=1)
    draw.text((524, 10), "Longitudinal X / normalized Z", fill=(15, 23, 42))

    grid_transform = _panel_transform(bounds, 2)
    grid_size = 64
    grid = np.full((grid_size, grid_size), -np.inf, dtype=np.float32)
    px0, px1, py0, py1 = bounds
    if len(pack.points_local_xyzi):
        points = pack.points_local_xyzi
        gx = np.floor((points[:, 0] - px0) / max(px1 - px0, 1e-9) * grid_size).astype(np.int64)
        gy = np.floor((points[:, 1] - py0) / max(py1 - py0, 1e-9) * grid_size).astype(np.int64)
        inside = (gx >= 0) & (gx < grid_size) & (gy >= 0) & (gy < grid_size)
        flat = gy[inside] * grid_size + gx[inside]
        np.maximum.at(grid.ravel(), flat, points[inside, 2])
    for gy in range(grid_size):
        for gx in range(grid_size):
            z = float(grid[gy, gx])
            if not np.isfinite(z):
                continue
            cell_x0 = px0 + gx / grid_size * (px1 - px0)
            cell_x1 = px0 + (gx + 1) / grid_size * (px1 - px0)
            cell_y0 = py0 + gy / grid_size * (py1 - py0)
            cell_y1 = py0 + (gy + 1) / grid_size * (py1 - py0)
            left, bottom = grid_transform(cell_x0, cell_y0)
            right, top = grid_transform(cell_x1, cell_y1)
            draw.rectangle((left, top, right, bottom), fill=_height_color(z))
    for offset in range(len(pack.adjacent_polygon_offsets) - 1):
        start = int(pack.adjacent_polygon_offsets[offset])
        end = int(pack.adjacent_polygon_offsets[offset + 1])
        _draw_polygon(
            draw,
            grid_transform,
            pack.adjacent_polygon_points_local_m[start:end],
            (100, 116, 139),
            width=1,
        )
    _draw_polygon(draw, grid_transform, pack.margin_polygon_local_m, (245, 158, 11), width=2)
    _draw_polygon(draw, grid_transform, pack.polygon_local_m, (37, 99, 235), width=2)
    _draw_polygon(draw, grid_transform, pack.core_polygon_local_m, (22, 163, 74), width=2)
    draw.text((1036, 10), "Ground-normalized max height", fill=(15, 23, 42))

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _require_evidence_id(evidence_id: str) -> None:
    if not isinstance(evidence_id, str) or EVIDENCE_ID_PATTERN.fullmatch(evidence_id) is None:
        raise EvidenceMediaError(
            "invalid_evidence_identity",
            "media binding requires an opaque evidence_id",
        )


def _require_rgb_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceMediaError(
            "rgb_identity_invalid",
            f"{name} must be a non-empty string",
        )
    return value


def _require_rgb_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceMediaError(
            "rgb_identity_invalid",
            f"{name} must be a non-negative integer",
        )
    return value


def _require_rgb_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceMediaError("rgb_identity_invalid", f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise EvidenceMediaError("rgb_identity_invalid", f"{name} must be finite")
    return result


def _polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    return abs(
        sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(
                polygon,
                (*polygon[1:], polygon[0]),
            )
        )
        / 2.0
    )


def _normalize_polygon(
    value: Any,
    *,
    name: str,
    width: int,
    height: int,
    require_in_bounds: bool,
    missing_code: str,
    invalid_code: str,
    out_of_bounds_code: str,
) -> tuple[tuple[float, float], ...]:
    if value is None or (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 0
    ):
        raise EvidenceMediaError(missing_code, f"{name} is required")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        raise EvidenceMediaError(invalid_code, f"{name} must contain at least three points")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 2
        ):
            raise EvidenceMediaError(
                invalid_code,
                f"{name}[{index}] must be a two-dimensional point",
            )
        coordinates: list[float] = []
        for coordinate in point:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise EvidenceMediaError(invalid_code, f"{name} contains a non-finite point")
            numeric = float(coordinate)
            if not np.isfinite(numeric):
                raise EvidenceMediaError(invalid_code, f"{name} contains a non-finite point")
            coordinates.append(numeric)
        x, y = coordinates
        if require_in_bounds and not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
            raise EvidenceMediaError(
                out_of_bounds_code,
                f"{name} contains an out-of-bounds point",
            )
        points.append((x, y))
    result = tuple(points)
    if _polygon_area(result) <= 1e-6:
        raise EvidenceMediaError(invalid_code, f"{name} is degenerate")
    return result


def _clip_polygon_to_image(
    polygon: Sequence[tuple[float, float]],
    width: int,
    height: int,
) -> tuple[tuple[float, float], ...]:
    """Clip a finite polygon to the decoded image rectangle."""

    points = list(polygon)

    def clip_edge(
        candidates: list[tuple[float, float]],
        inside: Any,
        intersection: Any,
    ) -> list[tuple[float, float]]:
        if not candidates:
            return []
        output: list[tuple[float, float]] = []
        previous = candidates[-1]
        previous_inside = inside(previous)
        for current in candidates:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside
        return output

    x_max = float(width - 1)
    y_max = float(height - 1)

    def vertical(boundary: float, first: tuple[float, float], second: tuple[float, float]):
        x0, y0 = first
        x1, y1 = second
        ratio = 0.0 if x1 == x0 else (boundary - x0) / (x1 - x0)
        return boundary, y0 + ratio * (y1 - y0)

    def horizontal(boundary: float, first: tuple[float, float], second: tuple[float, float]):
        x0, y0 = first
        x1, y1 = second
        ratio = 0.0 if y1 == y0 else (boundary - y0) / (y1 - y0)
        return x0 + ratio * (x1 - x0), boundary

    points = clip_edge(points, lambda point: point[0] >= 0.0, lambda a, b: vertical(0.0, a, b))
    points = clip_edge(points, lambda point: point[0] <= x_max, lambda a, b: vertical(x_max, a, b))
    points = clip_edge(points, lambda point: point[1] >= 0.0, lambda a, b: horizontal(0.0, a, b))
    points = clip_edge(points, lambda point: point[1] <= y_max, lambda a, b: horizontal(y_max, a, b))
    return tuple(points)


def _clipped_polygon_or_none(
    polygon: Sequence[tuple[float, float]],
    width: int,
    height: int,
) -> tuple[tuple[float, float], ...] | None:
    """Return a drawable clipped polygon, or ``None`` when no area is visible."""

    clipped = _clip_polygon_to_image(polygon, width, height)
    if len(clipped) < 3 or _polygon_area(clipped) <= 1e-6:
        return None
    return clipped


def _validate_rgb_identity(frame: RgbFrameMediaInput) -> None:
    for name in ("resource_id", "task_id", "slot_id", "encounter_id", "visual_frame_id"):
        _require_rgb_text(getattr(frame, name), name)
    lidar_frame = _require_rgb_integer(frame.lidar_frame, "lidar_frame")
    camera_frame = _require_rgb_integer(frame.camera_frame, "camera_frame")
    if lidar_frame == camera_frame:
        raise EvidenceMediaError(
            "rgb_identity_invalid",
            "LiDAR and camera frame identities must remain distinct",
        )
    _require_rgb_number(frame.camera_timestamp, "camera_timestamp")
    if not isinstance(frame.source_sha256, str) or SHA256_PATTERN.fullmatch(frame.source_sha256) is None:
        raise EvidenceMediaError(
            "rgb_identity_invalid",
            "RGB source_sha256 must be a canonical identity",
        )


def _load_and_annotate_rgb(
    frame: RgbFrameMediaInput,
) -> tuple[Image.Image, str]:
    _validate_rgb_identity(frame)
    try:
        source = Path(frame.source_path)
    except (TypeError, ValueError) as exc:
        raise EvidenceMediaError(
            "rgb_identity_invalid",
            "RGB source path is invalid",
        ) from exc
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(source, flags), "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise EvidenceMediaError(
                    "rgb_image_invalid",
                    "RGB source is not a concrete file",
                )
            source_content = handle.read(MAX_RGB_FILE_BYTES + 1)
        if len(source_content) > MAX_RGB_FILE_BYTES:
            raise EvidenceMediaError(
                "rgb_image_invalid",
                "RGB source exceeds the file size limit",
            )
        actual_sha256 = _bytes_sha256(source_content)
    except EvidenceMediaError:
        raise
    except OSError as exc:
        raise EvidenceMediaError("rgb_image_invalid", "RGB source cannot be read") from exc
    if actual_sha256 != frame.source_sha256:
        raise EvidenceMediaError(
            "rgb_source_identity_mismatch",
            "RGB source sha256 does not match its resource",
        )

    try:
        # Decode the exact byte buffer that was hashed above.  Re-opening the
        # path here would permit a concurrent rename/symlink swap to change
        # the pixels sent to the local model after identity verification.
        with Image.open(io.BytesIO(source_content)) as opened:
            width, height = opened.size
            if (
                isinstance(width, bool)
                or isinstance(height, bool)
                or width < 2
                or height < 2
                or width > MAX_RGB_WIDTH
                or height > MAX_RGB_HEIGHT
                or width * height > MAX_RGB_PIXELS
            ):
                raise EvidenceMediaError(
                    "rgb_image_dimensions_invalid",
                    "RGB source dimensions are outside safe limits",
                )
            if (frame.expected_width is None) != (frame.expected_height is None):
                raise EvidenceMediaError(
                    "rgb_image_dimensions_invalid",
                    "expected RGB dimensions must be declared together",
                )
            if frame.expected_width is not None and frame.expected_height is not None:
                if (
                    isinstance(frame.expected_width, bool)
                    or isinstance(frame.expected_height, bool)
                    or not isinstance(frame.expected_width, int)
                    or not isinstance(frame.expected_height, int)
                    or frame.expected_width < 2
                    or frame.expected_height < 2
                ):
                    raise EvidenceMediaError(
                        "rgb_image_dimensions_invalid",
                        "expected RGB dimensions are invalid",
                    )
                if (width, height) != (frame.expected_width, frame.expected_height):
                    raise EvidenceMediaError(
                        "rgb_image_dimensions_mismatch",
                        "decoded RGB dimensions do not match the projection identity",
                    )
            if getattr(opened, "n_frames", 1) != 1:
                raise EvidenceMediaError(
                    "rgb_image_invalid",
                    "animated RGB evidence is not supported",
                )
            raw_target = _normalize_polygon(
                frame.polygon_uv,
                name="polygon_uv",
                width=width,
                height=height,
                require_in_bounds=False,
                missing_code="rgb_target_polygon_missing",
                invalid_code="rgb_target_polygon_invalid",
                out_of_bounds_code="rgb_target_polygon_out_of_bounds",
            )
            target = _clipped_polygon_or_none(raw_target, width, height)
            if target is None:
                raise EvidenceMediaError(
                    "rgb_target_polygon_out_of_bounds",
                    "polygon_uv has no non-degenerate area inside the image",
                )
            target_was_clipped = target != raw_target
            adjacent_payload = frame.adjacent_polygons_uv
            if adjacent_payload is None:
                adjacent_payload = {}
            if not isinstance(adjacent_payload, Mapping):
                raise EvidenceMediaError(
                    "rgb_adjacent_polygon_invalid",
                    "adjacent_polygons_uv must be an object",
                )
            adjacent_slot_ids = tuple(adjacent_payload)
            if any(
                not isinstance(adjacent_slot_id, str) or not adjacent_slot_id
                for adjacent_slot_id in adjacent_slot_ids
            ):
                raise EvidenceMediaError(
                    "rgb_adjacent_polygon_invalid",
                    "adjacent polygon identities must be non-empty strings",
                )
            adjacent: list[tuple[str, tuple[tuple[float, float], ...]]] = []
            omitted_adjacent_count = 0
            for adjacent_slot_id in sorted(adjacent_slot_ids):
                if adjacent_slot_id == frame.slot_id:
                    raise EvidenceMediaError(
                        "rgb_identity_invalid",
                        "target slot cannot also be an adjacent polygon",
                    )
                try:
                    raw_polygon = _normalize_polygon(
                        adjacent_payload[adjacent_slot_id],
                        name=f"adjacent_polygons_uv.{adjacent_slot_id}",
                        width=width,
                        height=height,
                        require_in_bounds=False,
                        missing_code="rgb_adjacent_polygon_invalid",
                        invalid_code="rgb_adjacent_polygon_invalid",
                        out_of_bounds_code="rgb_adjacent_polygon_invalid",
                    )
                except EvidenceMediaError as exc:
                    if exc.code != "rgb_adjacent_polygon_invalid":
                        raise
                    omitted_adjacent_count += 1
                    continue
                polygon = _clipped_polygon_or_none(raw_polygon, width, height)
                if polygon is None:
                    omitted_adjacent_count += 1
                    continue
                adjacent.append((adjacent_slot_id, polygon))
            opened.load()
            image = opened.convert("RGB")
    except EvidenceMediaError:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise EvidenceMediaError("rgb_image_invalid", "RGB source cannot be decoded") from exc

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    secondary_width = max(2, min(width, height) // 360)
    target_width = max(4, min(width, height) // 160)
    font = ImageFont.load_default()
    for adjacent_slot_id, polygon in adjacent:
        draw.line(
            [*polygon, polygon[0]],
            fill=(250, 204, 21, 210),
            width=secondary_width,
            joint="curve",
        )
        center_x = sum(point[0] for point in polygon) / len(polygon)
        center_y = sum(point[1] for point in polygon) / len(polygon)
        draw.text(
            (center_x + 2, center_y + 2),
            adjacent_slot_id,
            fill=(254, 240, 138, 235),
            font=font,
            stroke_width=1,
            stroke_fill=(15, 23, 42, 220),
        )
    draw.polygon(target, fill=(236, 0, 204, 56))
    draw.line(
        [*target, target[0]],
        fill=(255, 0, 221, 255),
        width=target_width,
        joint="curve",
    )
    marker_radius = max(4, target_width + 1)
    for x, y in target:
        draw.ellipse(
            (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius),
            fill=(255, 255, 255, 255),
            outline=(255, 0, 221, 255),
            width=max(2, target_width // 2),
        )
    banner_height = min(height, max(28, min(width, height) // 16))
    draw.rectangle((0, 0, width, banner_height), fill=(15, 23, 42, 220))
    label = (
        f"TARGET {frame.slot_id} | visual {frame.visual_frame_id} | "
        f"lidar {frame.lidar_frame} / camera {frame.camera_frame}"
    )
    if target_was_clipped:
        label += " | TARGET CLIPPED"
    if omitted_adjacent_count:
        label += f" | ADJ OMITTED {omitted_adjacent_count}"
    draw.text(
        (8, max(2, (banner_height - 11) // 2)),
        label,
        fill=(255, 255, 255, 255),
        font=font,
    )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"), actual_sha256


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def render_rgb_frame(frame: RgbFrameMediaInput) -> tuple[bytes, int, int, str]:
    """Validate, annotate, and re-encode one path-private RGB source."""

    image, source_sha256 = _load_and_annotate_rgb(frame)
    return _png_bytes(image), image.width, image.height, source_sha256


def render_rgb_sequence(
    frames: Sequence[RgbFrameMediaInput],
) -> tuple[bytes, int, int, str]:
    """Render one deterministic, ordered contact sheet of at most five frames."""

    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
        raise EvidenceMediaError(
            "rgb_sequence_invalid",
            "RGB sequence inputs must be an ordered sequence",
        )
    inputs = tuple(frames)
    if not 1 <= len(inputs) <= RGB_SEQUENCE_MAX_FRAMES:
        raise EvidenceMediaError(
            "rgb_sequence_invalid",
            f"RGB sequence must contain 1..{RGB_SEQUENCE_MAX_FRAMES} frames",
        )
    if any(not isinstance(frame, RgbFrameMediaInput) for frame in inputs):
        raise EvidenceMediaError(
            "rgb_sequence_invalid",
            "RGB sequence contains an unsupported frame input",
        )
    for frame in inputs:
        _validate_rgb_identity(frame)
    identity = (inputs[0].task_id, inputs[0].slot_id, inputs[0].encounter_id)
    if any((frame.task_id, frame.slot_id, frame.encounter_id) != identity for frame in inputs):
        raise EvidenceMediaError(
            "rgb_sequence_identity_mismatch",
            "RGB sequence frames must belong to one task, slot, and encounter",
        )
    visual_ids = tuple(frame.visual_frame_id for frame in inputs)
    resource_ids = tuple(frame.resource_id for frame in inputs)
    if len(set(visual_ids)) != len(visual_ids) or len(set(resource_ids)) != len(resource_ids):
        raise EvidenceMediaError(
            "rgb_sequence_identity_mismatch",
            "RGB sequence frame and resource identities must be unique",
        )
    order = tuple(
        (float(frame.camera_timestamp), frame.camera_frame, frame.visual_frame_id)
        for frame in inputs
    )
    if order != tuple(sorted(order)):
        raise EvidenceMediaError(
            "rgb_sequence_identity_mismatch",
            "RGB sequence frames must be in chronological order",
        )

    rendered: list[Image.Image] = []
    source_hashes: list[str] = []
    for frame in inputs:
        image, source_sha256 = _load_and_annotate_rgb(frame)
        rendered.append(image)
        source_hashes.append(source_sha256)

    columns = min(RGB_SEQUENCE_COLUMNS, len(rendered))
    rows = (len(rendered) + columns - 1) // columns
    width = columns * RGB_SEQUENCE_TILE_WIDTH + (columns - 1) * RGB_SEQUENCE_TILE_GAP
    height = rows * RGB_SEQUENCE_TILE_HEIGHT + (rows - 1) * RGB_SEQUENCE_TILE_GAP
    sheet = Image.new("RGB", (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    image_height = RGB_SEQUENCE_TILE_HEIGHT - RGB_SEQUENCE_HEADER_HEIGHT
    for index, (frame, source_image) in enumerate(zip(inputs, rendered)):
        column = index % columns
        row = index // columns
        left = column * (RGB_SEQUENCE_TILE_WIDTH + RGB_SEQUENCE_TILE_GAP)
        top = row * (RGB_SEQUENCE_TILE_HEIGHT + RGB_SEQUENCE_TILE_GAP)
        scale = min(
            RGB_SEQUENCE_TILE_WIDTH / source_image.width,
            image_height / source_image.height,
        )
        resized_width = max(1, int(round(source_image.width * scale)))
        resized_height = max(1, int(round(source_image.height * scale)))
        resized = source_image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.LANCZOS,
        )
        paste_x = left + (RGB_SEQUENCE_TILE_WIDTH - resized_width) // 2
        paste_y = top + RGB_SEQUENCE_HEADER_HEIGHT + (image_height - resized_height) // 2
        sheet.paste(resized, (paste_x, paste_y))
        draw.rectangle(
            (left, top, left + RGB_SEQUENCE_TILE_WIDTH - 1, top + RGB_SEQUENCE_TILE_HEIGHT - 1),
            outline=(100, 116, 139),
            width=1,
        )
        draw.rectangle(
            (left, top, left + RGB_SEQUENCE_TILE_WIDTH - 1, top + RGB_SEQUENCE_HEADER_HEIGHT - 1),
            fill=(30, 41, 59),
        )
        draw.text(
            (left + 8, top + 8),
            (
                f"{index + 1}/{len(inputs)} TARGET {frame.slot_id} | "
                f"visual {frame.visual_frame_id} | lidar {frame.lidar_frame} / camera {frame.camera_frame}"
            ),
            fill=(255, 255, 255),
            font=font,
        )

    aggregate_sha256 = _bytes_sha256(
        _canonical_json_bytes(
            {
                "renderer_version": RGB_SEQUENCE_RENDERER_VERSION,
                "frames": [
                    {
                        "camera_frame": frame.camera_frame,
                        "encounter_id": frame.encounter_id,
                        "lidar_frame": frame.lidar_frame,
                        "resource_id": frame.resource_id,
                        "sha256": source_sha256,
                        "slot_id": frame.slot_id,
                        "task_id": frame.task_id,
                        "visual_frame_id": frame.visual_frame_id,
                    }
                    for frame, source_sha256 in zip(inputs, source_hashes)
                ],
            }
        )
    )
    return _png_bytes(sheet), sheet.width, sheet.height, aggregate_sha256


class EvidenceMediaStore:
    """Content-address rendered media and bind it to successful evidence IDs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._bindings: dict[str, MediaArtifact] = {}
        self._lock = threading.RLock()

    def _bind_png(
        self,
        evidence_id: str,
        content: bytes,
        *,
        kind: str,
        width: int,
        height: int,
        source_sha256: str,
        renderer_version: str,
    ) -> MediaArtifact:
        _require_evidence_id(evidence_id)
        media_sha256 = _bytes_sha256(content)
        digest = media_sha256.split(":", 1)[1]
        destination = self.root / "sha256" / digest[:2] / f"{digest}.png"
        with self._lock:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                try:
                    if destination.stat().st_size != len(content):
                        raise EvidenceMediaError(
                            "media_cache_corrupt",
                            "content-addressed media cache has the wrong size",
                        )
                    if _file_sha256(destination) != media_sha256:
                        raise EvidenceMediaError(
                            "media_cache_corrupt",
                            "content-addressed media cache is corrupted",
                        )
                except EvidenceMediaError:
                    raise
                except OSError as exc:
                    raise EvidenceMediaError(
                        "media_cache_unavailable",
                        "content-addressed media cache cannot be read",
                    ) from exc
            else:
                temporary_name: str | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=destination.parent,
                        prefix=f".{destination.name}.",
                        suffix=".tmp",
                        delete=False,
                    ) as handle:
                        temporary_name = handle.name
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary_name, destination)
                except OSError as exc:
                    raise EvidenceMediaError(
                        "media_cache_unavailable",
                        "content-addressed media cache cannot be written",
                    ) from exc
                finally:
                    if temporary_name is not None:
                        temporary = Path(temporary_name)
                        try:
                            if temporary.exists():
                                temporary.unlink()
                        except OSError:
                            pass
            artifact = MediaArtifact(
                evidence_id=evidence_id,
                kind=kind,
                path=destination,
                sha256=media_sha256,
                size_bytes=len(content),
                width=width,
                height=height,
                source_sha256=source_sha256,
                renderer_version=renderer_version,
            )
            self._bindings[evidence_id] = artifact
            return artifact

    def publish_lidar(
        self,
        evidence_id: str,
        pack_path: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_slot_id: str | None = None,
        expected_task_id: str | None = None,
        expected_encounter_id: str | None = None,
        expected_dataset_id: str | None = None,
        expected_config_hash: str | None = None,
        expected_slot_map_hash: str | None = None,
    ) -> MediaArtifact:
        if not isinstance(evidence_id, str) or EVIDENCE_ID_PATTERN.fullmatch(evidence_id) is None:
            raise EvidencePackError("media binding requires an opaque evidence_id")
        pack = load_lidar_evidence_pack(
            pack_path,
            expected_sha256=expected_sha256,
            expected_slot_id=expected_slot_id,
            expected_task_id=expected_task_id,
            expected_encounter_id=expected_encounter_id,
            expected_dataset_id=expected_dataset_id,
            expected_config_hash=expected_config_hash,
            expected_slot_map_hash=expected_slot_map_hash,
        )
        content = render_lidar_triptych(pack)
        return self._bind_png(
            evidence_id,
            content,
            kind="lidar_triptych_png",
            width=LIDAR_MEDIA_WIDTH,
            height=LIDAR_MEDIA_HEIGHT,
            source_sha256=pack.source_sha256,
            renderer_version=LIDAR_MEDIA_RENDERER_VERSION,
        )

    def publish_rgb_frame(
        self,
        evidence_id: str,
        frame: RgbFrameMediaInput,
    ) -> MediaArtifact:
        """Publish a newly generated target-marked RGB frame."""

        _require_evidence_id(evidence_id)
        if not isinstance(frame, RgbFrameMediaInput):
            raise EvidenceMediaError(
                "rgb_identity_invalid",
                "RGB frame media requires an identity-bound frame input",
            )
        content, width, height, source_sha256 = render_rgb_frame(frame)
        return self._bind_png(
            evidence_id,
            content,
            kind="rgb_target_frame_png",
            width=width,
            height=height,
            source_sha256=source_sha256,
            renderer_version=RGB_FRAME_RENDERER_VERSION,
        )

    def publish_rgb_sequence(
        self,
        evidence_id: str,
        frames: Sequence[RgbFrameMediaInput],
    ) -> MediaArtifact:
        """Publish a deterministic contact sheet for one ordered RGB sequence."""

        _require_evidence_id(evidence_id)
        content, width, height, source_sha256 = render_rgb_sequence(frames)
        return self._bind_png(
            evidence_id,
            content,
            kind="rgb_target_sequence_contact_sheet_png",
            width=width,
            height=height,
            source_sha256=source_sha256,
            renderer_version=RGB_SEQUENCE_RENDERER_VERSION,
        )

    def resolve(self, evidence_id: str) -> MediaArtifact | None:
        """Return only media published by a completed tool execution."""

        with self._lock:
            return self._bindings.get(evidence_id)

    def read_image(self, evidence_id: str) -> tuple[str, bytes] | None:
        """Read verified bound media without disclosing its filesystem path."""

        if not isinstance(evidence_id, str) or EVIDENCE_ID_PATTERN.fullmatch(evidence_id) is None:
            return None
        with self._lock:
            artifact = self._bindings.get(evidence_id)
            if artifact is None:
                return None
            try:
                content = artifact.path.read_bytes()
            except OSError:
                self._bindings.pop(evidence_id, None)
                return None
            if len(content) != artifact.size_bytes or _bytes_sha256(content) != artifact.sha256:
                self._bindings.pop(evidence_id, None)
                return None
            if content.startswith(b"\x89PNG\r\n\x1a\n"):
                media_type = "image/png"
            elif content.startswith(b"\xff\xd8\xff"):
                media_type = "image/jpeg"
            else:
                self._bindings.pop(evidence_id, None)
                return None
            return media_type, content

    def manifest_records(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(
                self._bindings[evidence_id].manifest_record()
                for evidence_id in sorted(self._bindings)
            )


__all__ = [
    "LIDAR_MEDIA_HEIGHT",
    "LIDAR_MEDIA_RENDERER_VERSION",
    "LIDAR_MEDIA_WIDTH",
    "RGB_FRAME_RENDERER_VERSION",
    "RGB_SEQUENCE_MAX_FRAMES",
    "RGB_SEQUENCE_RENDERER_VERSION",
    "EvidenceMediaError",
    "EvidenceMediaStore",
    "EvidencePackError",
    "LidarEvidencePack",
    "MediaArtifact",
    "RgbFrameMediaInput",
    "is_lidar_evidence_pack",
    "load_lidar_evidence_pack",
    "render_lidar_triptych",
    "render_rgb_frame",
    "render_rgb_sequence",
]
