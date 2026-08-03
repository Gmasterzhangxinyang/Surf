from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import FrameRecord, KnownSlot


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrameLoadError(RuntimeError):
    def __init__(self, frame_id: int, path: Path | None, reason: str) -> None:
        self.frame_id = int(frame_id)
        self.path = path
        self.reason = str(reason)
        super().__init__(f"frame {self.frame_id}: {self.reason}: {self.path}")


def _required_float(row: Mapping[str, str], field: str, frame_id: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"frame {frame_id} has invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"frame {frame_id} has non-finite pose field {field}")
    return value


def _optional_float(row: Mapping[str, str], field: str) -> float | None:
    raw = row.get(field, "")
    if raw is None or not str(raw).strip():
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite optional field {field}")
    return value


def _optional_int(row: Mapping[str, str], field: str) -> int | None:
    raw = row.get(field, "")
    if raw is None or not str(raw).strip():
        return None
    return int(float(raw))


def _optional_path(row: Mapping[str, str], field: str) -> Path | None:
    raw = row.get(field, "")
    return Path(raw) if raw is not None and str(raw).strip() else None


def _parse_bool(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "y"}


def load_frame_records(path: str | Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    seen: set[int] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                frame_id = int(row.get("frame", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError("frame row has invalid frame ID") from exc
            if frame_id in seen:
                raise ValueError(f"duplicate frame_id {frame_id}")
            seen.add(frame_id)
            try:
                record = FrameRecord(
                    frame_id=frame_id,
                    map_x=_required_float(row, "map_x", frame_id),
                    map_y=_required_float(row, "map_y", frame_id),
                    map_yaw=_required_float(row, "map_yaw", frame_id),
                    map_points_path=_optional_path(row, "map_points_path"),
                    lidar_path=_optional_path(row, "lidar_path"),
                    lidar_timestamp=_optional_float(row, "lidar_timestamp"),
                    camera_frame=_optional_int(row, "camera_frame"),
                    camera_image_path=_optional_path(row, "camera_image_path"),
                    camera_timestamp=_optional_float(row, "camera_timestamp"),
                    camera_lidar_dt_sec=_optional_float(row, "camera_lidar_dt_sec"),
                    camera_match_valid=_parse_bool(row.get("camera_match_valid")),
                )
            except ValueError as exc:
                if "non-finite pose" in str(exc):
                    raise ValueError(f"frame {frame_id} has non-finite pose") from exc
                raise
            records.append(record)
    return sorted(records, key=lambda record: record.frame_id)


def _slot_polygon(row: Mapping[str, Any], primary: str, fallbacks: Sequence[str]) -> np.ndarray:
    for field in (primary, *fallbacks):
        value = row.get(field)
        if value is not None:
            return np.asarray(value, dtype=np.float64)
    raise ValueError(f"slot {row.get('slot_id', '')} is missing {primary}")


def load_known_slots(path: str | Path) -> tuple[list[KnownSlot], float]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        raw_slots = payload.get("slots")
        scale = float(payload.get("map_units_per_meter", 1.0))
        declared_count = payload.get("slot_count")
    elif isinstance(payload, list):
        raw_slots = payload
        scale = 1.0
        declared_count = None
    else:
        raise ValueError("slot database must be an object or list")
    if not isinstance(raw_slots, list):
        raise ValueError("slot database has no slots list")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("map_units_per_meter must be positive and finite")
    if declared_count is not None and int(declared_count) != len(raw_slots):
        raise ValueError("slot_count does not match slots list")

    slots: list[KnownSlot] = []
    seen: set[str] = set()
    for row in raw_slots:
        if not isinstance(row, dict):
            raise ValueError("slot entry must be an object")
        slot_id = str(row.get("slot_id", ""))
        if not slot_id:
            raise ValueError("slot_id is required")
        if slot_id in seen:
            raise ValueError(f"duplicate slot_id {slot_id}")
        seen.add(slot_id)
        polygon = _slot_polygon(row, "polygon_map", ())
        core = _slot_polygon(row, "core_polygon_map", ("inner_polygon", "polygon_map"))
        margin = _slot_polygon(row, "margin_polygon_map", ("margin_polygon", "polygon_map"))
        if row.get("center_map") is None:
            raise ValueError(f"slot {slot_id} is missing center_map")
        if row.get("heading_deg") is None:
            raise ValueError(f"slot {slot_id} is missing heading_deg")
        heading = float(row["heading_deg"])
        if not math.isfinite(heading):
            raise ValueError(f"slot {slot_id} has non-finite heading_deg")
        slots.append(
            KnownSlot(
                slot_id=slot_id,
                polygon_map=polygon,
                core_polygon_map=core,
                margin_polygon_map=margin,
                center_map=np.asarray(row["center_map"], dtype=np.float64),
                heading_deg=heading,
                adjacent_slots=tuple(str(value) for value in row.get("adjacent_slots", ()) or ()),
            )
        )
    return sorted(slots, key=lambda slot: slot.slot_id), scale


class FramePointProvider:
    def __init__(
        self,
        frame_by_id: Mapping[int, FrameRecord],
        map_points_dir: str | Path,
        cache_size: int = 32,
        project_root: str | Path = PROJECT_ROOT,
    ) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self._frame_by_id = dict(frame_by_id)
        self._map_points_dir = Path(map_points_dir)
        self._project_root = Path(project_root)
        self._cache_size = int(cache_size)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._load_errors = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "load_errors": self._load_errors,
        }

    @property
    def cached_frame_ids(self) -> tuple[int, ...]:
        return tuple(self._cache)

    def _resolve_path(self, record: FrameRecord) -> Path:
        candidates: list[Path] = []
        if record.map_points_path is not None:
            raw = record.map_points_path
            candidates.append(raw)
            if not raw.is_absolute():
                candidates.append(self._project_root / raw)
                candidates.append(self._map_points_dir / raw)
        candidates.append(self._map_points_dir / f"{record.frame_id:06d}.npz")
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return candidates[0]

    def load(self, frame_id: int) -> np.ndarray:
        frame_id = int(frame_id)
        cached = self._cache.pop(frame_id, None)
        if cached is not None:
            self._cache[frame_id] = cached
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        record = self._frame_by_id.get(frame_id)
        if record is None:
            self._load_errors += 1
            raise FrameLoadError(frame_id, None, "unknown_frame")
        path = self._resolve_path(record)
        if not path.exists():
            self._load_errors += 1
            raise FrameLoadError(frame_id, path, "missing_map_points")
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "points_map_xyzi" not in payload.files:
                    raise FrameLoadError(frame_id, path, "missing_points_map_xyzi")
                points = np.asarray(payload["points_map_xyzi"], dtype=np.float64)
        except FrameLoadError:
            self._load_errors += 1
            raise
        except Exception as exc:
            self._load_errors += 1
            raise FrameLoadError(frame_id, path, "unreadable_map_points") from exc
        if points.ndim != 2 or points.shape[1] < 4:
            self._load_errors += 1
            raise FrameLoadError(frame_id, path, "invalid_map_points_shape")
        points = np.ascontiguousarray(points[:, :4], dtype=np.float64)
        if not np.isfinite(points).all():
            self._load_errors += 1
            raise FrameLoadError(frame_id, path, "nonfinite_map_points")
        points.setflags(write=False)
        self._cache[frame_id] = points
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return points


def _canonicalize(value: Any, key: str | None = None) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _canonicalize(value.tolist(), key=key)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(name): _canonicalize(item, key=str(name)) for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        items = [_canonicalize(item) for item in value]
        if items and all(isinstance(item, dict) and "slot_id" in item for item in items):
            return sorted(items, key=lambda item: str(item["slot_id"]))
        if key in {"frames", "selected_frames", "support_frames", "reference_frames"} and all(
            isinstance(item, int) and not isinstance(item, bool) for item in items
        ):
            return sorted(set(int(item) for item in items))
        if key in {"reasons", "unknown_reasons", "failures", "suggested_tools"}:
            return sorted(set(str(item) for item in items))
        return items
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical output cannot contain non-finite floats")
    return value


def _atomic_write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def write_json_atomic(path: str | Path, data: Any) -> None:
    canonical = _canonicalize(data)
    text = json.dumps(canonical, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _atomic_write_text(path, text)


def write_text_atomic(path: str | Path, text: str) -> None:
    _atomic_write_text(path, str(text))


def write_jsonl_atomic(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    canonical_rows = _canonicalize(list(rows))
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in canonical_rows
    )
    _atomic_write_text(path, text)


def _csv_value(value: Any, field_name: str) -> Any:
    canonical = _canonicalize(value, key=field_name)
    if isinstance(canonical, (dict, list)):
        return json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return canonical


def write_csv_atomic(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    canonical_rows = _canonicalize(list(rows))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in canonical_rows:
        writer.writerow({field: _csv_value(row.get(field, ""), field) for field in fieldnames})
    _atomic_write_text(path, buffer.getvalue())
