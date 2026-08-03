from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .timestamps import nearest_timestamp_match


def load_timestamp_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    timestamps: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2:
                continue
            frames.append(int(fields[0]))
            timestamps.append(float(fields[1]))
    order = np.argsort(timestamps)
    return np.asarray(frames, dtype=np.int64)[order], np.asarray(timestamps, dtype=np.float64)[order]


def synchronize_frame_rows(
    rows: Sequence[Mapping[str, object]],
    camera_frames: np.ndarray,
    camera_timestamps: np.ndarray,
    *,
    image_dir: str | Path,
    max_delta_sec: float,
) -> list[dict[str, object]]:
    image_dir = Path(image_dir)
    synchronized: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        lidar_timestamp = float(row["lidar_timestamp"])
        match = nearest_timestamp_match(camera_frames, camera_timestamps, lidar_timestamp)
        row.update(
            {
                "camera_frame": match.frame,
                "camera_image_path": str(image_dir / f"left{match.frame:06d}.png"),
                "camera_timestamp": match.timestamp,
                "camera_lidar_dt_sec": match.delta_sec,
                "camera_match_valid": int(abs(match.delta_sec) <= max_delta_sec),
            }
        )
        synchronized.append(row)
    return synchronized


def yaw_to_rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def lidar_to_map_xyzi(raw_xyzi: np.ndarray, pose_map: np.ndarray, map_scale: float) -> np.ndarray:
    raw = np.asarray(raw_xyzi, dtype=np.float32)
    pose = np.asarray(pose_map, dtype=np.float64)
    out = np.empty_like(raw, dtype=np.float32)
    out[:, :2] = (
        pose[:2] + float(map_scale) * (raw[:, :2].astype(np.float64) @ yaw_to_rotation(float(pose[2])).T)
    ).astype(np.float32)
    out[:, 2:] = raw[:, 2:]
    return out
