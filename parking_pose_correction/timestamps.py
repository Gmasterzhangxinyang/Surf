from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimestampMatch:
    frame: int
    timestamp: float
    delta_sec: float


def nearest_timestamp_match(
    frames: np.ndarray,
    timestamps: np.ndarray,
    query_timestamp: float,
) -> TimestampMatch:
    frames = np.asarray(frames, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if frames.ndim != 1 or timestamps.ndim != 1 or len(frames) != len(timestamps):
        raise ValueError("frames and timestamps must be one-dimensional arrays of equal length")
    if len(frames) == 0:
        raise ValueError("cannot match against an empty timestamp sequence")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be sorted")

    right = int(np.searchsorted(timestamps, query_timestamp, side="left"))
    candidates = [max(0, min(len(timestamps) - 1, right))]
    if right > 0:
        candidates.append(right - 1)
    index = min(candidates, key=lambda i: (abs(float(timestamps[i]) - query_timestamp), i))
    timestamp = float(timestamps[index])
    return TimestampMatch(int(frames[index]), timestamp, timestamp - float(query_timestamp))
