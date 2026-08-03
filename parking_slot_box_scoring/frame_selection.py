from __future__ import annotations

import ast
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SlotFrameSelection:
    slot_id: str
    anchor_frame: int
    selected_frames: list[int]
    baseline_state: str = ""
    baseline_score: float = 0.0


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def parse_frame_ids(value: str) -> list[int]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = []
    return [int(v) for v in parsed]


def load_baseline_selection(baseline_dir: str | Path) -> dict[str, SlotFrameSelection]:
    baseline_path = Path(baseline_dir) / "slot_aligned_evidence.csv"
    if not baseline_path.exists():
        return {}
    selections: dict[str, SlotFrameSelection] = {}
    for row in read_csv_rows(baseline_path):
        slot_id = row.get("slot_id", "")
        if not slot_id:
            continue
        frames = parse_frame_ids(row.get("sampled_frame_ids", ""))
        anchor_frame = int(float(row.get("anchor_frame") or (frames[len(frames) // 2] if frames else 0)))
        try:
            baseline_score = float(row.get("max_vehicle_like_score") or 0.0)
        except ValueError:
            baseline_score = 0.0
        selections[slot_id] = SlotFrameSelection(
            slot_id=slot_id,
            anchor_frame=anchor_frame,
            selected_frames=frames,
            baseline_state=row.get("state_by_slot_aligned_accumulation", ""),
            baseline_score=baseline_score,
        )
    return selections


def sample_anchor_window(frame_numbers: list[int], anchor_frame: int, before: int, after: int, stride: int) -> list[int]:
    if not frame_numbers:
        return []
    frame_set = set(frame_numbers)
    raw = list(range(anchor_frame - before, anchor_frame + after + 1, max(stride, 1)))
    if anchor_frame not in raw:
        raw.append(anchor_frame)
    sampled = sorted(f for f in raw if f in frame_set)
    if sampled:
        return sampled
    nearest = min(frame_numbers, key=lambda f: abs(f - anchor_frame))
    return [nearest]


def select_pose_aligned_anchor(frame_rows: list[dict[str, str]], slot: dict, map_units_per_meter: float) -> int:
    center = np.asarray(slot.get("center_np", slot.get("center_map")), dtype=np.float64)
    best_frame = None
    best_score = math.inf
    for row in frame_rows:
        try:
            frame_id = int(row["frame"])
            ego = np.asarray([float(row["map_x"]), float(row["map_y"])], dtype=np.float64)
        except (KeyError, ValueError):
            continue
        distance_m = float(np.linalg.norm(ego - center) / max(map_units_per_meter, 1e-9))
        score = distance_m
        if score < best_score:
            best_score = score
            best_frame = frame_id
    if best_frame is None:
        raise ValueError("could not select pose-aligned anchor frame")
    return best_frame


def build_slot_frame_selections(
    slots: list[dict],
    frame_rows: list[dict[str, str]],
    baseline_dir: str | Path,
    map_units_per_meter: float,
    window_before: int,
    window_after: int,
    frame_stride: int,
    frame_selection: str = "anchor_window",
) -> list[SlotFrameSelection]:
    frame_numbers = sorted(int(row["frame"]) for row in frame_rows if row.get("frame"))
    baseline = load_baseline_selection(baseline_dir)
    if baseline:
        return list(baseline.values())
    selections: list[SlotFrameSelection] = []
    for slot in slots:
        slot_id = str(slot["slot_id"])
        anchor_frame = select_pose_aligned_anchor(frame_rows, slot, map_units_per_meter)
        selected = sample_anchor_window(frame_numbers, anchor_frame, window_before, window_after, frame_stride)
        selections.append(SlotFrameSelection(slot_id=slot_id, anchor_frame=anchor_frame, selected_frames=selected))
    return selections
