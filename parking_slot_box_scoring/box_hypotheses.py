from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import BoxScoringConfig
from .geometry import SlotFrame


@dataclass(frozen=True)
class CandidateBox:
    center: np.ndarray
    yaw: float
    length: float
    width: float


def enumerate_box_hypotheses(frame: SlotFrame, config: BoxScoringConfig) -> list[CandidateBox]:
    boxes: list[CandidateBox] = []
    for length_scale in config.length_scales:
        for width_scale in config.width_scales:
            length = frame.length * length_scale
            width = frame.width * width_scale
            for yaw_offset_deg in config.yaw_offsets_deg:
                yaw = frame.yaw + math.radians(yaw_offset_deg)
                for long_offset in config.longitudinal_offsets:
                    for lat_offset in config.lateral_offsets:
                        center = frame.center + frame.long_axis * (long_offset * frame.length) + frame.short_axis * (lat_offset * frame.width)
                        boxes.append(CandidateBox(center=center, yaw=yaw, length=length, width=width))
    return boxes
