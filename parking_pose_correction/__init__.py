"""Map-pose drift diagnostics and correction utilities."""

from .registration import AlignmentMetrics, RegistrationResult, estimate_local_correction, evaluate_alignment
from .se2 import (
    apply_pose_corrections,
    interpolate_corrections,
    select_keyframes,
    stabilize_dense_corrections,
    smooth_keyframe_corrections,
)
from .timestamps import TimestampMatch, nearest_timestamp_match

__all__ = [
    "RegistrationResult",
    "AlignmentMetrics",
    "TimestampMatch",
    "apply_pose_corrections",
    "estimate_local_correction",
    "evaluate_alignment",
    "interpolate_corrections",
    "nearest_timestamp_match",
    "select_keyframes",
    "stabilize_dense_corrections",
    "smooth_keyframe_corrections",
]
