"""Conservative hybrid-3D parking-slot point-cloud evidence."""

from .config import Hybrid3DConfig
from .contracts import DecisionState, KnownSlot, ScopeStatus, SlotDecision
from .camera import (
    CameraSelectionConfig,
    load_camera_model,
    select_camera_assessments,
)
from .part2_evidence import (
    LIDAR_EVIDENCE_PACK_SCHEMA_VERSION,
    build_lidar_evidence_pack,
)
from .local_map import (
    LOCAL_MAP_SCHEMA_VERSION,
    LocalMapConfig,
    actual_lidar_coverage_polygon,
    build_local_map_snapshot,
    restrict_result_to_local_map,
    select_local_frame_window,
)

__all__ = [
    "DecisionState",
    "CameraSelectionConfig",
    "Hybrid3DConfig",
    "LIDAR_EVIDENCE_PACK_SCHEMA_VERSION",
    "LOCAL_MAP_SCHEMA_VERSION",
    "LocalMapConfig",
    "KnownSlot",
    "ScopeStatus",
    "SlotDecision",
    "build_lidar_evidence_pack",
    "actual_lidar_coverage_polygon",
    "build_local_map_snapshot",
    "load_camera_model",
    "select_camera_assessments",
    "restrict_result_to_local_map",
    "select_local_frame_window",
]
