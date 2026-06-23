from dataclasses import dataclass, field
from typing import Any

import numpy as np


UNKNOWN = 0
FREE = 1
OCCUPIED = 2
OCCLUDED_UNKNOWN = 3


@dataclass
class SlotHypothesis:
    """Synthetic parking slot hypothesis."""

    slot_id: str
    polygon_xy: np.ndarray
    entrance_xy: np.ndarray
    is_target_candidate: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticSample:
    """Synthetic parking sample with observed data and ground truth."""

    sample_id: str
    front_images: dict[str, np.ndarray]

    # Full point cloud is used only for ground truth generation and local tool evidence.
    lidar_points_full: np.ndarray

    # Observed point cloud is used to build initial perception.
    lidar_points_observed: np.ndarray

    fake_slots: list[SlotHypothesis]

    gt_occupancy: np.ndarray
    gt_risk_zones: np.ndarray
    gt_occlusion_zones: np.ndarray
    target_slot_mask: np.ndarray

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegionProposal:
    """A high-priority region selected by the active perception agent."""

    region_id: str
    center_cell: tuple[int, int]
    bbox_cells: tuple[int, int, int, int]
    priority: float
    issue_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result returned by an active perception tool."""

    tool_name: str
    region_id: str
    confidence_delta: float
    updates: dict[str, Any]
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParkingBeliefState:
    """Structured parking perception state."""

    occupancy: np.ndarray
    height: np.ndarray
    unknown: np.ndarray
    occlusion: np.ndarray
    boundary: np.ndarray
    uncertainty: np.ndarray
    decision_impact: np.ndarray
    priority: np.ndarray
    selected_regions: list[RegionProposal]
    metadata: dict[str, Any] = field(default_factory=dict)
