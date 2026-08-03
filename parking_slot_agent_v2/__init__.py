"""Independent data contracts for the Parking Slot Agent v2 pipeline."""

from .contracts import (
    PART1_OUTPUT_SCHEMA_VERSION,
    CaseStatus,
    ConfidenceScores,
    EvidenceRecord,
    FovResult,
    FovVisibility,
    MapSlot,
    Part1Output,
    ReasoningRound,
    SceneSnapshot,
    SensorFrame,
    SlotCase,
    SlotState,
)
from .io import (
    atomic_save_json,
    load_json,
    load_json_object,
    load_part1_output,
    load_slot_case,
    save_json_atomic,
    save_part1_output,
    save_slot_case,
)


__all__ = [
    "PART1_OUTPUT_SCHEMA_VERSION",
    "CaseStatus",
    "ConfidenceScores",
    "EvidenceRecord",
    "FovResult",
    "FovVisibility",
    "MapSlot",
    "Part1Output",
    "ReasoningRound",
    "SceneSnapshot",
    "SensorFrame",
    "SlotCase",
    "SlotState",
    "atomic_save_json",
    "load_json",
    "load_json_object",
    "load_part1_output",
    "load_slot_case",
    "save_json_atomic",
    "save_part1_output",
    "save_slot_case",
]
