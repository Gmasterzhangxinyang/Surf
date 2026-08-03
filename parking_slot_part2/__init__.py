"""Provider-neutral Part2 queue contracts.

The local-VLM HTTP integration is an optional dependency boundary.  Its
symbols are loaded lazily so queue validation and deterministic replay do not
require ``httpx`` merely to import this package.
"""

from importlib import import_module

from .contracts import (
    EncounterRef,
    QueueEnvelope,
    QueueItem,
    RelationshipRef,
    ResourceRef,
    VisualFrameRef,
)
from .decision import (
    DECISION_POLICY_VERSION,
    FINAL_ROUTE_STATES_SCHEMA_VERSION,
    RESOLUTIONS_SCHEMA_VERSION,
    GatedAssessment,
    SlotResolution,
    consolidate_group_assessments,
    gate_assessment,
    merge_final_route_states,
)
from .grouping import GROUPING_POLICY_VERSION, TaskGroup, build_groups
from .model import (
    MODEL_TURN_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
    GroupFinalProposal,
    ModelAdapter,
    ReplayModelAdapter,
    SlotAssessment,
    ToolRequest,
)
from .media import (
    EvidenceMediaError,
    EvidenceMediaStore,
    MediaArtifact,
    RgbFrameMediaInput,
)
from .orchestrator import (
    MAX_MODEL_TURNS,
    MAX_TOOL_ATTEMPTS,
    GroupRunResult,
    ToolAttemptRecord,
    run_group,
)
from .preflight import CameraPreflight, EvidenceCatalog, EvidenceRecord
from .queueing import (
    QUEUE_SCHEMA_VERSION,
    TOOL_NAMES,
    TOOL_REGISTRY_VERSION,
    canonical_json_bytes,
    canonical_sha256,
    load_queue,
    make_queue_id,
    make_task_id,
    validate_queue,
)
from .reporting import derive_part2_run_id, load_base_slot_decisions, write_run_outputs
from .shadow import ShadowSelection, ShadowSubset, build_shadow_subset, select_shadow_tasks
from .shadow_replay import (
    SHADOW_REPLAY_POLICY_VERSION,
    BlindSlotJudgement,
    compile_shadow_replay,
)
from .tools import ToolRegistry, ToolResult
from .trace import TRACE_SCHEMA_VERSION, EvidenceLedger


_LOCAL_VLM_EXPORTS = frozenset(
    {
        "LOCAL_VLM_ADAPTER_VERSION",
        "LocalImage",
        "LocalOpenAICompatibleVLMAdapter",
        "LocalVLMAdapter",
        "LocalVLMError",
        "LocalVLMHTTPError",
        "LocalVLMMediaError",
        "LocalVLMProtocolError",
        "LocalVLMRequestError",
        "LocalVLMResponseTooLarge",
        "LocalVLMTimeoutError",
        "MediaResolver",
    }
)


def __getattr__(name: str):
    if name not in _LOCAL_VLM_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".local_vlm", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LOCAL_VLM_EXPORTS)

__all__ = [
    "QUEUE_SCHEMA_VERSION",
    "DECISION_POLICY_VERSION",
    "FINAL_ROUTE_STATES_SCHEMA_VERSION",
    "GROUPING_POLICY_VERSION",
    "LOCAL_VLM_ADAPTER_VERSION",
    "MODEL_TURN_SCHEMA_VERSION",
    "REPLAY_SCHEMA_VERSION",
    "RESOLUTIONS_SCHEMA_VERSION",
    "SHADOW_REPLAY_POLICY_VERSION",
    "TRACE_SCHEMA_VERSION",
    "TOOL_NAMES",
    "TOOL_REGISTRY_VERSION",
    "MAX_MODEL_TURNS",
    "MAX_TOOL_ATTEMPTS",
    "BlindSlotJudgement",
    "CameraPreflight",
    "EncounterRef",
    "EvidenceCatalog",
    "EvidenceRecord",
    "EvidenceLedger",
    "EvidenceMediaError",
    "EvidenceMediaStore",
    "GatedAssessment",
    "GroupFinalProposal",
    "GroupRunResult",
    "LocalImage",
    "LocalOpenAICompatibleVLMAdapter",
    "LocalVLMAdapter",
    "LocalVLMError",
    "LocalVLMHTTPError",
    "LocalVLMMediaError",
    "LocalVLMProtocolError",
    "LocalVLMRequestError",
    "LocalVLMResponseTooLarge",
    "LocalVLMTimeoutError",
    "ModelAdapter",
    "MediaArtifact",
    "MediaResolver",
    "QueueEnvelope",
    "QueueItem",
    "RelationshipRef",
    "ReplayModelAdapter",
    "ResourceRef",
    "RgbFrameMediaInput",
    "TaskGroup",
    "ToolAttemptRecord",
    "ToolRegistry",
    "ToolRequest",
    "ToolResult",
    "SlotAssessment",
    "SlotResolution",
    "ShadowSelection",
    "ShadowSubset",
    "VisualFrameRef",
    "canonical_json_bytes",
    "canonical_sha256",
    "build_groups",
    "build_shadow_subset",
    "compile_shadow_replay",
    "consolidate_group_assessments",
    "derive_part2_run_id",
    "gate_assessment",
    "load_base_slot_decisions",
    "load_queue",
    "make_queue_id",
    "make_task_id",
    "run_group",
    "select_shadow_tasks",
    "validate_queue",
    "merge_final_route_states",
    "write_run_outputs",
]
