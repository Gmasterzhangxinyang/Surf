from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

from .accumulation import build_slot_accumulation
from .config import Hybrid3DConfig
from .contracts import (
    AgentContext,
    DecisionState,
    FrameRecord,
    FreeEvidence,
    GateResult,
    KnownSlot,
    OccupiedEvidence,
    ScopeEvidence,
    ScopeStatus,
    SlotDecision,
    StabilityEvidence,
)
from .decision import QualityEvidence, route_decision
from .free_space import evaluate_free_space
from .geometry import metric_slot
from .known_slot_scope import KnownSlotObservationScopeEvaluator
from .occupied import evaluate_occupied
from .stability import evaluate_stability, terminal_variant_passes
from .static_semantics import SemanticStaticOccupiedVeto


class PipelinePointProvider(Protocol):
    def load(self, frame_id: int): ...


@dataclass(frozen=True)
class PipelineResult:
    phase: str
    scopes: tuple[ScopeEvidence, ...]
    decisions: tuple[SlotDecision, ...]
    traces: tuple[Mapping[str, Any], ...]
    processing_seconds: float
    cache_stats: Mapping[str, int]
    map_total: int
    frames: tuple[FrameRecord, ...]
    slots: tuple[KnownSlot, ...]
    all_slots: tuple[KnownSlot, ...]
    map_units_per_meter: float


def _trace_id(slot_id: str) -> str:
    return f"trace:{slot_id}"


def _usable_camera_frame(frame: FrameRecord) -> bool:
    return bool(
        frame.camera_match_valid
        and frame.camera_frame is not None
        and frame.camera_image_path is not None
        and frame.camera_image_path.is_file()
        and frame.lidar_timestamp is not None
        and frame.camera_timestamp is not None
        and frame.camera_lidar_dt_sec is not None
    )


def _agent_context(
    scope: ScopeEvidence,
    frame_ids: Sequence[int],
    frame_by_id: Mapping[int, FrameRecord],
    *,
    lidar_available: bool,
) -> AgentContext:
    has_camera = any(
        _usable_camera_frame(frame_by_id[frame_id])
        for frame_id in frame_ids
        if frame_id in frame_by_id
    )
    tools: list[str] = []
    if lidar_available:
        tools.append("inspect_lidar_map")
    if has_camera:
        tools.extend(("inspect_rgb_frame", "inspect_rgb_sequence"))
    observable = bool(scope.agent_observable and tools)
    return AgentContext(
        agent_observable=observable,
        priority="inspect_vehicle_shape_and_occlusion" if observable else "",
        suggested_tools=tuple(tools if observable else ()),
    )


def _contract_error_decision(
    scope: ScopeEvidence,
    occupied: OccupiedEvidence,
    free: FreeEvidence,
    stability: StabilityEvidence,
    agent_context: AgentContext,
) -> SlotDecision:
    return SlotDecision(
        slot_id=scope.slot_id,
        scope_status=scope.scope_status,
        state=DecisionState.UNKNOWN,
        decision_reason="decision_contract_error",
        unknown_reasons=("decision_contract_error",),
        occupied_evidence=occupied,
        free_evidence=free,
        stability=stability,
        reference_frames=tuple(sorted(set(scope.crossing_frames) | set(scope.hit_frames))),
        agent_context=agent_context,
    )


class Hybrid3DPipeline:
    def __init__(
        self,
        slots: Sequence[KnownSlot],
        frames: Sequence[FrameRecord],
        provider: PipelinePointProvider,
        map_units_per_meter: float,
        config: Hybrid3DConfig | None = None,
        *,
        phase: str = "full",
        slot_ids: Sequence[str] | None = None,
        max_slots: int | None = None,
        static_occupied_veto: SemanticStaticOccupiedVeto | None = None,
    ) -> None:
        if phase not in {"scope", "occupied", "full"}:
            raise ValueError("phase must be scope, occupied, or full")
        if max_slots is not None and max_slots <= 0:
            raise ValueError("max_slots must be positive")
        self.config = config or Hybrid3DConfig()
        self.config.validate()
        self.phase = phase
        self.provider = provider
        self.static_occupied_veto = static_occupied_veto
        self.map_units_per_meter = float(map_units_per_meter)
        self.frames = tuple(sorted(frames, key=lambda item: item.frame_id))
        all_slots = tuple(sorted(slots, key=lambda item: item.slot_id))
        if slot_ids is not None:
            requested = tuple(dict.fromkeys(str(slot_id) for slot_id in slot_ids))
            available = {slot.slot_id for slot in all_slots}
            missing = sorted(set(requested) - available)
            if missing:
                raise ValueError(f"unknown slot IDs: {', '.join(missing)}")
            requested_set = set(requested)
            selected = tuple(slot for slot in all_slots if slot.slot_id in requested_set)
        else:
            selected = all_slots
        if max_slots is not None:
            selected = selected[:max_slots]
        self.all_slots = all_slots
        self.slots = selected

    def _cache_stats(self) -> Mapping[str, int]:
        stats = getattr(self.provider, "stats", {})
        try:
            return {str(key): int(value) for key, value in dict(stats).items()}
        except Exception:
            return {}

    def run(self) -> PipelineResult:
        started = time.perf_counter()
        frame_by_id = {frame.frame_id: frame for frame in self.frames}
        raw_slot_by_id = {slot.slot_id: slot for slot in self.all_slots}
        metric_by_id = {
            slot.slot_id: metric_slot(slot, self.map_units_per_meter)
            for slot in self.all_slots
        }
        scopes = tuple(
            KnownSlotObservationScopeEvaluator(
                self.config,
                self.map_units_per_meter,
            ).evaluate(self.slots, self.frames, self.provider)
        )
        decisions: list[SlotDecision] = []
        traces: list[Mapping[str, Any]] = []

        for scope in scopes:
            trace: dict[str, Any] = {
                "trace_event_id": _trace_id(scope.slot_id),
                "slot_id": scope.slot_id,
                "phase": self.phase,
                "scope": asdict(scope),
                "stages": {"scope": {"status": "ok"}},
            }
            if self.phase == "scope" or scope.scope_status is ScopeStatus.OUT_OF_ROUTE:
                traces.append(trace)
                continue

            if scope.scope_status is ScopeStatus.PARTIAL_ROUTE:
                candidate_frames = tuple(
                    sorted(
                        set(scope.crossing_frames)
                        | set(scope.hit_frames)
                        | set(scope.near_frames)
                    )
                )
                agent_context = _agent_context(
                    scope,
                    candidate_frames,
                    frame_by_id,
                    lidar_available=False,
                )
                decision = route_decision(
                    scope,
                    OccupiedEvidence(),
                    FreeEvidence(),
                    QualityEvidence(),
                    StabilityEvidence(),
                    agent_context,
                )
                if decision is not None:
                    decisions.append(decision)
                    trace["decision"] = asdict(decision)
                trace["stages"]["routing"] = {"status": "partial_route_unknown"}
                traces.append(trace)
                continue

            slot = raw_slot_by_id[scope.slot_id]
            converted_slot = metric_by_id[scope.slot_id]
            try:
                accumulation = build_slot_accumulation(
                    slot,
                    scope,
                    self.frames,
                    self.provider,
                    self.map_units_per_meter,
                    self.config,
                )
                trace["stages"]["accumulation"] = {
                    "status": "ok",
                    "anchor_frame": accumulation.anchor_frame,
                    "selected_frames": accumulation.selected_frames,
                    "valid_frames": tuple(
                        observation.frame_id for observation in accumulation.observations
                    ),
                    "excluded_frames": accumulation.excluded_frames,
                    "point_count": len(accumulation.points_local_xyzi),
                    "ray_endpoint_count": sum(
                        len(observation.ray_endpoints_local_xyz)
                        for observation in accumulation.observations
                    ),
                }
            except Exception as exc:
                agent_context = _agent_context(
                    scope,
                    tuple(sorted(set(scope.crossing_frames) | set(scope.hit_frames))),
                    frame_by_id,
                    lidar_available=False,
                )
                quality = QualityEvidence(
                    passed=False,
                    failures=("accumulation_error",),
                )
                decision = route_decision(
                    scope,
                    OccupiedEvidence(),
                    FreeEvidence(),
                    quality,
                    StabilityEvidence(),
                    agent_context,
                )
                if decision is not None:
                    decisions.append(decision)
                    trace["decision"] = asdict(decision)
                trace["stages"]["accumulation"] = {
                    "status": "error",
                    "reason": "accumulation_error",
                    "error_type": type(exc).__name__,
                }
                traces.append(trace)
                continue

            agent_context = _agent_context(
                scope,
                accumulation.selected_frames,
                frame_by_id,
                lidar_available=bool(accumulation.observations),
            )
            quality_failures: list[str] = (
                [] if accumulation.observations else ["no_valid_observations"]
            )
            occupied = evaluate_occupied(
                converted_slot,
                accumulation,
                metric_by_id,
                self.config,
            )
            static_assessment = None
            if occupied.strong and self.static_occupied_veto is not None:
                try:
                    static_assessment = self.static_occupied_veto.assess(
                        converted_slot,
                        accumulation,
                    )
                    if static_assessment.veto:
                        occupied = replace(
                            occupied,
                            strong=False,
                            weak=True,
                            gate_results=occupied.gate_results
                            + (
                                GateResult(
                                    "semantic_static_map_veto",
                                    False,
                                    (
                                        f"ratio={static_assessment.static_explained_ratio:.6g},"
                                        f"residual_short={static_assessment.residual_short_extent_m:.6g}"
                                    ),
                                    (
                                        f"ratio<{self.static_occupied_veto.max_static_explained_ratio}"
                                        f" and residual_short>="
                                        f"{self.static_occupied_veto.min_residual_short_extent_m}"
                                    ),
                                ),
                            ),
                            failures=tuple(
                                sorted(
                                    set(occupied.failures)
                                    | {static_assessment.reason}
                                )
                            ),
                        )
                except Exception:
                    occupied = replace(
                        occupied,
                        strong=False,
                        weak=False,
                        gate_results=occupied.gate_results
                        + (
                            GateResult(
                                "semantic_static_map_veto",
                                False,
                                "evaluation_error",
                                "successful_evaluation",
                            ),
                        ),
                        failures=tuple(
                            sorted(
                                set(occupied.failures)
                                | {"semantic_static_evaluation_error"}
                            )
                        ),
                    )
            trace["stages"]["semantic_static"] = {
                "status": (
                    "not_configured"
                    if self.static_occupied_veto is None
                    else "not_required"
                    if static_assessment is None and not occupied.strong
                    else "ok"
                ),
                "assessment": (
                    None if static_assessment is None else asdict(static_assessment)
                ),
            }
            trace["stages"]["occupied"] = {
                "status": "error" if "occupied_evaluation_error" in occupied.failures else "ok",
                "evidence": asdict(occupied),
            }

            free = FreeEvidence()
            if self.phase == "full":
                free = evaluate_free_space(
                    converted_slot,
                    accumulation,
                    occupied,
                    self.config,
                )
                trace["stages"]["free"] = {
                    "status": "error" if "free_evaluation_error" in free.failures else "ok",
                    "evidence": asdict(free),
                }

            if "occupied_evaluation_error" in occupied.failures:
                quality_failures.append("occupied_evaluation_error")
            if self.phase == "full" and "free_evaluation_error" in free.failures:
                quality_failures.append("free_evaluation_error")
            quality = QualityEvidence(
                passed=not quality_failures,
                failures=tuple(quality_failures),
            )
            trace["stages"]["quality"] = {
                "status": "ok" if quality.passed else "error",
                "failures": quality.failures,
            }

            stability = StabilityEvidence()
            should_check_occupied = (
                quality.passed
                and occupied.strong
                and not free.conflict
                and not free.strong
            )
            should_check_free = (
                quality.passed
                and self.phase == "full"
                and free.strong
                and not occupied.strong
                and not free.weak_obstacle
            )
            if should_check_occupied and occupied.best_box:
                stability = evaluate_stability(
                    converted_slot,
                    accumulation,
                    lambda current_slot, current_accumulation: terminal_variant_passes(
                        current_slot,
                        current_accumulation,
                        metric_by_id,
                        self.config,
                        terminal_state=DecisionState.OCCUPIED,
                        reference_occupied=occupied,
                        verify_free_context=self.phase == "full",
                    ),
                    self.config,
                )
            elif should_check_free:
                stability = evaluate_stability(
                    converted_slot,
                    accumulation,
                    lambda current_slot, current_accumulation: terminal_variant_passes(
                        current_slot,
                        current_accumulation,
                        metric_by_id,
                        self.config,
                        terminal_state=DecisionState.FREE,
                        reference_occupied=occupied,
                    ),
                    self.config,
                )
            trace["stages"]["stability"] = {
                "status": "error" if stability.failures else "ok",
                "evidence": asdict(stability),
            }

            try:
                decision = route_decision(
                    scope,
                    occupied,
                    free,
                    quality,
                    stability,
                    agent_context,
                )
                routing_status = "ok"
            except ValueError:
                decision = _contract_error_decision(
                    scope,
                    occupied,
                    free,
                    stability,
                    agent_context,
                )
                routing_status = "decision_contract_error"
            if decision is not None:
                if self.phase == "occupied" and decision.state is DecisionState.FREE:
                    raise AssertionError("occupied phase emitted free")
                decisions.append(decision)
                trace["decision"] = asdict(decision)
            trace["stages"]["routing"] = {"status": routing_status}
            traces.append(trace)

        return PipelineResult(
            phase=self.phase,
            scopes=scopes,
            decisions=tuple(sorted(decisions, key=lambda item: item.slot_id)),
            traces=tuple(traces),
            processing_seconds=float(time.perf_counter() - started),
            cache_stats=self._cache_stats(),
            map_total=len(self.slots),
            frames=self.frames,
            slots=self.slots,
            all_slots=self.all_slots,
            map_units_per_meter=self.map_units_per_meter,
        )
