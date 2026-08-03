from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from parking_slot_part2.contracts import (
    EncounterRef,
    QueueEnvelope,
    QueueItem,
    RelationshipRef,
    ResourceRef,
    VisualFrameRef,
    freeze_json,
)
from parking_slot_part2.decision import (
    SlotResolution,
    consolidate_group_assessments,
    gate_assessment,
    merge_final_route_states,
)
from parking_slot_part2.grouping import TaskGroup, build_groups
from parking_slot_part2.model import GroupFinalProposal, ReplayModelAdapter, SlotAssessment
from parking_slot_part2.orchestrator import GroupRunResult, ToolAttemptRecord, run_group
from parking_slot_part2.preflight import EvidenceCatalog
from parking_slot_part2.queueing import (
    canonical_json_bytes,
    canonical_sha256,
    load_queue,
    make_queue_id,
    make_task_id,
    validate_queue,
)
from parking_slot_part2.reporting import (
    derive_part2_run_id,
    load_base_slot_decisions,
    write_run_outputs,
)
from parking_slot_part2.trace import EvidenceLedger
from parking_slot_part2.tools import ToolRegistry


HASH = "sha256:" + "a" * 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "run_part2_agent.py"


def _content_hash(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _resource(resource_id: str, kind: str, uri: str, content: bytes) -> ResourceRef:
    return ResourceRef(
        resource_id=resource_id,
        kind=kind,
        uri=uri,
        sha256=_content_hash(content),
        manifest_hash=None,
        dataset_id="route-a",
        config_hash=HASH,
        slot_map_hash=HASH,
    )


def _visual(number: int) -> VisualFrameRef:
    return VisualFrameRef(
        visual_frame_id=f"visual-{number}",
        lidar_frame=2,
        camera_frame=102 + number,
        lidar_timestamp=2.0,
        camera_timestamp=1.99,
        camera_lidar_dt_sec=-0.01,
        image_resource_id=f"rgb-{number}",
        capabilities=("can_assess_occupied", "can_assess_free"),
        projection_quality=freeze_json(
            {
                "sync_source": "corrected",
                "bearing_deg": 5.0,
                "distance_m": 8.0,
                "finite_vertices": 4,
                "projected_area_px": 1_200.0,
                "bbox_px": [10.0, 10.0, 70.0, 40.0],
                "visible_fraction": 0.95,
            }
        ),
    )


def _item(
    number: int,
    *,
    scope_status: str = "in_route_scope",
    shared: tuple[str, ...] = (),
    allowed: tuple[str, ...] = ("occupied", "free", "unknown"),
) -> QueueItem:
    return QueueItem(
        task_id=f"task-{number}",
        slot_id=f"slot-{number}",
        scope_status=scope_status,
        state="unknown",
        agent_observable=True,
        unknown_reasons=("weak_vehicle_evidence", "unresolved_occlusion"),
        priority="normal",
        available_modalities=("lidar", "rgb"),
        suggested_tools=("inspect_lidar_map", "inspect_rgb_frame"),
        allowed_final_states=allowed,
        occupied_evidence=freeze_json({}),
        free_evidence=freeze_json({}),
        audit=freeze_json({}),
        relationships=RelationshipRef((), (), shared),
        encounter=EncounterRef(
            encounter_id="encounter-a",
            part1_trace_event_ids=(),
            start_lidar_frame=1,
            anchor_frame=2,
            end_lidar_frame=3,
            start_timestamp=1.0,
            anchor_timestamp=2.0,
            end_timestamp=3.0,
            support_frames=(1, 2),
            pointcloud_resource_ids=(f"lidar-{number}",),
            visual_frames=(_visual(number),),
        ),
    )


def _world(root: Path, count: int = 1, *, clique: bool = False) -> tuple[QueueEnvelope, EvidenceCatalog]:
    items: list[QueueItem] = []
    resources: dict[str, ResourceRef] = {}
    for number in range(1, count + 1):
        lidar_content = f"lidar-{number}".encode()
        rgb_content = f"rgb-{number}".encode()
        lidar_path = root / f"lidar-{number}.bin"
        rgb_path = root / f"rgb-{number}.jpg"
        lidar_path.write_bytes(lidar_content)
        rgb_path.write_bytes(rgb_content)
        resources[f"lidar-{number}"] = _resource(
            f"lidar-{number}", "lidar_map", lidar_path.name, lidar_content
        )
        resources[f"rgb-{number}"] = _resource(
            f"rgb-{number}", "rgb_frame", rgb_path.name, rgb_content
        )
        items.append(_item(number, shared=("shared-all",) if clique else ()))
    queue = QueueEnvelope(
        schema_version="unknown-agent-queue/1.0",
        queue_id="queue-a",
        producer=freeze_json({"dataset_id": "route-a"}),
        resources=freeze_json(resources),
        tool_registry_version="part2-tools/1.0",
        items=tuple(items),
    )
    return queue, EvidenceCatalog(queue, base_dir=root)


def _evidence_id(catalog: EvidenceCatalog, task_id: str, tool_name: str) -> str:
    return next(
        evidence_id
        for evidence_id, record in catalog.entries.items()
        if record.task_id == task_id and record.tool_name == tool_name
    )


def _assessment(
    item: QueueItem,
    state: str,
    evidence_refs: tuple[str, ...] = (),
    **overrides: object,
) -> SlotAssessment:
    defaults: dict[str, object]
    if state == "occupied":
        defaults = {
            "target_visibility": "clear_full",
            "target_ownership": "target",
            "semantic_finding": "vehicle_or_occupying_object",
            "resolved_unknown_reasons": (),
            "unresolved_blockers": (),
            "reason_codes": ("occupying_object_observed",),
        }
    elif state == "free":
        defaults = {
            "target_visibility": "clear_full",
            "target_ownership": "target",
            "semantic_finding": "empty",
            "resolved_unknown_reasons": item.unknown_reasons,
            "unresolved_blockers": (),
            "reason_codes": ("full_clear_frame",),
        }
    else:
        defaults = {
            "target_visibility": "unknown",
            "target_ownership": "uncertain",
            "semantic_finding": "unclear",
            "resolved_unknown_reasons": (),
            "unresolved_blockers": item.unknown_reasons,
            "reason_codes": ("insufficient_evidence",),
        }
    defaults.update(overrides)
    return SlotAssessment(
        task_id=item.task_id,
        slot_id=item.slot_id,
        proposed_state=state,
        target_visibility=str(defaults["target_visibility"]),
        target_ownership=str(defaults["target_ownership"]),
        semantic_finding=str(defaults["semantic_finding"]),
        resolved_unknown_reasons=tuple(defaults["resolved_unknown_reasons"]),
        unresolved_blockers=tuple(defaults["unresolved_blockers"]),
        evidence_refs=evidence_refs,
        reason_codes=tuple(defaults["reason_codes"]),
    )


def _attempt(
    catalog: EvidenceCatalog,
    evidence_id: str,
    index: int = 1,
    *,
    disposition: str = "executed",
    status: str = "ok",
) -> ToolAttemptRecord:
    record = catalog.get(evidence_id)
    executed = disposition == "executed"
    return ToolAttemptRecord(
        attempt_id=f"attempt-{index}",
        attempt_index=index,
        model_turn=index,
        tool_name=record.tool_name,
        arguments={"evidence_id": evidence_id},
        evidence_id=evidence_id,
        disposition=disposition,
        executed=executed,
        result_status=status,
        error_code=None if status == "ok" else "test_failure",
        error_message=None,
        data={},
    )


def _run_result(
    group_id: str,
    assessments: tuple[SlotAssessment, ...],
    attempts: tuple[ToolAttemptRecord, ...] = (),
) -> GroupRunResult:
    ledger = EvidenceLedger(group_id)
    return GroupRunResult(
        group_id=group_id,
        proposal=GroupFinalProposal(group_id, assessments),
        model_turns=1,
        tool_attempts=len(attempts),
        stop_reason="final_proposal",
        trace_span_id=ledger.trace_span_id,
        ledger=ledger,
        tool_attempt_records=attempts,
        tool_results=(),
    )


class DecisionGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.queue, self.catalog = _world(self.root)
        self.item = self.queue.items[0]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _gate(self, assessment: SlotAssessment, attempts: tuple[ToolAttemptRecord, ...]):
        result = _run_result("group-a", (assessment,), attempts)
        return gate_assessment(self.item, assessment, result, self.catalog)

    def test_target_owned_lidar_can_resolve_occupied(self) -> None:
        evidence_id = _evidence_id(self.catalog, self.item.task_id, "inspect_lidar_map")
        gated = self._gate(
            _assessment(self.item, "occupied", (evidence_id,)),
            (_attempt(self.catalog, evidence_id),),
        )
        self.assertEqual(gated.state, "occupied")
        self.assertEqual(gated.evidence_refs, (evidence_id,))

    def test_one_clear_capable_rgb_frame_can_resolve_occupied(self) -> None:
        evidence_id = _evidence_id(self.catalog, self.item.task_id, "inspect_rgb_frame")
        gated = self._gate(
            _assessment(self.item, "occupied", (evidence_id,)),
            (_attempt(self.catalog, evidence_id),),
        )
        self.assertEqual(gated.state, "occupied")

    def test_rgb_sequence_is_context_only_for_terminal_occupied(self) -> None:
        evidence_id = _evidence_id(self.catalog, self.item.task_id, "inspect_rgb_sequence")
        gated = self._gate(
            _assessment(self.item, "occupied", (evidence_id,)),
            (_attempt(self.catalog, evidence_id),),
        )
        self.assertEqual(gated.state, "unknown")
        self.assertIn("occupied_terminal_support_missing", gated.reason_codes)

    def test_rgb_sequence_is_context_only_for_terminal_free(self) -> None:
        evidence_id = _evidence_id(self.catalog, self.item.task_id, "inspect_rgb_sequence")
        gated = self._gate(
            _assessment(self.item, "free", (evidence_id,)),
            (_attempt(self.catalog, evidence_id),),
        )
        self.assertEqual(gated.state, "unknown")
        self.assertIn("free_terminal_support_missing", gated.reason_codes)

    def test_one_full_clear_capable_rgb_frame_can_resolve_free(self) -> None:
        evidence_id = _evidence_id(self.catalog, self.item.task_id, "inspect_rgb_frame")
        gated = self._gate(
            _assessment(self.item, "free", (evidence_id,)),
            (_attempt(self.catalog, evidence_id),),
        )
        self.assertEqual(gated.state, "free")

    def test_free_vetoes_unresolved_blocker(self) -> None:
        evidence_id = _evidence_id(self.catalog, self.item.task_id, "inspect_rgb_frame")
        gated = self._gate(
            _assessment(
                self.item,
                "free",
                (evidence_id,),
                unresolved_blockers=("occluded_corner",),
            ),
            (_attempt(self.catalog, evidence_id),),
        )
        self.assertEqual(gated.state, "unknown")
        self.assertIn("free_unresolved_blockers", gated.reason_codes)

    def test_free_requires_exact_original_unknown_reason_resolution(self) -> None:
        evidence_id = _evidence_id(self.catalog, self.item.task_id, "inspect_rgb_frame")
        assessment = _assessment(
            self.item,
            "free",
            (evidence_id,),
            resolved_unknown_reasons=(self.item.unknown_reasons[0], "invented_reason"),
        )
        gated = self._gate(assessment, (_attempt(self.catalog, evidence_id),))
        self.assertEqual(gated.state, "unknown")
        self.assertIn("free_unknown_reasons_not_exactly_resolved", gated.reason_codes)

    def test_free_requires_lidar_no_contradiction_after_successful_lidar_attempt(self) -> None:
        rgb_id = _evidence_id(self.catalog, self.item.task_id, "inspect_rgb_frame")
        lidar_id = _evidence_id(self.catalog, self.item.task_id, "inspect_lidar_map")
        attempts = (
            _attempt(self.catalog, rgb_id, 1),
            _attempt(self.catalog, lidar_id, 2),
        )
        vetoed = self._gate(_assessment(self.item, "free", (rgb_id,)), attempts)
        accepted = self._gate(
            _assessment(
                self.item,
                "free",
                (rgb_id,),
                reason_codes=("full_clear_frame", "lidar_no_contradiction"),
            ),
            attempts,
        )
        self.assertEqual(vetoed.state, "unknown")
        self.assertIn("free_lidar_no_contradiction_missing", vetoed.reason_codes)
        self.assertEqual(accepted.state, "free")

    def test_failed_duplicate_and_unattempted_refs_never_support_terminal_state(self) -> None:
        evidence_id = _evidence_id(self.catalog, self.item.task_id, "inspect_rgb_frame")
        cases = {
            "failed": (_attempt(self.catalog, evidence_id, status="failed"),),
            "duplicate": (
                _attempt(
                    self.catalog,
                    evidence_id,
                    disposition="duplicate",
                    status="duplicate",
                ),
            ),
            "unattempted": (),
        }
        for name, attempts in cases.items():
            with self.subTest(name=name):
                gated = self._gate(_assessment(self.item, "occupied", (evidence_id,)), attempts)
                self.assertEqual(gated.state, "unknown")
                self.assertIn("terminal_evidence_unusable", gated.reason_codes)

    def test_one_usable_and_one_unusable_ref_still_vetoes_terminal_state(self) -> None:
        lidar_id = _evidence_id(self.catalog, self.item.task_id, "inspect_lidar_map")
        rgb_id = _evidence_id(self.catalog, self.item.task_id, "inspect_rgb_frame")
        assessment = _assessment(self.item, "occupied", (lidar_id, rgb_id))
        gated = self._gate(
            assessment,
            (
                _attempt(self.catalog, lidar_id, 1),
                _attempt(self.catalog, rgb_id, 2, status="failed"),
            ),
        )
        self.assertEqual(gated.state, "unknown")
        self.assertIn("terminal_evidence_unusable", gated.reason_codes)

    def test_forged_ok_attempt_with_wrong_arguments_is_unusable(self) -> None:
        evidence_id = _evidence_id(self.catalog, self.item.task_id, "inspect_lidar_map")
        forged = replace(_attempt(self.catalog, evidence_id), arguments={})
        gated = self._gate(
            _assessment(self.item, "occupied", (evidence_id,)),
            (forged,),
        )
        self.assertEqual(gated.state, "unknown")
        self.assertIn("terminal_evidence_unusable", gated.reason_codes)

    def test_terminal_state_must_be_allowed_for_the_task(self) -> None:
        evidence_id = _evidence_id(self.catalog, self.item.task_id, "inspect_lidar_map")
        restricted = replace(self.item, allowed_final_states=("free", "unknown"))
        assessment = _assessment(restricted, "occupied", (evidence_id,))
        gated = gate_assessment(
            restricted,
            assessment,
            _run_result("group-a", (assessment,), (_attempt(self.catalog, evidence_id),)),
            self.catalog,
        )
        self.assertEqual(gated.state, "unknown")
        self.assertIn("state_not_allowed", gated.reason_codes)

    def test_non_target_semantics_and_ownership_never_resolve_terminal(self) -> None:
        evidence_id = _evidence_id(self.catalog, self.item.task_id, "inspect_rgb_frame")
        cases = (
            {"semantic_finding": "static_structure"},
            {"semantic_finding": "lane_object"},
            {"target_ownership": "adjacent"},
            {"target_ownership": "shared"},
            {"target_ownership": "uncertain"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                gated = self._gate(
                    _assessment(self.item, "occupied", (evidence_id,), **overrides),
                    (_attempt(self.catalog, evidence_id),),
                )
                self.assertEqual(gated.state, "unknown")

    def test_cross_task_reference_never_supports_terminal_state(self) -> None:
        queue, catalog = _world(self.root, count=2)
        first, second = queue.items
        other_id = _evidence_id(catalog, second.task_id, "inspect_rgb_frame")
        assessment = _assessment(first, "occupied", (other_id,))
        gated = gate_assessment(
            first,
            assessment,
            _run_result("group-a", (assessment,), (_attempt(catalog, other_id),)),
            catalog,
        )
        self.assertEqual(gated.state, "unknown")
        self.assertIn("terminal_evidence_cross_task", gated.reason_codes)

    def test_partial_route_scope_uses_the_same_terminal_gate(self) -> None:
        partial = replace(self.item, scope_status="partial_route_scope")
        queue = replace(self.queue, items=(partial,))
        catalog = EvidenceCatalog(queue, base_dir=self.root)
        evidence_id = _evidence_id(catalog, partial.task_id, "inspect_lidar_map")
        assessment = _assessment(partial, "occupied", (evidence_id,))
        gated = gate_assessment(
            partial,
            assessment,
            _run_result("group-a", (assessment,), (_attempt(catalog, evidence_id),)),
            catalog,
        )
        self.assertEqual(gated.state, "occupied")

    def test_unknown_is_safe_even_when_not_listed_as_allowed(self) -> None:
        item = replace(self.item, allowed_final_states=("occupied", "free"))
        assessment = _assessment(item, "unknown")
        gated = gate_assessment(
            item,
            assessment,
            _run_result("group-a", (assessment,)),
            self.catalog,
        )
        self.assertEqual(gated.state, "unknown")


class ConsolidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.queue, self.catalog = _world(self.root, count=5, clique=True)
        self.groups = build_groups(self.queue)
        occurrences: dict[str, int] = {}
        for group in self.groups:
            for task_id in group.task_ids:
                occurrences[task_id] = occurrences.get(task_id, 0) + 1
        self.repeated_task_id = next(task_id for task_id, count in occurrences.items() if count > 1)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _results(self, repeated_states: tuple[str, ...]) -> tuple[GroupRunResult, ...]:
        item_by_task = {item.task_id: item for item in self.queue.items}
        state_index = 0
        results: list[GroupRunResult] = []
        for group in self.groups:
            assessments: list[SlotAssessment] = []
            attempts: list[ToolAttemptRecord] = []
            for task_id in group.task_ids:
                item = item_by_task[task_id]
                if task_id == self.repeated_task_id:
                    state = repeated_states[state_index]
                    state_index += 1
                    if state == "occupied":
                        evidence_id = _evidence_id(self.catalog, task_id, "inspect_lidar_map")
                        assessments.append(_assessment(item, state, (evidence_id,)))
                        attempts.append(_attempt(self.catalog, evidence_id, len(attempts) + 1))
                    elif state == "free":
                        evidence_id = _evidence_id(self.catalog, task_id, "inspect_rgb_frame")
                        assessments.append(_assessment(item, state, (evidence_id,)))
                        attempts.append(_attempt(self.catalog, evidence_id, len(attempts) + 1))
                    else:
                        assessments.append(_assessment(item, "unknown"))
                else:
                    assessments.append(_assessment(item, "unknown"))
            results.append(_run_result(group.group_id, tuple(assessments), tuple(attempts)))
        return tuple(results)

    def test_overlap_same_terminal_state_stays_terminal(self) -> None:
        count = sum(self.repeated_task_id in group.task_ids for group in self.groups)
        resolutions = consolidate_group_assessments(
            self.queue,
            self._results(("occupied",) * count),
            self.catalog,
        )
        repeated = next(row for row in resolutions if row.task_id == self.repeated_task_id)
        self.assertEqual(repeated.state, "occupied")
        self.assertEqual(repeated.input_state, "unknown")
        self.assertEqual(repeated.decision_source, "part2_agent_validated")
        self.assertEqual(repeated.completion_status, "resolved")
        self.assertEqual(repeated.model_turns, count)
        self.assertEqual(repeated.tool_calls_attempted, count)
        self.assertEqual(repeated.stop_reasons, ("final_proposal",))
        self.assertEqual(len(resolutions), len(self.queue.items))
        self.assertEqual(tuple(row.task_id for row in resolutions), tuple(sorted(row.task_id for row in resolutions)))

    def test_overlap_terminal_and_unknown_degrades_unknown(self) -> None:
        count = sum(self.repeated_task_id in group.task_ids for group in self.groups)
        states = ("occupied",) + ("unknown",) * (count - 1)
        resolutions = consolidate_group_assessments(
            self.queue,
            self._results(states),
            self.catalog,
        )
        repeated = next(row for row in resolutions if row.task_id == self.repeated_task_id)
        self.assertEqual(repeated.state, "unknown")
        self.assertIn("overlap_disagreement", repeated.reason_codes)

    def test_overlap_occupied_and_free_degrades_unknown(self) -> None:
        count = sum(self.repeated_task_id in group.task_ids for group in self.groups)
        states = ("occupied", "free") + ("occupied",) * (count - 2)
        resolutions = consolidate_group_assessments(
            self.queue,
            self._results(states),
            self.catalog,
        )
        repeated = next(row for row in resolutions if row.task_id == self.repeated_task_id)
        self.assertEqual(repeated.state, "unknown")
        self.assertIn("overlap_disagreement", repeated.reason_codes)

    def test_missing_overlap_occurrence_degrades_unknown(self) -> None:
        count = sum(self.repeated_task_id in group.task_ids for group in self.groups)
        results = list(self._results(("occupied",) * count))
        results.pop(next(index for index, row in enumerate(results) if self.repeated_task_id in self.groups[index].task_ids))
        resolutions = consolidate_group_assessments(self.queue, tuple(results), self.catalog)
        repeated = next(row for row in resolutions if row.task_id == self.repeated_task_id)
        self.assertEqual(repeated.state, "unknown")
        self.assertIn("overlap_result_missing", repeated.reason_codes)

    def test_extra_and_duplicate_group_results_are_rejected(self) -> None:
        count = sum(self.repeated_task_id in group.task_ids for group in self.groups)
        results = self._results(("occupied",) * count)
        extra = _run_result("extra-group", ())
        with self.assertRaisesRegex(ValueError, "extra group"):
            consolidate_group_assessments(self.queue, results + (extra,), self.catalog)
        with self.assertRaisesRegex(ValueError, "duplicate group"):
            consolidate_group_assessments(self.queue, results + (results[0],), self.catalog)

    def test_malformed_group_occurrences_are_checked_against_expected_tasks(self) -> None:
        count = sum(self.repeated_task_id in group.task_ids for group in self.groups)
        results = list(self._results(("occupied",) * count))
        first = results[0]
        missing_task_id = first.assessments[0].task_id
        results[0] = _run_result(first.group_id, first.assessments[1:], first.tool_attempt_records)
        resolutions = consolidate_group_assessments(self.queue, tuple(results), self.catalog)
        missing = next(row for row in resolutions if row.task_id == missing_task_id)
        self.assertEqual(missing.state, "unknown")
        self.assertIn("overlap_result_missing", missing.reason_codes)

        unexpected = replace(first.assessments[0], task_id="unexpected-task")
        malformed = _run_result(first.group_id, first.assessments + (unexpected,))
        with self.assertRaisesRegex(ValueError, "extra assessment"):
            consolidate_group_assessments(
                self.queue,
                (malformed,) + tuple(results[1:]),
                self.catalog,
            )


class BaseMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.queue, _ = _world(self.root)
        self.item = self.queue.items[0]
        self.resolution = SlotResolution(
            task_id=self.item.task_id,
            slot_id=self.item.slot_id,
            scope_status=self.item.scope_status,
            input_state="unknown",
            state="occupied",
            decision_source="part2_agent_validated",
            model_turns=2,
            tool_calls_attempted=1,
            stop_reasons=("final_proposal",),
            completion_status="resolved",
            group_ids=("group-a",),
            trace_span_ids=("span-a",),
            attempt_ids=("attempt-a",),
            evidence_refs=("ev_" + "a" * 64,),
            reason_codes=("gate_occupied",),
        )
        self.terminal = {
            "slot_id": "terminal-slot",
            "scope_status": "in_route_scope",
            "state": "occupied",
            "decision_reason": "part1_terminal",
            "extension": {"keep": [1, 2]},
        }
        self.nonqueued = {
            "slot_id": "nonqueued-unknown",
            "scope_status": "partial_route_scope",
            "state": "unknown",
            "unknown_reasons": ["not_agent_observable"],
        }
        self.base = {
            "schema_version": "1.0",
            "pipeline": "slot_hybrid_3d_pose_corrected_v1",
            "phase": "full",
            "decisions": [
                {
                    "slot_id": self.item.slot_id,
                    "scope_status": self.item.scope_status,
                    "state": "unknown",
                    "decision_reason": "ambiguous",
                    "extension": {"preserve": True},
                },
                self.terminal,
                self.nonqueued,
                {
                    "slot_id": "outside-slot",
                    "scope_status": "out_of_route_scope",
                    "state": "free",
                    "extension": "excluded",
                },
            ],
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_merge_changes_only_queued_unknown_and_preserves_extensions(self) -> None:
        merged = merge_final_route_states(
            self.base,
            self.queue,
            (self.resolution,),
            part2_run_id="sha256:" + "b" * 64,
        )
        rows = {row["slot_id"]: row for row in merged["decisions"]}
        self.assertEqual(rows[self.item.slot_id]["state"], "occupied")
        self.assertEqual(merged["queue_id"], self.queue.queue_id)
        self.assertEqual(rows[self.item.slot_id]["extension"], {"preserve": True})
        self.assertEqual(rows["terminal-slot"], self.terminal)
        self.assertEqual(rows["nonqueued-unknown"], self.nonqueued)
        self.assertNotIn("outside-slot", rows)
        self.assertEqual(rows[self.item.slot_id]["part2_resolution"]["task_id"], self.item.task_id)
        self.assertEqual(
            rows[self.item.slot_id]["part2_resolution"]["decision_source"],
            "part2_agent_validated",
        )

    def test_resolution_set_must_match_queue_exactly(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing resolution"):
            merge_final_route_states(self.base, self.queue, (), part2_run_id="run")
        extra = replace(self.resolution, task_id="extra-task", slot_id="extra-slot")
        with self.assertRaisesRegex(ValueError, "extra resolution"):
            merge_final_route_states(
                self.base,
                self.queue,
                (self.resolution, extra),
                part2_run_id="run",
            )

    def test_base_rows_are_unique_and_queue_rows_are_matching_unknowns(self) -> None:
        duplicate = {**self.base, "decisions": self.base["decisions"] + [self.base["decisions"][0]]}
        with self.assertRaisesRegex(ValueError, "duplicate base slot"):
            merge_final_route_states(duplicate, self.queue, (self.resolution,), part2_run_id="run")
        terminalized = {
            **self.base,
            "decisions": [
                {**row, "state": "free"} if row["slot_id"] == self.item.slot_id else row
                for row in self.base["decisions"]
            ],
        }
        with self.assertRaisesRegex(ValueError, "base unknown"):
            merge_final_route_states(terminalized, self.queue, (self.resolution,), part2_run_id="run")

    def test_base_envelope_is_strict_and_versioned(self) -> None:
        with self.assertRaisesRegex(ValueError, "base decisions envelope"):
            merge_final_route_states(
                {**self.base, "slots": []},
                self.queue,
                (self.resolution,),
                part2_run_id="run",
            )
        with self.assertRaisesRegex(ValueError, "schema_version"):
            merge_final_route_states(
                {**self.base, "schema_version": "2.0"},
                self.queue,
                (self.resolution,),
                part2_run_id="run",
            )


def _write_canonical_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _resource_payload(kind: str, uri: str, sha256: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": kind,
        "uri": uri,
        "dataset_id": "route-a",
        "config_hash": HASH,
        "slot_map_hash": HASH,
    }
    if kind == "map_points_root":
        payload["manifest_hash"] = HASH
    else:
        payload["sha256"] = sha256
    return payload


def _cli_fixture(root: Path, *, empty: bool = False) -> tuple[Path, Path, Path, QueueEnvelope]:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    producer = {
        "pipeline": "slot_hybrid_3d_pose_corrected_v1",
        "run_id": "sha256:" + "b" * 64,
        "config_hash": HASH,
        "dataset_id": "route-a",
        "slot_map_hash": HASH,
    }
    base_payload = {
        "schema_version": "1.0",
        "pipeline": producer["pipeline"],
        "phase": "full",
        "decisions": (
            [
                {
                    "slot_id": "nonqueued-unknown",
                    "scope_status": "in_route_scope",
                    "state": "unknown",
                    "decision_reason": "not_agent_observable",
                    "extension": {"preserved": True},
                }
            ]
            if empty
            else [
                {
                    "slot_id": "slot-1",
                    "scope_status": "in_route_scope",
                    "state": "unknown",
                    "decision_reason": "ambiguous",
                    "unknown_reasons": ["weak_vehicle_evidence", "unresolved_occlusion"],
                    "extension": {"preserved": True},
                },
                {
                    "slot_id": "terminal-slot",
                    "scope_status": "partial_route_scope",
                    "state": "free",
                    "decision_reason": "part1_terminal",
                    "extension": ["unchanged"],
                },
            ]
        ),
    }
    files: dict[str, bytes] = {
        "slot_database.json": b"{}\n",
        "corrected_frames.csv": b"frame_id\n1\n",
        "camera_calibration.json": b"{}\n",
        "slot_decisions.json": canonical_json_bytes(base_payload) + b"\n",
    }
    if not empty:
        files.update(
            {
                "lidar.bin": b"lidar-evidence",
                "rgb.jpg": b"P6\n80 50\n255\n" + bytes((40, 80, 120)) * (80 * 50),
            }
        )
    for name, content in files.items():
        (artifacts / name).write_bytes(content)

    resources: dict[str, dict[str, object]] = {
        "slot-database": _resource_payload(
            "slot_database",
            "artifacts/slot_database.json",
            _content_hash(files["slot_database.json"]),
        ),
        "corrected-frames": _resource_payload(
            "corrected_frames",
            "artifacts/corrected_frames.csv",
            _content_hash(files["corrected_frames.csv"]),
        ),
        "map-points-root": _resource_payload("map_points_root", "artifacts/map_points"),
        "camera-calibration": _resource_payload(
            "camera_calibration",
            "artifacts/camera_calibration.json",
            _content_hash(files["camera_calibration.json"]),
        ),
        "base-decisions": _resource_payload(
            "base_slot_decisions",
            "artifacts/slot_decisions.json",
            _content_hash(files["slot_decisions.json"]),
        ),
    }
    items: list[dict[str, object]] = []
    if not empty:
        resources["lidar-1"] = _resource_payload(
            "lidar_map", "artifacts/lidar.bin", _content_hash(files["lidar.bin"])
        )
        resources["rgb-1"] = _resource_payload(
            "rgb_frame", "artifacts/rgb.jpg", _content_hash(files["rgb.jpg"])
        )
        task_id = make_task_id(producer, "slot-1", "encounter-1")
        items.append(
            {
                "task_id": task_id,
                "slot_id": "slot-1",
                "scope_status": "in_route_scope",
                "state": "unknown",
                "agent_observable": True,
                "unknown_reasons": ["weak_vehicle_evidence", "unresolved_occlusion"],
                "priority": "normal",
                "available_modalities": ["lidar", "rgb"],
                "suggested_tools": ["inspect_lidar_map", "inspect_rgb_frame"],
                "allowed_final_states": ["occupied", "free", "unknown"],
                "occupied_evidence": {},
                "free_evidence": {},
                "audit": {},
                "relationships": {
                    "adjacent_slot_ids": [],
                    "conflict_slot_ids": [],
                    "shared_evidence_ids": [],
                },
                "encounter": {
                    "encounter_id": "encounter-1",
                    "part1_trace_event_ids": [],
                    "start_lidar_frame": 1,
                    "anchor_frame": 2,
                    "end_lidar_frame": 3,
                    "start_timestamp": 1.0,
                    "anchor_timestamp": 2.0,
                    "end_timestamp": 3.0,
                    "support_frames": [1, 2],
                    "pointcloud_resource_ids": ["lidar-1"],
                    "visual_frames": [
                        {
                            "visual_frame_id": "visual-1",
                            "lidar_frame": 2,
                            "camera_frame": 102,
                            "lidar_timestamp": 2.0,
                            "camera_timestamp": 1.99,
                            "camera_lidar_dt_sec": -0.01,
                            "image_resource_id": "rgb-1",
                            "capabilities": ["can_assess_occupied", "can_assess_free"],
                            "projection_quality": {
                                "adjacent_polygons_uv": {},
                                "sync_source": "corrected",
                                "bearing_deg": 5.0,
                                "distance_m": 8.0,
                                "finite_vertices": 4,
                                "polygon_uv": [
                                    [10.0, 10.0],
                                    [70.0, 10.0],
                                    [70.0, 40.0],
                                    [10.0, 40.0],
                                ],
                                "projected_area_px": 1200.0,
                                "bbox_px": [10.0, 10.0, 70.0, 40.0],
                                "visible_fraction": 0.95,
                            },
                        }
                    ],
                },
            }
        )
    queue_payload: dict[str, object] = {
        "schema_version": "unknown-agent-queue/1.0",
        "producer": producer,
        "resources": resources,
        "tool_registry_version": "part2-tools/1.0",
        "items": items,
    }
    queue_payload["queue_id"] = make_queue_id(queue_payload)
    queue = validate_queue(queue_payload, base_dir=root)
    queue_path = root / "unknown_agent_queue.json"
    _write_canonical_json(queue_path, queue_payload)

    actions: dict[str, list[object]] = {}
    if not empty:
        group = build_groups(queue)[0]
        catalog = EvidenceCatalog(queue, base_dir=root)
        evidence_id = _evidence_id(catalog, queue.items[0].task_id, "inspect_rgb_frame")
        actions[group.group_id] = [
            {
                "type": "tool_request",
                "tool_name": "inspect_rgb_frame",
                "arguments": {"evidence_id": evidence_id},
            },
            {
                "type": "final_proposal",
                "assessments": [
                    {
                        "task_id": queue.items[0].task_id,
                        "slot_id": queue.items[0].slot_id,
                        "proposed_state": "occupied",
                        "target_visibility": "clear_full",
                        "target_ownership": "target",
                        "semantic_finding": "vehicle_or_occupying_object",
                        "resolved_unknown_reasons": [],
                        "unresolved_blockers": [],
                        "evidence_refs": [evidence_id],
                        "reason_codes": ["occupying_object_observed"],
                    }
                ],
            },
        ]
    replay_payload = {"schema_version": "part2-replay-actions/1.0", "actions": actions}
    replay_path = root / "replay_actions.json"
    _write_canonical_json(replay_path, replay_payload)
    return queue_path, replay_path, artifacts / "slot_decisions.json", queue


def _run_cli(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *(str(arg) for arg in args)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tree_snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): None if path.is_dir() else path.read_bytes()
        for path in sorted(root.rglob("*"))
    }


def _writer_inputs(root: Path) -> dict[str, object]:
    root.mkdir(parents=True)
    queue, _ = _world(root)
    item = queue.items[0]
    base_payload = {
        "schema_version": "1.0",
        "pipeline": "slot_hybrid_3d_pose_corrected_v1",
        "phase": "full",
        "decisions": [
            {
                "slot_id": item.slot_id,
                "scope_status": item.scope_status,
                "state": "unknown",
                "decision_reason": "ambiguous",
            },
            {
                "slot_id": "terminal-slot",
                "scope_status": "in_route_scope",
                "state": "occupied",
                "decision_reason": "part1_terminal",
                "extension": {"preserve": "terminal"},
            },
            {
                "slot_id": "nonqueued-unknown",
                "scope_status": "partial_route_scope",
                "state": "unknown",
                "decision_reason": "not_agent_observable",
                "extension": {"preserve": "unknown"},
            },
            {
                "slot_id": "outside-slot",
                "scope_status": "out_of_route_scope",
                "state": "free",
                "decision_reason": "outside_route",
            },
        ],
    }
    base_content = canonical_json_bytes(base_payload) + b"\n"
    base_path = root / "base_slot_decisions.json"
    base_path.write_bytes(base_content)
    resources = dict(queue.resources)
    resources["base-decisions"] = _resource(
        "base-decisions",
        "base_slot_decisions",
        base_path.name,
        base_content,
    )
    queue = replace(queue, resources=freeze_json(resources))
    catalog = EvidenceCatalog(queue, base_dir=root)
    group = build_groups(queue)[0]
    evidence_id = _evidence_id(catalog, item.task_id, "inspect_lidar_map")
    assessment = _assessment(item, "occupied", (evidence_id,))
    replay_payload = {
        "schema_version": "part2-replay-actions/1.0",
        "actions": {
            group.group_id: [
                {
                    "type": "tool_request",
                    "tool_name": "inspect_lidar_map",
                    "arguments": {"evidence_id": evidence_id},
                },
                {
                    "type": "final_proposal",
                    "assessments": [assessment.to_dict()],
                },
            ]
        },
    }
    # The writer fixtures intentionally cross the same orchestration boundary as
    # the CLI. Passive hand-built GroupRunResult objects remain limited to the
    # decision-gate/consolidation unit tests above.
    result = run_group(
        group,
        queue,
        ReplayModelAdapter(replay_payload),
        ToolRegistry(catalog),
        basic_context={},
    )
    resolutions = consolidate_group_assessments(queue, (result,), catalog)
    part2_run_id = derive_part2_run_id(queue, replay_payload)
    final = merge_final_route_states(
        base_payload,
        queue,
        resolutions,
        part2_run_id=part2_run_id,
    )
    return {
        "part2_run_id": part2_run_id,
        "queue": queue,
        "catalog": catalog,
        "base_dir": root,
        "replay_identity": canonical_sha256(replay_payload),
        "group_results": (result,),
        "resolutions": resolutions,
        "final_route_states": final,
    }


def _rerun_writer_inputs(
    inputs: dict[str, object],
    actions: list[object],
) -> GroupRunResult:
    queue = inputs["queue"]
    catalog = inputs["catalog"]
    group = build_groups(queue)[0]
    replay_payload = {
        "schema_version": "part2-replay-actions/1.0",
        "actions": {group.group_id: actions},
    }
    result = run_group(
        group,
        queue,
        ReplayModelAdapter(replay_payload),
        ToolRegistry(catalog),
        basic_context={},
    )
    resolutions = consolidate_group_assessments(queue, (result,), catalog)
    part2_run_id = derive_part2_run_id(queue, replay_payload)
    base = load_base_slot_decisions(queue, base_dir=inputs["base_dir"])
    inputs.update(
        {
            "part2_run_id": part2_run_id,
            "replay_identity": canonical_sha256(replay_payload),
            "group_results": (result,),
            "resolutions": resolutions,
            "final_route_states": merge_final_route_states(
                base,
                queue,
                resolutions,
                part2_run_id=part2_run_id,
            ),
        }
    )
    return result


class ReportingAndCliTest(unittest.TestCase):
    def _assert_writer_rejects_without_mutation(
        self,
        root: Path,
        inputs: dict[str, object],
        *,
        message_pattern: str,
    ) -> None:
        output = root / "output"
        output.mkdir(parents=True)
        (output / "summary.json").write_bytes(b"sentinel-summary\n")
        before = _tree_snapshot(output)
        with self.assertRaisesRegex((TypeError, ValueError), message_pattern):
            write_run_outputs(output, **inputs)
        self.assertEqual(_tree_snapshot(output), before)

    def test_writer_rejects_unsealed_group_ledger_before_output_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _writer_inputs(root / "input")
            result = inputs["group_results"][0]
            result.ledger._sealed = False
            self._assert_writer_rejects_without_mutation(
                root,
                inputs,
                message_pattern="ledger.*sealed",
            )

    def test_writer_rejects_empty_group_ledger_before_output_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _writer_inputs(root / "input")
            result = inputs["group_results"][0]
            empty_ledger = EvidenceLedger(result.group_id)
            empty_ledger.seal()
            inputs["group_results"] = (
                replace(
                    result,
                    trace_span_id=empty_ledger.trace_span_id,
                    ledger=empty_ledger,
                ),
            )
            self._assert_writer_rejects_without_mutation(
                root,
                inputs,
                message_pattern="ledger.*empty|trace.*empty",
            )

    def test_writer_rejects_nonderived_attempt_id_before_output_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _writer_inputs(root / "input")
            result = inputs["group_results"][0]
            forged = replace(
                result.tool_attempt_records[0],
                attempt_id="attempt_" + "f" * 64,
            )
            inputs["group_results"] = (
                replace(result, tool_attempt_records=(forged,)),
            )
            self._assert_writer_rejects_without_mutation(
                root,
                inputs,
                message_pattern="attempt.*derived|attempt.*identity|attempt ID",
            )

    def test_writer_rejects_inconsistent_turn_attempt_and_budget_counters(self) -> None:
        variants = ("tool_attempt_count", "model_turn_count", "attempt_index", "attempt_turn")
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = _writer_inputs(root / "input")
                result = inputs["group_results"][0]
                attempt = result.tool_attempt_records[0]
                if variant == "tool_attempt_count":
                    forged_result = replace(result, tool_attempts=0)
                elif variant == "model_turn_count":
                    forged_result = replace(result, model_turns=1)
                elif variant == "attempt_index":
                    forged_result = replace(
                        result,
                        tool_attempt_records=(replace(attempt, attempt_index=2),),
                    )
                else:
                    forged_result = replace(
                        result,
                        tool_attempt_records=(replace(attempt, model_turn=3),),
                    )
                inputs["group_results"] = (forged_result,)
                self._assert_writer_rejects_without_mutation(
                    root,
                    inputs,
                    message_pattern="model turn|tool attempt|attempt index|budget|counter",
                )

    def test_writer_rejects_executed_attempt_without_matching_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _writer_inputs(root / "input")
            result = inputs["group_results"][0]
            inputs["group_results"] = (replace(result, tool_results=()),)
            self._assert_writer_rejects_without_mutation(
                root,
                inputs,
                message_pattern="executed.*tool result|tool result.*executed|one-to-one",
            )

    def test_writer_reexecutes_tools_and_rejects_self_consistent_forged_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _writer_inputs(root / "input")
            result = inputs["group_results"][0]
            forged_data = freeze_json({"forged": True})
            forged_attempt = replace(result.tool_attempt_records[0], data=forged_data)
            forged_tool_result = replace(result.tool_results[0], data=forged_data)
            forged_ledger = EvidenceLedger(result.group_id)
            for event in result.trace_events:
                data = dict(event["data"])
                if event["event_type"] == "tool_result":
                    data["data"] = forged_data
                forged_ledger.record(event["category"], event["event_type"], data)
            forged_ledger.seal()
            inputs["group_results"] = (
                replace(
                    result,
                    ledger=forged_ledger,
                    trace_span_id=forged_ledger.trace_span_id,
                    tool_attempt_records=(forged_attempt,),
                    tool_results=(forged_tool_result,),
                ),
            )
            self._assert_writer_rejects_without_mutation(
                root,
                inputs,
                message_pattern="re-executed|recomputed tool result|tool result.*registry",
            )

    def test_writer_rejects_forged_trace_with_forbidden_reasoning_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _writer_inputs(root / "input")
            result = inputs["group_results"][0]
            forged_ledger = EvidenceLedger(result.group_id)
            rows: list[tuple[str, str, dict[str, object]]] = []
            for event in result.trace_events:
                data = dict(event["data"])
                if event["event_type"] == "model_turn_started" and data["model_turn"] == 2:
                    data["observation_count"] = 2
                rows.append((event["category"], event["event_type"], data))
                if event["event_type"] == "tool_result":
                    rows.append(
                        (
                            "validation",
                            "validation_feedback",
                            {
                                "model_turn": 1,
                                "code": "forged_feedback",
                                "details": {"reasoning": "must never reach the trace"},
                            },
                        )
                    )
            forged_events = []
            for sequence, (category, event_type, data) in enumerate(rows, start=1):
                identity = {
                    "schema_version": "part2-evidence-ledger/1.0",
                    "trace_span_id": forged_ledger.trace_span_id,
                    "sequence": sequence,
                    "category": category,
                    "event_type": event_type,
                    "data": data,
                }
                forged_events.append(
                    freeze_json(
                        {
                            **identity,
                            "event_id": "event_"
                            + canonical_sha256(identity).split(":", 1)[1],
                        }
                    )
                )
            forged_ledger._events = forged_events
            forged_ledger.seal()
            inputs["group_results"] = (
                replace(
                    result,
                    ledger=forged_ledger,
                    trace_span_id=forged_ledger.trace_span_id,
                ),
            )
            self._assert_writer_rejects_without_mutation(
                root,
                inputs,
                message_pattern="forbidden trace|forbidden.*reasoning|trace payload",
            )

    def test_writer_requires_exact_lowercase_sha256_replay_identity(self) -> None:
        malformed = (
            "sha256:abc",
            "sha256:" + "A" * 64,
            "sha256:" + "a" * 65,
            "sha256:" + "a" * 64 + "\n",
        )
        for replay_identity in malformed:
            with self.subTest(replay_identity=replay_identity), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = _writer_inputs(root / "input")
                inputs["replay_identity"] = replay_identity
                self._assert_writer_rejects_without_mutation(
                    root,
                    inputs,
                    message_pattern="replay_identity.*sha256|canonical sha256",
                )

    def test_writer_binds_part2_run_id_to_replay_identity_and_policies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _writer_inputs(root / "input")
            inputs["replay_identity"] = "sha256:" + "c" * 64
            self._assert_writer_rejects_without_mutation(
                root,
                inputs,
                message_pattern="part2_run_id.*replay|run identity.*polic|expected run",
            )

    def test_writer_rejects_self_consistent_but_nonderived_part2_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _writer_inputs(root / "input")
            forged_run_id = "sha256:" + "d" * 64
            base = load_base_slot_decisions(inputs["queue"], base_dir=inputs["base_dir"])
            inputs["part2_run_id"] = forged_run_id
            inputs["final_route_states"] = merge_final_route_states(
                base,
                inputs["queue"],
                inputs["resolutions"],
                part2_run_id=forged_run_id,
            )
            self._assert_writer_rejects_without_mutation(
                root,
                inputs,
                message_pattern="part2_run_id.*derived|run identity.*polic|expected run",
            )

    def test_manifest_and_run_identity_bind_exact_code_owned_tool_policies(self) -> None:
        from parking_slot_part2 import preflight, queueing, reporting, tools

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _writer_inputs(root / "input")
            output = root / "output"
            manifest = write_run_outputs(output, **inputs)
            self.assertIs(tools.TOOL_NAMES, queueing.TOOL_NAMES)
            self.assertEqual(
                getattr(preflight, "PREFLIGHT_POLICY_VERSION", None),
                "part2-evidence-preflight/1.1",
            )
            self.assertEqual(
                getattr(tools, "TOOLS_POLICY_VERSION", None),
                "part2-tool-execution/2.0",
            )
            self.assertEqual(
                set(manifest["policies"]),
                {
                    "decision",
                    "grouping",
                    "model_turn_schema",
                    "orchestrator",
                    "orchestrator_limits",
                    "preflight",
                    "reporting",
                    "tool_registry",
                    "tools",
                    "trace",
                },
            )
            self.assertEqual(
                manifest["policies"]["preflight"],
                preflight.PREFLIGHT_POLICY_VERSION,
            )
            self.assertEqual(manifest["policies"]["tools"], tools.TOOLS_POLICY_VERSION)
            self.assertEqual(
                manifest["policies"]["tool_registry"],
                queueing.TOOL_REGISTRY_VERSION,
            )
            replay_payload = {
                "schema_version": "part2-replay-actions/1.0",
                "actions": {},
            }
            baseline = derive_part2_run_id(inputs["queue"], replay_payload)
            with mock.patch.object(
                reporting,
                "PREFLIGHT_POLICY_VERSION",
                "part2-evidence-preflight/next",
            ):
                changed = derive_part2_run_id(inputs["queue"], replay_payload)
            self.assertNotEqual(baseline, changed)

    def test_writer_accepts_real_invalid_duplicate_unavailable_and_failed_paths(self) -> None:
        variants = ("invalid", "duplicate", "unavailable", "failed")
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = _writer_inputs(root / "input")
                queue = inputs["queue"]
                catalog = inputs["catalog"]
                item = queue.items[0]
                evidence_id = _evidence_id(catalog, item.task_id, "inspect_lidar_map")
                request = {
                    "type": "tool_request",
                    "tool_name": "inspect_lidar_map",
                    "arguments": {"evidence_id": evidence_id},
                }
                final = {
                    "type": "final_proposal",
                    "assessments": [_assessment(item, "unknown").to_dict()],
                }
                if variant == "invalid":
                    actions = [
                        {
                            "type": "tool_request",
                            "tool_name": "inspect_lidar_map",
                            "arguments": {},
                        },
                        final,
                    ]
                elif variant == "duplicate":
                    actions = [request, request, final]
                elif variant == "unavailable":
                    (Path(inputs["base_dir"]) / "lidar-1.bin").unlink()
                    actions = [request, final]
                else:
                    actions = [request, final]
                registry_failure = (
                    mock.patch.object(
                        catalog,
                        "artifact_paths",
                        side_effect=OSError("deterministic test failure"),
                    )
                    if variant == "failed"
                    else nullcontext()
                )
                with registry_failure:
                    result = _rerun_writer_inputs(inputs, actions)
                    manifest = write_run_outputs(root / "output", **inputs)
                self.assertEqual(
                    len(manifest["artifacts"]),
                    5 + len(result.tool_attempt_records),
                )
                if variant == "invalid":
                    self.assertEqual(
                        tuple(row.disposition for row in result.tool_attempt_records),
                        ("prevalidation_rejected",),
                    )
                    self.assertEqual(result.tool_results, ())
                elif variant == "duplicate":
                    self.assertEqual(
                        tuple(row.disposition for row in result.tool_attempt_records),
                        ("executed", "duplicate"),
                    )
                    self.assertEqual(len(result.tool_results), 1)
                elif variant == "unavailable":
                    self.assertEqual(result.tool_attempt_records[0].result_status, "unavailable")
                    self.assertEqual(result.tool_results[0].status, "unavailable")
                else:
                    self.assertEqual(result.tool_attempt_records[0].result_status, "failed")
                    self.assertEqual(result.tool_results[0].status, "failed")

    def test_writer_rejects_resolution_set_inconsistency_before_output_mutation(self) -> None:
        variants = ("missing", "extra", "duplicate", "slot_mismatch", "scope_mismatch")
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = _writer_inputs(root / "input")
                resolution = inputs["resolutions"][0]
                if variant == "missing":
                    inputs["resolutions"] = ()
                elif variant == "extra":
                    inputs["resolutions"] = (
                        resolution,
                        replace(resolution, task_id="extra-task", slot_id="extra-slot"),
                    )
                elif variant == "duplicate":
                    inputs["resolutions"] = (resolution, resolution)
                elif variant == "slot_mismatch":
                    inputs["resolutions"] = (replace(resolution, slot_id="wrong-slot"),)
                else:
                    inputs["resolutions"] = (
                        replace(resolution, scope_status="partial_route_scope"),
                    )
                self._assert_writer_rejects_without_mutation(
                    root,
                    inputs,
                    message_pattern="resolution",
                )

    def test_writer_recomputes_gate_and_rejects_self_consistent_forged_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _writer_inputs(root / "input")
            resolution = inputs["resolutions"][0]
            forged = replace(
                resolution,
                state="free",
                decision_source="part2_agent_validated",
                completion_status="resolved",
            )
            final = dict(inputs["final_route_states"])
            rows = [dict(row) for row in final["decisions"]]
            rows[0]["state"] = "free"
            rows[0]["part2_resolution"] = forged.to_dict()
            final["decisions"] = rows
            inputs["resolutions"] = (forged,)
            inputs["final_route_states"] = final
            self._assert_writer_rejects_without_mutation(
                root,
                inputs,
                message_pattern="gate|consolidated|recomputed",
            )

    def test_writer_rejects_group_and_proposal_inconsistency_before_output_mutation(self) -> None:
        variants = (
            "missing_group",
            "extra_group",
            "duplicate_group",
            "missing_assessment",
            "extra_assessment",
            "duplicate_assessment",
        )
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = _writer_inputs(root / "input")
                result = inputs["group_results"][0]
                assessment = result.assessments[0]
                if variant == "missing_group":
                    inputs["group_results"] = ()
                elif variant == "extra_group":
                    extra = _run_result("extra-group", (assessment,))
                    inputs["group_results"] = (result, extra)
                elif variant == "duplicate_group":
                    inputs["group_results"] = (result, result)
                elif variant == "missing_assessment":
                    inputs["group_results"] = (
                        replace(result, proposal=GroupFinalProposal(result.group_id, ())),
                    )
                elif variant == "extra_assessment":
                    unexpected = replace(
                        assessment,
                        task_id="unexpected-task",
                        slot_id="unexpected-slot",
                    )
                    inputs["group_results"] = (
                        replace(
                            result,
                            proposal=GroupFinalProposal(
                                result.group_id,
                                (assessment, unexpected),
                            ),
                        ),
                    )
                else:
                    inputs["group_results"] = (
                        replace(
                            result,
                            proposal=GroupFinalProposal(
                                result.group_id,
                                (assessment, assessment),
                            ),
                        ),
                    )
                self._assert_writer_rejects_without_mutation(
                    root,
                    inputs,
                    message_pattern="group|assessment",
                )

    def test_writer_rejects_final_state_inconsistency_before_output_mutation(self) -> None:
        variants = (
            "schema",
            "run_id",
            "queue_id",
            "state",
            "audit",
            "missing_row",
            "duplicate_row",
            "extra_part2_row",
        )
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = _writer_inputs(root / "input")
                final = dict(inputs["final_route_states"])
                rows = [dict(row) for row in final["decisions"]]
                queued_slot_id = inputs["resolutions"][0].slot_id
                queued_row = next(row for row in rows if row["slot_id"] == queued_slot_id)
                if variant == "schema":
                    final["schema_version"] = "wrong-version"
                elif variant == "run_id":
                    final["part2_run_id"] = "sha256:" + "c" * 64
                elif variant == "queue_id":
                    final["queue_id"] = "wrong-queue"
                elif variant == "state":
                    queued_row["state"] = "free"
                elif variant == "audit":
                    queued_row["part2_resolution"] = {
                        **queued_row["part2_resolution"],
                        "state": "free",
                    }
                elif variant == "missing_row":
                    rows = []
                elif variant == "duplicate_row":
                    rows.append(dict(rows[0]))
                else:
                    rows.append(
                        {
                            "slot_id": "extra-slot",
                            "scope_status": "in_route_scope",
                            "state": "unknown",
                            "part2_resolution": {
                                **queued_row["part2_resolution"],
                                "task_id": "extra-task",
                                "slot_id": "extra-slot",
                                "state": "unknown",
                            },
                        }
                    )
                final["decisions"] = rows
                inputs["final_route_states"] = final
                self._assert_writer_rejects_without_mutation(
                    root,
                    inputs,
                    message_pattern="final|queued|decision|part2_resolution",
                )

    def test_writer_rebuilds_full_final_envelope_before_output_mutation(self) -> None:
        variants = (
            "tampered_terminal",
            "tampered_nonqueued_unknown",
            "injected_route_row",
            "injected_out_of_route_row",
        )
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = _writer_inputs(root / "input")
                final = dict(inputs["final_route_states"])
                rows = [dict(row) for row in final["decisions"]]
                if variant == "tampered_terminal":
                    row = next(row for row in rows if row["slot_id"] == "terminal-slot")
                    row["state"] = "free"
                elif variant == "tampered_nonqueued_unknown":
                    row = next(
                        row for row in rows if row["slot_id"] == "nonqueued-unknown"
                    )
                    row["extension"] = {"preserve": "tampered"}
                elif variant == "injected_route_row":
                    rows.append(
                        {
                            "slot_id": "injected-slot",
                            "scope_status": "in_route_scope",
                            "state": "occupied",
                            "decision_reason": "injected",
                        }
                    )
                else:
                    rows.append(
                        {
                            "slot_id": "outside-slot",
                            "scope_status": "out_of_route_scope",
                            "state": "free",
                            "decision_reason": "injected_outside",
                        }
                    )
                final["decisions"] = rows
                inputs["final_route_states"] = final
                self._assert_writer_rejects_without_mutation(
                    root,
                    inputs,
                    message_pattern="full final route states do not match freshly merged base",
                )

    def test_writer_prevalidates_attempt_ids_and_canonical_bytes_before_mutation(self) -> None:
        variants = ("duplicate_attempt", "invalid_attempt_json", "invalid_main_json")
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = _writer_inputs(root / "input")
                result = inputs["group_results"][0]
                attempt = result.tool_attempt_records[0]
                if variant == "duplicate_attempt":
                    duplicate = replace(attempt, attempt_index=2, model_turn=2)
                    inputs["group_results"] = (
                        replace(
                            result,
                            tool_attempts=2,
                            tool_attempt_records=(attempt, duplicate),
                        ),
                    )
                    pattern = "duplicate tool attempt"
                elif variant == "invalid_attempt_json":
                    invalid = replace(attempt, data={"invalid": object()})
                    inputs["group_results"] = (
                        replace(result, tool_attempt_records=(invalid,)),
                    )
                    pattern = "JSON-compatible|type object"
                else:
                    final = dict(inputs["final_route_states"])
                    final["base_decisions"] = {
                        **final["base_decisions"],
                        "invalid": object(),
                    }
                    inputs["final_route_states"] = final
                    pattern = "JSON-compatible|type object"
                self._assert_writer_rejects_without_mutation(
                    root,
                    inputs,
                    message_pattern=pattern,
                )

    def test_task4_public_contracts_are_exported_from_package(self) -> None:
        import parking_slot_part2 as part2

        for name in (
            "DECISION_POLICY_VERSION",
            "FINAL_ROUTE_STATES_SCHEMA_VERSION",
            "RESOLUTIONS_SCHEMA_VERSION",
            "GatedAssessment",
            "SlotResolution",
            "gate_assessment",
            "consolidate_group_assessments",
            "merge_final_route_states",
            "derive_part2_run_id",
            "load_base_slot_decisions",
            "write_run_outputs",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(part2, name), name)

    def test_base_resource_is_rehashed_immediately_before_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path, _, base_path, queue = _cli_fixture(root)
            self.assertEqual(load_queue(queue_path), queue)
            base_path.write_bytes(base_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "base.*sha256"):
                load_base_slot_decisions(queue, base_dir=queue_path.parent)

    def test_base_hash_and_json_parse_use_one_identical_byte_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path, _, _, queue = _cli_fixture(root)
            original_read_bytes = Path.read_bytes
            with mock.patch.object(
                Path,
                "read_bytes",
                autospec=True,
                side_effect=lambda path: original_read_bytes(path),
            ) as read_bytes:
                payload = load_base_slot_decisions(queue, base_dir=queue_path.parent)
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(read_bytes.call_count, 1)

    def test_run_identity_is_stable_and_policy_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, replay_path, _, queue = _cli_fixture(root)
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            first = derive_part2_run_id(queue, replay)
            second = derive_part2_run_id(queue, replay)
            changed = derive_part2_run_id(
                queue,
                {**replay, "actions": {**replay["actions"], "unused-group": []}},
            )
            self.assertEqual(first, second)
            self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")
            self.assertNotEqual(first, changed)

    def test_validate_queue_cli_prints_stable_summary_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path, _, _, queue = _cli_fixture(root)
            before = _tree_bytes(root)
            first = _run_cli("validate-queue", "--queue", queue_path)
            second = _run_cli("validate-queue", "--queue", queue_path)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(first.stdout, second.stdout)
            summary = json.loads(first.stdout)
            self.assertEqual(summary["queue_id"], queue.queue_id)
            self.assertEqual(summary["item_count"], 1)
            self.assertEqual(summary["group_count"], 1)
            self.assertEqual(_tree_bytes(root), before)

    def test_run_cli_writes_exact_path_free_auditable_output_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path, replay_path, _, _ = _cli_fixture(root / "input")
            output = root / "output"
            completed = _run_cli(
                "run",
                "--queue",
                queue_path,
                "--replay-actions",
                replay_path,
                "--output-dir",
                output,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "part2_resolutions.json",
                    "final_route_states.json",
                    "decision_trace.jsonl",
                    "summary.json",
                    "report.html",
                    "run_manifest.json",
                    "tool_artifacts",
                },
            )
            attempt_files = tuple((output / "tool_artifacts").glob("*.json"))
            self.assertEqual(len(attempt_files), 1)
            resolutions = json.loads((output / "part2_resolutions.json").read_text())
            self.assertEqual(resolutions["resolutions"][0]["state"], "occupied")
            self.assertEqual(resolutions["resolutions"][0]["input_state"], "unknown")
            final = json.loads((output / "final_route_states.json").read_text())
            rows = {row["slot_id"]: row for row in final["decisions"]}
            self.assertEqual(rows["slot-1"]["state"], "occupied")
            self.assertEqual(rows["terminal-slot"]["extension"], ["unchanged"])
            manifest = json.loads((output / "run_manifest.json").read_text())
            artifact_paths = [row["path"] for row in manifest["artifacts"]]
            self.assertNotIn("run_manifest.json", artifact_paths)
            self.assertEqual(artifact_paths, sorted(artifact_paths))
            all_bytes = b"\n".join(_tree_bytes(output).values())
            self.assertNotIn(str(root).encode(), all_bytes)
            self.assertNotIn(b"timestamp", all_bytes.lower())
            self.assertNotIn(b"provider", all_bytes.lower())

    def test_run_cli_empty_queue_is_successful_and_keeps_route_base_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path, replay_path, _, _ = _cli_fixture(root / "input", empty=True)
            output = root / "output"
            completed = _run_cli(
                "run",
                "--queue",
                queue_path,
                "--replay-actions",
                replay_path,
                "--output-dir",
                output,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            resolutions = json.loads((output / "part2_resolutions.json").read_text())
            self.assertEqual(resolutions["resolutions"], [])
            self.assertEqual((output / "decision_trace.jsonl").read_bytes(), b"")
            final = json.loads((output / "final_route_states.json").read_text())
            self.assertEqual(final["decisions"][0]["slot_id"], "nonqueued-unknown")
            self.assertEqual(final["decisions"][0]["state"], "unknown")
            self.assertEqual(tuple((output / "tool_artifacts").iterdir()), ())

    def test_run_cli_is_byte_deterministic_across_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path, replay_path, _, _ = _cli_fixture(root / "input")
            outputs = (root / "run-a", root / "run-b")
            for output in outputs:
                completed = _run_cli(
                    "run",
                    "--queue",
                    queue_path,
                    "--replay-actions",
                    replay_path,
                    "--output-dir",
                    output,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(_tree_bytes(outputs[0]), _tree_bytes(outputs[1]))

    def test_cli_input_errors_are_nonzero_without_tracebacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path, replay_path, base_path, _ = _cli_fixture(root / "input")
            base_path.write_bytes(base_path.read_bytes() + b"tampered")
            invalid = _run_cli(
                "run",
                "--queue",
                queue_path,
                "--replay-actions",
                replay_path,
                "--output-dir",
                root / "output",
            )
            unsupported = _run_cli(
                "run",
                "--queue",
                queue_path,
                "--replay-actions",
                replay_path,
                "--provider",
                "live",
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertNotIn("Traceback", invalid.stderr)
            self.assertNotEqual(unsupported.returncode, 0)
            self.assertNotIn("Traceback", unsupported.stderr)


if __name__ == "__main__":
    unittest.main()
