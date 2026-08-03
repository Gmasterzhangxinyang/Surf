from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace
import unittest

from parking_slot_part2.contracts import (
    EncounterRef,
    QueueEnvelope,
    QueueItem,
    RelationshipRef,
    freeze_json,
)
from parking_slot_part2.grouping import TaskGroup
from parking_slot_part2.model import (
    REPLAY_SCHEMA_VERSION,
    GroupFinalProposal,
    ModelAdapter,
    ReplayModelAdapter,
    SlotAssessment,
    ToolRequest,
)
from parking_slot_part2.orchestrator import ToolAttemptRecord, run_group
from parking_slot_part2.tools import ToolResult
from parking_slot_part2.trace import EvidenceLedger


EV_A = "ev_" + "a" * 64
EV_B = "ev_" + "b" * 64
EV_C = "ev_" + "c" * 64
EV_D = "ev_" + "d" * 64


def _item(number: int, *, allowed: tuple[str, ...] = ("occupied", "free", "unknown")) -> QueueItem:
    return QueueItem(
        task_id=f"task-{number}",
        slot_id=f"slot-{number}",
        scope_status="in_route_scope",
        state="unknown",
        agent_observable=True,
        unknown_reasons=("weak_vehicle_evidence", "unresolved_occlusion"),
        priority="normal",
        available_modalities=("lidar",),
        suggested_tools=("inspect_lidar_map",),
        allowed_final_states=allowed,
        occupied_evidence=freeze_json({}),
        free_evidence=freeze_json({}),
        audit=freeze_json({}),
        relationships=RelationshipRef((), (), ()),
        encounter=EncounterRef(
            encounter_id="encounter-a",
            part1_trace_event_ids=(),
            start_lidar_frame=1,
            anchor_frame=2,
            end_lidar_frame=3,
            start_timestamp=1.0,
            anchor_timestamp=2.0,
            end_timestamp=3.0,
            support_frames=(),
            pointcloud_resource_ids=(),
            visual_frames=(),
        ),
    )


def _queue(*items: QueueItem) -> QueueEnvelope:
    return QueueEnvelope(
        schema_version="unknown-agent-queue/1.0",
        queue_id="queue-a",
        producer=freeze_json({"dataset_id": "route-a"}),
        resources=freeze_json({}),
        tool_registry_version="part2-tools/1.0",
        items=items,
    )


def _assessment(
    item: QueueItem,
    *,
    state: str = "unknown",
    evidence_refs: list[str] | None = None,
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "task_id": item.task_id,
        "slot_id": item.slot_id,
        "proposed_state": state,
        "target_visibility": "unknown",
        "target_ownership": "uncertain",
        "semantic_finding": "unclear",
        "resolved_unknown_reasons": [],
        "unresolved_blockers": list(item.unknown_reasons),
        "evidence_refs": list(evidence_refs or []),
        "reason_codes": ["insufficient_evidence"],
    }
    payload.update(overrides)
    return payload


def _final(*assessments: dict[str, object], **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "final_proposal",
        "assessments": list(assessments),
    }
    payload.update(extra)
    return payload


def _tool(
    evidence_id: str,
    *,
    tool_name: str = "inspect_lidar_map",
    arguments: dict[str, object] | None = None,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "tool_request",
        "tool_name": tool_name,
        "arguments": arguments if arguments is not None else {"evidence_id": evidence_id},
    }
    payload.update(extra)
    return payload


def _replay(group_id: str, actions: list[object]) -> ReplayModelAdapter:
    return ReplayModelAdapter(
        {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "actions": {group_id: actions},
        }
    )


class RecordingAdapter:
    def __init__(self, actions: list[object]) -> None:
        self.actions = list(actions)
        self.requests: list[dict[str, object]] = []

    def next_action(self, request):
        self.requests.append(json.loads(json.dumps(request)))
        if not self.actions:
            raise AssertionError("unexpected model turn")
        return self.actions.pop(0)


class RecordingRegistry:
    names = ("inspect_lidar_map", "inspect_rgb_frame", "inspect_rgb_sequence")

    def __init__(
        self,
        queue: QueueEnvelope,
        *,
        task_by_evidence: dict[str, str] | None = None,
        results: list[ToolResult] | None = None,
    ) -> None:
        evidence_map = task_by_evidence or {
            EV_A: queue.items[0].task_id,
            EV_B: queue.items[0].task_id,
            EV_C: queue.items[0].task_id,
            EV_D: queue.items[0].task_id,
        }
        records = {
            evidence_id: SimpleNamespace(
                evidence_id=evidence_id,
                task_id=task_id,
                tool_name="inspect_lidar_map",
            )
            for evidence_id, task_id in evidence_map.items()
        }
        self.catalog = SimpleNamespace(
            queue=queue,
            get=lambda evidence_id: records[evidence_id],
        )
        self.results = list(results or [])
        self.invocations: list[tuple[str, dict[str, object]]] = []

    def invoke(self, tool_name, arguments):
        self.invocations.append((tool_name, dict(arguments)))
        if self.results:
            return self.results.pop(0)
        return ToolResult(
            tool_name=tool_name,
            status="ok",
            evidence_id=arguments["evidence_id"],
            data=freeze_json({"semantic_inference_performed": False}),
        )


class ReplayAdapterTest(unittest.TestCase):
    def test_task3_public_contracts_are_exported_from_the_package(self):
        import parking_slot_part2 as part2

        for name in (
            "EvidenceLedger",
            "GroupFinalProposal",
            "GroupRunResult",
            "ModelAdapter",
            "ReplayModelAdapter",
            "SlotAssessment",
            "ToolAttemptRecord",
            "ToolRequest",
            "run_group",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(part2, name))

    def test_replay_actions_are_consumed_in_order_independently_per_group(self):
        payload = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "actions": {
                "group-a": [_tool(EV_A), _final()],
                "group-b": [_tool(EV_B)],
            },
        }
        adapter = ReplayModelAdapter(payload)

        first_a = adapter.next_action({"group_id": "group-a", "turn": 1})
        first_b = adapter.next_action({"group_id": "group-b", "turn": 1})
        second_a = adapter.next_action({"group_id": "group-a", "turn": 2})

        self.assertEqual(first_a, _tool(EV_A))
        self.assertEqual(first_b, _tool(EV_B))
        self.assertEqual(second_a, _final())
        first_a["tool_name"] = "mutated-by-caller"
        self.assertEqual(payload["actions"]["group-a"][0]["tool_name"], "inspect_lidar_map")

    def test_replay_envelope_is_versioned_strict_and_has_no_provider_fields(self):
        valid = {"schema_version": REPLAY_SCHEMA_VERSION, "actions": {"group-a": []}}
        ReplayModelAdapter(valid)

        invalid_payloads = (
            {"schema_version": "part2-replay-actions/0.9", "actions": {}},
            {**valid, "provider": "live-provider"},
            {"schema_version": REPLAY_SCHEMA_VERSION, "actions": []},
            {"schema_version": REPLAY_SCHEMA_VERSION, "actions": {"": []}},
            {"schema_version": REPLAY_SCHEMA_VERSION, "actions": {"group-a": "not-a-list"}},
            {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "actions": {"group-a": [{"type": "final_proposal", "provider": "live"}]},
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ReplayModelAdapter(payload)

    def test_public_action_records_are_typed_and_immutable(self):
        request = ToolRequest("inspect_lidar_map", freeze_json({"evidence_id": EV_A}))
        assessment = SlotAssessment(
            task_id="task-1",
            slot_id="slot-1",
            proposed_state="unknown",
            target_visibility="unknown",
            target_ownership="uncertain",
            semantic_finding="unclear",
            resolved_unknown_reasons=(),
            unresolved_blockers=("weak_vehicle_evidence", "unresolved_occlusion"),
            evidence_refs=(),
            reason_codes=("insufficient_evidence",),
        )
        proposal = GroupFinalProposal("group-a", (assessment,))

        self.assertEqual(request.to_action(), _tool(EV_A))
        self.assertEqual(proposal.to_action(), _final(_assessment(_item(1))))
        with self.assertRaises(FrozenInstanceError):
            assessment.proposed_state = "free"  # type: ignore[misc]

        self.assertTrue(hasattr(ModelAdapter, "next_action"))


class OrchestratorValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first = _item(1)
        self.second = _item(2)
        self.queue = _queue(self.first, self.second)
        self.group = TaskGroup("group-a", "encounter-a", ("task-1", "task-2"))

    def _run(self, actions, *, registry=None):
        adapter = RecordingAdapter(list(actions))
        registry = registry or RecordingRegistry(self.queue)
        result = run_group(
            self.group,
            self.queue,
            adapter,
            registry,
            basic_context=freeze_json({"scene": "opaque-basic-context"}),
        )
        return result, adapter, registry

    def test_invalid_final_schema_and_enum_feed_structured_repair_context(self):
        invalid = _assessment(self.first, target_visibility="mostly_visible")
        result, adapter, _ = self._run(
            [
                _final(invalid, _assessment(self.second)),
                _final(_assessment(self.first), _assessment(self.second)),
            ]
        )

        self.assertEqual(result.stop_reason, "final_proposal")
        self.assertEqual(result.model_turns, 2)
        feedback = adapter.requests[1]["observations"][-1]
        self.assertEqual(feedback["kind"], "validation_feedback")
        self.assertEqual(feedback["code"], "invalid_target_visibility")
        self.assertNotIn("mostly_visible", json.dumps(feedback))

    def test_model_request_receives_structured_part1_evidence(self):
        result, adapter, _ = self._run(
            [_final(_assessment(self.first), _assessment(self.second))]
        )

        self.assertEqual(result.stop_reason, "final_proposal")
        tasks = adapter.requests[0]["decision_tasks"]
        self.assertEqual(len(tasks), 2)
        self.assertIn("occupied_evidence", tasks[0])
        self.assertIn("free_evidence", tasks[0])

    def test_final_requires_one_exact_task_and_slot_assessment(self):
        duplicate = _assessment(self.first)
        extra = dict(_assessment(self.second), task_id="task-3", slot_id="slot-3")
        result, adapter, _ = self._run(
            [
                _final(_assessment(self.first)),
                _final(duplicate, duplicate, _assessment(self.second)),
                _final(_assessment(self.first), extra),
                _final(_assessment(self.first), _assessment(self.second)),
            ]
        )

        self.assertEqual(result.stop_reason, "final_proposal")
        self.assertEqual(result.model_turns, 4)
        codes = [request["observations"][-1]["code"] for request in adapter.requests[1:]]
        self.assertEqual(codes, ["missing_assessment", "duplicate_assessment", "unexpected_assessment"])
        self.assertEqual(tuple(row.task_id for row in result.assessments), ("task-1", "task-2"))

    def test_slot_pair_and_item_allowed_state_are_enforced(self):
        restricted = _item(1, allowed=("occupied", "unknown"))
        queue = _queue(restricted)
        group = TaskGroup("group-a", "encounter-a", (restricted.task_id,))
        adapter = RecordingAdapter(
            [
                _final(_assessment(restricted, slot_id="slot-wrong")),
                _final(_assessment(restricted, state="free")),
                _final(_assessment(restricted)),
            ]
        )

        result = run_group(
            group,
            queue,
            adapter,
            RecordingRegistry(queue),
            basic_context={},
        )

        self.assertEqual(result.stop_reason, "final_proposal")
        self.assertEqual(
            [request["observations"][-1]["code"] for request in adapter.requests[1:]],
            ["assessment_slot_mismatch", "state_not_allowed"],
        )

    def test_extra_provider_or_assessment_fields_are_rejected_then_repaired(self):
        with_provider = _final(
            _assessment(self.first),
            _assessment(self.second),
            provider="not-part-of-replay-contract",
        )
        with_extra_assessment = _assessment(self.first, confidence=0.8)
        result, adapter, _ = self._run(
            [
                with_provider,
                _final(with_extra_assessment, _assessment(self.second)),
                _final(_assessment(self.first), _assessment(self.second)),
            ]
        )

        self.assertEqual(result.stop_reason, "final_proposal")
        self.assertEqual(
            [request["observations"][-1]["code"] for request in adapter.requests[1:]],
            ["invalid_final_proposal_schema", "invalid_assessment_schema"],
        )

    def test_assessment_evidence_must_belong_to_the_same_task(self):
        registry = RecordingRegistry(
            self.queue,
            task_by_evidence={EV_A: self.second.task_id},
        )
        adapter = RecordingAdapter(
            [
                _final(
                    _assessment(self.first, evidence_refs=[EV_A]),
                    _assessment(self.second),
                ),
                _final(_assessment(self.first), _assessment(self.second)),
            ]
        )

        result = run_group(
            self.group,
            self.queue,
            adapter,
            registry,
            basic_context={},
        )

        self.assertEqual(result.model_turns, 2)
        self.assertEqual(adapter.requests[1]["observations"][-1]["code"], "evidence_task_mismatch")

    def test_unexpected_task_value_is_not_echoed_into_feedback_or_trace(self):
        unexpected = dict(
            _assessment(self.second),
            task_id="/tmp/model-controlled-secret",
            slot_id="slot-untrusted",
        )
        adapter = RecordingAdapter(
            [
                _final(_assessment(self.first), unexpected),
                _final(_assessment(self.first), _assessment(self.second)),
            ]
        )

        result = run_group(
            self.group,
            self.queue,
            adapter,
            RecordingRegistry(self.queue),
            basic_context={},
        )

        self.assertEqual(adapter.requests[1]["observations"][-1]["code"], "unexpected_assessment")
        self.assertNotIn("model-controlled-secret", json.dumps(adapter.requests[1]))
        self.assertNotIn("model-controlled-secret", result.trace_jsonl)


class OrchestratorBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first = _item(1)
        self.queue = _queue(self.first)
        self.group = TaskGroup("group-a", "encounter-a", (self.first.task_id,))

    def _run(self, actions, registry):
        adapter = RecordingAdapter(list(actions))
        result = run_group(
            self.group,
            self.queue,
            adapter,
            registry,
            basic_context={"basic_context_is_free": True},
        )
        return result, adapter

    def test_invalid_failed_and_duplicate_calls_each_consume_attempt_budget(self):
        failed = ToolResult(
            tool_name="inspect_lidar_map",
            status="failed",
            evidence_id=EV_A,
            data=freeze_json({}),
            error_code="backend_failed",
            error_message="deterministic failure",
        )
        registry = RecordingRegistry(self.queue, results=[failed])
        valid = _tool(EV_A)
        result, adapter = self._run(
            [
                _tool(EV_A, tool_name="read_file", arguments={"path": "/tmp/secret"}),
                valid,
                dict(reversed(list(valid.items()))),
                _final(_assessment(self.first)),
            ],
            registry,
        )

        self.assertEqual(result.model_turns, 4)
        self.assertEqual(result.tool_attempts, 3)
        self.assertEqual(result.tool_attempt_count, 3)
        self.assertEqual(len(result.tool_attempt_records), 3)
        self.assertEqual(len(registry.invocations), 1, "duplicate requests must not execute twice")
        self.assertEqual(result.stop_reason, "final_proposal")
        self.assertEqual(
            tuple(attempt.disposition for attempt in result.tool_attempt_records),
            ("prevalidation_rejected", "executed", "duplicate"),
        )
        self.assertEqual(tuple(attempt.attempt_index for attempt in result.tool_attempt_records), (1, 2, 3))
        self.assertEqual(len({attempt.attempt_id for attempt in result.tool_attempt_records}), 3)
        self.assertFalse(result.tool_attempt_records[0].executed)
        self.assertEqual(dict(result.tool_attempt_records[0].arguments), {})
        self.assertEqual(result.tool_attempt_records[1].result_status, "failed")
        self.assertTrue(result.tool_attempt_records[1].executed)
        self.assertEqual(result.tool_attempt_records[1].evidence_id, EV_A)
        self.assertFalse(result.tool_attempt_records[2].executed)
        self.assertEqual(result.tool_attempt_records[2].error_code, "duplicate_tool_request")
        self.assertIsInstance(result.tool_attempt_records[2], ToolAttemptRecord)
        all_observations = [row for request in adapter.requests for row in request["observations"]]
        self.assertTrue(any(row.get("code") == "tool_not_allowlisted" for row in all_observations))
        self.assertTrue(any(row.get("code") == "duplicate_tool_request" for row in all_observations))
        self.assertNotIn("/tmp/secret", result.trace_jsonl)

    def test_unavailable_calls_consume_budget_and_fourth_turn_may_finalize(self):
        unavailable = [
            ToolResult(
                tool_name="inspect_lidar_map",
                status="unavailable",
                evidence_id=evidence_id,
                data=freeze_json({"evidence_id": evidence_id}),
                error_code="artifact_missing",
                error_message="not available",
            )
            for evidence_id in (EV_A, EV_B, EV_C)
        ]
        registry = RecordingRegistry(self.queue, results=unavailable)
        result, _ = self._run(
            [
                _tool(EV_A),
                _tool(EV_B),
                _tool(EV_C),
                _final(_assessment(self.first)),
            ],
            registry,
        )

        self.assertEqual((result.model_turns, result.tool_attempt_count), (4, 3))
        self.assertEqual(len(registry.invocations), 3)
        self.assertEqual(result.stop_reason, "final_proposal")

    def test_duplicate_exact_invalid_request_is_reported_without_execution(self):
        registry = RecordingRegistry(self.queue)
        invalid = _tool(
            EV_A,
            tool_name="read_file",
            arguments={"path": "/tmp/model-controlled"},
        )
        result, adapter = self._run(
            [invalid, dict(reversed(list(invalid.items()))), _final(_assessment(self.first))],
            registry,
        )

        self.assertEqual(result.tool_attempts, 2)
        self.assertEqual(registry.invocations, [])
        self.assertEqual(
            tuple(row.disposition for row in result.tool_attempt_records),
            ("prevalidation_rejected", "duplicate"),
        )
        self.assertEqual(adapter.requests[2]["observations"][-1]["code"], "duplicate_tool_request")
        self.assertNotIn("/tmp/model-controlled", result.trace_jsonl)

    def test_fourth_tool_request_is_not_executed_and_stops_with_unknown(self):
        registry = RecordingRegistry(self.queue)
        result, _ = self._run(
            [_tool(EV_A), _tool(EV_B), _tool(EV_C), _tool(EV_D)],
            registry,
        )

        self.assertEqual((result.model_turns, result.tool_attempt_count), (4, 3))
        self.assertEqual(len(registry.invocations), 3)
        self.assertEqual(result.stop_reason, "tool_budget_exhausted")
        self.assertTrue(all(row.proposed_state == "unknown" for row in result.assessments))

    def test_cross_group_evidence_is_rejected_without_registry_execution(self):
        outsider = _item(2)
        queue = _queue(self.first, outsider)
        registry = RecordingRegistry(queue, task_by_evidence={EV_A: outsider.task_id})
        adapter = RecordingAdapter([_tool(EV_A), _final(_assessment(self.first))])

        result = run_group(
            self.group,
            queue,
            adapter,
            registry,
            basic_context={},
        )

        self.assertEqual(result.tool_attempt_count, 1)
        self.assertEqual(registry.invocations, [])
        self.assertEqual(adapter.requests[1]["observations"][-1]["code"], "evidence_outside_group")

    def test_invalid_argument_field_names_cannot_echo_model_controlled_paths(self):
        registry = RecordingRegistry(self.queue)
        adapter = RecordingAdapter(
            [
                _tool(EV_A, arguments={"evidence_id": EV_A, "/tmp/model-path": "ignored"}),
                _final(_assessment(self.first)),
            ]
        )

        result = run_group(
            self.group,
            self.queue,
            adapter,
            registry,
            basic_context={},
        )

        self.assertEqual(registry.invocations, [])
        self.assertEqual(adapter.requests[1]["observations"][-1]["code"], "invalid_tool_arguments")
        self.assertNotIn("/tmp/model-path", json.dumps(adapter.requests[1]))
        self.assertNotIn("/tmp/model-path", result.trace_jsonl)

    def test_non_allowlisted_tool_name_is_not_persisted_in_audit_data(self):
        registry = RecordingRegistry(self.queue)
        secret_name = "model_secret_reasoning"
        adapter = RecordingAdapter(
            [_tool(EV_A, tool_name=secret_name), _final(_assessment(self.first))]
        )

        result = run_group(
            self.group,
            self.queue,
            adapter,
            registry,
            basic_context={},
        )

        self.assertEqual(registry.invocations, [])
        self.assertIsNone(result.tool_attempt_records[0].tool_name)
        self.assertNotIn(secret_name, result.trace_jsonl)
        self.assertNotIn(secret_name, json.dumps(result.tool_attempt_records[0].to_dict()))


class OrchestratorFallbackAndTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first = _item(1)
        self.second = _item(2)
        self.queue = _queue(self.first, self.second)
        self.group = TaskGroup("group-a", "encounter-a", ("task-1", "task-2"))

    def test_four_invalid_model_turns_fall_back_unknown_for_every_task(self):
        adapter = RecordingAdapter([None, [], {"type": "mystery"}, {"type": "mystery"}])
        result = run_group(
            self.group,
            self.queue,
            adapter,
            RecordingRegistry(self.queue),
            basic_context={},
        )

        self.assertEqual(result.model_turns, 4)
        self.assertEqual(result.tool_attempt_count, 0)
        self.assertEqual(result.stop_reason, "model_turn_limit")
        self.assertEqual(tuple(row.task_id for row in result.assessments), self.group.task_ids)
        for row in result.assessments:
            self.assertEqual(row.proposed_state, "unknown")
            self.assertEqual(row.target_visibility, "unknown")
            self.assertEqual(row.target_ownership, "uncertain")
            self.assertEqual(row.semantic_finding, "unclear")
            self.assertEqual(row.evidence_refs, ())
            self.assertTrue(row.unresolved_blockers)

    def test_missing_exhausted_and_model_error_paths_are_safe_structured_stops(self):
        missing = ReplayModelAdapter(
            {"schema_version": REPLAY_SCHEMA_VERSION, "actions": {"other-group": []}}
        )
        exhausted = _replay(self.group.group_id, [])

        class ErrorAdapter:
            def next_action(self, request):
                raise RuntimeError("provider details must not leak")

        for expected, adapter in (
            ("missing_replay", missing),
            ("replay_exhausted", exhausted),
            ("model_error", ErrorAdapter()),
        ):
            with self.subTest(expected=expected):
                result = run_group(
                    self.group,
                    self.queue,
                    adapter,
                    RecordingRegistry(self.queue),
                    basic_context={},
                )
                self.assertEqual(result.stop_reason, expected)
                self.assertEqual(result.model_turns, 1)
                self.assertTrue(all(row.proposed_state == "unknown" for row in result.assessments))
                self.assertNotIn("provider details", result.trace_jsonl)

    def test_trace_is_canonical_deterministic_and_covers_event_categories(self):
        failed = ToolResult(
            tool_name="inspect_lidar_map",
            status="failed",
            evidence_id=EV_A,
            data=freeze_json({}),
            error_code="backend_failed",
            error_message="stable failure",
        )

        def once():
            return run_group(
                self.group,
                self.queue,
                _replay(
                    self.group.group_id,
                    [
                        _tool(EV_A),
                        _final(_assessment(self.first), _assessment(self.second)),
                    ],
                ),
                RecordingRegistry(self.queue, results=[failed]),
                basic_context=freeze_json({"scene": "same"}),
            )

        first = once()
        second = once()

        self.assertEqual(first.trace_span_id, second.trace_span_id)
        self.assertEqual(first.trace_jsonl, second.trace_jsonl)
        self.assertTrue(first.trace_jsonl.endswith("\n"))
        events = [json.loads(line) for line in first.trace_jsonl.splitlines()]
        self.assertEqual([event["sequence"] for event in events], list(range(1, len(events) + 1)))
        self.assertEqual(len({event["event_id"] for event in events}), len(events))
        self.assertTrue({"lifecycle", "model", "tool", "validation", "stop"} <= {event["category"] for event in events})
        accepted = next(event for event in events if event["event_type"] == "final_proposal_accepted")
        self.assertEqual(
            [row["proposed_state"] for row in accepted["data"]["proposal"]["assessments"]],
            ["unknown", "unknown"],
        )
        for line, event in zip(first.trace_jsonl.splitlines(), events):
            self.assertEqual(
                line,
                json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            )
        lowered = first.trace_jsonl.lower()
        self.assertNotIn("timestamp", lowered)
        self.assertNotIn("chain_of_thought", lowered)
        self.assertNotIn("provider details", lowered)

    def test_ledger_rejects_reasoning_and_nondeterministic_timestamp_payloads(self):
        ledger = EvidenceLedger("group-a")
        for payload in (
            {"chain_of_thought": "secret"},
            {"reasoning": "secret"},
            {"timestamp": "now"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ledger.record("model", "model_action_received", payload)


if __name__ == "__main__":
    unittest.main()
