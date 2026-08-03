from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from parking_slot_part2.contracts import (
    EncounterRef,
    QueueEnvelope,
    QueueItem,
    RelationshipRef,
    ResourceRef,
    VisualFrameRef,
    freeze_json,
)
from parking_slot_part2.decision import consolidate_group_assessments
from parking_slot_part2.grouping import build_groups
from parking_slot_part2.model import REPLAY_SCHEMA_VERSION, ReplayModelAdapter
from parking_slot_part2.orchestrator import run_group
from parking_slot_part2.preflight import EvidenceCatalog
from parking_slot_part2.shadow_replay import compile_shadow_replay
from parking_slot_part2.tools import ToolRegistry


HASH = "sha256:" + "a" * 64


def _sha(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _resource(resource_id: str, kind: str, path: Path, content: bytes) -> ResourceRef:
    path.write_bytes(content)
    return ResourceRef(
        resource_id=resource_id,
        kind=kind,
        uri=path.name,
        sha256=_sha(content),
        manifest_hash=None,
        dataset_id="route-a",
        config_hash=HASH,
        slot_map_hash=HASH,
    )


def _visual(
    number: int,
    resource_id: str,
    *,
    capabilities: tuple[str, ...] = ("can_assess_occupied", "can_assess_free"),
) -> VisualFrameRef:
    return VisualFrameRef(
        visual_frame_id=f"visual-{number}",
        lidar_frame=90,
        camera_frame=1_000 + number,
        lidar_timestamp=9.0,
        camera_timestamp=8.99,
        camera_lidar_dt_sec=-0.01,
        image_resource_id=resource_id,
        capabilities=capabilities,
        projection_quality=freeze_json(
            {
                "sync_source": "corrected",
                "bearing_deg": 5.0,
                "distance_m": 8.0,
                "finite_vertices": 4,
                "projected_area_px": 1_400.0,
                "bbox_px": [10.0, 20.0, 90.0, 60.0],
                "visible_fraction": 0.95,
                "image_width_px": 100,
                "image_height_px": 80,
                "polygon_uv": [[10.0, 20.0], [90.0, 20.0], [90.0, 60.0], [10.0, 60.0]],
                "adjacent_polygons_uv": {},
            }
        ),
    )


def _item(
    number: int,
    *,
    conflicts: tuple[str, ...] = (),
    shared: tuple[str, ...] = (),
    lidar: bool = True,
    rgb: bool = True,
    rgb_capabilities: tuple[str, ...] = (
        "can_assess_occupied",
        "can_assess_free",
    ),
) -> QueueItem:
    lidar_id = f"lidar-{number}"
    rgb_id = f"rgb-{number}"
    return QueueItem(
        task_id=f"task-{number}",
        slot_id=f"slot-{number}",
        scope_status="in_route_scope",
        state="unknown",
        agent_observable=True,
        unknown_reasons=("weak_vehicle_evidence", "unresolved_occlusion"),
        priority="normal",
        available_modalities=tuple(
            name for name, present in (("lidar", lidar), ("rgb", rgb)) if present
        ),
        suggested_tools=(),
        allowed_final_states=("occupied", "free", "unknown"),
        occupied_evidence=freeze_json({}),
        free_evidence=freeze_json({}),
        audit=freeze_json({}),
        relationships=RelationshipRef(
            adjacent_slot_ids=(),
            conflict_slot_ids=conflicts,
            shared_evidence_ids=shared,
        ),
        encounter=EncounterRef(
            encounter_id="encounter-a",
            part1_trace_event_ids=(),
            start_lidar_frame=1,
            anchor_frame=100,
            end_lidar_frame=110,
            start_timestamp=1.0,
            anchor_timestamp=10.0,
            end_timestamp=11.0,
            support_frames=(90,),
            pointcloud_resource_ids=(lidar_id,) if lidar else (),
            visual_frames=(
                (_visual(number, rgb_id, capabilities=rgb_capabilities),)
                if rgb
                else ()
            ),
        ),
    )


def _queue(root: Path, items: tuple[QueueItem, ...]) -> QueueEnvelope:
    resources: dict[str, ResourceRef] = {}
    for item in items:
        number = int(item.task_id.rsplit("-", 1)[1])
        for resource_id in item.encounter.pointcloud_resource_ids:
            content = f"lidar-{number}".encode()
            resources[resource_id] = _resource(
                resource_id,
                "lidar_map",
                root / f"{resource_id}.bin",
                content,
            )
        for frame in item.encounter.visual_frames:
            content = b"P6\n100 80\n255\n" + bytes((40 + number, 80, 120)) * (100 * 80)
            resources[frame.image_resource_id] = _resource(
                frame.image_resource_id,
                "rgb_frame",
                root / f"{frame.image_resource_id}.png",
                content,
            )
    return QueueEnvelope(
        schema_version="unknown-agent-queue/1.0",
        queue_id="queue-shadow-test",
        producer=freeze_json({"dataset_id": "route-a"}),
        resources=freeze_json(resources),
        tool_registry_version="part2-tools/1.0",
        items=items,
    )


def _blind(
    state: str,
    *,
    confidence: float | None = 0.8,
    visibility: str | None = None,
    ownership: str | None = None,
    finding: str | None = None,
    reason: str = "blind_semantic_review",
) -> dict[str, object]:
    defaults = {
        "occupied": ("clear_partial", "target", "vehicle_or_occupying_object"),
        "free": ("clear_full", "target", "empty"),
        # Deliberately terminal-looking values verify that the compiler replaces
        # every unknown with the fixed safe assessment fields.
        "unknown": ("clear_full", "target", "empty"),
    }
    default_visibility, default_ownership, default_finding = defaults[state]
    record: dict[str, object] = {
        "state": state,
        "visibility": visibility or default_visibility,
        "ownership": ownership or default_ownership,
        "finding": finding or default_finding,
        "reason_codes": [reason],
    }
    if confidence is not None:
        record["confidence"] = confidence
    return record


def _final_assessments(actions: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    final = actions[-1]
    assert final["type"] == "final_proposal"
    return {row["task_id"]: row for row in final["assessments"]}


class ShadowReplayCompilerTest(unittest.TestCase):
    def test_compiles_target_tools_and_complete_safe_group_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = tuple(_item(number, shared=("shared-all",)) for number in range(1, 5))
            queue = _queue(root, items)
            blind = {
                "slot-1": _blind("occupied", confidence=0.9, visibility="clear_full"),
                "slot-2": _blind("free", confidence=0.8),
                "slot-3": _blind("occupied", confidence=0.7, visibility="clear_full"),
                "slot-4": _blind("unknown", confidence=0.6),
            }

            payload = compile_shadow_replay(queue, blind, base_dir=root)
            ReplayModelAdapter(payload)
            group = build_groups(queue)[0]
            actions = payload["actions"][group.group_id]

            self.assertEqual(payload["schema_version"], REPLAY_SCHEMA_VERSION)
            self.assertEqual(len(actions), 4)
            self.assertEqual([row["type"] for row in actions], [
                "tool_request",
                "tool_request",
                "tool_request",
                "final_proposal",
            ])
            assessments = _final_assessments(actions)
            self.assertEqual(set(assessments), set(group.task_ids))

            catalog = EvidenceCatalog(queue, base_dir=root)
            for action in actions[:-1]:
                record = catalog.get(action["arguments"]["evidence_id"])
                assessment = assessments[record.task_id]
                self.assertEqual(record.slot_id, assessment["slot_id"])
                self.assertEqual(assessment["evidence_refs"], [record.evidence_id])
            unknown = assessments["task-4"]
            self.assertEqual(unknown["proposed_state"], "unknown")
            self.assertEqual(unknown["target_visibility"], "unknown")
            self.assertEqual(unknown["target_ownership"], "uncertain")
            self.assertEqual(unknown["semantic_finding"], "unclear")
            self.assertEqual(unknown["resolved_unknown_reasons"], [])
            self.assertEqual(unknown["unresolved_blockers"], list(items[3].unknown_reasons))
            self.assertEqual(unknown["evidence_refs"], [])

            result = run_group(
                group,
                queue,
                ReplayModelAdapter(payload),
                ToolRegistry(catalog),
                basic_context={},
            )
            resolutions = consolidate_group_assessments(queue, (result,), catalog)
            self.assertEqual(
                {row.slot_id: row.state for row in resolutions},
                {
                    "slot-1": "occupied",
                    "slot-2": "free",
                    "slot-3": "occupied",
                    "slot-4": "unknown",
                },
            )

    def test_free_without_eligible_single_rgb_frame_is_downgraded_without_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = _item(
                1,
                rgb_capabilities=("can_assess_occupied",),
            )
            queue = _queue(root, (item,))

            payload = compile_shadow_replay(
                queue,
                {"slot-1": _blind("free")},
                base_dir=root,
            )

        actions = next(iter(payload["actions"].values()))
        self.assertEqual(len(actions), 1)
        assessment = _final_assessments(actions)["task-1"]
        self.assertEqual(assessment["proposed_state"], "unknown")
        self.assertEqual(assessment["evidence_refs"], [])
        self.assertIn(
            "shadow_free_rgb_evidence_unavailable",
            assessment["reason_codes"],
        )

    def test_lowest_confidence_overlap_slot_is_globally_fixed_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = (
                _item(1, conflicts=("slot-2",)),
                _item(2, conflicts=("slot-3",)),
                _item(3, conflicts=("slot-4",)),
                _item(4, conflicts=("slot-5",)),
                _item(5, conflicts=("slot-6",)),
                _item(6),
            )
            queue = _queue(root, items)
            blind = {
                item.slot_id: _blind(
                    "occupied",
                    confidence=0.01 if item.slot_id == "slot-4" else 0.9,
                    visibility="clear_full",
                )
                for item in items
            }

            payload = compile_shadow_replay(queue, blind, base_dir=root)

        groups = build_groups(queue)
        self.assertEqual(
            tuple(group.task_ids for group in groups),
            (("task-1", "task-2", "task-3", "task-4"), ("task-4", "task-5", "task-6")),
        )
        for group in groups:
            actions = payload["actions"][group.group_id]
            self.assertLessEqual(len(actions), 4)
            self.assertLessEqual(
                sum(action["type"] == "tool_request" for action in actions),
                3,
            )
            assessments = _final_assessments(actions)
            if "task-4" in assessments:
                self.assertEqual(assessments["task-4"]["proposed_state"], "unknown")
                self.assertIn(
                    "shadow_tool_budget_downgrade",
                    assessments["task-4"]["reason_codes"],
                )
        serialized = json.dumps(payload, sort_keys=True)
        catalog = EvidenceCatalog(queue, base_dir=root)
        task4_evidence = {
            evidence_id
            for evidence_id, record in catalog.entries.items()
            if record.task_id == "task-4"
        }
        self.assertTrue(task4_evidence)
        self.assertTrue(all(evidence_id not in serialized for evidence_id in task4_evidence))

    def test_recursively_rejects_path_gt_and_provider_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = _queue(root, (_item(1),))
            forbidden = (
                {"metadata": {"image_path": "/tmp/not-allowed"}},
                {"metadata": {"artifact_uri": "file:///tmp/not-allowed"}},
                {"metadata": {"review": {"ground_truth": "occupied"}}},
                {"metadata": {"gt_score": 1.0}},
                {"metadata": {"provider": "live"}},
                {"metadata": [{"model": "provider-specific"}]},
            )
            for extra in forbidden:
                record = _blind("unknown")
                record.update(extra)
                with self.subTest(extra=extra), self.assertRaisesRegex(
                    ValueError, "forbidden"
                ):
                    compile_shadow_replay(queue, {"slot-1": record}, base_dir=root)
            with self.assertRaisesRegex(ValueError, "forbidden provider"):
                compile_shadow_replay(
                    queue,
                    {"slot-1": _blind("unknown"), "provider": {"name": "live"}},
                    base_dir=root,
                )

    def test_requires_exact_slot_coverage_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = (_item(1, rgb=False), _item(2, rgb=False))
            queue = _queue(root, items)
            records = {
                "slot-2": _blind("unknown", confidence=None),
                "slot-1": _blind("occupied", confidence=None),
            }
            first = compile_shadow_replay(queue, records, base_dir=root)
            second = compile_shadow_replay(
                queue,
                dict(reversed(tuple(records.items()))),
                base_dir=root,
            )
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, "exactly one record"):
                compile_shadow_replay(
                    queue,
                    {"slot-1": records["slot-1"]},
                    base_dir=root,
                )


if __name__ == "__main__":
    unittest.main()
