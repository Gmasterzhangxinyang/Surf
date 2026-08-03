from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import tempfile
import unittest
from pathlib import Path

from parking_slot_part2 import (
    EncounterRef,
    QueueEnvelope,
    QueueItem,
    RelationshipRef,
    ResourceRef,
    VisualFrameRef,
    canonical_json_bytes,
    canonical_sha256,
    load_queue,
    make_queue_id,
    make_task_id,
    validate_queue,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


def _producer() -> dict[str, str]:
    return {
        "pipeline": "slot_hybrid_3d",
        "run_id": "part1-run-7",
        "config_hash": HASH_A,
        "dataset_id": "icpark-route-a",
        "slot_map_hash": HASH_B,
    }


def _resources() -> dict[str, dict[str, str]]:
    resources = {
        "slot-database": {
            "kind": "slot_database",
            "uri": "artifacts/slot_decisions.json",
            "sha256": HASH_A,
        },
        "corrected-frames": {
            "kind": "corrected_frames",
            "uri": "artifacts/frames.csv",
            "sha256": HASH_B,
        },
        "map-points-root": {
            "kind": "map_points_root",
            "uri": "artifacts/map_points",
            "sha256": HASH_C,
        },
        "camera-calibration": {
            "kind": "camera_calibration",
            "uri": "artifacts/camera_calibration.json",
            "sha256": HASH_A,
        },
        "base-decisions": {
            "kind": "base_slot_decisions",
            "uri": "artifacts/base_slot_decisions.json",
            "sha256": HASH_B,
        },
        "lidar-map": {
            "kind": "lidar_map",
            "uri": "artifacts/encounter-10.npz",
            "sha256": HASH_C,
        },
        "rgb-901": {
            "kind": "rgb_frame",
            "uri": "artifacts/front/000901.png",
            "sha256": HASH_A,
        },
    }
    for resource in resources.values():
        resource.update(
            {
                "dataset_id": "icpark-route-a",
                "config_hash": HASH_A,
                "slot_map_hash": HASH_B,
            }
        )
    return resources


def _item(slot_number: int = 1, **overrides: object) -> dict[str, object]:
    slot_id = str(overrides.pop("slot_id", f"slot_{slot_number:04d}"))
    encounter_id = str(overrides.pop("encounter_id", "encounter-10"))
    item: dict[str, object] = {
        "task_id": make_task_id(_producer(), slot_id, encounter_id),
        "slot_id": slot_id,
        "scope_status": "in_route_scope",
        "state": "unknown",
        "agent_observable": True,
        "unknown_reasons": ["weak_vehicle_evidence"],
        "priority": "inspect_vehicle_shape_and_occlusion",
        "available_modalities": ["lidar", "rgb"],
        "suggested_tools": ["inspect_lidar_map", "inspect_rgb_frame"],
        "allowed_final_states": ["occupied", "free", "unknown"],
        "occupied_evidence": {"strength": 0.42, "support_frame_count": 3},
        "free_evidence": {"strength": 0.18, "core_ray_coverage": 0.31},
        "audit": {"part1_contract": "slot-hybrid-3d/1.0"},
        "relationships": {
            "adjacent_slot_ids": ["slot_0002"],
            "conflict_slot_ids": [],
            "shared_evidence_ids": ["shared-window-10"],
        },
        "encounter": {
            "encounter_id": encounter_id,
            "part1_trace_event_ids": ["trace-0007"],
            "start_lidar_frame": 8,
            "anchor_frame": 10,
            "end_lidar_frame": 12,
            "start_timestamp": 99.0,
            "anchor_timestamp": 100.05,
            "end_timestamp": 101.0,
            "support_frames": [8, 10, 12],
            "pointcloud_resource_ids": ["lidar-map"],
            "visual_frames": [
                {
                    "visual_frame_id": "visual-901",
                    "lidar_frame": 10,
                    "camera_frame": 901,
                    "lidar_timestamp": 100.0,
                    "camera_timestamp": 100.012,
                    "camera_lidar_dt_sec": 0.012,
                    "image_resource_id": "rgb-901",
                    "capabilities": ["can_assess_occupied", "can_assess_free"],
                    "projection_quality": {"status": "ok", "visible_core_ratio": 0.83},
                }
            ],
        },
    }
    item.update(overrides)
    return item


def _payload(items: list[dict[str, object]] | None = None, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "unknown-agent-queue/1.0",
        "producer": _producer(),
        "resources": _resources(),
        "tool_registry_version": "part2-tools/1.0",
        "items": [_item()] if items is None else items,
    }
    payload.update(overrides)
    payload["queue_id"] = make_queue_id(payload)
    return payload


class QueueContractTest(unittest.TestCase):
    def test_valid_envelope_parses_to_deeply_immutable_contracts(self):
        queue = validate_queue(_payload())

        self.assertIsInstance(queue, QueueEnvelope)
        self.assertIsInstance(queue.items[0], QueueItem)
        self.assertIsInstance(queue.items[0].encounter, EncounterRef)
        self.assertIsInstance(queue.items[0].relationships, RelationshipRef)
        self.assertIsInstance(queue.items[0].encounter.visual_frames[0], VisualFrameRef)
        self.assertIsInstance(queue.resources["lidar-map"], ResourceRef)
        self.assertTrue(hasattr(queue.resources["lidar-map"], "dataset_id"))
        self.assertEqual(queue.resources["lidar-map"].dataset_id, "icpark-route-a")
        self.assertEqual(queue.items[0].encounter.start_lidar_frame, 8)
        self.assertEqual(queue.items[0].encounter.anchor_timestamp, 100.05)
        self.assertEqual(queue.items[0].encounter.end_timestamp, 101.0)
        self.assertEqual(queue.items[0].encounter.support_frames, (8, 10, 12))
        self.assertEqual(queue.items[0].encounter.visual_frames[0].camera_frame, 901)

        with self.assertRaises(FrozenInstanceError):
            queue.items[0].state = "free"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            queue.producer["run_id"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            queue.items[0].occupied_evidence["strength"] = 1.0  # type: ignore[index]
        with self.assertRaises(TypeError):
            queue.items[0].audit["part1_contract"] = "changed"  # type: ignore[index]

    def test_load_queue_reads_json_and_runs_the_same_validation(self):
        payload = _payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unknown_agent_queue.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_queue(path)
        self.assertEqual(loaded.queue_id, payload["queue_id"])
        self.assertEqual(loaded.items[0].slot_id, "slot_0001")

    def test_canonical_identities_ignore_mapping_order_and_reject_non_json_numbers(self):
        self.assertEqual(canonical_json_bytes({"b": 2, "a": 1}), b'{"a":1,"b":2}')
        self.assertEqual(canonical_sha256({"b": 2, "a": 1}), canonical_sha256({"a": 1, "b": 2}))
        self.assertRegex(canonical_sha256({"a": 1}), r"^sha256:[0-9a-f]{64}$")
        with self.assertRaises((TypeError, ValueError)):
            canonical_json_bytes({"strength": float("nan")})

    def test_invalid_envelope_shapes_and_versions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "object"):
            validate_queue([])

        wrong_version = _payload(schema_version="unknown-agent-queue/2.0")
        with self.assertRaisesRegex(ValueError, "schema_version"):
            validate_queue(wrong_version)

        missing_items = _payload()
        del missing_items["items"]
        missing_items["queue_id"] = make_queue_id(missing_items)
        with self.assertRaisesRegex(ValueError, "items"):
            validate_queue(missing_items)

        wrong_tools = _payload(tool_registry_version="part2-tools/2.0")
        with self.assertRaisesRegex(ValueError, "tool_registry_version"):
            validate_queue(wrong_tools)

    def test_terminal_unobservable_and_path_out_items_are_rejected(self):
        invalid_cases = (
            ("state", "occupied", "state=unknown"),
            ("state", "free", "state=unknown"),
            ("scope_status", "out_of_route_scope", "scope_status"),
            ("agent_observable", False, "agent_observable=true"),
        )
        for field, value, message in invalid_cases:
            with self.subTest(field=field, value=value):
                item = _item(**{field: value})
                payload = _payload([item])
                with self.assertRaisesRegex(ValueError, message):
                    validate_queue(payload)

    def test_partial_route_scope_item_is_valid(self):
        queue = validate_queue(_payload([_item(scope_status="partial_route_scope")]))
        self.assertEqual(queue.items[0].scope_status, "partial_route_scope")

    def test_encounter_bounds_are_required_and_ordered_around_the_anchor(self):
        required = (
            "start_lidar_frame",
            "end_lidar_frame",
            "start_timestamp",
            "anchor_timestamp",
            "end_timestamp",
        )
        for field in required:
            with self.subTest(missing=field):
                item = _item()
                del item["encounter"][field]  # type: ignore[index]
                with self.assertRaisesRegex(ValueError, field):
                    validate_queue(_payload([item]))

        invalid_values = (
            ("start_lidar_frame", 11, "start_lidar_frame.*anchor_frame"),
            ("end_lidar_frame", 9, "anchor_frame.*end_lidar_frame"),
            ("start_timestamp", 100.1, "start_timestamp.*anchor_timestamp"),
            ("end_timestamp", 100.0, "anchor_timestamp.*end_timestamp"),
        )
        for field, value, message in invalid_values:
            with self.subTest(field=field):
                item = _item()
                item["encounter"][field] = value  # type: ignore[index]
                with self.assertRaisesRegex(ValueError, message):
                    validate_queue(_payload([item]))

    def test_support_and_visual_frames_stay_within_encounter_bounds(self):
        outside_support = _item()
        outside_support["encounter"]["support_frames"] = [7, 10]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "support_frames.*encounter frame range"):
            validate_queue(_payload([outside_support]))

        invalid_visuals = (
            ({"lidar_frame": 7}, "lidar_frame.*encounter frame range"),
            ({"lidar_frame": 11}, "lidar_frame.*anchor_frame"),
            (
                {
                    "lidar_timestamp": 98.99,
                    "camera_timestamp": 99.0,
                    "camera_lidar_dt_sec": 0.01,
                },
                "lidar_timestamp.*encounter time range",
            ),
            (
                {
                    "lidar_timestamp": 100.98,
                    "camera_timestamp": 101.01,
                    "camera_lidar_dt_sec": 0.03,
                },
                "camera_timestamp.*encounter time range",
            ),
            (
                {
                    "lidar_timestamp": 100.03,
                    "camera_timestamp": 100.06,
                    "camera_lidar_dt_sec": 0.03,
                },
                "camera_timestamp.*anchor_timestamp",
            ),
            (
                {
                    "lidar_timestamp": 100.06,
                    "camera_timestamp": 100.04,
                    "camera_lidar_dt_sec": -0.02,
                },
                "lidar_timestamp.*anchor_timestamp",
            ),
        )
        for overrides, message in invalid_visuals:
            with self.subTest(overrides=overrides):
                item = _item()
                item["encounter"]["visual_frames"][0].update(overrides)  # type: ignore[index]
                with self.assertRaisesRegex(ValueError, message):
                    validate_queue(_payload([item]))

        at_anchor = _item()
        at_anchor["encounter"]["visual_frames"][0].update(  # type: ignore[index]
            {
                "lidar_timestamp": 100.05,
                "camera_timestamp": 100.04,
                "camera_lidar_dt_sec": -0.01,
            }
        )
        self.assertEqual(
            validate_queue(_payload([at_anchor])).items[0].encounter.visual_frames[0].lidar_timestamp,
            100.05,
        )

    def test_reused_encounter_id_requires_common_bounds_but_allows_distinct_anchors(self):
        first = _item(1)
        different_anchor = _item(2)
        different_anchor["encounter"]["anchor_frame"] = 11  # type: ignore[index]
        different_anchor["encounter"]["anchor_timestamp"] = 100.5  # type: ignore[index]
        queue = validate_queue(_payload([first, different_anchor]))
        self.assertEqual(queue.items[0].encounter.anchor_frame, 10)
        self.assertEqual(queue.items[1].encounter.anchor_frame, 11)

        disjoint = _item(2)
        encounter = disjoint["encounter"]  # type: ignore[assignment]
        encounter.update(  # type: ignore[union-attr]
            {
                "start_lidar_frame": 20,
                "anchor_frame": 21,
                "end_lidar_frame": 22,
                "start_timestamp": 200.0,
                "anchor_timestamp": 200.05,
                "end_timestamp": 201.0,
                "support_frames": [20, 21, 22],
            }
        )
        visual = encounter["visual_frames"][0]  # type: ignore[index]
        visual.update(  # type: ignore[union-attr]
            {
                "visual_frame_id": "visual-902",
                "lidar_frame": 21,
                "lidar_timestamp": 200.0,
                "camera_timestamp": 200.012,
                "camera_lidar_dt_sec": 0.012,
            }
        )
        with self.assertRaisesRegex(ValueError, "encounter_id.*common bounds"):
            validate_queue(_payload([first, disjoint]))

    def test_visual_camera_delta_must_match_timestamps_and_stay_within_40ms(self):
        inconsistent = _item()
        inconsistent_visual = inconsistent["encounter"]["visual_frames"][0]  # type: ignore[index]
        inconsistent_visual["camera_lidar_dt_sec"] = 0.011  # type: ignore[index]
        with self.assertRaisesRegex(
            ValueError,
            "camera_lidar_dt_sec.*camera_timestamp.*lidar_timestamp",
        ):
            validate_queue(_payload([inconsistent]))

        too_far = _item()
        too_far_visual = too_far["encounter"]["visual_frames"][0]  # type: ignore[index]
        too_far_visual["camera_timestamp"] = 100.041  # type: ignore[index]
        too_far_visual["camera_lidar_dt_sec"] = 0.041  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "camera_lidar_dt_sec.*0.04"):
            validate_queue(_payload([too_far]))

    def test_duplicate_task_and_slot_ids_are_rejected(self):
        first = _item(1)
        duplicate_task = _item(2, task_id=first["task_id"])
        with self.assertRaisesRegex(ValueError, "duplicate task_id"):
            validate_queue(_payload([first, duplicate_task]))

        duplicate_slot = _item(1, encounter_id="encounter-11")
        with self.assertRaisesRegex(ValueError, "duplicate slot_id"):
            validate_queue(_payload([first, duplicate_slot]))

    def test_mismatched_queue_and_task_hashes_are_rejected(self):
        mismatched_queue = _payload()
        mismatched_queue["queue_id"] = HASH_A
        with self.assertRaisesRegex(ValueError, "queue_id.*hash"):
            validate_queue(mismatched_queue)

        item = _item(task_id=HASH_B)
        with self.assertRaisesRegex(ValueError, "task_id.*hash"):
            validate_queue(_payload([item]))

        invalid_resource_hash = _payload()
        invalid_resource_hash["resources"]["lidar-map"]["sha256"] = "not-a-sha256"  # type: ignore[index]
        invalid_resource_hash["queue_id"] = make_queue_id(invalid_resource_hash)
        with self.assertRaisesRegex(ValueError, "resources.*sha256"):
            validate_queue(invalid_resource_hash)

    def test_required_v1_global_resource_kinds_are_enforced(self):
        required_kinds = (
            "slot_database",
            "corrected_frames",
            "map_points_root",
            "camera_calibration",
            "base_slot_decisions",
        )
        for missing_kind in required_kinds:
            with self.subTest(missing_kind=missing_kind):
                payload = _payload()
                resources = payload["resources"]
                resource_id = next(
                    key for key, value in resources.items() if value["kind"] == missing_kind  # type: ignore[union-attr]
                )
                del resources[resource_id]  # type: ignore[index]
                payload["queue_id"] = make_queue_id(payload)
                with self.assertRaisesRegex(ValueError, f"required resource kind.*{missing_kind}"):
                    validate_queue(payload)

    def test_resource_identity_linkage_must_match_the_producer(self):
        mismatches = (
            ("dataset_id", "another-dataset"),
            ("config_hash", HASH_C),
            ("slot_map_hash", HASH_C),
        )
        for field, value in mismatches:
            with self.subTest(field=field):
                payload = _payload()
                payload["resources"]["slot-database"][field] = value  # type: ignore[index]
                payload["queue_id"] = make_queue_id(payload)
                with self.assertRaisesRegex(ValueError, f"resources.*{field}.*producer"):
                    validate_queue(payload)

    def test_well_formed_sha256_must_match_a_resolved_local_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "base_slot_decisions.json"
            artifact.write_text('{"slots":[]}', encoding="utf-8")
            payload = _payload()
            payload["resources"]["base-decisions"]["uri"] = artifact.name  # type: ignore[index]
            payload["resources"]["base-decisions"]["sha256"] = HASH_A  # type: ignore[index]
            payload["queue_id"] = make_queue_id(payload)
            queue_path = root / "unknown_agent_queue.json"
            queue_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "base-decisions.*sha256.*file"):
                load_queue(queue_path)

    def test_map_points_root_uses_declared_manifest_hash_without_requiring_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = _payload()
            map_root = Path(tmp) / "map_points"
            map_root.mkdir()
            map_resource = payload["resources"]["map-points-root"]  # type: ignore[index]
            map_resource["uri"] = str(map_root)
            map_resource["manifest_hash"] = map_resource.pop("sha256")
            payload["queue_id"] = make_queue_id(payload)

            try:
                queue = validate_queue(payload)
            except ValueError as exc:
                self.fail(f"manifest-backed map_points_root should be valid: {exc}")
            self.assertEqual(queue.resources["map-points-root"].manifest_hash, HASH_C)
            self.assertIsNone(queue.resources["map-points-root"].sha256)

    def test_item_pointcloud_requires_a_concrete_file_resource_with_sha256(self):
        item = _item()
        item["encounter"]["pointcloud_resource_ids"] = ["map-points-root"]  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "pointcloud_resource_ids.*concrete"):
            validate_queue(_payload([item]))

    def test_unknown_root_fields_are_rejected_in_strict_v1(self):
        payload = _payload(live_provider={"name": "forbidden"})
        with self.assertRaisesRegex(ValueError, "unsupported field.*live_provider"):
            validate_queue(payload)

    def test_metric_depth_item_fields_are_rejected_in_strict_v1(self):
        item = _item(metric_depth_png="artifacts/depth.png")
        with self.assertRaisesRegex(ValueError, "unsupported field.*metric_depth_png"):
            validate_queue(_payload([item]))

    def test_suggested_tools_are_limited_to_the_v1_registry(self):
        item = _item(suggested_tools=["inspect_lidar_map", "read_arbitrary_file"])
        with self.assertRaisesRegex(ValueError, "suggested_tools.*unsupported"):
            validate_queue(_payload([item]))

    def test_extension_data_is_preserved_only_in_evidence_and_audit_mappings(self):
        item = _item(
            occupied_evidence={"vendor_extension": {"layers": [1, 2]}},
            audit={"producer_extension": {"version": 7}},
        )
        parsed = validate_queue(_payload([item])).items[0]
        self.assertTrue(hasattr(parsed, "audit"))
        self.assertEqual(parsed.occupied_evidence["vendor_extension"]["layers"], (1, 2))
        self.assertEqual(parsed.audit["producer_extension"]["version"], 7)

    def test_lidar_and_camera_references_are_structured_and_cross_checked(self):
        queue = validate_queue(_payload())
        encounter = queue.items[0].encounter
        visual = encounter.visual_frames[0]
        self.assertEqual(encounter.pointcloud_resource_ids, ("lidar-map",))
        self.assertEqual(queue.resources[encounter.pointcloud_resource_ids[0]].kind, "lidar_map")
        self.assertEqual(queue.resources[visual.image_resource_id].kind, "rgb_frame")
        self.assertEqual(visual.capabilities, ("can_assess_occupied", "can_assess_free"))

        missing_lidar = _payload()
        del missing_lidar["resources"]["lidar-map"]  # type: ignore[index]
        missing_lidar["queue_id"] = make_queue_id(missing_lidar)
        with self.assertRaisesRegex(ValueError, "pointcloud_resource_ids.*lidar-map"):
            validate_queue(missing_lidar)

        wrong_camera_kind = _payload()
        wrong_camera_kind["resources"]["rgb-901"]["kind"] = "lidar_map"  # type: ignore[index]
        wrong_camera_kind["queue_id"] = make_queue_id(wrong_camera_kind)
        with self.assertRaisesRegex(ValueError, "image_resource_id.*rgb_frame"):
            validate_queue(wrong_camera_kind)

    def test_empty_queue_is_valid(self):
        queue = validate_queue(_payload([]))
        self.assertEqual(queue.items, ())

    def test_queue_larger_than_legacy_300_case_fixture_is_valid(self):
        items = [_item(index, encounter_id=f"encounter-{index}") for index in range(1, 302)]
        queue = validate_queue(_payload(items))
        self.assertEqual(len(queue.items), 301)


if __name__ == "__main__":
    unittest.main()
