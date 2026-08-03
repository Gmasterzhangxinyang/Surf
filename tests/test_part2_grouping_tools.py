from __future__ import annotations

from dataclasses import replace
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
from parking_slot_part2.grouping import GROUPING_POLICY_VERSION, build_groups
from parking_slot_part2.preflight import EvidenceCatalog
from parking_slot_part2.tools import ToolRegistry


HASH = "sha256:" + "a" * 64


def _resource(
    resource_id: str,
    kind: str,
    uri: str,
    *,
    sha256: str = HASH,
) -> ResourceRef:
    return ResourceRef(
        resource_id=resource_id,
        kind=kind,
        uri=uri,
        sha256=sha256,
        manifest_hash=None,
        dataset_id="route-a",
        config_hash=HASH,
        slot_map_hash=HASH,
    )


def _content_hash(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _quality(**overrides: object):
    quality: dict[str, object] = {
        "sync_source": "corrected",
        "bearing_deg": 10.0,
        "distance_m": 8.0,
        "finite_vertices": 4,
        "projected_area_px": 1_200.0,
        "bbox_px": [20.0, 30.0, 80.0, 60.0],
        "visible_fraction": 0.95,
    }
    quality.update(overrides)
    return freeze_json(quality)


def _visual(
    visual_id: str,
    lidar_frame: int,
    camera_frame: int,
    timestamp: float,
    resource_id: str,
    *,
    capabilities: tuple[str, ...] = ("can_assess_occupied", "can_assess_free"),
    **quality: object,
) -> VisualFrameRef:
    return VisualFrameRef(
        visual_frame_id=visual_id,
        lidar_frame=lidar_frame,
        camera_frame=camera_frame,
        lidar_timestamp=timestamp,
        camera_timestamp=timestamp - 0.01,
        camera_lidar_dt_sec=-0.01,
        image_resource_id=resource_id,
        capabilities=capabilities,
        projection_quality=_quality(**quality),
    )


def _item(
    number: int,
    *,
    task_id: str | None = None,
    encounter_id: str = "encounter-a",
    conflicts: tuple[str, ...] = (),
    adjacent: tuple[str, ...] = (),
    shared: tuple[str, ...] = (),
    visuals: tuple[VisualFrameRef, ...] = (),
    pointclouds: tuple[str, ...] = (),
    anchor_frame: int = 100,
    anchor_timestamp: float = 10.0,
) -> QueueItem:
    slot_id = f"slot-{number}"
    return QueueItem(
        task_id=task_id or f"t{number}",
        slot_id=slot_id,
        scope_status="in_route_scope",
        state="unknown",
        agent_observable=True,
        unknown_reasons=("ambiguous",),
        priority="normal",
        available_modalities=tuple(
            modality
            for modality, present in (("lidar", bool(pointclouds)), ("rgb", bool(visuals)))
            if present
        ),
        suggested_tools=(),
        allowed_final_states=("occupied", "free", "unknown"),
        occupied_evidence=freeze_json({}),
        free_evidence=freeze_json({}),
        audit=freeze_json({}),
        relationships=RelationshipRef(
            adjacent_slot_ids=adjacent,
            conflict_slot_ids=conflicts,
            shared_evidence_ids=shared,
        ),
        encounter=EncounterRef(
            encounter_id=encounter_id,
            part1_trace_event_ids=(),
            start_lidar_frame=0,
            anchor_frame=anchor_frame,
            end_lidar_frame=anchor_frame,
            start_timestamp=0.0,
            anchor_timestamp=anchor_timestamp,
            end_timestamp=1_000.0,
            support_frames=(),
            pointcloud_resource_ids=pointclouds,
            visual_frames=visuals,
        ),
    )


def _queue(
    items: tuple[QueueItem, ...],
    resources: dict[str, ResourceRef] | None = None,
) -> QueueEnvelope:
    return QueueEnvelope(
        schema_version="unknown-agent-queue/1.0",
        queue_id="queue-stable-id",
        producer=freeze_json({"dataset_id": "route-a"}),
        resources=freeze_json(resources or {}),
        tool_registry_version="part2-tools/1.0",
        items=items,
    )


class DeterministicGroupingTest(unittest.TestCase):
    def test_task2_public_contracts_are_exported_from_the_package(self):
        import parking_slot_part2 as part2

        for name in (
            "CameraPreflight",
            "EvidenceCatalog",
            "TaskGroup",
            "ToolRegistry",
            "ToolResult",
            "build_groups",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(part2, name))

    def test_only_conflict_or_shared_evidence_edges_group_same_encounter_tasks(self):
        items = (
            _item(1, adjacent=("slot-2",)),
            _item(2, conflicts=("slot-3",)),
            _item(3),
            _item(4, shared=("shared-a",)),
            _item(5, shared=("shared-a",)),
            _item(6, encounter_id="encounter-b", conflicts=("slot-3",)),
        )

        groups = build_groups(_queue(items))
        task_sets = {group.task_ids for group in groups}

        self.assertIn(("t1",), task_sets, "plain adjacency must remain context only")
        self.assertIn(("t2", "t3"), task_sets, "one-sided conflict declarations form edges")
        self.assertIn(("t4", "t5"), task_sets, "intersecting shared evidence forms edges")
        self.assertIn(("t6",), task_sets, "edges may not cross encounter boundaries")
        self.assertTrue(all(len(group.task_ids) <= 4 for group in groups))

    def test_large_component_uses_deterministic_overlapping_edge_cover(self):
        items = (
            _item(1, conflicts=("slot-2",)),
            _item(2, conflicts=("slot-3",)),
            _item(3, conflicts=("slot-4",)),
            _item(4, conflicts=("slot-5",)),
            _item(5, conflicts=("slot-6",)),
            _item(6),
        )
        queue = _queue(items)

        groups = build_groups(queue)
        reordered = build_groups(replace(queue, items=tuple(reversed(items))))

        self.assertEqual(
            tuple(group.task_ids for group in groups),
            (("t1", "t2", "t3", "t4"), ("t4", "t5", "t6")),
        )
        self.assertEqual(groups, reordered)
        self.assertEqual(len({group.group_id for group in groups}), 2)
        self.assertTrue(all(group.grouping_policy_version == GROUPING_POLICY_VERSION for group in groups))


class PreflightAndToolsTest(unittest.TestCase):
    def test_camera_preflight_applies_capability_thresholds_and_caps_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources: dict[str, ResourceRef] = {}
            frames: list[VisualFrameRef] = []
            for index in range(7):
                resource_id = f"rgb-{index}"
                image = root / f"frame-{index}.png"
                image.write_bytes(b"declared-rgb-artifact")
                resources[resource_id] = _resource(resource_id, "rgb_frame", image.name)
                frames.append(_visual(f"vf-{index}", 90 + index, 300 + index, 9.0 + index / 10, resource_id))
            item = _item(1, visuals=tuple(frames), anchor_frame=100)

            catalog = EvidenceCatalog(_queue((item,), resources), base_dir=root)
            result = catalog.camera_preflight("t1")

        self.assertEqual(result.status, "ready")
        self.assertEqual(
            result.capabilities,
            ("can_assess_free", "can_assess_occupied"),
        )
        self.assertEqual(len(result.frame_evidence_ids), 5)
        self.assertIsNotNone(result.sequence_evidence_id)
        self.assertTrue(all(evidence_id.startswith("ev_") for evidence_id in result.frame_evidence_ids))

    def test_camera_preflight_enforces_every_geometry_sync_and_chronology_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.write_bytes(b"rgb")
            resources = {"rgb": _resource("rgb", "rgb_frame", image.name)}
            base = _visual("base", 100, 300, 10.0, "rgb")
            invalid = (
                replace(base, visual_frame_id="same-sensor", lidar_frame=80, camera_frame=80),
                replace(base, visual_frame_id="after-anchor", lidar_frame=99, camera_timestamp=10.1),
                replace(
                    base,
                    visual_frame_id="lidar-after-anchor-time",
                    lidar_frame=99,
                    lidar_timestamp=10.02,
                    camera_timestamp=9.99,
                    camera_lidar_dt_sec=-0.03,
                ),
                replace(base, visual_frame_id="bad-sync", projection_quality=_quality(sync_source="derived")),
                replace(base, visual_frame_id="bad-fov", projection_quality=_quality(bearing_deg=40.01)),
                replace(base, visual_frame_id="too-near", projection_quality=_quality(distance_m=2.99)),
                replace(base, visual_frame_id="too-far", projection_quality=_quality(distance_m=20.01)),
                replace(base, visual_frame_id="few-vertices", projection_quality=_quality(finite_vertices=2)),
                replace(base, visual_frame_id="small-area", projection_quality=_quality(projected_area_px=799.9)),
                replace(base, visual_frame_id="narrow-bbox", projection_quality=_quality(bbox_px=[0, 0, 39.9, 25])),
                replace(base, visual_frame_id="short-bbox", projection_quality=_quality(bbox_px=[0, 0, 50, 19.9])),
                replace(base, visual_frame_id="low-visible", projection_quality=_quality(visible_fraction=0.59)),
            )
            item = _item(1, visuals=invalid, anchor_frame=100)

            result = EvidenceCatalog(_queue((item,), resources), base_dir=root).camera_preflight("t1")

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.frame_evidence_ids, ())
        self.assertEqual(result.capabilities, ())
        self.assertTrue(result.rejections)
        self.assertIn(
            "after_anchor_timestamp",
            result.rejections["lidar-after-anchor-time"],
        )

    def test_camera_preflight_requires_passing_audit_for_bound_real_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.write_bytes(b"rgb")
            resources = {"rgb": _resource("rgb", "rgb_frame", image.name)}
            frame = _visual(
                "untrusted-real-frame",
                100,
                300,
                10.0,
                "rgb",
                calibration_sha256="sha256:" + "b" * 64,
                calibration_audit_status="untrusted",
            )
            item = _item(1, visuals=(frame,))

            result = EvidenceCatalog(_queue((item,), resources), base_dir=root).camera_preflight("t1")

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.capabilities, ())
        self.assertIn("calibration_untrusted", result.rejections["untrusted-real-frame"])

    def test_camera_preflight_grants_free_at_relaxed_visibility_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.write_bytes(b"rgb")
            resources = {"rgb": _resource("rgb", "rgb_frame", image.name)}
            frame = _visual("limited", 100, 300, 10.0, "rgb", visible_fraction=0.70)
            item = _item(1, visuals=(frame,))

            result = EvidenceCatalog(_queue((item,), resources), base_dir=root).camera_preflight("t1")

        self.assertEqual(result.status, "ready")
        self.assertEqual(result.capabilities, ("can_assess_free", "can_assess_occupied"))

    def test_camera_preflight_accepts_inclusive_boundaries_and_native_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.write_bytes(b"rgb")
            resources = {"rgb": _resource("rgb", "rgb_frame", image.name)}
            frame = _visual(
                "boundary-ready",
                100,
                300,
                10.0,
                "rgb",
                sync_source="native",
                bearing_deg=-40.0,
                distance_m=20.0,
                finite_vertices=3,
                projected_area_px=800.0,
                bbox_px=[0.0, 0.0, 40.0, 20.0],
                visible_fraction=0.60,
            )
            item = _item(1, visuals=(frame,))

            result = EvidenceCatalog(_queue((item,), resources), base_dir=root).camera_preflight("t1")

        self.assertEqual(result.status, "ready")
        self.assertEqual(result.capabilities, ("can_assess_free", "can_assess_occupied"))

    def test_camera_preflight_uses_declared_anchor_timestamp_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.write_bytes(b"rgb")
            resources = {"rgb": _resource("rgb", "rgb_frame", image.name)}
            frame = replace(
                _visual("after-declared-anchor", 90, 300, 10.01, "rgb"),
                camera_timestamp=10.0,
            )
            item = _item(1, visuals=(frame,), anchor_timestamp=9.5)

            result = EvidenceCatalog(_queue((item,), resources), base_dir=root).camera_preflight("t1")

        self.assertEqual(result.status, "unavailable")
        self.assertIn("after_anchor_timestamp", result.rejections["after-declared-anchor"])

    def test_catalog_ids_are_opaque_and_tools_are_exactly_allowlisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lidar = root / "map.npz"
            image = root / "frame.png"
            lidar.write_bytes(b"lidar")
            image.write_bytes(b"rgb")
            resources = {
                "lidar": _resource(
                    "lidar",
                    "lidar_map",
                    lidar.name,
                    sha256=_content_hash(b"lidar"),
                ),
                "rgb": _resource(
                    "rgb",
                    "rgb_frame",
                    image.name,
                    sha256=_content_hash(b"rgb"),
                ),
            }
            frame = _visual("ready", 100, 300, 10.0, "rgb")
            catalog = EvidenceCatalog(
                _queue((_item(1, visuals=(frame,), pointclouds=("lidar",)),), resources),
                base_dir=root,
            )
            registry = ToolRegistry(catalog)

            self.assertEqual(
                registry.names,
                ("inspect_lidar_map", "inspect_rgb_frame", "inspect_rgb_sequence"),
            )
            self.assertTrue(catalog.evidence_ids)
            self.assertTrue(all("/" not in item and "\\" not in item for item in catalog.evidence_ids))
            self.assertTrue(all(item.startswith("ev_") for item in catalog.evidence_ids))

            lidar_id = next(
                evidence_id
                for evidence_id in catalog.evidence_ids
                if catalog.get(evidence_id).tool_name == "inspect_lidar_map"
            )
            first = registry.execute("inspect_lidar_map", {"evidence_id": lidar_id})
            second = registry.execute("inspect_lidar_map", {"evidence_id": lidar_id})
            self.assertEqual(first, second)
            self.assertEqual(first.status, "unavailable")
            self.assertEqual(
                first.error_code,
                "unsupported_lidar_evidence_format",
            )
            self.assertNotIn("uri", first.data)
            self.assertNotIn("path", first.data)
            self.assertEqual(
                json.loads(json.dumps(first.to_dict()))["status"],
                "unavailable",
            )

            arbitrary_path = registry.execute("inspect_rgb_frame", {"path": "/tmp/frame.png"})
            self.assertEqual(arbitrary_path.status, "invalid_arguments")
            unknown_tool = registry.execute("read_file", {"evidence_id": lidar_id})
            self.assertEqual(unknown_tool.status, "invalid_tool")

    def test_tool_rehashes_file_immediately_before_returning_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "map.npz"
            artifact.write_bytes(b"declared-content")
            resources = {
                "lidar": _resource(
                    "lidar",
                    "lidar_map",
                    artifact.name,
                    sha256=_content_hash(b"declared-content"),
                )
            }
            catalog = EvidenceCatalog(
                _queue((_item(1, pointclouds=("lidar",)),), resources),
                base_dir=root,
            )
            registry = ToolRegistry(catalog)
            evidence_id = catalog.evidence_ids[0]
            artifact.write_bytes(b"tampered-after-preflight")

            result = registry.execute("inspect_lidar_map", {"evidence_id": evidence_id})

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.error_code, "artifact_hash_mismatch")

    def test_missing_artifacts_return_structured_results_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = {"lidar": _resource("lidar", "lidar_map", "missing.npz")}
            catalog = EvidenceCatalog(
                _queue((_item(1, pointclouds=("lidar",)),), resources),
                base_dir=root,
            )
            registry = ToolRegistry(catalog)
            evidence_id = catalog.evidence_ids[0]

            result = registry.execute("inspect_lidar_map", {"evidence_id": evidence_id})

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.error_code, "artifact_missing")
        self.assertEqual(result.data["evidence_id"], evidence_id)

    def test_artifact_access_errors_return_structured_results_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources = {
                "lidar": _resource("lidar", "lidar_map", "x" * 5_000),
            }

            catalog = EvidenceCatalog(
                _queue((_item(1, pointclouds=("lidar",)),), resources),
                base_dir=tmp,
            )
            registry = ToolRegistry(catalog)
            result = registry.execute(
                "inspect_lidar_map",
                {"evidence_id": catalog.evidence_ids[0]},
            )

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.error_code, "artifact_access_failed")

    def test_preflight_ids_are_bound_to_one_task_and_encounter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.write_bytes(b"rgb")
            resources = {"rgb": _resource("rgb", "rgb_frame", image.name)}
            frame = _visual("same-public-frame-id", 100, 300, 10.0, "rgb")
            first = _item(1, visuals=(frame,), encounter_id="encounter-a")
            second = _item(2, visuals=(frame,), encounter_id="encounter-b")
            catalog = EvidenceCatalog(_queue((first, second), resources), base_dir=root)

            first_ids = catalog.camera_preflight("t1").frame_evidence_ids
            second_ids = catalog.camera_preflight("t2").frame_evidence_ids

        self.assertTrue(first_ids)
        self.assertTrue(second_ids)
        self.assertTrue(set(first_ids).isdisjoint(second_ids))
        self.assertEqual(catalog.get(first_ids[0]).encounter_id, "encounter-a")
        self.assertEqual(catalog.get(second_ids[0]).encounter_id, "encounter-b")


if __name__ == "__main__":
    unittest.main()
