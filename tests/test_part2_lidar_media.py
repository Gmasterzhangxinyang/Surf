from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from parking_slot_hybrid_3d.contracts import (
    FrameObservation,
    GroundModel,
    KnownSlot,
    SlotAccumulation,
)
from parking_slot_hybrid_3d.geometry import metric_slot
from parking_slot_hybrid_3d.part2_evidence import (
    PACK_ARRAY_KEYS,
    build_lidar_evidence_pack,
)
from parking_slot_part2.contracts import (
    EncounterRef,
    QueueEnvelope,
    QueueItem,
    RelationshipRef,
    ResourceRef,
    freeze_json,
)
from parking_slot_part2.media import (
    EvidenceMediaStore,
    EvidencePackError,
    load_lidar_evidence_pack,
    render_lidar_triptych,
)
from parking_slot_part2.preflight import EvidenceCatalog
from parking_slot_part2.tools import ToolRegistry


IDENTITY = "sha256:" + "a" * 64


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _slot():
    known = KnownSlot(
        slot_id="slot_0007",
        polygon_map=np.asarray([[-2.2, -1.0], [2.2, -1.0], [2.2, 1.0], [-2.2, 1.0]]),
        core_polygon_map=np.asarray([[-1.8, -0.7], [1.8, -0.7], [1.8, 0.7], [-1.8, 0.7]]),
        margin_polygon_map=np.asarray([[-2.6, -1.4], [2.6, -1.4], [2.6, 1.4], [-2.6, 1.4]]),
        center_map=np.asarray([0.0, 0.0]),
        heading_deg=0.0,
    )
    return metric_slot(known, 1.0)


def _accumulation() -> SlotAccumulation:
    ground = GroundModel(method="plane", valid=True)
    observations = (
        FrameObservation(
            frame_id=10,
            origin_local_xyz=np.asarray([-5.0, 0.0, 1.5]),
            points_local_xyzi=np.empty((0, 4)),
            ray_endpoints_local_xyz=np.empty((0, 3)),
            ground_model=ground,
        ),
        FrameObservation(
            frame_id=14,
            origin_local_xyz=np.asarray([-3.0, 1.0, 1.5]),
            points_local_xyzi=np.empty((0, 4)),
            ray_endpoints_local_xyz=np.empty((0, 3)),
            ground_model=ground,
        ),
    )
    # Deliberately non-canonical order: the pack builder must canonicalize it.
    points = np.asarray(
        [
            [1.1, 0.3, 1.6, 4.0],
            [-1.0, -0.2, 0.05, 1.0],
            [0.2, 0.1, 0.8, 2.0],
            [-0.4, 0.4, 1.2, 3.0],
        ],
        dtype=np.float64,
    )
    return SlotAccumulation(
        slot_id="slot_0007",
        anchor_frame=10,
        selected_frames=(10, 12, 14),
        points_local_xyzi=points,
        point_frame_ids=np.asarray([14, 10, 10, 14], dtype=np.int64),
        observations=observations,
        excluded_frames=((12, ("invalid_ground_model",)),),
    )


def _write_sources(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for frame_id in (10, 12, 14):
        path = root / f"{frame_id:06d}.npz"
        path.write_bytes(f"source-frame-{frame_id}".encode("ascii"))
        result[frame_id] = path
    return result


def _build(root: Path, name: str = "evidence.npz"):
    sources = _write_sources(root)
    artifact = build_lidar_evidence_pack(
        root / name,
        accumulation=_accumulation(),
        slot=_slot(),
        source_paths=sources,
        task_id="task-a",
        encounter_id="encounter-a",
        dataset_id="route-a",
        config_hash=IDENTITY,
        slot_map_hash=IDENTITY,
        adjacent_polygons_local_m={
            "slot_0008": np.asarray(
                [[-2.2, 1.1], [2.2, 1.1], [2.2, 3.1], [-2.2, 3.1]]
            )
        },
    )
    return artifact, sources


def _queue(pack_path: Path, pack_sha256: str) -> QueueEnvelope:
    resource = ResourceRef(
        resource_id="lidar-pack",
        kind="pointcloud_artifact",
        uri=pack_path.name,
        sha256=pack_sha256,
        manifest_hash=None,
        dataset_id="route-a",
        config_hash=IDENTITY,
        slot_map_hash=IDENTITY,
    )
    item = QueueItem(
        task_id="task-a",
        slot_id="slot_0007",
        scope_status="in_route_scope",
        state="unknown",
        agent_observable=True,
        unknown_reasons=("ambiguous",),
        priority="normal",
        available_modalities=("lidar",),
        suggested_tools=("inspect_lidar_map",),
        allowed_final_states=("occupied", "free", "unknown"),
        occupied_evidence=freeze_json({}),
        free_evidence=freeze_json({}),
        audit=freeze_json({}),
        relationships=RelationshipRef((), (), ()),
        encounter=EncounterRef(
            encounter_id="encounter-a",
            part1_trace_event_ids=(),
            start_lidar_frame=10,
            anchor_frame=10,
            end_lidar_frame=14,
            start_timestamp=1.0,
            anchor_timestamp=2.0,
            end_timestamp=3.0,
            support_frames=(10, 12, 14),
            pointcloud_resource_ids=("lidar-pack",),
            visual_frames=(),
        ),
    )
    return QueueEnvelope(
        schema_version="unknown-agent-queue/1.0",
        queue_id="queue-a",
        producer=freeze_json({"dataset_id": "route-a"}),
        resources=freeze_json({"lidar-pack": resource}),
        tool_registry_version="part2-tools/1.0",
        items=(item,),
    )


class LidarEvidencePackTest(unittest.TestCase):
    def test_pack_and_render_are_byte_deterministic_and_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, _ = _build(root, "first.npz")
            second, _ = _build(root, "second.npz")

            self.assertEqual(first.sha256, second.sha256)
            self.assertEqual(first.path.read_bytes(), second.path.read_bytes())
            self.assertNotIn(str(root).encode("utf-8"), first.path.read_bytes())

            pack = load_lidar_evidence_pack(
                first.path,
                expected_sha256=first.sha256,
                expected_slot_id="slot_0007",
            )
            first_png = render_lidar_triptych(pack)
            second_png = render_lidar_triptych(pack)

        self.assertEqual(first_png, second_png)
        with Image.open(io.BytesIO(first_png)) as image:
            self.assertEqual(image.size, (1536, 512))
            self.assertEqual(image.mode, "RGB")
            self.assertGreater(len(image.getcolors(maxcolors=1_000_000) or []), 5)
        self.assertEqual(pack.points_local_xyzi.dtype, np.dtype(np.float32))
        self.assertEqual(pack.point_frame_ids.dtype, np.dtype(np.int32))
        self.assertFalse(pack.points_local_xyzi.flags.writeable)
        self.assertEqual(tuple(pack.selected_frames.tolist()), (10, 12, 14))
        self.assertEqual(tuple(pack.valid_frames.tolist()), (10, 14))

    def test_source_content_identity_propagates_to_pack_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, sources = _build(root, "first.npz")
            sources[10].write_bytes(b"changed-source-frame")
            second = build_lidar_evidence_pack(
                root / "second.npz",
                accumulation=_accumulation(),
                slot=_slot(),
                source_paths=sources,
                task_id="task-a",
                encounter_id="encounter-a",
                dataset_id="route-a",
                config_hash=IDENTITY,
                slot_map_hash=IDENTITY,
            )

        self.assertNotEqual(first.sha256, second.sha256)

    def test_builder_rejects_missing_source_and_nonfinite_float32_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = _write_sources(root)
            sources.pop(12)
            with self.assertRaisesRegex(ValueError, "no concrete source"):
                build_lidar_evidence_pack(
                    root / "missing.npz",
                    accumulation=_accumulation(),
                    slot=_slot(),
                    source_paths=sources,
                    task_id="task-a",
                    encounter_id="encounter-a",
                    dataset_id="route-a",
                    config_hash=IDENTITY,
                    slot_map_hash=IDENTITY,
                )

            broken = _accumulation()
            broken.points_local_xyzi[0, 0] = np.inf
            with self.assertRaisesRegex(ValueError, "non-finite"):
                build_lidar_evidence_pack(
                    root / "nonfinite.npz",
                    accumulation=broken,
                    slot=_slot(),
                    source_paths=_write_sources(root),
                    task_id="task-a",
                    encounter_id="encounter-a",
                    dataset_id="route-a",
                    config_hash=IDENTITY,
                    slot_map_hash=IDENTITY,
                )

    def test_loader_rejects_wrong_dtype_and_archive_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact, _ = _build(root)
            with np.load(artifact.path, allow_pickle=False) as payload:
                arrays = {name: np.array(payload[name], copy=True) for name in PACK_ARRAY_KEYS}
            arrays["points_local_xyzi"] = arrays["points_local_xyzi"].astype(np.float64)
            wrong_dtype = root / "wrong-dtype.npz"
            np.savez(wrong_dtype, **arrays)

            with self.assertRaisesRegex(EvidencePackError, "exact dtype"):
                load_lidar_evidence_pack(wrong_dtype)
            with self.assertRaisesRegex(EvidencePackError, "sha256"):
                load_lidar_evidence_pack(artifact.path, expected_sha256=_sha256(b"wrong"))

    def test_task_and_encounter_identity_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact, _ = _build(root)
            with self.assertRaisesRegex(EvidencePackError, "task_id"):
                load_lidar_evidence_pack(artifact.path, expected_task_id="another-task")
            with self.assertRaisesRegex(EvidencePackError, "encounter_id"):
                load_lidar_evidence_pack(
                    artifact.path,
                    expected_encounter_id="another-encounter",
                )


class LidarMediaToolBridgeTest(unittest.TestCase):
    def test_media_is_published_only_after_successful_opaque_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact, _ = _build(root)
            catalog = EvidenceCatalog(_queue(artifact.path, artifact.sha256), base_dir=root)
            store = EvidenceMediaStore(root / "media")
            registry = ToolRegistry(catalog, media_store=store)
            evidence_id = catalog.evidence_ids[0]

            self.assertIsNone(store.resolve(evidence_id))
            result = registry.execute("inspect_lidar_map", {"evidence_id": evidence_id})
            media = store.resolve(evidence_id)

            self.assertEqual(result.status, "ok")
            self.assertIsNotNone(media)
            assert media is not None
            self.assertTrue(media.path.is_file())
            self.assertEqual(media.sha256, _sha256(media.path.read_bytes()))
            self.assertNotIn("path", json.dumps(result.to_dict(), sort_keys=True))
            self.assertNotIn(".png", json.dumps(result.to_dict(), sort_keys=True))
            manifest = store.manifest_records()
            self.assertEqual(len(manifest), 1)
            self.assertNotIn("path", manifest[0])

            replay_result = ToolRegistry(catalog).execute(
                "inspect_lidar_map",
                {"evidence_id": evidence_id},
            )
            self.assertEqual(result, replay_result)

    def test_legacy_pointcloud_npz_is_not_reported_as_visual_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.npz"
            np.savez(raw, points_map_xyzi=np.zeros((2, 4), dtype=np.float32))
            queue = _queue(raw, _sha256(raw.read_bytes()))
            catalog = EvidenceCatalog(queue, base_dir=root)
            store = EvidenceMediaStore(root / "media")
            registry = ToolRegistry(catalog, media_store=store)
            evidence_id = catalog.evidence_ids[0]

            result = registry.execute("inspect_lidar_map", {"evidence_id": evidence_id})
            replay_result = ToolRegistry(catalog).execute(
                "inspect_lidar_map",
                {"evidence_id": evidence_id},
            )

            self.assertEqual(result.status, "unavailable")
            self.assertEqual(result, replay_result)
            self.assertEqual(
                result.error_code,
                "unsupported_lidar_evidence_format",
            )
            self.assertIsNone(store.resolve(evidence_id))

    def test_advertised_but_invalid_pack_fails_closed_without_media_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact, _ = _build(root)
            with np.load(artifact.path, allow_pickle=False) as payload:
                arrays = {name: np.array(payload[name], copy=True) for name in PACK_ARRAY_KEYS}
            arrays["points_local_xyzi"] = arrays["points_local_xyzi"].astype(np.float64)
            invalid = root / "invalid-pack.npz"
            np.savez(invalid, **arrays)
            queue = _queue(invalid, _sha256(invalid.read_bytes()))
            catalog = EvidenceCatalog(queue, base_dir=root)
            store = EvidenceMediaStore(root / "media")
            registry = ToolRegistry(catalog, media_store=store)
            evidence_id = catalog.evidence_ids[0]

            result = registry.execute("inspect_lidar_map", {"evidence_id": evidence_id})
            replay_result = ToolRegistry(catalog).execute(
                "inspect_lidar_map",
                {"evidence_id": evidence_id},
            )

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.error_code, "evidence_media_unavailable")
            self.assertEqual(result, replay_result)
            self.assertIsNone(store.resolve(evidence_id))


if __name__ == "__main__":
    unittest.main()
