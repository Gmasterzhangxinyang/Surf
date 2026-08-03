from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from parking_slot_part2.contracts import (
    EncounterRef,
    QueueEnvelope,
    QueueItem,
    RelationshipRef,
    ResourceRef,
    VisualFrameRef,
    freeze_json,
)
from parking_slot_part2.media import (
    EvidenceMediaError,
    EvidenceMediaStore,
    RGB_SEQUENCE_MAX_FRAMES,
    RgbFrameMediaInput,
    render_rgb_frame,
)
from parking_slot_part2.preflight import EvidenceCatalog
from parking_slot_part2.tools import ToolRegistry


IDENTITY = "sha256:" + "a" * 64


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _quality(**overrides: object):
    value: dict[str, object] = {
        "sync_source": "corrected",
        "bearing_deg": 8.0,
        "distance_m": 8.0,
        "finite_vertices": 4,
        "projected_area_px": 6_000.0,
        "bbox_px": [30.0, 40.0, 130.0, 100.0],
        "visible_fraction": 0.95,
        "image_width_px": 240,
        "image_height_px": 140,
        "polygon_uv": [[30.0, 40.0], [130.0, 40.0], [130.0, 100.0], [30.0, 100.0]],
        # Partially off-screen polygons are legal and are clipped for display.
        "adjacent_polygons_uv": {
            "slot-2": [[205.0, 45.0], [280.0, 45.0], [280.0, 105.0], [205.0, 105.0]],
        },
    }
    value.update(overrides)
    return freeze_json(value)


def _resource(resource_id: str, path: Path) -> ResourceRef:
    return ResourceRef(
        resource_id=resource_id,
        kind="rgb_frame",
        uri=path.name,
        sha256=_sha256(path.read_bytes()),
        manifest_hash=None,
        dataset_id="route-a",
        config_hash=IDENTITY,
        slot_map_hash=IDENTITY,
    )


def _environment(
    root: Path,
    qualities: tuple[object, ...],
) -> tuple[EvidenceCatalog, EvidenceMediaStore, ToolRegistry, tuple[Path, ...]]:
    resources: dict[str, ResourceRef] = {}
    visuals: list[VisualFrameRef] = []
    paths: list[Path] = []
    colors = ((24, 48, 72), (112, 136, 160), (48, 92, 36), (160, 80, 40), (70, 30, 130))
    for index, quality in enumerate(qualities):
        path = root / f"private-source-{index}.png"
        Image.new("RGB", (240, 140), colors[index % len(colors)]).save(path, format="PNG")
        paths.append(path)
        resource_id = f"rgb-{index}"
        resources[resource_id] = _resource(resource_id, path)
        visuals.append(
            VisualFrameRef(
                visual_frame_id=f"visual-{index}",
                lidar_frame=90 + index,
                camera_frame=300 + index,
                lidar_timestamp=9.0 + index * 0.1,
                camera_timestamp=8.99 + index * 0.1,
                camera_lidar_dt_sec=-0.01,
                image_resource_id=resource_id,
                capabilities=("can_assess_occupied", "can_assess_free"),
                projection_quality=quality,
            )
        )
    item = QueueItem(
        task_id="task-a",
        slot_id="slot-1",
        scope_status="in_route_scope",
        state="unknown",
        agent_observable=True,
        unknown_reasons=("ambiguous",),
        priority="normal",
        available_modalities=("rgb",),
        suggested_tools=("inspect_rgb_frame", "inspect_rgb_sequence"),
        allowed_final_states=("occupied", "free", "unknown"),
        occupied_evidence=freeze_json({}),
        free_evidence=freeze_json({}),
        audit=freeze_json({}),
        relationships=RelationshipRef((), (), ()),
        encounter=EncounterRef(
            encounter_id="encounter-a",
            part1_trace_event_ids=(),
            start_lidar_frame=0,
            anchor_frame=100,
            end_lidar_frame=100,
            start_timestamp=0.0,
            anchor_timestamp=20.0,
            end_timestamp=20.0,
            support_frames=(),
            pointcloud_resource_ids=(),
            visual_frames=tuple(visuals),
        ),
    )
    queue = QueueEnvelope(
        schema_version="unknown-agent-queue/1.0",
        queue_id="queue-a",
        producer=freeze_json({"dataset_id": "route-a"}),
        resources=freeze_json(resources),
        tool_registry_version="part2-tools/1.0",
        items=(item,),
    )
    catalog = EvidenceCatalog(queue, base_dir=root)
    store = EvidenceMediaStore(root / "media")
    return catalog, store, ToolRegistry(catalog, media_store=store), tuple(paths)


def _direct_frame(path: Path, *, index: int = 0, **overrides: object) -> RgbFrameMediaInput:
    values: dict[str, object] = {
        "source_path": path,
        "source_sha256": _sha256(path.read_bytes()),
        "resource_id": f"rgb-{index}",
        "task_id": "task-a",
        "slot_id": "slot-1",
        "encounter_id": "encounter-a",
        "visual_frame_id": f"visual-{index}",
        "lidar_frame": 90 + index,
        "camera_frame": 300 + index,
        "camera_timestamp": 8.99 + index * 0.1,
        "polygon_uv": _quality()["polygon_uv"],
        "adjacent_polygons_uv": _quality()["adjacent_polygons_uv"],
        "expected_width": 240,
        "expected_height": 140,
    }
    values.update(overrides)
    return RgbFrameMediaInput(**values)  # type: ignore[arg-type]


class RgbMediaToolBridgeTest(unittest.TestCase):
    def test_rgb_decode_uses_the_exact_bytes_that_were_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            Image.new("RGB", (240, 140), (20, 40, 60)).save(source)
            frame = _direct_frame(source)
            original_hash = frame.source_sha256

            def hash_then_replace(content: bytes) -> str:
                source.unlink()
                Image.new("RGB", (240, 140), (200, 210, 220)).save(source)
                return _sha256(content)

            with patch(
                "parking_slot_part2.media._bytes_sha256",
                side_effect=hash_then_replace,
            ):
                content, _, _, source_sha256 = render_rgb_frame(frame)

            self.assertEqual(source_sha256, original_hash)
            with Image.open(io.BytesIO(content)) as rendered:
                # Bottom-right is outside both target and adjacent overlays.
                self.assertEqual(rendered.convert("RGB").getpixel((239, 139)), (20, 40, 60))

    def test_frame_is_annotated_content_addressed_and_path_private(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, store, registry, sources = _environment(root, (_quality(),))
            evidence_id = catalog.camera_preflight("task-a").frame_evidence_ids[0]
            original = sources[0].read_bytes()

            self.assertIsNone(store.resolve(evidence_id))
            result = registry.execute("inspect_rgb_frame", {"evidence_id": evidence_id})
            artifact = store.resolve(evidence_id)
            safe_media = store.read_image(evidence_id)

            self.assertEqual(result.status, "ok")
            self.assertIsNotNone(artifact)
            self.assertIsNotNone(safe_media)
            assert artifact is not None and safe_media is not None
            media_type, content = safe_media
            self.assertEqual(media_type, "image/png")
            self.assertEqual(content, artifact.path.read_bytes())
            self.assertNotEqual(content, original)
            self.assertEqual(artifact.kind, "rgb_target_frame_png")
            self.assertEqual((artifact.width, artifact.height), (240, 140))
            self.assertEqual(artifact.sha256, _sha256(content))
            self.assertEqual(sources[0].read_bytes(), original)
            self.assertEqual(artifact.path.name, artifact.sha256.split(":", 1)[1] + ".png")
            rendered = Image.open(io.BytesIO(content)).convert("RGB")
            raw_pixels = rendered.tobytes()
            pixels = tuple(zip(raw_pixels[0::3], raw_pixels[1::3], raw_pixels[2::3]))
            self.assertTrue(any(red > 240 and green < 80 and blue > 180 for red, green, blue in pixels))
            self.assertTrue(any(red > 190 and green > 145 and blue < 100 for red, green, blue in pixels))
            public = json.dumps(result.to_dict(), sort_keys=True)
            self.assertNotIn(str(root), public)
            self.assertNotIn(sources[0].name, public)
            manifest = store.manifest_records()[0]
            self.assertNotIn("path", manifest)

    def test_sequence_is_an_ordered_deterministic_contact_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, store, registry, _ = _environment(root, (_quality(), _quality()))
            evidence_id = catalog.camera_preflight("task-a").sequence_evidence_id
            assert evidence_id is not None

            first = registry.execute("inspect_rgb_sequence", {"evidence_id": evidence_id})
            first_artifact = store.resolve(evidence_id)
            second = registry.execute("inspect_rgb_sequence", {"evidence_id": evidence_id})
            second_artifact = store.resolve(evidence_id)

            self.assertEqual(first.status, "ok")
            self.assertEqual(second.status, "ok")
            self.assertIsNotNone(first_artifact)
            self.assertEqual(first_artifact, second_artifact)
            assert first_artifact is not None
            self.assertEqual(first_artifact.kind, "rgb_target_sequence_contact_sheet_png")
            self.assertEqual((first_artifact.width, first_artifact.height), (1288, 400))
            self.assertEqual(store.read_image(evidence_id)[0], "image/png")  # type: ignore[index]

    def test_missing_invalid_and_fully_out_of_bounds_targets_fail_closed_with_stable_codes(self) -> None:
        cases = (
            (None, None, "rgb_target_polygon_missing"),
            (
                "polygon_uv",
                [[250.0, 40.0], [300.0, 40.0], [300.0, 100.0], [250.0, 100.0]],
                "rgb_target_polygon_out_of_bounds",
            ),
            (
                "polygon_uv",
                [[30.0, 40.0], [130.0, float("nan")], [130.0, 100.0], [30.0, 100.0]],
                "rgb_target_polygon_invalid",
            ),
            ("image_width_px", 241, "rgb_image_dimensions_mismatch"),
        )
        for field, value, expected_code in cases:
            with self.subTest(expected_code=expected_code), tempfile.TemporaryDirectory() as tmp:
                overrides = {} if field is None else {field: value}
                if field is None:
                    overrides["polygon_uv"] = value
                quality = _quality(**overrides)
                root = Path(tmp)
                catalog, store, registry, _ = _environment(root, (quality,))
                evidence_id = catalog.camera_preflight("task-a").frame_evidence_ids[0]

                result = registry.execute("inspect_rgb_frame", {"evidence_id": evidence_id})

                self.assertEqual(result.status, "failed")
                self.assertEqual(result.error_code, expected_code)
                self.assertIsNone(store.resolve(evidence_id))

    def test_partial_target_and_bad_adjacent_polygons_render_without_weakening_target_gate(self) -> None:
        adjacent = {
            "partial": [[205.0, 45.0], [280.0, 45.0], [280.0, 105.0], [205.0, 105.0]],
            "missing": None,
            "non-finite": [[10.0, 10.0], [20.0, float("nan")], [20.0, 20.0]],
            "degenerate": [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
            "outside": [[250.0, 40.0], [300.0, 40.0], [300.0, 100.0], [250.0, 100.0]],
        }
        quality = _quality(
            polygon_uv=[[-30.0, 40.0], [130.0, 40.0], [130.0, 100.0], [-30.0, 100.0]],
            adjacent_polygons_uv=adjacent,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, store, registry, _ = _environment(root, (quality,))
            evidence_id = catalog.camera_preflight("task-a").frame_evidence_ids[0]

            with_store = registry.execute("inspect_rgb_frame", {"evidence_id": evidence_id})
            without_store = ToolRegistry(catalog).execute(
                "inspect_rgb_frame",
                {"evidence_id": evidence_id},
            )

            self.assertEqual(with_store, without_store)
            self.assertEqual(with_store.status, "ok")
            self.assertIsNotNone(store.resolve(evidence_id))

    def test_registry_runs_the_same_rgb_validation_with_and_without_media_store(self) -> None:
        quality = _quality(
            polygon_uv=[[250.0, 40.0], [300.0, 40.0], [300.0, 100.0], [250.0, 100.0]],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, store, registry, _ = _environment(root, (quality,))
            evidence_id = catalog.camera_preflight("task-a").frame_evidence_ids[0]

            with_store = registry.execute("inspect_rgb_frame", {"evidence_id": evidence_id})
            without_store = ToolRegistry(catalog).execute(
                "inspect_rgb_frame",
                {"evidence_id": evidence_id},
            )

            self.assertEqual(with_store, without_store)
            self.assertEqual(with_store.status, "failed")
            self.assertEqual(with_store.error_code, "rgb_target_polygon_out_of_bounds")
            self.assertIsNone(store.resolve(evidence_id))

    def test_tool_fails_in_the_same_call_when_verified_media_cannot_be_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, _, _, _ = _environment(root, (_quality(),))
            evidence_id = catalog.camera_preflight("task-a").frame_evidence_ids[0]
            blocked_root = root / "blocked-media-root"
            blocked_root.write_bytes(b"not-a-directory")
            store = EvidenceMediaStore(blocked_root)

            result = ToolRegistry(catalog, media_store=store).execute(
                "inspect_rgb_frame",
                {"evidence_id": evidence_id},
            )

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.error_code, "rgb_media_unavailable")
            self.assertIsNone(store.resolve(evidence_id))

    def test_sequence_limits_and_identities_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            Image.new("RGB", (240, 140), (20, 40, 60)).save(source)
            frame = _direct_frame(source)
            store = EvidenceMediaStore(root / "media")
            evidence_id = "ev_" + "b" * 64

            with self.assertRaises(EvidenceMediaError) as too_many:
                store.publish_rgb_sequence(
                    evidence_id,
                    tuple(replace(frame, visual_frame_id=f"visual-{index}") for index in range(6)),
                )
            self.assertEqual(too_many.exception.code, "rgb_sequence_invalid")
            self.assertEqual(RGB_SEQUENCE_MAX_FRAMES, 5)

            other_slot = replace(
                frame,
                resource_id="rgb-1",
                visual_frame_id="visual-1",
                camera_frame=301,
                camera_timestamp=9.09,
                slot_id="slot-other",
            )
            with self.assertRaises(EvidenceMediaError) as wrong_identity:
                store.publish_rgb_sequence(evidence_id, (frame, other_slot))
            self.assertEqual(wrong_identity.exception.code, "rgb_sequence_identity_mismatch")
            self.assertIsNone(store.resolve(evidence_id))

    def test_source_identity_mismatch_and_bound_media_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            Image.new("RGB", (240, 140), (20, 40, 60)).save(source)
            store = EvidenceMediaStore(root / "media")
            bad_identity = _direct_frame(source, source_sha256="sha256:" + "0" * 64)
            with self.assertRaises(EvidenceMediaError) as mismatch:
                store.publish_rgb_frame("ev_" + "c" * 64, bad_identity)
            self.assertEqual(mismatch.exception.code, "rgb_source_identity_mismatch")

            evidence_id = "ev_" + "d" * 64
            artifact = store.publish_rgb_frame(evidence_id, _direct_frame(source))
            artifact.path.write_bytes(artifact.path.read_bytes() + b"tampered")

            self.assertIsNone(store.read_image(evidence_id))
            self.assertIsNone(store.resolve(evidence_id))

    def test_tool_rehashes_rgb_before_publish_and_rejects_unsafe_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, store, registry, sources = _environment(root, (_quality(),))
            evidence_id = catalog.camera_preflight("task-a").frame_evidence_ids[0]
            sources[0].write_bytes(b"changed after preflight")

            result = registry.execute("inspect_rgb_frame", {"evidence_id": evidence_id})

            self.assertEqual(result.status, "unavailable")
            self.assertEqual(result.error_code, "artifact_hash_mismatch")
            self.assertIsNone(store.resolve(evidence_id))

            tiny = root / "tiny.png"
            Image.new("RGB", (1, 1), (0, 0, 0)).save(tiny)
            tiny_frame = _direct_frame(
                tiny,
                expected_width=1,
                expected_height=1,
                polygon_uv=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            )
            with self.assertRaises(EvidenceMediaError) as invalid_dimensions:
                store.publish_rgb_frame("ev_" + "e" * 64, tiny_frame)
            self.assertEqual(
                invalid_dimensions.exception.code,
                "rgb_image_dimensions_invalid",
            )


if __name__ == "__main__":
    unittest.main()
