from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from PIL import Image

from parking_slot_hybrid_3d.camera import (
    CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION,
    LEGACY_CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION,
    load_camera_model,
    load_projection_audit,
)
from parking_slot_hybrid_3d.projection_audit import (
    CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION,
    ProjectionAuditThresholds,
    build_projection_audit,
    write_projection_audit,
)
from scripts.build_camera_projection_audit import main as audit_cli_main


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _calibration_payload() -> dict[str, Any]:
    return {
        "schema_version": "camera-calibration/1.0",
        "camera_id": "synthetic-test-camera",
        "image_size_px": [300, 180],
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 150.0, "cy": 90.0},
        "camera_from_lidar": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "distortion": {"model": "none", "coefficients": []},
    }


def _point_for_pixel(u: float, v: float, depth: float = 10.0) -> list[float]:
    return [(u - 150.0) * depth / 100.0, (v - 90.0) * depth / 100.0, depth]


def _reference_set(
    root: Path,
    calibration_sha256: str,
    *,
    offsets: dict[str, float] | None = None,
    include_references: bool = True,
) -> dict[str, Any]:
    offsets = offsets or {}
    pixels = {"left": (45.0, 90.0), "center": (150.0, 90.0), "right": (255.0, 90.0)}
    references: list[dict[str, Any]] = []
    for zone, (base_u, base_v) in pixels.items():
        for sample_index in range(5):
            u = base_u + float(sample_index - 2)
            v = base_v + float(2 * (sample_index - 2))
            reference: dict[str, Any] = {
                "id": f"{zone}-real-pixel-{sample_index}",
                "image_id": "camera-frame-42",
                "point_source_id": "lidar-frame-42",
                "observed_pixel_uv": [u + offsets.get(zone, 0.0), v],
            }
            if zone == "center":
                reference["point_map_xyz_m"] = _point_for_pixel(u, v)
            else:
                reference["point_lidar_xyz_m"] = _point_for_pixel(u, v)
            references.append(reference)
    image_path = root / "camera-frame-42.png"
    Image.new("RGB", (300, 180), color=(24, 48, 72)).save(image_path, format="PNG")
    image_bytes = image_path.read_bytes()
    annotation_path = root / "independent-annotation-export.json"
    annotation_path.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "image_id": reference["image_id"],
                        "observed_pixel_uv": reference["observed_pixel_uv"],
                        "point_source_id": reference["point_source_id"],
                        "reference_id": reference["id"],
                    }
                    for reference in references
                ],
                "dataset_id": "route-a",
                "exporter": "independent-test-fixture",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    annotation_bytes = annotation_path.read_bytes()
    lidar_path = root / "lidar-frame-42.bin"
    lidar_path.write_bytes(b"synthetic-lidar-frame-42\x00\x01\x02")
    lidar_bytes = lidar_path.read_bytes()
    pose_path = root / "pose-frame-42.json"
    pose_path.write_text(
        json.dumps(
            {
                "schema_version": "camera-projection-pose-source/1.0",
                "lidar_frame": 42,
                "lidar_timestamp_sec": 1234.48,
                "lidar_from_map": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pose_bytes = pose_path.read_bytes()
    payload: dict[str, Any] = {
        "schema_version": CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION,
        "dataset_id": "route-a",
        "calibration_sha256": calibration_sha256,
        "observation_source": {
            "kind": "manual_pixel_labels",
            "artifact_uri": annotation_path.name,
            "artifact_sha256": _sha256(annotation_bytes),
        },
        "images": [
            {
                "image_id": "camera-frame-42",
                "image_uri": image_path.name,
                "image_sha256": _sha256(image_bytes),
                "decoded_width_px": 300,
                "decoded_height_px": 180,
                "camera_frame": 42,
                "camera_timestamp_sec": 1234.5,
            }
        ],
        "point_sources": [
            {
                "point_source_id": "lidar-frame-42",
                "lidar_uri": lidar_path.name,
                "lidar_sha256": _sha256(lidar_bytes),
                "lidar_frame": 42,
                "lidar_timestamp_sec": 1234.48,
                "pose_uri": pose_path.name,
                "pose_sha256": _sha256(pose_bytes),
            }
        ],
    }
    if include_references:
        payload["references"] = references
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


class ProjectionAuditBuilderTest(unittest.TestCase):
    def _paths(self, root: Path) -> tuple[Path, Path, str]:
        calibration_path = root / "camera.json"
        calibration_bytes = json.dumps(_calibration_payload(), sort_keys=True).encode()
        calibration_path.write_bytes(calibration_bytes)
        reference_path = root / "references.json"
        _write_json(reference_path, _reference_set(root, _sha256(calibration_bytes)))
        return calibration_path, reference_path, _sha256(calibration_bytes)

    def test_passes_only_with_total_zone_and_error_thresholds_and_loads_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path, reference_path, calibration_hash = self._paths(root)
            audit = build_projection_audit(
                calibration_path,
                reference_path,
                dataset_id="route-a",
                thresholds=ProjectionAuditThresholds(),
            )
            audit_path = write_projection_audit(root / "audit.json", audit)
            loaded = load_projection_audit(
                audit_path,
                calibration_path=calibration_path,
            )
            model = load_camera_model(
                calibration_path,
                dataset_id="route-a",
                projection_audit_path=audit_path,
            )
            annotation_source_sha256 = _sha256(
                (root / "independent-annotation-export.json").read_bytes()
            )

        self.assertEqual(audit["schema_version"], CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION)
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["failures"], [])
        self.assertEqual(audit["calibration_sha256"], calibration_hash)
        self.assertEqual(audit["reference_set_schema_version"], "camera-projection-reference-set/2.0")
        self.assertEqual(audit["image_count"], 1)
        self.assertEqual(audit["point_source_count"], 1)
        self.assertTrue(audit["image_registry_sha256"].startswith("sha256:"))
        self.assertTrue(audit["point_source_registry_sha256"].startswith("sha256:"))
        self.assertEqual(
            audit["thresholds"]["maximum_camera_lidar_delta_sec"],
            0.04,
        )
        self.assertEqual(
            audit["observation_source"]["artifact_sha256"],
            annotation_source_sha256,
        )
        self.assertEqual(audit["metrics"]["valid_projection_count"], 15)
        self.assertEqual(
            {zone: group["count"] for zone, group in audit["metrics"]["zones"].items()},
            {"left": 5, "center": 5, "right": 5},
        )
        self.assertEqual(loaded.observation_source_kind, "manual_pixel_labels")
        self.assertEqual(loaded.reference_set_uri, reference_path.resolve().as_uri())
        self.assertEqual(loaded.reference_set_path, reference_path.resolve())
        self.assertEqual(loaded.metrics, audit["metrics"])
        self.assertTrue(model.trusted)

    def test_missing_references_is_persisted_as_untrusted_not_passed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path, _, calibration_hash = self._paths(root)
            reference_path = root / "empty.json"
            _write_json(
                reference_path,
                _reference_set(root, calibration_hash, include_references=False),
            )

            audit = build_projection_audit(
                calibration_path,
                reference_path,
                dataset_id="route-a",
                thresholds=ProjectionAuditThresholds(),
            )

        self.assertEqual(audit["status"], "untrusted")
        self.assertEqual(audit["metrics"]["reference_count"], 0)
        self.assertEqual(
            audit["failures"],
            [
                "minimum_total_references_not_met",
                "minimum_left_references_not_met",
                "minimum_center_references_not_met",
                "minimum_right_references_not_met",
            ],
        )

    def test_a_bad_edge_zone_fails_even_when_overall_rmse_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path, _, calibration_hash = self._paths(root)
            reference_path = root / "edge-error.json"
            _write_json(
                reference_path,
                _reference_set(root, calibration_hash, offsets={"right": 4.0}),
            )

            audit = build_projection_audit(
                calibration_path,
                reference_path,
                dataset_id="route-a",
                thresholds=ProjectionAuditThresholds(),
            )

        self.assertLess(audit["metrics"]["overall"]["rmse_px"], 3.0)
        self.assertEqual(audit["metrics"]["zones"]["right"]["rmse_px"], 4.0)
        self.assertIn("right_rmse_exceeded", audit["failures"])
        self.assertEqual(audit["status"], "untrusted")

    def test_automatic_projection_cannot_claim_to_be_an_independent_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path, reference_path, calibration_hash = self._paths(root)
            payload = _reference_set(root, calibration_hash)
            payload["observation_source"]["kind"] = "automatic_projection"
            _write_json(reference_path, payload)

            with self.assertRaisesRegex(ValueError, "independent"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

    def test_v1_reference_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path, reference_path, calibration_hash = self._paths(root)
            legacy = _reference_set(root, calibration_hash)
            legacy["schema_version"] = "camera-projection-reference-set/1.0"
            _write_json(reference_path, legacy)
            with self.assertRaisesRegex(ValueError, "schema_version"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

    def test_annotation_source_and_legacy_reference_hash_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path, reference_path, calibration_hash = self._paths(root)

            phantom = _reference_set(root, calibration_hash)
            phantom["observation_source"]["artifact_sha256"] = _sha256(
                b"phantom-annotation"
            )
            _write_json(reference_path, phantom)
            with self.assertRaisesRegex(ValueError, "does not match source bytes"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            missing = _reference_set(root, calibration_hash)
            (root / missing["observation_source"]["artifact_uri"]).unlink()
            _write_json(reference_path, missing)
            with self.assertRaisesRegex(ValueError, "artifact_uri is unavailable"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            current = _reference_set(root, calibration_hash)
            current["references"][0]["image_sha256"] = _sha256(b"phantom")
            _write_json(reference_path, current)
            with self.assertRaisesRegex(ValueError, "forbidden; use image_id"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

    def test_image_registry_rejects_phantom_missing_invalid_resized_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path, reference_path, calibration_hash = self._paths(root)

            phantom = _reference_set(root, calibration_hash)
            phantom["images"][0]["image_sha256"] = _sha256(b"phantom-image")
            _write_json(reference_path, phantom)
            with self.assertRaisesRegex(ValueError, "does not match image bytes"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            missing = _reference_set(root, calibration_hash)
            (root / missing["images"][0]["image_uri"]).unlink()
            _write_json(reference_path, missing)
            with self.assertRaisesRegex(ValueError, "image_uri is unavailable"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            invalid = _reference_set(root, calibration_hash)
            image_path = root / invalid["images"][0]["image_uri"]
            image_path.write_bytes(b"not-an-image")
            invalid["images"][0]["image_sha256"] = _sha256(image_path.read_bytes())
            _write_json(reference_path, invalid)
            with self.assertRaisesRegex(ValueError, "safely decoded"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            resized = _reference_set(root, calibration_hash)
            image_path = root / resized["images"][0]["image_uri"]
            Image.new("RGB", (200, 100), color=(1, 2, 3)).save(image_path, format="PNG")
            resized["images"][0]["image_sha256"] = _sha256(image_path.read_bytes())
            _write_json(reference_path, resized)
            with self.assertRaisesRegex(ValueError, "decoded dimensions do not match registry"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            duplicate = _reference_set(root, calibration_hash)
            duplicate_entry = dict(duplicate["images"][0])
            duplicate_entry["image_id"] = "duplicate-frame"
            duplicate_entry["camera_frame"] = 43
            duplicate_entry["camera_timestamp_sec"] = 1234.6
            duplicate["images"].append(duplicate_entry)
            _write_json(reference_path, duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate image registry path"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

    def test_reference_must_resolve_image_and_point_source_and_decoded_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path, reference_path, calibration_hash = self._paths(root)

            missing_image = _reference_set(root, calibration_hash)
            missing_image["references"][0]["image_id"] = "phantom-image"
            _write_json(reference_path, missing_image)
            with self.assertRaisesRegex(ValueError, "not present in the image registry"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            missing_point_source = _reference_set(root, calibration_hash)
            missing_point_source["references"][0]["point_source_id"] = "phantom-lidar"
            _write_json(reference_path, missing_point_source)
            with self.assertRaisesRegex(ValueError, "not present in the point source registry"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            outside = _reference_set(root, calibration_hash)
            outside["references"][0]["observed_pixel_uv"] = [300.0, 10.0]
            _write_json(reference_path, outside)
            with self.assertRaisesRegex(ValueError, "outside the decoded image"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

    def test_point_source_and_pose_provenance_are_read_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path, reference_path, calibration_hash = self._paths(root)

            phantom = _reference_set(root, calibration_hash)
            phantom["point_sources"][0]["lidar_sha256"] = _sha256(b"phantom-lidar")
            _write_json(reference_path, phantom)
            with self.assertRaisesRegex(ValueError, "does not match LiDAR source bytes"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            missing_lidar = _reference_set(root, calibration_hash)
            (root / missing_lidar["point_sources"][0]["lidar_uri"]).unlink()
            _write_json(reference_path, missing_lidar)
            with self.assertRaisesRegex(ValueError, "lidar_uri is unavailable"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            missing_pose = _reference_set(root, calibration_hash)
            (root / missing_pose["point_sources"][0]["pose_uri"]).unlink()
            _write_json(reference_path, missing_pose)
            with self.assertRaisesRegex(ValueError, "pose_uri is unavailable"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            pose_less = _reference_set(root, calibration_hash)
            del pose_less["point_sources"][0]["pose_uri"]
            del pose_less["point_sources"][0]["pose_sha256"]
            _write_json(reference_path, pose_less)
            with self.assertRaisesRegex(ValueError, "requires a pose-bound point source"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            pose_mismatch = _reference_set(root, calibration_hash)
            pose_path = root / pose_mismatch["point_sources"][0]["pose_uri"]
            pose_payload = json.loads(pose_path.read_text(encoding="utf-8"))
            pose_payload["lidar_frame"] = 99
            _write_json(pose_path, pose_payload)
            pose_mismatch["point_sources"][0]["pose_sha256"] = _sha256(
                pose_path.read_bytes()
            )
            _write_json(reference_path, pose_mismatch)
            with self.assertRaisesRegex(ValueError, "pose source lidar_frame mismatch"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            duplicate = _reference_set(root, calibration_hash)
            duplicate_source = dict(duplicate["point_sources"][0])
            duplicate_source["point_source_id"] = "duplicate-lidar-source"
            duplicate_source["lidar_frame"] = 43
            duplicate_source["lidar_timestamp_sec"] = 1234.58
            duplicate["point_sources"].append(duplicate_source)
            _write_json(reference_path, duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate point source LiDAR path"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

    def test_threshold_flags_can_only_tighten_the_production_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "only tighten"):
            ProjectionAuditThresholds(
                minimum_total_references=3,
                minimum_references_per_zone=1,
                maximum_rmse_px=20.0,
                maximum_error_px=50.0,
            )
        with self.assertRaisesRegex(ValueError, "only tighten"):
            ProjectionAuditThresholds(maximum_camera_lidar_delta_sec=0.041)

    def test_each_reference_enforces_the_camera_lidar_sync_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path, reference_path, calibration_hash = self._paths(root)
            out_of_sync = _reference_set(root, calibration_hash)
            out_of_sync["images"][0]["camera_timestamp_sec"] = 1234.55
            _write_json(reference_path, out_of_sync)
            with self.assertRaisesRegex(ValueError, "timestamp delta exceeds"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )

            tightened = _reference_set(root, calibration_hash)
            _write_json(reference_path, tightened)
            with self.assertRaisesRegex(ValueError, "timestamp delta exceeds"):
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                    thresholds=ProjectionAuditThresholds(
                        maximum_camera_lidar_delta_sec=0.01
                    ),
                )

    def test_invalid_geometric_projection_is_counted_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path, reference_path, calibration_hash = self._paths(root)
            payload = _reference_set(root, calibration_hash)
            payload["references"][0]["point_lidar_xyz_m"] = [0.0, 0.0, -1.0]
            _write_json(reference_path, payload)

            audit = build_projection_audit(
                calibration_path,
                reference_path,
                dataset_id="route-a",
                thresholds=ProjectionAuditThresholds(),
            )

        self.assertEqual(audit["metrics"]["invalid_projection_count"], 1)
        self.assertIn("invalid_projection_references_present", audit["failures"])
        self.assertIn("minimum_left_references_not_met", audit["failures"])
        self.assertEqual(audit["status"], "untrusted")


class ProjectionAuditLoaderTest(unittest.TestCase):
    def test_oversized_audit_and_reference_set_are_rejected_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oversized_audit = root / "oversized-audit.json"
            with oversized_audit.open("wb") as handle:
                handle.seek(4 * 1024 * 1024)
                handle.write(b"x")
            with self.assertRaisesRegex(ValueError, "exceeds the safe size limit"):
                load_projection_audit(oversized_audit)

            calibration_bytes = json.dumps(_calibration_payload(), sort_keys=True).encode()
            calibration_path = root / "camera.json"
            calibration_path.write_bytes(calibration_bytes)
            reference_path = root / "references.json"
            _write_json(
                reference_path,
                _reference_set(root, _sha256(calibration_bytes)),
            )
            audit_path = write_projection_audit(
                root / "audit.json",
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                ),
            )
            with reference_path.open("wb") as handle:
                handle.seek(32 * 1024 * 1024)
                handle.write(b"x")
            with self.assertRaisesRegex(ValueError, "exceeds the safe size limit"):
                load_projection_audit(
                    audit_path,
                    calibration_path=calibration_path,
                )

    def test_forged_internally_consistent_aggregate_cannot_grant_trust(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_bytes = json.dumps(_calibration_payload(), sort_keys=True).encode()
            calibration_path = root / "camera.json"
            calibration_path.write_bytes(calibration_bytes)
            reference_path = root / "references.json"
            _write_json(
                reference_path,
                _reference_set(
                    root,
                    _sha256(calibration_bytes),
                    offsets={"right": 4.0},
                ),
            )
            audit = build_projection_audit(
                calibration_path,
                reference_path,
                dataset_id="route-a",
            )
            zero_metrics = {
                "reference_count": 15,
                "valid_projection_count": 15,
                "invalid_projection_count": 0,
                "overall": {
                    "count": 15,
                    "mean_error_px": 0.0,
                    "rmse_px": 0.0,
                    "max_error_px": 0.0,
                },
                "zones": {
                    zone: {
                        "count": 5,
                        "mean_error_px": 0.0,
                        "rmse_px": 0.0,
                        "max_error_px": 0.0,
                    }
                    for zone in ("left", "center", "right")
                },
            }
            audit["metrics"] = zero_metrics
            audit["failures"] = []
            audit["status"] = "passed"
            audit_path = write_projection_audit(root / "forged-aggregate.json", audit)

            with self.assertRaisesRegex(ValueError, "calibration_path is required"):
                load_projection_audit(audit_path)
            with self.assertRaisesRegex(ValueError, "do not match recomputed"):
                load_projection_audit(
                    audit_path,
                    calibration_path=calibration_path,
                )
            model = load_camera_model(
                calibration_path,
                dataset_id="route-a",
                projection_audit_path=audit_path,
            )

        self.assertFalse(model.trusted)
        self.assertEqual(model.calibration_audit_status, "invalid")
        self.assertEqual(model.trust_reasons, ("projection_audit_invalid",))

    def test_registry_and_annotation_provenance_summary_tampering_is_rejected(self) -> None:
        mutations = (
            (("image_count",), 2),
            (("image_registry_sha256",), _sha256(b"forged-image-registry")),
            (("point_source_count",), 2),
            (("point_source_registry_sha256",), _sha256(b"forged-point-registry")),
            (
                ("observation_source", "artifact_sha256"),
                _sha256(b"forged-annotation-source"),
            ),
        )
        for keys, replacement in mutations:
            with self.subTest(field=".".join(keys)), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                calibration_bytes = json.dumps(
                    _calibration_payload(), sort_keys=True
                ).encode()
                calibration_path = root / "camera.json"
                calibration_path.write_bytes(calibration_bytes)
                reference_path = root / "references.json"
                _write_json(
                    reference_path,
                    _reference_set(root, _sha256(calibration_bytes)),
                )
                audit = build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                )
                target: dict[str, Any] = audit
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = replacement
                audit_path = write_projection_audit(root / "forged.json", audit)

                with self.assertRaises(ValueError):
                    load_projection_audit(
                        audit_path,
                        calibration_path=calibration_path,
                    )

    def test_reference_set_tamper_after_build_is_rejected_by_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_bytes = json.dumps(_calibration_payload(), sort_keys=True).encode()
            calibration_path = root / "camera.json"
            calibration_path.write_bytes(calibration_bytes)
            reference_path = root / "references.json"
            _write_json(
                reference_path,
                _reference_set(root, _sha256(calibration_bytes)),
            )
            audit_path = write_projection_audit(
                root / "audit.json",
                build_projection_audit(
                    calibration_path,
                    reference_path,
                    dataset_id="route-a",
                ),
            )
            reference_path.write_text(
                reference_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not match referenced bytes"):
                load_projection_audit(
                    audit_path,
                    calibration_path=calibration_path,
                )
            model = load_camera_model(
                calibration_path,
                dataset_id="route-a",
                projection_audit_path=audit_path,
            )

        self.assertFalse(model.trusted)
        self.assertEqual(model.calibration_audit_status, "invalid")

    def test_bound_image_annotation_lidar_and_pose_tamper_after_build_are_rejected(self) -> None:
        cases = (
            ("camera-frame-42.png", "does not match image bytes"),
            (
                "independent-annotation-export.json",
                "artifact_sha256 does not match source bytes",
            ),
            ("lidar-frame-42.bin", "does not match LiDAR source bytes"),
            ("pose-frame-42.json", "pose_sha256 does not match pose source bytes"),
        )
        for source_name, expected_error in cases:
            with self.subTest(source_name=source_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                calibration_bytes = json.dumps(
                    _calibration_payload(), sort_keys=True
                ).encode()
                calibration_path = root / "camera.json"
                calibration_path.write_bytes(calibration_bytes)
                reference_path = root / "references.json"
                _write_json(
                    reference_path,
                    _reference_set(root, _sha256(calibration_bytes)),
                )
                audit_path = write_projection_audit(
                    root / "audit.json",
                    build_projection_audit(
                        calibration_path,
                        reference_path,
                        dataset_id="route-a",
                    ),
                )
                source_path = root / source_name
                source_path.write_bytes(source_path.read_bytes() + b"tampered")

                with self.assertRaisesRegex(ValueError, expected_error):
                    load_projection_audit(
                        audit_path,
                        calibration_path=calibration_path,
                    )

    def test_bare_v2_status_passed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forged.json"
            _write_json(
                path,
                {
                    "schema_version": CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION,
                    "status": "passed",
                    "dataset_id": "route-a",
                    "calibration_sha256": _sha256(b"calibration"),
                },
            )

            with self.assertRaisesRegex(ValueError, "reference schema_version"):
                load_projection_audit(path)

    def test_metrics_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_bytes = json.dumps(_calibration_payload(), sort_keys=True).encode()
            calibration_path = root / "camera.json"
            calibration_path.write_bytes(calibration_bytes)
            reference_path = root / "references.json"
            _write_json(
                reference_path,
                _reference_set(root, _sha256(calibration_bytes)),
            )
            audit = build_projection_audit(
                calibration_path,
                reference_path,
                dataset_id="route-a",
                thresholds=ProjectionAuditThresholds(),
            )
            audit["metrics"]["overall"]["rmse_px"] = 0.5
            path = root / "tampered.json"
            write_projection_audit(path, audit)

            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_projection_audit(path)

    def test_loader_rejects_a_hand_edited_weaker_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_bytes = json.dumps(_calibration_payload(), sort_keys=True).encode()
            calibration_path = root / "camera.json"
            calibration_path.write_bytes(calibration_bytes)
            reference_path = root / "references.json"
            _write_json(
                reference_path,
                _reference_set(root, _sha256(calibration_bytes)),
            )
            audit = build_projection_audit(
                calibration_path,
                reference_path,
                dataset_id="route-a",
            )
            audit["thresholds"] = {
                "minimum_total_references": 1,
                "minimum_references_per_zone": 1,
                "maximum_rmse_px": 50.0,
                "maximum_error_px": 100.0,
                "maximum_camera_lidar_delta_sec": 0.05,
            }
            path = write_projection_audit(root / "weakened.json", audit)

            with self.assertRaisesRegex(ValueError, "weaken"):
                load_projection_audit(path)

    def test_v1_audit_is_loadable_but_forced_untrusted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.json"
            _write_json(
                path,
                {
                    "schema_version": LEGACY_CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION,
                    "status": "passed",
                    "dataset_id": "route-a",
                    "calibration_sha256": _sha256(b"calibration"),
                },
            )

            audit = load_projection_audit(path)

        self.assertEqual(audit.status, "untrusted")
        self.assertEqual(audit.failures, ("legacy_projection_audit_schema",))

    def test_audited_image_size_must_match_the_loaded_camera_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_bytes = json.dumps(_calibration_payload(), sort_keys=True).encode()
            calibration_path = root / "camera.json"
            calibration_path.write_bytes(calibration_bytes)
            reference_path = root / "references.json"
            _write_json(
                reference_path,
                _reference_set(root, _sha256(calibration_bytes)),
            )
            audit = build_projection_audit(
                calibration_path,
                reference_path,
                dataset_id="route-a",
            )
            audit["image_size_px"] = [640, 480]
            audit_path = write_projection_audit(root / "wrong-size-audit.json", audit)

            model = load_camera_model(
                calibration_path,
                dataset_id="route-a",
                projection_audit_path=audit_path,
            )

        self.assertFalse(model.trusted)
        self.assertEqual(model.calibration_audit_status, "invalid")
        self.assertEqual(model.trust_reasons, ("projection_audit_invalid",))

    def test_cli_persists_untrusted_audit_and_returns_nonzero_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_bytes = json.dumps(_calibration_payload(), sort_keys=True).encode()
            calibration_path = root / "camera.json"
            calibration_path.write_bytes(calibration_bytes)
            references_path = root / "references.json"
            _write_json(
                references_path,
                _reference_set(
                    root,
                    _sha256(calibration_bytes),
                    include_references=False,
                ),
            )
            output_path = root / "audit.json"

            with redirect_stdout(io.StringIO()):
                exit_code = audit_cli_main(
                    [
                        "--calibration",
                        str(calibration_path),
                        "--references",
                        str(references_path),
                        "--dataset-id",
                        "route-a",
                        "--output",
                        str(output_path),
                    ]
                )

            persisted = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 3)
        self.assertEqual(persisted["status"], "untrusted")


if __name__ == "__main__":
    unittest.main()
