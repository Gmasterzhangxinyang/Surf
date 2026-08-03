from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from parking_slot_hybrid_3d.camera import (
    CAMERA_CALIBRATION_SCHEMA_VERSION,
    CameraFrameAssessment,
    CameraModel,
    CameraSelectionConfig,
    assess_camera_frame,
    assess_pre_anchor_candidates,
    load_camera_calibration,
    load_camera_model,
    preselect_pre_anchor_frames,
    projection_quality,
    select_camera_assessments,
)
from parking_slot_hybrid_3d.contracts import FrameRecord
from parking_slot_hybrid_3d.projection_audit import (
    CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION,
    ProjectionAuditThresholds,
    build_projection_audit,
    write_projection_audit,
)


CALIBRATION_TEXT = (
    "P0: 527.525085 0 636.297913 0 527.525085 357.787354 0 0 1\n"
    "Tr: 1 0 0 0.6 0 1 0 0.0 0 0 1 -0.07 0 0 0 1\n"
)


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _write_passing_audit(
    root: Path,
    calibration_path: Path,
    *,
    dataset_id: str,
) -> Path:
    """Create a strict audit from a synthetic independent-reference fixture."""
    calibration = load_camera_calibration(calibration_path)
    points_lidar = [
        [9.8 + 0.1 * sample_index, lateral, -0.8]
        for lateral in (9.5, 0.0, -9.5)
        for sample_index in range(5)
    ]
    references = []
    for index, point in enumerate(points_lidar):
        camera_xyz = calibration.camera_from_lidar @ np.asarray([*point, 1.0])
        pixel_h = calibration.intrinsic_matrix @ camera_xyz[:3]
        observed_uv = (pixel_h[:2] / pixel_h[2]).tolist()
        references.append(
            {
                "id": f"synthetic-independent-{index}",
                "image_id": "synthetic-camera-frame-42",
                "point_source_id": "synthetic-lidar-frame-42",
                "point_lidar_xyz_m": list(point),
                "observed_pixel_uv": observed_uv,
            }
        )
    image_path = root / "synthetic-camera-frame-42.png"
    Image.new(
        "RGB",
        (calibration.image_width_px, calibration.image_height_px),
        color=(16, 32, 64),
    ).save(image_path, format="PNG")
    image_bytes = image_path.read_bytes()
    lidar_path = root / "synthetic-lidar-frame-42.bin"
    lidar_path.write_bytes(b"synthetic-lidar-source\x00\x01")
    lidar_bytes = lidar_path.read_bytes()
    annotation_path = root / "synthetic-independent-label-export.json"
    annotation_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "annotations": [
                    {
                        "image_id": reference["image_id"],
                        "observed_pixel_uv": reference["observed_pixel_uv"],
                        "point_source_id": reference["point_source_id"],
                        "reference_id": reference["id"],
                    }
                    for reference in references
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    annotation_bytes = annotation_path.read_bytes()
    reference_path = root / "projection-references.json"
    reference_path.write_text(
        json.dumps(
            {
                "schema_version": CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "calibration_sha256": calibration.source_sha256,
                "observation_source": {
                    "kind": "ground_truth_correspondences",
                    "artifact_uri": annotation_path.name,
                    "artifact_sha256": _sha256(annotation_bytes),
                },
                "images": [
                    {
                        "image_id": "synthetic-camera-frame-42",
                        "image_uri": image_path.name,
                        "image_sha256": _sha256(image_bytes),
                        "decoded_width_px": calibration.image_width_px,
                        "decoded_height_px": calibration.image_height_px,
                        "camera_frame": 42,
                        "camera_timestamp_sec": 1234.5,
                    }
                ],
                "point_sources": [
                    {
                        "point_source_id": "synthetic-lidar-frame-42",
                        "lidar_uri": lidar_path.name,
                        "lidar_sha256": _sha256(lidar_bytes),
                        "lidar_frame": 42,
                        "lidar_timestamp_sec": 1234.48,
                    }
                ],
                "references": references,
            }
        ),
        encoding="utf-8",
    )
    audit = build_projection_audit(
        calibration_path,
        reference_path,
        dataset_id=dataset_id,
        thresholds=ProjectionAuditThresholds(),
    )
    return write_projection_audit(root / "projection-audit.json", audit)


def _frame(
    frame_id: int,
    image_path: Path,
    *,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    camera_frame: int | None = None,
    lidar_timestamp: float = 10.0,
    camera_timestamp: float = 9.99,
    delta: float = -0.01,
    match_valid: bool = True,
) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        map_x=x,
        map_y=y,
        map_yaw=yaw,
        map_points_path=None,
        lidar_timestamp=lidar_timestamp,
        camera_frame=camera_frame if camera_frame is not None else 1000 + frame_id,
        camera_image_path=image_path,
        camera_timestamp=camera_timestamp,
        camera_lidar_dt_sec=delta,
        camera_match_valid=match_valid,
    )


class CameraCalibrationTest(unittest.TestCase):
    def test_loads_dataset_p0_tr_calibration_with_axis_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib.txt"
            path.write_text(CALIBRATION_TEXT, encoding="utf-8")

            calibration = load_camera_calibration(path)

        point_lidar = np.asarray([10.0, 1.0, -0.8, 1.0])
        point_camera = calibration.camera_from_lidar @ point_lidar
        np.testing.assert_allclose(point_camera[:3], [-1.0, 0.73, 9.4])
        self.assertEqual(calibration.source_format, "p0-tr-text")
        self.assertEqual(calibration.image_width_px, 1280)
        self.assertEqual(calibration.image_height_px, 720)
        self.assertEqual(calibration.source_sha256, _sha256(CALIBRATION_TEXT.encode()))

    def test_loads_explicit_json_calibration_without_hidden_defaults(self) -> None:
        payload = {
            "schema_version": CAMERA_CALIBRATION_SCHEMA_VERSION,
            "camera_id": "front-left",
            "image_size_px": [640, 480],
            "intrinsics": {"fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0},
            "extrinsics": {
                "optical_from_lidar": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
                "camera_origin_lidar_m": [0.6, 0.0, -0.07],
            },
            "distortion": {"model": "none", "coefficients": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            calibration = load_camera_calibration(path)

        self.assertEqual(calibration.camera_id, "front-left")
        self.assertEqual((calibration.image_width_px, calibration.image_height_px), (640, 480))
        self.assertAlmostEqual(calibration.fx, 500.0)
        self.assertAlmostEqual(calibration.fy, 501.0)

    def test_incomplete_fixture_style_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.json"
            path.write_text(
                json.dumps({"schema_version": "camera-calibration-fixture/1.0", "camera_id": "front"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_camera_calibration(path)

    def test_model_is_untrusted_without_audit_and_trusted_only_when_hash_and_dataset_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / "calib.txt"
            calibration_path.write_text(CALIBRATION_TEXT, encoding="utf-8")

            missing = load_camera_model(calibration_path, dataset_id="route-a")
            audit_path = _write_passing_audit(
                root,
                calibration_path,
                dataset_id="route-a",
            )

            trusted = load_camera_model(
                calibration_path,
                dataset_id="route-a",
                projection_audit_path=audit_path,
            )
            wrong_dataset = load_camera_model(
                calibration_path,
                dataset_id="route-b",
                projection_audit_path=audit_path,
            )

        self.assertFalse(missing.trusted)
        self.assertEqual(missing.calibration_audit_status, "missing")
        self.assertTrue(trusted.trusted)
        self.assertEqual(trusted.trust_reasons, ())
        self.assertFalse(wrong_dataset.trusted)
        self.assertEqual(wrong_dataset.calibration_audit_status, "dataset_mismatch")

    def test_calibration_byte_change_invalidates_passing_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / "calib.txt"
            calibration_path.write_text(CALIBRATION_TEXT, encoding="utf-8")
            audit_path = _write_passing_audit(
                root,
                calibration_path,
                dataset_id="route-a",
            )
            calibration_path.write_text(CALIBRATION_TEXT + "\n", encoding="utf-8")

            model = load_camera_model(
                calibration_path,
                dataset_id="route-a",
                projection_audit_path=audit_path,
            )

        self.assertFalse(model.trusted)
        self.assertEqual(model.calibration_audit_status, "calibration_mismatch")

    def test_camera_model_cannot_be_forged_passed_without_bound_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib.txt"
            path.write_text(CALIBRATION_TEXT, encoding="utf-8")
            calibration = load_camera_calibration(path)

            with self.assertRaisesRegex(ValueError, "calibration and audit"):
                CameraModel(calibration, None, "passed", ())


class CameraProjectionTest(unittest.TestCase):
    def test_projection_quality_uses_clipped_polygon_area(self) -> None:
        quality = projection_quality(
            np.asarray([[-50.0, 0.0], [50.0, 0.0], [50.0, 100.0], [-50.0, 100.0]]),
            100,
            100,
        )

        self.assertAlmostEqual(quality["raw_projected_area_px"], 10_000.0)
        self.assertAlmostEqual(quality["projected_area_px"], 5_000.0)
        self.assertAlmostEqual(quality["visible_fraction"], 0.5)
        self.assertEqual(quality["bbox_px"], [0.0, 0.0, 50.0, 100.0])

    def test_untrusted_calibration_returns_geometry_and_intended_but_no_effective_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / "calib.txt"
            calibration_path.write_text(CALIBRATION_TEXT, encoding="utf-8")
            image_path = root / "frame.png"
            image_path.write_bytes(b"image")
            model = load_camera_model(calibration_path, dataset_id="route-a")

            assessment = assess_camera_frame(
                _frame(100, image_path),
                np.asarray([[3.0, -1.5], [12.0, -1.5], [12.0, 1.5], [3.0, 1.5]]),
                [8.0, 0.0],
                1.0,
                -0.8,
                model,
                anchor_frame=100,
                anchor_timestamp=10.0,
                sync_source="corrected",
            )

        self.assertEqual(
            assessment.intended_capabilities,
            ("can_assess_free", "can_assess_occupied"),
        )
        self.assertEqual(assessment.effective_capabilities, ())
        self.assertEqual(assessment.rejection_reasons, ("calibration_untrusted",))
        self.assertEqual(assessment.projection_quality["calibration_audit_status"], "missing")
        self.assertEqual(assessment.projection_quality["calibration_sha256"], model.calibration_sha256)
        self.assertEqual(len(assessment.polygon_uv), 4)

    def test_passing_audit_grants_occupied_and_free_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / "calib.txt"
            calibration_path.write_text(CALIBRATION_TEXT, encoding="utf-8")
            audit_path = _write_passing_audit(
                root,
                calibration_path,
                dataset_id="route-a",
            )
            image_path = root / "frame.png"
            image_path.write_bytes(b"image")
            model = load_camera_model(
                calibration_path,
                dataset_id="route-a",
                projection_audit_path=audit_path,
            )

            assessment = assess_camera_frame(
                _frame(100, image_path),
                np.asarray([[3.0, -1.5], [12.0, -1.5], [12.0, 1.5], [3.0, 1.5]]),
                [8.0, 0.0],
                1.0,
                -0.8,
                model,
                anchor_frame=100,
                anchor_timestamp=10.0,
                sync_source="native",
            )

        self.assertEqual(assessment.effective_capabilities, assessment.intended_capabilities)
        self.assertEqual(assessment.projection_quality["calibration_audit_status"], "passed")
        self.assertEqual(assessment.rejection_reasons, ())

    def test_invalid_sync_prevents_even_intended_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / "calib.txt"
            calibration_path.write_text(CALIBRATION_TEXT, encoding="utf-8")
            image_path = root / "frame.png"
            image_path.write_bytes(b"image")
            calibration = load_camera_calibration(calibration_path)
            model = CameraModel(calibration, None, "missing", ("projection_audit_missing",))

            assessment = assess_camera_frame(
                _frame(100, image_path, camera_timestamp=9.9, delta=-0.01),
                np.asarray([[3.0, -1.5], [12.0, -1.5], [12.0, 1.5], [3.0, 1.5]]),
                [8.0, 0.0],
                1.0,
                -0.8,
                model,
                anchor_frame=100,
                anchor_timestamp=10.0,
                sync_source="derived",
            )

        self.assertEqual(assessment.intended_capabilities, ())
        self.assertIn("camera_sync_delta_mismatch", assessment.rejection_reasons)
        self.assertIn("unsupported_or_missing_sync_metadata", assessment.rejection_reasons)


class CameraCandidateSelectionTest(unittest.TestCase):
    def test_pose_prefilter_honors_pre_anchor_window_stride_fov_and_limit(self) -> None:
        missing_image = Path("/does/not/exist.png")
        frames = [
            _frame(frame_id, missing_image, x=float(frame_id - 90))
            for frame_id in range(90, 111, 5)
        ]
        # Frame 110 is after the declared anchor. Frame 90 is outside lookback.
        settings = CameraSelectionConfig(
            lookback_frames=15,
            frame_stride=5,
            pose_prefilter_limit=2,
        )

        selected = preselect_pre_anchor_frames(
            frames,
            [20.0, 0.0],
            105,
            1.0,
            config=settings,
        )

        self.assertEqual([frame.frame_id for frame in selected], [100, 105])

    def test_ranking_keeps_best_frames_then_returns_chronological_order(self) -> None:
        image = Path("/does/not/exist.png")
        rows = []
        for frame_id, visible in ((90, 0.70), (95, 0.95), (100, 0.80)):
            rows.append(
                CameraFrameAssessment(
                    frame=_frame(frame_id, image),
                    polygon_uv=(),
                    projection_quality={
                        "visible_fraction": visible,
                        "projected_area_px": 1000.0,
                        "bearing_deg": 0.0,
                    },
                    intended_capabilities=(
                        ("can_assess_free", "can_assess_occupied")
                        if visible >= 0.90
                        else ("can_assess_occupied",)
                    ),
                    effective_capabilities=(),
                    rejection_reasons=("calibration_untrusted",),
                )
            )

        selected = select_camera_assessments(rows, max_frames=2)

        self.assertEqual([assessment.frame.frame_id for assessment in selected], [95, 100])
        self.assertEqual(select_camera_assessments(rows, max_frames=2, trusted_only=True), ())

    def test_batch_exposes_all_assessments_and_ranked_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / "calib.txt"
            calibration_path.write_text(CALIBRATION_TEXT, encoding="utf-8")
            image_path = root / "frame.png"
            image_path.write_bytes(b"image")
            model = load_camera_model(calibration_path, dataset_id="route-a")
            frames = [
                _frame(90, image_path, x=-2.0, lidar_timestamp=9.0, camera_timestamp=8.99),
                _frame(95, image_path, x=-1.0, lidar_timestamp=9.5, camera_timestamp=9.49),
                _frame(100, image_path, x=0.0),
            ]

            batch = assess_pre_anchor_candidates(
                frames,
                np.asarray([[3.0, -1.5], [12.0, -1.5], [12.0, 1.5], [3.0, 1.5]]),
                [8.0, 0.0],
                100,
                10.0,
                1.0,
                {90: -0.8, 95: -0.8, 100: -0.8},
                model,
                sync_source="corrected",
                config=CameraSelectionConfig(lookback_frames=10, frame_stride=5),
            )

        self.assertEqual([row.lidar_frame for row in batch.assessments], [90, 95, 100])
        self.assertTrue(batch.selected)
        self.assertTrue(all(row.intended_capabilities for row in batch.selected))


if __name__ == "__main__":
    unittest.main()
