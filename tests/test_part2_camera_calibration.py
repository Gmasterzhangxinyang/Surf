from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from parking_slot_part2.camera_calibration import (
    CAMERA_CALIBRATION_SCHEMA_VERSION,
    CalibrationError,
    CameraCalibration,
    camera_calibration_from_mapping,
    read_camera_calibration,
    read_legacy_camera_calibration,
)


def _placeholder_payload(*, reverse: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CAMERA_CALIBRATION_SCHEMA_VERSION,
        "camera_model": "fisheye",
        "image_width": 400,
        "image_height": 200,
        "K": [[100.0, 0.0, 200.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        "D": [0.0, 0.0, 0.0, 0.0],
        "calibration_id": "placeholder-unit-test-calibration",
        "calibration_time": "2026-01-01T00:00:00Z",
        "vehicle_frame": "vehicle",
        "camera_frame": "camera_front",
        "is_placeholder": True,
        "extrinsic_translation_unit": "m",
    }
    if reverse:
        payload["T_camera_vehicle"] = [
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, -3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    else:
        payload["T_vehicle_camera"] = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    return payload


class StrictCameraCalibrationTest(unittest.TestCase):
    def test_repository_example_is_explicitly_placeholder_and_runtime_rejected(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "camera_front.placeholder.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertIs(payload["is_placeholder"], True)
        self.assertIn("PLACEHOLDER", payload["calibration_id"])
        with self.assertRaisesRegex(CalibrationError, "placeholder"):
            read_camera_calibration(path)

    def test_from_mapping_normalizes_declared_reverse_extrinsic(self) -> None:
        calibration = CameraCalibration.from_mapping(
            _placeholder_payload(reverse=True),
            allow_placeholder=True,
        )

        np.testing.assert_allclose(
            calibration.T_vehicle_camera[:3, 3],
            [1.0, 2.0, 3.0],
            atol=1e-12,
        )
        self.assertEqual(calibration.source_format, "strict-mapping")

    def test_mapping_parser_requires_exactly_one_extrinsic_direction(self) -> None:
        payload = _placeholder_payload()
        payload["T_camera_vehicle"] = np.eye(4).tolist()

        with self.assertRaisesRegex(CalibrationError, "exactly one"):
            camera_calibration_from_mapping(
                payload,
                allow_placeholder=True,
            )

    def test_placeholder_is_rejected_by_default_for_mapping_and_path(self) -> None:
        payload = _placeholder_payload()
        with self.assertRaisesRegex(CalibrationError, "placeholder"):
            camera_calibration_from_mapping(payload)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "placeholder-calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CalibrationError, "placeholder"):
                read_camera_calibration(path)
            parsed = read_camera_calibration(path, allow_placeholder=True)
        self.assertEqual(parsed.calibration_id, payload["calibration_id"])
        self.assertEqual(parsed.source_format, "strict-json")

    def test_missing_real_metadata_fails_closed(self) -> None:
        payload = _placeholder_payload()
        del payload["calibration_time"]
        with self.assertRaisesRegex(CalibrationError, "calibration_time"):
            CameraCalibration.from_mapping(
                payload,
                allow_placeholder=True,
            )

    def test_fisheye_projection_is_opencv_equidistant_polynomial(self) -> None:
        calibration = CameraCalibration.from_mapping(
            _placeholder_payload(),
            allow_placeholder=True,
        )
        points = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )

        uv, valid = calibration.project_camera_points(points)

        np.testing.assert_array_equal(valid, [True, True, True, False])
        np.testing.assert_allclose(uv[0], [200.0, 100.0], atol=1e-12)
        np.testing.assert_allclose(
            uv[1],
            [200.0 + 100.0 * math.pi / 2.0, 100.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            uv[2],
            [200.0, 100.0 + 100.0 * math.pi / 2.0],
            atol=1e-12,
        )
        self.assertTrue(np.isnan(uv[3]).all())

    def test_fisheye_projection_applies_theta_polynomial_not_linear_fov(self) -> None:
        payload = _placeholder_payload()
        payload["D"] = [0.1, 0.0, 0.0, 0.0]
        calibration = CameraCalibration.from_mapping(
            payload,
            allow_placeholder=True,
        )
        uv, valid = calibration.project_camera_points([[1.0, 0.0, 1.0]])
        theta = math.pi / 4.0
        expected_u = 200.0 + 100.0 * theta * (1.0 + 0.1 * theta**2)

        self.assertTrue(valid[0])
        self.assertAlmostEqual(uv[0, 0], expected_u, places=12)
        self.assertAlmostEqual(uv[0, 1], 100.0, places=12)

    def test_legacy_reader_never_infers_reverse_direction_from_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            intrinsic_path = Path(directory) / "calib_cam_to_cam.txt"
            extrinsic_path = Path(directory) / "calib_velo_to_cam.txt"
            intrinsic_path.write_text(
                "K: 100 0 200 0 100 100 0 0 1\n",
                encoding="utf-8",
            )
            extrinsic_path.write_text(
                "Tr: 1 0 0 -1 0 1 0 0 0 0 1 0\n",
                encoding="utf-8",
            )
            calibration = read_legacy_camera_calibration(
                intrinsic_path,
                extrinsic_path,
                stored_extrinsic="T_camera_vehicle",
                camera_model="fisheye",
                image_width=400,
                image_height=200,
                distortion_coefficients=[0.0, 0.0, 0.0, 0.0],
                calibration_id="placeholder-legacy-unit-test",
                calibration_time="2026-01-01T00:00:00Z",
                vehicle_frame="vehicle",
                camera_frame="camera_front",
                is_placeholder=True,
                extrinsic_translation_unit="m",
                allow_placeholder=True,
            )
            np.testing.assert_allclose(
                calibration.T_vehicle_camera[:3, 3],
                [1.0, 0.0, 0.0],
                atol=1e-12,
            )
            with self.assertRaisesRegex(CalibrationError, "explicitly declare"):
                read_legacy_camera_calibration(
                    intrinsic_path,
                    extrinsic_path,
                    stored_extrinsic="unknown",  # type: ignore[arg-type]
                    camera_model="fisheye",
                    image_width=400,
                    image_height=200,
                    distortion_coefficients=[0.0, 0.0, 0.0, 0.0],
                    calibration_id="placeholder-legacy-unit-test",
                    calibration_time="2026-01-01T00:00:00Z",
                    vehicle_frame="vehicle",
                    camera_frame="camera_front",
                    is_placeholder=True,
                    extrinsic_translation_unit="m",
                    allow_placeholder=True,
                )


if __name__ == "__main__":
    unittest.main()
