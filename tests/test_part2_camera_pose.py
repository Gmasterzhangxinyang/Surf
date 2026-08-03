from __future__ import annotations

import math
import unittest

import numpy as np

from parking_slot_part2.camera_calibration import CameraCalibration
from parking_slot_part2.camera_pose import (
    CameraMapPoseEstimate,
    PoseInterpolationPolicy,
    PoseValidationError,
    VehiclePoseSample,
    estimate_camera_map_pose,
    interpolate_vehicle_pose,
    vehicle_pose_samples_from_records,
)


def _transform(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    angle = math.radians(yaw_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    transform[:3, 3] = [x, y, z]
    return transform


def _covariance(
    position_variance: float = 1e-4,
    orientation_variance: float = 1e-6,
) -> np.ndarray:
    return np.diag(
        [position_variance] * 3 + [orientation_variance] * 3
    )


def _sample(
    timestamp_ns: int,
    transform: np.ndarray,
    *,
    covariance: np.ndarray | None = None,
    status: str = "valid",
    frame_id: str = "vehicle",
    parent_frame: str = "map",
    translation_unit: str = "m",
    map_units_per_meter: float | None = None,
) -> VehiclePoseSample:
    return VehiclePoseSample(
        timestamp_ns=timestamp_ns,
        parent_frame=parent_frame,
        frame_id=frame_id,
        T_map_vehicle=transform,
        pose_covariance=(
            _covariance() if covariance is None else covariance
        ),
        localization_status=status,  # type: ignore[arg-type]
        translation_unit=translation_unit,  # type: ignore[arg-type]
        map_units_per_meter=map_units_per_meter,
    )


def _placeholder_calibration(
    transform: np.ndarray | None = None,
    *,
    vehicle_frame: str = "vehicle",
) -> CameraCalibration:
    return CameraCalibration(
        camera_model="fisheye",
        image_width=400,
        image_height=200,
        K=np.asarray([[100.0, 0.0, 200.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]]),
        D=np.zeros(4),
        T_vehicle_camera=(np.eye(4) if transform is None else transform),
        calibration_id="placeholder-camera-pose-unit-test",
        calibration_time="2026-01-01T00:00:00Z",
        vehicle_frame=vehicle_frame,
        camera_frame="camera_front",
        is_placeholder=True,
        extrinsic_translation_unit="m",
        source_format="unit-test-placeholder",
    )


def _policy(
    *,
    max_offset_ms: float = 1500.0,
    max_gap_ms: float = 2500.0,
    degraded_position: float = 0.05,
    invalid_position: float = 0.5,
    degraded_orientation: float = 0.01,
    invalid_orientation: float = 0.1,
) -> PoseInterpolationPolicy:
    return PoseInterpolationPolicy(
        max_abs_time_offset_ms=max_offset_ms,
        max_interpolation_gap_ms=max_gap_ms,
        degraded_position_variance_m2=degraded_position,
        invalid_position_variance_m2=invalid_position,
        degraded_orientation_variance_rad2=degraded_orientation,
        invalid_orientation_variance_rad2=invalid_orientation,
    )


class CameraPoseCompositionTest(unittest.TestCase):
    def test_composes_only_t_map_vehicle_times_t_vehicle_camera(self) -> None:
        vehicle = _sample(1_000_000_000, _transform(10.0, 0.0, 0.0, 90.0))
        extrinsic = _transform(1.0, 0.0, 0.0, 30.0)
        calibration = _placeholder_calibration(extrinsic)

        estimate = estimate_camera_map_pose(
            [vehicle],
            vehicle.timestamp_ns,
            calibration,
            _policy(),
            allow_placeholder_calibration=True,
        )

        self.assertEqual(estimate.localization_status, "valid")
        np.testing.assert_allclose(
            estimate.T_map_camera,
            vehicle.T_map_vehicle @ extrinsic,
            atol=1e-12,
        )
        np.testing.assert_allclose(estimate.position_xyz, [10.0, 1.0, 0.0])
        expected_yaw = math.radians(120.0)
        np.testing.assert_allclose(
            estimate.orientation_xyzw,
            [0.0, 0.0, math.sin(expected_yaw / 2.0), math.cos(expected_yaw / 2.0)],
            atol=1e-12,
        )
        payload = estimate.to_dict()
        for field in (
            "timestamp_ns",
            "frame_id",
            "parent_frame",
            "T_map_camera",
            "position_xyz",
            "orientation_xyzw",
            "pose_covariance",
            "localization_status",
            "time_offset_ms",
            "calibration_id",
        ):
            self.assertIn(field, payload)
        self.assertEqual(len(payload["pose_covariance"]), 36)
        self.assertEqual(payload["frame_id"], "camera_front")
        self.assertEqual(payload["parent_frame"], "map")

    def test_map_unit_pose_scales_metric_extrinsic_before_composition(self) -> None:
        map_scale = 0.05
        vehicle = _sample(
            1_000_000_000,
            _transform(5.0, 2.0),
            covariance=_covariance(1e-8, 1e-6),
            translation_unit="map_unit",
            map_units_per_meter=map_scale,
        )
        estimate = estimate_camera_map_pose(
            [vehicle],
            vehicle.timestamp_ns,
            _placeholder_calibration(_transform(1.0, 0.0, 0.0)),
            _policy(),
            allow_placeholder_calibration=True,
        )

        self.assertEqual(estimate.localization_status, "valid")
        np.testing.assert_allclose(estimate.position_xyz, [5.05, 2.0, 0.0])
        self.assertEqual(estimate.translation_unit, "map_unit")
        self.assertEqual(estimate.map_units_per_meter, map_scale)
        self.assertIn(
            "extrinsic_translation_scaled_to_map_units",
            estimate.reason_codes,
        )

    def test_map_unit_pose_without_scale_is_rejected(self) -> None:
        with self.assertRaises(PoseValidationError) as context:
            _sample(
                0,
                np.eye(4),
                translation_unit="map_unit",
            )
        self.assertEqual(context.exception.code, "map_scale_missing")


class CameraPoseInterpolationTest(unittest.TestCase):
    def test_translation_rotation_and_covariance_are_interpolated(self) -> None:
        first_covariance = _covariance(0.01, 0.001)
        second_covariance = _covariance(0.03, 0.003)
        first = _sample(1_000_000_000, _transform(yaw_deg=0.0), covariance=first_covariance)
        second = _sample(3_000_000_000, _transform(2.0, 0.0, yaw_deg=180.0), covariance=second_covariance)

        pose = interpolate_vehicle_pose(
            [second, first],
            2_000_000_000,
            _policy(
                degraded_position=1.0,
                invalid_position=2.0,
                degraded_orientation=1.0,
                invalid_orientation=2.0,
            ),
        )

        self.assertTrue(pose.interpolated)
        self.assertEqual(pose.time_offset_ms, -1000.0)
        np.testing.assert_allclose(pose.T_map_vehicle[:3, 3], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            pose.T_map_vehicle[:3, :3],
            _transform(yaw_deg=90.0)[:3, :3],
            atol=1e-12,
        )
        self.assertEqual(pose.T_map_vehicle.shape, (4, 4))
        np.testing.assert_allclose(
            pose.pose_covariance,
            0.5 * (first_covariance + second_covariance),
        )

    def test_inconsistent_pose_units_are_rejected(self) -> None:
        metric = _sample(0, np.eye(4))
        map_unit = _sample(
            2_000_000_000,
            np.eye(4),
            translation_unit="map_unit",
            map_units_per_meter=0.05,
        )
        with self.assertRaises(PoseValidationError) as context:
            interpolate_vehicle_pose(
                [metric, map_unit],
                1_000_000_000,
                _policy(),
            )
        self.assertEqual(
            context.exception.code,
            "translation_unit_mismatch",
        )


class CameraPoseFailClosedTest(unittest.TestCase):
    def _assert_invalid_empty(self, estimate: CameraMapPoseEstimate) -> None:
        self.assertEqual(estimate.localization_status, "invalid")
        payload = estimate.to_dict()
        self.assertEqual(payload["T_map_camera"], [])
        self.assertEqual(payload["position_xyz"], [])
        self.assertEqual(payload["orientation_xyzw"], [])
        self.assertEqual(payload["pose_covariance"], [])

    def test_frame_mismatch_fails_closed(self) -> None:
        sample = _sample(0, np.eye(4))
        estimate = estimate_camera_map_pose(
            [sample],
            0,
            _placeholder_calibration(vehicle_frame="base_link"),
            _policy(),
            allow_placeholder_calibration=True,
        )
        self._assert_invalid_empty(estimate)
        self.assertEqual(estimate.reason_codes, ("frame_mismatch",))

    def test_stale_nearest_pose_fails_closed_with_signed_offset(self) -> None:
        samples = [
            _sample(0, _transform()),
            _sample(2_000_000_000, _transform(2.0, 0.0)),
        ]
        estimate = estimate_camera_map_pose(
            samples,
            1_000_000_000,
            _placeholder_calibration(),
            _policy(max_offset_ms=100.0),
            allow_placeholder_calibration=True,
        )
        self._assert_invalid_empty(estimate)
        self.assertEqual(estimate.reason_codes, ("pose_timestamp_stale",))
        self.assertEqual(estimate.time_offset_ms, -1000.0)

    def test_excessive_covariance_is_invalid_and_intermediate_is_degraded(self) -> None:
        calibration = _placeholder_calibration()
        invalid = estimate_camera_map_pose(
            [_sample(0, np.eye(4), covariance=_covariance(0.6, 0.001))],
            0,
            calibration,
            _policy(),
            allow_placeholder_calibration=True,
        )
        self._assert_invalid_empty(invalid)
        self.assertEqual(invalid.reason_codes, ("covariance_exceeds_limit",))

        degraded = estimate_camera_map_pose(
            [_sample(0, np.eye(4), covariance=_covariance(0.1, 0.001))],
            0,
            calibration,
            _policy(),
            allow_placeholder_calibration=True,
        )
        self.assertEqual(degraded.localization_status, "degraded")
        self.assertIn("localization_degraded", degraded.reason_codes)

    def test_placeholder_calibration_is_rejected_by_runtime_default(self) -> None:
        estimate = estimate_camera_map_pose(
            [_sample(0, np.eye(4))],
            0,
            _placeholder_calibration(),
            _policy(),
        )
        self._assert_invalid_empty(estimate)
        self.assertEqual(estimate.reason_codes, ("placeholder_calibration",))

    def test_strict_records_reject_missing_covariance_status_and_units(self) -> None:
        with self.assertRaises(PoseValidationError) as context:
            vehicle_pose_samples_from_records(
                [
                    {
                        "timestamp_ns": 0,
                        "parent_frame": "map",
                        "frame_id": "vehicle",
                        "T_map_vehicle": np.eye(4).tolist(),
                    }
                ]
            )
        self.assertEqual(context.exception.code, "pose_fields_missing")


if __name__ == "__main__":
    unittest.main()
