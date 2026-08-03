import unittest
from pathlib import Path

import numpy as np

from parking_slot_hybrid_3d.accumulation import (
    build_slot_accumulation,
    select_slot_frames,
    split_frame_ids,
)
from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import FrameRecord, KnownSlot, ScopeEvidence, ScopeStatus
from parking_slot_hybrid_3d.io import FrameLoadError


def slot() -> KnownSlot:
    return KnownSlot(
        slot_id="slot_0001",
        polygon_map=np.asarray([[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]]),
        core_polygon_map=np.asarray([[-1.8, -0.8], [1.8, -0.8], [1.8, 0.8], [-1.8, 0.8]]),
        margin_polygon_map=np.asarray([[-2.5, -1.5], [2.5, -1.5], [2.5, 1.5], [-2.5, 1.5]]),
        center_map=np.asarray([0.0, 0.0]),
        heading_deg=0.0,
    )


def frame(frame_id: int, map_x: float, map_y: float = 0.0) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        map_x=map_x,
        map_y=map_y,
        map_yaw=0.0,
        map_points_path=Path(f"{frame_id:06d}.npz"),
    )


def flat_frame_points(extra_points: list[list[float]] | None = None) -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(-3.0, 3.0, 10), np.linspace(-1.5, 1.5, 5))
    ground = np.column_stack(
        [xs.ravel(), ys.ravel(), np.full(xs.size, -1.5), np.ones(xs.size)]
    )
    extras = np.asarray(extra_points or [], dtype=np.float64).reshape(-1, 4)
    return np.vstack([ground, extras])


class DictionaryPointProvider:
    def __init__(self, values: dict[int, np.ndarray], missing: set[int] | None = None) -> None:
        self.values = values
        self.missing = missing or set()

    def load(self, frame_id: int) -> np.ndarray:
        if frame_id in self.missing:
            raise FrameLoadError(frame_id, Path(f"{frame_id:06d}.npz"), "missing_map_points")
        return self.values[frame_id]


class Hybrid3DAccumulationTest(unittest.TestCase):
    def test_anchor_is_nearest_only_among_frames_that_reached_slot(self) -> None:
        frames = [frame(index, map_x=20.0) for index in range(0, 61)]
        replacements = {30: frame(30, -3.0), 31: frame(31, -2.0), 32: frame(32, -4.0)}
        frames = [replacements.get(item.frame_id, item) for item in frames]
        scope = ScopeEvidence(
            slot_id="slot_0001",
            scope_status=ScopeStatus.IN_ROUTE,
            crossing_frames=(30, 31, 32),
        )
        config = Hybrid3DConfig(window_before=8, window_after=8, frame_stride=4)

        anchor, selected = select_slot_frames(slot(), scope, frames, 1.0, config)

        self.assertEqual(anchor, 31)
        self.assertEqual(selected, (23, 27, 31, 35, 39))

    def test_accumulation_preserves_frame_point_origin_and_relevant_ray_provenance(self) -> None:
        frames = [frame(6, -5.0), frame(10, -4.0), frame(14, -5.0)]
        scope = ScopeEvidence(
            slot_id="slot_0001",
            scope_status=ScopeStatus.IN_ROUTE,
            crossing_frames=(6, 10, 14),
        )
        values = {
            item.frame_id: flat_frame_points(
                [
                    [0.0, 0.0, -0.5, 7.0],
                    [8.0, 0.0, -1.5, 3.0],
                    [-8.0, 8.0, -1.5, 2.0],
                ]
            )
            for item in frames
        }
        config = Hybrid3DConfig(window_before=4, window_after=4, frame_stride=4)

        result = build_slot_accumulation(
            slot(), scope, frames, DictionaryPointProvider(values), 1.0, config
        )

        self.assertEqual(result.anchor_frame, 10)
        self.assertEqual(result.selected_frames, (6, 10, 14))
        self.assertEqual([observation.frame_id for observation in result.observations], [6, 10, 14])
        self.assertEqual(set(result.point_frame_ids.tolist()), {6, 10, 14})
        self.assertTrue(np.any(np.isclose(result.points_local_xyzi[:, 2], 1.0, atol=0.03)))
        self.assertTrue(np.any(np.isclose(result.points_local_xyzi[:, 3], 7.0)))
        for observation in result.observations:
            self.assertAlmostEqual(observation.origin_local_xyz[2], 1.5, delta=0.03)
            self.assertTrue(np.any(np.isclose(observation.ray_endpoints_local_xyz[:, 0], 8.0)))
            self.assertFalse(np.any(np.isclose(observation.ray_endpoints_local_xyz[:, 1], 8.0)))
        self.assertFalse(result.points_local_xyzi.flags.writeable)
        self.assertFalse(result.point_frame_ids.flags.writeable)

    def test_invalid_ground_and_missing_frame_are_recorded_not_treated_as_empty(self) -> None:
        frames = [frame(6, -5.0), frame(10, -4.0), frame(14, -5.0)]
        scope = ScopeEvidence(
            slot_id="slot_0001",
            scope_status=ScopeStatus.IN_ROUTE,
            crossing_frames=(6, 10, 14),
        )
        values = {
            6: flat_frame_points(),
            10: np.asarray([[0.0, 0.0, -1.5, 1.0], [1.0, 0.0, -1.5, 1.0]]),
        }
        config = Hybrid3DConfig(window_before=4, window_after=4, frame_stride=4)

        result = build_slot_accumulation(
            slot(), scope, frames, DictionaryPointProvider(values, missing={14}), 1.0, config
        )

        self.assertEqual([observation.frame_id for observation in result.observations], [6])
        self.assertEqual(
            result.excluded_frames,
            ((10, ("invalid_ground_model",)), (14, ("missing_map_points",))),
        )

    def test_split_membership_is_chronological_and_deterministic(self) -> None:
        splits = split_frame_ids((6, 10, 14, 18, 22))

        self.assertEqual(splits["first_half"], (6, 10))
        self.assertEqual(splits["second_half"], (14, 18, 22))
        self.assertEqual(splits["odd_index"], (6, 14, 22))
        self.assertEqual(splits["even_index"], (10, 18))
        self.assertEqual(splits, split_frame_ids((6, 10, 14, 18, 22)))


if __name__ == "__main__":
    unittest.main()
