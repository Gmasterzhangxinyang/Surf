import unittest

import numpy as np

from parking_slot_box_scoring.config import BoxScoringConfig
from parking_slot_box_scoring.geometry import compute_slot_frame
from parking_slot_box_scoring.scoring import score_slot_points


def rectangular_slot() -> dict:
    polygon = np.asarray([[-2.5, -1.0], [2.5, -1.0], [2.5, 1.0], [-2.5, 1.0]], dtype=np.float64)
    core = np.asarray([[-2.0, -0.7], [2.0, -0.7], [2.0, 0.7], [-2.0, 0.7]], dtype=np.float64)
    margin = np.asarray([[-2.8, -1.2], [2.8, -1.2], [2.8, 1.2], [-2.8, 1.2]], dtype=np.float64)
    return {
        "slot_id": "slot_test",
        "polygon_np": polygon,
        "core_np": core,
        "margin_np": margin,
        "center_np": np.asarray([0.0, 0.0], dtype=np.float64),
        "adjacent_slots": [],
    }


class SlotBoxScoringSyntheticTest(unittest.TestCase):
    def test_centered_vehicle_box_scores_high(self) -> None:
        slot = rectangular_slot()
        xs = np.linspace(-1.7, 1.7, 10)
        ys = np.linspace(-0.55, 0.55, 5)
        points = []
        for x in xs:
            for y in ys:
                points.append([x, y, 0.75, 1.0])
                points.append([x, y, 1.25, 1.0])
        points = np.asarray(points, dtype=np.float64)
        config = BoxScoringConfig(min_vehicle_points_accumulated=30, min_frame_vehicle_points=3)

        score = score_slot_points(slot, points, [points, points, points], {}, 1.0, config)

        self.assertEqual(score.state, "box_vehicle_core_supported")
        self.assertGreaterEqual(score.score, 0.55)
        self.assertGreater(score.height_support, 0.35)

    def test_low_height_residual_scores_low(self) -> None:
        slot = rectangular_slot()
        xs = np.linspace(-1.5, 1.5, 12)
        ys = np.linspace(-0.5, 0.5, 4)
        points = np.asarray([[x, y, 0.12, 1.0] for x in xs for y in ys], dtype=np.float64)
        config = BoxScoringConfig(min_vehicle_points_accumulated=20)

        score = score_slot_points(slot, points, [points, points, points], {}, 1.0, config)

        self.assertIn(score.state, {"box_low_height_residual", "box_no_vehicle_evidence"})
        self.assertLess(score.height_support, 0.35)

    def test_boundary_only_points_not_vehicle(self) -> None:
        slot = rectangular_slot()
        xs = np.linspace(-2.4, 2.4, 16)
        points = np.asarray([[x, 0.95, 0.8, 1.0] for x in xs], dtype=np.float64)
        config = BoxScoringConfig(min_vehicle_points_accumulated=8, min_frame_vehicle_points=2)

        score = score_slot_points(slot, points, [points, points, points], {}, 1.0, config)

        self.assertNotEqual(score.state, "box_vehicle_core_supported")
        self.assertIn(
            score.state,
            {"box_boundary_conflict", "box_wall_like_or_static_suspect", "box_no_vehicle_evidence", "box_low_height_residual"},
        )


if __name__ == "__main__":
    unittest.main()
