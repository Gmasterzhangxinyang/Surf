from __future__ import annotations

import unittest

import numpy as np

from parking_slot_hybrid_3d.contracts import MetricSlot, SlotAccumulation
from parking_slot_hybrid_3d.static_semantics import SemanticStaticOccupiedVeto


def _slot() -> MetricSlot:
    polygon = np.asarray([[-2.5, -1.2], [2.5, -1.2], [2.5, 1.2], [-2.5, 1.2]])
    return MetricSlot(
        slot_id="slot_test",
        center_map=np.asarray([0.0, 0.0]),
        long_axis_map=np.asarray([1.0, 0.0]),
        short_axis_map=np.asarray([0.0, 1.0]),
        polygon_local_m=polygon,
        core_polygon_local_m=polygon * 0.8,
        margin_polygon_local_m=polygon * 1.1,
        adjacent_slots=(),
        map_units_per_meter=1.0,
    )


def _accumulation(points_xy: np.ndarray) -> SlotAccumulation:
    points = np.column_stack(
        [
            points_xy,
            np.full(len(points_xy), 1.0),
            np.ones(len(points_xy)),
        ]
    )
    return SlotAccumulation(
        slot_id="slot_test",
        anchor_frame=10,
        selected_frames=(10,),
        points_local_xyzi=points,
        point_frame_ids=np.full(len(points), 10, dtype=np.int64),
        observations=(),
    )


class StaticSemanticOccupiedVetoTests(unittest.TestCase):
    def test_vetoes_semantic_static_explained_returns(self) -> None:
        x, y = np.meshgrid(np.linspace(-2.0, 2.0, 16), np.linspace(-0.8, 0.8, 8))
        points = np.column_stack([x.ravel(), y.ravel()])
        veto = SemanticStaticOccupiedVeto(points.copy())
        assessment = veto.assess(_slot(), _accumulation(points))
        self.assertTrue(assessment.veto)
        self.assertEqual(assessment.reason, "semantic_static_explained_ratio")
        self.assertGreaterEqual(assessment.static_explained_ratio, 0.99)

    def test_vetoes_unexplained_narrow_residual(self) -> None:
        x, y = np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-0.15, 0.15, 4))
        points = np.column_stack([x.ravel(), y.ravel()])
        veto = SemanticStaticOccupiedVeto(np.asarray([[20.0, 20.0]]))
        assessment = veto.assess(_slot(), _accumulation(points))
        self.assertTrue(assessment.veto)
        self.assertEqual(assessment.reason, "semantic_static_residual_too_narrow")
        self.assertLess(assessment.residual_short_extent_m, 0.75)

    def test_allows_unexplained_vehicle_sized_footprint(self) -> None:
        x, y = np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-0.8, 0.8, 8))
        points = np.column_stack([x.ravel(), y.ravel()])
        veto = SemanticStaticOccupiedVeto(np.asarray([[20.0, 20.0]]))
        assessment = veto.assess(_slot(), _accumulation(points))
        self.assertFalse(assessment.veto)
        self.assertEqual(assessment.reason, "semantic_static_veto_passed")
        self.assertGreaterEqual(assessment.residual_short_extent_m, 0.75)


if __name__ == "__main__":
    unittest.main()
