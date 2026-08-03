import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import DecisionState, KnownSlot, ScopeStatus, SlotDecision
from parking_slot_hybrid_3d.geometry import inset_local_rectangle, map_xy_to_slot_m, metric_slot


def known_slot(center_map: tuple[float, float] = (10.0, 20.0), heading_deg: float = 90.0) -> KnownSlot:
    center = np.asarray(center_map, dtype=np.float64)
    polygon = center + np.asarray(
        [[-0.05, -0.125], [0.05, -0.125], [0.05, 0.125], [-0.05, 0.125]],
        dtype=np.float64,
    )
    core = center + np.asarray(
        [[-0.04, -0.10], [0.04, -0.10], [0.04, 0.10], [-0.04, 0.10]],
        dtype=np.float64,
    )
    margin = center + np.asarray(
        [[-0.06, -0.14], [0.06, -0.14], [0.06, 0.14], [-0.06, 0.14]],
        dtype=np.float64,
    )
    return KnownSlot(
        slot_id="slot_0001",
        polygon_map=polygon,
        core_polygon_map=core,
        margin_polygon_map=margin,
        center_map=center,
        heading_deg=heading_deg,
        adjacent_slots=("slot_0002",),
    )


class Hybrid3DContractTest(unittest.TestCase):
    def test_map_xy_converts_to_metric_slot_local_coordinates(self) -> None:
        slot = known_slot(center_map=(10.0, 20.0), heading_deg=90.0)
        converted = metric_slot(slot, map_units_per_meter=0.05)

        local = map_xy_to_slot_m(np.asarray([[10.0, 20.10]], dtype=np.float64), converted)

        np.testing.assert_allclose(local, [[2.0, 0.0]], atol=1e-8)
        np.testing.assert_allclose(slot.center_map, [10.0, 20.0])
        self.assertEqual(converted.slot_id, slot.slot_id)
        self.assertEqual(converted.adjacent_slots, slot.adjacent_slots)

    def test_invalid_ratio_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "scope_core_coverage_min"):
            Hybrid3DConfig(scope_core_coverage_min=1.1).validate()

    def test_localization_uncertainty_defaults_to_exact_legacy_geometry(self) -> None:
        polygon = np.asarray(
            [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]],
            dtype=np.float64,
        )

        np.testing.assert_allclose(inset_local_rectangle(polygon, 0.0), polygon)
        np.testing.assert_allclose(
            inset_local_rectangle(polygon, 0.25),
            [[-1.75, -0.75], [1.75, -0.75], [1.75, 0.75], [-1.75, 0.75]],
        )

    def test_invalid_localization_uncertainty_is_rejected(self) -> None:
        for value in (-0.01, 0.51):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "occupied_localization_uncertainty_m"
            ):
                Hybrid3DConfig(
                    occupied_localization_uncertainty_m=value
                ).validate()

    def test_overlapping_height_layers_are_rejected(self) -> None:
        config = Hybrid3DConfig(height_layers_m=((0.30, 0.90), (0.80, 1.40)))

        with self.assertRaisesRegex(ValueError, "height_layers_m"):
            config.validate()

    def test_box_hypothesis_config_preserves_current_search_space(self) -> None:
        box_config = Hybrid3DConfig().box_hypothesis_config()

        self.assertEqual(box_config.length_scales, [0.70, 0.85, 1.00, 1.10])
        self.assertEqual(box_config.width_scales, [0.55, 0.70, 0.85, 1.00])
        self.assertEqual(box_config.yaw_offsets_deg, [-8.0, -4.0, 0.0, 4.0, 8.0])

    def test_decision_contract_rejects_out_of_scope_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "out_of_route_scope"):
            SlotDecision(
                slot_id="slot_0001",
                scope_status=ScopeStatus.OUT_OF_ROUTE,
                state=DecisionState.FREE,
                decision_reason="invalid",
            )

    def test_decision_contract_is_frozen(self) -> None:
        decision = SlotDecision(
            slot_id="slot_0001",
            scope_status=ScopeStatus.IN_ROUTE,
            state=DecisionState.UNKNOWN,
            decision_reason="insufficient_evidence",
            unknown_reasons=("insufficient_evidence",),
        )

        with self.assertRaises(FrozenInstanceError):
            decision.state = DecisionState.FREE


if __name__ == "__main__":
    unittest.main()
