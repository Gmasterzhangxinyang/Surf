from __future__ import annotations

import unittest

from parking_slot_agent_v2.fov import (
    DEFAULT_HALF_FOV_DEG,
    DEFAULT_NOMINAL_HALF_FOV_DEG,
    evaluate_horizontal_projection_fov,
    evaluate_map_bearing_fov,
)


class CoarseFovPolicyTest(unittest.TestCase):
    def test_default_policy_uses_intrinsic_envelope_with_80_degree_reliable_zone(self):
        self.assertAlmostEqual(DEFAULT_NOMINAL_HALF_FOV_DEG * 2.0, 101.004231, places=5)
        self.assertEqual(DEFAULT_HALF_FOV_DEG, 40.0)

    def test_target_near_90_degrees_is_not_reported_visible(self):
        result = evaluate_map_bearing_fov(
            ego_map_xy=[0.0, 0.0],
            ego_yaw_rad=0.0,
            target_polygon_map=[
                [0.8, 3.8],
                [1.2, 3.8],
                [1.2, 4.2],
                [0.8, 4.2],
            ],
            yaw_uncertainty_deg=15.0,
        )
        self.assertNotEqual(result.visibility.value, "visible")
        self.assertAlmostEqual(result.details["half_fov_deg"], 40.0)
        self.assertEqual(
            result.details["bearing_sign_convention"],
            "positive_is_camera_left",
        )

    def test_horizontal_projection_forbids_center_outside_even_when_corner_clips(self):
        result = evaluate_horizontal_projection_fov(
            ego_map_xy=[0.0, 0.0],
            ego_yaw_rad=0.0,
            target_center_map=[3.0, 5.0],
            target_polygon_map=[
                [3.0, 2.0],
                [4.0, 2.0],
                [4.0, 6.0],
                [3.0, 6.0],
            ],
            map_units_per_meter=1.0,
        )
        self.assertEqual(result.visibility.value, "not_visible")
        self.assertFalse(result.details["camera_candidate"])
        self.assertTrue(any(
            row["inside_horizontal_image"]
            for row in result.details["sample_horizontal_projection"][1:]
        ))

    def test_horizontal_projection_allows_target_center_inside(self):
        result = evaluate_horizontal_projection_fov(
            ego_map_xy=[0.0, 0.0],
            ego_yaw_rad=0.0,
            target_center_map=[10.0, 3.0],
            target_polygon_map=[
                [9.0, 2.0],
                [11.0, 2.0],
                [11.0, 4.0],
                [9.0, 4.0],
            ],
            map_units_per_meter=1.0,
        )
        self.assertIn(result.visibility.value, {"visible", "partially_visible"})
        self.assertTrue(result.details["camera_candidate"])
        self.assertGreaterEqual(result.details["target_center_pixel_u"], 0.0)
        self.assertLess(result.details["target_center_pixel_u"], 1280.0)
        self.assertEqual(
            result.details["projection_role"], "camera_tool_feasibility_only"
        )


if __name__ == "__main__":
    unittest.main()
