import math
import unittest

import numpy as np

from parking_slot_box_scoring.geometry import (
    box_polygon,
    compute_slot_frame,
    global_to_local,
    local_to_global,
    polygon_area,
)


class SlotBoxGeometryTest(unittest.TestCase):
    def test_slot_local_transform_roundtrip(self) -> None:
        yaw = math.radians(32.0)
        long_axis = np.asarray([math.cos(yaw), math.sin(yaw)])
        short_axis = np.asarray([-math.sin(yaw), math.cos(yaw)])
        center = np.asarray([10.0, -4.0])
        length = 5.0
        width = 2.0
        local_corners = np.asarray(
            [[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2], [-length / 2, width / 2]]
        )
        polygon = center + local_corners[:, 0:1] * long_axis + local_corners[:, 1:2] * short_axis
        frame = compute_slot_frame(polygon)
        points = np.asarray([[9.1, -4.2], [10.8, -3.6], [10.0, -4.0]])

        local = global_to_local(points, frame)
        reconstructed = local_to_global(local, frame)

        self.assertTrue(np.allclose(points, reconstructed, atol=1e-9))
        self.assertAlmostEqual(frame.length, length, places=6)
        self.assertAlmostEqual(frame.width, width, places=6)

    def test_box_polygon_area_matches_length_width(self) -> None:
        polygon = box_polygon(np.asarray([1.0, 2.0]), math.radians(17), 4.0, 1.5)

        self.assertAlmostEqual(polygon_area(polygon), 6.0, places=6)


if __name__ == "__main__":
    unittest.main()
