from __future__ import annotations

import unittest

from parking_slot_part2.static_obstacle_map import (
    StaticObstacleMap,
    StaticObstacleMapValidationError,
)


def _rectangle(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _obstacle(
    object_id: str,
    polygon: list[list[float]],
    *,
    object_type: str = "wall",
    min_z: float = 0.0,
    max_z: float = 2.8,
) -> dict[str, object]:
    return {
        "id": object_id,
        "type": object_type,
        "polygon_xy": polygon,
        "min_z": min_z,
        "max_z": max_z,
        "confidence": 0.95,
        "source": "manual",
    }


def _layer(
    obstacles: list[dict[str, object]],
    *,
    complete: bool = True,
    regions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "static-obstacle-map/1.0",
        "coordinate_frame": "map-test",
        "map_units_per_meter": 1.0,
        "static_obstacles": obstacles,
        "mapped_regions": regions
        if regions is not None
        else [
            {
                "region_id": "local",
                "polygon_xy": _rectangle(-10.0, -10.0, 110.0, 10.0),
                "static_obstacle_layer_complete": complete,
            }
        ],
    }


class StaticObstacleMapTest(unittest.TestCase):
    def _query(self, static_map: StaticObstacleMap):
        return static_map.query_corridor(
            [0.0, 0.0],
            [10.0, 0.0],
            0.5,
            coordinate_frame="map-test",
            map_units_per_meter=1.0,
        )

    def test_complete_empty_region_is_known_clear(self) -> None:
        query = self._query(StaticObstacleMap.from_mapping(_layer([])))

        self.assertTrue(query.information_sufficient)
        self.assertTrue(query.local_coverage_complete)
        self.assertTrue(query.complete_empty)
        self.assertEqual(query.obstacles, ())

    def test_explicit_present_complete_layer_may_omit_empty_obstacle_array(self) -> None:
        payload = _layer([])
        payload.pop("static_obstacles")
        payload["static_obstacle_layer_present"] = True

        query = self._query(StaticObstacleMap.from_mapping(payload))

        self.assertTrue(query.information_sufficient)
        self.assertTrue(query.complete_empty)

    def test_only_local_corridor_completeness_is_required(self) -> None:
        regions = [
            {
                "region_id": "local-complete",
                "polygon_xy": _rectangle(-1.0, -1.0, 11.0, 1.0),
                "static_obstacle_layer_complete": True,
            },
            {
                "region_id": "remote-incomplete",
                "polygon_xy": _rectangle(50.0, 50.0, 60.0, 60.0),
                "static_obstacle_layer_complete": False,
            },
        ]
        query = self._query(
            StaticObstacleMap.from_mapping(_layer([], regions=regions))
        )

        self.assertTrue(query.information_sufficient)
        self.assertEqual(query.mapped_region_ids, ("local-complete",))

    def test_incomplete_corridor_is_fail_closed(self) -> None:
        query = self._query(
            StaticObstacleMap.from_mapping(_layer([], complete=False))
        )

        self.assertFalse(query.information_sufficient)
        self.assertIn(
            "corridor_outside_complete_mapped_region",
            query.reason_codes,
        )

    def test_long_wall_is_found_by_aabb_grid_not_centroid(self) -> None:
        wall = _obstacle("wall-long", _rectangle(1.0, -0.2, 100.0, 0.2))
        query = self._query(
            StaticObstacleMap.from_mapping(_layer([wall]))
        )

        self.assertEqual(
            [item.object_id for item in query.obstacles],
            ["wall-long"],
        )
        self.assertLess(query.spatial_candidate_count, query.total_obstacle_count + 1)

    def test_low_curb_is_retained_but_not_a_full_height_occluder(self) -> None:
        curb = _obstacle(
            "curb-1",
            _rectangle(4.0, -0.2, 4.5, 0.2),
            object_type="curb",
            max_z=0.2,
        )
        query = self._query(
            StaticObstacleMap.from_mapping(_layer([curb]))
        )

        self.assertEqual(query.low_obstacle_ids, ("curb-1",))
        self.assertEqual(query.camera_occluder_obstacles, ())
        self.assertEqual(len(query.obstacles), 1)
        self.assertEqual(query.obstacles[0].to_mapping()["type"], "other_static")

    def test_dynamic_vehicle_cannot_be_written_into_static_layer(self) -> None:
        vehicle = _obstacle(
            "vehicle-1",
            _rectangle(4.0, -1.0, 6.0, 1.0),
            object_type="vehicle",
        )

        with self.assertRaises(StaticObstacleMapValidationError) as caught:
            StaticObstacleMap.from_mapping(_layer([vehicle]))

        self.assertEqual(
            caught.exception.reason_code,
            "dynamic_object_in_static_layer",
        )

    def test_invalid_or_self_intersecting_polygon_is_rejected(self) -> None:
        crossed = _obstacle(
            "bad-wall",
            [[0.0, 0.0], [3.0, 2.0], [0.0, 3.0], [2.0, 0.0]],
        )

        with self.assertRaises(StaticObstacleMapValidationError):
            StaticObstacleMap.from_mapping(_layer([crossed]))

    def test_frame_or_scale_mismatch_makes_query_insufficient(self) -> None:
        static_map = StaticObstacleMap.from_mapping(_layer([]))
        query = static_map.query_corridor(
            [0.0, 0.0],
            [10.0, 0.0],
            0.5,
            coordinate_frame="another-map",
            map_units_per_meter=2.0,
        )

        self.assertFalse(query.information_sufficient)
        self.assertIn(
            "static_obstacle_coordinate_frame_mismatch",
            query.reason_codes,
        )
        self.assertIn("static_obstacle_map_units_mismatch", query.reason_codes)


if __name__ == "__main__":
    unittest.main()
