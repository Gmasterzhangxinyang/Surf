import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_static_obstacle_layer_from_lidar import (
    COVERAGE_AUDIT_SCHEMA_VERSION,
    DYNAMIC_AUDIT_SCHEMA_VERSION,
    BuildConfig,
    build_static_obstacle_layer,
)
from parking_slot_part2.static_obstacle_map import StaticObstacleMap


class StaticObstacleLayerFromLidarTest(unittest.TestCase):
    MAP_FRAME_ID = "test-map/frame-v1"

    def _fixture(self, root: Path) -> tuple[Path, Path, BuildConfig]:
        points_dir = root / "points"
        points_dir.mkdir()
        manifest = root / "frames.csv"
        slot_database = root / "slot_database.json"

        # This slot contains a parked-vehicle-shaped return cluster.  Slot
        # exclusion must remove it before persistence is calculated.
        slot_database.write_text(
            json.dumps(
                {
                    "map_frame_id": self.MAP_FRAME_ID,
                    "map_units_per_meter": 1.0,
                    "slot_count": 1,
                    "slots": [
                        {
                            "slot_id": "slot-1",
                            "polygon_map": [[-0.5, -0.5], [0.75, -0.5], [0.75, 0.75], [-0.5, 0.75]],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        rows = []
        ground_x, ground_y = np.meshgrid(np.linspace(-2.0, 12.0, 29), np.linspace(-2.0, 2.0, 9))
        ground_z = 3.0
        ground = np.column_stack(
            [
                ground_x.ravel(),
                ground_y.ravel(),
                np.full(ground_x.size, ground_z),
                np.ones(ground_x.size),
            ]
        )
        for frame_id in range(5):
            jitter = frame_id * 0.001
            # Four persistent voxels for a true static wall.
            wall = np.asarray(
                [
                    [5.02 + jitter, 0.02, 4.0, 1.0],
                    [5.27 + jitter, 0.02, 4.1, 1.0],
                    [5.02 + jitter, 0.27, 4.2, 1.0],
                    [5.27 + jitter, 0.27, 4.3, 1.0],
                ]
            )
            # A vehicle is persistent in map XY, so persistence alone cannot
            # safely distinguish it from static infrastructure.
            dynamic_vehicle = np.asarray(
                [
                    [10.02 + jitter, 0.02, 4.0, 1.0],
                    [10.27 + jitter, 0.02, 4.1, 1.0],
                    [10.02 + jitter, 0.27, 4.2, 1.0],
                    [10.27 + jitter, 0.27, 4.3, 1.0],
                ]
            )
            parked_in_slot = np.asarray(
                [
                    [0.02, 0.02, 4.0, 1.0],
                    [0.27, 0.02, 4.1, 1.0],
                    [0.02, 0.27, 4.2, 1.0],
                    [0.27, 0.27, 4.3, 1.0],
                ]
            )
            cloud = np.vstack([ground, wall, dynamic_vehicle, parked_in_slot]).astype(np.float32)
            points_path = points_dir / f"{frame_id:06d}.npz"
            np.savez(points_path, points_map_xyzi=cloud)
            rows.append([frame_id, str(points_path)])

        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "map_points_path"])
            writer.writerows(rows)

        config = BuildConfig(
            ground_quantile=0.08,
            ground_clearance_m=0.20,
            max_height_above_ground_m=2.50,
            voxel_size_m=0.25,
            min_persistent_frames=3,
            min_persistence_ratio=0.60,
            cluster_neighbor_radius_m=0.36,
            min_cluster_voxels=4,
        )
        return manifest, slot_database, config

    def test_without_dynamic_audit_emits_review_candidates_but_no_static_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, slot_database, config = self._fixture(root)

            result = build_static_obstacle_layer(
                manifest,
                slot_database,
                self.MAP_FRAME_ID,
                points_xy_unit="map_unit",
                points_z_unit="m",
                config=config,
            )

            self.assertEqual(result["static_obstacles"], [])
            self.assertFalse(result["audits"]["dynamic_filter"]["trusted"])
            self.assertTrue(result["review_required"]["required"])
            self.assertFalse(result["mapped_regions_complete"])
            self.assertEqual(len(result["candidate_obstacles"]), 2)
            self.assertTrue(all(item["review_required"] for item in result["candidate_obstacles"]))
            # The parked cluster at x ~= 0 was excluded by the slot polygon.
            self.assertTrue(all(item["centroid_map"][0] > 4.0 for item in result["candidate_obstacles"]))

    def test_trusted_dynamic_audit_excludes_vehicle_and_coverage_is_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, slot_database, config = self._fixture(root)
            candidate_run = build_static_obstacle_layer(
                manifest,
                slot_database,
                self.MAP_FRAME_ID,
                points_xy_unit="map_unit",
                points_z_unit="m",
                config=config,
            )
            identity = candidate_run["input_identity"]

            dynamic_audit = root / "dynamic_audit.json"
            dynamic_audit.write_text(
                json.dumps(
                    {
                        "schema_version": DYNAMIC_AUDIT_SCHEMA_VERSION,
                        "status": "passed",
                        "map_frame_id": self.MAP_FRAME_ID,
                        "input_identity": identity,
                        "method": "independent synthetic tracker audit",
                        "filter_scope_complete": True,
                        "dynamic_regions": [
                            {
                                "region_id": "moving-vehicle-1",
                                "polygon_map": [[9.75, -0.25], [10.75, -0.25], [10.75, 0.75], [9.75, 0.75]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dynamic_only = build_static_obstacle_layer(
                manifest,
                slot_database,
                self.MAP_FRAME_ID,
                points_xy_unit="map_unit",
                points_z_unit="m",
                dynamic_filter_audit_path=dynamic_audit,
                config=config,
            )

            self.assertTrue(dynamic_only["audits"]["dynamic_filter"]["trusted"])
            self.assertEqual(dynamic_only["candidate_obstacles"], [])
            self.assertEqual(len(dynamic_only["static_obstacles"]), 1)
            self.assertEqual(dynamic_only["static_obstacles"][0]["type"], "other_static")
            self.assertLess(dynamic_only["static_obstacles"][0]["centroid_map"][0], 7.0)
            self.assertGreaterEqual(dynamic_only["static_obstacles"][0]["min_z"], 4.0)
            self.assertGreater(dynamic_only["processing"]["dynamic_region_removed"], 0)
            # A dynamic audit never promotes coverage completeness by itself.
            self.assertFalse(dynamic_only["mapped_regions_complete"])

            coverage_audit = root / "coverage_audit.json"
            coverage_audit.write_text(
                json.dumps(
                    {
                        "schema_version": COVERAGE_AUDIT_SCHEMA_VERSION,
                        "status": "passed",
                        "map_frame_id": self.MAP_FRAME_ID,
                        "input_identity": identity,
                        "method": "independent synthetic visibility audit",
                        "complete": True,
                        "regions": [
                            {
                                "region_id": "test-mapped-area",
                                "polygon_map": [[-2.0, -2.0], [12.0, -2.0], [12.0, 2.0], [-2.0, 2.0]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            fully_audited = build_static_obstacle_layer(
                manifest,
                slot_database,
                self.MAP_FRAME_ID,
                points_xy_unit="map_unit",
                points_z_unit="m",
                dynamic_filter_audit_path=dynamic_audit,
                coverage_audit_path=coverage_audit,
                config=config,
            )

            self.assertTrue(fully_audited["audits"]["coverage"]["trusted"])
            self.assertTrue(fully_audited["mapped_regions_complete"])
            self.assertEqual(fully_audited["mapped_regions"][0]["region_id"], "test-mapped-area")
            parsed = StaticObstacleMap.from_mapping(fully_audited)
            self.assertTrue(parsed.layer_present)
            self.assertEqual(len(parsed.static_obstacles), 1)
            self.assertLess(parsed.static_obstacles[0].object_id.find("vehicle"), 0)


if __name__ == "__main__":
    unittest.main()
