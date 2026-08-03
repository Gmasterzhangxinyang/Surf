import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class DatasetIntegrityLinkageTest(unittest.TestCase):
    def test_minimal_complete_dataset_links_all_data_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "reports" / "dataset_integrity_linkage"
            dataset_root = root / "raw_dataset"
            image = dataset_root / "image" / "left000006.png"
            lidar = dataset_root / "velodyne" / "000006.bin"
            image.parent.mkdir(parents=True)
            lidar.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            lidar.write_bytes(b"lidar")

            map_points = root / "outputs" / "frame_map_dataset" / "map_points" / "000006.npz"
            map_points.parent.mkdir(parents=True)
            np.savez(
                map_points,
                points_map_xyzi=np.array([[1.0, 2.0, 0.5, 10.0], [1.1, 2.1, 0.6, 20.0]], dtype=np.float32),
                ego_map_pose=np.array([5.0, 6.0, 0.1], dtype=np.float32),
                frame=np.array([6], dtype=np.int32),
                map_scale=np.array([0.04558499], dtype=np.float32),
            )

            write_csv(
                root / "outputs" / "frame_map_dataset" / "frames.csv",
                [
                    "frame",
                    "image_path",
                    "lidar_path",
                    "map_points_path",
                    "map_x",
                    "map_y",
                    "map_yaw",
                    "lidar_timestamp",
                    "odom_timestamp",
                    "time_delta_sec",
                    "num_points",
                    "missing_image",
                    "missing_lidar",
                ],
                [
                    {
                        "frame": 6,
                        "image_path": str(image),
                        "lidar_path": str(lidar),
                        "map_points_path": "outputs/frame_map_dataset/map_points/000006.npz",
                        "map_x": 5.0,
                        "map_y": 6.0,
                        "map_yaw": 0.1,
                        "lidar_timestamp": 1.0,
                        "odom_timestamp": 1.0,
                        "time_delta_sec": 0.0,
                        "num_points": 2,
                        "missing_image": 0,
                        "missing_lidar": 0,
                    }
                ],
            )
            (root / "outputs" / "frame_map_dataset" / "metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_root": str(dataset_root),
                        "selected_rows": 1,
                        "processed_lidar_frames": 1,
                        "missing_lidar": 0,
                        "missing_image": 0,
                        "total_projected_points": 2,
                    }
                ),
                encoding="utf-8",
            )
            write_csv(
                root / "outputs" / "azimuth_time_odometry_compatible.csv",
                ["seq", "frame", "x", "y", "yaw_rad", "z", "lidar_timestamp", "odom_timestamp", "time_delta_sec"],
                [
                    {
                        "seq": 0,
                        "frame": 6,
                        "x": 0.0,
                        "y": 0.0,
                        "yaw_rad": 0.1,
                        "z": 0.0,
                        "lidar_timestamp": 1.0,
                        "odom_timestamp": 1.0,
                        "time_delta_sec": 0.0,
                    }
                ],
            )
            write_csv(
                root / "outputs" / "aligned_trajectory_final.csv",
                ["frame", "x", "y", "yaw"],
                [{"frame": 6, "x": 5.0, "y": 6.0, "yaw": 0.1}],
            )

            part1 = root / "outputs" / "full_icpark_allframes_vehicle_cluster"
            part1.mkdir(parents=True)
            (part1 / "slot_database.json").write_text(
                json.dumps({"slot_count": 1, "slots": [{"slot_id": "slot_0001", "polygon_map": []}]}),
                encoding="utf-8",
            )
            write_csv(part1 / "slot_scores.csv", ["slot_id", "state"], [{"slot_id": "slot_0001", "state": "free"}])
            write_csv(
                part1 / "slot_decision_table" / "slot_decision_table.csv",
                ["slot_id", "final_decision"],
                [{"slot_id": "slot_0001", "final_decision": "free"}],
            )

            module = importlib.import_module("scripts.dataset_integrity_linkage")
            summary = module.run_audit(root, output_dir, npz_samples=1, check_all_npz=False)

            self.assertEqual(summary["overall_status"], "PASS")
            self.assertEqual(summary["frame_checks"]["rows"], 1)
            self.assertEqual(summary["frame_checks"]["bad_rows"], 0)
            self.assertEqual(summary["slot_checks"]["slot_count"], 1)
            self.assertEqual(summary["slot_checks"]["bad_rows"], 0)
            self.assertTrue((output_dir / "integrity_summary.json").exists())
            self.assertTrue((output_dir / "frame_linkage_audit.csv").exists())
            self.assertTrue((output_dir / "slot_linkage_audit.csv").exists())
            self.assertTrue((output_dir / "dataset_integrity_linkage_report.md").exists())


if __name__ == "__main__":
    unittest.main()
