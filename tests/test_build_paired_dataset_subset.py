import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_paired_dataset_subset import load_eligible_rows, uniform_sample


class PairedDatasetSubsetTest(unittest.TestCase):
    def test_uniform_sample_includes_endpoints_without_duplicates(self):
        rows = [{"frame": str(index)} for index in range(11)]

        selected = uniform_sample(rows, 3)

        self.assertEqual([int(row["frame"]) for row in selected], [0, 5, 10])

    def test_uniform_sample_rejects_insufficient_rows(self):
        with self.assertRaisesRegex(ValueError, "need 4 eligible rows"):
            uniform_sample([{"frame": "1"}], 4)

    def test_load_eligible_rows_filters_flags_and_sync_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            points = root / "points.npz"
            image.write_bytes(b"present")
            points.write_bytes(b"present")
            manifest = root / "frames.csv"
            fields = [
                "frame",
                "camera_match_valid",
                "missing_image",
                "missing_lidar",
                "camera_lidar_dt_sec",
                "camera_image_path",
                "map_points_path",
                "lidar_timestamp",
                "camera_frame",
                "camera_timestamp",
                "map_x",
                "map_y",
                "map_yaw",
                "num_points",
            ]
            rows = [
                [
                    "1",
                    "1",
                    "0",
                    "0",
                    "0.010",
                    str(image),
                    str(points),
                    "1.0",
                    "3",
                    "1.01",
                    "0.0",
                    "0.0",
                    "0.0",
                    "1",
                ],
                [
                    "2",
                    "1",
                    "0",
                    "0",
                    "0.050",
                    str(image),
                    str(points),
                    "2.0",
                    "6",
                    "2.05",
                    "0.0",
                    "0.0",
                    "0.0",
                    "1",
                ],
            ]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)

            eligible = load_eligible_rows(manifest, root, 0.04)

            self.assertEqual([row["frame"] for row in eligible], ["1"])

    def test_build_subset_creates_self_contained_valid_archive(self):
        import json
        import tarfile

        import numpy as np
        from PIL import Image

        from scripts.build_paired_dataset_subset import build_subset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            manifest = root / "frames.csv"
            fields = [
                "frame",
                "camera_match_valid",
                "missing_image",
                "missing_lidar",
                "camera_lidar_dt_sec",
                "camera_image_path",
                "map_points_path",
                "lidar_timestamp",
                "camera_frame",
                "camera_timestamp",
                "map_x",
                "map_y",
                "map_yaw",
                "num_points",
            ]
            rows = []
            for frame in range(5):
                image = source / f"camera_{frame}.png"
                points = source / f"points_{frame}.npz"
                Image.new("RGB", (8, 6), (frame, 10, 20)).save(image)
                cloud = np.full((frame + 1, 4), frame, dtype=np.float32)
                np.savez(points, points_map_xyzi=cloud)
                rows.append(
                    [
                        str(frame),
                        "1",
                        "0",
                        "0",
                        "0.001",
                        str(image),
                        str(points),
                        str(frame),
                        str(frame * 3),
                        str(frame + 0.001),
                        "1.0",
                        "2.0",
                        "0.5",
                        str(len(cloud)),
                    ]
                )
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)

            output = root / "parking_dataset_3"
            archive = root / "parking_dataset_3.tar"
            summary = build_subset(manifest, root, output, archive, count=3)

            self.assertEqual(summary["pair_count"], 3)
            self.assertEqual(len(list((output / "images").glob("*.png"))), 3)
            self.assertEqual(len(list((output / "pointclouds").glob("*.npz"))), 3)
            with (output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                built_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [Path(row["image_path"]).stem for row in built_rows],
                [Path(row["pointcloud_path"]).stem for row in built_rows],
            )
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["pair_count"], 3)
            self.assertEqual(metadata["sync_abs_max_sec"], 0.001)
            with tarfile.open(archive, "r") as handle:
                names = handle.getnames()
            self.assertIn("parking_dataset_3/manifest.csv", names)


if __name__ == "__main__":
    unittest.main()
