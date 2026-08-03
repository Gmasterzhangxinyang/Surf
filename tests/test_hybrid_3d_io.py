import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parking_slot_hybrid_3d.io import (
    FrameLoadError,
    FramePointProvider,
    load_frame_records,
    load_known_slots,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)


FRAME_FIELDS = [
    "frame",
    "map_x",
    "map_y",
    "map_yaw",
    "map_points_path",
    "lidar_path",
    "lidar_timestamp",
    "camera_frame",
    "camera_image_path",
    "camera_timestamp",
    "camera_lidar_dt_sec",
    "camera_match_valid",
]


def write_frames(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def frame_row(frame: int, point_path: str = "") -> dict[str, str]:
    return {
        "frame": str(frame),
        "map_x": str(frame * 0.1),
        "map_y": "2.0",
        "map_yaw": "0.25",
        "map_points_path": point_path,
        "lidar_path": f"/lidar/{frame:06d}.bin",
        "lidar_timestamp": f"{1000 + frame}.0",
        "camera_frame": str(frame * 3),
        "camera_image_path": f"/images/left{frame * 3:06d}.png",
        "camera_timestamp": f"{1000 + frame}.01",
        "camera_lidar_dt_sec": "0.01",
        "camera_match_valid": "1",
    }


def slot_row(slot_id: str) -> dict[str, object]:
    return {
        "slot_id": slot_id,
        "polygon_map": [[-2.5, -1.0], [2.5, -1.0], [2.5, 1.0], [-2.5, 1.0]],
        "core_polygon_map": [[-2.0, -0.7], [2.0, -0.7], [2.0, 0.7], [-2.0, 0.7]],
        "margin_polygon_map": [[-2.8, -1.2], [2.8, -1.2], [2.8, 1.2], [-2.8, 1.2]],
        "center_map": [0.0, 0.0],
        "heading_deg": 0.0,
        "adjacent_slots": [],
    }


class Hybrid3DIOTest(unittest.TestCase):
    def test_frame_loader_sorts_and_parses_corrected_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.csv"
            write_frames(path, [frame_row(9), frame_row(3)])

            records = load_frame_records(path)

        self.assertEqual([record.frame_id for record in records], [3, 9])
        self.assertEqual(records[0].camera_frame, 9)
        self.assertTrue(records[0].camera_match_valid)
        self.assertAlmostEqual(records[0].camera_lidar_dt_sec, 0.01)
        self.assertIsNone(records[0].map_points_path)

    def test_frame_loader_rejects_duplicate_and_nonfinite_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.csv"
            write_frames(path, [frame_row(3), frame_row(3)])
            with self.assertRaisesRegex(ValueError, "duplicate frame_id 3"):
                load_frame_records(path)

            invalid = frame_row(4)
            invalid["map_x"] = "nan"
            write_frames(path, [invalid])
            with self.assertRaisesRegex(ValueError, "non-finite pose"):
                load_frame_records(path)

    def test_slot_loader_preserves_and_sorts_all_known_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slots.json"
            slots = [slot_row(f"slot_{index:04d}") for index in reversed(range(1397))]
            path.write_text(
                json.dumps({"map_units_per_meter": 0.05, "slot_count": 1397, "slots": slots}),
                encoding="utf-8",
            )

            loaded, scale = load_known_slots(path)

        self.assertEqual(len(loaded), 1397)
        self.assertEqual(loaded[0].slot_id, "slot_0000")
        self.assertEqual(loaded[-1].slot_id, "slot_1396")
        self.assertAlmostEqual(scale, 0.05)

    def test_slot_loader_rejects_duplicate_id_and_missing_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slots.json"
            duplicate = [slot_row("slot_1"), slot_row("slot_1")]
            path.write_text(json.dumps({"map_units_per_meter": 1.0, "slots": duplicate}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate slot_id slot_1"):
                load_known_slots(path)

            bad = slot_row("slot_bad")
            del bad["polygon_map"]
            path.write_text(json.dumps({"map_units_per_meter": 1.0, "slots": [bad]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "polygon_map"):
                load_known_slots(path)

    def test_point_provider_loads_npz_and_reports_bounded_cache_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_by_id = {}
            rows = []
            for frame in (1, 2):
                point_path = root / f"{frame:06d}.npz"
                np.savez(point_path, points_map_xyzi=np.asarray([[frame, 2.0, 3.0, 4.0]], dtype=np.float32))
                rows.append(frame_row(frame, str(point_path)))
            csv_path = root / "frames.csv"
            write_frames(csv_path, rows)
            frame_by_id = {record.frame_id: record for record in load_frame_records(csv_path)}
            provider = FramePointProvider(frame_by_id, root, cache_size=1)

            first = provider.load(1)
            provider.load(1)
            provider.load(2)

            self.assertEqual(first.shape, (1, 4))
            self.assertEqual(first.dtype, np.float64)
            self.assertEqual(provider.stats, {"cache_hits": 1, "cache_misses": 2, "load_errors": 0})
            self.assertEqual(provider.cached_frame_ids, (2,))

    def test_point_provider_raises_structured_errors_for_missing_and_malformed_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "frames.csv"
            missing = root / "missing.npz"
            malformed = root / "malformed.npz"
            np.savez(malformed, points_map_xyzi=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
            write_frames(csv_path, [frame_row(1, str(missing)), frame_row(2, str(malformed))])
            records = {record.frame_id: record for record in load_frame_records(csv_path)}
            provider = FramePointProvider(records, root)

            with self.assertRaises(FrameLoadError) as missing_error:
                provider.load(1)
            self.assertEqual(missing_error.exception.frame_id, 1)
            self.assertEqual(missing_error.exception.reason, "missing_map_points")

            with self.assertRaises(FrameLoadError) as malformed_error:
                provider.load(2)
            self.assertEqual(malformed_error.exception.reason, "invalid_map_points_shape")
            self.assertEqual(provider.stats["load_errors"], 2)

    def test_atomic_writers_produce_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"slot_id": "slot_2", "reasons": ["z", "a"], "frames": [9, 3]},
                {"slot_id": "slot_1", "reasons": ["b", "a"], "frames": [4, 2]},
            ]
            json_path = root / "rows.json"
            jsonl_path = root / "rows.jsonl"
            csv_path = root / "rows.csv"

            write_json_atomic(json_path, rows)
            first_json = json_path.read_bytes()
            write_json_atomic(json_path, list(reversed(rows)))
            write_jsonl_atomic(jsonl_path, rows)
            write_csv_atomic(csv_path, rows, ["slot_id", "reasons", "frames"])

            self.assertEqual(first_json, json_path.read_bytes())
            self.assertEqual(json.loads(json_path.read_text())[0]["slot_id"], "slot_1")
            self.assertEqual(json.loads(json_path.read_text())[0]["frames"], [2, 4])
            self.assertEqual(json.loads(jsonl_path.read_text().splitlines()[0])["slot_id"], "slot_1")
            self.assertEqual(csv_path.read_text().splitlines()[1].split(",", 1)[0], "slot_1")


if __name__ == "__main__":
    unittest.main()
