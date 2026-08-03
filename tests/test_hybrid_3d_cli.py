import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_hybrid_3d_slot_evidence.py"


class Hybrid3DCLITest(unittest.TestCase):
    def test_help_runs_when_invoked_by_path(self) -> None:
        completed = subprocess.run(
            ["python3", str(SCRIPT), "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--phase", completed.stdout)
        self.assertIn("--slot-ids", completed.stdout)
        self.assertIn("--anchor-frame", completed.stdout)
        self.assertIn("--local-frame-count", completed.stdout)

    def test_scope_cli_writes_scope_artifacts_without_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            points_dir = root / "map_points"
            points_dir.mkdir()
            frame_rows = []
            xs, ys = np.meshgrid(np.linspace(-3.5, 3.5, 20), np.linspace(-1.5, 1.5, 10))
            points = np.column_stack(
                [xs.ravel(), ys.ravel(), np.full(xs.size, -1.5), np.ones(xs.size)]
            )
            for index, frame_id in enumerate((1, 5, 9, 13, 17)):
                path = points_dir / f"{frame_id:06d}.npz"
                np.savez_compressed(path, points_map_xyzi=points)
                frame_rows.append(
                    {
                        "frame": frame_id,
                        "map_x": -3.0,
                        "map_y": 0.0,
                        "map_yaw": 0.0,
                        "map_points_path": str(path),
                        "lidar_timestamp": 100.0 + index,
                    }
                )
            frames_csv = root / "frames.csv"
            with frames_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=frame_rows[0].keys())
                writer.writeheader()
                writer.writerows(frame_rows)
            slot_db = root / "slots.json"
            slot_db.write_text(
                json.dumps(
                    {
                        "map_units_per_meter": 1.0,
                        "slot_count": 1,
                        "slots": [
                            {
                                "slot_id": "slot_0001",
                                "polygon_map": [[-2, -1], [2, -1], [2, 1], [-2, 1]],
                                "core_polygon_map": [[-1.8, -0.8], [1.8, -0.8], [1.8, 0.8], [-1.8, 0.8]],
                                "margin_polygon_map": [[-2.5, -1.5], [2.5, -1.5], [2.5, 1.5], [-2.5, 1.5]],
                                "center_map": [0, 0],
                                "heading_deg": 0,
                                "adjacent_slots": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "output"

            completed = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--frames-csv",
                    str(frames_csv),
                    "--slot-db",
                    str(slot_db),
                    "--map-points-dir",
                    str(points_dir),
                    "--output-dir",
                    str(output_dir),
                    "--phase",
                    "scope",
                    "--slot-ids",
                    "slot_0001",
                    "--max-slots",
                    "1",
                    "--overwrite",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output_dir / "known_slot_scope.csv").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "local_map.json").is_file())
            self.assertTrue((output_dir / "local_map.png").is_file())
            self.assertFalse((output_dir / "slot_decisions.json").exists())
            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["phase"], "scope")
            self.assertEqual(summary["map_semantics"], "local_incomplete_lidar_snapshot")
            self.assertEqual(summary["lidar_frame_count"], 5)


if __name__ == "__main__":
    unittest.main()
