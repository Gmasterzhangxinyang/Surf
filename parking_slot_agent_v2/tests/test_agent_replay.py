from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

from parking_slot_agent_v2.reporting_agent_replay import build_agent_replay


class AgentReplayReportTest(unittest.TestCase):
    def test_real_trace_builds_with_parseable_embedded_data(self) -> None:
        root = Path(__file__).resolve().parents[2]
        run = root / "outputs/parking_slot_agent_v2_frame_9277/openai_extended60_exhaustive_v1"
        if not run.is_dir():
            self.skipTest("real audited run is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            result = build_agent_replay(run_dir=run, output_dir=directory)
            report = Path(result["html"])
            text = report.read_text(encoding="utf-8")
            match = re.search(
                r'<script id="replay-data" type="application/json">(.*?)</script>',
                text,
                re.DOTALL,
            )
            self.assertIsNotNone(match)
            payload = json.loads(match.group(1))
            self.assertEqual(payload["overview"]["total_cases"], 24)
            self.assertEqual(payload["overview"]["unknown_inputs"], 22)
            self.assertEqual(payload["overview"]["resolved_unknown"], 9)
            self.assertAlmostEqual(payload["overview"]["unknown_resolution_rate"], 9 / 22)
            self.assertEqual(payload["overview"]["final_counts"], {"free": 4, "occupied": 7, "unknown": 13})
            self.assertEqual(len(payload["overview"]["all_cases"]), 24)
            self.assertEqual([row["final_state"] for row in payload["cases"]], ["free", "occupied", "unknown"])
            self.assertEqual([len(row["steps"]) for row in payload["cases"]], [4, 4, 6])
            self.assertIn("camera_context", json.dumps(payload["cases"][2], ensure_ascii=False))
            for case in payload["cases"]:
                lidar_step = next(step for step in case["steps"] if step["key"] == "lidar")
                self.assertIn("lidar_explained", lidar_step["image"])
                self.assertEqual(lidar_step["image_label"], "同一LiDAR证据的可解释重绘")
                for step in case["steps"]:
                    if step["image"]:
                        self.assertTrue((report.parent / step["image"]).is_file())
            markdown = Path(result["report"]).read_text(encoding="utf-8")
            self.assertIn("解决率 40.9%", markdown)
            self.assertIn("解决率不是准确率", markdown)
            self.assertEqual(len(result["decision_boards"]), 3)
            for board in result["decision_boards"]:
                self.assertTrue(Path(board).is_file())


if __name__ == "__main__":
    unittest.main()
