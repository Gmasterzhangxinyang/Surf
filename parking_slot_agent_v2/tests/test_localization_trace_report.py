from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from parking_slot_agent_v2.reporting_localization_trace import (
    build_localization_trace_report,
)


class LocalizationTraceReportTest(unittest.TestCase):
    def test_real_audited_trace_builds_visual_and_machine_readable_outputs(self) -> None:
        root = Path(__file__).resolve().parents[2]
        run = root / "outputs/parking_slot_agent_v2_frame_9277/openai_agent_localization_slot_1248_v1"
        part1 = root / "outputs/parking_slot_agent_v2_frame_9277/agent_localization_slot_1248_input.json"
        if not run.is_dir() or not part1.is_file():
            self.skipTest("real slot_1248 localization trace is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            result = build_localization_trace_report(
                run_dir=run,
                part1_path=part1,
                output_dir=directory,
                slot_id="slot_1248",
            )
            payload = json.loads(Path(result["trace"]).read_text(encoding="utf-8"))
            self.assertEqual(len(payload["turns"]), 4)
            self.assertEqual(payload["turns"][1]["localization"]["target_side"], "right")
            self.assertEqual(payload["turns"][2]["localization"]["target_side"], "left")
            self.assertEqual(payload["final_localization"]["stage"], "ambiguous")
            self.assertAlmostEqual(payload["final_localization"]["confidence_after"], 0.42)
            with Image.open(result["board"]) as image:
                self.assertEqual(image.size, (2400, 1760))
            text = Path(result["html"]).read_text(encoding="utf-8")
            self.assertIn("右侧 → 左侧", text)
            self.assertIn("不是Ground Truth", text)


if __name__ == "__main__":
    unittest.main()
