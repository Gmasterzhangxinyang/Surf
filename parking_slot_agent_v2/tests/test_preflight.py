from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from parking_slot_agent_v2.preflight import run_openai_preflight


class PreflightTest(unittest.TestCase):
    def test_existing_output_and_bad_key_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            key = root / "key"
            key.write_text("x", encoding="utf-8")
            os.chmod(key, 0o644)
            output = root / "output"
            output.mkdir()
            (output / "part2_result.json").write_text("{}", encoding="utf-8")
            missing_input = root / "missing.json"
            with patch("importlib.metadata.version", return_value="2.46.0"):
                result = run_openai_preflight(
                    part1_path=missing_input,
                    output_dir=output,
                    key_file=key,
                )
            failures = {
                item["name"] for item in result["checks"] if not item["ok"]
            }
            self.assertFalse(result["ok"])
            self.assertIn("api_key_file", failures)
            self.assertIn("part1_contract", failures)
            self.assertIn("output_isolation", failures)
            self.assertFalse(result["secret_recorded"])


if __name__ == "__main__":
    unittest.main()
