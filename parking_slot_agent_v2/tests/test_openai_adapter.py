from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from parking_slot_agent_v2.openai_adapter import (
    OpenAIResponsesAdapter,
    _normalize_structured_action,
    openai_action_schema,
)


def _structured_tool() -> dict[str, object]:
    return {
        "type": "tool",
        "tool": "camera_sequence",
        "rationale": "inspect temporal context",
        "arguments": {"bbox_norm": None, "enhancement": None},
        "belief": {
            "state": "unknown",
            "free_confidence": 0.4,
            "occupied_confidence": 0.4,
            "unknown_confidence": 0.6,
            "resolved_unknown_reasons": [],
            "remaining_unknown_reasons": ["needs_temporal_context"],
        },
        "state": None,
        "free_confidence": None,
        "occupied_confidence": None,
        "localization_confidence": None,
        "occupancy_confidence": None,
        "evidence_ids": None,
        "reason": None,
        "reason_codes": None,
    }


class OpenAIAdapterTest(unittest.TestCase):
    def test_retry_budget_is_bounded(self) -> None:
        adapter = OpenAIResponsesAdapter(max_retries=2)
        self.assertEqual(adapter.max_retries, 2)
        with self.assertRaisesRegex(ValueError, "max_retries"):
            OpenAIResponsesAdapter(max_retries=6)

    def test_schema_is_strict_root_object(self) -> None:
        schema = openai_action_schema()
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_structured_tool_is_normalized_to_runtime_contract(self) -> None:
        action = _normalize_structured_action(_structured_tool())
        self.assertEqual(
            action,
            {
                "type": "tool",
                "tool": "camera_sequence",
                "rationale": "inspect temporal context",
                "arguments": {},
                "belief": _structured_tool()["belief"],
            },
        )

    def test_missing_key_fails_before_network(self) -> None:
        adapter = OpenAIResponsesAdapter(model="gpt-test")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                adapter.next_action({"case_id": "case-1", "turn": 1})

    def test_key_file_requires_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "key"
            path.write_text("secret", encoding="utf-8")
            path.chmod(0o644)
            adapter = OpenAIResponsesAdapter(model="gpt-test", api_key_file=path)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "permissions"):
                    adapter._load_api_key()
            path.chmod(0o600)
            with patch.dict(os.environ, {}, clear=True):
                value, source = adapter._load_api_key()
            self.assertEqual(value, "secret")
            self.assertEqual(source, "file:permission_checked")


if __name__ == "__main__":
    unittest.main()
