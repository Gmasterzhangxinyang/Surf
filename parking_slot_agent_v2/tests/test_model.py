from __future__ import annotations

import unittest

from parking_slot_agent_v2.model import (
    ActionError,
    FinalAction,
    LocalVLMAdapter,
    ReplayModelAdapter,
    ToolAction,
    parse_action,
)


def _belief() -> dict[str, object]:
    return {
        "state": "unknown",
        "free_confidence": 0.4,
        "occupied_confidence": 0.4,
        "unknown_confidence": 0.6,
        "resolved_unknown_reasons": [],
        "remaining_unknown_reasons": ["needs_detail_evidence"],
    }


def _localization() -> dict[str, object]:
    return {
        "stage": "hypothesis",
        "hypothesis_id": "loc_h1",
        "target_side": "left",
        "depth_band": "far",
        "target_row": "front_left_row",
        "target_order_in_row": 4,
        "bbox_norm": [0.2, 0.3, 0.8, 0.9],
        "matched_landmarks": ["left_pillar_row", "neighboring_slots"],
        "missing_landmarks": [],
        "confidence_before": 0.0,
        "confidence_after": 0.62,
        "ambiguity_reasons": ["partial_pillar_occlusion"],
    }


class ModelActionTest(unittest.TestCase):
    def test_local_vlm_bounds_output_tokens(self) -> None:
        adapter = LocalVLMAdapter(
            base_url="http://127.0.0.1:8000/v1",
            model="Qwen/Qwen3-VL-4B-Instruct-FP8",
            max_output_tokens=1024,
        )
        self.assertEqual(adapter.max_output_tokens, 1024)
        with self.assertRaisesRegex(ValueError, "max_output_tokens"):
            LocalVLMAdapter(
                base_url="http://127.0.0.1:8000/v1",
                model="Qwen/Qwen3-VL-4B-Instruct-FP8",
                max_output_tokens=0,
            )

    def test_parses_camera_crop(self) -> None:
        action = parse_action(
            {
                "type": "tool",
                "tool": "camera_crop",
                "rationale": "inspect the likely third slot",
                "arguments": {
                    "bbox_norm": [0.2, 0.3, 0.8, 0.9],
                    "enhancement": "contrast",
                },
                "belief": _belief(),
                "localization": _localization(),
            }
        )
        self.assertIsInstance(action, ToolAction)
        self.assertEqual(action.arguments["bbox_norm"], (0.2, 0.3, 0.8, 0.9))
        self.assertEqual(action.localization.hypothesis_id, "loc_h1")
        self.assertEqual(action.localization.target_order_in_row, 4)

    def test_parses_camera_sequence_without_arguments(self) -> None:
        action = parse_action(
            {
                "type": "tool",
                "tool": "camera_sequence",
                "rationale": "inspect strictly causal temporal context",
                "arguments": {},
                "belief": _belief(),
            }
        )
        self.assertIsInstance(action, ToolAction)
        self.assertEqual(action.tool, "camera_sequence")

    def test_parses_final_with_separate_camera_confidences(self) -> None:
        action = parse_action(
            {
                "type": "final",
                "state": "free",
                "free_confidence": 0.94,
                "occupied_confidence": 0.06,
                "localization_confidence": 0.96,
                "occupancy_confidence": 0.95,
                "evidence_ids": ["ev_camera_1"],
                "reason": "target slot is localized and visibly empty",
                "reason_codes": ["camera_target_empty"],
            }
        )
        self.assertIsInstance(action, FinalAction)
        self.assertEqual(action.localization_confidence, 0.96)

    def test_rejects_bad_bbox_and_extra_fields(self) -> None:
        with self.assertRaises(ActionError):
            parse_action(
                {
                    "type": "tool",
                    "tool": "camera_crop",
                    "rationale": "bad",
                    "arguments": {
                        "bbox_norm": [0.8, 0.2, 0.1, 0.9],
                        "enhancement": "none",
                    },
                    "belief": _belief(),
                }
            )
        with self.assertRaises(ActionError):
            parse_action(
                {
                    "type": "tool",
                    "tool": "lidar_detail",
                    "rationale": "inspect geometry",
                    "arguments": {},
                    "belief": _belief(),
                    "path": "/tmp/not-allowed",
                }
            )

    def test_replay_is_per_case(self) -> None:
        payload = {
            "type": "tool",
            "tool": "lidar_detail",
            "rationale": "inspect geometry",
            "arguments": {},
            "belief": _belief(),
        }
        model = ReplayModelAdapter({"case-a": [payload]})
        returned = model.next_action({"case_id": "case-a"})
        self.assertEqual(returned, payload)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            model.next_action({"case_id": "case-a"})

    def test_replay_rebinds_final_evidence_ids_to_current_case(self) -> None:
        payload = {
            "type": "final",
            "state": "occupied",
            "free_confidence": 0.04,
            "occupied_confidence": 0.92,
            "localization_confidence": None,
            "occupancy_confidence": 0.92,
            "evidence_ids": ["ev_original_output_path"],
            "reason": "persistent target-local obstacle",
            "reason_codes": ["lidar_persistent_obstacle"],
        }
        model = ReplayModelAdapter({"case-a": [payload]})
        returned = model.next_action(
            {
                "case_id": "case-a",
                "observations": [
                    {"evidence_id": "ev_current_fov", "tool_name": "check_fov"},
                    {"evidence_id": "ev_current_lidar", "tool_name": "lidar_detail"},
                ],
            }
        )
        self.assertEqual(
            returned["evidence_ids"],
            ["ev_current_fov", "ev_current_lidar"],
        )
        self.assertEqual(payload["evidence_ids"], ["ev_original_output_path"])


if __name__ == "__main__":
    unittest.main()
