from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from parking_slot_agent_v2.agent import _available_tools, _validate_localization_action
from parking_slot_agent_v2.calibrated_tools import CalibratedCameraToolSuite
from parking_slot_agent_v2.contracts import FovResult
from parking_slot_agent_v2.model import parse_action
from parking_slot_agent_v2.tests.test_agent_pipeline import _scene_and_cases


def test_calibrated_contact_sheet_unlocks_agent_selected_crop() -> None:
    scene, cases = _scene_and_cases()
    case = cases[1]
    case.update_fov(
        FovResult(
            visibility="visible",
            confidence=0.95,
            reason="fixture_visible",
        )
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sheet = root / "slot_b.jpg"
        Image.new("RGB", (1280, 1498), (45, 55, 65)).save(sheet)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "slot_id": case.slot_id,
                            "frames": [100, 103, 106, 109, 112, 115, 118, 121],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        case.resources["calibrated_camera_sheet_path"] = str(sheet)
        case.resources["calibrated_camera_manifest_path"] = str(manifest)
        suite = CalibratedCameraToolSuite()

        context = suite.inspect_camera_context(scene, case, root / "context", 1)
        assert context.status == "ok"
        assert context.metadata["bbox_norm_coordinate_space"] == "calibrated_contact_sheet"
        assert context.metadata["contact_sheet_layout"]["tiles"][0][
            "lidar_frame_id"
        ] == 100
        case.record_evidence(context)
        assert "camera_crop" in _available_tools(case)
        action = parse_action(
            {
                "type": "tool",
                "tool": "camera_crop",
                "rationale": "zoom tile 1 without inventing metric depth",
                "arguments": {
                    "bbox_norm": [0.05, 0.05, 0.46, 0.26],
                    "enhancement": "sharpen",
                },
                "belief": {
                    "state": "unknown",
                    "free_confidence": 0.2,
                    "occupied_confidence": 0.3,
                    "unknown_confidence": 0.7,
                    "resolved_unknown_reasons": [],
                    "remaining_unknown_reasons": ["target_too_small"],
                },
                "localization": {
                    "stage": "hypothesis",
                    "hypothesis_id": "cyan_target_crop",
                    "target_side": "unknown",
                    "depth_band": "unknown",
                    "target_row": "tile_1",
                    "target_order_in_row": None,
                    "bbox_norm": [0.05, 0.05, 0.46, 0.26],
                    "matched_landmarks": ["cyan_target_polygon"],
                    "missing_landmarks": [],
                    "confidence_before": 0.4,
                    "confidence_after": 0.7,
                    "ambiguity_reasons": [],
                },
            }
        )
        _validate_localization_action(action, case)

        crop = suite.crop_camera(
            scene,
            case,
            root / "crop",
            2,
            [0.05, 0.05, 0.46, 0.26],
            "sharpen",
        )
        assert crop.status == "ok"
        assert crop.metadata["bbox_norm_coordinate_space"] == "calibrated_contact_sheet"
        assert crop.metadata["selected_tile_indices"] == [1]
        assert crop.metadata["selected_frame_ids"] == [100]
        assert crop.metadata["agent_selected_crop"] is True
        assert len(crop.metadata["model_image_paths"]) == 2
        assert all(Path(path).is_file() for path in crop.metadata["model_image_paths"])
