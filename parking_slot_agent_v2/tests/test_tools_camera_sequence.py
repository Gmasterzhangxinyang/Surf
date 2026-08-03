from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from parking_slot_agent_v2.contracts import (
    ConfidenceScores,
    FovResult,
    MapSlot,
    SceneSnapshot,
    SensorFrame,
    SlotCase,
)
from parking_slot_agent_v2.tools import V2ToolSuite


class CameraSequenceToolTest(unittest.TestCase):
    def test_sequence_excludes_post_t0_camera_and_keeps_last_five(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames: list[SensorFrame] = []
            for frame_id in range(1, 16):
                image_path = root / f"camera_{frame_id:02d}.png"
                Image.new("RGB", (64, 36), (frame_id * 8, 20, 30)).save(image_path)
                camera_timestamp = (
                    15.01 if frame_id == 15 else float(frame_id) - 0.05
                )
                frames.append(
                    SensorFrame(
                        frame_id=frame_id,
                        lidar_timestamp=float(frame_id),
                        map_x=0.0,
                        map_y=0.0,
                        map_yaw_rad=0.0,
                        camera_frame_id=100 + frame_id,
                        camera_timestamp=camera_timestamp,
                        camera_image_path=str(image_path),
                        camera_lidar_dt_sec=camera_timestamp - float(frame_id),
                        camera_match_valid=True,
                    )
                )
            slot = MapSlot(
                slot_id="slot_a",
                polygon_map=[[3.0, -1.0], [3.0, 1.0], [6.0, 1.0], [6.0, -1.0]],
                center_map=[4.5, 0.0],
                heading_deg=0.0,
                state="unknown",
                observed=True,
            )
            scene = SceneSnapshot(
                snapshot_id="camera-sequence-fixture",
                anchor_frame_id=15,
                anchor_timestamp=15.0,
                anchor_pose_map=[0.0, 0.0, 0.0],
                radius_m=30.0,
                map_units_per_meter=1.0,
                coordinate_frame="fixture",
                slots=[slot],
                frames=frames,
            )
            scores = ConfidenceScores(0.3, 0.3, 0.7, calibrated=False)
            case = SlotCase(
                case_id="case-a",
                snapshot_id=scene.snapshot_id,
                snapshot_frame_id=15,
                snapshot_timestamp=15.0,
                snapshot_pose_map=[0.0, 0.0, 0.0],
                slot=slot,
                evidence_anchor_frame_id=15,
                evidence_frame_ids=list(range(1, 16)),
                part1_state="unknown",
                part1_scores=scores,
                current_state="unknown",
                current_scores=scores,
                decision_reason="fixture",
                unknown_reasons=["needs_detail"],
            )
            case.update_fov(
                FovResult(
                    visibility="visible",
                    confidence=0.9,
                    reason="fixture_visible",
                )
            )
            case.resources["camera_image_path"] = str(root / "camera_14.png")
            context = V2ToolSuite(max_sequence_frames=5).inspect_camera_context(
                scene,
                case,
                root / "context_output",
                1,
            )
            self.assertEqual(context.status, "ok")
            self.assertEqual(
                context.metadata["map_panel_source"],
                "dedicated_ego_semantic_map",
            )
            self.assertEqual(
                context.metadata["semantic_map_convention"]["map_left"],
                "camera_left",
            )
            self.assertEqual(len(context.metadata["model_image_paths"]), 2)

            crop = V2ToolSuite(max_sequence_frames=5).crop_camera(
                scene,
                case,
                root / "crop_output",
                2,
                [0.1, 0.2, 0.5, 0.9],
                "contrast",
            )
            self.assertEqual(crop.status, "ok")
            self.assertTrue(crop.metadata["agent_hypothesis_overlay"])
            self.assertFalse(crop.metadata["hypothesis_overlay_is_ground_truth"])
            self.assertEqual(len(crop.metadata["model_image_paths"]), 2)
            for path in crop.metadata["model_image_paths"]:
                self.assertTrue(Path(path).is_file())

            evidence = V2ToolSuite(max_sequence_frames=5).inspect_camera_sequence(
                scene,
                case,
                root / "output",
                1,
            )

            self.assertEqual(evidence.status, "ok")
            self.assertEqual(evidence.metadata["frame_count"], 5)
            self.assertEqual(
                [item["lidar_frame_id"] for item in evidence.metadata["frames"]],
                [10, 11, 12, 13, 14],
            )
            self.assertTrue(evidence.metadata["strictly_causal_to_t0"])
            self.assertTrue(Path(evidence.artifact_paths[0]).is_file())


if __name__ == "__main__":
    unittest.main()
