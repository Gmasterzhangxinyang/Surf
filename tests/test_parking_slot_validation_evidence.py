import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from parking_slot_validation.evidence import (
    EvidenceSettings,
    FrameAssessment,
    build_manifest,
    draw_overlay,
    pose_prefilter,
    quality_reasons,
    select_evidence_frames,
)
from parking_slot_validation.models import Prediction


def assessment(
    frame: int,
    *,
    bearing: float = 0.0,
    score: float = 0.9,
    area: float = 1200.0,
    finite_vertices: int = 4,
    camera_dt: float = 0.005,
    image_exists: bool = True,
) -> FrameAssessment:
    return FrameAssessment(
        lidar_frame=frame,
        lidar_timestamp=float(frame) / 10.0,
        camera_frame=frame * 3,
        camera_timestamp=float(frame) / 10.0 + camera_dt,
        camera_lidar_dt_sec=camera_dt,
        camera_image_path=f"left{frame * 3:06d}.png",
        camera_image_exists=image_exists,
        bearing_deg=bearing,
        distance_m=5.0,
        projection_score=score,
        projected_area_px=area,
        finite_vertices=finite_vertices,
        target_polygon_uv=((20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)),
        adjacent_polygons_uv={
            "slot_adjacent": ((5.0, 25.0), (15.0, 25.0), (15.0, 70.0), (5.0, 70.0))
        },
    )


class EvidenceSelectionTest(unittest.TestCase):
    def test_quality_reasons_cover_all_trust_gates(self):
        row = assessment(
            10,
            bearing=42.0,
            score=0.4,
            area=100.0,
            finite_vertices=2,
            camera_dt=0.08,
            image_exists=False,
        )
        self.assertEqual(
            set(quality_reasons(row)),
            {
                "outside_safe_fov",
                "camera_timestamp_delta_too_large",
                "insufficient_projected_corners",
                "projection_score_too_low",
                "projected_area_too_small",
                "missing_camera_image",
            },
        )

    def test_pose_prefilter_searches_before_and_after_anchor(self):
        rows = {}
        for frame in range(80, 121, 5):
            rows[frame] = {
                "map_x": str((frame - 100) / 100.0),
                "map_y": "0.0",
                "map_yaw": "0.0",
                "camera_match_valid": "1",
            }
        rows[100]["camera_match_valid"] = "0"
        selected = pose_prefilter(
            rows,
            np.asarray([2.0, 0.0]),
            anchor_frame=100,
            search_before=20,
            search_after=20,
            frame_stride=5,
            half_fov_deg=40.0,
            limit=8,
        )
        self.assertTrue(any(frame < 100 for frame in selected))
        self.assertTrue(any(frame > 100 for frame in selected))
        self.assertNotIn(100, selected)
        self.assertEqual(selected, sorted(selected))

    def test_selection_spreads_before_center_and_after(self):
        rows = [assessment(frame=i, area=1000.0 + 100.0 * (5 - abs(15 - i))) for i in range(10, 20)]
        selected = select_evidence_frames(rows, max_frames=7, min_frames=3)
        frames = [row.lidar_frame for row in selected]
        self.assertEqual(len(selected), 7)
        self.assertEqual(frames, sorted(frames))
        self.assertIn(15, frames)
        self.assertTrue(any(frame < 15 for frame in frames))
        self.assertTrue(any(frame > 15 for frame in frames))

    def test_selection_rejects_bad_rows_and_abstains_below_minimum(self):
        rows = [assessment(10), assessment(11, bearing=50.0), assessment(12, finite_vertices=2)]
        self.assertEqual(select_evidence_frames(rows, max_frames=7, min_frames=3), [])

    def test_draw_overlay_marks_target_green_and_adjacent_blue(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            output = Path(tmp) / "overlay.png"
            Image.new("RGB", (100, 100), "black").save(source)
            draw_overlay(source, output, assessment(10), "slot_target")
            self.assertTrue(output.exists())
            pixels = np.asarray(Image.open(output).convert("RGB"))
            self.assertGreater(int(pixels[:, :, 1].max()), 200)
            self.assertGreater(int(pixels[:, :, 2].max()), 150)




    def test_manifest_builds_once_and_reuses_same_evidence_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "camera.png"
            Image.new("RGB", (100, 100), "black").save(image_path)
            frames_path = root / "frames.csv"
            fieldnames = [
                "frame", "map_x", "map_y", "map_yaw", "lidar_timestamp",
                "camera_frame", "camera_timestamp", "camera_lidar_dt_sec",
                "camera_match_valid", "camera_image_path", "map_points_path",
            ]
            with frames_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for frame in range(80, 121, 5):
                    writer.writerow(
                        {
                            "frame": frame,
                            "map_x": 0.0,
                            "map_y": 0.0,
                            "map_yaw": 0.0,
                            "lidar_timestamp": frame / 10.0,
                            "camera_frame": frame * 3,
                            "camera_timestamp": frame / 10.0 + 0.001,
                            "camera_lidar_dt_sec": 0.001,
                            "camera_match_valid": 1,
                            "camera_image_path": image_path,
                            "map_points_path": root / "unused.npz",
                        }
                    )
            slot_db_path = root / "slots.json"
            slot_db_path.write_text(
                json.dumps(
                    {
                        "map_units_per_meter": 1.0,
                        "slots": [
                            {
                                "slot_id": "slot_1",
                                "center_map": [2.0, 0.0],
                                "polygon_map": [[1.5, -0.5], [2.5, -0.5], [2.5, 0.5], [1.5, 0.5]],
                                "adjacent_slots": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            prediction = Prediction("run-a", "data-a", "slot_1", "occupied", 0.8, 100, {})

            def fake_assessor(frame_id, row, slot, slots, scale, settings, base_dir):
                return FrameAssessment(
                    lidar_frame=frame_id,
                    lidar_timestamp=float(row["lidar_timestamp"]),
                    camera_frame=int(row["camera_frame"]),
                    camera_timestamp=float(row["camera_timestamp"]),
                    camera_lidar_dt_sec=float(row["camera_lidar_dt_sec"]),
                    camera_image_path=str(image_path),
                    camera_image_exists=True,
                    bearing_deg=0.0,
                    distance_m=2.0,
                    projection_score=0.9,
                    projected_area_px=2000.0 - abs(frame_id - 100),
                    finite_vertices=4,
                    target_polygon_uv=((20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)),
                    adjacent_polygons_uv={},
                )

            output = root / "review"
            settings = EvidenceSettings(search_before=20, search_after=20, frame_stride=5, pose_prefilter_limit=20)
            first = build_manifest(
                [prediction], frames_path, slot_db_path, output, settings, assessor=fake_assessor
            )
            self.assertEqual(len(first["cases"]), 1)
            self.assertEqual(first["cases"][0]["review_status"], "reviewable")
            self.assertEqual(len(first["cases"][0]["evidence_frames"]), 7)
            for frame in first["cases"][0]["evidence_frames"]:
                self.assertTrue((output / frame["asset_path"]).exists())

            def should_not_run(*args, **kwargs):
                raise AssertionError("existing immutable evidence should be reused")

            reused = build_manifest(
                [Prediction("run-b", "data-a", "slot_1", "free", 0.3, 105, {})],
                frames_path,
                slot_db_path,
                output,
                settings,
                assessor=should_not_run,
            )
            self.assertEqual(reused["manifest_id"], first["manifest_id"])

    def test_rebuild_is_rejected_after_labels_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "evidence_manifest.json").write_text(
                json.dumps({"dataset_id": "data-a", "manifest_id": "old", "settings": {}, "cases": []}),
                encoding="utf-8",
            )
            (output / "human_labels.json").write_text(
                json.dumps({"labels": {"sample": {"human_label": "occupied"}}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "labels already exist"):
                build_manifest(
                    [],
                    output / "missing.csv",
                    output / "missing.json",
                    output,
                    EvidenceSettings(),
                    rebuild=True,
                )
