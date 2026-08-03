import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import (
    FrameRecord,
    FreeEvidence,
    KnownSlot,
    OccupiedEvidence,
    ScopeStatus,
    StabilityEvidence,
)
from parking_slot_hybrid_3d.io import FrameLoadError
from parking_slot_hybrid_3d.pipeline import Hybrid3DPipeline
from parking_slot_hybrid_3d.reporting import (
    OutputContext,
    part2_candidate_in_forward_fov,
    write_pipeline_outputs,
)
from parking_slot_part2 import validate_queue


def rectangle(center_x: float, half_length: float = 2.0, half_width: float = 1.0) -> np.ndarray:
    return np.asarray(
        [
            [center_x - half_length, -half_width],
            [center_x + half_length, -half_width],
            [center_x + half_length, half_width],
            [center_x - half_length, half_width],
        ],
        dtype=np.float64,
    )


def known_slot(slot_id: str, center_x: float) -> KnownSlot:
    return KnownSlot(
        slot_id=slot_id,
        polygon_map=rectangle(center_x),
        core_polygon_map=rectangle(center_x, 1.8, 0.8),
        margin_polygon_map=rectangle(center_x, 2.5, 1.5),
        center_map=np.asarray([center_x, 0.0]),
        heading_deg=0.0,
    )


def frames(
    camera_image: Path | None = None,
    map_points_dir: Path | None = None,
) -> list[FrameRecord]:
    return [
        FrameRecord(
            frame_id=frame_id,
            map_x=-3.0,
            map_y=0.0,
            map_yaw=0.0,
            map_points_path=(map_points_dir / f"{frame_id:06d}.npz") if map_points_dir else Path(f"{frame_id:06d}.npz"),
            lidar_timestamp=100.0 + index,
            camera_frame=1000 + index if camera_image else None,
            camera_image_path=camera_image,
            camera_timestamp=99.99 + index if camera_image else None,
            camera_lidar_dt_sec=-0.01 if camera_image else None,
            camera_match_valid=camera_image is not None,
        )
        for index, frame_id in enumerate((1, 5, 9, 13, 17))
    ]


def flat_points() -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(-3.5, 3.5, 20), np.linspace(-1.5, 1.5, 10))
    return np.column_stack(
        [xs.ravel(), ys.ravel(), np.full(xs.size, -1.5), np.ones(xs.size)]
    )


class DictionaryProvider:
    def __init__(self, records: list[FrameRecord], *, fail: bool = False) -> None:
        self.records = {record.frame_id: flat_points() for record in records}
        self.fail = fail

    @property
    def stats(self) -> dict[str, int]:
        return {"cache_hits": 0, "cache_misses": len(self.records), "load_errors": int(self.fail)}

    def load(self, frame_id: int) -> np.ndarray:
        if self.fail:
            raise FrameLoadError(frame_id, Path(f"{frame_id:06d}.npz"), "corrupt_test_frame")
        return self.records[frame_id]


class Hybrid3DPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = frames()
        self.slots = [known_slot("slot_0001", 0.0), known_slot("slot_0002", 100.0)]
        self.config = Hybrid3DConfig()

    def test_full_pipeline_emits_one_decision_per_path_relevant_slot(self) -> None:
        result = Hybrid3DPipeline(
            self.slots,
            self.frames,
            DictionaryProvider(self.frames),
            1.0,
            self.config,
            phase="full",
        ).run()

        self.assertEqual(len(result.scopes), 2)
        self.assertEqual({item.slot_id for item in result.decisions}, {"slot_0001"})
        self.assertEqual(
            next(item for item in result.scopes if item.slot_id == "slot_0002").scope_status,
            ScopeStatus.OUT_OF_ROUTE,
        )
        self.assertEqual(len(result.traces), 2)

    def test_scope_and_occupied_phases_preserve_phase_boundaries(self) -> None:
        scope_result = Hybrid3DPipeline(
            self.slots,
            self.frames,
            DictionaryProvider(self.frames),
            1.0,
            self.config,
            phase="scope",
        ).run()
        occupied_result = Hybrid3DPipeline(
            self.slots,
            self.frames,
            DictionaryProvider(self.frames),
            1.0,
            self.config,
            phase="occupied",
        ).run()

        self.assertEqual(scope_result.decisions, ())
        self.assertNotEqual(occupied_result.decisions, ())
        self.assertNotIn("free", {item.state.value for item in occupied_result.decisions})

    def test_corrupt_near_route_frames_degrade_affected_slot_to_unknown(self) -> None:
        result = Hybrid3DPipeline(
            self.slots,
            self.frames,
            DictionaryProvider(self.frames, fail=True),
            1.0,
            self.config,
            phase="full",
        ).run()

        decision = result.decisions[0]
        self.assertEqual(decision.state.value, "unknown")
        self.assertEqual(decision.scope_status, ScopeStatus.PARTIAL_ROUTE)
        self.assertNotEqual(decision.state.value, "free")

    def test_evidence_evaluation_errors_are_quality_failures_and_skip_stability(self) -> None:
        cases = (
            (
                OccupiedEvidence(failures=("occupied_evaluation_error",)),
                FreeEvidence(strong=True, positive_geometry=True),
                "occupied_evaluation_error",
            ),
            (
                OccupiedEvidence(strong=True),
                FreeEvidence(failures=("free_evaluation_error",)),
                "free_evaluation_error",
            ),
        )
        for occupied, free, expected_failure in cases:
            with self.subTest(expected_failure=expected_failure):
                with (
                    patch(
                        "parking_slot_hybrid_3d.pipeline.evaluate_occupied",
                        return_value=occupied,
                    ),
                    patch(
                        "parking_slot_hybrid_3d.pipeline.evaluate_free_space",
                        return_value=free,
                    ),
                    patch(
                        "parking_slot_hybrid_3d.pipeline.evaluate_stability",
                        side_effect=AssertionError("stability must be skipped"),
                    ),
                ):
                    result = Hybrid3DPipeline(
                        self.slots,
                        self.frames,
                        DictionaryProvider(self.frames),
                        1.0,
                        self.config,
                        phase="full",
                    ).run()

                decision = result.decisions[0]
                self.assertEqual(decision.state.value, "unknown")
                self.assertEqual(decision.decision_reason, "quality_gate_failed")
                self.assertIn(expected_failure, decision.unknown_reasons)
                self.assertEqual(
                    result.traces[0]["stages"]["quality"]["status"],
                    "error",
                )

    def test_stability_evaluation_error_is_marked_in_trace_and_stays_unknown(self) -> None:
        occupied = OccupiedEvidence(
            strong=True,
            best_box=(("center_x_m", 0.0),),
        )
        stability = StabilityEvidence(
            total_variants=7,
            failures=("stability_evaluation_error:dyaw_minus",),
        )
        with (
            patch(
                "parking_slot_hybrid_3d.pipeline.evaluate_occupied",
                return_value=occupied,
            ),
            patch(
                "parking_slot_hybrid_3d.pipeline.evaluate_free_space",
                return_value=FreeEvidence(),
            ),
            patch(
                "parking_slot_hybrid_3d.pipeline.evaluate_stability",
                return_value=stability,
            ),
        ):
            result = Hybrid3DPipeline(
                self.slots,
                self.frames,
                DictionaryProvider(self.frames),
                1.0,
                self.config,
                phase="full",
            ).run()

        self.assertEqual(result.decisions[0].state.value, "unknown")
        self.assertEqual(
            result.decisions[0].decision_reason,
            "stability_evaluation_error",
        )
        self.assertEqual(result.traces[0]["stages"]["stability"]["status"], "error")

    def test_reporting_writes_canonical_artifacts_and_a_valid_filtered_queue(self) -> None:
        result = Hybrid3DPipeline(
            self.slots,
            self.frames,
            DictionaryProvider(self.frames),
            1.0,
            self.config,
            phase="full",
        ).run()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "run"
            write_pipeline_outputs(
                result,
                output_dir,
                self.config,
                OutputContext(dataset_id="synthetic-test"),
                overwrite=True,
            )

            expected = {
                "local_map.json",
                "local_map.png",
                "local_slot_database.json",
                "local_frame_manifest.json",
                "known_slot_scope.csv",
                "slot_decisions.csv",
                "slot_decisions.json",
                "unknown_agent_queue.json",
                "decision_trace.jsonl",
                "summary.json",
                "report.html",
            }
            self.assertTrue(expected <= {path.name for path in output_dir.iterdir()})
            queue_payload = json.loads((output_dir / "unknown_agent_queue.json").read_text())
            queue = validate_queue(queue_payload, base_dir=output_dir)
            self.assertEqual(queue.items, ())
            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(
                summary["metric_semantics"],
                "local_evidence_and_coverage_without_gt",
            )
            self.assertEqual(summary["map_semantics"], "local_incomplete_lidar_snapshot")
            self.assertEqual(summary["map_total"], 1)
            self.assertEqual(summary["known_slot_database_total"], 2)
            self.assertEqual(summary["unobserved_slots_omitted"], 1)

    def test_observable_filter_changes_queue_eligibility_not_decision_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "frame.png"
            image.write_bytes(b"synthetic-image")
            for frame_id in (1, 5, 9, 13, 17):
                np.savez_compressed(root / f"{frame_id:06d}.npz", points_map_xyzi=flat_points())
            camera_frames = frames(image, root)
            result = Hybrid3DPipeline(
                self.slots,
                camera_frames,
                DictionaryProvider(camera_frames),
                1.0,
                self.config,
                phase="full",
            ).run()
            output_dir = root / "run"
            write_pipeline_outputs(
                result,
                output_dir,
                self.config,
                OutputContext(dataset_id="synthetic-camera", map_points_dir=root),
                overwrite=True,
            )

            decision = result.decisions[0]
            queue = json.loads((output_dir / "unknown_agent_queue.json").read_text())
            self.assertEqual(decision.state.value, "unknown")
            self.assertEqual([item["slot_id"] for item in queue["items"]], ["slot_0001"])
            self.assertEqual(queue["items"][0]["audit"]["depth_capability"], "relative_only")
            self.assertEqual(
                queue["items"][0]["audit"]["candidate_policy"],
                "forward_field_of_regard",
            )
            self.assertEqual(
                queue["items"][0]["audit"]["candidate_total_fov_deg"],
                180.0,
            )
            validate_queue(queue, base_dir=output_dir)

    def test_part2_candidate_forward_fov_is_a_closed_180_degree_half_plane(self) -> None:
        frame = FrameRecord(
            frame_id=10,
            map_x=0.0,
            map_y=0.0,
            map_yaw=0.0,
            map_points_path=None,
        )
        for center, expected, bearing in (
            ((1.0, 0.0), True, 0.0),
            ((0.0, 1.0), True, 90.0),
            ((0.0, -1.0), True, -90.0),
            ((-1.0, 0.0), False, -180.0),
        ):
            with self.subTest(center=center):
                accepted, measured = part2_candidate_in_forward_fov(
                    frame, np.asarray(center), 90.0
                )
                self.assertEqual(accepted, expected)
                self.assertAlmostEqual(measured, bearing)

    def test_canonical_replay_hashes_ignore_the_output_directory_name(self) -> None:
        result = Hybrid3DPipeline(
            self.slots,
            self.frames,
            DictionaryProvider(self.frames),
            1.0,
            self.config,
            phase="full",
        ).run()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "primary"
            replay = root / "replay"
            for output_dir in (first, replay):
                write_pipeline_outputs(
                    result,
                    output_dir,
                    self.config,
                    OutputContext(dataset_id="deterministic-replay"),
                    overwrite=True,
                )

            canonical_files = (
                "local_map.json",
                "local_slot_database.json",
                "local_frame_manifest.json",
                "known_slot_scope.csv",
                "slot_decisions.csv",
                "slot_decisions.json",
                "unknown_agent_queue.json",
                "decision_trace.jsonl",
            )
            for filename in canonical_files:
                with self.subTest(filename=filename):
                    first_hash = hashlib.sha256((first / filename).read_bytes()).hexdigest()
                    replay_hash = hashlib.sha256((replay / filename).read_bytes()).hexdigest()
                    self.assertEqual(first_hash, replay_hash)


if __name__ == "__main__":
    unittest.main()
