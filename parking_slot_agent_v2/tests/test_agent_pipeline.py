from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parking_slot_agent_v2.agent import (
    SingleSlotAgent,
    _available_tools,
    _image_paths,
    _validate_final,
    _validate_localization_action,
    _validate_lidar_terminal_geometry,
)
from parking_slot_agent_v2.contracts import (
    ConfidenceScores,
    EvidenceRecord,
    FovResult,
    FovVisibility,
    MapSlot,
    Part1Output,
    SceneSnapshot,
    SensorFrame,
    SlotCase,
)
from parking_slot_agent_v2.model import (
    ActionError,
    FinalAction,
    LocalizationEstimate,
    ReplayModelAdapter,
    parse_action,
)
from parking_slot_agent_v2.pipeline import run_candidate_queue
from parking_slot_agent_v2.tools import describe_lidar_capability


def _scores(free: float, occupied: float) -> ConfidenceScores:
    return ConfidenceScores(
        free_confidence=free,
        occupied_confidence=occupied,
        unknown_confidence=max(0.0, 1.0 - max(free, occupied)),
        calibrated=False,
    )


def _scene_and_cases() -> tuple[SceneSnapshot, list[SlotCase]]:
    frames = [
        SensorFrame(
            frame_id=index,
            lidar_timestamp=float(index),
            map_x=0.0,
            map_y=0.0,
            map_yaw_rad=0.0,
        )
        for index in range(1, 16)
    ]
    free_slot = MapSlot(
        slot_id="slot_free",
        polygon_map=[[3.0, -1.0], [3.0, 1.0], [6.0, 1.0], [6.0, -1.0]],
        center_map=[4.5, 0.0],
        heading_deg=0.0,
        state="free",
        observed=True,
    )
    unknown_slot = MapSlot(
        slot_id="slot_unknown",
        polygon_map=[[8.0, -1.0], [8.0, 1.0], [11.0, 1.0], [11.0, -1.0]],
        center_map=[9.5, 0.0],
        heading_deg=0.0,
        state="unknown",
        observed=True,
    )
    scene = SceneSnapshot(
        snapshot_id="fixture-t0",
        anchor_frame_id=15,
        anchor_timestamp=15.0,
        anchor_pose_map=[0.0, 0.0, 0.0],
        radius_m=30.0,
        map_units_per_meter=1.0,
        coordinate_frame="fixture_map",
        slots=[free_slot, unknown_slot],
        frames=frames,
    )
    free_case = SlotCase(
        case_id="case-free",
        snapshot_id=scene.snapshot_id,
        snapshot_frame_id=15,
        snapshot_timestamp=15.0,
        snapshot_pose_map=[0.0, 0.0, 0.0],
        slot=free_slot,
        evidence_anchor_frame_id=15,
        evidence_frame_ids=list(range(1, 16)),
        part1_state="free",
        part1_scores=_scores(0.92, 0.1),
        current_state="free",
        current_scores=_scores(0.92, 0.1),
        decision_reason="strong_free_space_evidence",
        unknown_reasons=[],
    )
    unknown_case = SlotCase(
        case_id="case-unknown",
        snapshot_id=scene.snapshot_id,
        snapshot_frame_id=15,
        snapshot_timestamp=15.0,
        snapshot_pose_map=[0.0, 0.0, 0.0],
        slot=unknown_slot,
        evidence_anchor_frame_id=15,
        evidence_frame_ids=list(range(1, 16)),
        part1_state="unknown",
        part1_scores=_scores(0.4, 0.5),
        current_state="unknown",
        current_scores=_scores(0.4, 0.5),
        decision_reason="weak_obstacle_evidence",
        unknown_reasons=["weak_obstacle_evidence"],
    )
    return scene, [free_case, unknown_case]


class FakeTools:
    def check_fov(self, scene, case):
        case.update_fov(
            FovResult(
                visibility=FovVisibility.VISIBLE,
                confidence=0.8,
                reason="map_bearing_inside",
                camera_frame_id=150,
                details={"method": "map_bearing_only"},
            )
        )
        return EvidenceRecord(
            evidence_id=f"fov-{case.slot_id}",
            tool_name="check_fov",
            round_index=0,
            status="ok",
            artifact_paths=[],
            summary="coarse visible",
            metadata={},
            modality="fov",
        )

    def inspect_camera_context(self, scene, case, output_dir, round_index):
        return EvidenceRecord(
            evidence_id=f"detail-{case.slot_id}",
            tool_name="camera_context",
            round_index=round_index,
            status="ok",
            artifact_paths=[],
            summary="target map and raw camera inspected",
            metadata={},
            modality="camera",
        )

    def crop_camera(self, *args, **kwargs):
        raise AssertionError("crop not expected")

    def inspect_lidar_detail(self, *args, **kwargs):
        raise AssertionError("lidar not expected")


class NotVisibleTools(FakeTools):
    def check_fov(self, scene, case):
        case.update_fov(
            FovResult(
                visibility=FovVisibility.NOT_VISIBLE,
                confidence=0.8,
                reason="well_outside_conservative_fov",
                camera_frame_id=150,
                details={"method": "map_bearing_only"},
            )
        )
        return EvidenceRecord(
            evidence_id=f"fov-{case.slot_id}",
            tool_name="check_fov",
            round_index=0,
            status="ok",
            artifact_paths=[],
            summary="coarse not visible",
            metadata={},
            modality="fov",
        )

    def inspect_lidar_detail(self, scene, case, output_dir, round_index):
        return EvidenceRecord(
            evidence_id=f"lidar-{case.slot_id}",
            tool_name="lidar_detail",
            round_index=round_index,
            status="unavailable",
            artifact_paths=[],
            summary="target-local lidar evidence is unavailable",
            metadata={"reason_codes": ["lidar_evidence_pack_missing"]},
            modality="lidar",
        )


class AgentPipelineTest(unittest.TestCase):
    def test_camera_legal_case_requires_context_before_other_tools(self) -> None:
        _, cases = _scene_and_cases()
        case = cases[1]
        case.update_fov(
            FovResult(
                visibility="visible",
                confidence=0.9,
                reason="fixture_visible",
            )
        )
        self.assertEqual(_available_tools(case), ("camera_context",))
        case.record_evidence(
            EvidenceRecord(
                evidence_id="camera-context",
                tool_name="camera_context",
                round_index=1,
                status="ok",
                artifact_paths=[],
                summary="semantic map and raw camera",
                metadata={},
                modality="camera",
            )
        )
        self.assertEqual(
            _available_tools(case),
            ("camera_sequence", "camera_crop"),
        )

    def test_uncertain_projection_gate_does_not_advertise_camera(self) -> None:
        _, cases = _scene_and_cases()
        case = cases[1]
        case.update_fov(
            FovResult(
                visibility="uncertain",
                confidence=0.0,
                reason="projection_input_unavailable",
            )
        )
        self.assertEqual(_available_tools(case), ())

    def test_failed_camera_context_does_not_unlock_dependent_tools(self) -> None:
        _, cases = _scene_and_cases()
        case = cases[1]
        case.update_fov(
            FovResult(
                visibility="visible",
                confidence=0.9,
                reason="fixture_visible",
            )
        )
        case.record_evidence(
            EvidenceRecord(
                evidence_id="camera-context-failed",
                tool_name="camera_context",
                round_index=1,
                status="failed",
                artifact_paths=[],
                summary="camera context render failed",
                metadata={},
                modality="camera",
            )
        )
        self.assertEqual(_available_tools(case), ())

    def test_unadvertised_sequence_is_rejected_before_context(self) -> None:
        _, cases = _scene_and_cases()
        case = cases[1]
        case.update_fov(
            FovResult(
                visibility="visible",
                confidence=0.9,
                reason="fixture_visible",
            )
        )
        action = parse_action(
            {
                "type": "tool",
                "tool": "camera_sequence",
                "rationale": "inspect temporal consistency",
                "arguments": {},
                "belief": {
                    "state": "unknown",
                    "free_confidence": 0.2,
                    "occupied_confidence": 0.2,
                    "unknown_confidence": 0.8,
                    "resolved_unknown_reasons": [],
                    "remaining_unknown_reasons": ["needs_temporal_check"],
                },
            }
        )
        with self.assertRaisesRegex(ActionError, "not currently available"):
            _validate_localization_action(action, case)

    def test_lidar_is_available_only_for_identity_bound_incremental_pack(self) -> None:
        _, cases = _scene_and_cases()
        case = cases[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack_path = root / "extended.npz"
            decision_path = root / "extended.json"
            selected = np.arange(1, 31, dtype=np.int64)
            np.savez(
                pack_path,
                selected_frames=selected,
                valid_frames=selected[::2],
            )
            decision_path.write_text(
                json.dumps(
                    {
                        "slot_id": case.slot_id,
                        "selected_frames": selected.tolist(),
                        "decision": {
                            "slot_id": case.slot_id,
                            "stability": {
                                "total_variants": 7,
                                "passing_variants": 7,
                                "stable": True,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            case.resources["extended_lidar_evidence_path"] = str(pack_path)
            case.resources["extended_lidar_decision_path"] = str(decision_path)
            capability = describe_lidar_capability(case)

        self.assertTrue(capability["available"])
        self.assertTrue(capability["is_incremental_over_part1"])
        self.assertEqual(capability["selected_frame_count"], 30)
        self.assertEqual(capability["valid_frame_count"], 15)
        self.assertTrue(capability["robustness_available"])
        self.assertEqual(capability["expected_information_gain"], "potential")

    def test_part1_pack_is_reviewable_but_not_a_part2_action(self) -> None:
        _, cases = _scene_and_cases()
        case = cases[1]
        with tempfile.TemporaryDirectory() as temporary:
            part1_path = Path(temporary) / "part1.npz"
            part1_path.write_bytes(b"fixture")
            case.resources["lidar_evidence_path"] = str(part1_path)
            capability = describe_lidar_capability(case)

        self.assertFalse(capability["available"])
        self.assertTrue(capability["review_available"])
        self.assertFalse(capability["is_incremental_over_part1"])
        self.assertEqual(capability["blocking_reason"], "same_as_part1")

    def test_camera_crop_requires_grounded_matching_localization_hypothesis(self) -> None:
        scene, cases = _scene_and_cases()
        case = cases[1]
        case.update_fov(
            FovResult(
                visibility="visible",
                confidence=0.9,
                reason="fixture_visible",
            )
        )
        case.record_evidence(
            EvidenceRecord(
                evidence_id="camera-context",
                tool_name="camera_context",
                round_index=1,
                status="ok",
                artifact_paths=[],
                summary="semantic map and raw camera",
                metadata={},
                modality="camera",
            )
        )
        raw = {
            "type": "tool",
            "tool": "camera_crop",
            "rationale": "verify the left-row hypothesis",
            "arguments": {
                "bbox_norm": [0.1, 0.2, 0.5, 0.8],
                "enhancement": "contrast",
            },
            "belief": {
                "state": "unknown",
                "free_confidence": 0.2,
                "occupied_confidence": 0.2,
                "unknown_confidence": 0.8,
                "resolved_unknown_reasons": [],
                "remaining_unknown_reasons": ["needs_crop_verification"],
            },
        }
        with self.assertRaisesRegex(ActionError, "explicit Camera localization"):
            _validate_localization_action(parse_action(raw), case)
        raw["localization"] = {
            "stage": "hypothesis",
            "hypothesis_id": "loc_h1",
            "target_side": "left",
            "depth_band": "far",
            "target_row": "front_left_row",
            "target_order_in_row": 4,
            "bbox_norm": [0.2, 0.2, 0.6, 0.8],
            "matched_landmarks": ["pillar_row"],
            "missing_landmarks": [],
            "confidence_before": 0.0,
            "confidence_after": 0.6,
            "ambiguity_reasons": [],
        }
        with self.assertRaisesRegex(ActionError, "bbox must match"):
            _validate_localization_action(parse_action(raw), case)
        raw["localization"]["bbox_norm"] = [0.1, 0.2, 0.5, 0.8]
        _validate_localization_action(parse_action(raw), case)
        case.record_evidence(
            EvidenceRecord(
                evidence_id="camera-crop-h1",
                tool_name="camera_crop",
                round_index=2,
                status="ok",
                artifact_paths=[],
                summary="first crop",
                metadata={
                    "bbox_norm": [0.1, 0.2, 0.5, 0.8],
                    "enhancement": "contrast",
                },
                modality="camera",
            )
        )
        self.assertIn("camera_crop", _available_tools(case))
        with self.assertRaisesRegex(ActionError, "must change bbox_norm or enhancement"):
            _validate_localization_action(parse_action(raw), case)
        raw["arguments"]["enhancement"] = "sharpen"
        _validate_localization_action(parse_action(raw), case)

    def test_final_after_camera_crop_must_report_verification_outcome(self) -> None:
        _, cases = _scene_and_cases()
        case = cases[1]
        case.update_fov(
            FovResult(
                visibility="visible",
                confidence=0.95,
                reason="fixture_visible",
            )
        )
        case.record_evidence(
            EvidenceRecord(
                evidence_id="crop",
                tool_name="camera_crop",
                round_index=1,
                status="ok",
                artifact_paths=[],
                summary="Agent-proposed target crop",
                metadata={},
                modality="camera",
            )
        )
        hypothesis = LocalizationEstimate(
            stage="hypothesis",
            hypothesis_id="loc_h1",
            target_side="left",
            depth_band="far",
            target_row="front_left_row",
            target_order_in_row=4,
            bbox_norm=(0.1, 0.2, 0.5, 0.8),
            matched_landmarks=("pillar_row",),
            missing_landmarks=(),
            confidence_before=0.5,
            confidence_after=0.95,
            ambiguity_reasons=(),
        )
        action = FinalAction(
            state="free",
            free_confidence=0.95,
            occupied_confidence=0.03,
            localization_confidence=0.95,
            occupancy_confidence=0.95,
            evidence_ids=("crop",),
            reason="fixture",
            reason_codes=("fixture",),
            localization=hypothesis,
        )
        with self.assertRaisesRegex(ActionError, "supported, refuted, or ambiguous"):
            _validate_final(action, case)

    @staticmethod
    def _terminal_action(state: str) -> FinalAction:
        return FinalAction(
            state=state,
            free_confidence=0.95 if state == "free" else 0.03,
            occupied_confidence=0.95 if state == "occupied" else 0.03,
            localization_confidence=None,
            occupancy_confidence=0.95,
            evidence_ids=("lidar",),
            reason="fixture",
            reason_codes=("fixture",),
        )

    @staticmethod
    def _geometry_evidence(card: dict) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id="lidar",
            tool_name="lidar_detail",
            round_index=1,
            status="ok",
            artifact_paths=[],
            summary="fixture",
            metadata={"geometry_card": card},
            modality="lidar",
        )

    def test_geometry_hard_gate_accepts_only_stable_strong_free(self) -> None:
        card = {
            "free_geometry": {
                "support_level": "strong_free_geometry_candidate",
                "strong_gate": True,
                "failures": [],
                "unresolved_core_hit": False,
            },
            "occupied_geometry": {
                "strong_gate": False,
                "failures": ["boundary_dominated"],
            },
            "robustness": {
                "stable": True,
                "passing_variants": 7,
                "total_variants": 7,
            },
        }
        _validate_lidar_terminal_geometry(
            self._terminal_action("free"),
            (self._geometry_evidence(card),),
        )
        card["robustness"]["passing_variants"] = 5
        with self.assertRaisesRegex(ActionError, "hard gates"):
            _validate_lidar_terminal_geometry(
                self._terminal_action("free"),
                (self._geometry_evidence(card),),
            )

    def test_geometry_hard_gate_vetoes_boundary_occupied(self) -> None:
        card = {
            "free_geometry": {"strong_gate": False},
            "occupied_geometry": {
                "support_level": "boundary_dominated_not_terminal",
                "strong_gate": False,
                "failures": ["boundary_dominated"],
                "boundary_ratio": 1.0,
                "linearity_risk": 0.0,
                "core_point_count": 0,
            },
            "robustness": {
                "stable": True,
                "passing_variants": 7,
                "total_variants": 7,
            },
        }
        with self.assertRaisesRegex(ActionError, "hard gates"):
            _validate_lidar_terminal_geometry(
                self._terminal_action("occupied"),
                (self._geometry_evidence(card),),
            )

    def test_explicit_model_image_does_not_duplicate_audit_artifacts(self) -> None:
        _, cases = _scene_and_cases()
        case = cases[1]
        case.resources.update(
            {
                "raw": "/tmp/raw.png",
                "card": "/tmp/card.png",
                "combined": "/tmp/combined.png",
            }
        )
        case.record_evidence(
            EvidenceRecord(
                evidence_id="lidar-composite",
                tool_name="lidar_detail",
                round_index=1,
                status="ok",
                artifact_paths=["/tmp/raw.png", "/tmp/card.png", "/tmp/combined.png"],
                summary="fixture",
                metadata={"model_image_paths": ["/tmp/combined.png"]},
                modality="lidar",
                resource_keys=["raw", "card", "combined"],
            )
        )
        self.assertEqual(_image_paths(case), (Path("/tmp/combined.png"),))

    def test_free_first_and_global_early_stop(self) -> None:
        scene, cases = _scene_and_cases()
        part1 = Part1Output(scene=scene, slot_cases=cases)
        model = ReplayModelAdapter(
            {
                "case-free": [
                    {
                        "type": "tool",
                        "tool": "camera_context",
                        "rationale": "locate the map-marked target in the raw image",
                        "arguments": {},
                        "belief": {
                            "state": "unknown",
                            "free_confidence": 0.6,
                            "occupied_confidence": 0.3,
                            "unknown_confidence": 0.4,
                            "resolved_unknown_reasons": [],
                            "remaining_unknown_reasons": ["needs_camera_detail"],
                        },
                    },
                    {
                        "type": "final",
                        "state": "free",
                        "free_confidence": 0.95,
                        "occupied_confidence": 0.05,
                        "localization_confidence": 0.96,
                        "occupancy_confidence": 0.95,
                        "evidence_ids": ["detail-slot_free"],
                        "reason": "target localized and empty",
                        "reason_codes": ["camera_target_empty"],
                        "localization": {
                            "stage": "supported",
                            "hypothesis_id": "loc_free_1",
                            "target_side": "left",
                            "depth_band": "middle",
                            "target_row": "front_left_row",
                            "target_order_in_row": 2,
                            "bbox_norm": [0.1, 0.2, 0.5, 0.8],
                            "matched_landmarks": ["pillar_row", "neighbor_slot"],
                            "missing_landmarks": [],
                            "confidence_before": 0.7,
                            "confidence_after": 0.96,
                            "ambiguity_reasons": [],
                        },
                    },
                ]
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = run_candidate_queue(
                part1,
                SingleSlotAgent(model, FakeTools()),  # type: ignore[arg-type]
                Path(temporary),
            )
            self.assertTrue((Path(temporary) / "part2_result.json").is_file())

        self.assertEqual(result.selected_slot_id, "slot_free")
        self.assertEqual(result.processed_case_ids, ("case-free",))
        self.assertEqual(cases[0].status.value, "selected")
        self.assertEqual(cases[1].status.value, "pending")

    def test_exhaustive_mode_checkpoints_every_case_and_can_resume(self) -> None:
        scene, cases = _scene_and_cases()
        part1 = Part1Output(scene=scene, slot_cases=cases)
        model = ReplayModelAdapter(
            {
                "case-free": [
                    {
                        "type": "tool",
                        "tool": "camera_context",
                        "rationale": "inspect target",
                        "arguments": {},
                        "belief": {
                            "state": "unknown",
                            "free_confidence": 0.6,
                            "occupied_confidence": 0.2,
                            "unknown_confidence": 0.4,
                            "resolved_unknown_reasons": [],
                            "remaining_unknown_reasons": ["needs_detail"],
                        },
                    },
                    {
                        "type": "final",
                        "state": "free",
                        "free_confidence": 0.95,
                        "occupied_confidence": 0.05,
                        "localization_confidence": 0.96,
                        "occupancy_confidence": 0.95,
                        "evidence_ids": ["detail-slot_free"],
                        "reason": "target empty",
                        "reason_codes": ["camera_target_empty"],
                    },
                ],
                "case-unknown": [
                    {
                        "type": "tool",
                        "tool": "camera_context",
                        "rationale": "inspect target",
                        "arguments": {},
                        "belief": {
                            "state": "unknown",
                            "free_confidence": 0.4,
                            "occupied_confidence": 0.4,
                            "unknown_confidence": 0.6,
                            "resolved_unknown_reasons": [],
                            "remaining_unknown_reasons": ["ambiguous"],
                        },
                    },
                    {
                        "type": "final",
                        "state": "unknown",
                        "free_confidence": 0.4,
                        "occupied_confidence": 0.4,
                        "localization_confidence": None,
                        "occupancy_confidence": None,
                        "evidence_ids": ["detail-slot_unknown"],
                        "reason": "still ambiguous",
                        "reason_codes": ["ambiguous"],
                    },
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = run_candidate_queue(
                part1,
                SingleSlotAgent(model, FakeTools()),  # type: ignore[arg-type]
                output,
                stop_at_first_free=False,
            )
            self.assertEqual(result.processed_case_ids, ("case-free", "case-unknown"))
            self.assertEqual(result.stop_reason, "evaluation_queue_exhausted")
            fresh_scene, fresh_cases = _scene_and_cases()
            resumed = run_candidate_queue(
                Part1Output(scene=fresh_scene, slot_cases=fresh_cases),
                SingleSlotAgent(ReplayModelAdapter({}), FakeTools()),  # type: ignore[arg-type]
                output,
                stop_at_first_free=False,
                resume=True,
            )
        self.assertEqual(resumed.processed_case_ids, result.processed_case_ids)
        self.assertTrue(resumed.evaluation_exhaustive)

    def test_not_visible_can_finish_unknown_when_no_tools_remain(self) -> None:
        scene, cases = _scene_and_cases()
        case = cases[1]
        model = ReplayModelAdapter(
            {
                "case-unknown": [
                    {
                        "type": "final",
                        "state": "unknown",
                        "free_confidence": 0.35,
                        "occupied_confidence": 0.45,
                        "localization_confidence": None,
                        "occupancy_confidence": None,
                        "evidence_ids": ["fov-slot_unknown"],
                        "reason": "Camera is outside and no incremental LiDAR exists",
                        "reason_codes": ["no_useful_tools_remaining"],
                    }
                ]
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = SingleSlotAgent(model, NotVisibleTools()).run(  # type: ignore[arg-type]
                scene,
                case,
                Path(temporary),
            )

        self.assertEqual(result.stop_reason, "no_useful_tools_remaining")
        self.assertEqual(result.tool_rounds, 0)
        self.assertEqual(case.status.value, "exhausted")
        self.assertEqual(case.final_state.value, "unknown")
        request = model.requests[0]
        self.assertEqual(request["available_tools"], [])
        self.assertTrue(request["budget"]["no_useful_tools_remaining"])
        self.assertFalse(request["budget"]["part2_detail_attempt_required"])
        self.assertEqual(
            request["tool_capabilities"]["lidar_detail"]["blocking_reason"],
            "lidar_evidence_pack_missing",
        )


if __name__ == "__main__":
    unittest.main()
