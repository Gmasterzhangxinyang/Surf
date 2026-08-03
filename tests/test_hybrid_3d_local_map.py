import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from parking_slot_hybrid_3d.contracts import (
    DecisionState,
    FrameRecord,
    KnownSlot,
    ScopeEvidence,
    ScopeStatus,
    SlotDecision,
)
from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.local_map import (
    LocalMapConfig,
    build_local_map_snapshot,
    restrict_result_to_local_map,
    select_local_frame_window,
)
from parking_slot_hybrid_3d.pipeline import PipelineResult
from parking_slot_hybrid_3d.local_map_visualization import render_local_map
from parking_slot_hybrid_3d.reporting import _summary


def _frame(frame_id: int, x: float = 0.0, y: float = 0.0) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        map_x=x,
        map_y=y,
        map_yaw=0.0,
        map_points_path=None,
        lidar_timestamp=float(frame_id),
    )


def _slot(slot_id: str, x: float, y: float) -> KnownSlot:
    # Parking-slot depth is along map Y and the aisle runs along map X.
    polygon = np.asarray(
        [
            [x - 1.0, y - 2.0],
            [x + 1.0, y - 2.0],
            [x + 1.0, y + 2.0],
            [x - 1.0, y + 2.0],
        ],
        dtype=np.float64,
    )
    return KnownSlot(
        slot_id=slot_id,
        polygon_map=polygon,
        core_polygon_map=polygon,
        margin_polygon_map=polygon,
        center_map=np.asarray([x, y], dtype=np.float64),
        heading_deg=90.0,
    )


def _scope(
    slot_id: str,
    *,
    status: ScopeStatus = ScopeStatus.IN_ROUTE,
    evidence_frame: int | None = 3,
) -> ScopeEvidence:
    crossing_frames = () if evidence_frame is None else (evidence_frame,)
    return ScopeEvidence(
        slot_id=slot_id,
        scope_status=status,
        near_frames=crossing_frames,
        crossing_frames=crossing_frames,
        core_ray_coverage=0.8 if crossing_frames else 0.0,
        agent_observable=bool(crossing_frames),
    )


def _decision(
    slot_id: str,
    state: DecisionState,
    *,
    status: ScopeStatus = ScopeStatus.IN_ROUTE,
) -> SlotDecision:
    return SlotDecision(
        slot_id=slot_id,
        scope_status=status,
        state=state,
        decision_reason=(
            "partial_route_scope"
            if status is ScopeStatus.PARTIAL_ROUTE
            else f"synthetic_{state.value}_evidence"
        ),
        unknown_reasons=("partial_route_scope",)
        if status is ScopeStatus.PARTIAL_ROUTE
        else (),
        reference_frames=(3,),
    )


def _result(
    slots: list[KnownSlot],
    states: dict[str, DecisionState] | None = None,
    *,
    statuses: dict[str, ScopeStatus] | None = None,
    evidence_frames: dict[str, int | None] | None = None,
    frames: tuple[FrameRecord, ...] | None = None,
) -> PipelineResult:
    states = states or {}
    statuses = statuses or {}
    evidence_frames = evidence_frames or {}
    frame_records = frames or (_frame(1), _frame(2), _frame(3))
    scopes = tuple(
        _scope(
            slot.slot_id,
            status=statuses.get(slot.slot_id, ScopeStatus.IN_ROUTE),
            evidence_frame=evidence_frames.get(slot.slot_id, 3),
        )
        for slot in slots
    )
    decisions = tuple(
        _decision(
            slot.slot_id,
            state,
            status=statuses.get(slot.slot_id, ScopeStatus.IN_ROUTE),
        )
        for slot in slots
        if (state := states.get(slot.slot_id)) is not None
        and statuses.get(slot.slot_id, ScopeStatus.IN_ROUTE)
        is not ScopeStatus.OUT_OF_ROUTE
    )
    traces = tuple(
        {"trace_event_id": f"trace:{slot.slot_id}", "slot_id": slot.slot_id}
        for slot in slots
    )
    return PipelineResult(
        phase="full",
        scopes=scopes,
        decisions=decisions,
        traces=traces,
        processing_seconds=0.01,
        cache_stats={"loads": len(frame_records)},
        map_total=len(slots),
        frames=frame_records,
        slots=tuple(slots),
        all_slots=tuple(slots),
        map_units_per_meter=1.0,
    )


def _row(snapshot: dict, slot_id: str) -> dict:
    return next(item for item in snapshot["slots"] if item["slot_id"] == slot_id)


class Hybrid3DLocalMapTest(unittest.TestCase):
    def test_selects_sorted_causal_trailing_frame_window(self) -> None:
        frames = tuple(_frame(frame_id) for frame_id in range(110, 99, -1))

        selected = select_local_frame_window(
            frames,
            anchor_frame_id=108,
            frame_count=5,
        )

        self.assertEqual([frame.frame_id for frame in selected], [104, 105, 106, 107, 108])
        self.assertNotIn(109, {frame.frame_id for frame in selected})
        self.assertNotIn(110, {frame.frame_id for frame in selected})

    def test_local_window_rejects_fewer_than_three_causal_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three causal LiDAR frames"):
            select_local_frame_window(
                (_frame(10), _frame(11), _frame(12)),
                anchor_frame_id=11,
                frame_count=3,
            )

    def test_outside_radius_out_of_route_and_no_evidence_slots_are_hidden(self) -> None:
        local = _slot("local", 4.0, 4.0)
        far = _slot("far_global", 25.0, 4.0)
        out_of_route = _slot("near_but_out_of_route", 4.0, -4.0)
        unseen = _slot("near_without_local_evidence", 8.0, 4.0)
        result = _result(
            [local, far, out_of_route, unseen],
            {
                "local": DecisionState.UNKNOWN,
                "far_global": DecisionState.OCCUPIED,
                "near_without_local_evidence": DecisionState.FREE,
            },
            statuses={"near_but_out_of_route": ScopeStatus.OUT_OF_ROUTE},
            evidence_frames={"near_without_local_evidence": None},
        )

        snapshot = build_local_map_snapshot(result)

        self.assertEqual([item["slot_id"] for item in snapshot["slots"]], ["local"])
        serialized = json.dumps(snapshot, sort_keys=True)
        self.assertNotIn("far_global", serialized)
        self.assertNotIn("near_but_out_of_route", serialized)
        self.assertNotIn("near_without_local_evidence", serialized)
        self.assertFalse(snapshot["is_global_truth_map"])
        self.assertEqual(snapshot["counts"]["unobserved_slots_omitted"], 3)

    def test_partial_route_scope_is_exposed_as_unknown(self) -> None:
        partial = _slot("partial", 4.0, 4.0)
        result = _result(
            [partial],
            {"partial": DecisionState.UNKNOWN},
            statuses={"partial": ScopeStatus.PARTIAL_ROUTE},
        )

        snapshot = build_local_map_snapshot(result)
        row = _row(snapshot, "partial")

        self.assertEqual(row["state"], "unknown")
        self.assertEqual(row["decision_reason"], "partial_route_scope")
        self.assertEqual(snapshot["state_domain"], ["free", "occupied", "unknown"])
        self.assertIsNone(snapshot["candidate_slot_id"])
        self.assertIsNone(snapshot["candidate"])
        self.assertEqual(snapshot["provisional_candidates"], [])

    def test_free_slot_is_only_a_visualization_provisional_candidate(self) -> None:
        free = _slot("free_candidate", 4.0, 4.0)
        occupied = _slot("occupied_neighbor", 8.0, 4.0)
        result = _result(
            [free, occupied],
            {
                "free_candidate": DecisionState.FREE,
                "occupied_neighbor": DecisionState.OCCUPIED,
            },
        )

        snapshot = build_local_map_snapshot(result)

        self.assertIsNone(snapshot["candidate_slot_id"])
        self.assertIsNone(snapshot["candidate"])
        self.assertFalse(snapshot["candidate_selection"]["selected"])
        self.assertEqual(_row(snapshot, "free_candidate")["state"], "free")
        self.assertTrue(
            _row(snapshot, "free_candidate")["visualization_shortlist_eligible"]
        )
        self.assertFalse(
            _row(snapshot, "occupied_neighbor")["visualization_shortlist_eligible"]
        )
        self.assertEqual(snapshot["provisional_candidate_ids"], ["free_candidate"])
        record = snapshot["provisional_candidates"][0]
        self.assertEqual(record["slot_id"], "free_candidate")
        self.assertEqual(record["state"], "free")
        self.assertTrue(record["provisional"])
        self.assertTrue(record["visualization_only"])
        self.assertTrue(
            "final_selection" in record or "is_final_selection" in record
        )
        if "final_selection" in record:
            self.assertFalse(record["final_selection"])
        if "is_final_selection" in record:
            self.assertFalse(record["is_final_selection"])

    def test_unknown_is_provisional_but_occupied_is_excluded(self) -> None:
        unknown = _slot("unknown", 4.0, 4.0)
        occupied = _slot("occupied", 8.0, 4.0)
        result = _result(
            [unknown, occupied],
            {
                "unknown": DecisionState.UNKNOWN,
                "occupied": DecisionState.OCCUPIED,
            },
        )

        snapshot = build_local_map_snapshot(result)

        self.assertIsNone(snapshot["candidate_slot_id"])
        self.assertIsNone(snapshot["candidate"])
        self.assertEqual(snapshot["provisional_candidate_ids"], ["unknown"])
        self.assertEqual(
            [(item["slot_id"], item["state"]) for item in snapshot["provisional_candidates"]],
            [("unknown", "unknown")],
        )
        unknown_row = _row(snapshot, "unknown")
        occupied_row = _row(snapshot, "occupied")
        self.assertTrue(unknown_row["visualization_shortlist_eligible"])
        self.assertTrue(unknown_row["visualization_shortlist_metrics"]["reachable"])
        self.assertFalse(occupied_row["visualization_shortlist_eligible"])
        self.assertIn(
            "occupied_not_shortlistable",
            occupied_row["visualization_shortlist_metrics"]["rejection_reasons"],
        )

    def test_provisional_shortlist_is_capped_at_two_and_deterministic(self) -> None:
        occupied_nearest = _slot("occupied_nearest", 2.0, 4.0)
        free_near = _slot("free_near", 4.0, 4.0)
        unknown_near = _slot("unknown_near", 6.0, 4.0)
        free_farther = _slot("free_farther", 8.0, 4.0)
        slots = [occupied_nearest, free_near, unknown_near, free_farther]
        states = {
            "occupied_nearest": DecisionState.OCCUPIED,
            "free_near": DecisionState.FREE,
            "unknown_near": DecisionState.UNKNOWN,
            "free_farther": DecisionState.FREE,
        }

        forward = build_local_map_snapshot(_result(slots, states))
        reversed_input = build_local_map_snapshot(_result(list(reversed(slots)), states))

        self.assertEqual(forward["provisional_candidate_ids"], ["free_near", "unknown_near"])
        self.assertEqual(
            reversed_input["provisional_candidate_ids"],
            forward["provisional_candidate_ids"],
        )
        self.assertEqual(len(forward["provisional_candidates"]), 2)
        self.assertEqual(
            forward["provisional_candidate_display"]["eligible_local_slot_count"],
            3,
        )
        self.assertNotIn("occupied_nearest", forward["provisional_candidate_ids"])
        self.assertIsNone(forward["candidate_slot_id"])
        self.assertIsNone(forward["candidate"])
        self.assertFalse(forward["candidate_selection"]["selected"])
        self.assertFalse(
            forward["provisional_candidate_display"]["is_final_parking_decision"]
        )
        self.assertFalse(
            forward["provisional_candidate_display"]["consumed_by_part2"]
        )

    def test_visualization_shortlist_limit_cannot_exceed_two(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "visualization_shortlist_max_count must be 1 or 2"
        ):
            LocalMapConfig(visualization_shortlist_max_count=3).validate()

    def test_nearest_row_across_the_current_aisle_is_allowed(self) -> None:
        opposite_1 = _slot("opposite_1", 3.0, -4.0)
        opposite_2 = _slot("opposite_2", 8.0, -4.0)
        target = _slot("adjacent_free", 4.0, 4.0)
        same_row_neighbor = _slot("adjacent_neighbor", 8.0, 4.0)
        result = _result(
            [opposite_1, opposite_2, target, same_row_neighbor],
            {
                "opposite_1": DecisionState.OCCUPIED,
                "opposite_2": DecisionState.OCCUPIED,
                "adjacent_free": DecisionState.FREE,
                "adjacent_neighbor": DecisionState.OCCUPIED,
            },
        )

        snapshot = build_local_map_snapshot(result)
        metrics = _row(snapshot, "adjacent_free")["visualization_shortlist_metrics"]

        self.assertIsNone(snapshot["candidate_slot_id"])
        self.assertEqual(snapshot["provisional_candidate_ids"], ["adjacent_free"])
        self.assertLessEqual(metrics["row_distance"], 1)
        self.assertEqual(metrics["same_side_row_rank"], 0)
        self.assertEqual(metrics["intervening_slot_ids"], [])
        self.assertTrue(metrics["reachable"])

    def test_second_row_is_rejected_by_hard_row_distance(self) -> None:
        slots = [
            _slot("left_1", 3.0, -4.0),
            _slot("left_2", 8.0, -4.0),
            _slot("right_near_1", 2.0, 4.0),
            _slot("right_near_2", 7.0, 4.0),
            _slot("second_row_free", 4.0, 8.0),
            _slot("second_row_neighbor", 8.0, 8.0),
        ]
        states = {slot.slot_id: DecisionState.OCCUPIED for slot in slots}
        states["second_row_free"] = DecisionState.FREE

        snapshot = build_local_map_snapshot(_result(slots, states))
        row = _row(snapshot, "second_row_free")

        self.assertIsNone(snapshot["candidate_slot_id"])
        self.assertEqual(snapshot["provisional_candidates"], [])
        self.assertFalse(row["visualization_shortlist_eligible"])
        self.assertGreater(
            row["visualization_shortlist_metrics"]["row_distance"], 1
        )
        self.assertIn(
            "row_distance_exceeded",
            row["visualization_shortlist_metrics"]["geometry_rejection_reasons"],
        )

    def test_intervening_row_rejects_target_even_when_row_distance_is_one(self) -> None:
        slots = [
            _slot("middle_blocker", 2.0, 4.0),
            _slot("middle_neighbor", 7.0, 4.0),
            _slot("behind_middle_free", 4.0, 8.0),
            _slot("behind_middle_neighbor", 8.0, 8.0),
        ]
        states = {slot.slot_id: DecisionState.OCCUPIED for slot in slots}
        states["behind_middle_free"] = DecisionState.FREE

        snapshot = build_local_map_snapshot(_result(slots, states))
        metrics = _row(snapshot, "behind_middle_free")[
            "visualization_shortlist_metrics"
        ]

        self.assertEqual(metrics["row_distance"], 1)
        self.assertIn("middle_blocker", metrics["intervening_slot_ids"])
        self.assertIn("intervening_slot_row", metrics["rejection_reasons"])
        self.assertIn("not_adjacent_to_current_channel", metrics["rejection_reasons"])
        self.assertIsNone(snapshot["candidate_slot_id"])
        self.assertEqual(snapshot["provisional_candidates"], [])

    def test_unobserved_intervening_row_geometry_still_blocks_candidate(self) -> None:
        middle = _slot("unobserved_middle", 2.0, 4.0)
        middle_mate = _slot("unobserved_middle_mate", 7.0, 4.0)
        target = _slot("behind_unobserved_free", 4.0, 8.0)
        target_mate = _slot("behind_unobserved_mate", 8.0, 8.0)
        result = _result(
            [middle, middle_mate, target, target_mate],
            {
                "behind_unobserved_free": DecisionState.FREE,
                "behind_unobserved_mate": DecisionState.OCCUPIED,
            },
            evidence_frames={
                "unobserved_middle": None,
                "unobserved_middle_mate": None,
            },
        )

        snapshot = build_local_map_snapshot(result)
        metrics = _row(snapshot, "behind_unobserved_free")[
            "visualization_shortlist_metrics"
        ]

        self.assertEqual(
            {item["slot_id"] for item in snapshot["slots"]},
            {"behind_unobserved_free", "behind_unobserved_mate"},
        )
        self.assertIsNone(snapshot["candidate_slot_id"])
        self.assertIn("unobserved_middle", metrics["intervening_slot_ids"])
        self.assertIn("intervening_slot_row", metrics["rejection_reasons"])
        self.assertEqual(snapshot["provisional_candidates"], [])

    def test_unknown_shortlist_still_obeys_hard_distance_limits(self) -> None:
        target = _slot("too_far_unknown", 16.0, 4.0)
        row_neighbor = _slot("far_row_neighbor", 12.0, 4.0)
        result = _result(
            [target, row_neighbor],
            {
                "too_far_unknown": DecisionState.UNKNOWN,
                "far_row_neighbor": DecisionState.OCCUPIED,
            },
        )

        snapshot = build_local_map_snapshot(result)
        metrics = _row(snapshot, "too_far_unknown")[
            "visualization_shortlist_metrics"
        ]

        self.assertIsNone(snapshot["candidate_slot_id"])
        self.assertEqual(snapshot["provisional_candidates"], [])
        self.assertTrue(metrics["state_shortlistable"])
        self.assertFalse(metrics["reachable"])
        self.assertIn(
            "euclidean_distance_exceeded", metrics["geometry_rejection_reasons"]
        )
        self.assertIn(
            "along_aisle_distance_exceeded", metrics["geometry_rejection_reasons"]
        )

    def test_missing_row_topology_fails_closed(self) -> None:
        target = _slot("isolated_free", 4.0, 4.0)

        snapshot = build_local_map_snapshot(
            _result([target], {"isolated_free": DecisionState.FREE})
        )
        row = _row(snapshot, "isolated_free")

        self.assertIsNone(snapshot["candidate_slot_id"])
        self.assertEqual(snapshot["provisional_candidates"], [])
        self.assertFalse(row["visualization_shortlist_eligible"])
        self.assertEqual(
            row["visualization_shortlist_metrics"]["inferred_row_slot_count"], 1
        )
        self.assertIn(
            "row_topology_unresolved",
            row["visualization_shortlist_metrics"]["geometry_rejection_reasons"],
        )

    def test_restrict_result_removes_every_global_slot_artifact(self) -> None:
        local = _slot("local", 4.0, 4.0)
        leaked_global = _slot("global_truth_slot", 30.0, 4.0)
        result = _result(
            [local, leaked_global],
            {
                "local": DecisionState.FREE,
                "global_truth_slot": DecisionState.OCCUPIED,
            },
        )
        snapshot = {"slots": [{"slot_id": "local"}]}

        restricted = restrict_result_to_local_map(result, snapshot)

        self.assertEqual(restricted.map_total, 1)
        self.assertEqual([slot.slot_id for slot in restricted.slots], ["local"])
        self.assertEqual([slot.slot_id for slot in restricted.all_slots], ["local"])
        self.assertEqual([scope.slot_id for scope in restricted.scopes], ["local"])
        self.assertEqual([decision.slot_id for decision in restricted.decisions], ["local"])
        self.assertEqual([trace["slot_id"] for trace in restricted.traces], ["local"])
        self.assertNotIn("global_truth_slot", repr(restricted))

    def test_reporting_does_not_promote_provisional_to_formal_candidate(self) -> None:
        free = _slot("provisional_free", 4.0, 4.0)
        unknown = _slot("provisional_unknown", 8.0, 4.0)
        result = _result(
            [free, unknown],
            {
                "provisional_free": DecisionState.FREE,
                "provisional_unknown": DecisionState.UNKNOWN,
            },
        )
        snapshot = build_local_map_snapshot(result)

        self.assertEqual(len(snapshot["provisional_candidates"]), 2)
        summary = _summary(result, Hybrid3DConfig(), snapshot)

        self.assertIsNone(summary["candidate_slot_id"])
        self.assertIsNone(summary["candidate"])
        self.assertEqual(
            summary["provisional_candidate_ids"],
            snapshot["provisional_candidate_ids"],
        )
        self.assertEqual(
            summary["provisional_candidate_count"],
            len(snapshot["provisional_candidates"]),
        )
        self.assertNotEqual(
            summary["candidate_slot_id"], snapshot["provisional_candidate_ids"][0]
        )

    def test_local_visualization_uses_snapshot_and_writes_png(self) -> None:
        local_free = _slot("local_free", 4.0, 4.0)
        local_unknown = _slot("local_unknown", 8.0, 4.0)
        local_occupied = _slot("local_occupied", 12.0, 4.0)
        far_geometry = _slot("far_geometry_without_state", 100.0, 100.0)
        snapshot = build_local_map_snapshot(
            _result(
                [local_free, local_unknown, local_occupied, far_geometry],
                {
                    "local_free": DecisionState.FREE,
                    "local_unknown": DecisionState.UNKNOWN,
                    "local_occupied": DecisionState.OCCUPIED,
                    "far_geometry_without_state": DecisionState.OCCUPIED,
                },
            )
        )
        self.assertIsNone(snapshot["candidate_slot_id"])
        self.assertEqual(
            snapshot["provisional_candidate_ids"], ["local_free", "local_unknown"]
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "local_map.png"
            rendered = render_local_map(
                snapshot,
                [local_free, local_unknown, local_occupied, far_geometry],
                output,
            )

            self.assertEqual(rendered, output)
            self.assertGreater(output.stat().st_size, 1_000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_local_visualization_rejects_invalid_provisional_candidates(self) -> None:
        local_free = _slot("local_free", 4.0, 4.0)
        local_unknown = _slot("local_unknown", 8.0, 4.0)
        local_occupied = _slot("local_occupied", 12.0, 4.0)
        slots = [local_free, local_unknown, local_occupied]
        snapshot = build_local_map_snapshot(
            _result(
                slots,
                {
                    "local_free": DecisionState.FREE,
                    "local_unknown": DecisionState.UNKNOWN,
                    "local_occupied": DecisionState.OCCUPIED,
                },
            )
        )
        first, second = snapshot["provisional_candidates"]

        too_many = deepcopy(snapshot)
        third = deepcopy(first)
        third.update(
            {"display_rank": 3, "slot_id": "local_occupied", "state": "occupied"}
        )
        too_many["provisional_candidates"] = [first, second, third]

        duplicate = deepcopy(snapshot)
        duplicate["provisional_candidates"] = [first, deepcopy(first)]

        nonlocal_slot = deepcopy(snapshot)
        ghost = deepcopy(first)
        ghost["slot_id"] = "not_in_local_snapshot"
        nonlocal_slot["provisional_candidates"] = [ghost]

        occupied = deepcopy(snapshot)
        occupied_record = deepcopy(first)
        occupied_record.update({"slot_id": "local_occupied", "state": "occupied"})
        occupied["provisional_candidates"] = [occupied_record]

        state_mismatch = deepcopy(snapshot)
        mismatch = deepcopy(first)
        mismatch.update({"slot_id": "local_unknown", "state": "free"})
        state_mismatch["provisional_candidates"] = [mismatch]

        invalid_cases = (
            (too_many, "at most two"),
            (duplicate, "duplicate provisional candidate"),
            (nonlocal_slot, "must reference a local snapshot slot"),
            (occupied, "only free or unknown"),
            (state_mismatch, "state does not match local slot state"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (invalid, message) in enumerate(invalid_cases):
                with self.subTest(message=message):
                    output = Path(temporary) / f"invalid_{index}.png"
                    with self.assertRaisesRegex(ValueError, message):
                        render_local_map(invalid, slots, output)

    def test_local_visualization_keeps_legacy_free_candidate_compatible(self) -> None:
        local_free = _slot("legacy_free", 4.0, 4.0)
        local_mate = _slot("legacy_mate", 8.0, 4.0)
        snapshot = build_local_map_snapshot(
            _result(
                [local_free, local_mate],
                {
                    "legacy_free": DecisionState.FREE,
                    "legacy_mate": DecisionState.OCCUPIED,
                },
            )
        )
        legacy_snapshot = deepcopy(snapshot)
        legacy_snapshot.pop("provisional_candidates")
        legacy_snapshot["candidate_slot_id"] = "legacy_free"

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "legacy.png"
            render_local_map(legacy_snapshot, [local_free, local_mate], output)

            self.assertGreater(output.stat().st_size, 1_000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
