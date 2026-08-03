from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from parking_slot_part2.contracts import (
    EncounterRef,
    QueueEnvelope,
    QueueItem,
    RelationshipRef,
    ResourceRef,
    freeze_json,
)
from parking_slot_part2.queueing import canonical_sha256, make_task_id, validate_queue
from parking_slot_part2.shadow import (
    REGRESSION_SLOT_IDS,
    SHADOW_SELECTION_POLICY_VERSION,
    SHADOW_SELECTION_SCHEMA_VERSION,
    STRATUM_ORDER,
    build_shadow_subset,
    select_shadow_tasks,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
REASONS = {
    "conflict": "occupied_free_conflict",
    "pose": "pose_sensitive_terminal",
    "ownership_outside": "weak_shared_ownership",
    "boundary_static": "boundary_dominated",
    "observation_deficit": "insufficient_valid_frames",
}


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _producer() -> dict[str, str]:
    return {
        "pipeline": "slot_hybrid_3d",
        "run_id": "shadow-source-run",
        "config_hash": HASH_A,
        "dataset_id": "shadow-test-route",
        "slot_map_hash": HASH_B,
    }


def _resource(resource_id: str, kind: str, uri: str, identity: str) -> ResourceRef:
    return ResourceRef(
        resource_id=resource_id,
        kind=kind,
        uri=uri,
        sha256=None if kind == "map_points_root" else identity,
        manifest_hash=identity if kind == "map_points_root" else None,
        dataset_id="shadow-test-route",
        config_hash=HASH_A,
        slot_map_hash=HASH_B,
    )


def _item(
    slot_id: str,
    reason: str,
    occupied: float,
    free: float,
    *,
    encounter_id: str | None = None,
    conflicts: tuple[str, ...] = (),
    pointcloud: str = "lidar-selected",
) -> QueueItem:
    encounter = encounter_id or f"encounter-{slot_id}"
    return QueueItem(
        task_id=make_task_id(_producer(), slot_id, encounter),
        slot_id=slot_id,
        scope_status="in_route_scope",
        state="unknown",
        agent_observable=True,
        unknown_reasons=(reason,),
        priority="inspect_vehicle_shape_and_occlusion",
        available_modalities=("lidar",),
        suggested_tools=("inspect_lidar_map",),
        allowed_final_states=("occupied", "free", "unknown"),
        occupied_evidence=freeze_json({"strength": occupied}),
        free_evidence=freeze_json({"strength": free}),
        audit=freeze_json({}),
        relationships=RelationshipRef(
            adjacent_slot_ids=(),
            conflict_slot_ids=conflicts,
            shared_evidence_ids=(),
        ),
        encounter=EncounterRef(
            encounter_id=encounter,
            part1_trace_event_ids=(),
            start_lidar_frame=10,
            anchor_frame=10,
            end_lidar_frame=10,
            start_timestamp=1.0,
            anchor_timestamp=1.0,
            end_timestamp=1.0,
            support_frames=(10,),
            pointcloud_resource_ids=(pointcloud,),
            visual_frames=(),
        ),
    )


def _items() -> tuple[QueueItem, ...]:
    items: list[QueueItem] = [
        _item(slot_id, "fixed_regression_case", 0.01, 0.01)
        for slot_id in REGRESSION_SLOT_IDS
    ]
    strengths = (
        (0.99, 0.10),
        (0.98, 0.20),
        (0.10, 0.99),
        (0.20, 0.98),
        (0.50, 0.50),
        (0.40, 0.40),
    )
    for stratum in STRATUM_ORDER:
        for index, (occupied, free) in enumerate(strengths):
            items.append(
                _item(
                    f"slot_{stratum}_{index}",
                    REASONS[stratum],
                    occupied,
                    free,
                    pointcloud=(
                        "lidar-unselected" if stratum == "conflict" and index == 4
                        else "lidar-selected"
                    ),
                )
            )

    # A six-node chain makes build_groups produce overlapping groups.  Starting
    # at the fixed regression must therefore reach the tail by fixed-point
    # closure rather than by a single direct-neighbour expansion.
    chain_slots = (REGRESSION_SLOT_IDS[0], *(f"slot_closure_{i}" for i in range(1, 6)))
    chain_items: list[QueueItem] = []
    for index, slot_id in enumerate(chain_slots):
        neighbours: list[str] = []
        if index > 0:
            neighbours.append(chain_slots[index - 1])
        if index + 1 < len(chain_slots):
            neighbours.append(chain_slots[index + 1])
        if index == 0:
            original = next(item for item in items if item.slot_id == slot_id)
            chain_items.append(
                replace(
                    original,
                    task_id=make_task_id(_producer(), slot_id, "encounter-closure"),
                    relationships=replace(
                        original.relationships,
                        conflict_slot_ids=tuple(neighbours),
                    ),
                    encounter=replace(original.encounter, encounter_id="encounter-closure"),
                )
            )
        else:
            chain_items.append(
                _item(
                    slot_id,
                    "closure_context_only",
                    0.0,
                    0.0,
                    encounter_id="encounter-closure",
                    conflicts=tuple(neighbours),
                )
            )
    items = [item for item in items if item.slot_id != REGRESSION_SLOT_IDS[0]] + chain_items
    return tuple(items)


def _resources(root: Path) -> dict[str, ResourceRef]:
    artifacts = root / "artifacts"
    artifacts.mkdir()
    map_points = artifacts / "map_points"
    map_points.mkdir()
    resources: dict[str, ResourceRef] = {}
    definitions = {
        "slot-database": ("slot_database", "slot_database.json"),
        "corrected-frames": ("corrected_frames", "frames.csv"),
        "camera-calibration": ("camera_calibration", "camera.json"),
        "base-decisions": ("base_slot_decisions", "decisions.json"),
        "lidar-selected": ("pointcloud_artifact", "selected.npz"),
        "lidar-unselected": ("pointcloud_artifact", "unselected.npz"),
        "rgb-unreferenced": ("rgb_frame", "unused.png"),
    }
    for resource_id, (kind, name) in definitions.items():
        path = artifacts / name
        path.write_bytes(resource_id.encode("utf-8"))
        resources[resource_id] = _resource(
            resource_id,
            kind,
            f"artifacts/{name}",
            _file_hash(path),
        )
    resources["map-points-root"] = _resource(
        "map-points-root",
        "map_points_root",
        "artifacts/map_points",
        canonical_sha256({"frames": []}),
    )
    return resources


def _queue(root: Path) -> QueueEnvelope:
    return QueueEnvelope(
        schema_version="unknown-agent-queue/1.0",
        queue_id=canonical_sha256({"source": "shadow-test"}),
        producer=freeze_json(_producer()),
        resources=_resources(root),
        tool_registry_version="part2-tools/1.0",
        items=_items(),
    )


class Part2ShadowTest(unittest.TestCase):
    def test_shadow_selection_is_stratified_distinct_and_order_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = _queue(Path(directory))
            selection = select_shadow_tasks(queue)
            reordered = select_shadow_tasks(
                replace(queue, items=tuple(reversed(queue.items)))
            )

        self.assertEqual(selection, reordered)
        self.assertEqual(len(selection.seed_task_ids), 25)
        self.assertEqual(len(set(selection.seed_task_ids)), 25)
        self.assertEqual(
            tuple(record["slot_id"] for record in selection.seed_records[:5]),
            REGRESSION_SLOT_IDS,
        )

        for stratum in STRATUM_ORDER:
            with self.subTest(stratum=stratum):
                records = [
                    record
                    for record in selection.seed_records
                    if record.get("stratum") == stratum
                ]
                self.assertEqual(
                    [record["score_axis"] for record in records],
                    [
                        "occupied_strength",
                        "occupied_strength",
                        "free_strength",
                        "free_strength",
                    ],
                )
                self.assertEqual(
                    [record["slot_id"] for record in records],
                    [
                        f"slot_{stratum}_0",
                        f"slot_{stratum}_1",
                        f"slot_{stratum}_2",
                        f"slot_{stratum}_3",
                    ],
                )

        closure_slots = {
            item.slot_id
            for item in queue.items
            if item.task_id in set(selection.closure_added_task_ids)
        }
        self.assertEqual(
            closure_slots,
            {f"slot_closure_{index}" for index in range(1, 6)},
        )
        self.assertEqual(len(selection.group_expansions), 2)

    def test_shadow_subset_is_valid_absolute_filtered_and_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = _queue(root)
            subset = build_shadow_subset(queue, source_base_dir=root)

            # build_shadow_subset validates internally; validating the returned
            # wire payload again proves it is independently consumable.
            reparsed = validate_queue(subset.queue_payload)
            json.dumps(subset.queue_payload)
            json.dumps(subset.selection_manifest)

            self.assertEqual(reparsed.queue_id, subset.queue.queue_id)
            self.assertNotEqual(reparsed.queue_id, queue.queue_id)
            self.assertEqual(len(reparsed.items), 30)
            self.assertTrue(
                all(
                    Path(resource.uri).is_absolute()
                    for resource in reparsed.resources.values()
                )
            )
            self.assertIn("lidar-selected", reparsed.resources)
            self.assertNotIn("lidar-unselected", reparsed.resources)
            self.assertNotIn("rgb-unreferenced", reparsed.resources)
            self.assertTrue(
                {
                    "slot_database",
                    "corrected_frames",
                    "map_points_root",
                    "camera_calibration",
                    "base_slot_decisions",
                }.issubset(
                    {resource.kind for resource in reparsed.resources.values()}
                )
            )

            manifest = subset.selection_manifest
            self.assertEqual(
                manifest["schema_version"], SHADOW_SELECTION_SCHEMA_VERSION
            )
            self.assertEqual(
                manifest["policy_version"], SHADOW_SELECTION_POLICY_VERSION
            )
            self.assertEqual(manifest["source_queue_id"], queue.queue_id)
            self.assertEqual(manifest["subset_queue_id"], reparsed.queue_id)
            self.assertEqual(
                manifest["counts"],
                {
                    "fixed_regression_seeds": 5,
                    "stratified_seeds": 20,
                    "seed_tasks": 25,
                    "closure_added_tasks": 5,
                    "selected_tasks": 30,
                    "retained_resources": 6,
                },
            )
            identity_payload = dict(manifest)
            selection_id = identity_payload.pop("selection_id")
            self.assertEqual(selection_id, canonical_sha256(identity_payload))

    def test_shadow_selection_fails_closed_for_missing_seed_or_strength(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = _queue(Path(directory))
            missing_seed = replace(
                queue,
                items=tuple(
                    item
                    for item in queue.items
                    if item.slot_id != REGRESSION_SLOT_IDS[-1]
                ),
            )
            with self.assertRaisesRegex(ValueError, "missing fixed regression"):
                select_shadow_tasks(missing_seed)

            broken = next(
                item for item in queue.items if item.slot_id == "slot_conflict_0"
            )
            invalid_strength = replace(
                broken,
                occupied_evidence=freeze_json({"strength": math.nan}),
            )
            invalid_queue = replace(
                queue,
                items=tuple(
                    invalid_strength if item.task_id == broken.task_id else item
                    for item in queue.items
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "non-finite occupied_evidence.strength",
            ):
                select_shadow_tasks(invalid_queue)

    def test_shadow_subset_rejects_unresolvable_resource_uri(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = _queue(root)
            selected_resource = queue.resources["lidar-selected"]
            resources = dict(queue.resources)
            resources["lidar-selected"] = replace(
                selected_resource,
                uri="missing/selected.npz",
            )

            with self.assertRaisesRegex(ValueError, "cannot be resolved"):
                build_shadow_subset(
                    replace(queue, resources=resources),
                    source_base_dir=root,
                )


if __name__ == "__main__":
    unittest.main()
