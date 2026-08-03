#!/usr/bin/env python3
"""Build a deterministic fail-closed Part2 replay for an Unknown queue.

This policy is the reproducible control used when the queue exposes no trusted
target-corresponded camera evidence.  It preserves every Part1 Unknown instead
of manufacturing a terminal label from ambiguous LiDAR geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_part2.grouping import build_groups  # noqa: E402
from parking_slot_part2.preflight import EvidenceCatalog  # noqa: E402
from parking_slot_part2.queueing import canonical_json_bytes, load_queue  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    queue = load_queue(args.queue)
    catalog = EvidenceCatalog(queue, base_dir=args.queue.parent)
    item_by_task = {item.task_id: item for item in queue.items}
    lidar_by_task = {
        record.task_id: evidence_id
        for evidence_id, record in catalog.entries.items()
        if record.tool_name == "inspect_lidar_map" and record.available
    }
    actions: dict[str, list[dict[str, object]]] = {}
    for group in build_groups(queue):
        group_actions: list[dict[str, object]] = []
        assessments = []
        for task_id in group.task_ids:
            item = item_by_task[task_id]
            evidence_id = lidar_by_task.get(task_id)
            if evidence_id is not None:
                group_actions.append(
                    {
                        "type": "tool_request",
                        "tool_name": "inspect_lidar_map",
                        "arguments": {"evidence_id": evidence_id},
                    }
                )
            assessments.append(
                {
                    "task_id": item.task_id,
                    "slot_id": item.slot_id,
                    "proposed_state": "unknown",
                    "semantic_finding": "unclear",
                    "target_visibility": "unknown",
                    "target_ownership": "uncertain",
                    "evidence_refs": [] if evidence_id is None else [evidence_id],
                    "reason_codes": [
                        "calibrated_target_camera_evidence_unavailable",
                        "fail_closed_on_ambiguous_lidar",
                    ],
                    "resolved_unknown_reasons": [],
                    "unresolved_blockers": list(item.unknown_reasons),
                }
            )
        group_actions.append(
            {
                "type": "final_proposal",
                "assessments": assessments,
            }
        )
        actions[group.group_id] = group_actions

    payload = {
        "schema_version": "part2-replay-actions/1.0",
        "actions": actions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload) + b"\n")
    policy_path = args.output.with_name("fail_closed_policy.json")
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "part2-fail-closed-policy/1.0",
                "id": "deterministic-fail-closed-no-trusted-camera/1.0",
                "terminal_claims_allowed": False,
                "reason": (
                    "No hash-audited target-corresponded camera modality is available; "
                    "ambiguous LiDAR alone cannot safely override Part1 Unknown."
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "queue_id": queue.queue_id,
                "group_count": len(actions),
                "assessment_count": len(queue.items),
                "output": str(args.output),
                "policy_manifest": str(policy_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
