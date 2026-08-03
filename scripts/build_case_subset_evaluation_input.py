#!/usr/bin/env python3
"""Create a contract-valid evaluation input for an explicit SlotCase subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.contracts import Part1Output
from parking_slot_hybrid_3d.io import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--slot-id", action="append", dest="slot_ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested = tuple(dict.fromkeys(str(value) for value in args.slot_ids))
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cases = payload.get("slot_cases", [])
    by_id = {str(row.get("slot", {}).get("slot_id")): row for row in cases}
    missing = [slot_id for slot_id in requested if slot_id not in by_id]
    if missing:
        raise ValueError(f"requested SlotCases are missing: {missing}")

    requested_set = set(requested)
    candidate_ids = set(by_id)
    for slot in payload["scene"]["slots"]:
        slot_id = str(slot["slot_id"])
        if slot_id in candidate_ids and slot_id not in requested_set:
            slot["state"] = "occupied"
            slot["observed"] = True
    payload["slot_cases"] = [by_id[slot_id] for slot_id in requested]
    payload["producer"] = (
        str(payload.get("producer", "parking_slot_agent_v2"))
        + ":case_subset_evaluation_only"
    )
    validated = Part1Output.from_dict(payload)
    write_json_atomic(args.output, validated.to_dict())
    print(
        json.dumps(
            {
                "slot_ids": list(requested),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
