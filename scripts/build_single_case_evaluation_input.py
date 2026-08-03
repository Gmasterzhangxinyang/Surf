#!/usr/bin/env python3
"""Create a contract-valid, explicitly evaluation-only single-case Part1 input."""

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
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cases = payload.get("slot_cases", [])
    selected = [row for row in cases if row.get("slot", {}).get("slot_id") == args.slot_id]
    if len(selected) != 1:
        raise ValueError("requested slot must identify exactly one candidate case")
    candidate_ids = {row["slot"]["slot_id"] for row in cases}
    for slot in payload["scene"]["slots"]:
        if slot["slot_id"] in candidate_ids and slot["slot_id"] != args.slot_id:
            slot["state"] = "occupied"
            slot["observed"] = True
    payload["slot_cases"] = selected
    payload["producer"] = str(payload.get("producer", "parking_slot_agent_v2")) + ":single_case_evaluation_only"
    validated = Part1Output.from_dict(payload)
    write_json_atomic(args.output, validated.to_dict())
    print(json.dumps({"slot_id": args.slot_id, "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
