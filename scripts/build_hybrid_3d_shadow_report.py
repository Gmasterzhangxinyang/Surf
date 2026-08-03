#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_hybrid_3d.io import write_csv_atomic, write_json_atomic
from parking_slot_hybrid_3d.shadow import build_shadow_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a no-GT transition report between the 202-slot baseline and hybrid 3D shadow run."
    )
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--hybrid-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_shadow_comparison(
        _read_csv(args.baseline_csv),
        _read_csv(args.hybrid_output_dir / "known_slot_scope.csv"),
        _read_csv(args.hybrid_output_dir / "slot_decisions.csv"),
    )
    write_csv_atomic(
        args.output_dir / "common_baseline_transition.csv",
        comparison.common_rows,
        (
            "slot_id",
            "old_state",
            "old_score",
            "old_reason",
            "new_scope_status",
            "new_state",
            "new_decision_reason",
            "new_unknown_reasons",
            "migration_reasons",
        ),
    )
    write_csv_atomic(
        args.output_dir / "newly_evaluated_route_slots.csv",
        comparison.new_route_rows,
        (
            "slot_id",
            "new_scope_status",
            "new_state",
            "new_decision_reason",
            "new_unknown_reasons",
            "agent_observable",
        ),
    )
    write_json_atomic(args.output_dir / "shadow_summary.json", comparison.summary)
    print(json.dumps(comparison.summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
