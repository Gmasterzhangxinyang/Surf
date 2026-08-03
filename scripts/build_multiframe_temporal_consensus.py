#!/usr/bin/env python3
"""Build per-slot temporal consensus from multi-frame accumulation probe outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIRS = [
    Path("outputs/multiframe_accumulation_probe_900_1200_eps025_ms8"),
    Path("outputs/multiframe_accumulation_probe_5600_5900_eps025_ms8"),
    Path("outputs/multiframe_accumulation_probe_6300_6600_eps025_ms8"),
]
DEFAULT_OUTPUT = Path("outputs/multiframe_temporal_consensus_eps025_ms8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-slot temporal consensus from accumulated slot evidence")
    parser.add_argument("--input-dirs", type=Path, nargs="+", default=DEFAULT_INPUT_DIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-core-supported-anchors", type=int, default=3)
    parser.add_argument("--min-core-support-ratio", type=float, default=0.35)
    parser.add_argument("--min-owner-stability", type=float, default=0.80)
    parser.add_argument("--max-conflict-ratio", type=float, default=0.45)
    parser.add_argument("--min-score", type=float, default=0.70)
    parser.add_argument("--review-limit", type=int, default=80)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_inputs(input_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        evidence_path = input_dir / "accumulated_slot_evidence.csv"
        summary_path = input_dir / "accumulated_summary.json"
        if not evidence_path.exists():
            raise SystemExit(f"missing evidence file: {evidence_path}")
        source_name = input_dir.name
        source_rows = read_csv_rows(evidence_path)
        for row in source_rows:
            row = dict(row)
            row["source_window"] = source_name
            row["source_dir"] = str(input_dir)
            rows.append(row)
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        sources.append(
            {
                "source_window": source_name,
                "source_dir": str(input_dir),
                "row_count": len(source_rows),
                "summary": summary,
            }
        )
    return rows, sources


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def i(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def consensus_state(
    core_supported_anchors: int,
    observed_anchors: int,
    conflict_anchors: int,
    owner_stability: float,
    max_score: float,
    args: argparse.Namespace,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    support_ratio = core_supported_anchors / max(observed_anchors, 1)
    conflict_ratio = conflict_anchors / max(observed_anchors, 1)
    if core_supported_anchors < args.min_core_supported_anchors:
        reasons.append("insufficient repeated core-supported anchors")
    if support_ratio < args.min_core_support_ratio:
        reasons.append("core-supported ratio below threshold")
    if owner_stability < args.min_owner_stability:
        reasons.append("dominant owner is not stable")
    if conflict_ratio > args.max_conflict_ratio:
        reasons.append("boundary/adjacent conflict ratio too high")
    if max_score < args.min_score:
        reasons.append("vehicle-like score below threshold")
    if not reasons:
        return "possible_occupied_by_multiframe_accumulation", [
            "slot has repeated accumulated vehicle core support",
            "dominant ownership is stable across anchor windows",
            "conflict ratio is below threshold",
        ]
    if core_supported_anchors > 0 and conflict_ratio <= 0.65:
        return "multiframe_possible_occupied_unconfirmed", reasons
    if conflict_anchors > 0:
        return "multiframe_boundary_or_adjacent_conflict", reasons
    return "multiframe_no_stable_vehicle_evidence", reasons


def build_consensus(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_slot[str(row["slot_id"])].append(row)
    consensus_rows: list[dict[str, Any]] = []
    for slot_id, slot_rows in sorted(by_slot.items()):
        observed_anchors = len(slot_rows)
        core_rows = [row for row in slot_rows if row.get("accumulated_state") == "accumulated_vehicle_core_supported"]
        conflict_rows = [row for row in slot_rows if row.get("accumulated_state") in {"accumulated_adjacent_conflict", "accumulated_boundary_conflict"}]
        static_rows = [row for row in slot_rows if row.get("accumulated_state") == "accumulated_static_like"]
        owner_counts = Counter(str(row.get("cluster_owner_slot", "")) for row in core_rows if row.get("cluster_owner_slot"))
        dominant_owner, dominant_owner_count = ("", 0)
        if owner_counts:
            dominant_owner, dominant_owner_count = owner_counts.most_common(1)[0]
        owner_stability = dominant_owner_count / max(len(core_rows), 1)
        support_ratio = len(core_rows) / max(observed_anchors, 1)
        conflict_ratio = len(conflict_rows) / max(observed_anchors, 1)
        max_score = max((f(row, "max_vehicle_like_score") for row in slot_rows), default=0.0)
        mean_core_score = sum(f(row, "max_vehicle_like_score") for row in core_rows) / max(len(core_rows), 1)
        max_core_overlap = max((i(row, "core_overlap_count") for row in slot_rows), default=0)
        mean_boundary_ratio = sum(f(row, "boundary_ratio") for row in slot_rows) / max(observed_anchors, 1)
        max_boundary_ratio = max((f(row, "boundary_ratio") for row in slot_rows), default=0.0)
        mean_adjacent_ratio = sum(f(row, "adjacent_overlap_ratio") for row in slot_rows) / max(observed_anchors, 1)
        max_adjacent_ratio = max((f(row, "adjacent_overlap_ratio") for row in slot_rows), default=0.0)
        anchor_frames = sorted({i(row, "anchor_frame") for row in slot_rows})
        core_anchor_frames = sorted({i(row, "anchor_frame") for row in core_rows})
        source_windows = sorted({str(row.get("source_window", "")) for row in slot_rows})
        state, reasons = consensus_state(len(core_rows), observed_anchors, len(conflict_rows), owner_stability, max_score, args)
        consensus_rows.append(
            {
                "slot_id": slot_id,
                "consensus_state": state,
                "observed_anchor_count": observed_anchors,
                "core_supported_anchor_count": len(core_rows),
                "conflict_anchor_count": len(conflict_rows),
                "static_like_anchor_count": len(static_rows),
                "core_support_ratio": support_ratio,
                "conflict_ratio": conflict_ratio,
                "dominant_owner_slot": dominant_owner,
                "dominant_owner_count": dominant_owner_count,
                "owner_stability": owner_stability,
                "max_vehicle_like_score": max_score,
                "mean_core_vehicle_like_score": mean_core_score,
                "max_core_overlap_count": max_core_overlap,
                "mean_boundary_ratio": mean_boundary_ratio,
                "max_boundary_ratio": max_boundary_ratio,
                "mean_adjacent_overlap_ratio": mean_adjacent_ratio,
                "max_adjacent_overlap_ratio": max_adjacent_ratio,
                "source_windows": source_windows,
                "anchor_frames": anchor_frames,
                "core_anchor_frames": core_anchor_frames,
                "reason": reasons,
            }
        )
    consensus_rows.sort(
        key=lambda row: (
            row["consensus_state"] != "possible_occupied_by_multiframe_accumulation",
            row["consensus_state"] != "multiframe_possible_occupied_unconfirmed",
            -float(row["core_supported_anchor_count"]),
            -float(row["max_vehicle_like_score"]),
            float(row["conflict_ratio"]),
        )
    )
    return consensus_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "slot_id",
        "consensus_state",
        "observed_anchor_count",
        "core_supported_anchor_count",
        "conflict_anchor_count",
        "static_like_anchor_count",
        "core_support_ratio",
        "conflict_ratio",
        "dominant_owner_slot",
        "dominant_owner_count",
        "owner_stability",
        "max_vehicle_like_score",
        "mean_core_vehicle_like_score",
        "max_core_overlap_count",
        "mean_boundary_ratio",
        "max_boundary_ratio",
        "mean_adjacent_overlap_ratio",
        "max_adjacent_overlap_ratio",
        "source_windows",
        "anchor_frames",
        "core_anchor_frames",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ["source_windows", "anchor_frames", "core_anchor_frames", "reason"]:
                out[key] = json.dumps(out.get(key, []), ensure_ascii=False)
            writer.writerow(out)


def esc(value: Any) -> str:
    return html.escape(str(value))


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    state_groups = defaultdict(list)
    for row in rows:
        state_groups[row["consensus_state"]].append(row)

    def table(group_rows: list[dict[str, Any]], limit: int) -> str:
        parts = [
            "<table><tr><th>slot</th><th>state</th><th>core/obs</th><th>conflict</th><th>owner</th><th>score</th><th>core overlap</th><th>boundary</th><th>adjacent</th><th>core frames</th><th>reason</th></tr>"
        ]
        for row in group_rows[:limit]:
            parts.append(
                "<tr>"
                f"<td>{esc(row['slot_id'])}</td>"
                f"<td>{esc(row['consensus_state'])}</td>"
                f"<td>{row['core_supported_anchor_count']}/{row['observed_anchor_count']} ({float(row['core_support_ratio']):.2f})</td>"
                f"<td>{row['conflict_anchor_count']} ({float(row['conflict_ratio']):.2f})</td>"
                f"<td>{esc(row['dominant_owner_slot'])} ({float(row['owner_stability']):.2f})</td>"
                f"<td>{float(row['max_vehicle_like_score']):.3f}</td>"
                f"<td>{row['max_core_overlap_count']}</td>"
                f"<td>{float(row['mean_boundary_ratio']):.2f}/{float(row['max_boundary_ratio']):.2f}</td>"
                f"<td>{float(row['mean_adjacent_overlap_ratio']):.2f}/{float(row['max_adjacent_overlap_ratio']):.2f}</td>"
                f"<td>{esc(row['core_anchor_frames'])}</td>"
                f"<td>{esc(' | '.join(row['reason']))}</td>"
                "</tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Multiframe Temporal Consensus</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    .note {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; font-size: 12px; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 4px 6px; text-align: left; vertical-align: top; }}
    pre {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 10px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Multiframe Temporal Consensus</h1>
  <div class="note">
    Diagnostic only. This converts repeated accumulated evidence into per-slot consensus labels, but does not change Part 1 final free/occupied states.
  </div>
  <h2>Summary</h2>
  <pre>{esc(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
  <h2>Possible occupied by multiframe accumulation</h2>
  {table(state_groups['possible_occupied_by_multiframe_accumulation'], args.review_limit)}
  <h2>Possible occupied unconfirmed</h2>
  {table(state_groups['multiframe_possible_occupied_unconfirmed'], args.review_limit)}
  <h2>Boundary / adjacent conflict</h2>
  {table(state_groups['multiframe_boundary_or_adjacent_conflict'], args.review_limit)}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    rows, sources = load_inputs(args.input_dirs)
    consensus_rows = build_consensus(rows, args)
    counts = Counter(row["consensus_state"] for row in consensus_rows)
    summary = {
        "input_dirs": [str(path) for path in args.input_dirs],
        "thresholds": {
            "min_core_supported_anchors": args.min_core_supported_anchors,
            "min_core_support_ratio": args.min_core_support_ratio,
            "min_owner_stability": args.min_owner_stability,
            "max_conflict_ratio": args.max_conflict_ratio,
            "min_score": args.min_score,
        },
        "source_count": len(sources),
        "input_row_count": len(rows),
        "slot_count": len(consensus_rows),
        "consensus_state_counts": dict(counts),
        "sources": sources,
        "outputs": {
            "consensus_csv": str(args.output_dir / "multiframe_temporal_consensus.csv"),
            "consensus_json": str(args.output_dir / "multiframe_temporal_consensus.json"),
            "report_html": str(args.output_dir / "multiframe_temporal_consensus_report.html"),
        },
    }
    write_csv(args.output_dir / "multiframe_temporal_consensus.csv", consensus_rows)
    write_json(args.output_dir / "multiframe_temporal_consensus.json", {"summary": summary, "slots": consensus_rows})
    write_json(args.output_dir / "multiframe_temporal_consensus_summary.json", summary)
    write_report(args.output_dir / "multiframe_temporal_consensus_report.html", summary, consensus_rows, args)
    print(f"[done] output={args.output_dir}")
    print(json.dumps({"slot_count": len(consensus_rows), "consensus_state_counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
