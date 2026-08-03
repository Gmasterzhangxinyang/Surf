#!/usr/bin/env python3
"""Build a unified per-slot decision table from Part 1 diagnostics.

This is a conservative post-processing layer. It does not mutate Part 1
outputs; it merges Part 1 state, vehicle-cluster diagnostics, pose-assisted
diagnostics, and occupied-chain diagnostics into one readable decision table.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    HAS_MPL = True
except Exception:
    HAS_MPL = False


DEFAULT_PART1 = Path("outputs/part1_slot_scoring_10000_10994_vehicle_cluster")
DEFAULT_PART2 = Path("outputs/part2_case_review_pack_10000_10994")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified slot decision table")
    parser.add_argument("--part1-dir", type=Path, default=DEFAULT_PART1)
    parser.add_argument("--part2-pack", type=Path, default=DEFAULT_PART2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vehicle-score-occupied", type=float, default=0.75)
    parser.add_argument("--vehicle-support-frames", type=int, default=3)
    parser.add_argument("--owner-ratio-min", type=float, default=0.75)
    parser.add_argument("--conflict-score-max", type=float, default=0.65)
    parser.add_argument("--boundary-frame-ratio-max", type=float, default=0.45)
    parser.add_argument("--no-map", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_slot_database(part1_dir: Path) -> dict[str, dict[str, object]]:
    path = part1_dir / "slot_database.json"
    if not path.exists():
        return {}
    data = load_json(path)
    return {str(slot["slot_id"]): slot for slot in data["slots"]}  # type: ignore[index]


def load_pose_cases(part2_pack: Path) -> dict[str, dict[str, object]]:
    path = part2_pack / "pose_assisted_review" / "pose_assisted_cases.json"
    if not path.exists():
        return {}
    cases = load_json(path)["cases"]  # type: ignore[index]
    by_slot: dict[str, dict[str, object]] = {}
    for case in cases:
        by_slot[str(case["slot_id"])] = case
    return by_slot


def load_chain_results(part2_pack: Path) -> dict[str, dict[str, object]]:
    path = part2_pack / "occupied_chain_analysis" / "occupied_chain_analysis.json"
    if not path.exists():
        return {}
    rows = load_json(path).get("slot_results", [])  # type: ignore[union-attr]
    return {str(row["slot_id"]): row for row in rows}


def load_block_results(part2_pack: Path) -> dict[str, dict[str, object]]:
    path = part2_pack / "occupied_chain_analysis" / "occupied_blocks.json"
    if path.exists():
        rows = load_json(path).get("slot_results", [])  # type: ignore[union-attr]
        return {str(row["slot_id"]): row for row in rows if str(row.get("block_status", "")) == "occupied_block_confirmed"}
    legacy_path = part2_pack / "occupied_chain_analysis" / "occupied_chain_analysis.json"
    if not legacy_path.exists():
        return {}
    rows = load_json(legacy_path).get("block_slot_results", [])  # type: ignore[union-attr]
    return {str(row["slot_id"]): row for row in rows if str(row.get("block_status", "")) == "occupied_block_confirmed"}


def load_vehicle_audit(part1_dir: Path) -> dict[str, dict[str, object]]:
    path = part1_dir / "vehicle_cluster_summary.json"
    if not path.exists():
        return {}
    rows = load_json(path).get("slots", [])  # type: ignore[union-attr]
    return {str(row["slot_id"]): row for row in rows}


def parse_risk_counts(row: dict[str, object]) -> dict[str, int]:
    raw = row.get("vehicle_cluster_risk_counts", {})
    if isinstance(raw, dict):
        return {str(k): to_int(v) for k, v in raw.items()}
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): to_int(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            return {}
    return {}


def static_like_dominates(risk_counts: dict[str, int], vehicle_support: int) -> bool:
    static_count = sum(risk_counts.get(key, 0) for key in ["wall_like_linear", "low_height_structure", "too_small_static", "too_large_merged"])
    return static_count > 0 and static_count >= max(vehicle_support, 1)


def vehicle_promotion_blockers(row: dict[str, str], args: argparse.Namespace) -> list[str]:
    support = to_int(row.get("support_frame_count"))
    vehicle_support = to_int(row.get("vehicle_like_support_frames"))
    stable_frames = to_int(row.get("stable_vehicle_cluster_frames"))
    boundary_ratio = to_int(row.get("boundary_conflict_frames")) / max(support, 1)
    blockers: list[str] = []
    if str(row.get("diagnostic_label", "")) != "likely_occupied_vehicle_cluster":
        blockers.append("diagnostic_label_not_likely_vehicle_cluster")
    if to_float(row.get("max_vehicle_like_score")) < args.vehicle_score_occupied:
        blockers.append("vehicle_like_score_below_threshold")
    if vehicle_support < args.vehicle_support_frames:
        blockers.append("vehicle_support_frames_below_threshold")
    if stable_frames < args.vehicle_support_frames:
        blockers.append("stable_vehicle_frames_below_threshold")
    if str(row.get("dominant_vehicle_owner_slot", "")) != str(row.get("slot_id", "")):
        blockers.append("dominant_vehicle_owner_is_not_target_slot")
    if to_float(row.get("dominant_vehicle_owner_ratio")) < args.owner_ratio_min:
        blockers.append("dominant_vehicle_owner_ratio_below_threshold")
    if to_float(row.get("conflict_score")) > args.conflict_score_max:
        blockers.append("conflict_score_above_threshold")
    if boundary_ratio > args.boundary_frame_ratio_max:
        blockers.append("boundary_frame_ratio_above_threshold")
    return blockers


def decide_slot(
    row: dict[str, str],
    pose_case: dict[str, object] | None,
    chain: dict[str, object] | None,
    block: dict[str, object] | None,
    vehicle_audit: dict[str, object] | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    slot_id = str(row["slot_id"])
    part1_state = str(row.get("state", "unknown"))
    diagnostic = str(row.get("diagnostic_label", ""))
    support = to_int(row.get("support_frame_count"))
    vehicle_support = to_int(row.get("vehicle_like_support_frames"))
    risk_counts = parse_risk_counts(vehicle_audit or {})
    boundary_frame_ratio = to_int(row.get("boundary_conflict_frames")) / max(support, 1)
    reason_codes: list[str] = []
    blocking_reasons: list[str] = []
    promoted_by = ""
    evidence_level = str(row.get("evidence_level", ""))
    unified_label = diagnostic or str(row.get("uncertainty_type", "")) or "low_visibility"

    vehicle_blockers = vehicle_promotion_blockers(row, args)
    static_dominates = static_like_dominates(risk_counts, vehicle_support)
    chain_status = str((chain or {}).get("chain_status", ""))
    chain_position = str((chain or {}).get("chain_position", ""))
    chain_confidence = str((chain or {}).get("chain_confidence", ""))
    block_status = str((block or {}).get("block_status", ""))
    block_confidence = str((block or {}).get("block_confidence", ""))
    pose_status = str((pose_case or {}).get("pose_assisted_status", ""))

    if part1_state == "free":
        final = "free"
        unified_label = "part1_confirmed_free"
        evidence_level = evidence_level or "high"
        promoted_by = "part1_final_state"
        reason_codes.append("part1_final_free")
    elif part1_state == "occupied":
        final = "occupied"
        unified_label = "part1_confirmed_occupied"
        evidence_level = evidence_level or "high"
        promoted_by = "part1_final_state"
        reason_codes.append("part1_final_occupied")
    elif block_status == "occupied_block_confirmed":
        final = "occupied"
        unified_label = "occupied_block_confirmed"
        evidence_level = block_confidence or str((block or {}).get("evidence_level", "")) or "medium"
        promoted_by = "occupied_block"
        reason_codes.extend(["continuous_adjacent_occupied_evidence", "slot_assignment_ambiguity_absorbed_by_block"])
    elif chain_status == "chain_edge_uncertain" or diagnostic == "vehicle_cluster_boundary_conflict":
        final = "edge_uncertain"
        unified_label = "edge_uncertain"
        evidence_level = evidence_level or str((chain or {}).get("evidence_level", "")) or "medium"
        promoted_by = "boundary_guard"
        reason_codes.append("edge_or_boundary_conflict")
        if chain_status == "chain_edge_uncertain":
            reason_codes.append("chain_edge")
        if diagnostic == "vehicle_cluster_boundary_conflict":
            reason_codes.append("vehicle_cluster_boundary_conflict")
    elif not vehicle_blockers and not static_dominates:
        final = "occupied"
        unified_label = "confirmed_occupied_by_vehicle_cluster"
        evidence_level = "high"
        promoted_by = "vehicle_cluster"
        reason_codes.extend(["vehicle_like_cluster_high_score", "vehicle_owner_stable", "boundary_ratio_within_limit"])
    elif diagnostic == "likely_occupied_vehicle_cluster" and not static_dominates:
        final = "likely_occupied"
        unified_label = "likely_occupied_vehicle_cluster"
        evidence_level = evidence_level or "medium"
        promoted_by = "vehicle_cluster_diagnostic"
        reason_codes.append("vehicle_like_cluster_present")
        blocking_reasons.extend(vehicle_blockers)
    elif chain_status == "likely_occupied_chain_supported" and not static_dominates:
        final = "likely_occupied"
        unified_label = "likely_occupied_chain_supported"
        evidence_level = chain_confidence or evidence_level or "medium"
        promoted_by = "occupied_chain"
        reason_codes.extend(["chain_interior", "occupied_chain_supported"])
    elif diagnostic == "static_like_cluster" or static_dominates:
        final = "ambiguous"
        unified_label = "static_or_structure_likely"
        evidence_level = "low"
        promoted_by = "static_guard"
        reason_codes.append("static_like_cluster")
        if static_dominates:
            reason_codes.append("static_risk_dominates_vehicle_support")
    elif diagnostic == "possible_occupied_vehicle_cluster" or pose_status == "possible_occupied_pose_supported_camera_check":
        final = "ambiguous"
        unified_label = "possible_occupied_needs_review"
        evidence_level = evidence_level or "medium"
        promoted_by = "review_needed"
        reason_codes.append("occupied_evidence_not_strict_enough")
        blocking_reasons.extend(vehicle_blockers)
    elif diagnostic == "possible_free_unconfirmed" or str((pose_case or {}).get("uncertainty_type", "")) == "possible_free_unconfirmed":
        final = "possible_free_unconfirmed"
        unified_label = "possible_free_unconfirmed"
        evidence_level = evidence_level or "medium"
        promoted_by = "free_guard"
        reason_codes.append("free_not_promoted_in_v1")
    elif part1_state == "unknown" or to_float(row.get("visibility_score")) < 0.18:
        final = "unknown"
        unified_label = "unknown_or_low_visibility"
        evidence_level = evidence_level or "low"
        promoted_by = "visibility_guard"
        reason_codes.append("low_visibility_or_unobserved")
    else:
        final = "ambiguous"
        unified_label = unified_label or "ambiguous"
        evidence_level = evidence_level or "low"
        promoted_by = "default_conservative"
        reason_codes.append("no_strict_rule_matched")

    if static_dominates and final in {"occupied", "likely_occupied"}:
        blocking_reasons.append("static_risk_dominates_vehicle_support")
    blocking_reasons = list(dict.fromkeys(blocking_reasons))
    reason_codes = list(dict.fromkeys(reason_codes))

    return {
        "slot_id": slot_id,
        "part1_state": part1_state,
        "final_decision": final,
        "diagnostic_label": unified_label,
        "raw_diagnostic_label": diagnostic,
        "evidence_level": evidence_level,
        "promoted_by": promoted_by,
        "reason_codes": reason_codes,
        "blocking_reasons": blocking_reasons,
        "support_frame_count": support,
        "visibility_score": to_float(row.get("visibility_score")),
        "max_ray_free_ratio": to_float(row.get("max_ray_free_ratio")),
        "max_core_hit_cell_ratio": to_float(row.get("max_core_hit_cell_ratio")),
        "max_vehicle_like_score": to_float(row.get("max_vehicle_like_score")),
        "vehicle_like_support_frames": vehicle_support,
        "stable_vehicle_cluster_frames": to_int(row.get("stable_vehicle_cluster_frames")),
        "dominant_vehicle_owner_slot": str(row.get("dominant_vehicle_owner_slot", "")),
        "dominant_vehicle_owner_ratio": to_float(row.get("dominant_vehicle_owner_ratio")),
        "boundary_conflict_frames": to_int(row.get("boundary_conflict_frames")),
        "boundary_frame_ratio": boundary_frame_ratio,
        "conflict_score": to_float(row.get("conflict_score")),
        "chain_id": str((chain or {}).get("chain_id", "")),
        "chain_position": chain_position,
        "chain_status": chain_status,
        "chain_confidence": chain_confidence,
        "block_id": str((block or {}).get("block_id", "")),
        "block_position": str((block or {}).get("block_position", "")),
        "block_status": block_status,
        "block_confidence": block_confidence,
        "pose_assisted_status": pose_status,
        "vehicle_cluster_risk_counts": risk_counts,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "slot_id",
        "part1_state",
        "final_decision",
        "diagnostic_label",
        "raw_diagnostic_label",
        "evidence_level",
        "promoted_by",
        "reason_codes",
        "blocking_reasons",
        "support_frame_count",
        "visibility_score",
        "max_ray_free_ratio",
        "max_core_hit_cell_ratio",
        "max_vehicle_like_score",
        "vehicle_like_support_frames",
        "stable_vehicle_cluster_frames",
        "dominant_vehicle_owner_slot",
        "dominant_vehicle_owner_ratio",
        "boundary_conflict_frames",
        "boundary_frame_ratio",
        "conflict_score",
        "chain_id",
        "chain_position",
        "chain_status",
        "chain_confidence",
        "block_id",
        "block_position",
        "block_status",
        "block_confidence",
        "pose_assisted_status",
        "vehicle_cluster_risk_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["reason_codes"] = "|".join(row.get("reason_codes", []))
            out["blocking_reasons"] = "|".join(row.get("blocking_reasons", []))
            out["vehicle_cluster_risk_counts"] = json.dumps(row.get("vehicle_cluster_risk_counts", {}), ensure_ascii=False)
            writer.writerow(out)


def row_color(decision: str) -> str:
    return {
        "occupied": "#fee2e2",
        "likely_occupied": "#ffedd5",
        "edge_uncertain": "#fef9c3",
        "ambiguous": "#f3e8ff",
        "possible_free_unconfirmed": "#dbeafe",
        "unknown": "#f1f5f9",
        "free": "#dcfce7",
    }.get(decision, "#ffffff")


def draw_map(path: Path, slots: dict[str, dict[str, object]], rows: list[dict[str, object]]) -> str:
    if not HAS_MPL or not slots:
        return ""
    row_by_slot = {str(row["slot_id"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(11, 8), dpi=170)
    color_map = {
        "occupied": ("#ef4444", "#991b1b", 0.72, 1.0),
        "likely_occupied": ("#f97316", "#9a3412", 0.62, 0.9),
        "edge_uncertain": ("#facc15", "#a16207", 0.58, 0.8),
        "ambiguous": ("#a855f7", "#6b21a8", 0.36, 0.55),
        "possible_free_unconfirmed": ("#60a5fa", "#1d4ed8", 0.36, 0.55),
        "unknown": ("#e5e7eb", "#cbd5e1", 0.16, 0.25),
        "free": ("#22c55e", "#166534", 0.62, 1.0),
    }
    highlighted = []
    for slot_id, slot in slots.items():
        row = row_by_slot.get(slot_id, {})
        decision = str(row.get("final_decision", "unknown"))
        face, edge, alpha, lw = color_map.get(decision, color_map["unknown"])
        if row.get("diagnostic_label") == "occupied_block_confirmed":
            face, edge, alpha, lw = "#b91c1c", "#450a0a", 0.78, 1.8
        if decision not in {"unknown"}:
            highlighted.append(slot_id)
        poly = np.asarray(slot["polygon_map"], dtype=np.float64)
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw))
    for slot_id in highlighted:
        row = row_by_slot[slot_id]
        if row["final_decision"] in {"occupied", "likely_occupied", "edge_uncertain"}:
            c = np.asarray(slots[slot_id]["center_map"], dtype=np.float64)
            ax.text(c[0], c[1], slot_id.replace("slot_", ""), fontsize=5, ha="center", va="center", color="#111827")
    pts = np.vstack([np.asarray(slot["polygon_map"], dtype=np.float64) for slot in slots.values()])
    ax.set_xlim(float(pts[:, 0].min() - 0.5), float(pts[:, 0].max() + 0.5))
    ax.set_ylim(float(pts[:, 1].min() - 0.5), float(pts[:, 1].max() + 0.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Unified slot decision map")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path.name


def esc(value: object) -> str:
    return html.escape(str(value))


def write_report(path: Path, summary: dict[str, object], rows: list[dict[str, object]], map_name: str) -> None:
    groups = {
        "Block Confirmed Occupied": [r for r in rows if r["diagnostic_label"] == "occupied_block_confirmed"],
        "Confirmed Occupied": [r for r in rows if r["final_decision"] == "occupied"],
        "Likely Occupied": [r for r in rows if r["final_decision"] == "likely_occupied"],
        "Edge Uncertain": [r for r in rows if r["final_decision"] == "edge_uncertain"],
        "Static/Structure Likely": [r for r in rows if r["diagnostic_label"] == "static_or_structure_likely"],
        "Possible Free Unconfirmed": [r for r in rows if r["final_decision"] == "possible_free_unconfirmed"],
    }

    def table(group_rows: list[dict[str, object]], limit: int = 40) -> str:
        parts = ["<table><tr><th>slot</th><th>decision</th><th>label</th><th>level</th><th>score</th><th>veh frames</th><th>owner</th><th>conflict</th><th>chain</th><th>block</th><th>reasons</th><th>blocks</th></tr>"]
        for row in group_rows[:limit]:
            parts.append(
                f"<tr style='background:{row_color(str(row['final_decision']))}'>"
                f"<td>{esc(row['slot_id'])}</td>"
                f"<td>{esc(row['final_decision'])}</td>"
                f"<td>{esc(row['diagnostic_label'])}</td>"
                f"<td>{esc(row['evidence_level'])}</td>"
                f"<td>{float(row['max_vehicle_like_score']):.3f}</td>"
                f"<td>{row['vehicle_like_support_frames']}/{row['stable_vehicle_cluster_frames']}</td>"
                f"<td>{esc(row['dominant_vehicle_owner_slot'])} ({float(row['dominant_vehicle_owner_ratio']):.2f})</td>"
                f"<td>{float(row['conflict_score']):.2f}</td>"
                f"<td>{esc(row['chain_status'])}</td>"
                f"<td>{esc(row['block_id'])} {esc(row['block_position'])}</td>"
                f"<td>{esc(' | '.join(row.get('reason_codes', [])))}</td>"
                f"<td>{esc(' | '.join(row.get('blocking_reasons', [])))}</td>"
                "</tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    sections = []
    for title, group_rows in groups.items():
        sections.append(f"<h2>{esc(title)} ({len(group_rows)})</h2>{table(group_rows)}")

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Unified Slot Decision Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    pre {{ background: #f9fafb; border: 1px solid #d1d5db; padding: 12px; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; }}
  </style>
</head>
<body>
  <h1>Unified Slot Decision Report</h1>
  <p>This is a post-processing decision table. It does not mutate Part 1 outputs.</p>
  <h2>Summary</h2>
  <pre>{esc(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
  <p><strong>Map colors:</strong> dark red = occupied_block_confirmed, red = single-slot occupied, orange = likely occupied, yellow = edge uncertain, purple = ambiguous, blue = possible free unconfirmed, grey = unknown.</p>
  {f'<h2>Map</h2><img src="{esc(map_name)}" alt="decision map">' if map_name else ''}
  {''.join(sections)}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.part1_dir / "slot_decision_table")
    ensure_dir(output_dir)

    scores = read_csv(args.part1_dir / "slot_scores.csv")
    pose_by_slot = load_pose_cases(args.part2_pack)
    chain_by_slot = load_chain_results(args.part2_pack)
    block_by_slot = load_block_results(args.part2_pack)
    vehicle_audit_by_slot = load_vehicle_audit(args.part1_dir)
    slots = load_slot_database(args.part1_dir)

    rows = [
        decide_slot(
            row,
            pose_by_slot.get(str(row["slot_id"])),
            chain_by_slot.get(str(row["slot_id"])),
            block_by_slot.get(str(row["slot_id"])),
            vehicle_audit_by_slot.get(str(row["slot_id"])),
            args,
        )
        for row in scores
    ]
    priority = {
        "occupied": 0,
        "likely_occupied": 1,
        "edge_uncertain": 2,
        "ambiguous": 3,
        "possible_free_unconfirmed": 4,
        "unknown": 5,
        "free": 6,
    }
    rows.sort(key=lambda r: (priority.get(str(r["final_decision"]), 9), -float(r["max_vehicle_like_score"]), str(r["slot_id"])))
    decision_counts = Counter(str(row["final_decision"]) for row in rows)
    label_counts = Counter(str(row["diagnostic_label"]) for row in rows)
    promoted_by_counts = Counter(str(row["promoted_by"]) for row in rows)
    summary = {
        "slot_count": len(rows),
        "final_decision_counts": dict(decision_counts),
        "diagnostic_label_counts": dict(label_counts),
        "promoted_by_counts": dict(promoted_by_counts),
        "occupied_block_confirmed_count": label_counts.get("occupied_block_confirmed", 0),
        "thresholds": {
            "vehicle_score_occupied": args.vehicle_score_occupied,
            "vehicle_support_frames": args.vehicle_support_frames,
            "owner_ratio_min": args.owner_ratio_min,
            "conflict_score_max": args.conflict_score_max,
            "boundary_frame_ratio_max": args.boundary_frame_ratio_max,
        },
        "note": "Free promotion is disabled unless Part 1 already outputs free. occupied_block_confirmed absorbs adjacent-slot attribution ambiguity inside confirmed blocks.",
    }

    write_json(output_dir / "slot_decision_table.json", {"summary": summary, "slots": rows})
    write_csv(output_dir / "slot_decision_table.csv", rows)
    map_name = "" if args.no_map else draw_map(output_dir / "slot_decision_map.png", slots, rows)
    write_report(output_dir / "slot_decision_report.html", summary, rows, map_name)
    print(f"[done] output={output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
