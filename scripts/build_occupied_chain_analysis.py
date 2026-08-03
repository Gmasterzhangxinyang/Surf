#!/usr/bin/env python3
"""Build occupied-chain diagnostics from pose-assisted slot evidence.

The analysis is intentionally non-authoritative: it adds likely-occupied
diagnostics without changing Part 1 final free/occupied states.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
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


DEFAULT_PART1 = Path("outputs/part1_slot_scoring_10000_10994_boundary_fixed")
DEFAULT_PART2 = Path("outputs/part2_case_review_pack_10000_10994")

LEVEL_RANK = {"strong": 3, "medium": 2, "weak": 1}
LEVEL_LABEL = {3: "strong", 2: "medium", 1: "weak"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build occupied chain diagnostics")
    parser.add_argument("--part1-dir", type=Path, default=DEFAULT_PART1)
    parser.add_argument("--part2-pack", type=Path, default=DEFAULT_PART2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--heading-diff-deg", type=float, default=15.0)
    parser.add_argument("--min-adjacent-distance-m", type=float, default=0.3)
    parser.add_argument("--max-adjacent-distance-m", type=float, default=4.0)
    parser.add_argument("--min-chain-length", type=int, default=3)
    parser.add_argument("--min-block-length", type=int, default=3)
    parser.add_argument("--max-boundary-ratio", type=float, default=0.60)
    parser.add_argument("--static-min-level", choices=["weak", "medium", "strong"], default="strong")
    parser.add_argument("--slot-vs-lane-min-level", choices=["weak", "medium", "strong"], default="medium")
    return parser.parse_args()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def slot_map(slot_db: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(slot["slot_id"]): slot for slot in slot_db["slots"]}  # type: ignore[index]


def center(slot: dict[str, object]) -> np.ndarray:
    return np.asarray(slot["center_map"], dtype=np.float64)


def heading_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(diff, abs(180.0 - diff))


def distance_m(slot_a: dict[str, object], slot_b: dict[str, object], scale: float) -> float:
    return float(np.linalg.norm(center(slot_a) - center(slot_b))) / max(scale, 1e-9)


def numeric(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def merge_candidate(existing: dict[str, object] | None, incoming: dict[str, object]) -> dict[str, object]:
    if existing is None:
        return incoming
    old_rank = LEVEL_RANK[str(existing["evidence_level"])]
    new_rank = LEVEL_RANK[str(incoming["evidence_level"])]
    if new_rank > old_rank:
        return {**existing, **incoming}
    if new_rank == old_rank and numeric(incoming.get("max_core_hit_cell_ratio")) > numeric(existing.get("max_core_hit_cell_ratio")):
        return {**existing, **incoming}
    return existing


def candidate_flags(subtype: str, level: str, boundary: float, args: argparse.Namespace) -> tuple[bool, bool, list[str]]:
    risk_flags: list[str] = []
    if boundary > args.max_boundary_ratio:
        risk_flags.append("high_boundary_ratio")
    if subtype == "static_vs_vehicle" and LEVEL_RANK[level] < LEVEL_RANK[args.static_min_level]:
        risk_flags.append("static_evidence_not_strong")
    if subtype == "slot_vs_lane" and LEVEL_RANK[level] < LEVEL_RANK[args.slot_vs_lane_min_level]:
        risk_flags.append("slot_vs_lane_evidence_not_medium")
    chain_eligible = "high_boundary_ratio" not in risk_flags and "static_evidence_not_strong" not in risk_flags and "slot_vs_lane_evidence_not_medium" not in risk_flags
    block_eligible = "static_evidence_not_strong" not in risk_flags and "slot_vs_lane_evidence_not_medium" not in risk_flags
    return chain_eligible, block_eligible, risk_flags


def level_from_part1(row: dict[str, str]) -> str:
    level = str(row.get("evidence_level", "")).lower()
    if level in LEVEL_RANK:
        return level
    score = numeric(row.get("max_vehicle_like_score"))
    support = int(numeric(row.get("vehicle_like_support_frames")))
    if score >= 0.85 and support >= 5:
        return "strong"
    if score >= 0.70 and support >= 3:
        return "medium"
    return "weak"


def build_candidates(part1_dir: Path, part2_pack: Path, args: argparse.Namespace) -> dict[str, dict[str, object]]:
    pose_path = part2_pack / "pose_assisted_review" / "pose_assisted_cases.json"
    high_conf_path = part2_pack / "pose_assisted_review" / "high_confidence_occupied_candidates.json"
    pose_cases = load_json(pose_path)["cases"]  # type: ignore[index]
    high_conf = (
        load_json(high_conf_path)
        if high_conf_path.exists()
        else {"strict_high_confidence_occupied_candidates": [], "regional_occupied_needs_slot_attribution": []}
    )
    part1_scores = read_csv(part1_dir / "slot_scores.csv")

    candidates: dict[str, dict[str, object]] = {}

    def add_candidate(row: dict[str, object], level: str, source: str) -> None:
        slot_id = str(row["slot_id"])
        subtype = str(row.get("uncertainty_type", ""))
        boundary = numeric(row.get("boundary_conflict_frame_ratio"))
        chain_eligible, block_eligible, risk_flags = candidate_flags(subtype, level, boundary, args)
        incoming = {
            "slot_id": slot_id,
            "case_id": row.get("case_id", ""),
            "uncertainty_type": subtype,
            "evidence_level": level,
            "evidence_source": source,
            "chain_eligible": chain_eligible,
            "block_eligible": block_eligible,
            "risk_flags": risk_flags,
            "support_frame_count": int(numeric(row.get("support_frame_count"))),
            "clear_core_owned_frames": int(numeric(row.get("clear_core_owned_frames"))),
            "max_core_hit_cell_ratio": numeric(row.get("max_core_hit_cell_ratio")),
            "boundary_conflict_frame_ratio": boundary,
            "camera_visible_frame_count": int(numeric(row.get("camera_visible_frame_count"))),
            "best_camera_projection_score": numeric(row.get("best_camera_projection_score")),
            "dominant_core_owner_slot": row.get("dominant_core_owner_slot", ""),
            "dominant_core_owner_ratio": numeric(row.get("dominant_core_owner_ratio")),
            "min_distance_m": numeric(row.get("min_distance_m"), None),  # type: ignore[arg-type]
            "pose_baseline_m": numeric(row.get("pose_baseline_m")),
        }
        candidates[slot_id] = merge_candidate(candidates.get(slot_id), incoming)

    for row in high_conf["strict_high_confidence_occupied_candidates"]:  # type: ignore[index]
        add_candidate(row, "strong", "strict_high_confidence")
    for row in high_conf["regional_occupied_needs_slot_attribution"]:  # type: ignore[index]
        add_candidate(row, "medium", "regional_needs_slot_attribution")
    for row in pose_cases:
        if str(row.get("pose_assisted_status")) == "possible_occupied_pose_supported_camera_check":
            add_candidate(row, "weak", "pose_supported_possible_occupied")
    for row in part1_scores:
        diagnostic = str(row.get("diagnostic_label", ""))
        if diagnostic not in {"likely_occupied_vehicle_cluster", "vehicle_cluster_boundary_conflict"}:
            continue
        boundary = numeric(row.get("max_boundary_ratio"))
        level = level_from_part1(row)
        add_candidate(
            {
                "slot_id": row["slot_id"],
                "case_id": "",
                "uncertainty_type": row.get("uncertainty_type") or ("adjacent_slot_conflict" if diagnostic == "vehicle_cluster_boundary_conflict" else ""),
                "support_frame_count": row.get("support_frame_count"),
                "clear_core_owned_frames": row.get("clear_cluster_ownership_frames"),
                "max_core_hit_cell_ratio": row.get("max_core_hit_cell_ratio"),
                "boundary_conflict_frame_ratio": boundary,
                "camera_visible_frame_count": 0,
                "best_camera_projection_score": 0.0,
                "dominant_core_owner_slot": row.get("dominant_vehicle_owner_slot", ""),
                "dominant_core_owner_ratio": row.get("dominant_vehicle_owner_ratio", 0.0),
                "min_distance_m": 0.0,
                "pose_baseline_m": 0.0,
            },
            level,
            diagnostic,
        )

    return candidates


def connect_slots(
    slot_a: str,
    slot_b: str,
    slots: dict[str, dict[str, object]],
    candidates: dict[str, dict[str, object]],
    scale: float,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, object]]:
    if slot_b not in slots[slot_a].get("adjacent_slots", []) and slot_a not in slots[slot_b].get("adjacent_slots", []):
        return False, {"reason": "not_adjacent"}
    hdiff = heading_diff_deg(float(slots[slot_a]["heading_deg"]), float(slots[slot_b]["heading_deg"]))
    dist = distance_m(slots[slot_a], slots[slot_b], scale)
    if hdiff > args.heading_diff_deg:
        return False, {"reason": "heading_diff", "heading_diff_deg": hdiff, "distance_m": dist}
    if dist < args.min_adjacent_distance_m or dist > args.max_adjacent_distance_m:
        return False, {"reason": "distance", "heading_diff_deg": hdiff, "distance_m": dist}
    if slot_a not in candidates or slot_b not in candidates:
        return False, {"reason": "missing_candidate", "heading_diff_deg": hdiff, "distance_m": dist}
    return True, {"heading_diff_deg": hdiff, "distance_m": dist}


def connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in sorted(graph):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in sorted(graph[cur]):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    return components


def order_component(component: list[str], slots: dict[str, dict[str, object]]) -> list[str]:
    if len(component) <= 2:
        return sorted(component)
    pts = np.vstack([center(slots[slot_id]) for slot_id in component])
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    return [slot_id for _, slot_id in sorted(zip(projections, component), key=lambda item: (float(item[0]), item[1]))]


def slot_result(
    chain_id: str,
    slot_id: str,
    idx: int,
    ordered: list[str],
    candidates: dict[str, dict[str, object]],
    min_chain_length: int,
) -> dict[str, object]:
    cand = candidates[slot_id]
    chain_len = len(ordered)
    is_edge = idx == 0 or idx == chain_len - 1
    level = str(cand["evidence_level"])
    status = "chain_too_short"
    confidence = "low"
    if chain_len >= min_chain_length:
        if is_edge:
            status = "chain_edge_uncertain"
            confidence = "medium" if level == "strong" else "low"
        else:
            status = "likely_occupied_chain_supported"
            confidence = "high" if level == "strong" and numeric(cand.get("boundary_conflict_frame_ratio")) <= 0.35 else "medium"
    return {
        "chain_id": chain_id,
        "slot_id": slot_id,
        "case_id": cand.get("case_id", ""),
        "chain_position": "edge" if is_edge else "interior",
        "chain_index": idx,
        "chain_length": chain_len,
        "chain_status": status,
        "chain_confidence": confidence,
        "evidence_level": level,
        "evidence_source": cand.get("evidence_source", ""),
        "uncertainty_type": cand.get("uncertainty_type", ""),
        "support_frame_count": cand.get("support_frame_count", 0),
        "clear_core_owned_frames": cand.get("clear_core_owned_frames", 0),
        "max_core_hit_cell_ratio": cand.get("max_core_hit_cell_ratio", 0.0),
        "boundary_conflict_frame_ratio": cand.get("boundary_conflict_frame_ratio", 0.0),
        "camera_visible_frame_count": cand.get("camera_visible_frame_count", 0),
        "best_camera_projection_score": cand.get("best_camera_projection_score", 0.0),
        "reason": reason_for_slot(status, level, chain_len),
    }


def reason_for_slot(status: str, level: str, chain_len: int) -> list[str]:
    if status == "likely_occupied_chain_supported":
        return [
            f"slot is interior of a geometrically continuous occupied-like chain of length {chain_len}",
            f"slot has {level} pointcloud/pose occupied evidence",
            "chain reasoning reduces single-slot boundary attribution ambiguity",
        ]
    if status == "chain_edge_uncertain":
        return [
            f"slot is at the edge of a continuous occupied-like chain of length {chain_len}",
            "edge slots remain vulnerable to adjacent-slot, lane, or static-structure attribution errors",
        ]
    return [
        f"occupied-like component length {chain_len} is below promotion threshold",
        "kept as diagnostic evidence only",
    ]


def build_chains(
    slots: dict[str, dict[str, object]],
    candidates: dict[str, dict[str, object]],
    scale: float,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    chain_candidates = {slot_id: cand for slot_id, cand in candidates.items() if cand.get("chain_eligible", True)}
    graph: dict[str, set[str]] = {slot_id: set() for slot_id in chain_candidates}
    edge_rows: list[dict[str, object]] = []
    for slot_id in sorted(chain_candidates):
        for adj in slots[slot_id].get("adjacent_slots", []):
            if adj not in chain_candidates or adj not in slots or adj <= slot_id:
                continue
            ok, metrics = connect_slots(slot_id, str(adj), slots, chain_candidates, scale, args)
            if not ok:
                continue
            graph[slot_id].add(str(adj))
            graph[str(adj)].add(slot_id)
            edge_rows.append({"slot_a": slot_id, "slot_b": str(adj), **metrics})

    components = connected_components(graph)
    chains: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    for idx, component in enumerate(components, 1):
        ordered = order_component(component, slots)
        chain_id = f"occ_chain_{idx:03d}"
        results = [slot_result(chain_id, slot_id, pos, ordered, chain_candidates, args.min_chain_length) for pos, slot_id in enumerate(ordered)]
        level_counts = Counter(str(row["evidence_level"]) for row in results)
        status_counts = Counter(str(row["chain_status"]) for row in results)
        hvals = [float(slots[slot_id]["heading_deg"]) for slot_id in ordered]
        chain = {
            "chain_id": chain_id,
            "slots": ordered,
            "chain_length": len(ordered),
            "status_counts": dict(status_counts),
            "evidence_level_counts": dict(level_counts),
            "interior_likely_occupied_slots": [str(r["slot_id"]) for r in results if r["chain_status"] == "likely_occupied_chain_supported"],
            "edge_uncertain_slots": [str(r["slot_id"]) for r in results if r["chain_status"] == "chain_edge_uncertain"],
            "mean_heading_deg": float(np.mean(hvals)) if hvals else 0.0,
            "slot_results": results,
        }
        chains.append(chain)
        slot_rows.extend(results)

    chains.sort(key=lambda c: (-int(c["chain_length"]), str(c["chain_id"])))
    for new_idx, chain in enumerate(chains, 1):
        new_id = f"occ_chain_{new_idx:03d}"
        chain["chain_id"] = new_id
        for row in chain["slot_results"]:  # type: ignore[index]
            row["chain_id"] = new_id
    slot_rows = [row for chain in chains for row in chain["slot_results"]]  # type: ignore[index]
    slot_rows.sort(key=lambda r: (str(r["chain_id"]), int(r["chain_index"])))
    return chains, slot_rows


def block_reason(block_status: str, block_len: int, risk_flags: list[str]) -> list[str]:
    if block_status == "occupied_block_confirmed":
        return [
            f"continuous occupied-like adjacent-slot block length {block_len}",
            "single-slot attribution ambiguity is absorbed because every slot in the block has occupied-like evidence",
            "block passed static and slot-vs-lane rejection gates",
        ]
    return [
        f"occupied-like component length {block_len} is below block threshold or failed block gates",
        "not promoted to occupied block",
        *risk_flags,
    ]


def build_occupied_blocks(
    slots: dict[str, dict[str, object]],
    candidates: dict[str, dict[str, object]],
    scale: float,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    block_candidates = {slot_id: cand for slot_id, cand in candidates.items() if cand.get("block_eligible", True)}
    graph: dict[str, set[str]] = {slot_id: set() for slot_id in block_candidates}
    for slot_id in sorted(block_candidates):
        for adj in slots[slot_id].get("adjacent_slots", []):
            if adj not in block_candidates or adj not in slots or adj <= slot_id:
                continue
            ok, _ = connect_slots(slot_id, str(adj), slots, block_candidates, scale, args)
            if not ok:
                continue
            graph[slot_id].add(str(adj))
            graph[str(adj)].add(slot_id)

    blocks: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    for idx, component in enumerate(connected_components(graph), 1):
        ordered = order_component(component, slots)
        block_len = len(ordered)
        risk_flags = sorted({flag for slot_id in ordered for flag in block_candidates[slot_id].get("risk_flags", [])})
        status = "occupied_block_confirmed" if block_len >= args.min_block_length else "block_rejected"
        level_rank = min((LEVEL_RANK[str(block_candidates[slot_id]["evidence_level"])] for slot_id in ordered), default=1)
        evidence_level = LEVEL_LABEL[level_rank]
        block_id = f"occ_block_{idx:03d}"
        rows = []
        for pos, slot_id in enumerate(ordered):
            cand = block_candidates[slot_id]
            row = {
                "block_id": block_id,
                "slot_id": slot_id,
                "case_id": cand.get("case_id", ""),
                "block_index": pos,
                "block_length": block_len,
                "block_position": "edge" if pos == 0 or pos == block_len - 1 else "interior",
                "block_status": status,
                "block_confidence": "high" if status == "occupied_block_confirmed" and evidence_level == "strong" else ("medium" if status == "occupied_block_confirmed" else "low"),
                "evidence_level": cand.get("evidence_level", ""),
                "evidence_source": cand.get("evidence_source", ""),
                "uncertainty_type": cand.get("uncertainty_type", ""),
                "support_frame_count": cand.get("support_frame_count", 0),
                "clear_core_owned_frames": cand.get("clear_core_owned_frames", 0),
                "max_core_hit_cell_ratio": cand.get("max_core_hit_cell_ratio", 0.0),
                "boundary_conflict_frame_ratio": cand.get("boundary_conflict_frame_ratio", 0.0),
                "risk_flags": cand.get("risk_flags", []),
                "reason": block_reason(status, block_len, risk_flags),
            }
            rows.append(row)
            slot_rows.append(row)
        blocks.append(
            {
                "block_id": block_id,
                "slots": ordered,
                "block_length": block_len,
                "block_status": status,
                "block_confidence": "high" if status == "occupied_block_confirmed" and evidence_level == "strong" else ("medium" if status == "occupied_block_confirmed" else "low"),
                "evidence_level": evidence_level,
                "evidence_source_counts": dict(Counter(str(block_candidates[slot_id].get("evidence_source", "")) for slot_id in ordered)),
                "risk_flags": risk_flags,
                "confirmed_slots": ordered if status == "occupied_block_confirmed" else [],
                "reason": block_reason(status, block_len, risk_flags),
                "slot_results": rows,
            }
        )

    blocks.sort(key=lambda block: (-int(block["block_length"]), str(block["block_id"])))
    for new_idx, block in enumerate(blocks, 1):
        new_id = f"occ_block_{new_idx:03d}"
        block["block_id"] = new_id
        for row in block["slot_results"]:  # type: ignore[index]
            row["block_id"] = new_id
    slot_rows = [row for block in blocks for row in block["slot_results"]]  # type: ignore[index]
    slot_rows.sort(key=lambda row: (str(row["block_id"]), int(row["block_index"])))
    return blocks, slot_rows


def summarize(
    candidates: dict[str, dict[str, object]],
    chains: list[dict[str, object]],
    slot_rows: list[dict[str, object]],
    blocks: list[dict[str, object]],
    block_rows: list[dict[str, object]],
) -> dict[str, object]:
    slot_status_counts = Counter(str(row["chain_status"]) for row in slot_rows)
    block_status_counts = Counter(str(row["block_status"]) for row in block_rows)
    level_counts = Counter(str(cand["evidence_level"]) for cand in candidates.values())
    long_chains = [chain for chain in chains if int(chain["chain_length"]) >= 3]
    confirmed_blocks = [block for block in blocks if block["block_status"] == "occupied_block_confirmed"]
    return {
        "candidate_slot_count": len(candidates),
        "candidate_evidence_level_counts": dict(level_counts),
        "block_candidate_slot_count": sum(1 for cand in candidates.values() if cand.get("block_eligible", True)),
        "chain_count": len(chains),
        "chain_length_ge_3_count": len(long_chains),
        "likely_occupied_chain_supported_count": slot_status_counts.get("likely_occupied_chain_supported", 0),
        "chain_edge_uncertain_count": slot_status_counts.get("chain_edge_uncertain", 0),
        "chain_too_short_count": slot_status_counts.get("chain_too_short", 0),
        "occupied_block_count": len(confirmed_blocks),
        "occupied_block_confirmed_slot_count": block_status_counts.get("occupied_block_confirmed", 0),
        "block_rejected_slot_count": block_status_counts.get("block_rejected", 0),
        "longest_chain_length": max((int(chain["chain_length"]) for chain in chains), default=0),
        "longest_block_length": max((int(block["block_length"]) for block in blocks), default=0),
        "note": "Chain outputs are diagnostic. occupied_blocks.json is consumed by the unified decision table.",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "chain_id",
        "slot_id",
        "case_id",
        "chain_position",
        "chain_index",
        "chain_length",
        "chain_status",
        "chain_confidence",
        "evidence_level",
        "evidence_source",
        "uncertainty_type",
        "support_frame_count",
        "clear_core_owned_frames",
        "max_core_hit_cell_ratio",
        "boundary_conflict_frame_ratio",
        "camera_visible_frame_count",
        "best_camera_projection_score",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["reason"] = " | ".join(row.get("reason", []))
            writer.writerow(out)


def write_block_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "block_id",
        "slot_id",
        "case_id",
        "block_index",
        "block_length",
        "block_position",
        "block_status",
        "block_confidence",
        "evidence_level",
        "evidence_source",
        "uncertainty_type",
        "support_frame_count",
        "clear_core_owned_frames",
        "max_core_hit_cell_ratio",
        "boundary_conflict_frame_ratio",
        "risk_flags",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["risk_flags"] = " | ".join(row.get("risk_flags", []))
            out["reason"] = " | ".join(row.get("reason", []))
            writer.writerow(out)


def esc(value: object) -> str:
    return html.escape(str(value))


def row_class(row: dict[str, object]) -> str:
    status = str(row.get("chain_status", ""))
    confidence = str(row.get("chain_confidence", ""))
    if status == "likely_occupied_chain_supported" and confidence == "high":
        return "occ-high"
    if status == "likely_occupied_chain_supported":
        return "occ-medium"
    if status == "chain_edge_uncertain":
        return "edge-uncertain"
    return "neutral"


def block_row_class(row: dict[str, object]) -> str:
    if str(row.get("block_status", "")) == "occupied_block_confirmed":
        return "block-confirmed"
    return "neutral"


def status_badge(row: dict[str, object]) -> str:
    cls = row_class(row)
    return f"<span class='badge {cls}'>{esc(row.get('chain_status', ''))}</span>"


def block_status_badge(row: dict[str, object]) -> str:
    cls = block_row_class(row)
    return f"<span class='badge {cls}'>{esc(row.get('block_status', ''))}</span>"


def draw_chain_map(
    path: Path,
    slots: dict[str, dict[str, object]],
    slot_rows: list[dict[str, object]],
    zoom: bool,
) -> str:
    if not HAS_MPL:
        return ""
    status_by_slot = {str(row["slot_id"]): row for row in slot_rows}
    highlighted = [slot_id for slot_id, row in status_by_slot.items() if row["chain_status"] != "chain_too_short"]
    if zoom and not highlighted:
        return ""

    fig, ax = plt.subplots(figsize=(11, 8), dpi=170)
    for slot_id, slot in slots.items():
        poly = np.asarray(slot["polygon_map"], dtype=np.float64)
        row = status_by_slot.get(slot_id)
        face = "#e5e7eb"
        edge = "#cbd5e1"
        alpha = 0.20
        lw = 0.25
        zorder = 1
        if row:
            cls = row_class(row)
            if cls == "occ-high":
                face, edge, alpha, lw, zorder = "#ef4444", "#991b1b", 0.72, 1.5, 5
            elif cls == "occ-medium":
                face, edge, alpha, lw, zorder = "#f97316", "#9a3412", 0.66, 1.4, 4
            elif cls == "edge-uncertain":
                face, edge, alpha, lw, zorder = "#facc15", "#a16207", 0.62, 1.2, 3
            else:
                face, edge, alpha, lw, zorder = "#94a3b8", "#475569", 0.35, 0.7, 2
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=zorder))

    for row in slot_rows:
        if row["chain_status"] == "chain_too_short":
            continue
        c = center(slots[str(row["slot_id"])])
        ax.text(c[0], c[1], str(row["slot_id"]).replace("slot_", ""), fontsize=5.5, ha="center", va="center", color="#111827", zorder=8)

    all_pts = np.vstack([np.asarray(slot["polygon_map"], dtype=np.float64) for slot in slots.values()])
    if zoom:
        pts = np.vstack([np.asarray(slots[slot_id]["polygon_map"], dtype=np.float64) for slot_id in highlighted])
        pad = 0.7
    else:
        pts = all_pts
        pad = 0.5
    bmin = pts.min(axis=0) - pad
    bmax = pts.max(axis=0) + pad
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Occupied chain map view" + (" - zoom" if zoom else " - full map"))
    legend = [
        MplPolygon([[0, 0]], closed=True, facecolor="#ef4444", edgecolor="#991b1b", alpha=0.72, label="high likely occupied"),
        MplPolygon([[0, 0]], closed=True, facecolor="#f97316", edgecolor="#9a3412", alpha=0.66, label="medium likely occupied"),
        MplPolygon([[0, 0]], closed=True, facecolor="#facc15", edgecolor="#a16207", alpha=0.62, label="edge uncertain"),
        MplPolygon([[0, 0]], closed=True, facecolor="#e5e7eb", edgecolor="#cbd5e1", alpha=0.20, label="other slots"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path.name


def draw_block_map(
    path: Path,
    slots: dict[str, dict[str, object]],
    block_rows: list[dict[str, object]],
    zoom: bool,
) -> str:
    if not HAS_MPL:
        return ""
    row_by_slot = {str(row["slot_id"]): row for row in block_rows}
    highlighted = [slot_id for slot_id, row in row_by_slot.items() if row["block_status"] == "occupied_block_confirmed"]
    if zoom and not highlighted:
        return ""

    fig, ax = plt.subplots(figsize=(11, 8), dpi=170)
    for slot_id, slot in slots.items():
        row = row_by_slot.get(slot_id)
        face, edge, alpha, lw, zorder = "#e5e7eb", "#cbd5e1", 0.18, 0.25, 1
        if row:
            if row["block_status"] == "occupied_block_confirmed":
                face, edge, alpha, lw, zorder = "#b91c1c", "#450a0a", 0.78, 1.8, 5
            else:
                face, edge, alpha, lw, zorder = "#94a3b8", "#475569", 0.34, 0.7, 2
        poly = np.asarray(slot["polygon_map"], dtype=np.float64)
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=zorder))

    for slot_id in highlighted:
        c = center(slots[slot_id])
        ax.text(c[0], c[1], slot_id.replace("slot_", ""), fontsize=5.5, ha="center", va="center", color="#111827", zorder=8)

    all_pts = np.vstack([np.asarray(slot["polygon_map"], dtype=np.float64) for slot in slots.values()])
    if zoom:
        pts = np.vstack([np.asarray(slots[slot_id]["polygon_map"], dtype=np.float64) for slot_id in highlighted])
        pad = 0.7
    else:
        pts = all_pts
        pad = 0.5
    bmin = pts.min(axis=0) - pad
    bmax = pts.max(axis=0) + pad
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Occupied block map view" + (" - zoom" if zoom else " - full map"))
    legend = [
        MplPolygon([[0, 0]], closed=True, facecolor="#b91c1c", edgecolor="#450a0a", alpha=0.78, label="occupied block confirmed"),
        MplPolygon([[0, 0]], closed=True, facecolor="#94a3b8", edgecolor="#475569", alpha=0.34, label="block candidate rejected"),
        MplPolygon([[0, 0]], closed=True, facecolor="#e5e7eb", edgecolor="#cbd5e1", alpha=0.18, label="other slots"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path.name


def write_report(path: Path, summary: dict[str, object], chains: list[dict[str, object]], slot_rows: list[dict[str, object]], map_images: list[str]) -> None:
    top_chains = [chain for chain in chains if int(chain["chain_length"]) >= 2][:30]
    chain_parts = []
    for chain in top_chains:
        rows_html = []
        for row in chain["slot_results"]:  # type: ignore[index]
            rows_html.append(
                f"<tr class='{row_class(row)}'>"
                f"<td>{esc(row['slot_id'])}</td>"
                f"<td>{esc(row['chain_position'])}</td>"
                f"<td>{status_badge(row)}</td>"
                f"<td>{esc(row['chain_confidence'])}</td>"
                f"<td>{esc(row['evidence_level'])}</td>"
                f"<td>{float(row['max_core_hit_cell_ratio']):.3f}</td>"
                f"<td>{float(row['boundary_conflict_frame_ratio']):.2f}</td>"
                f"<td>{esc(row['uncertainty_type'])}</td>"
                "</tr>"
            )
        chain_parts.append(
            "<section>"
            f"<h2>{esc(chain['chain_id'])}: length {chain['chain_length']}</h2>"
            f"<p>slots: {esc(', '.join(chain['slots']))}</p>"
            f"<p>interior likely occupied: {esc(', '.join(chain['interior_likely_occupied_slots'])) or 'none'} | "
            f"edge uncertain: {esc(', '.join(chain['edge_uncertain_slots'])) or 'none'}</p>"
            "<table><tr><th>slot</th><th>position</th><th>status</th><th>confidence</th><th>level</th><th>core hit</th><th>boundary</th><th>type</th></tr>"
            f"{''.join(rows_html)}</table>"
            "</section>"
        )

    likely_rows = [row for row in slot_rows if row["chain_status"] == "likely_occupied_chain_supported"]
    likely_html = []
    for row in likely_rows[:80]:
        likely_html.append(
            f"<tr class='{row_class(row)}'>"
            f"<td>{esc(row['slot_id'])}</td>"
            f"<td>{esc(row['chain_id'])}</td>"
            f"<td>{esc(row['chain_confidence'])}</td>"
            f"<td>{esc(row['evidence_level'])}</td>"
            f"<td>{float(row['max_core_hit_cell_ratio']):.3f}</td>"
            f"<td>{float(row['boundary_conflict_frame_ratio']):.2f}</td>"
            f"<td>{esc(row['case_id'])}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Occupied Chain Analysis</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    tr.occ-high td {{ background: #fee2e2; border-color: #fecaca; }}
    tr.occ-medium td {{ background: #ffedd5; border-color: #fed7aa; }}
    tr.edge-uncertain td {{ background: #fef9c3; border-color: #fde68a; }}
    .badge {{ display: inline-block; border-radius: 4px; padding: 2px 6px; font-weight: 650; }}
    .badge.occ-high {{ background: #dc2626; color: white; }}
    .badge.occ-medium {{ background: #ea580c; color: white; }}
    .badge.edge-uncertain {{ background: #ca8a04; color: white; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 18px; }}
    img.map {{ width: min(100%, 1100px); border: 1px solid #d1d5db; background: #fff; }}
    section {{ border-top: 1px solid #d1d5db; padding: 14px 0; }}
    pre {{ background: #f9fafb; border: 1px solid #d1d5db; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Occupied Chain Analysis</h1>
  <p>This report is diagnostic only. It does not change Part 1 final states.</p>
  <div class="legend">
    <span class="badge occ-high">high-confidence likely occupied</span>
    <span class="badge occ-medium">medium likely occupied</span>
    <span class="badge edge-uncertain">edge uncertain</span>
  </div>
  <h2>Summary</h2>
  <pre>{esc(json.dumps(summary, indent=2))}</pre>
  <h2>Map View</h2>
  {''.join(f'<p><img class="map" src="{esc(img)}" alt="{esc(img)}"></p>' for img in map_images if img)}
  <h2>Likely Occupied Chain-Supported Interior Slots</h2>
  <table><tr><th>slot</th><th>chain</th><th>confidence</th><th>level</th><th>core hit</th><th>boundary</th><th>case</th></tr>{''.join(likely_html)}</table>
  <h2>Top Chains</h2>
  {''.join(chain_parts)}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_block_report(path: Path, summary: dict[str, object], blocks: list[dict[str, object]], block_rows: list[dict[str, object]], map_images: list[str]) -> None:
    confirmed = [block for block in blocks if block["block_status"] == "occupied_block_confirmed"]
    block_parts = []
    for block in confirmed[:50]:
        rows_html = []
        for row in block["slot_results"]:  # type: ignore[index]
            rows_html.append(
                f"<tr class='{block_row_class(row)}'>"
                f"<td>{esc(row['slot_id'])}</td>"
                f"<td>{esc(row['block_position'])}</td>"
                f"<td>{block_status_badge(row)}</td>"
                f"<td>{esc(row['evidence_level'])}</td>"
                f"<td>{esc(row['evidence_source'])}</td>"
                f"<td>{float(row['max_core_hit_cell_ratio']):.3f}</td>"
                f"<td>{float(row['boundary_conflict_frame_ratio']):.2f}</td>"
                f"<td>{esc(row['uncertainty_type'])}</td>"
                f"<td>{esc(' | '.join(row.get('risk_flags', [])))}</td>"
                "</tr>"
            )
        block_parts.append(
            "<section>"
            f"<h2>{esc(block['block_id'])}: length {block['block_length']} | {esc(block['block_confidence'])}</h2>"
            f"<p>slots: {esc(', '.join(block['slots']))}</p>"
            f"<p>sources: {esc(json.dumps(block['evidence_source_counts'], ensure_ascii=False))}</p>"
            f"<p>reason: {esc(' | '.join(block['reason']))}</p>"
            "<table><tr><th>slot</th><th>position</th><th>status</th><th>level</th><th>source</th><th>core hit</th><th>boundary</th><th>type</th><th>risk flags</th></tr>"
            f"{''.join(rows_html)}</table>"
            "</section>"
        )

    confirmed_rows = [row for row in block_rows if row["block_status"] == "occupied_block_confirmed"]
    rows_html = []
    for row in confirmed_rows[:100]:
        rows_html.append(
            f"<tr class='{block_row_class(row)}'>"
            f"<td>{esc(row['slot_id'])}</td>"
            f"<td>{esc(row['block_id'])}</td>"
            f"<td>{esc(row['block_position'])}</td>"
            f"<td>{esc(row['evidence_level'])}</td>"
            f"<td>{esc(row['evidence_source'])}</td>"
            f"<td>{float(row['max_core_hit_cell_ratio']):.3f}</td>"
            f"<td>{float(row['boundary_conflict_frame_ratio']):.2f}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Occupied Block Confirmation</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    tr.block-confirmed td {{ background: #fee2e2; border-color: #fecaca; }}
    .badge {{ display: inline-block; border-radius: 4px; padding: 2px 6px; font-weight: 650; }}
    .badge.block-confirmed {{ background: #b91c1c; color: white; }}
    img.map {{ width: min(100%, 1100px); border: 1px solid #d1d5db; background: #fff; }}
    section {{ border-top: 1px solid #d1d5db; padding: 14px 0; }}
    pre {{ background: #f9fafb; border: 1px solid #d1d5db; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Occupied Block Confirmation</h1>
  <p>This confirms contiguous occupied-like slot blocks. It absorbs adjacent-slot attribution ambiguity inside each confirmed block.</p>
  <h2>Summary</h2>
  <pre>{esc(json.dumps(summary, indent=2))}</pre>
  <h2>Map View</h2>
  {''.join(f'<p><img class="map" src="{esc(img)}" alt="{esc(img)}"></p>' for img in map_images if img)}
  <h2>Block-Confirmed Slots</h2>
  <table><tr><th>slot</th><th>block</th><th>position</th><th>level</th><th>source</th><th>core hit</th><th>boundary</th></tr>{''.join(rows_html)}</table>
  <h2>Confirmed Blocks</h2>
  {''.join(block_parts)}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.part2_pack / "occupied_chain_analysis")
    ensure_dir(output_dir)

    slot_db = load_json(args.part1_dir / "slot_database.json")
    slots = slot_map(slot_db)  # type: ignore[arg-type]
    scale = float(slot_db["map_units_per_meter"])  # type: ignore[index]

    candidates = build_candidates(args.part1_dir, args.part2_pack, args)
    chains, slot_rows = build_chains(slots, candidates, scale, args)
    blocks, block_rows = build_occupied_blocks(slots, candidates, scale, args)
    summary = summarize(candidates, chains, slot_rows, blocks, block_rows)
    map_images = [
        draw_chain_map(output_dir / "occupied_chain_map_full.png", slots, slot_rows, zoom=False),
        draw_chain_map(output_dir / "occupied_chain_map_zoom.png", slots, slot_rows, zoom=True),
    ]
    block_map_images = [
        draw_block_map(output_dir / "occupied_block_map_full.png", slots, block_rows, zoom=False),
        draw_block_map(output_dir / "occupied_block_map_zoom.png", slots, block_rows, zoom=True),
    ]

    write_json(
        output_dir / "occupied_chain_analysis.json",
        {"summary": summary, "chains": chains, "slot_results": slot_rows, "occupied_blocks": blocks, "block_slot_results": block_rows},
    )
    write_json(output_dir / "occupied_blocks.json", {"summary": summary, "blocks": blocks, "slot_results": block_rows})
    write_csv(output_dir / "occupied_chain_analysis.csv", slot_rows)
    write_block_csv(output_dir / "occupied_blocks.csv", block_rows)
    write_report(output_dir / "occupied_chain_report.html", summary, chains, slot_rows, map_images)
    write_block_report(output_dir / "occupied_block_report.html", summary, blocks, block_rows, block_map_images)

    print(f"[done] output={output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
