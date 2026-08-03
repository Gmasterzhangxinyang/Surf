#!/usr/bin/env python3
"""Build a legacy route-history audit, not an official Part1 map.

The current Part1 contract is a local snapshot from a few consecutive LiDAR
frames.  This script is retained only for explicitly requested historical
audits of old route-wide artifacts.  Its full slot-database backdrop must not
be presented as current perception or parking-lot occupancy truth.

The report deliberately keeps route scope separate from occupancy state:

* ``occupied`` / ``free`` / ``unknown`` are decisions for in-route slots;
* ``partial`` means the slot was only partially observable and therefore has no
  terminal occupancy decision;
* ``out_of_route`` means the recorded path did not observe the slot.

Every slot in the slot database must have one scope row.  This strict merge
prevents a missing decision or scope record from being rendered as a genuine
``unknown`` decision.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon as MplPolygon


DEFAULT_SLOT_DATABASE = Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json")
DEFAULT_OUTPUT_SUBDIR = "global_map_review"


@dataclass(frozen=True)
class StateStyle:
    face: str
    edge: str
    alpha: float
    label: str
    zorder: int


STATE_STYLES = {
    "occupied": StateStyle("#dc2626", "#7f1d1d", 0.86, "occupied", 6),
    "free": StateStyle("#16a34a", "#14532d", 0.76, "free", 5),
    "unknown": StateStyle("#facc15", "#a16207", 0.58, "unknown", 4),
    "partial": StateStyle("#8b5cf6", "#5b21b6", 0.62, "partial route scope", 3),
    "out_of_route": StateStyle("#e5e7eb", "#94a3b8", 0.22, "out of route", 1),
}

VALID_SCOPE_STATUSES = {
    "in_route_scope",
    "partial_route_scope",
    "out_of_route_scope",
}
VALID_DECISION_STATES = {"occupied", "free", "unknown"}


@dataclass(frozen=True)
class MapSlotState:
    slot_id: str
    category: str
    scope_status: str
    decision_state: str
    decision_reason: str
    unknown_reasons: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a LEGACY route-history audit (not a Part1 perception output)"
    )
    parser.add_argument(
        "--legacy-route-audit",
        action="store_true",
        help="Required acknowledgement that this is not the current local Part1 map.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing slot_decisions.csv, known_slot_scope.csv, and summary.json",
    )
    parser.add_argument("--slot-database", type=Path, default=DEFAULT_SLOT_DATABASE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"Output directory (default: INPUT_DIR/{DEFAULT_OUTPUT_SUBDIR})",
    )
    parser.add_argument("--output-name", default="global_map_hybrid_3d_report.html")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_unique(rows: Iterable[dict[str, str]], source: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        slot_id = str(row.get("slot_id", "")).strip()
        if not slot_id:
            raise ValueError(f"{source} contains an empty slot_id")
        if slot_id in indexed:
            raise ValueError(f"duplicate slot_id in {source}: {slot_id}")
        indexed[slot_id] = row
    return indexed


def _json_string_list(raw: str, *, slot_id: str) -> tuple[str, ...]:
    if not raw:
        return ()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid unknown_reasons JSON for {slot_id}") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"unknown_reasons must be a JSON string list for {slot_id}")
    return tuple(value)


def merge_slot_states(
    slots: dict[str, dict[str, Any]],
    scope_rows: Iterable[dict[str, str]],
    decision_rows: Iterable[dict[str, str]],
) -> list[MapSlotState]:
    """Strictly merge map, scope, and decision records into render states."""

    scopes = _index_unique(scope_rows, "known_slot_scope.csv")
    decisions = _index_unique(decision_rows, "slot_decisions.csv")
    slot_ids = set(slots)
    scope_ids = set(scopes)

    missing_scope = sorted(slot_ids - scope_ids)
    extra_scope = sorted(scope_ids - slot_ids)
    if missing_scope or extra_scope:
        raise ValueError(
            "scope/database slot mismatch: "
            f"missing_scope={missing_scope[:5]}, extra_scope={extra_scope[:5]}"
        )

    route_slot_ids: set[str] = set()
    for slot_id, row in scopes.items():
        scope_status = row.get("scope_status", "")
        if scope_status not in VALID_SCOPE_STATUSES:
            raise ValueError(f"invalid scope_status for {slot_id}: {scope_status!r}")
        if scope_status != "out_of_route_scope":
            route_slot_ids.add(slot_id)

    missing_decision = sorted(route_slot_ids - set(decisions))
    unexpected_decision = sorted(set(decisions) - route_slot_ids)
    if missing_decision or unexpected_decision:
        raise ValueError(
            "decision/scope slot mismatch: "
            f"missing_decision={missing_decision[:5]}, unexpected_decision={unexpected_decision[:5]}"
        )

    merged: list[MapSlotState] = []
    for slot_id in sorted(slots):
        scope_status = scopes[slot_id]["scope_status"]
        if scope_status == "out_of_route_scope":
            merged.append(MapSlotState(slot_id, "out_of_route", scope_status, "", "", ()))
            continue

        decision = decisions[slot_id]
        decision_scope = decision.get("scope_status", "")
        if decision_scope and decision_scope != scope_status:
            raise ValueError(
                f"scope_status disagrees between files for {slot_id}: "
                f"{scope_status!r} != {decision_scope!r}"
            )
        decision_state = decision.get("state", "")
        if decision_state not in VALID_DECISION_STATES:
            raise ValueError(f"invalid decision state for {slot_id}: {decision_state!r}")
        reason = decision.get("decision_reason", "")
        unknown_reasons = _json_string_list(decision.get("unknown_reasons", ""), slot_id=slot_id)
        category = "partial" if scope_status == "partial_route_scope" else decision_state
        merged.append(
            MapSlotState(
                slot_id=slot_id,
                category=category,
                scope_status=scope_status,
                decision_state=decision_state,
                decision_reason=reason,
                unknown_reasons=unknown_reasons,
            )
        )
    return merged


def _slot_polygon(slot: dict[str, Any], slot_id: str) -> np.ndarray:
    polygon = np.asarray(slot.get("polygon_map"), dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError(f"invalid polygon_map for {slot_id}")
    if not np.isfinite(polygon).all():
        raise ValueError(f"non-finite polygon_map for {slot_id}")
    return polygon


def draw_map(
    path: Path,
    slots: dict[str, dict[str, Any]],
    states: list[MapSlotState],
    *,
    zoom_to_route: bool,
) -> None:
    state_by_slot = {state.slot_id: state for state in states}
    polygons = {slot_id: _slot_polygon(slot, slot_id) for slot_id, slot in slots.items()}
    all_points = np.vstack([polygons[slot_id] for slot_id in sorted(polygons)])
    counts = Counter(state.category for state in states)

    fig, ax = plt.subplots(figsize=(13, 10), dpi=190)
    # Draw lower-priority categories first so terminal states remain visible at
    # the rare boundaries where annotated polygons overlap.
    ordered_ids = sorted(slots, key=lambda slot_id: (STATE_STYLES[state_by_slot[slot_id].category].zorder, slot_id))
    for slot_id in ordered_ids:
        state = state_by_slot[slot_id]
        style = STATE_STYLES[state.category]
        linewidth = 0.25 if state.category == "out_of_route" else 0.58
        if state.category in {"occupied", "free"}:
            linewidth = 1.10
        ax.add_patch(
            MplPolygon(
                polygons[slot_id],
                closed=True,
                facecolor=style.face,
                edgecolor=style.edge,
                alpha=style.alpha,
                linewidth=linewidth,
                zorder=style.zorder,
            )
        )

    # IDs are limited to terminal decisions and partial slots to avoid turning
    # hundreds of unknown labels into an unreadable solid block.
    for state in states:
        if state.category not in {"occupied", "free", "partial"}:
            continue
        slot = slots[state.slot_id]
        center = np.asarray(slot.get("center_map"), dtype=np.float64)
        if center.shape != (2,) or not np.isfinite(center).all():
            center = polygons[state.slot_id].mean(axis=0)
        ax.text(
            float(center[0]),
            float(center[1]),
            state.slot_id.removeprefix("slot_"),
            fontsize=4.6,
            ha="center",
            va="center",
            color="#111827",
            zorder=10,
        )

    if zoom_to_route:
        route_ids = [state.slot_id for state in states if state.category != "out_of_route"]
        focus_points = np.vstack([polygons[slot_id] for slot_id in route_ids]) if route_ids else all_points
        title = "Hybrid 3D occupancy decisions — route-scope zoom"
    else:
        focus_points = all_points
        title = "Hybrid 3D occupancy decisions — full annotated map"
    extent = focus_points.max(axis=0) - focus_points.min(axis=0)
    margin = max(0.25, float(max(extent)) * 0.025)
    lower = focus_points.min(axis=0) - margin
    upper = focus_points.max(axis=0) + margin
    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    ax.set_title(title)
    handles = [
        Patch(
            facecolor=style.face,
            edgecolor=style.edge,
            alpha=style.alpha,
            label=f"{style.label} ({counts.get(category, 0)})",
        )
        for category, style in STATE_STYLES.items()
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": "ParkingAgent hybrid 3D global-map reporter"})
    plt.close(fig)


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _embedded_png(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def write_html_report(
    path: Path,
    full_path: Path,
    zoom_path: Path,
    states: list[MapSlotState],
    summary: dict[str, Any],
) -> None:
    counts = Counter(state.category for state in states)
    decision_reasons = Counter(
        state.decision_reason
        for state in states
        if state.category == "unknown" and state.decision_reason
    )
    reason_rows = "".join(
        f"<tr><td>{_esc(reason)}</td><td>{count}</td></tr>"
        for reason, count in sorted(decision_reasons.items(), key=lambda item: (-item[1], item[0]))
    )
    count_cards = "".join(
        f'<div class="card"><span class="swatch" style="background:{style.face}"></span>'
        f"<b>{_esc(style.label)}</b><strong>{counts.get(category, 0)}</strong></div>"
        for category, style in STATE_STYLES.items()
    )
    pipeline = summary.get("pipeline", "") if isinstance(summary, dict) else ""
    gt_status = summary.get("gt_status", "") if isinstance(summary, dict) else ""
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Hybrid 3D Global Occupancy Map</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1500px; margin: auto; }}
    .meta {{ color: #475569; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0; }}
    .card {{ display: grid; grid-template-columns: 16px auto auto; gap: 8px; align-items: center;
             padding: 9px 12px; border: 1px solid #cbd5e1; border-radius: 8px; background: white; }}
    .card strong {{ margin-left: 8px; font-size: 1.15rem; }}
    .swatch {{ width: 14px; height: 14px; border: 1px solid #64748b; }}
    figure {{ margin: 18px 0 30px; padding: 12px; background: white; border: 1px solid #d1d5db; }}
    img {{ display: block; width: 100%; height: auto; }}
    table {{ border-collapse: collapse; min-width: 360px; background: white; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }}
    th {{ background: #e2e8f0; }}
    code {{ background: #e2e8f0; padding: 1px 4px; }}
  </style>
</head>
<body><main>
  <h1>LEGACY Route-History Audit — Not Current Part1 Perception</h1>
  <p style="padding:12px;border:2px solid #b91c1c;background:#fef2f2;color:#7f1d1d">
    This page combines historical observations with the complete slot-database backdrop.
    It is not a local LiDAR snapshot, not a current Camera/LiDAR view, and not parking-lot
    occupancy ground truth. Current Part1 uses <code>local_map.json</code> and
    <code>local_map.png</code>; unobserved far slots receive no state there.
  </p>
  <p class="meta">Pipeline: <code>{_esc(pipeline)}</code>; GT status: <code>{_esc(gt_status)}</code>;
    annotated slots rendered: <b>{len(states)}</b>.</p>
  <p>Occupied, free, and unknown apply only to fully in-route slots. Partial scope and out-of-route
    are coverage states, not occupancy decisions.</p>
  <div class="cards">{count_cards}</div>
  <h2>Full annotated map</h2>
  <figure><img alt="Full hybrid 3D global occupancy map" src="{_embedded_png(full_path)}"></figure>
  <h2>Route-scope zoom</h2>
  <figure><img alt="Hybrid 3D route-scope occupancy map" src="{_embedded_png(zoom_path)}"></figure>
  <h2>Unknown decision reasons</h2>
  <table><thead><tr><th>Decision reason</th><th>Slots</th></tr></thead>
    <tbody>{reason_rows or '<tr><td colspan="2">None</td></tr>'}</tbody></table>
</main></body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def build_report(input_dir: Path, slot_database: Path, output_dir: Path, output_name: str) -> dict[str, Any]:
    decision_path = input_dir / "slot_decisions.csv"
    scope_path = input_dir / "known_slot_scope.csv"
    summary_path = input_dir / "summary.json"
    for required in (decision_path, scope_path, summary_path, slot_database):
        if not required.is_file():
            raise FileNotFoundError(f"required input does not exist: {required}")

    slot_db = load_json(slot_database)
    raw_slots = slot_db.get("slots") if isinstance(slot_db, dict) else None
    if not isinstance(raw_slots, list) or not raw_slots:
        raise ValueError("slot database must contain a non-empty 'slots' list")
    slots = _index_unique(raw_slots, "slot database")
    states = merge_slot_states(slots, read_csv(scope_path), read_csv(decision_path))
    summary = load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("summary.json must contain a JSON object")

    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "global_map_hybrid_3d_full.png"
    zoom_path = output_dir / "global_map_hybrid_3d_route_zoom.png"
    report_path = output_dir / output_name
    draw_map(full_path, slots, states, zoom_to_route=False)
    draw_map(zoom_path, slots, states, zoom_to_route=True)
    write_html_report(report_path, full_path, zoom_path, states, summary)

    counts = Counter(state.category for state in states)
    return {
        "report": str(report_path),
        "full_map": str(full_path),
        "route_zoom": str(zoom_path),
        "slot_count": len(states),
        "counts": {category: counts.get(category, 0) for category in STATE_STYLES},
    }


def main() -> None:
    args = parse_args()
    if not args.legacy_route_audit:
        raise SystemExit(
            "Refusing to render a global state map as Part1 output. "
            "Use local_map.png from run_hybrid_3d_slot_evidence.py. "
            "Pass --legacy-route-audit only for an explicitly labelled historical audit."
        )
    output_dir = args.output_dir or args.input_dir / DEFAULT_OUTPUT_SUBDIR
    result = build_report(args.input_dir, args.slot_database, output_dir, args.output_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
