#!/usr/bin/env python3
"""Build a self-contained, read-only Part2 shadow global-map report.

The reporter treats Part1 as immutable source data.  It verifies that
``final_route_states.json`` is an exact overlay of those source decisions,
renders the overlaid route state with the established Hybrid 3D map style, and
labels every result as shadow output without ground-truth validation.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from dataclasses import dataclass, fields
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_part2.decision import (  # noqa: E402
    FINAL_ROUTE_STATES_SCHEMA_VERSION,
    SlotResolution,
)
from parking_slot_part2.queueing import canonical_json_bytes, canonical_sha256  # noqa: E402
from parking_slot_part2.shadow_replay import (  # noqa: E402
    BlindSlotJudgement,
    _parse_blind_record,
    _reject_forbidden_fields,
)
from scripts.build_hybrid_3d_global_map_report import (  # noqa: E402
    DEFAULT_SLOT_DATABASE,
    STATE_STYLES,
    MapSlotState,
    _index_unique,
    _slot_polygon,
    load_json,
    merge_slot_states,
    read_csv,
)


DEFAULT_OUTPUT_NAME = "global_map_part2_shadow_report.html"
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
EVIDENCE_ID_PATTERN = re.compile(r"^ev_[0-9a-f]{64}$")
_FINAL_FIELDS = frozenset(
    {"schema_version", "part2_run_id", "queue_id", "base_decisions", "decisions"}
)
_BASE_METADATA_FIELDS = frozenset({"schema_version", "pipeline", "phase"})
_RESOLUTION_FIELDS = frozenset(field.name for field in fields(SlotResolution))


@dataclass(frozen=True, slots=True)
class ShadowOverlay:
    slot_id: str
    base_state: str
    final_state: str
    resolution: SlotResolution

    @property
    def changed(self) -> bool:
        return self.base_state != self.final_state


@dataclass(frozen=True, slots=True)
class EmbeddedMedia:
    evidence_id: str
    slot_id: str
    task_id: str
    sha256: str
    data_url: str


@dataclass(frozen=True, slots=True)
class ProvenanceSets:
    """Disjoint map-border sets; state fill remains the final decision state."""

    evaluated_unknown: frozenset[str]
    resolved: frozenset[str]
    no_card: frozenset[str]


PROVENANCE_STYLES = {
    "evaluated_unknown": {
        "color": "#06b6d4",
        "linewidth": 2.8,
        "linestyle": "solid",
        "label": "Part2 evaluated; final unknown",
    },
    "resolved": {
        "color": "#2563eb",
        "linewidth": 2.8,
        "linestyle": "solid",
        "label": "Part2 resolved / changed",
    },
    "no_card": {
        "color": "#64748b",
        "linewidth": 2.2,
        "linestyle": "dashed",
        "label": "Selected for Part2; no resolution card",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained Part2 shadow global occupancy-map report"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Part1 output directory whose selected decisions are bound to Part2",
    )
    parser.add_argument(
        "--map-input-dir",
        type=Path,
        help="Optional complete Part1 output directory used as the global-map base",
    )
    parser.add_argument("--final-route-states", required=True, type=Path)
    parser.add_argument("--slot-database", type=Path, default=DEFAULT_SLOT_DATABASE)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--media-manifest", type=Path)
    parser.add_argument("--media-dir", type=Path)
    parser.add_argument(
        "--blind-records",
        type=Path,
        help=(
            "Optional strict slot-keyed blind proposals; when supplied they must "
            "cover exactly every Part2 resolution slot"
        ),
    )
    return parser.parse_args()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _json_list(raw: str, *, field_name: str, slot_id: str) -> list[Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {field_name} JSON for {slot_id}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON list for {slot_id}")
    return value


def _index_json_rows(rows: Any, source: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{source} decisions must be an array")
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{source} decision {index} must be an object")
        slot_id = row.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id:
            raise ValueError(f"{source} decision {index} requires slot_id")
        if slot_id in indexed:
            raise ValueError(f"duplicate slot_id in {source}: {slot_id}")
        indexed[slot_id] = row
    return indexed


def _validate_metadata(
    final_payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    base_json: Mapping[str, Any] | None,
) -> None:
    metadata = final_payload.get("base_decisions")
    if not isinstance(metadata, dict) or set(metadata) != _BASE_METADATA_FIELDS:
        raise ValueError("final base_decisions metadata does not match the strict schema")
    expected_from_summary = {
        "schema_version": summary.get("schema_version"),
        "pipeline": summary.get("pipeline"),
        "phase": summary.get("phase"),
    }
    if metadata != expected_from_summary:
        raise ValueError("final base_decisions identity does not match Part1 summary.json")
    if base_json is not None:
        if set(base_json) != _BASE_METADATA_FIELDS | {"decisions"}:
            raise ValueError("Part1 slot_decisions.json does not match the strict envelope")
        json_metadata = {key: base_json[key] for key in _BASE_METADATA_FIELDS}
        if metadata != json_metadata:
            raise ValueError("final base_decisions identity does not match slot_decisions.json")


def _parse_resolution(raw: Any, *, slot_id: str) -> SlotResolution:
    if not isinstance(raw, dict) or set(raw) != _RESOLUTION_FIELDS:
        raise ValueError(f"invalid part2_resolution schema for {slot_id}")
    try:
        resolution = SlotResolution(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid part2_resolution for {slot_id}: {exc}") from exc
    if resolution.slot_id != slot_id:
        raise ValueError(f"Part2 resolution slot mismatch for {slot_id}")
    return resolution


def _validate_csv_identity(
    slot_id: str,
    final_row: Mapping[str, Any],
    base_csv_row: Mapping[str, str],
    resolution: SlotResolution | None,
) -> None:
    base_state = resolution.input_state if resolution is not None else final_row.get("state")
    comparisons = {
        "slot_id": slot_id,
        "scope_status": base_csv_row.get("scope_status"),
        "state": base_csv_row.get("state"),
        "decision_reason": base_csv_row.get("decision_reason", ""),
    }
    recovered = {
        "slot_id": final_row.get("slot_id"),
        "scope_status": final_row.get("scope_status"),
        "state": base_state,
        "decision_reason": final_row.get("decision_reason", ""),
    }
    if recovered != comparisons:
        raise ValueError(f"final decision does not preserve Part1 CSV identity for {slot_id}")

    expected_reasons = _json_list(
        base_csv_row.get("unknown_reasons", "[]"),
        field_name="unknown_reasons",
        slot_id=slot_id,
    )
    if final_row.get("unknown_reasons", []) != expected_reasons:
        raise ValueError(f"final decision changes Part1 unknown_reasons for {slot_id}")
    if "reference_frames" in base_csv_row:
        expected_frames = _json_list(
            base_csv_row.get("reference_frames", "[]"),
            field_name="reference_frames",
            slot_id=slot_id,
        )
        if final_row.get("reference_frames") != expected_frames:
            raise ValueError(f"final decision changes Part1 reference_frames for {slot_id}")


def validate_and_overlay(
    base_states: list[MapSlotState],
    base_csv_rows: Iterable[dict[str, str]],
    final_payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    base_json: Mapping[str, Any] | None,
) -> tuple[list[MapSlotState], list[ShadowOverlay]]:
    """Validate the immutable merge and return map states plus Part2 audit rows."""

    if set(final_payload) != _FINAL_FIELDS:
        raise ValueError("final_route_states fields do not match the strict schema")
    if final_payload.get("schema_version") != FINAL_ROUTE_STATES_SCHEMA_VERSION:
        raise ValueError("final_route_states schema_version is invalid")
    for identity_name in ("part2_run_id", "queue_id"):
        identity = final_payload.get(identity_name)
        if not isinstance(identity, str) or SHA256_PATTERN.fullmatch(identity) is None:
            raise ValueError(f"{identity_name} must be a canonical sha256 identity")
    _validate_metadata(final_payload, summary, base_json)

    base_by_slot = {state.slot_id: state for state in base_states}
    csv_by_slot = _index_unique(base_csv_rows, "slot_decisions.csv")
    route_ids = {
        state.slot_id for state in base_states if state.scope_status != "out_of_route_scope"
    }
    if set(csv_by_slot) != route_ids:
        raise ValueError("Part1 CSV decisions do not exactly match route-scope slots")
    final_by_slot = _index_json_rows(final_payload.get("decisions"), "final_route_states.json")
    if set(final_by_slot) != route_ids:
        missing = sorted(route_ids - set(final_by_slot))
        extra = sorted(set(final_by_slot) - route_ids)
        raise ValueError(
            "final/Part1 route slot mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    json_by_slot: dict[str, dict[str, Any]] | None = None
    if base_json is not None:
        json_by_slot = _index_json_rows(base_json.get("decisions"), "slot_decisions.json")
        if set(json_by_slot) != route_ids:
            raise ValueError("Part1 JSON decisions do not exactly match route-scope slots")

    final_state_by_slot: dict[str, str] = {}
    overlays: list[ShadowOverlay] = []
    task_ids: set[str] = set()
    for slot_id in sorted(route_ids):
        base_state = base_by_slot[slot_id]
        row = final_by_slot[slot_id]
        final_state = row.get("state")
        if final_state not in {"occupied", "free", "unknown"}:
            raise ValueError(f"invalid final decision state for {slot_id}: {final_state!r}")
        if row.get("scope_status") != base_state.scope_status:
            raise ValueError(f"final decision scope mismatch for {slot_id}")

        raw_resolution = row.get("part2_resolution")
        resolution = (
            _parse_resolution(raw_resolution, slot_id=slot_id)
            if raw_resolution is not None
            else None
        )
        if resolution is not None:
            if resolution.task_id in task_ids:
                raise ValueError(f"duplicate Part2 resolution task_id: {resolution.task_id}")
            task_ids.add(resolution.task_id)
            if resolution.scope_status != base_state.scope_status:
                raise ValueError(f"Part2 resolution scope mismatch for {slot_id}")
            if resolution.state != final_state:
                raise ValueError(f"Part2 resolution state mismatch for {slot_id}")
            if base_state.decision_state != "unknown":
                raise ValueError(f"Part2 may only overlay a Part1 unknown: {slot_id}")
            overlays.append(
                ShadowOverlay(
                    slot_id=slot_id,
                    base_state=base_state.decision_state,
                    final_state=final_state,
                    resolution=resolution,
                )
            )
        elif final_state != base_state.decision_state:
            raise ValueError(f"final decision changes a slot without Part2 resolution: {slot_id}")

        _validate_csv_identity(slot_id, row, csv_by_slot[slot_id], resolution)
        if json_by_slot is not None:
            recovered = dict(row)
            recovered.pop("part2_resolution", None)
            if resolution is not None:
                recovered["state"] = resolution.input_state
            if canonical_json_bytes(recovered) != canonical_json_bytes(json_by_slot[slot_id]):
                raise ValueError(
                    f"final decision does not exactly preserve Part1 JSON row for {slot_id}"
                )
        final_state_by_slot[slot_id] = final_state

    rendered: list[MapSlotState] = []
    for base in base_states:
        if base.scope_status == "out_of_route_scope":
            rendered.append(base)
            continue
        final_state = final_state_by_slot[base.slot_id]
        category = "partial" if base.scope_status == "partial_route_scope" else final_state
        rendered.append(
            MapSlotState(
                slot_id=base.slot_id,
                category=category,
                scope_status=base.scope_status,
                decision_state=final_state,
                decision_reason=base.decision_reason,
                unknown_reasons=base.unknown_reasons,
            )
        )
    return rendered, overlays


def _validate_map_base(
    selected_states: list[MapSlotState],
    full_states: list[MapSlotState],
    selected_summary: Mapping[str, Any],
    full_summary: Mapping[str, Any],
) -> None:
    """Bind every selected Part1 decision to the complete immutable map base."""

    identity_fields = ("schema_version", "pipeline", "phase")
    selected_identity = {field: selected_summary.get(field) for field in identity_fields}
    full_identity = {field: full_summary.get(field) for field in identity_fields}
    if selected_identity != full_identity:
        raise ValueError("selected and full-map Part1 pipeline identities do not match")

    full_by_slot = {state.slot_id: state for state in full_states}
    for selected in selected_states:
        full = full_by_slot.get(selected.slot_id)
        if full is None:
            raise ValueError(f"selected Part1 slot is missing from full-map base: {selected.slot_id}")
        selected_identity_row = (
            selected.scope_status,
            selected.decision_state,
            selected.decision_reason,
            selected.unknown_reasons,
        )
        full_identity_row = (
            full.scope_status,
            full.decision_state,
            full.decision_reason,
            full.unknown_reasons,
        )
        if selected_identity_row != full_identity_row:
            raise ValueError(
                f"selected Part1 decision does not match full-map base for {selected.slot_id}"
            )


def _overlay_full_map(
    full_states: list[MapSlotState],
    overlays: list[ShadowOverlay],
) -> list[MapSlotState]:
    overlay_by_slot = {overlay.slot_id: overlay for overlay in overlays}
    rendered: list[MapSlotState] = []
    for base in full_states:
        overlay = overlay_by_slot.get(base.slot_id)
        if overlay is None:
            rendered.append(base)
            continue
        category = (
            "partial"
            if base.scope_status == "partial_route_scope"
            else overlay.final_state
        )
        rendered.append(
            MapSlotState(
                slot_id=base.slot_id,
                category=category,
                scope_status=base.scope_status,
                decision_state=overlay.final_state,
                decision_reason=base.decision_reason,
                unknown_reasons=base.unknown_reasons,
            )
        )
    return rendered


def derive_provenance_sets(
    overlays: Iterable[ShadowOverlay],
    selected_states: Iterable[MapSlotState],
    *,
    enable_no_card: bool,
) -> ProvenanceSets:
    """Return disjoint provenance borders for the two maps.

    ``no_card`` is intentionally disabled for a complete Part1 input.  Without
    a strict selected/full-map distinction, interpreting every non-resolution
    route slot as a failed Part2 selection would be false provenance.
    """

    overlay_rows = tuple(overlays)
    evaluated_unknown = frozenset(
        row.slot_id for row in overlay_rows if row.final_state == "unknown"
    )
    resolved = frozenset(
        row.slot_id for row in overlay_rows if row.final_state in {"occupied", "free"}
    )
    overlay_ids = evaluated_unknown | resolved
    selected_route_ids = frozenset(
        state.slot_id
        for state in selected_states
        if state.scope_status != "out_of_route_scope"
    )
    if not overlay_ids <= selected_route_ids:
        raise ValueError("Part2 overlay slots are not contained in selected route slots")
    no_card = (
        frozenset(selected_route_ids - overlay_ids) if enable_no_card else frozenset()
    )
    return ProvenanceSets(
        evaluated_unknown=evaluated_unknown,
        resolved=resolved,
        no_card=no_card,
    )


def _load_blind_records(
    path: Path | None,
    overlays: Iterable[ShadowOverlay],
) -> dict[str, BlindSlotJudgement]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"blind records do not exist: {path}")
    payload = _load_json_object(path)
    expected = {overlay.slot_id for overlay in overlays}
    supplied = set(payload)
    if any(not isinstance(slot_id, str) or not slot_id for slot_id in payload):
        raise ValueError("blind_records keys must be non-empty slot_id strings")
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        raise ValueError(
            "blind_records must cover exactly every Part2 overlay slot; "
            f"missing={missing}, extra={extra}"
        )

    records: dict[str, BlindSlotJudgement] = {}
    for slot_id in sorted(expected):
        raw = payload[slot_id]
        _reject_forbidden_fields(raw, f"blind_records.{slot_id}")
        records[slot_id] = _parse_blind_record(slot_id, raw)
    return records


def _draw_map_with_provenance(
    path: Path,
    slots: dict[str, dict[str, Any]],
    states: list[MapSlotState],
    provenance: ProvenanceSets,
    *,
    zoom_to_route: bool,
) -> None:
    """Render established final-state fills, then add provenance-only borders."""

    state_by_slot = {state.slot_id: state for state in states}
    polygons = {
        slot_id: _slot_polygon(slot, slot_id) for slot_id, slot in slots.items()
    }
    all_points = np.vstack([polygons[slot_id] for slot_id in sorted(polygons)])
    counts = Counter(state.category for state in states)

    fig, ax = plt.subplots(figsize=(13, 10), dpi=190)
    ordered_ids = sorted(
        slots,
        key=lambda slot_id: (
            STATE_STYLES[state_by_slot[slot_id].category].zorder,
            slot_id,
        ),
    )
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

    provenance_by_name = {
        "evaluated_unknown": provenance.evaluated_unknown,
        "resolved": provenance.resolved,
        "no_card": provenance.no_card,
    }
    for name, slot_ids in provenance_by_name.items():
        style = PROVENANCE_STYLES[name]
        for slot_id in sorted(slot_ids):
            ax.add_patch(
                MplPolygon(
                    polygons[slot_id],
                    closed=True,
                    fill=False,
                    edgecolor=style["color"],
                    linewidth=style["linewidth"],
                    linestyle=style["linestyle"],
                    zorder=20,
                )
            )

    labelled_ids = {
        state.slot_id
        for state in states
        if state.category in {"occupied", "free", "partial"}
    }
    labelled_ids.update(
        provenance.evaluated_unknown | provenance.resolved | provenance.no_card
    )
    for slot_id in sorted(labelled_ids):
        slot = slots[slot_id]
        center = np.asarray(slot.get("center_map"), dtype=np.float64)
        if center.shape != (2,) or not np.isfinite(center).all():
            center = polygons[slot_id].mean(axis=0)
        ax.text(
            float(center[0]),
            float(center[1]),
            slot_id.removeprefix("slot_"),
            fontsize=4.6,
            ha="center",
            va="center",
            color="#111827",
            zorder=21,
        )

    if zoom_to_route:
        route_ids = [
            state.slot_id for state in states if state.category != "out_of_route"
        ]
        focus_points = (
            np.vstack([polygons[slot_id] for slot_id in route_ids])
            if route_ids
            else all_points
        )
        title = "Part2 shadow occupancy — route-scope zoom with provenance"
    else:
        focus_points = all_points
        title = "Part2 shadow occupancy — full annotated map with provenance"
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
    handles: list[Any] = [
        Patch(
            facecolor=style.face,
            edgecolor=style.edge,
            alpha=style.alpha,
            label=f"{style.label} ({counts.get(category, 0)})",
        )
        for category, style in STATE_STYLES.items()
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=PROVENANCE_STYLES[name]["color"],
            linewidth=PROVENANCE_STYLES[name]["linewidth"],
            linestyle=PROVENANCE_STYLES[name]["linestyle"],
            label=(
                f"{PROVENANCE_STYLES[name]['label']} "
                f"({len(provenance_by_name[name])})"
            ),
        )
        for name in ("evaluated_unknown", "resolved", "no_card")
    )
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(
        path,
        metadata={"Software": "ParkingAgent Part2 shadow global-map reporter"},
    )
    plt.close(fig)


def _draw_evaluated_before_after_map(
    path: Path,
    slots: dict[str, dict[str, Any]],
    overlays: Iterable[ShadowOverlay],
) -> None:
    """Compare Part1 and validated Part2 for resolution-bearing slots only."""

    comparison_rows = tuple(sorted(overlays, key=lambda row: row.slot_id))
    comparison_ids = {row.slot_id for row in comparison_rows}
    if len(comparison_ids) != len(comparison_rows):
        raise ValueError("duplicate slot_id in Part2 before/after comparison")
    missing = sorted(comparison_ids - set(slots))
    if missing:
        raise ValueError(
            "Part2 before/after slots are missing map geometry: "
            + ", ".join(missing[:5])
        )

    polygons = {
        slot_id: _slot_polygon(slot, slot_id) for slot_id, slot in slots.items()
    }
    all_points = np.vstack([polygons[slot_id] for slot_id in sorted(polygons)])
    extent = all_points.max(axis=0) - all_points.min(axis=0)
    margin = max(0.25, float(max(extent)) * 0.025)
    lower = all_points.min(axis=0) - margin
    upper = all_points.max(axis=0) + margin

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=190, sharex=True, sharey=True)
    panels = (
        (
            axes[0],
            "Part1 before",
            {row.slot_id: row.base_state for row in comparison_rows},
        ),
        (
            axes[1],
            "Part2 validated after",
            {row.slot_id: row.final_state for row in comparison_rows},
        ),
    )
    for ax, title, state_by_slot in panels:
        # Every non-evaluated slot, including selected/no-card slots, is only a
        # faint unnumbered spatial reference in this dedicated comparison.
        for slot_id in sorted(slots):
            ax.add_patch(
                MplPolygon(
                    polygons[slot_id],
                    closed=True,
                    facecolor="#f8fafc",
                    edgecolor="#94a3b8",
                    alpha=0.16,
                    linewidth=0.28,
                    zorder=1,
                )
            )

        for slot_id in sorted(comparison_ids):
            state = state_by_slot[slot_id]
            if state not in {"occupied", "free", "unknown"}:
                raise ValueError(
                    f"invalid before/after occupancy state for {slot_id}: {state!r}"
                )
            style = STATE_STYLES[state]
            ax.add_patch(
                MplPolygon(
                    polygons[slot_id],
                    closed=True,
                    facecolor=style.face,
                    edgecolor=style.edge,
                    alpha=style.alpha,
                    linewidth=1.35,
                    zorder=10,
                )
            )
            slot = slots[slot_id]
            center = np.asarray(slot.get("center_map"), dtype=np.float64)
            if center.shape != (2,) or not np.isfinite(center).all():
                center = polygons[slot_id].mean(axis=0)
            ax.text(
                float(center[0]),
                float(center[1]),
                slot_id.removeprefix("slot_"),
                fontsize=5.0,
                ha="center",
                va="center",
                color="#111827",
                zorder=11,
            )

        counts = Counter(state_by_slot.values())
        handles = [
            Patch(
                facecolor="#f8fafc",
                edgecolor="#94a3b8",
                alpha=0.35,
                label="context only (not evaluated)",
            )
        ]
        handles.extend(
            Patch(
                facecolor=STATE_STYLES[state].face,
                edgecolor=STATE_STYLES[state].edge,
                alpha=STATE_STYLES[state].alpha,
                label=f"{state} ({counts.get(state, 0)})",
            )
            for state in ("occupied", "free", "unknown")
            if counts.get(state, 0)
        )
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        ax.set_xlim(float(lower[0]), float(upper[0]))
        ax.set_ylim(float(lower[1]), float(upper[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("map x")
        ax.set_ylabel("map y")
        ax.set_title(f"{title} — {len(comparison_rows)} evaluated slots")

    fig.suptitle(
        "SHADOW / NO GT — Part1 before vs Part2 after (resolution-bearing slots only)",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "All other slots are pale, unnumbered context; selected/no-card slots are not evaluated.",
        ha="center",
        color="#475569",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    fig.savefig(
        path,
        metadata={"Software": "ParkingAgent Part2 shadow before/after reporter"},
    )
    plt.close(fig)


def _load_media(
    manifest_path: Path | None,
    media_dir: Path | None,
    final_payload: Mapping[str, Any],
    overlays: list[ShadowOverlay],
) -> dict[str, EmbeddedMedia]:
    if (manifest_path is None) != (media_dir is None):
        raise ValueError("--media-manifest and --media-dir must be supplied together")
    if manifest_path is None or media_dir is None:
        return {}
    if not manifest_path.is_file():
        raise FileNotFoundError(f"media manifest does not exist: {manifest_path}")
    if not media_dir.is_dir():
        raise FileNotFoundError(f"media directory does not exist: {media_dir}")

    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != "part2-shadow-media-manifest/1.0":
        raise ValueError("media manifest schema_version is invalid")
    if manifest.get("queue_id") != final_payload["queue_id"]:
        raise ValueError("media manifest queue_id does not match final_route_states")
    if "manifest_id" in manifest:
        manifest_id = manifest["manifest_id"]
        unsigned = {key: value for key, value in manifest.items() if key != "manifest_id"}
        if manifest_id != canonical_sha256(unsigned):
            raise ValueError("media manifest_id is invalid")
    raw_media = manifest.get("media")
    if not isinstance(raw_media, list):
        raise ValueError("media manifest media must be an array")

    overlay_by_slot = {overlay.slot_id: overlay for overlay in overlays}
    seen_evidence: set[str] = set()
    by_slot: dict[str, EmbeddedMedia] = {}
    for index, record in enumerate(raw_media):
        if not isinstance(record, dict):
            raise ValueError(f"media record {index} must be an object")
        evidence_id = record.get("evidence_id")
        slot_id = record.get("slot_id")
        task_id = record.get("task_id")
        sha256 = record.get("sha256")
        if not isinstance(evidence_id, str) or EVIDENCE_ID_PATTERN.fullmatch(evidence_id) is None:
            raise ValueError(f"media record {index} has invalid evidence_id")
        if evidence_id in seen_evidence:
            raise ValueError(f"duplicate media evidence_id: {evidence_id}")
        seen_evidence.add(evidence_id)
        if not isinstance(slot_id, str) or slot_id not in overlay_by_slot:
            raise ValueError(f"media record {index} does not belong to a Part2 slot")
        if task_id != overlay_by_slot[slot_id].resolution.task_id:
            raise ValueError(f"media task/slot mismatch for {evidence_id}")
        if not isinstance(sha256, str) or SHA256_PATTERN.fullmatch(sha256) is None:
            raise ValueError(f"media record {index} has invalid sha256")
        if record.get("kind") != "lidar_triptych_png":
            raise ValueError(f"media record {index} is not a LiDAR triptych")
        size_bytes = record.get("size_bytes")
        width = record.get("width")
        height = record.get("height")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise ValueError(f"media record {index} has invalid dimensions or size")

        digest = sha256.split(":", 1)[1]
        media_path = media_dir / "sha256" / digest[:2] / f"{digest}.png"
        if not media_path.is_file():
            raise FileNotFoundError(f"content-addressed media is missing: {media_path}")
        if media_path.stat().st_size != size_bytes or _file_sha256(media_path) != sha256:
            raise ValueError(f"media content identity mismatch for {evidence_id}")
        try:
            with Image.open(media_path) as image:
                if image.format != "PNG" or image.size != (width, height):
                    raise ValueError(f"media PNG metadata mismatch for {evidence_id}")
                image.verify()
        except OSError as exc:
            raise ValueError(f"media is not a readable PNG for {evidence_id}") from exc

        content = media_path.read_bytes()
        embedded = EmbeddedMedia(
            evidence_id=evidence_id,
            slot_id=slot_id,
            task_id=task_id,
            sha256=sha256,
            data_url="data:image/png;base64," + base64.b64encode(content).decode("ascii"),
        )
        current = by_slot.get(slot_id)
        if current is None or evidence_id < current.evidence_id:
            by_slot[slot_id] = embedded
    return by_slot


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _embedded_png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _list_text(values: Iterable[str]) -> str:
    values = tuple(values)
    return ", ".join(_esc(value) for value in values) if values else "none"


def write_html_report(
    path: Path,
    before_after_map_path: Path,
    full_map_path: Path,
    route_map_path: Path,
    base_states: list[MapSlotState],
    final_states: list[MapSlotState],
    overlays: list[ShadowOverlay],
    final_payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    media_by_slot: Mapping[str, EmbeddedMedia],
    provenance: ProvenanceSets,
    blind_records: Mapping[str, BlindSlotJudgement],
) -> None:
    base_counts = Counter(state.category for state in base_states)
    final_counts = Counter(state.category for state in final_states)
    transitions = Counter(
        (overlay.base_state, overlay.final_state)
        for overlay in overlays
        if overlay.changed
    )
    unresolved = [overlay for overlay in overlays if overlay.final_state == "unknown"]
    vetoed = [
        overlay
        for overlay in unresolved
        if "terminal_proposal_vetoed" in overlay.resolution.reason_codes
    ]
    veto_reasons = Counter(
        reason
        for overlay in unresolved
        for reason in overlay.resolution.reason_codes
        if reason not in {"gate_unknown", "terminal_proposal_vetoed"}
    )
    unresolved_findings = Counter(
        blind_records[overlay.slot_id].finding
        for overlay in unresolved
        if overlay.slot_id in blind_records
    )
    total_map_slots = len(final_states)
    route_slots = sum(
        state.scope_status != "out_of_route_scope" for state in final_states
    )
    remaining_in_route_unknown = sum(
        state.scope_status == "in_route_scope" and state.decision_state == "unknown"
        for state in final_states
    )
    remaining_partial_unknown = sum(
        state.scope_status == "partial_route_scope" and state.decision_state == "unknown"
        for state in final_states
    )
    remaining_unknown = remaining_in_route_unknown + remaining_partial_unknown

    count_cards = "".join(
        f'<div class="metric"><span class="swatch" style="background:{style.face}"></span>'
        f"<b>{_esc(style.label)}</b><strong>{final_counts.get(category, 0)}</strong>"
        f'<small>Part1 {base_counts.get(category, 0)}</small></div>'
        for category, style in STATE_STYLES.items()
    )
    transition_rows = "".join(
        f"<tr><td>{_esc(source)} &rarr; {_esc(target)}</td><td>{count}</td></tr>"
        for (source, target), count in sorted(transitions.items())
    )
    veto_rows = "".join(
        f"<tr><td><code>{_esc(reason)}</code></td><td>{count}</td></tr>"
        for reason, count in sorted(veto_reasons.items(), key=lambda item: (-item[1], item[0]))
    )
    finding_rows = "".join(
        f"<tr><td><code>{_esc(finding)}</code></td><td>{count}</td></tr>"
        for finding, count in sorted(
            unresolved_findings.items(), key=lambda item: (-item[1], item[0])
        )
    )

    unknown_semantic_rows: list[str] = []
    for overlay in sorted(unresolved, key=lambda row: row.slot_id):
        proposal = blind_records.get(overlay.slot_id)
        if proposal is None:
            proposal_cells = '<td colspan="6"><i>Blind proposal not supplied</i></td>'
        else:
            proposal_cells = (
                f"<td>{_esc(proposal.state)}</td>"
                f"<td><code>{_esc(proposal.finding)}</code></td>"
                f"<td>{_esc(proposal.ownership)}</td>"
                f"<td>{_esc(proposal.visibility)}</td>"
                f"<td>{proposal.confidence:.3f}</td>"
                f"<td>{_list_text(proposal.reason_codes)}</td>"
            )
        unknown_semantic_rows.append(
            f"<tr><td><code>{_esc(overlay.slot_id)}</code></td>{proposal_cells}"
            f"<td><b>{_esc(overlay.final_state)}</b><br>"
            f"{_list_text(overlay.resolution.reason_codes)}</td></tr>"
        )

    card_rows: list[str] = []
    for overlay in sorted(overlays, key=lambda row: (not row.changed, row.slot_id)):
        resolution = overlay.resolution
        proposal = blind_records.get(overlay.slot_id)
        media = media_by_slot.get(overlay.slot_id)
        media_html = ""
        if media is not None:
            media_html = (
                f'<img class="triptych" alt="LiDAR evidence for {_esc(overlay.slot_id)}" '
                f'src="{media.data_url}">'
                f'<p class="hash">media {_esc(media.evidence_id)} / {_esc(media.sha256)}</p>'
            )
        badge_class = "changed" if overlay.changed else "unresolved"
        proposal_html = "<p><b>Blind proposal:</b> not supplied.</p>"
        if proposal is not None:
            proposal_html = (
                f"<p><b>Blind proposal:</b> state <b>{_esc(proposal.state)}</b>; "
                f"finding <code>{_esc(proposal.finding)}</code>; "
                f"ownership {_esc(proposal.ownership)}; visibility {_esc(proposal.visibility)}; "
                f"confidence {proposal.confidence:.3f}.</p>"
                f"<p><b>Blind reason codes:</b> {_list_text(proposal.reason_codes)}</p>"
            )
        card_rows.append(
            f'<article class="case {badge_class}">'
            f"<h3>{_esc(overlay.slot_id)} "
            f'<span>{_esc(overlay.base_state)} &rarr; {_esc(overlay.final_state)}</span></h3>'
            f"{proposal_html}"
            f"<p><b>Validated final:</b> <b>{_esc(overlay.final_state)}</b>; "
            f"<b>completion:</b> {_esc(resolution.completion_status)}; "
            f"<b>source:</b> {_esc(resolution.decision_source)}; "
            f"<b>turns/tools:</b> {resolution.model_turns}/{resolution.tool_calls_attempted}</p>"
            f"<p><b>stop:</b> {_list_text(resolution.stop_reasons)}</p>"
            f"<p><b>reason codes:</b> {_list_text(resolution.reason_codes)}</p>"
            f"<p><b>evidence:</b> {_list_text(resolution.evidence_refs)}</p>"
            f"{media_html}</article>"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Part2 Shadow Global Occupancy Map</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1500px; margin: auto; }}
    .warning {{ padding: 14px 16px; border: 2px solid #b45309; background: #fffbeb; border-radius: 9px; }}
    .meta, .hash {{ color: #475569; overflow-wrap: anywhere; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0; }}
    .metric {{ display: grid; grid-template-columns: 16px auto auto; gap: 8px; align-items: center;
               padding: 9px 12px; border: 1px solid #cbd5e1; border-radius: 8px; background: white; }}
    .metric strong {{ margin-left: 8px; font-size: 1.15rem; }}
    .metric small {{ grid-column: 2 / 4; color: #64748b; }}
    .swatch {{ width: 14px; height: 14px; border: 1px solid #64748b; }}
    figure {{ margin: 18px 0 30px; padding: 12px; background: white; border: 1px solid #d1d5db; }}
    figure img, .triptych {{ display: block; width: 100%; height: auto; }}
    table {{ border-collapse: collapse; min-width: 390px; background: white; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }}
    th {{ background: #e2e8f0; }}
    code {{ background: #e2e8f0; padding: 1px 4px; }}
    .cases {{ display: grid; gap: 12px; }}
    .case {{ padding: 12px; border: 1px solid #cbd5e1; border-left: 6px solid #ca8a04;
             background: white; border-radius: 8px; }}
    .case.changed {{ border-left-color: #2563eb; }}
    .case h3 {{ margin: 0 0 8px; }}
    .case h3 span {{ font-size: .85rem; color: #475569; margin-left: 8px; }}
    .triptych {{ margin-top: 10px; border: 1px solid #cbd5e1; }}
    .hash {{ font-size: .75rem; }}
    .provenance-key {{ display:flex; flex-wrap:wrap; gap:12px; margin:12px 0 20px; }}
    .provenance-key span {{ background:white; padding:7px 10px; border-radius:6px; }}
    .cyan {{ border:3px solid #06b6d4; }}
    .blue {{ border:3px solid #2563eb; }}
    .gray {{ border:3px dashed #64748b; }}
    .semantic {{ font-size:.88rem; min-width:100%; }}
  </style>
</head>
<body><main>
  <h1>Part2 Shadow Global Occupancy Map</h1>
  <p class="warning"><b>SHADOW / NO GT.</b> These Part2 states are not promoted to Part1 and are not
    accuracy claims. The report overlays a validated copy in memory; all source artifacts remain unchanged.</p>
  <p class="meta">Part1 pipeline: <code>{_esc(summary.get('pipeline', ''))}</code>; source GT status:
    <code>{_esc(summary.get('gt_status', 'unavailable'))}</code>; Part2 run:
    <code>{_esc(final_payload['part2_run_id'])}</code>; queue: <code>{_esc(final_payload['queue_id'])}</code>.</p>
  <div class="metrics">{count_cards}</div>
  <div class="metrics">
    <div class="metric"><b>Total map slots</b><strong>{total_map_slots}</strong></div>
    <div class="metric"><b>Route-scope slots</b><strong>{route_slots}</strong></div>
    <div class="metric"><b>Part2 evaluated</b><strong>{len(overlays)}</strong></div>
    <div class="metric"><b>Resolved / changed</b><strong>{len(provenance.resolved)}</strong></div>
    <div class="metric"><b>Selected, no card</b><strong>{len(provenance.no_card)}</strong></div>
    <div class="metric"><b>Remaining decision unknown</b><strong>{remaining_unknown}</strong>
      <small>{remaining_unknown} total = {remaining_in_route_unknown} in-route +
        {remaining_partial_unknown} partial</small></div>
    <div class="metric"><b>Evaluated but unknown</b><strong>{len(unresolved)}</strong></div>
    <div class="metric"><b>Vetoed unknown</b><strong>{len(vetoed)}</strong></div>
  </div>
  <p>Polygon fill is the <b>final occupancy/scope state</b>. Border color is independent
    Part2 provenance:</p>
  <div class="provenance-key">
    <span class="cyan">cyan solid: evaluated, final unknown ({len(provenance.evaluated_unknown)})</span>
    <span class="blue">blue solid: Part2 resolved / changed ({len(provenance.resolved)})</span>
    <span class="gray">gray dashed: selected but no Part2 resolution card ({len(provenance.no_card)})</span>
  </div>
  <h2>Base &rarr; Part2 state changes</h2>
  <table><thead><tr><th>Transition</th><th>Slots</th></tr></thead>
    <tbody>{transition_rows or '<tr><td colspan="2">No state changes</td></tr>'}</tbody></table>
  <h2>Veto / unresolved reason counts</h2>
  <table><thead><tr><th>Reason code</th><th>Slots</th></tr></thead>
    <tbody>{veto_rows or '<tr><td colspan="2">None</td></tr>'}</tbody></table>
  <h2>Evaluated slots: Part1 before vs Part2 after</h2>
  <p>This comparison contains only the {len(overlays)} slots with an actual Part2 resolution.
    The {len(provenance.no_card)} selected/no-card slots remain pale, unnumbered context and are
    not presented as Part2 evaluations.</p>
  <figure><img alt="Part1 before and Part2 validated-after comparison for evaluated slots only"
    src="{_embedded_png(before_after_map_path)}"></figure>
  <h2>Full annotated map</h2>
  <figure><img alt="Full Part2 shadow occupancy map" src="{_embedded_png(full_map_path)}"></figure>
  <h2>Route-scope zoom</h2>
  <figure><img alt="Part2 shadow route-scope occupancy map" src="{_embedded_png(route_map_path)}"></figure>
  <h2>Evaluated-unknown semantic review</h2>
  <p>These are blind model proposals, shown separately from the validated final state.
    They explain why a slot remained unknown; they are not terminal occupancy labels.</p>
  <h3>Finding summary</h3>
  <table><thead><tr><th>Blind finding</th><th>Slots</th></tr></thead>
    <tbody>{finding_rows or '<tr><td colspan="2">Blind records not supplied</td></tr>'}</tbody></table>
  <table class="semantic"><thead><tr><th>Slot</th><th>Blind proposal</th><th>Finding</th>
    <th>Ownership</th><th>Visibility</th><th>Confidence</th><th>Blind reason codes</th>
    <th>Validated final / reason codes</th></tr></thead>
    <tbody>{''.join(unknown_semantic_rows) or '<tr><td colspan="8">No evaluated unknown slots</td></tr>'}</tbody></table>
  <h2>Part2 task cards</h2>
  <section class="cases">{''.join(card_rows) or '<p>No Part2 task rows.</p>'}</section>
</main></body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def build_report(
    input_dir: Path,
    final_route_states_path: Path,
    slot_database: Path,
    output_dir: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
    *,
    map_input_dir: Path | None = None,
    media_manifest: Path | None = None,
    media_dir: Path | None = None,
    blind_records: Path | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    final_route_states_path = Path(final_route_states_path)
    slot_database = Path(slot_database)
    output_dir = Path(output_dir)
    map_input_dir = Path(map_input_dir) if map_input_dir is not None else None
    blind_records = Path(blind_records) if blind_records is not None else None
    required = (
        input_dir / "known_slot_scope.csv",
        input_dir / "slot_decisions.csv",
        input_dir / "summary.json",
        final_route_states_path,
        slot_database,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"required input does not exist: {path}")

    slot_db = load_json(slot_database)
    raw_slots = slot_db.get("slots") if isinstance(slot_db, dict) else None
    if not isinstance(raw_slots, list) or not raw_slots:
        raise ValueError("slot database must contain a non-empty 'slots' list")
    slots = _index_unique(raw_slots, "slot database")
    scope_rows = read_csv(input_dir / "known_slot_scope.csv")
    decision_rows = read_csv(input_dir / "slot_decisions.csv")
    selected_scope_ids = set(_index_unique(scope_rows, "known_slot_scope.csv"))
    missing_selected_geometry = sorted(selected_scope_ids - set(slots))
    if missing_selected_geometry:
        raise ValueError(
            "selected scope contains slots missing from the slot database: "
            + ", ".join(missing_selected_geometry[:5])
        )
    if map_input_dir is None and selected_scope_ids != set(slots):
        raise ValueError(
            "selected Part1 does not cover the complete slot database; "
            "--map-input-dir is required for a full global map"
        )
    selected_slots = {slot_id: slots[slot_id] for slot_id in selected_scope_ids}
    base_states = merge_slot_states(selected_slots, scope_rows, decision_rows)
    summary = _load_json_object(input_dir / "summary.json")
    final_payload = _load_json_object(final_route_states_path)
    base_json_path = input_dir / "slot_decisions.json"
    base_json = _load_json_object(base_json_path) if base_json_path.is_file() else None

    selected_final_states, overlays = validate_and_overlay(
        base_states,
        decision_rows,
        final_payload,
        summary,
        base_json=base_json,
    )
    map_summary = summary
    map_base_states = base_states
    final_states = selected_final_states
    if map_input_dir is not None:
        map_required = (
            map_input_dir / "known_slot_scope.csv",
            map_input_dir / "slot_decisions.csv",
            map_input_dir / "summary.json",
        )
        for path in map_required:
            if not path.is_file():
                raise FileNotFoundError(f"required full-map input does not exist: {path}")
        map_scope_rows = read_csv(map_input_dir / "known_slot_scope.csv")
        map_decision_rows = read_csv(map_input_dir / "slot_decisions.csv")
        map_base_states = merge_slot_states(slots, map_scope_rows, map_decision_rows)
        map_summary = _load_json_object(map_input_dir / "summary.json")
        _validate_map_base(base_states, map_base_states, summary, map_summary)
        final_states = _overlay_full_map(map_base_states, overlays)
    provenance = derive_provenance_sets(
        overlays,
        base_states,
        enable_no_card=(
            map_input_dir is not None and selected_scope_ids < set(slots)
        ),
    )
    media_by_slot = _load_media(media_manifest, media_dir, final_payload, overlays)
    blind_by_slot = _load_blind_records(blind_records, overlays)

    output_dir.mkdir(parents=True, exist_ok=True)
    before_after_map_path = output_dir / "global_map_part2_evaluated_before_after.png"
    full_map_path = output_dir / "global_map_part2_shadow_full.png"
    route_map_path = output_dir / "global_map_part2_shadow_route_zoom.png"
    report_path = output_dir / output_name
    _draw_evaluated_before_after_map(
        before_after_map_path,
        slots,
        overlays,
    )
    _draw_map_with_provenance(
        full_map_path,
        slots,
        final_states,
        provenance,
        zoom_to_route=False,
    )
    _draw_map_with_provenance(
        route_map_path,
        slots,
        final_states,
        provenance,
        zoom_to_route=True,
    )
    write_html_report(
        report_path,
        before_after_map_path,
        full_map_path,
        route_map_path,
        map_base_states,
        final_states,
        overlays,
        final_payload,
        map_summary,
        media_by_slot,
        provenance,
        blind_by_slot,
    )

    final_counts = Counter(state.category for state in final_states)
    transitions = Counter(
        f"{overlay.base_state}->{overlay.final_state}"
        for overlay in overlays
        if overlay.changed
    )
    unresolved = [overlay for overlay in overlays if overlay.final_state == "unknown"]
    remaining_in_route_unknown = sum(
        state.scope_status == "in_route_scope" and state.decision_state == "unknown"
        for state in final_states
    )
    remaining_partial_unknown = sum(
        state.scope_status == "partial_route_scope" and state.decision_state == "unknown"
        for state in final_states
    )
    return {
        "schema_version": "part2-shadow-global-map-report-result/1.0",
        "report": str(report_path),
        "before_after_map": str(before_after_map_path),
        "full_map": str(full_map_path),
        "route_zoom": str(route_map_path),
        "slot_count": len(final_states),
        "route_slot_count": sum(
            state.scope_status != "out_of_route_scope" for state in final_states
        ),
        "part2_task_count": len(overlays),
        "evaluated_unknown_count": len(provenance.evaluated_unknown),
        "resolved_count": len(provenance.resolved),
        "no_card_count": len(provenance.no_card),
        "changed_count": sum(overlay.changed for overlay in overlays),
        "unresolved_count": len(unresolved),
        "remaining_unknown_count": (
            remaining_in_route_unknown + remaining_partial_unknown
        ),
        "remaining_in_route_unknown_count": remaining_in_route_unknown,
        "remaining_partial_unknown_count": remaining_partial_unknown,
        "vetoed_unknown_count": sum(
            "terminal_proposal_vetoed" in overlay.resolution.reason_codes
            for overlay in unresolved
        ),
        "embedded_media_count": len(media_by_slot),
        "blind_record_count": len(blind_by_slot),
        "counts": {category: final_counts.get(category, 0) for category in STATE_STYLES},
        "transitions": dict(sorted(transitions.items())),
    }


def main() -> None:
    args = parse_args()
    result = build_report(
        args.input_dir,
        args.final_route_states,
        args.slot_database,
        args.output_dir,
        args.output_name,
        map_input_dir=args.map_input_dir,
        media_manifest=args.media_manifest,
        media_dir=args.media_dir,
        blind_records=args.blind_records,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
