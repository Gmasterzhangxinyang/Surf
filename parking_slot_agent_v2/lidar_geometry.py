"""Compact, auditable geometry evidence derived from Part1 and a validated pack.

The VLM used to receive a LiDAR triptych but almost none of the deterministic
measurements that produced the Part1 decision.  This module exposes a bounded
subset of those measurements and renders the same values beside the triptych.
It does not make a terminal parking decision and it does not calibrate the
upstream heuristic scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCHEMA_VERSION = "parking-slot-agent-v2-lidar-geometry-card/1.0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _rounded(value: Any, digits: int = 4) -> float | None:
    parsed = _finite(value)
    return None if parsed is None else round(parsed, digits)


def _integer(value: Any) -> int | None:
    parsed = _finite(value)
    return None if parsed is None else int(parsed)


def _part1_decision(case: Any) -> Mapping[str, Any]:
    for evidence in getattr(case, "evidence", ()):
        if getattr(evidence, "tool_name", None) != "part1_15frame":
            continue
        return _mapping(_mapping(getattr(evidence, "metadata", {})).get("decision"))
    return {}


def resolve_effective_decision(
    case: Any,
    decision_override: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return the one decision payload used by cards and visual renderers.

    Extended evidence supplies an explicit decision.  Part1 review mode must
    reuse the decision already attached to the SlotCase instead of silently
    rendering the pack with an empty decision.
    """

    return (
        _mapping(decision_override)
        if decision_override is not None
        else _part1_decision(case)
    )


def _occupied_support(
    occupied: Mapping[str, Any],
    features: Mapping[str, Any],
    stability: Mapping[str, Any],
) -> str:
    failures = {str(item) for item in _sequence(occupied.get("failures"))}
    if bool(occupied.get("strong")) and not failures and not bool(stability.get("stable", False)):
        return "strong_but_pose_unstable_not_terminal"
    if bool(occupied.get("strong")) and not failures:
        return "strong_occupied_geometry_candidate"
    if "linear_static_structure" in failures:
        return "linear_static_structure_risk"
    if "boundary_dominated" in failures or (_finite(features.get("boundary_ratio")) or 0.0) >= 0.5:
        return "boundary_dominated_not_terminal"
    if bool(occupied.get("weak")):
        return "weak_occupied_geometry_not_terminal"
    return "no_occupied_geometry_support"


def _free_support(free: Mapping[str, Any], stability: Mapping[str, Any]) -> str:
    failures = {str(item) for item in _sequence(free.get("failures"))}
    if bool(free.get("strong")) and not failures and not bool(stability.get("stable", False)):
        return "strong_but_pose_unstable_not_terminal"
    if bool(free.get("strong")) and not failures:
        return "strong_free_geometry_candidate"
    if bool(free.get("positive_geometry")):
        return "partial_free_geometry_not_terminal"
    return "insufficient_free_observation"


def build_geometry_card(
    case: Any,
    pack: Any,
    *,
    decision_override: Mapping[str, Any] | None = None,
    decision_source: str = "part1_15frame",
) -> dict[str, Any]:
    """Return bounded numerical evidence for one already validated LiDAR pack."""

    decision = resolve_effective_decision(case, decision_override)
    occupied = _mapping(decision.get("occupied_evidence"))
    free = _mapping(decision.get("free_evidence"))
    features = _mapping(occupied.get("features"))
    stability = _mapping(decision.get("stability"))
    unresolved = [str(item) for item in getattr(case, "unknown_reasons", ())]
    robustness_failures = [str(item) for item in _sequence(stability.get("failures"))]
    if not bool(stability.get("stable", False)) and not robustness_failures:
        robustness_failures = [
            item
            for item in unresolved
            if item in {"pose_sensitive_terminal", "pose_unstable"}
        ] or ["not_stable"]
    valid_frames = tuple(int(value) for value in getattr(pack, "valid_frames", ()))
    pack_selected_frames = tuple(
        int(value) for value in getattr(pack, "selected_frames", ())
    )
    decision_reference_frames = tuple(
        int(value)
        for value in (
            _sequence(decision.get("selected_frames"))
            or _sequence(decision.get("reference_frames"))
        )
    )
    selected_frames = (
        pack_selected_frames or decision_reference_frames or valid_frames
    )
    point_count = len(getattr(pack, "points_local_xyzi", ()))
    occupied_card = {
        "support_level": _occupied_support(occupied, features, stability),
        "heuristic_strength_uncalibrated": _rounded(occupied.get("strength")),
        "strong_gate": bool(occupied.get("strong", False)),
        "weak_gate": bool(occupied.get("weak", False)),
        "failures": [str(item) for item in _sequence(occupied.get("failures"))],
        "candidate_return_count": _integer(features.get("point_count")),
        "core_point_count": _integer(features.get("core_point_count")),
        "support_frame_count": _integer(features.get("supported_frame_count")),
        "temporal_support": _rounded(features.get("temporal_support")),
        "temporal_consistency": _rounded(features.get("temporal_consistency")),
        "supported_height_layers": _integer(features.get("supported_layer_count")),
        "z95_m": _rounded(features.get("z95_m")),
        "height_span_m": _rounded(features.get("height_span_m")),
        "extent_slot_x_m": _rounded(features.get("extent_x_m")),
        "extent_slot_y_m": _rounded(features.get("extent_y_m")),
        "core_overlap": _rounded(features.get("core_overlap")),
        "adjacent_overlap": _rounded(features.get("adjacent_overlap")),
        "boundary_ratio": _rounded(features.get("boundary_ratio")),
        "linearity_risk": _rounded(features.get("linearity")),
        "outside_residual_ratio": _rounded(features.get("outside_residual_ratio")),
    }
    free_card = {
        "support_level": _free_support(free, stability),
        "heuristic_strength_uncalibrated": _rounded(free.get("strength")),
        "strong_gate": bool(free.get("strong", False)),
        "failures": [str(item) for item in _sequence(free.get("failures"))],
        "ray_frame_count": _integer(free.get("ray_frame_count")),
        "viewpoint_count": _integer(free.get("viewpoint_count")),
        "viewpoint_separation_deg": _rounded(free.get("viewpoint_separation_deg")),
        "core_ray_coverage": _rounded(free.get("core_ray_coverage")),
        "observed_volume_ratio": _rounded(free.get("observed_volume_ratio")),
        "near_ground_bev_coverage": _rounded(free.get("near_ground_bev_coverage")),
        "unobserved_component_ratio": _rounded(free.get("unobserved_component_ratio")),
        "occlusion_ratio": _rounded(free.get("occlusion_ratio")),
        "unresolved_core_hit": bool(free.get("unresolved_core_hit", False)),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "terminal_decision": False,
        "scores_calibrated": False,
        "slot_id": str(getattr(case, "slot_id", "")),
        "decision_context": {
            "part1_decision_reason": str(getattr(case, "decision_reason", "")),
            "part1_unresolved_reasons": unresolved,
            "decision_reason": decision.get("decision_reason"),
            "unresolved_reasons": [
                str(item) for item in _sequence(decision.get("unknown_reasons"))
            ],
            "scope_status": decision.get("scope_status"),
            "evidence_source": str(decision_source),
            "evidence_window_frames": len(selected_frames),
            "decision_reference_frame_count": len(decision_reference_frames),
            "extended_decision_reason": decision.get("decision_reason"),
        },
        "pack": {
            "point_count": int(point_count),
            "selected_frame_count": len(selected_frames),
            "selected_frames": list(selected_frames),
            "valid_frame_count": len(valid_frames),
            "valid_frames": list(valid_frames),
        },
        "occupied_geometry": occupied_card,
        "free_geometry": free_card,
        "robustness": {
            "stable": bool(stability.get("stable", False)),
            "passing_variants": _integer(stability.get("passing_variants")),
            "total_variants": _integer(stability.get("total_variants")),
            "pass_ratio": _rounded(stability.get("pass_ratio")),
            "failures": robustness_failures,
        },
        "interpretation_policy": [
            "This card is measured evidence, not a terminal decision.",
            "High boundary ratio or linearity risk is evidence against treating returns as a vehicle.",
            "Free requires adequate observed volume and near-ground coverage without unresolved core hits.",
            "Part1 strengths are uncalibrated and cannot alone satisfy the 0.50 terminal contract.",
        ],
    }


def assess_terminal_geometry(card: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate balanced global geometry gates without using model scores.

    The balanced policy accepts medium clear-space evidence and temporally
    supported vehicle geometry, while retaining explicit pillar/wall/boundary
    vetoes for Occupied.  Thresholds are global and never slot-specific.
    """

    occupied = _mapping(card.get("occupied_geometry"))
    free = _mapping(card.get("free_geometry"))
    robustness = _mapping(card.get("robustness"))
    passed = robustness.get("passing_variants")
    total = robustness.get("total_variants")
    stable_all = bool(
        robustness.get("stable") is True
        and isinstance(passed, int)
        and isinstance(total, int)
        and total > 0
        and passed == total
    )
    pass_ratio_value = _finite(robustness.get("pass_ratio"))
    pass_ratio = (
        pass_ratio_value
        if pass_ratio_value is not None
        else float(passed) / float(total)
        if isinstance(passed, int) and isinstance(total, int) and total > 0
        else 0.0
    )

    occupied_failures = {str(item) for item in _sequence(occupied.get("failures"))}
    static_risk = bool(
        occupied_failures & {
            "boundary_dominated",
            "linear_static_structure",
            "compact_vertical_structure",
        }
    )
    boundary = _finite(occupied.get("boundary_ratio"))
    linearity = _finite(occupied.get("linearity_risk"))
    core_points = occupied.get("core_point_count")
    temporal_support = _finite(occupied.get("temporal_support"))
    support_frames = occupied.get("support_frame_count")
    height_span = _finite(occupied.get("height_span_m"))

    occupied_blockers: list[str] = []
    if occupied.get("strong_gate") is not True:
        occupied_blockers.append("occupied_strong_gate_failed")
    if occupied_failures:
        occupied_blockers.append("occupied_geometry_has_failures")
    if static_risk:
        occupied_blockers.append("pillar_wall_or_boundary_risk")
    if boundary is None or boundary >= 0.5:
        occupied_blockers.append("boundary_ratio_not_below_0_5")
    if linearity is None or linearity >= 0.6:
        occupied_blockers.append("linearity_risk_not_below_0_6")
    if not isinstance(core_points, int) or core_points <= 0:
        occupied_blockers.append("no_core_obstacle_points")
    if not stable_all and (temporal_support is None or temporal_support < 0.30):
        occupied_blockers.append("temporal_support_below_0_30")
    if not stable_all and (not isinstance(support_frames, int) or support_frames < 20):
        occupied_blockers.append("support_frames_below_20")
    if height_span is None or height_span < 0.80:
        occupied_blockers.append("height_span_below_0_80")
    if free.get("strong_gate") is True:
        occupied_blockers.append("opposing_free_strong_gate")

    occupied_vetoed = bool(
        occupied.get("strong_gate") is not True
        or static_risk
        or boundary is None
        or boundary >= 0.5
        or not isinstance(core_points, int)
        or core_points <= 0
    )
    observed_volume = _finite(free.get("observed_volume_ratio"))
    near_ground = _finite(free.get("near_ground_bev_coverage"))
    core_coverage = _finite(free.get("core_ray_coverage"))
    occlusion = _finite(free.get("occlusion_ratio"))
    no_core_hit = free.get("unresolved_core_hit") is False
    strong_free = bool(
        free.get("strong_gate") is True
        and not free.get("failures")
        and no_core_hit
        and pass_ratio >= 0.85
    )
    medium_free = bool(
        no_core_hit
        and observed_volume is not None
        and observed_volume >= 0.80
        and near_ground is not None
        and near_ground >= 0.80
        and core_coverage is not None
        and core_coverage >= 0.80
        and occlusion is not None
        and occlusion <= 0.25
        and occupied_vetoed
    )
    free_blockers: list[str] = []
    if not no_core_hit:
        free_blockers.append("unresolved_core_hit")
    if not strong_free and not medium_free:
        free_blockers.append("neither_strong_nor_medium_free_geometry")
    if not occupied_vetoed:
        free_blockers.append("opposing_occupied_geometry_not_vetoed")

    return {
        "schema_version": "parking-slot-agent-v2-terminal-geometry-gate/1.1-balanced",
        "policy": "balanced_global_no_slot_specific_thresholds",
        "free_eligible": not free_blockers,
        "occupied_eligible": not occupied_blockers,
        "free_support_tier": "strong" if strong_free else "medium" if medium_free else "none",
        "occupied_support_tier": "balanced_strong" if not occupied_blockers else "none",
        "free_blockers": free_blockers,
        "occupied_blockers": occupied_blockers,
        "thresholds": {
            "free_min_observed_volume": 0.80,
            "free_min_near_ground_coverage": 0.80,
            "free_min_core_coverage": 0.80,
            "free_max_occlusion": 0.25,
            "occupied_max_boundary_ratio": 0.50,
            "occupied_max_linearity": 0.60,
            "occupied_min_temporal_support": 0.30,
            "occupied_min_support_frames": 20,
            "occupied_min_height_span_m": 0.80,
        },
    }


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if bold else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf") if bold else Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    *,
    x: int,
    y: int,
    width: int,
    font: ImageFont.ImageFont,
    fill: str,
    line_gap: int = 8,
) -> int:
    cursor = y
    for raw in lines:
        words = str(raw).split()
        current = ""
        wrapped: list[str] = []
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if draw.textlength(candidate, font=font) <= width:
                current = candidate
            else:
                if current:
                    wrapped.append(current)
                current = word
        if current or not wrapped:
            wrapped.append(current)
        for line in wrapped:
            draw.text((x, cursor), line, font=font, fill=fill)
            bbox = draw.textbbox((x, cursor), line or " ", font=font)
            cursor += bbox[3] - bbox[1] + line_gap
    return cursor


def render_geometry_card(card: Mapping[str, Any], path: str | Path) -> Path:
    """Render the exact compact JSON fields as a model/audit image."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1536, 780), "#f8fafc")
    draw = ImageDraw.Draw(image)
    title = _font(34, bold=True)
    heading = _font(25, bold=True)
    body = _font(20)
    small = _font(18)
    draw.rectangle((0, 0, 1536, 72), fill="#0f172a")
    context = _mapping(card.get("decision_context"))
    window_frames = context.get("evidence_window_frames") or _mapping(card.get("pack")).get("valid_frame_count", 0)
    draw.text(
        (28, 18),
        f"TARGET-ALIGNED {window_frames}-FRAME LIDAR EVIDENCE | {card.get('slot_id')}",
        font=title,
        fill="white",
    )
    occupied = _mapping(card.get("occupied_geometry"))
    free = _mapping(card.get("free_geometry"))
    robust = _mapping(card.get("robustness"))
    pack = _mapping(card.get("pack"))
    gate = _mapping(card.get("terminal_geometry_gate"))
    panels = (
        (24, 92, 744, 630, "OCCUPIED GEOMETRY", "#fff1f2", "#be123c", occupied),
        (792, 92, 1512, 630, "FREE-SPACE GEOMETRY", "#ecfdf5", "#047857", free),
    )
    occupied_keys = (
        "support_level", "strong_gate", "weak_gate", "failures", "candidate_return_count",
        "core_point_count", "support_frame_count", "temporal_support", "temporal_consistency",
        "supported_height_layers", "z95_m", "height_span_m", "extent_slot_x_m", "extent_slot_y_m",
        "core_overlap", "adjacent_overlap", "boundary_ratio", "linearity_risk",
    )
    free_keys = (
        "support_level", "strong_gate", "failures", "ray_frame_count", "viewpoint_count",
        "viewpoint_separation_deg", "core_ray_coverage", "observed_volume_ratio",
        "near_ground_bev_coverage", "unobserved_component_ratio", "occlusion_ratio",
        "unresolved_core_hit",
    )
    for panel_index, (x0, y0, x1, y1, label, background, color, values) in enumerate(panels):
        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=background, outline=color, width=3)
        draw.text((x0 + 20, y0 + 16), label, font=heading, fill=color)
        keys = occupied_keys if panel_index == 0 else free_keys
        rows = []
        for key in keys:
            value = values.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                value = ", ".join(str(item) for item in value) or "none"
            rows.append(f"{key}: {_fmt(value)}")
        _draw_lines(draw, rows, x=x0 + 20, y=y0 + 62, width=x1 - x0 - 40, font=body, fill="#1e293b", line_gap=5)
    footer = (
        f"PACK: {pack.get('point_count', 0)} points | {pack.get('valid_frame_count', 0)} valid frames    "
        f"ROBUSTNESS: stable={robust.get('stable')} | variants={robust.get('passing_variants')}/{robust.get('total_variants')} "
        f"| pass_ratio={_fmt(robust.get('pass_ratio'))}    "
        f"HARD GATE: Free={gate.get('free_eligible', False)} | Occupied={gate.get('occupied_eligible', False)}"
    )
    draw.text((28, 656), footer, font=heading, fill="#0f172a")
    _draw_lines(
        draw,
        ("AUDIT POLICY: numerical strengths are uncalibrated; boundary/linear-static risks veto naive vehicle interpretation; this card cannot by itself bypass the 0.50 evidence contract.",),
        x=28,
        y=706,
        width=1480,
        font=small,
        fill="#475569",
        line_gap=4,
    )
    temporary = destination.with_name(f".{destination.name}.tmp")
    image.save(temporary, format="PNG", optimize=False, compress_level=9)
    temporary.replace(destination)
    return destination


def combine_triptych_and_card(
    triptych_path: str | Path,
    card_path: str | Path,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(triptych_path) as raw_triptych, Image.open(card_path) as raw_card:
        triptych = raw_triptych.convert("RGB")
        card = raw_card.convert("RGB")
        if triptych.width != card.width:
            card = card.resize((triptych.width, round(card.height * triptych.width / card.width)), Image.Resampling.LANCZOS)
        combined = Image.new("RGB", (triptych.width, triptych.height + card.height), "white")
        combined.paste(triptych, (0, 0))
        combined.paste(card, (0, triptych.height))
    temporary = destination.with_name(f".{destination.name}.tmp")
    combined.save(temporary, format="PNG", optimize=False, compress_level=9)
    temporary.replace(destination)
    return destination


__all__ = [
    "SCHEMA_VERSION",
    "assess_terminal_geometry",
    "build_geometry_card",
    "combine_triptych_and_card",
    "render_geometry_card",
    "resolve_effective_decision",
]
