"""Deterministic map visualizations for the map-only Camera precheck.

This module renders two deliberately different artifacts:

* ``camera_gate_bev.png`` is a text-free, non-antialiased RGB raster with a
  fixed palette.  It is suitable for machine inspection and never contains
  global truth outside Part1's local LiDAR footprint.
* ``map_only_camera_precheck.png`` is a human-readable audit figure showing
  the Part1 anchor used as a conservative Camera proxy, the +/-80 degree
  policy, pose-uncertainty margins, exact rays returned by the precheck, and
  the resulting slot marks.

Rendering never calls a Camera model and never recomputes a precheck decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon as MplPolygon, Wedge
from PIL import Image, ImageDraw


MACHINE_BEV_SCHEMA_VERSION = "camera-gate-bev-raster/1.0"
ANNOTATED_PRECHECK_SCHEMA_VERSION = "map-only-camera-precheck-visualization/1.0"

# Exact RGB tuples are part of the machine-raster contract.  Do not add
# transparency or antialiasing to ``render_camera_gate_bev``.
MACHINE_BEV_COLORS: dict[str, tuple[int, int, int]] = {
    "unobserved": (0, 0, 0),
    "lidar_observed": (37, 99, 135),
    "free": (34, 197, 94),
    "occupied": (239, 68, 68),
    "unknown": (245, 158, 11),
    "local_obstacle": (100, 116, 139),
    "target": (6, 182, 212),
    "camera_proxy": (37, 99, 235),
}

_STATE_COLORS = {
    "free": ("#22c55e", "#166534"),
    "occupied": ("#ef4444", "#991b1b"),
    "unknown": ("#f59e0b", "#92400e"),
}
_TARGET_COLORS = {
    "marked_candidate": "#0891b2",
    "not_marked": "#7c3aed",
    "insufficient_information": "#64748b",
}
_RAY_COLORS = {
    "clear": "#16a34a",
    "blocked": "#dc2626",
    "uncertain": "#d97706",
    "outside_fov": "#64748b",
    "edge_unreliable": "#9333ea",
}
_TARGET_STATUSES = frozenset(_TARGET_COLORS)


def _finite_xy(value: Any, name: str, *, minimum_rows: int = 1) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != 2
        or len(array) < minimum_rows
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite [N,2] array")
    return array


def _finite_pair(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite XY pair")
    return array


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def _snapshot_geometry(
    local_map: Mapping[str, Any],
) -> tuple[float, np.ndarray, float, list[dict[str, Any]], np.ndarray | None]:
    if not isinstance(local_map, Mapping):
        raise TypeError("local_map must be a mapping")
    scale = float(local_map.get("map_units_per_meter", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("map_units_per_meter must be positive and finite")
    anchor = local_map.get("anchor_pose")
    if not isinstance(anchor, Mapping):
        raise ValueError("Part1 anchor_pose is required")
    anchor_xy = _finite_pair(anchor.get("map_xy"), "anchor_pose.map_xy")
    anchor_yaw = float(anchor.get("map_yaw_rad"))
    if not np.isfinite(anchor_yaw):
        raise ValueError("anchor_pose.map_yaw_rad must be finite")

    slots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(_sequence(local_map.get("slots", ()), "slots")):
        if not isinstance(raw, Mapping):
            raise ValueError(f"slot {index} must be a mapping")
        slot_id = str(raw.get("slot_id", "")).strip()
        if not slot_id or slot_id in seen:
            raise ValueError("local slots require unique non-empty slot_id values")
        state = str(raw.get("state", ""))
        if state not in _STATE_COLORS:
            raise ValueError(f"unsupported Part1 local state for {slot_id}: {state}")
        polygon = _finite_xy(
            raw.get("polygon_map"), f"slot {slot_id} polygon_map", minimum_rows=3
        )
        slots.append(
            {
                "slot_id": slot_id,
                "state": state,
                "polygon_map": polygon,
                "center_map": np.mean(polygon, axis=0),
            }
        )
        seen.add(slot_id)

    coverage = local_map.get("lidar_coverage", {})
    if not isinstance(coverage, Mapping):
        raise ValueError("lidar_coverage must be a mapping")
    raw_coverage = coverage.get("polygon_map")
    coverage_polygon = (
        None
        if raw_coverage is None
        else _finite_xy(raw_coverage, "lidar_coverage.polygon_map", minimum_rows=3)
    )
    return scale, anchor_xy, anchor_yaw, slots, coverage_polygon


def _normalise_report(
    report: Mapping[str, Any],
    *,
    slot_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise TypeError("precheck_report must be a mapping")
    if report.get("algorithm_executed") is not True:
        raise ValueError("precheck report must come from an executed algorithm")
    if report.get("semantic_camera_model_called") is not False:
        raise ValueError("map-only precheck must not call a Camera semantic model")
    if report.get("camera_call_requested") is not False:
        raise ValueError("map-only precheck visualization requires Camera NOT CALLED")

    proxy = report.get("proxy_pose")
    if not isinstance(proxy, Mapping):
        raise ValueError("precheck_report.proxy_pose is required")
    position = _finite_pair(
        proxy.get("position_map_xy", proxy.get("map_xy")),
        "proxy_pose.position_map_xy",
    )
    if proxy.get("yaw_rad") is not None:
        yaw_rad = float(proxy["yaw_rad"])
    else:
        yaw_rad = np.radians(float(proxy.get("yaw_deg")))
    if not np.isfinite(yaw_rad):
        raise ValueError("proxy_pose yaw must be finite")
    position_uncertainty_m = float(
        proxy.get(
            "position_uncertainty_m",
            proxy.get("position_uncertainty_radius_m", 0.0),
        )
    )
    yaw_uncertainty_deg = float(proxy.get("yaw_uncertainty_deg", 0.0))
    if (
        not np.isfinite(position_uncertainty_m)
        or position_uncertainty_m < 0.0
        or not np.isfinite(yaw_uncertainty_deg)
        or yaw_uncertainty_deg < 0.0
    ):
        raise ValueError("proxy pose uncertainties must be finite and non-negative")

    policy = report.get("policy", report.get("config"))
    if not isinstance(policy, Mapping):
        raise ValueError("precheck_report.policy is required")
    half_fov = float(
        policy.get("reliable_half_fov_deg", policy.get("candidate_half_fov_deg", 80.0))
    )
    if not np.isfinite(half_fov) or not 0.0 < half_fov <= 90.0:
        raise ValueError("map-only reliable_half_fov_deg must be within (0,90]")

    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_targets = _sequence(report.get("targets", ()), "precheck_report.targets")
    if len(raw_targets) > 2:
        raise ValueError("map-only visualization supports at most two A/B targets")
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            raise ValueError(f"precheck target {index} must be a mapping")
        slot_id = str(raw.get("target_slot_id", "")).strip()
        if not slot_id or slot_id not in slot_ids or slot_id in seen:
            raise ValueError("precheck targets must be unique local Part1 slots")
        status = str(raw.get("camera_gate_status", raw.get("status", "")))
        if status not in _TARGET_STATUSES:
            raise ValueError(f"unsupported map-only target status: {status}")
        candidate = bool(raw.get("camera_candidate", False))
        likely = bool(raw.get("camera_likely_observable", False))
        if candidate != (status == "marked_candidate") or likely != candidate:
            raise ValueError("target status/camera_candidate fields disagree")
        if raw.get("camera_call_requested") is not False:
            raise ValueError("individual target must say Camera NOT CALLED")
        rays: list[dict[str, Any]] = []
        for ray_index, ray in enumerate(_sequence(raw.get("rays", ()), "target.rays")):
            if not isinstance(ray, Mapping):
                raise ValueError(f"ray {ray_index} must be a mapping")
            start = _finite_pair(
                ray.get("origin_map_xy", ray.get("start_map_xy", position)),
                "ray origin/start_map_xy",
            )
            end = _finite_pair(
                ray.get("sample_map_xy", ray.get("end_map_xy")),
                "ray sample/end_map_xy",
            )
            status_name = str(ray.get("status", ""))
            if status_name not in _RAY_COLORS:
                raise ValueError(f"unsupported ray status: {status_name}")
            rays.append(
                {
                    "start_map_xy": start,
                    "end_map_xy": end,
                    "status": status_name,
                    "object_id": (
                        None if ray.get("object_id") is None else str(ray["object_id"])
                    ),
                }
            )
        targets.append(
            {
                "target_slot_id": slot_id,
                "display_label": str(raw.get("display_label", chr(65 + index))),
                "status": status,
                "camera_candidate": candidate,
                "camera_likely_observable": likely,
                "camera_call_requested": False,
                "target_bearing_deg": raw.get("target_bearing_deg"),
                "target_distance_m": raw.get("target_distance_m"),
                "target_angular_width_deg": raw.get("target_angular_width_deg"),
                "reliable_fov_coverage": raw.get(
                    "robust_center_fov_coverage",
                    raw.get("reliable_fov_coverage"),
                ),
                "clear_ray_ratio": raw.get("clear_ray_ratio"),
                "blocked_ray_ratio": raw.get("blocked_ray_ratio"),
                "uncertain_ray_ratio": raw.get("uncertain_ray_ratio"),
                "blocking_object_ids": tuple(
                    str(item)
                    for item in _sequence(
                        raw.get("blocking_object_ids", ()), "blocking_object_ids"
                    )
                ),
                "potential_occluder_ids": tuple(
                    str(item)
                    for item in _sequence(
                        raw.get("potential_occluder_ids", ()),
                        "potential_occluder_ids",
                    )
                ),
                "reason_codes": tuple(
                    str(item)
                    for item in _sequence(raw.get("reason_codes", ()), "reason_codes")
                ),
                "rays": tuple(rays),
            }
        )
        seen.add(slot_id)

    return {
        "mode": str(report.get("mode", "conservative_map_only_gate")),
        "proxy_pose": {
            "source": str(proxy.get("source", "part1_anchor_pose")),
            "position_map_xy": position,
            "yaw_rad": float(yaw_rad),
            "position_uncertainty_m": position_uncertainty_m,
            "yaw_uncertainty_deg": yaw_uncertainty_deg,
        },
        "policy": dict(policy),
        "reliable_half_fov_deg": half_fov,
        "targets": tuple(targets),
        "limitations": tuple(
            str(item)
            for item in _sequence(report.get("input_limitations", ()), "input_limitations")
        ),
    }


def _map_to_relative_m(points: np.ndarray, anchor: np.ndarray, scale: float) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - anchor) / scale


def _atomic_image_save(image: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.png")
    image.save(temporary, format="PNG", optimize=False)
    temporary.replace(output)


def render_camera_gate_bev(
    local_map: Mapping[str, Any],
    precheck_report: Mapping[str, Any],
    output_path: str | Path,
    *,
    resolution_m_per_pixel: float = 0.05,
) -> dict[str, Any]:
    """Write the fixed-palette, text-free Part1 input raster.

    The raster contains only Part1-local evidence, the target outlines, and
    the Part1-anchor Camera proxy.  Decision rays and labels are deliberately
    excluded so this artifact remains an algorithm-input view rather than a
    screenshot that leaks output annotations back into a later model.
    """

    scale, anchor, anchor_yaw, slots, coverage = _snapshot_geometry(local_map)
    report = _normalise_report(
        precheck_report, slot_ids={item["slot_id"] for item in slots}
    )
    resolution = float(resolution_m_per_pixel)
    if not np.isfinite(resolution) or not 0.01 <= resolution <= 1.0:
        raise ValueError("resolution_m_per_pixel must be within [0.01,1.0]")

    points: list[np.ndarray] = [anchor.reshape(1, 2)]
    points.extend(item["polygon_map"] for item in slots)
    if coverage is not None:
        points.append(coverage)
    proxy_xy = report["proxy_pose"]["position_map_xy"]
    points.append(proxy_xy.reshape(1, 2))
    bounds = np.vstack(points)
    padding_map = max(scale * 1.0, scale * resolution * 4.0)
    minimum = np.min(bounds, axis=0) - padding_map
    maximum = np.max(bounds, axis=0) + padding_map
    resolution_map = scale * resolution
    width = int(np.ceil((maximum[0] - minimum[0]) / resolution_map)) + 1
    height = int(np.ceil((maximum[1] - minimum[1]) / resolution_map)) + 1
    if width < 2 or height < 2 or width > 4096 or height > 4096:
        raise ValueError("machine BEV dimensions must be within [2,4096]")

    def pixel(point: np.ndarray) -> tuple[int, int]:
        column = int(np.floor((float(point[0]) - minimum[0]) / resolution_map))
        bottom_row = int(np.floor((float(point[1]) - minimum[1]) / resolution_map))
        return (
            min(max(column, 0), width - 1),
            min(max(height - 1 - bottom_row, 0), height - 1),
        )

    image = Image.new("RGB", (width, height), MACHINE_BEV_COLORS["unobserved"])
    draw = ImageDraw.Draw(image)
    if coverage is not None:
        draw.polygon(
            [pixel(point) for point in coverage],
            fill=MACHINE_BEV_COLORS["lidar_observed"],
        )
    for item in sorted(slots, key=lambda row: row["slot_id"]):
        draw.polygon(
            [pixel(point) for point in item["polygon_map"]],
            fill=MACHINE_BEV_COLORS[item["state"]],
        )

    # Local non-ground geometry is optional.  It is rendered only when Part1
    # actually supplies a polygon; no wall/column layer is fabricated here.
    raw_obstacles = local_map.get("local_obstacles", ())
    if raw_obstacles is None:
        raw_obstacles = ()
    for index, raw in enumerate(_sequence(raw_obstacles, "local_obstacles")):
        if not isinstance(raw, Mapping):
            raise ValueError(f"local obstacle {index} must be a mapping")
        polygon = _finite_xy(
            raw.get("polygon_map", raw.get("polygon_xy")),
            f"local obstacle {index} polygon",
            minimum_rows=3,
        )
        draw.polygon(
            [pixel(point) for point in polygon],
            fill=MACHINE_BEV_COLORS["local_obstacle"],
        )

    by_id = {item["slot_id"]: item for item in slots}
    for target in report["targets"]:
        draw.line(
            [pixel(point) for point in by_id[target["target_slot_id"]]["polygon_map"]]
            + [pixel(by_id[target["target_slot_id"]]["polygon_map"][0])],
            fill=MACHINE_BEV_COLORS["target"],
            width=3,
            joint="curve",
        )

    proxy_px = pixel(proxy_xy)
    proxy_radius = 4
    draw.ellipse(
        [
            proxy_px[0] - proxy_radius,
            proxy_px[1] - proxy_radius,
            proxy_px[0] + proxy_radius,
            proxy_px[1] + proxy_radius,
        ],
        fill=MACHINE_BEV_COLORS["camera_proxy"],
    )
    heading_length_px = max(8, int(round(1.5 / resolution)))
    yaw = report["proxy_pose"]["yaw_rad"]
    heading_end = (
        proxy_px[0] + int(round(np.cos(yaw) * heading_length_px)),
        proxy_px[1] - int(round(np.sin(yaw) * heading_length_px)),
    )
    draw.line(
        [proxy_px, heading_end],
        fill=MACHINE_BEV_COLORS["camera_proxy"],
        width=3,
    )

    output = Path(output_path)
    _atomic_image_save(image, output)
    return {
        "schema_version": MACHINE_BEV_SCHEMA_VERSION,
        "path": str(output),
        "width_px": width,
        "height_px": height,
        "resolution_m_per_pixel": resolution,
        "resolution_map_units_per_pixel": resolution_map,
        "origin_map_xy_lower_left": [float(minimum[0]), float(minimum[1])],
        "pixel_convention": "u right, v down; lower-left map origin maps to bottom row",
        "anti_aliased": False,
        "contains_text": False,
        "contains_decision_rays": False,
        "contains_decision_status": False,
        "input_only": True,
        "color_legend_rgb": {
            name: list(color) for name, color in MACHINE_BEV_COLORS.items()
        },
    }


def _safe_metric(value: Any, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}{suffix}" if np.isfinite(number) else "n/a"


def render_annotated_map_only_precheck(
    local_map: Mapping[str, Any],
    precheck_report: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Render an auditable Part1-map-only Camera candidate-marking figure."""

    scale, anchor, anchor_yaw, slots, coverage = _snapshot_geometry(local_map)
    del anchor_yaw
    report = _normalise_report(
        precheck_report, slot_ids={item["slot_id"] for item in slots}
    )
    proxy = report["proxy_pose"]
    proxy_m = _map_to_relative_m(
        proxy["position_map_xy"].reshape(1, 2), anchor, scale
    )[0]
    yaw = proxy["yaw_rad"]
    half_fov = report["reliable_half_fov_deg"]
    yaw_uncertainty = proxy["yaw_uncertainty_deg"]
    robust_half = max(0.0, half_fov - yaw_uncertainty)
    position_uncertainty = proxy["position_uncertainty_m"]

    slot_by_id = {item["slot_id"]: item for item in slots}
    slot_polygons_m = {
        item["slot_id"]: _map_to_relative_m(item["polygon_map"], anchor, scale)
        for item in slots
    }
    extent_rows: list[np.ndarray] = [np.asarray([[0.0, 0.0]]), proxy_m.reshape(1, 2)]
    extent_rows.extend(slot_polygons_m.values())
    coverage_m = None if coverage is None else _map_to_relative_m(coverage, anchor, scale)
    if coverage_m is not None:
        extent_rows.append(coverage_m)
    bounds = np.vstack(extent_rows)
    minimum = np.min(bounds, axis=0)
    maximum = np.max(bounds, axis=0)
    center = (minimum + maximum) * 0.5
    half = np.maximum((maximum - minimum) * 0.5, np.asarray([6.0, 6.0]))
    half += np.maximum(half * 0.10, 0.8)
    x_limits = (float(center[0] - half[0]), float(center[0] + half[0]))
    y_limits = (float(center[1] - half[1]), float(center[1] + half[1]))
    fov_radius = min(
        18.0,
        max(6.0, max(np.linalg.norm(row["center_map"] - proxy["position_map_xy"]) / scale for row in slots) * 1.08),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#f1f5f9",
            "savefig.facecolor": "#f8fafc",
            "savefig.dpi": 160,
            "path.simplify": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(14.5, 8.3), dpi=160)
        figure.subplots_adjust(left=0.06, right=0.69, bottom=0.10, top=0.86)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)

        if coverage_m is not None:
            axis.add_patch(
                MplPolygon(
                    coverage_m,
                    closed=True,
                    facecolor="#38bdf8",
                    edgecolor="#0284c7",
                    linewidth=1.5,
                    alpha=0.14,
                    zorder=1,
                )
            )

        yaw_deg = float(np.degrees(yaw))
        if robust_half > 0.0:
            axis.add_patch(
                Wedge(
                    proxy_m,
                    fov_radius,
                    yaw_deg - robust_half,
                    yaw_deg + robust_half,
                    facecolor="#22c55e",
                    edgecolor="#16a34a",
                    linewidth=0.9,
                    alpha=0.075,
                    zorder=1.8,
                )
            )
        for theta1, theta2 in (
            (yaw_deg - half_fov, yaw_deg - robust_half),
            (yaw_deg + robust_half, yaw_deg + half_fov),
        ):
            if theta2 > theta1:
                axis.add_patch(
                    Wedge(
                        proxy_m,
                        fov_radius,
                        theta1,
                        theta2,
                        facecolor="#f59e0b",
                        edgecolor="#d97706",
                        linewidth=0.9,
                        alpha=0.14,
                        zorder=1.9,
                    )
                )
        for offset, color, linestyle in (
            (-half_fov, "#0891b2", "--"),
            (half_fov, "#0891b2", "--"),
            (-robust_half, "#16a34a", ":"),
            (robust_half, "#16a34a", ":"),
        ):
            angle = yaw + np.radians(offset)
            axis.plot(
                [proxy_m[0], proxy_m[0] + fov_radius * np.cos(angle)],
                [proxy_m[1], proxy_m[1] + fov_radius * np.sin(angle)],
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
                alpha=0.85,
                zorder=2.1,
            )

        for item in sorted(slots, key=lambda row: row["slot_id"]):
            polygon = slot_polygons_m[item["slot_id"]]
            face, edge = _STATE_COLORS[item["state"]]
            axis.add_patch(
                MplPolygon(
                    polygon,
                    closed=True,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.1,
                    alpha=0.52,
                    zorder=3,
                )
            )
            center_m = np.mean(polygon, axis=0)
            axis.text(
                center_m[0],
                center_m[1],
                f"{item['slot_id']}\n{item['state']}",
                ha="center",
                va="center",
                fontsize=6.0,
                color="#0f172a",
                zorder=4,
            )

        blocker_ids = {
            object_id
            for target in report["targets"]
            for object_id in target["blocking_object_ids"]
        }
        potential_ids = {
            object_id
            for target in report["targets"]
            for object_id in target["potential_occluder_ids"]
        } - blocker_ids
        for ids, color, label in (
            (blocker_ids, "#b91c1c", "BLOCKER"),
            (potential_ids, "#d97706", "POTENTIAL"),
        ):
            for object_id in sorted(ids):
                if object_id not in slot_polygons_m:
                    continue
                polygon = slot_polygons_m[object_id]
                axis.add_patch(
                    MplPolygon(
                        polygon,
                        closed=True,
                        fill=False,
                        edgecolor=color,
                        linewidth=2.8,
                        zorder=5.0,
                    )
                )
                center_m = np.mean(polygon, axis=0)
                axis.annotate(
                    f"{label}: {object_id}",
                    center_m,
                    xytext=(5, -18),
                    textcoords="offset points",
                    fontsize=6.3,
                    fontweight="bold",
                    color=color,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": color,
                        "alpha": 0.92,
                    },
                    zorder=7,
                )

        rendered_ray_count = 0
        for target in report["targets"]:
            for ray in target["rays"]:
                start = _map_to_relative_m(
                    ray["start_map_xy"].reshape(1, 2), anchor, scale
                )[0]
                end = _map_to_relative_m(
                    ray["end_map_xy"].reshape(1, 2), anchor, scale
                )[0]
                color = _RAY_COLORS[ray["status"]]
                axis.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color=color,
                    linewidth=1.5 if ray["status"] == "clear" else 2.0,
                    alpha=0.78,
                    zorder=5.5,
                )
                axis.scatter(
                    [end[0]],
                    [end[1]],
                    s=12,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.9,
                    zorder=5.7,
                )
                rendered_ray_count += 1

        for index, target in enumerate(report["targets"]):
            slot_id = target["target_slot_id"]
            polygon = slot_polygons_m[slot_id]
            color = _TARGET_COLORS[target["status"]]
            axis.add_patch(
                MplPolygon(
                    polygon,
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=3.2,
                    linestyle=(0, (5, 2)),
                    zorder=6.5,
                )
            )
            label = target["display_label"] or chr(65 + index)
            status_text = {
                "marked_candidate": "MARKED",
                "not_marked": "NOT MARKED",
                "insufficient_information": "INSUFFICIENT",
            }[target["status"]]
            axis.annotate(
                f"{label} · {status_text}\nCAMERA NOT CALLED",
                np.mean(polygon, axis=0),
                xytext=(7, 14),
                textcoords="offset points",
                fontsize=7.2,
                fontweight="bold",
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.95,
                },
                zorder=8,
            )

        if position_uncertainty > 0.0:
            axis.add_patch(
                Circle(
                    proxy_m,
                    position_uncertainty,
                    facecolor="#2563eb",
                    edgecolor="#1d4ed8",
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.12,
                    zorder=7.2,
                )
            )
        axis.scatter(
            [proxy_m[0]],
            [proxy_m[1]],
            marker="D",
            s=82,
            c="#2563eb",
            edgecolors="white",
            linewidths=1.1,
            zorder=9,
        )
        heading_length = 2.0
        axis.arrow(
            proxy_m[0],
            proxy_m[1],
            np.cos(yaw) * heading_length,
            np.sin(yaw) * heading_length,
            width=0.07,
            head_width=0.42,
            head_length=0.52,
            length_includes_head=True,
            color="#1d4ed8",
            zorder=9,
        )
        axis.text(
            proxy_m[0],
            proxy_m[1] - 0.58,
            "CAMERA PROXY\n(PART1 ANCHOR POSE)",
            ha="center",
            va="top",
            fontsize=6.8,
            fontweight="bold",
            color="#1e3a8a",
            zorder=9,
        )

        axis.set_title(
            "MAP-ONLY CAMERA PRECHECK · SLOT MARKING ONLY\n"
            "PART1 LOCAL MAP → CAMERA PROXY ±80° · CAMERA NOT CALLED",
            fontsize=14,
            fontweight="bold",
            color="#0f172a",
            pad=10,
        )
        axis.set_xlabel("x relative to Part1 anchor [m]")
        axis.set_ylabel("y relative to Part1 anchor [m]")
        axis.grid(True, color="#cbd5e1", linewidth=0.45, alpha=0.55)

        note = (
            "This is a conservative map-only precheck, not a Camera result.\n"
            "The Part1 ego/anchor pose is used as a Camera proxy; pose uncertainty "
            "is tested by the algorithm.\n"
            f"Solid-green core: ±{robust_half:.0f}° robust-to-yaw range; "
            f"orange margin reaches the configured ±{half_fov:.0f}°."
        )
        axis.text(
            0.01,
            0.015,
            note,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.0,
            color="#334155",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#94a3b8",
                "alpha": 0.94,
            },
            zorder=20,
        )

        panel: list[str] = [
            "MAP-ONLY CAMERA PRECHECK",
            "algorithm executed: YES",
            "semantic Camera model: NOT CALLED",
            "Camera call requested: NO",
            f"proxy source: {proxy['source']}",
            f"proxy position uncertainty: ±{position_uncertainty:.2f} m",
            f"proxy yaw uncertainty: ±{yaw_uncertainty:.1f}°",
            f"candidate FOV policy: central ±{half_fov:.1f}°",
            f"rendered nominal rays: {rendered_ray_count}",
            "",
        ]
        for target in report["targets"]:
            status_text = {
                "marked_candidate": "MARKED",
                "not_marked": "NOT MARKED",
                "insufficient_information": "INSUFFICIENT",
            }[target["status"]]
            panel.extend(
                [
                    f"{target['display_label']}  {target['target_slot_id']}",
                    f"result: {status_text}",
                    "CAMERA: NOT CALLED",
                    (
                        "bearing / distance / width: "
                        f"{_safe_metric(target['target_bearing_deg'], '°', 1)} / "
                        f"{_safe_metric(target['target_distance_m'], ' m', 2)} / "
                        f"{_safe_metric(target['target_angular_width_deg'], '°', 1)}"
                    ),
                    "coverage / clear / blocked / uncertain: "
                    + " / ".join(
                        _safe_metric(target[name])
                        for name in (
                            "reliable_fov_coverage",
                            "clear_ray_ratio",
                            "blocked_ray_ratio",
                            "uncertain_ray_ratio",
                        )
                    ),
                    "blockers: " + (", ".join(target["blocking_object_ids"]) or "none"),
                    "potential: "
                    + (", ".join(target["potential_occluder_ids"]) or "none"),
                ]
            )
            reasons = ", ".join(target["reason_codes"]) or "none"
            wrapped = textwrap.wrap(reasons, width=49) or ["none"]
            panel.append("reasons: " + wrapped[0])
            panel.extend("  " + row for row in wrapped[1:])
            panel.append("")
        if report["limitations"]:
            panel.append("INPUT LIMITATIONS")
            for limitation in report["limitations"]:
                wrapped = textwrap.wrap(limitation, width=48)
                panel.extend("- " + row for row in wrapped)
        figure.text(
            0.715,
            0.88,
            "\n".join(panel),
            ha="left",
            va="top",
            fontsize=7.2,
            family="DejaVu Sans Mono",
            color="#0f172a",
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "white",
                "edgecolor": "#0891b2",
                "linewidth": 1.6,
                "alpha": 0.98,
            },
        )

        legend_handles = [
            Patch(facecolor="#38bdf8", edgecolor="#0284c7", alpha=0.18, label="Part1 LiDAR coverage"),
            Patch(facecolor="#22c55e", edgecolor="#166534", alpha=0.55, label="free"),
            Patch(facecolor="#ef4444", edgecolor="#991b1b", alpha=0.55, label="occupied / blocker"),
            Patch(facecolor="#f59e0b", edgecolor="#92400e", alpha=0.55, label="unknown / potential"),
            Line2D([0], [0], color="#16a34a", linewidth=1.8, label="clear ray"),
            Line2D([0], [0], color="#dc2626", linewidth=2.0, label="blocked ray"),
            Line2D([0], [0], color="#d97706", linewidth=2.0, label="uncertain ray"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#2563eb", label="Camera proxy"),
        ]
        axis.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            framealpha=0.94,
            fontsize=7.0,
        )

        temporary = output.with_name(output.name + ".tmp.png")
        figure.savefig(temporary, bbox_inches="tight", format="png")
        plt.close(figure)
        temporary.replace(output)

    return {
        "schema_version": ANNOTATED_PRECHECK_SCHEMA_VERSION,
        "path": str(output),
        "camera_proxy_drawn": True,
        "candidate_half_fov_deg": half_fov,
        "yaw_uncertainty_deg": yaw_uncertainty,
        "position_uncertainty_m": position_uncertainty,
        "rendered_ray_count": rendered_ray_count,
        "camera_called": False,
        "target_labels": {
            target["target_slot_id"]: (
                "MARKED"
                if target["status"] == "marked_candidate"
                else (
                    "NOT MARKED"
                    if target["status"] == "not_marked"
                    else "INSUFFICIENT"
                )
            )
            for target in report["targets"]
        },
    }
