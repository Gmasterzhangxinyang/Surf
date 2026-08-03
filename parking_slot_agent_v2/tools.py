"""Evidence tools for the isolated ParkingAgent v2 single-slot agent.

Camera tools use a conservative horizontal projection only as a physical
availability gate. The model is shown a target-marked ego-centric map beside an
unmodified full Camera frame; the slot polygon is never drawn on Camera pixels,
then may request bounded crops in normalized Camera coordinates.  LiDAR detail
reuses the audited v1 evidence-pack loader and triptych renderer when a pack is
available.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps, UnidentifiedImageError

from parking_slot_part2.media import (
    EvidenceMediaError,
    EvidencePackError,
    is_lidar_evidence_pack,
    load_lidar_evidence_pack,
    render_lidar_triptych,
)
from parking_slot_hybrid_3d.local_map_visualization import render_local_map

from .contracts import (
    EvidenceRecord,
    FovResult,
    FovVisibility,
    SceneSnapshot,
    SlotCase,
)
from .fov_left_camera import (
    DEFAULT_HALF_FOV_DEG,
    DEFAULT_NOMINAL_HALF_FOV_DEG,
    evaluate_horizontal_projection_fov,
    evaluate_map_bearing_fov,
)
from .lidar_geometry import (
    assess_terminal_geometry,
    build_geometry_card,
    combine_triptych_and_card,
    render_geometry_card,
    resolve_effective_decision,
)
from .lidar_explainability import render_explainable_lidar


CAMERA_RESOURCE_KEYS = (
    "camera_image_path",
    "anchor_camera_image_path",
    "rgb_image_path",
    "camera_path",
)
LIDAR_RESOURCE_KEYS = (
    "lidar_evidence_path",
    "part2_lidar_evidence_path",
    "lidar_pack_path",
    "pointcloud_artifact_path",
)
EXTENDED_LIDAR_RESOURCE_KEYS = ("extended_lidar_evidence_path",)
EXTENDED_LIDAR_DECISION_KEYS = ("extended_lidar_decision_path",)
LOCAL_MAP_JSON_RESOURCE_KEYS = ("local_map_json_path",)
FULL_SLOT_DATABASE_RESOURCE_KEYS = ("full_slot_database_path",)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_MISSING = object()


def _ui_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/home/ParkingAgent/DASP/simulation_carla/CARLA_0.9.16/Engine/Content/Slate/Fonts/DroidSansFallback.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _first(value: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        candidate = _get(value, name, _MISSING)
        if candidate is not _MISSING and candidate is not None:
            return candidate
    return default


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().lower()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _xy(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must be [x,y]")
    return _finite(value[0], f"{name}[0]"), _finite(value[1], f"{name}[1]")


def _polygon(value: Any, name: str) -> tuple[tuple[float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a coordinate sequence")
    result = tuple(_xy(point, f"{name}[{index}]") for index, point in enumerate(value))
    if len(result) < 3:
        raise ValueError(f"{name} must have at least three points")
    return result


def _slot_id(slot: Any) -> str:
    value = _first(slot, ("slot_id", "id"), "")
    result = str(value).strip()
    if not result:
        raise ValueError("slot_id is required")
    return result


def _case_id(case: Any) -> str:
    return str(_first(case, ("case_id", "task_id", "slot_id"), _slot_id(case)))


def _slot_polygon(slot: Any) -> tuple[tuple[float, float], ...]:
    raw = _first(slot, ("polygon_map", "slot_polygon_map", "polygon"), None)
    if raw is None:
        nested = _get(slot, "slot", None)
        raw = _first(nested, ("polygon_map", "slot_polygon_map", "polygon"), None)
    if raw is None:
        geometry = _get(slot, "geometry", {})
        raw = _first(geometry, ("polygon_map", "polygon"), None)
    return _polygon(raw, f"{_slot_id(slot)}.polygon_map")


def _slot_center(slot: Any) -> tuple[float, float]:
    raw = _first(slot, ("center_map", "slot_center_map", "center"), None)
    if raw is None:
        nested = _get(slot, "slot", None)
        raw = _first(nested, ("center_map", "slot_center_map", "center"), None)
    if raw is not None:
        return _xy(raw, f"{_slot_id(slot)}.center_map")
    polygon = _slot_polygon(slot)
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )


def _scene_pose(scene: Any) -> tuple[tuple[float, float], float]:
    anchor_pose_map = _get(scene, "anchor_pose_map", None)
    if (
        isinstance(anchor_pose_map, Sequence)
        and not isinstance(anchor_pose_map, (str, bytes))
        and len(anchor_pose_map) == 3
    ):
        return (
            _xy(anchor_pose_map[:2], "scene.anchor_pose_map[:2]"),
            _finite(anchor_pose_map[2], "scene.anchor_pose_map[2]"),
        )
    direct_xy = _first(scene, ("ego_map_xy", "anchor_map_xy"), None)
    direct_yaw = _first(scene, ("ego_yaw_rad", "anchor_yaw_rad"), None)
    if direct_xy is not None and direct_yaw is not None:
        return _xy(direct_xy, "scene.ego_map_xy"), _finite(
            direct_yaw, "scene.ego_yaw_rad"
        )
    pose = _first(scene, ("ego_pose", "anchor_pose", "reference_pose", "t0_pose"), None)
    if pose is None:
        raise ValueError("scene has no reference-time ego pose")
    return (
        _xy(_first(pose, ("map_xy", "xy", "position_map_xy"), None), "ego_pose.map_xy"),
        _finite(
            _first(pose, ("map_yaw_rad", "yaw_rad", "map_yaw"), None),
            "ego_pose.map_yaw_rad",
        ),
    )


def _map_units_per_meter(scene: Any) -> float:
    result = _finite(_get(scene, "map_units_per_meter", None), "map_units_per_meter")
    if result <= 0.0:
        raise ValueError("map_units_per_meter must be positive")
    return result


def _iter_scene_slots(scene: Any, case: Any) -> tuple[Any, ...]:
    raw = _first(
        scene,
        ("nearby_slots", "slots", "map_slots", "slot_map", "all_slots"),
        (),
    )
    if isinstance(raw, Mapping):
        values = list(raw.values())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        raise ValueError("scene slots must be a mapping or sequence")
    case_slot_id = _slot_id(case)
    if not any(_slot_id(slot) == case_slot_id for slot in values):
        values.append(case)
    unique: dict[str, Any] = {}
    for slot in values:
        identifier = _slot_id(slot)
        if identifier not in unique:
            unique[identifier] = slot
    return tuple(unique[key] for key in sorted(unique))


def _resources(value: Any) -> Mapping[str, Any]:
    raw = _get(value, "resources", {})
    return raw if isinstance(raw, Mapping) else {}


def _resource_path(scene: Any, case: Any, keys: Sequence[str]) -> Path | None:
    t0_frame = _get(scene, "t0_frame", None)
    for owner in (case, scene, t0_frame):
        if owner is None:
            continue
        resources = _resources(owner)
        for key in keys:
            raw = resources.get(key)
            if isinstance(raw, str) and raw.strip():
                return Path(raw)
        for key in keys:
            raw = _get(owner, key, None)
            if isinstance(raw, (str, Path)) and str(raw).strip():
                return Path(raw)
    return None


def describe_lidar_capability(case: Any) -> dict[str, Any]:
    """Describe whether ``lidar_detail`` can add evidence beyond Part1.

    A Part1 evidence pack remains reviewable for audit/rendering, but replaying
    that same pack is not a useful Part2 action and must not consume one of the
    Agent's three evidence rounds.  An extended pack is callable only when its
    identity-bound decision sidecar is present and declares genuinely additional
    causal frames.
    """

    part1_path = _resource_path(None, case, LIDAR_RESOURCE_KEYS)
    extended_path = _resource_path(None, case, EXTENDED_LIDAR_RESOURCE_KEYS)
    decision_path = _resource_path(None, case, EXTENDED_LIDAR_DECISION_KEYS)
    result: dict[str, Any] = {
        "available": False,
        "review_available": bool(part1_path and part1_path.is_file()),
        "evidence_source": "none",
        "selected_frame_count": 0,
        "valid_frame_count": None,
        "is_incremental_over_part1": False,
        "robustness_available": False,
        "expected_information_gain": "none",
        "blocking_reason": "lidar_evidence_pack_missing",
    }
    if extended_path is None:
        if part1_path is not None and part1_path.is_file():
            result.update(
                {
                    "evidence_source": "part1_15frame",
                    "selected_frame_count": len(
                        tuple(_get(case, "evidence_frame_ids", ()))
                    ),
                    "blocking_reason": "same_as_part1",
                }
            )
        return result
    result.update(
        {
            "evidence_source": "part2_extended_causal_lidar",
            "review_available": bool(extended_path.is_file()),
            "blocking_reason": "extended_lidar_contract_incomplete",
        }
    )
    if not extended_path.is_file() or decision_path is None or not decision_path.is_file():
        return result
    try:
        payload = json.loads(decision_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return result
        decision = payload.get("decision")
        if not isinstance(decision, Mapping):
            return result
        if str(payload.get("slot_id", decision.get("slot_id", ""))) != _slot_id(case):
            result["blocking_reason"] = "extended_lidar_slot_identity_mismatch"
            return result
        selected_frames = tuple(int(value) for value in payload.get("selected_frames", ()))
        if not selected_frames:
            return result
        part1_frames = {
            int(value) for value in tuple(_get(case, "evidence_frame_ids", ()))
        }
        incremental = any(value not in part1_frames for value in selected_frames)
        stability = decision.get("stability", {})
        robustness_available = (
            isinstance(stability, Mapping)
            and int(stability.get("total_variants", 0) or 0) > 0
        )
        valid_count: int | None = None
        # Reading these two small arrays is lazy; the large points array in the
        # NPZ is not decompressed. Failure leaves the count unknown and the
        # actual tool performs the authoritative evidence-pack validation.
        try:
            import numpy as np

            with np.load(extended_path, allow_pickle=False) as archive:
                pack_selected = tuple(
                    int(value) for value in archive["selected_frames"].tolist()
                )
                valid_count = int(len(archive["valid_frames"]))
            if pack_selected != selected_frames:
                result["blocking_reason"] = "extended_lidar_frame_contract_mismatch"
                return result
        except (OSError, KeyError, TypeError, ValueError):
            result["blocking_reason"] = "extended_lidar_pack_unreadable"
            return result
        result.update(
            {
                "available": incremental,
                "selected_frame_count": len(selected_frames),
                "valid_frame_count": valid_count,
                "is_incremental_over_part1": incremental,
                "robustness_available": robustness_available,
                "expected_information_gain": "potential" if incremental else "none",
                "blocking_reason": None if incremental else "same_as_part1",
            }
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return result


def _safe_component(value: str) -> str:
    result = _SAFE_NAME.sub("_", str(value)).strip("._")
    return result[:96] or "slot"


def _canonical_id(payload: Mapping[str, Any]) -> str:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "ev_" + hashlib.sha256(content).hexdigest()


def _record(
    *,
    case: Any,
    tool_name: str,
    round_index: int,
    status: str,
    summary: str,
    artifact_paths: Sequence[Path] = (),
    metadata: Mapping[str, Any] | None = None,
    modality: str = "unknown",
    reason_codes: Sequence[str] = (),
    resource_keys: Sequence[str] = (),
) -> EvidenceRecord:
    data = dict(metadata or {})
    normalized_reason_codes = [str(item) for item in reason_codes]
    if not normalized_reason_codes:
        raw_reason_codes = data.get("reason_codes", [])
        if isinstance(raw_reason_codes, Sequence) and not isinstance(
            raw_reason_codes, (str, bytes)
        ):
            normalized_reason_codes = [str(item) for item in raw_reason_codes]
    evidence_id = _canonical_id(
        {
            "case_id": _case_id(case),
            "slot_id": _slot_id(case),
            "tool_name": tool_name,
            "round_index": int(round_index),
            "status": status,
            "metadata": data,
        }
    )
    return EvidenceRecord(
        evidence_id=evidence_id,
        tool_name=tool_name,
        round_index=int(round_index),
        status=status,
        artifact_paths=[str(path) for path in artifact_paths],
        summary=summary,
        metadata=data,
        modality=modality,
        reason_codes=normalized_reason_codes,
        resource_keys=[str(item) for item in resource_keys],
    )


def _register_artifact(case: Any, base_key: str, path: Path) -> str:
    """Attach a generated artifact to the mutable SlotCase before evidence."""

    resources = _get(case, "resources", None)
    if not isinstance(resources, dict):
        raise ValueError("SlotCase.resources must be a mutable dictionary")
    path_text = str(path)
    key = _safe_component(base_key)
    existing = resources.get(key)
    if existing == path_text:
        return key
    if existing is not None:
        suffix = hashlib.sha256(path_text.encode("utf-8")).hexdigest()[:10]
        key = f"{key}_{suffix}"
        counter = 2
        while key in resources and resources[key] != path_text:
            key = f"{_safe_component(base_key)}_{suffix}_{counter}"
            counter += 1
    resources[key] = path_text
    return key


def _atomic_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG", optimize=False, compress_level=9)
    temporary.replace(path)


def _atomic_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _read_json_mapping(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{name} is not a concrete file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} cannot be read as JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _open_camera_image(path: Path, *, max_bytes: int, max_pixels: int) -> Image.Image:
    try:
        if not path.is_file():
            raise ValueError("camera image is not a concrete file")
        if path.stat().st_size > max_bytes:
            raise ValueError("camera image exceeds the byte limit")
        with Image.open(path) as opened:
            width, height = opened.size
            if (
                width < 2
                or height < 2
                or width * height > max_pixels
                or getattr(opened, "n_frames", 1) != 1
            ):
                raise ValueError("camera image dimensions are outside safe limits")
            opened.load()
            return ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("camera image cannot be decoded safely") from exc


def _slot_state(slot: Any) -> str:
    return _enum_value(
        _first(slot, ("current_state", "part1_state", "state"), "not_evaluated")
    )


def _adjacent_ids(case: Any) -> frozenset[str]:
    raw = _first(case, ("adjacent_slot_ids", "adjacent_slots"), ())
    if not raw:
        raw = _first(_get(case, "slot", None), ("adjacent_slot_ids", "adjacent_slots"), ())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return frozenset(str(value) for value in raw)
    return frozenset()


def _ego_relative_m(
    point_map: tuple[float, float],
    ego_map: tuple[float, float],
    ego_yaw_rad: float,
    map_units_per_meter: float,
) -> tuple[float, float]:
    dx = (point_map[0] - ego_map[0]) / map_units_per_meter
    dy = (point_map[1] - ego_map[1]) / map_units_per_meter
    cosine = math.cos(ego_yaw_rad)
    sine = math.sin(ego_yaw_rad)
    # x is forward and y is left in the rendered ego frame.
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def _render_semantic_map(
    scene: Any,
    case: Any,
    *,
    radius_m: float,
    half_fov_deg: float,
    size: int,
) -> Image.Image:
    ego_xy, ego_yaw = _scene_pose(scene)
    scale = _map_units_per_meter(scene)
    slots = _iter_scene_slots(scene, case)
    target_id = _slot_id(case)
    adjacent = _adjacent_ids(case)
    margin = 58
    pixels_per_metre = (size - 2 * margin) / (2.0 * radius_m)

    image = Image.new("RGB", (size, size), (247, 249, 252))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(image)
    font = _ui_font(15)
    label_font = _ui_font(18, bold=True)
    header_font = _ui_font(17, bold=True)
    center_px = (size / 2.0, size / 2.0)

    def pixel(relative: tuple[float, float]) -> tuple[int, int]:
        forward, left = relative
        return (
            int(round(center_px[0] - left * pixels_per_metre)),
            int(round(center_px[1] - forward * pixels_per_metre)),
        )

    # Range rings and ego-centric axes make map/image spatial reasoning easier.
    for distance in range(5, int(radius_m) + 1, 5):
        radius_px = int(round(distance * pixels_per_metre))
        draw.ellipse(
            (
                center_px[0] - radius_px,
                center_px[1] - radius_px,
                center_px[0] + radius_px,
                center_px[1] + radius_px,
            ),
            outline=(215, 222, 232),
            width=1,
        )
        draw.text(
            (center_px[0] + 4, center_px[1] - radius_px + 2),
            f"{distance}m",
            fill=(100, 116, 139),
            font=font,
        )
    draw.line((margin, center_px[1], size - margin, center_px[1]), fill=(225, 230, 238))
    draw.line((center_px[0], margin, center_px[0], size - margin), fill=(225, 230, 238))

    wedge_points = [pixel((0.0, 0.0))]
    for angle_deg in range(-int(round(half_fov_deg)), int(round(half_fov_deg)) + 1, 4):
        angle = math.radians(angle_deg)
        wedge_points.append(pixel((radius_m * math.cos(angle), radius_m * math.sin(angle))))
    wedge_points.append(pixel((0.0, 0.0)))
    draw_overlay.polygon(wedge_points, fill=(37, 99, 235, 24))
    draw_overlay.line(wedge_points, fill=(37, 99, 235, 120), width=2)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    state_fill = {
        "free": (194, 240, 205),
        "occupied": (254, 202, 202),
        "unknown": (254, 230, 171),
        "not_evaluated": (226, 232, 240),
        "unobserved": (226, 232, 240),
    }
    for slot in slots:
        identifier = _slot_id(slot)
        polygon_map = _slot_polygon(slot)
        relative = tuple(
            _ego_relative_m(point, ego_xy, ego_yaw, scale) for point in polygon_map
        )
        slot_center = _slot_center(slot)
        center_relative = _ego_relative_m(slot_center, ego_xy, ego_yaw, scale)
        if math.hypot(*center_relative) > radius_m + 4.0:
            continue
        polygon_px = [pixel(point) for point in relative]
        state = _slot_state(slot)
        fill = state_fill.get(state, state_fill["not_evaluated"])
        outline = (100, 116, 139)
        width = 2
        if identifier in adjacent:
            outline = (234, 179, 8)
            width = 4
        if identifier == target_id:
            fill = (238, 153, 220)
            outline = (190, 24, 126)
            width = 6
        draw.polygon(polygon_px, fill=fill)
        draw.line([*polygon_px, polygon_px[0]], fill=outline, width=width, joint="curve")
        if identifier == target_id or identifier in adjacent:
            label_position = pixel(center_relative)
            draw.text(
                (label_position[0] + 3, label_position[1] + 3),
                identifier,
                fill=(91, 33, 84) if identifier == target_id else (113, 75, 0),
                font=label_font,
                stroke_width=1,
                stroke_fill=(255, 255, 255),
            )

    # Draw the Agent's map-space target ray. This is not an image projection;
    # it makes the left/right and depth relation readable for semantic matching.
    target_relative = _ego_relative_m(_slot_center(case), ego_xy, ego_yaw, scale)
    target_px = pixel(target_relative)
    ego = pixel((0.0, 0.0))
    draw.line((ego, target_px), fill=(190, 24, 126), width=4)
    draw.ellipse((target_px[0] - 8, target_px[1] - 8, target_px[0] + 8, target_px[1] + 8), fill=(190, 24, 126))

    # Ego always points upward because the map is ego-centric.
    arrow_tip = pixel((2.8, 0.0))
    arrow_left = pixel((-0.8, 1.0))
    arrow_right = pixel((-0.8, -1.0))
    draw.polygon((arrow_tip, arrow_left, arrow_right), fill=(15, 118, 110), outline=(4, 78, 72))
    draw.text((ego[0] + 8, ego[1] + 8), "EGO", fill=(4, 78, 72), font=label_font)
    draw.rectangle((0, 0, size, 48), fill=(15, 23, 42))
    fov = _get(case, "fov", None)
    details = _get(fov, "details", {})
    bearing = float(_get(details, "target_bearing_deg", 0.0))
    distance = _get(details, "target_distance_m", None)
    distance_label = "?m" if distance is None else f"{float(distance):.1f}m"
    draw.text(
        (12, 13),
        f"TARGET {target_id} | bearing {bearing:+.1f}° (+LEFT / -RIGHT) | {distance_label} | MAGENTA",
        fill=(255, 255, 255),
        font=header_font,
    )
    draw.text((12, size - 30), "CAMERA LEFT", fill=(37, 99, 235), font=header_font)
    right_label = "CAMERA RIGHT"
    right_width = draw.textlength(right_label, font=header_font)
    draw.text((size - right_width - 12, size - 30), right_label, fill=(37, 99, 235), font=header_font)
    forward_label = "CAMERA FORWARD ↑"
    forward_width = draw.textlength(forward_label, font=header_font)
    draw.text(((size - forward_width) / 2, 55), forward_label, fill=(4, 78, 72), font=header_font)
    return image


def _fit_full_camera(image: Image.Image, *, size: int) -> Image.Image:
    tile = Image.new("RGB", (size, size), (15, 23, 42))
    title_height = 30
    available_height = size - title_height
    scale = min(size / image.width, available_height / image.height)
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    left = (size - width) // 2
    top = title_height + (available_height - height) // 2
    # No marks, polygons, or text are drawn onto these Camera pixels.
    tile.paste(resized, (left, top))
    draw = ImageDraw.Draw(tile)
    draw.text(
        (8, 9),
        "RAW FULL CAMERA | NO SLOT PROJECTION",
        fill=(255, 255, 255),
        font=ImageFont.load_default(),
    )
    return tile


def _fit_map_artifact(image: Image.Image, *, size: int) -> Image.Image:
    tile = Image.new("RGB", (size, size), (248, 250, 252))
    contained = ImageOps.contain(image.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    tile.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    return tile


def _causal_camera_frames(scene: SceneSnapshot, limit: int) -> tuple[Any, ...]:
    anchor_timestamp = _finite(
        _get(scene, "anchor_timestamp", None), "scene.anchor_timestamp"
    )
    candidates: list[Any] = []
    seen_camera_ids: set[int] = set()
    for frame in _get(scene, "frames", ()):
        if not bool(_get(frame, "camera_match_valid", False)):
            continue
        camera_timestamp = _get(frame, "camera_timestamp", None)
        camera_frame_id = _get(frame, "camera_frame_id", None)
        camera_path = _get(frame, "camera_image_path", None)
        if camera_timestamp is None or camera_frame_id is None or not camera_path:
            continue
        if _finite(camera_timestamp, "camera_timestamp") > anchor_timestamp + 1e-9:
            continue
        normalized_id = int(camera_frame_id)
        if normalized_id in seen_camera_ids:
            continue
        seen_camera_ids.add(normalized_id)
        candidates.append(frame)
    candidates.sort(key=lambda item: float(_get(item, "camera_timestamp", 0.0)))
    return tuple(candidates[-limit:])


def _render_camera_sequence(
    frames: Sequence[tuple[Any, Image.Image, FovResult]],
    *,
    anchor_timestamp: float,
    tile_width: int = 512,
) -> Image.Image:
    columns = min(3, len(frames))
    rows = int(math.ceil(len(frames) / columns))
    image_height = int(round(tile_width * 9.0 / 16.0))
    header_height = 42
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (image_height + header_height)),
        (15, 23, 42),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (frame, image, fov) in enumerate(frames):
        column = index % columns
        row = index // columns
        left = column * tile_width
        top = row * (image_height + header_height)
        fitted = ImageOps.fit(
            image.convert("RGB"),
            (tile_width, image_height),
            method=Image.Resampling.LANCZOS,
        )
        sheet.paste(fitted, (left, top + header_height))
        camera_timestamp = float(_get(frame, "camera_timestamp", anchor_timestamp))
        delta_ms = (camera_timestamp - anchor_timestamp) * 1000.0
        label = (
            f"LiDAR {_get(frame, 'frame_id')} | Camera {_get(frame, 'camera_frame_id')} "
            f"| t-t0 {delta_ms:+.1f} ms | bearing "
            f"{float(fov.details.get('target_bearing_deg', 0.0)):+.1f} deg "
            f"| FOV {fov.visibility.value}"
        )
        draw.text((left + 8, top + 14), label, fill=(255, 255, 255), font=font)
    return sheet


def _fraction_inside(bearings_deg: Sequence[float], half_fov_deg: float) -> float:
    if not bearings_deg:
        return 0.0
    return sum(abs(float(value)) <= half_fov_deg for value in bearings_deg) / len(
        bearings_deg
    )


def _camera_fov_overlay(
    scene: SceneSnapshot,
    case: SlotCase,
    result: FovResult,
    *,
    reliable_half_fov_deg: float,
    nominal_half_fov_deg: float,
    radius_m: float,
) -> dict[str, Any]:
    """Translate the coarse FOV result into the shared Part1 map renderer schema."""

    ego_xy, ego_yaw = _scene_pose(scene)
    polygon = _slot_polygon(case)
    center = _slot_center(case)
    samples = (center, *polygon)
    raw_bearings = result.details.get("sample_bearings_deg", ())
    bearings = (
        tuple(float(value) for value in raw_bearings)
        if isinstance(raw_bearings, Sequence)
        and not isinstance(raw_bearings, (str, bytes))
        and len(raw_bearings) == len(samples)
        else ()
    )
    edge_start = reliable_half_fov_deg + (
        nominal_half_fov_deg - reliable_half_fov_deg
    ) * 0.5
    if result.visibility is FovVisibility.VISIBLE:
        decision = "use_camera"
    elif result.visibility is FovVisibility.NOT_VISIBLE:
        decision = "do_not_use_camera"
    else:
        decision = "insufficient_information"

    robust_fraction = float(result.details.get("robust_inside_fraction", 0.0))
    nominal_fraction = _fraction_inside(bearings, nominal_half_fov_deg)
    debug_trace: list[dict[str, Any]] = []
    for index, point in enumerate(samples):
        bearing = None if not bearings else bearings[index]
        if bearing is None:
            status = "uncertain"
        elif abs(bearing) > nominal_half_fov_deg:
            status = "outside_fov"
        elif abs(bearing) > reliable_half_fov_deg:
            status = "edge_unreliable"
        else:
            # A map-bearing check has no image/occlusion evidence, so an
            # in-cone ray must remain uncertain rather than being called clear.
            status = "uncertain"
        debug_trace.append(
            {
                "sample_name": "center" if index == 0 else f"corner_{index}",
                "sample_map_xy": [float(point[0]), float(point[1])],
                "bearing_deg": bearing,
                "status": status,
            }
        )

    raw_reasons = result.details.get("reason_codes", ())
    reason_codes = (
        [str(value) for value in raw_reasons]
        if isinstance(raw_reasons, Sequence)
        and not isinstance(raw_reasons, (str, bytes))
        else []
    )
    limitations = result.details.get("proxy_limitations", ())
    input_limitations = (
        [str(value) for value in limitations]
        if isinstance(limitations, Sequence)
        and not isinstance(limitations, (str, bytes))
        else []
    )
    if result.details.get("projection_role") == "camera_tool_feasibility_only":
        input_limitations.extend(
            [
                "horizontal_projection_is_tool_gate_not_slot_localization",
                "slot_polygon_not_drawn_on_camera_pixels",
                "no_image_occlusion_test_in_projection_gate",
            ]
        )
    else:
        input_limitations.extend(
            [
                "camera_pose_is_t0_ego_pose_proxy",
                "nominal_horizontal_fov_derived_from_zed_intrinsics_about_101_deg",
                "no_image_occlusion_test_in_coarse_fov_gate",
            ]
        )
    return {
        "algorithm": (
            "parking_slot_agent_v2.horizontal_projection_camera_tool_gate"
            if result.details.get("projection_role") == "camera_tool_feasibility_only"
            else "parking_slot_agent_v2.coarse_map_bearing_fov"
        ),
        "algorithm_executed": True,
        "semantic_camera_model_called": False,
        "camera_pose_map_xyyaw": [ego_xy[0], ego_xy[1], ego_yaw],
        "target_center_pixel_u": result.details.get("target_center_pixel_u"),
        "targets": [
            {
                "target_slot_id": _slot_id(case),
                "display_label": "TARGET",
                "assessment": {
                    "target_slot_id": _slot_id(case),
                    "decision": decision,
                    "camera_usable": decision == "use_camera",
                    "nominal_fov_coverage": nominal_fraction,
                    "reliable_fov_coverage": max(0.0, min(robust_fraction, 1.0)),
                    "edge_quality_score": max(0.0, min(robust_fraction, 1.0)),
                    "clear_ray_ratio": 0.0,
                    "blocked_ray_ratio": 0.0,
                    "uncertain_ray_ratio": 1.0,
                    "blocking_object_ids": [],
                    "potential_occluder_ids": [],
                    "reason_codes": reason_codes,
                    "target_bearing_deg": result.details.get("target_bearing_deg"),
                    "target_distance_m": result.details.get("target_distance_m"),
                    "fov_visibility": result.visibility.value,
                    "confidence": result.confidence,
                },
                "debug_trace": debug_trace,
            }
        ],
        "fov_zones_deg": {
            "nominal_half": nominal_half_fov_deg,
            "reliable_half": reliable_half_fov_deg,
            "edge_unreliable_start": edge_start,
        },
        "fov_radius_m": radius_m,
        "static_obstacles": [],
        "occlusion_objects": [],
        "input_limitations": input_limitations,
    }


class V2ToolSuite:
    """Bounded per-slot tools used by the v2 orchestrator."""

    def __init__(
        self,
        *,
        half_fov_deg: float = DEFAULT_HALF_FOV_DEG,
        nominal_half_fov_deg: float = DEFAULT_NOMINAL_HALF_FOV_DEG,
        yaw_uncertainty_deg: float = 15.0,
        map_radius_m: float = 30.0,
        panel_size_px: int = 768,
        max_camera_bytes: int = 64 * 1024 * 1024,
        max_camera_pixels: int = 40_000_000,
        max_crop_side_px: int = 1800,
        max_sequence_frames: int = 5,
    ) -> None:
        self.half_fov_deg = _finite(half_fov_deg, "half_fov_deg")
        self.nominal_half_fov_deg = _finite(
            nominal_half_fov_deg, "nominal_half_fov_deg"
        )
        self.yaw_uncertainty_deg = _finite(
            yaw_uncertainty_deg, "yaw_uncertainty_deg"
        )
        self.map_radius_m = _finite(map_radius_m, "map_radius_m")
        self.panel_size_px = int(panel_size_px)
        self.max_camera_bytes = int(max_camera_bytes)
        self.max_camera_pixels = int(max_camera_pixels)
        self.max_crop_side_px = int(max_crop_side_px)
        self.max_sequence_frames = int(max_sequence_frames)
        if not 0.0 < self.half_fov_deg < 180.0:
            raise ValueError("half_fov_deg must be within (0,180)")
        if not self.half_fov_deg < self.nominal_half_fov_deg <= 180.0:
            raise ValueError(
                "nominal_half_fov_deg must be greater than half_fov_deg and <= 180"
            )
        if not 0.0 <= self.yaw_uncertainty_deg < self.half_fov_deg:
            raise ValueError("yaw_uncertainty_deg must be below half_fov_deg")
        if self.map_radius_m <= 0.0 or self.panel_size_px < 256:
            raise ValueError("map radius and panel size must be positive")
        if min(self.max_camera_bytes, self.max_camera_pixels, self.max_crop_side_px) <= 0:
            raise ValueError("Camera safety limits must be positive")
        if not 3 <= self.max_sequence_frames <= 8:
            raise ValueError("max_sequence_frames must be within [3,8]")
        self._fov_by_case: dict[str, FovResult] = {}

    def check_fov(
        self,
        scene: SceneSnapshot,
        case: SlotCase,
        output_dir: str | Path | None = None,
    ) -> EvidenceRecord:
        """Mandatory horizontal projection gate; it never reads Camera pixels."""

        t0_frame = _get(scene, "t0_frame", None)
        camera_frame_id = _get(t0_frame, "camera_frame_id", None)
        try:
            ego_xy, ego_yaw = _scene_pose(scene)
            result = evaluate_horizontal_projection_fov(
                ego_map_xy=ego_xy,
                ego_yaw_rad=ego_yaw,
                target_polygon_map=_slot_polygon(case),
                target_center_map=_slot_center(case),
                map_units_per_meter=_map_units_per_meter(scene),
                camera_frame_id=camera_frame_id,
            )
        except (TypeError, ValueError) as exc:
            result = FovResult(
                visibility=FovVisibility.UNCERTAIN,
                confidence=0.0,
                reason="coarse_map_bearing_input_unavailable",
                camera_frame_id=camera_frame_id,
                candidate_regions=[],
                details={
                    "reason_codes": ["invalid_or_missing_map_fov_input"],
                    "error_type": type(exc).__name__,
                    "proxy_limitations": [
                        "map_bearing_only_no_pixel_projection",
                    ],
                },
            )
            self._fov_by_case[_case_id(case)] = result
            case.update_fov(result)
            return _record(
                case=case,
                tool_name="check_fov",
                round_index=0,
                status="unavailable",
                summary="Coarse FOV could not be evaluated from the t0 map pose.",
                metadata={
                    "visibility": "uncertain",
                    "reason_codes": ["invalid_or_missing_map_fov_input"],
                    "error_type": type(exc).__name__,
                    "proxy_limitations": [
                        "map_bearing_only_no_pixel_projection",
                    ],
                },
                modality="map_geometry",
            )
        self._fov_by_case[_case_id(case)] = result
        case.update_fov(result)
        artifact_paths: tuple[Path, ...] = ()
        resource_keys: list[str] = []
        metadata = result.to_dict()
        if output_dir is not None:
            try:
                destination, resource_key = self.render_fov_occupancy_map(
                    scene,
                    case,
                    output_dir,
                    result=result,
                )
                artifact_paths = (destination,)
                resource_keys.append(resource_key)
                metadata["occupancy_map_fov_overlay_path"] = str(destination)
                metadata["fov_zones_deg"] = {
                    "nominal_half": self.nominal_half_fov_deg,
                    "reliable_half": self.half_fov_deg,
                    "edge_unreliable_start": self.half_fov_deg
                    + (self.nominal_half_fov_deg - self.half_fov_deg) * 0.5,
                }
            except (OSError, TypeError, ValueError) as exc:
                # The geometry decision remains valid even if its audit image
                # cannot be produced. Keep that failure explicit in metadata.
                metadata["occupancy_map_fov_overlay_error"] = type(exc).__name__
        return _record(
            case=case,
            tool_name="check_fov",
            round_index=0,
            status="ok",
            summary=(
                f"Horizontal Camera projection gate is {result.visibility.value}; "
                "it controls tool availability but is not occupancy evidence."
            ),
            artifact_paths=artifact_paths,
            metadata=metadata,
            modality="map_geometry",
            reason_codes=result.details.get("reason_codes", []),
            resource_keys=resource_keys,
        )

    def render_fov_occupancy_map(
        self,
        scene: SceneSnapshot,
        case: SlotCase,
        output_dir: str | Path,
        *,
        result: FovResult | None = None,
    ) -> tuple[Path, str]:
        """Draw the Camera FOV zones directly on the original Part1 local map."""

        local_map_path = _resource_path(scene, case, LOCAL_MAP_JSON_RESOURCE_KEYS)
        full_slot_db_path = _resource_path(
            scene, case, FULL_SLOT_DATABASE_RESOURCE_KEYS
        )
        if local_map_path is None:
            raise ValueError("local_map_json_path is unavailable")
        snapshot = _read_json_mapping(local_map_path, "local_map_json_path")
        all_slots: Any = None
        if full_slot_db_path is not None:
            all_slots = _read_json_mapping(
                full_slot_db_path, "full_slot_database_path"
            )
        active_result = result or self._fov_by_case.get(_case_id(case))
        if active_result is None or not active_result.checked:
            raise ValueError("a checked FOV result is required before rendering")
        overlay = _camera_fov_overlay(
            scene,
            case,
            active_result,
            reliable_half_fov_deg=self.half_fov_deg,
            nominal_half_fov_deg=self.nominal_half_fov_deg,
            radius_m=min(
                self.map_radius_m,
                float(_get(scene, "radius_m", self.map_radius_m)),
            ),
        )
        destination = (
            Path(output_dir).resolve(strict=False)
            / "fov"
            / f"{_safe_component(_slot_id(case))}_fov_occupancy_map.png"
        )
        render_local_map(
            snapshot,
            all_slots,
            destination,
            camera_overlay=overlay,
        )
        resource_key = _register_artifact(
            case,
            "part2_fov_occupancy_map",
            destination,
        )
        return destination, resource_key

    def _camera_allowed(self, case: SlotCase) -> bool:
        result = self._fov_by_case.get(_case_id(case))
        if result is None:
            candidate = _get(case, "fov", None)
            result = candidate if isinstance(candidate, FovResult) and candidate.checked else None
        return result is not None and result.visibility in {
            FovVisibility.VISIBLE,
            FovVisibility.PARTIALLY_VISIBLE,
        }

    def inspect_camera_context(
        self,
        scene: SceneSnapshot,
        case: SlotCase,
        output_dir: str | Path,
        round_index: int,
    ) -> EvidenceRecord:
        """Render target map and unmodified full Camera frame side by side."""

        if not self._camera_allowed(case):
            return _record(
                case=case,
                tool_name="camera_context",
                round_index=round_index,
                status="unavailable",
                summary="Camera context is forbidden because the horizontal projection gate is outside.",
                metadata={"reason_codes": ["target_outside_camera_projection"]},
                modality="camera",
            )
        source = _resource_path(scene, case, CAMERA_RESOURCE_KEYS)
        if source is None:
            return _record(
                case=case,
                tool_name="camera_context",
                round_index=round_index,
                status="unavailable",
                summary="No raw Camera resource is attached to this t0 scene.",
                metadata={"reason_codes": ["camera_resource_missing"]},
                modality="camera",
            )
        try:
            camera = _open_camera_image(
                source,
                max_bytes=self.max_camera_bytes,
                max_pixels=self.max_camera_pixels,
            )
            # The model always receives a dedicated, uncluttered ego-centric
            # semantic map. The original Part1/FOV plot remains an audit asset,
            # but its overlapping labels previously obscured the target row and
            # made active map-to-Camera reasoning unnecessarily difficult.
            map_panel = _render_semantic_map(
                scene,
                case,
                radius_m=self.map_radius_m,
                half_fov_deg=self.half_fov_deg,
                size=self.panel_size_px,
            )
            map_source = "dedicated_ego_semantic_map"
            map_destination = (
                Path(output_dir).resolve(strict=False)
                / "camera_context"
                / f"{_safe_component(_slot_id(case))}_round_{int(round_index):02d}_map.png"
            )
            _atomic_png(map_panel, map_destination)
            camera_panel = _fit_full_camera(camera, size=self.panel_size_px)
            combined = Image.new(
                "RGB",
                (self.panel_size_px * 2, self.panel_size_px),
                (15, 23, 42),
            )
            combined.paste(map_panel, (0, 0))
            combined.paste(camera_panel, (self.panel_size_px, 0))
            destination = (
                Path(output_dir).resolve(strict=False)
                / "camera_context"
                / f"{_safe_component(_slot_id(case))}_round_{int(round_index):02d}.png"
            )
            _atomic_png(combined, destination)
            resource_key = _register_artifact(
                case,
                f"part2_camera_context_round_{int(round_index)}",
                destination,
            )
            map_resource_key = _register_artifact(
                case,
                f"part2_camera_context_map_round_{int(round_index)}",
                map_destination,
            )
        except (OSError, TypeError, ValueError) as exc:
            return _record(
                case=case,
                tool_name="camera_context",
                round_index=round_index,
                status="failed",
                summary="Camera context could not be rendered safely.",
                metadata={
                    "reason_codes": ["camera_context_render_failed"],
                    "error_type": type(exc).__name__,
                },
                modality="camera",
            )
        return _record(
            case=case,
            tool_name="camera_context",
            round_index=round_index,
            status="ok",
            summary=(
                "Model evidence is ordered as the target-marked ego-centric map "
                "then the complete raw Camera frame; no slot projection is used."
            ),
            artifact_paths=(map_destination, source, destination),
            metadata={
                "target_slot_id": _slot_id(case),
                "target_bearing_deg": case.fov.details.get("target_bearing_deg"),
                "bearing_sign_convention": "positive_is_camera_left",
                "target_distance_m": case.fov.details.get("target_distance_m"),
                "semantic_map_convention": {
                    "forward": "up",
                    "map_left": "camera_left",
                    "map_right": "camera_right",
                },
                "map_radius_m": self.map_radius_m,
                "camera_source_size_px": [camera.width, camera.height],
                "composite_size_px": [combined.width, combined.height],
                "camera_pixels_modified": False,
                "map_to_camera_projection_used": False,
                "map_panel_source": map_source,
                "bbox_norm_coordinate_space": "raw_full_camera",
                "model_image_paths": [str(map_destination), str(source)],
                "audit_composite_path": str(destination),
            },
            modality="camera",
            resource_keys=[map_resource_key, resource_key],
        )

    # The model tool name is concise; retain the explicit method above for
    # callers that prefer verb-led APIs.
    camera_context = inspect_camera_context

    def inspect_camera_sequence(
        self,
        scene: SceneSnapshot,
        case: SlotCase,
        output_dir: str | Path,
        round_index: int,
    ) -> EvidenceRecord:
        """Render up to five strictly pre-t0 Camera frames as one contact sheet."""

        if not self._camera_allowed(case):
            return _record(
                case=case,
                tool_name="camera_sequence",
                round_index=round_index,
                status="unavailable",
                summary="Camera sequence is forbidden because the horizontal projection gate is outside.",
                metadata={"reason_codes": ["target_outside_camera_projection"]},
                modality="camera",
            )
        try:
            frames = _causal_camera_frames(scene, self.max_sequence_frames)
            if len(frames) < 3:
                return _record(
                    case=case,
                    tool_name="camera_sequence",
                    round_index=round_index,
                    status="unavailable",
                    summary="Fewer than three strictly causal Camera frames are available.",
                    metadata={"reason_codes": ["causal_camera_sequence_too_short"]},
                    modality="camera",
                )
            loaded: list[tuple[Any, Image.Image, FovResult]] = []
            source_rows: list[dict[str, Any]] = []
            total_bytes = 0
            for frame in frames:
                source = Path(str(_get(frame, "camera_image_path", "")))
                total_bytes += source.stat().st_size
                if total_bytes > self.max_camera_bytes:
                    raise ValueError("Camera sequence exceeds the byte limit")
                image = _open_camera_image(
                    source,
                    max_bytes=self.max_camera_bytes,
                    max_pixels=self.max_camera_pixels,
                )
                frame_fov = evaluate_map_bearing_fov(
                    ego_map_xy=[
                        float(_get(frame, "map_x")),
                        float(_get(frame, "map_y")),
                    ],
                    ego_yaw_rad=float(_get(frame, "map_yaw_rad")),
                    target_polygon_map=_slot_polygon(case),
                    target_center_map=_slot_center(case),
                    map_units_per_meter=_map_units_per_meter(scene),
                    half_fov_deg=self.half_fov_deg,
                    yaw_uncertainty_deg=self.yaw_uncertainty_deg,
                    camera_frame_id=int(_get(frame, "camera_frame_id")),
                )
                loaded.append((frame, image, frame_fov))
                source_rows.append(
                    {
                        "lidar_frame_id": int(_get(frame, "frame_id")),
                        "camera_frame_id": int(_get(frame, "camera_frame_id")),
                        "camera_timestamp": float(_get(frame, "camera_timestamp")),
                        "relative_to_t0_ms": round(
                            (float(_get(frame, "camera_timestamp")) - scene.anchor_timestamp)
                            * 1000.0,
                            3,
                        ),
                        "camera_lidar_dt_ms": round(
                            float(_get(frame, "camera_lidar_dt_sec")) * 1000.0,
                            3,
                        ),
                        "target_bearing_deg": frame_fov.details.get(
                            "target_bearing_deg"
                        ),
                        "fov_visibility": frame_fov.visibility.value,
                        "fov_confidence": frame_fov.confidence,
                        "path": str(source),
                    }
                )
            sheet = _render_camera_sequence(
                loaded,
                anchor_timestamp=scene.anchor_timestamp,
            )
            destination = (
                Path(output_dir).resolve(strict=False)
                / "camera_sequence"
                / f"{_safe_component(_slot_id(case))}_round_{int(round_index):02d}.png"
            )
            _atomic_png(sheet, destination)
            resource_key = _register_artifact(
                case,
                f"part2_camera_sequence_round_{int(round_index)}",
                destination,
            )
        except (OSError, TypeError, ValueError) as exc:
            return _record(
                case=case,
                tool_name="camera_sequence",
                round_index=round_index,
                status="failed",
                summary="The causal Camera sequence could not be rendered safely.",
                metadata={
                    "reason_codes": ["camera_sequence_render_failed"],
                    "error_type": type(exc).__name__,
                },
                modality="camera",
            )
        return _record(
            case=case,
            tool_name="camera_sequence",
            round_index=round_index,
            status="ok",
            summary=(
                "Strictly causal Camera contact sheet. It provides temporal context but "
                "still has no audited target-slot pixel projection."
            ),
            artifact_paths=(destination,),
            metadata={
                "target_slot_id": _slot_id(case),
                "frame_count": len(source_rows),
                "strictly_causal_to_t0": True,
                "frames": source_rows,
                "model_image_paths": [str(destination)],
                "map_to_camera_projection_used": False,
            },
            modality="camera",
            reason_codes=["temporal_camera_context_without_pixel_projection"],
            resource_keys=[resource_key],
        )

    camera_sequence = inspect_camera_sequence

    def crop_camera(
        self,
        scene: SceneSnapshot,
        case: SlotCase,
        output_dir: str | Path,
        round_index: int,
        bbox_norm: Sequence[float],
        enhancement: str = "none",
    ) -> EvidenceRecord:
        """Safely crop the raw Camera frame using an LLM-proposed normalized box."""

        if not self._camera_allowed(case):
            return _record(
                case=case,
                tool_name="camera_crop",
                round_index=round_index,
                status="unavailable",
                summary="Camera crop is forbidden because the horizontal projection gate is outside.",
                metadata={"reason_codes": ["target_outside_camera_projection"]},
                modality="camera",
            )
        source = _resource_path(scene, case, CAMERA_RESOURCE_KEYS)
        if source is None:
            return _record(
                case=case,
                tool_name="camera_crop",
                round_index=round_index,
                status="unavailable",
                summary="No raw Camera resource is attached to this t0 scene.",
                metadata={"reason_codes": ["camera_resource_missing"]},
                modality="camera",
            )
        try:
            if (
                isinstance(bbox_norm, (str, bytes))
                or not isinstance(bbox_norm, Sequence)
                or len(bbox_norm) != 4
            ):
                raise ValueError("bbox_norm must contain four coordinates")
            x1, y1, x2, y2 = tuple(
                _finite(value, f"bbox_norm[{index}]")
                for index, value in enumerate(bbox_norm)
            )
            if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
                raise ValueError("bbox_norm coordinates must be within [0,1]")
            if x2 - x1 < 0.02 or y2 - y1 < 0.02:
                raise ValueError("bbox_norm is too small or has non-positive area")
            enhancement_name = str(enhancement).strip().lower()
            if enhancement_name not in {"none", "contrast", "sharpen"}:
                raise ValueError("enhancement must be none, contrast, or sharpen")
            camera = _open_camera_image(
                source,
                max_bytes=self.max_camera_bytes,
                max_pixels=self.max_camera_pixels,
            )
            left = max(0, min(camera.width - 1, int(math.floor(x1 * camera.width))))
            top = max(0, min(camera.height - 1, int(math.floor(y1 * camera.height))))
            right = max(left + 1, min(camera.width, int(math.ceil(x2 * camera.width))))
            bottom = max(top + 1, min(camera.height, int(math.ceil(y2 * camera.height))))
            crop = camera.crop((left, top, right, bottom))
            if enhancement_name == "contrast":
                crop = ImageEnhance.Contrast(crop).enhance(1.35)
            elif enhancement_name == "sharpen":
                crop = ImageEnhance.Sharpness(crop).enhance(1.8)
            maximum_side = max(crop.size)
            if maximum_side > self.max_crop_side_px:
                resize_scale = self.max_crop_side_px / maximum_side
                crop = crop.resize(
                    (
                        max(1, int(round(crop.width * resize_scale))),
                        max(1, int(round(crop.height * resize_scale))),
                    ),
                    Image.Resampling.LANCZOS,
                )
            box_key = hashlib.sha256(
                json.dumps(
                    [round(value, 6) for value in (x1, y1, x2, y2)],
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest()[:10]
            destination = (
                Path(output_dir).resolve(strict=False)
                / "camera_crop"
                / (
                    f"{_safe_component(_slot_id(case))}_round_{int(round_index):02d}_"
                    f"{box_key}_{enhancement_name}.png"
                )
            )
            _atomic_png(crop, destination)
            hypothesis_view = camera.copy()
            hypothesis_draw = ImageDraw.Draw(hypothesis_view)
            hypothesis_draw.rectangle(
                (left, top, right - 1, bottom - 1),
                outline=(208, 43, 151),
                width=max(4, camera.width // 240),
            )
            label_font = _ui_font(max(18, camera.width // 48), bold=True)
            label = "AGENT HYPOTHESIS · NOT GROUND TRUTH"
            label_box = hypothesis_draw.textbbox((0, 0), label, font=label_font)
            label_width = label_box[2] - label_box[0] + 24
            label_height = label_box[3] - label_box[1] + 18
            label_top = max(0, top - label_height)
            hypothesis_draw.rectangle(
                (left, label_top, min(camera.width, left + label_width), label_top + label_height),
                fill=(208, 43, 151),
            )
            hypothesis_draw.text(
                (left + 12, label_top + 7),
                label,
                fill=(255, 255, 255),
                font=label_font,
            )
            hypothesis_destination = destination.with_name(
                destination.stem + "_hypothesis_on_full_camera.png"
            )
            _atomic_png(hypothesis_view, hypothesis_destination)
            resource_key = _register_artifact(
                case,
                (
                    f"part2_camera_crop_round_{int(round_index)}_"
                    f"{box_key}_{enhancement_name}"
                ),
                destination,
            )
            hypothesis_resource_key = _register_artifact(
                case,
                (
                    f"part2_camera_crop_hypothesis_round_{int(round_index)}_"
                    f"{box_key}_{enhancement_name}"
                ),
                hypothesis_destination,
            )
        except (OSError, TypeError, ValueError) as exc:
            return _record(
                case=case,
                tool_name="camera_crop",
                round_index=round_index,
                status="failed",
                summary="The requested normalized Camera crop was rejected or could not be rendered.",
                metadata={
                    "reason_codes": ["camera_crop_invalid_or_failed"],
                    "error_type": type(exc).__name__,
                },
                modality="camera",
            )
        return _record(
            case=case,
            tool_name="camera_crop",
            round_index=round_index,
            status="ok",
            summary=(
                f"Raw Camera crop for bbox {[round(value, 4) for value in (x1, y1, x2, y2)]} "
                f"with {enhancement_name} enhancement."
            ),
            artifact_paths=(hypothesis_destination, destination),
            metadata={
                "target_slot_id": _slot_id(case),
                "bbox_norm": [x1, y1, x2, y2],
                "bbox_px": [left, top, right, bottom],
                "bbox_norm_coordinate_space": "raw_full_camera",
                "source_size_px": [camera.width, camera.height],
                "crop_size_px": [crop.width, crop.height],
                "enhancement": enhancement_name,
                "map_to_camera_projection_used": False,
                "agent_hypothesis_overlay": True,
                "hypothesis_overlay_is_ground_truth": False,
                "model_image_paths": [str(hypothesis_destination), str(destination)],
            },
            modality="camera",
            resource_keys=[hypothesis_resource_key, resource_key],
        )

    camera_crop = crop_camera

    def inspect_lidar_detail(
        self,
        scene: SceneSnapshot,
        case: SlotCase,
        output_dir: str | Path,
        round_index: int,
    ) -> EvidenceRecord:
        """Validate and render an attached target-local LiDAR evidence pack."""

        extended_source = _resource_path(
            scene, case, EXTENDED_LIDAR_RESOURCE_KEYS
        )
        source = extended_source or _resource_path(scene, case, LIDAR_RESOURCE_KEYS)
        if source is None:
            return _record(
                case=case,
                tool_name="lidar_detail",
                round_index=round_index,
                status="unavailable",
                summary="No Part2 LiDAR evidence pack is attached to this SlotCase.",
                metadata={"reason_codes": ["lidar_evidence_pack_missing"]},
                modality="lidar",
            )
        try:
            if not is_lidar_evidence_pack(source):
                return _record(
                    case=case,
                    tool_name="lidar_detail",
                    round_index=round_index,
                    status="unavailable",
                    summary="The attached LiDAR artifact is not a supported evidence pack.",
                    metadata={
                        "reason_codes": ["lidar_evidence_pack_unsupported"],
                    },
                    modality="lidar",
                )
            pack = load_lidar_evidence_pack(
                source,
                expected_slot_id=_slot_id(case),
            )
            decision_override: Mapping[str, Any] | None = None
            decision_source = "part1_15frame"
            if extended_source is not None:
                decision_path = _resource_path(
                    scene, case, EXTENDED_LIDAR_DECISION_KEYS
                )
                if decision_path is None or not decision_path.is_file():
                    raise EvidencePackError("extended LiDAR decision resource is missing")
                raw_decision = json.loads(decision_path.read_text(encoding="utf-8"))
                if not isinstance(raw_decision, Mapping):
                    raise EvidencePackError("extended LiDAR decision must be an object")
                decision_override = raw_decision.get("decision")
                if not isinstance(decision_override, Mapping):
                    raise EvidencePackError("extended LiDAR decision payload is missing")
                if str(decision_override.get("slot_id", "")) != _slot_id(case):
                    raise EvidencePackError("extended LiDAR decision slot identity mismatch")
                declared_frames = tuple(
                    int(value) for value in raw_decision.get("selected_frames", ())
                )
                if tuple(int(value) for value in pack.selected_frames) != declared_frames:
                    raise EvidencePackError(
                        "extended evidence pack frame set does not match its decision"
                    )
                if int(raw_decision.get("anchor_frame")) != int(scene.anchor_frame_id):
                    raise EvidencePackError("extended LiDAR decision anchor mismatch")
                decision_source = "part2_extended_causal_lidar"
            else:
                expected_frames = tuple(int(value) for value in case.evidence_frame_ids)
                if tuple(int(value) for value in pack.selected_frames) != expected_frames:
                    raise EvidencePackError(
                        "evidence pack frame set does not match the current SlotCase"
                    )
            pack_anchor = pack.metadata.get("anchor_frame")
            expected_anchor = (
                raw_decision.get("alignment_anchor_frame")
                if extended_source is not None
                else case.evidence_anchor_frame_id
            )
            if int(pack_anchor) != int(expected_anchor):
                raise EvidencePackError(
                    "evidence pack anchor does not match the current SlotCase"
                )
            if any(int(value) > int(scene.anchor_frame_id) for value in pack.selected_frames):
                raise EvidencePackError("evidence pack contains post-t0 frames")
            content = render_lidar_triptych(pack)
            destination = (
                Path(output_dir).resolve(strict=False)
                / "lidar_detail"
                / f"{_safe_component(_slot_id(case))}_round_{int(round_index):02d}.png"
            )
            _atomic_bytes(content, destination)
            effective_decision = resolve_effective_decision(
                case,
                decision_override,
            )
            geometry_card = build_geometry_card(
                case,
                pack,
                decision_override=effective_decision,
                decision_source=decision_source,
            )
            geometry_card["terminal_geometry_gate"] = assess_terminal_geometry(
                geometry_card
            )
            card_destination = destination.with_name(
                f"{destination.stem}_geometry_card.png"
            )
            render_geometry_card(geometry_card, card_destination)
            combined_destination = destination.with_name(
                f"{destination.stem}_model_evidence.png"
            )
            combine_triptych_and_card(
                destination,
                card_destination,
                combined_destination,
            )
            explained_destination = destination.with_name(
                f"{destination.stem}_explained.png"
            )
            render_explainable_lidar(
                pack,
                geometry_card,
                explained_destination,
                decision=effective_decision,
            )
            resource_key = _register_artifact(
                case,
                f"part2_lidar_detail_round_{int(round_index)}",
                destination,
            )
            card_resource_key = _register_artifact(
                case,
                f"part2_lidar_geometry_card_round_{int(round_index)}",
                card_destination,
            )
            combined_resource_key = _register_artifact(
                case,
                f"part2_lidar_model_evidence_round_{int(round_index)}",
                combined_destination,
            )
            explained_resource_key = _register_artifact(
                case,
                f"part2_lidar_explained_round_{int(round_index)}",
                explained_destination,
            )
        except (EvidenceMediaError, EvidencePackError, OSError, TypeError, ValueError) as exc:
            return _record(
                case=case,
                tool_name="lidar_detail",
                round_index=round_index,
                status="failed",
                summary="The attached LiDAR evidence pack could not be validated or rendered.",
                metadata={
                    "reason_codes": ["lidar_evidence_render_failed"],
                    "error_type": type(exc).__name__,
                },
                modality="lidar",
            )
        metadata = pack.metadata
        is_incremental = extended_source is not None
        stability = geometry_card.get("robustness", {})
        robustness_available = (
            isinstance(stability, Mapping)
            and int(stability.get("total_variants", 0) or 0) > 0
        )
        return _record(
            case=case,
            tool_name="lidar_detail",
            round_index=round_index,
            status="ok",
            summary=(
                (
                    "Extended causal LiDAR evidence adds pre-t0 viewpoints beyond "
                    "the Part1 window. "
                    if is_incremental
                    else "Part1 LiDAR review only; this is not incremental Part2 evidence. "
                )
                + "The ownership view exposes free voxels, fitted obstacle "
                "candidate, boundary veto, viewpoints, vertical structure, and "
                "terminal hard gates."
            ),
            artifact_paths=(
                destination,
                card_destination,
                combined_destination,
                explained_destination,
            ),
            metadata={
                "target_slot_id": _slot_id(case),
                "source_sha256": pack.source_sha256,
                "selected_frames": [int(value) for value in pack.selected_frames],
                "valid_frames": [int(value) for value in pack.valid_frames],
                "selected_frame_count": int(len(pack.selected_frames)),
                "valid_frame_count": int(len(pack.valid_frames)),
                "point_count": int(len(pack.points_local_xyzi)),
                "evidence_schema_version": metadata.get("schema_version"),
                "renderer": "parking_slot_agent_v2.lidar_explainability",
                "geometry_card": geometry_card,
                "evidence_source": decision_source,
                "effective_decision_source": decision_source,
                "is_incremental_over_part1": is_incremental,
                "robustness_available": robustness_available,
                "expected_information_gain": (
                    "incremental" if is_incremental else "none_same_as_part1"
                ),
                "model_image_paths": [str(explained_destination)],
            },
            modality="lidar",
            resource_keys=[
                resource_key,
                card_resource_key,
                combined_resource_key,
                explained_resource_key,
            ],
        )

    lidar_detail = inspect_lidar_detail


__all__ = [
    "CAMERA_RESOURCE_KEYS",
    "EXTENDED_LIDAR_DECISION_KEYS",
    "EXTENDED_LIDAR_RESOURCE_KEYS",
    "LIDAR_RESOURCE_KEYS",
    "V2ToolSuite",
    "describe_lidar_capability",
]
