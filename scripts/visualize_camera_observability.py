#!/usr/bin/env python3
"""Render the exact pre-Camera observability geometry on a Camera image.

The script calls ``assess_camera_observability(..., debug_trace=...)`` and
draws that returned trace; it does not implement a second visibility decision
and does not invoke any Camera semantic model.

LiDAR points and reprojection references must explicitly use the scene map
frame, map-unit XY, and metric Z.  Missing references produce count=0 and
null errors instead of a fabricated all-image average.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from parking_slot_part2.camera_calibration import CameraCalibration
from parking_slot_part2.camera_observability import (
    CameraObservabilityConfig,
    CameraObservabilityInput,
    assess_camera_observability,
)


_STATUS_COLORS = {
    "clear": (30, 210, 80, 220),
    "blocked": (245, 45, 45, 235),
    "uncertain": (255, 160, 25, 235),
    "edge_unreliable": (180, 80, 230, 220),
    "outside_fov": (120, 120, 120, 180),
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _points(value: Any, name: str, columns: int = 3) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numbers") from exc
    if array.ndim != 2 or array.shape[1] < columns:
        raise ValueError(f"{name} must have shape [N, >={columns}]")
    array = array[:, :columns]
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _scene_geometry(
    scene: Mapping[str, Any],
) -> tuple[CameraObservabilityInput, CameraCalibration]:
    row = CameraObservabilityInput.from_mapping(scene)
    if not row.camera_pose_contract_present or row.camera_T_map_camera_m is None:
        raise ValueError("a valid camera-map-pose/1.0 transform is required")
    if row.map_units_per_meter is None or row.map_coordinate_frame is None:
        raise ValueError("scene map frame and map_units_per_meter are required")
    calibration = CameraCalibration.from_mapping(
        _mapping(scene.get("camera_calibration"), "camera_calibration")
    )
    if row.camera_frame_id != calibration.camera_frame:
        raise ValueError("Camera pose and calibration frames differ")
    return row, calibration


def _map_to_camera(
    points_map_xyz: np.ndarray,
    row: CameraObservabilityInput,
) -> np.ndarray:
    assert row.map_units_per_meter is not None
    assert row.camera_T_map_camera_m is not None
    points_map_m = np.asarray(points_map_xyz, dtype=np.float64).copy()
    points_map_m[:, :2] /= row.map_units_per_meter
    transform = np.asarray(row.camera_T_map_camera_m, dtype=np.float64)
    return (points_map_m - transform[:3, 3]) @ transform[:3, :3]


def _project_map(
    points_map_xyz: np.ndarray,
    row: CameraObservabilityInput,
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = _map_to_camera(points_map_xyz, row)
    pixels, valid = calibration.project_camera_points(camera_points)
    in_image = (
        valid
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= calibration.image_width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= calibration.image_height - 1)
    )
    return pixels, in_image, camera_points


def _project_horizontal_angle(
    angle_deg: float,
    calibration: CameraCalibration,
) -> float | None:
    angle = math.radians(angle_deg)
    point = np.asarray([[math.sin(angle), 0.0, math.cos(angle)]])
    pixels, valid = calibration.project_camera_points(point)
    return float(pixels[0, 0]) if bool(valid[0]) else None


def _draw_fov_zones(
    overlay: Image.Image,
    calibration: CameraCalibration,
    config: CameraObservabilityConfig,
) -> None:
    draw = ImageDraw.Draw(overlay, "RGBA")
    boundaries = [
        -config.nominal_half_fov_deg,
        -config.edge_unreliable_start_deg,
        -config.reliable_half_fov_deg,
        config.reliable_half_fov_deg,
        config.edge_unreliable_start_deg,
        config.nominal_half_fov_deg,
    ]
    projected = [_project_horizontal_angle(item, calibration) for item in boundaries]
    if all(item is not None for item in projected):
        x = [max(0.0, min(float(calibration.image_width), float(item))) for item in projected]
        spans = [
            (x[0], x[1], (245, 50, 50, 42)),
            (x[1], x[2], (255, 170, 30, 36)),
            (x[2], x[3], (30, 210, 80, 24)),
            (x[3], x[4], (255, 170, 30, 36)),
            (x[4], x[5], (245, 50, 50, 42)),
        ]
        for left, right, color in spans:
            draw.rectangle((left, 0, right, calibration.image_height), fill=color)

    # Boundary curves use the calibrated fisheye/pinhole model across
    # vertical angles; no horizontal-angle linear pixel mapping is used.
    for angle, color, width in (
        (-config.reliable_half_fov_deg, (30, 210, 80, 190), 2),
        (config.reliable_half_fov_deg, (30, 210, 80, 190), 2),
        (-config.edge_unreliable_start_deg, (255, 160, 25, 210), 2),
        (config.edge_unreliable_start_deg, (255, 160, 25, 210), 2),
        (-config.nominal_half_fov_deg, (245, 45, 45, 220), 2),
        (config.nominal_half_fov_deg, (245, 45, 45, 220), 2),
    ):
        horizontal = math.radians(angle)
        elevations = np.radians(np.linspace(-88.0, 88.0, 177))
        directions = np.column_stack(
            [
                np.cos(elevations) * math.sin(horizontal),
                np.sin(elevations),
                np.cos(elevations) * math.cos(horizontal),
            ]
        )
        pixels, valid = calibration.project_camera_points(directions)
        current: list[tuple[float, float]] = []
        for uv, is_valid in zip(pixels, valid):
            inside = bool(is_valid) and (
                0.0 <= uv[0] < calibration.image_width
                and 0.0 <= uv[1] < calibration.image_height
            )
            if inside:
                current.append((float(uv[0]), float(uv[1])))
            elif len(current) >= 2:
                draw.line(current, fill=color, width=width)
                current = []
            else:
                current = []
        if len(current) >= 2:
            draw.line(current, fill=color, width=width)


def _static_rows(scene: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    layer = scene.get("static_obstacle_map", scene.get("static_obstacle_layer"))
    if not isinstance(layer, Mapping):
        return ()
    rows = layer.get("static_obstacles", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    return tuple(item for item in rows if isinstance(item, Mapping))


def _draw_static_obstacles(
    draw: ImageDraw.ImageDraw,
    scene: Mapping[str, Any],
    row: CameraObservabilityInput,
    calibration: CameraCalibration,
) -> None:
    for obstacle in _static_rows(scene):
        polygon = obstacle.get("polygon_xy", obstacle.get("polygon_map"))
        try:
            xy = _points(polygon, "static obstacle polygon", columns=2)
            minimum = float(obstacle.get("min_z", obstacle.get("min_height_m")))
            maximum = float(obstacle.get("max_z", obstacle.get("max_height_m")))
        except (TypeError, ValueError):
            continue
        color = (255, 65, 65, 230) if str(obstacle.get("type")) == "wall" else (255, 200, 30, 230)
        level_pixels: list[np.ndarray] = []
        level_valid: list[np.ndarray] = []
        for z_value in (minimum, maximum):
            xyz = np.column_stack([xy, np.full(len(xy), z_value)])
            pixels, valid, _ = _project_map(xyz, row, calibration)
            level_pixels.append(pixels)
            level_valid.append(valid)
            if bool(np.all(valid)):
                points = [tuple(float(item) for item in uv) for uv in pixels]
                draw.line(points + [points[0]], fill=color, width=2)
        for index in range(len(xy)):
            if level_valid[0][index] and level_valid[1][index]:
                draw.line(
                    [tuple(level_pixels[0][index]), tuple(level_pixels[1][index])],
                    fill=color,
                    width=1,
                )


def _load_lidar_points(
    scene: Mapping[str, Any],
    scene_dir: Path,
    row: CameraObservabilityInput,
) -> np.ndarray:
    payload = scene.get("lidar_projection_input")
    if payload is None:
        return np.empty((0, 3), dtype=np.float64)
    contract = _mapping(payload, "lidar_projection_input")
    if contract.get("coordinate_frame") != row.map_coordinate_frame:
        raise ValueError("LiDAR and Camera map coordinate frames differ")
    if contract.get("xy_unit") != "map_unit" or contract.get("z_unit") != "m":
        raise ValueError("LiDAR projection units must be map_unit XY and metric Z")
    if "points_map_xyz" in contract:
        return _points(contract["points_map_xyz"], "points_map_xyz")
    raw_path = contract.get("points_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("LiDAR input needs points_map_xyz or points_path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = scene_dir / path
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "points_map_xyzi" in loaded:
                values = loaded["points_map_xyzi"]
            elif "points_map_xyz" in loaded:
                values = loaded["points_map_xyz"]
            else:
                raise ValueError("LiDAR NPZ has no explicit map-point array")
        finally:
            loaded.close()
    else:
        values = loaded
    return _points(values, "LiDAR map points")


def _empty_error_summary() -> dict[str, float | int | None]:
    return {"count": 0, "mean_px": None, "rmse_px": None, "max_px": None}


def _summarize_errors(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return _empty_error_summary()
    return {
        "count": int(len(array)),
        "mean_px": round(float(np.mean(array)), 6),
        "rmse_px": round(float(np.sqrt(np.mean(array * array))), 6),
        "max_px": round(float(np.max(array)), 6),
    }


def _reprojection_errors(
    scene: Mapping[str, Any],
    row: CameraObservabilityInput,
    calibration: CameraCalibration,
    draw: ImageDraw.ImageDraw,
    config: CameraObservabilityConfig,
) -> tuple[dict[str, dict[str, float | int | None]], list[dict[str, Any]]]:
    raw = scene.get("reprojection_references", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("reprojection_references must be an array")
    errors: dict[str, list[float]] = {"left_edge": [], "center": [], "right_edge": []}
    records: list[dict[str, Any]] = []
    for index, reference in enumerate(raw):
        item = _mapping(reference, f"reprojection_references[{index}]")
        point = _points([item.get("point_map_xyz")], "reference point")
        observed = np.asarray(item.get("observed_pixel_uv"), dtype=np.float64)
        if observed.shape != (2,) or not np.isfinite(observed).all():
            raise ValueError("observed_pixel_uv must contain two finite values")
        pixels, valid, camera_points = _project_map(point, row, calibration)
        if not bool(valid[0]):
            continue
        predicted = pixels[0]
        bearing = math.degrees(
            math.atan2(float(camera_points[0, 0]), float(camera_points[0, 2]))
        )
        zone = (
            "center"
            if abs(bearing) <= config.reliable_half_fov_deg
            else ("left_edge" if bearing < 0.0 else "right_edge")
        )
        error = float(np.linalg.norm(predicted - observed))
        errors[zone].append(error)
        draw.ellipse(
            (observed[0] - 3, observed[1] - 3, observed[0] + 3, observed[1] + 3),
            outline=(255, 50, 255, 255),
            width=2,
        )
        draw.line([tuple(observed), tuple(predicted)], fill=(40, 230, 255, 240), width=1)
        records.append(
            {
                "id": str(item.get("id", index)),
                "zone": zone,
                "predicted_pixel_uv": [round(float(v), 6) for v in predicted],
                "observed_pixel_uv": [round(float(v), 6) for v in observed],
                "error_px": round(error, 6),
            }
        )
    return ({key: _summarize_errors(value) for key, value in errors.items()}, records)


def render_camera_observability(
    scene: Mapping[str, Any],
    *,
    scene_dir: str | Path = ".",
    image_path: str | Path | None = None,
    output_dir: str | Path = "outputs/camera_observability_validation",
    stem: str = "camera_observability",
) -> dict[str, Any]:
    """Render one validated scene and return its report mapping."""

    row, calibration = _scene_geometry(scene)
    raw_config = scene.get("camera_observability_config", {})
    config = CameraObservabilityConfig.from_mapping(
        _mapping(raw_config, "camera_observability_config")
    )
    trace: list[dict[str, Any]] = []
    assessment = assess_camera_observability(scene, config=config, debug_trace=trace)

    base_dir = Path(scene_dir)
    source_image = image_path if image_path is not None else scene.get("camera_image_path")
    if not isinstance(source_image, (str, Path)):
        raise ValueError("camera_image_path is required")
    source = Path(source_image)
    if not source.is_absolute():
        source = base_dir / source
    image = Image.open(source).convert("RGBA")
    if image.size != (calibration.image_width, calibration.image_height):
        raise ValueError("Camera image dimensions differ from calibration")

    zone_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    _draw_fov_zones(zone_overlay, calibration, config)
    image = Image.alpha_composite(image, zone_overlay)
    draw = ImageDraw.Draw(image, "RGBA")
    principal = (float(calibration.K[0, 2]), float(calibration.K[1, 2]))

    _draw_static_obstacles(draw, scene, row, calibration)

    lidar = _load_lidar_points(scene, base_dir, row)
    lidar_pixels, lidar_valid, _ = _project_map(lidar, row, calibration) if len(lidar) else (
        np.empty((0, 2)),
        np.empty((0,), dtype=bool),
        np.empty((0, 3)),
    )
    for uv in lidar_pixels[lidar_valid]:
        draw.ellipse((uv[0] - 1, uv[1] - 1, uv[0] + 1, uv[1] + 1), fill=(30, 220, 255, 190))

    for ray in trace:
        pixel = ray.get("pixel_uv")
        if not isinstance(pixel, Sequence) or len(pixel) != 2:
            continue
        endpoint = (float(pixel[0]), float(pixel[1]))
        status = str(ray.get("status", "outside_fov"))
        color = _STATUS_COLORS.get(status, (200, 200, 200, 190))
        draw.line([principal, endpoint], fill=color, width=1)
        radius = 3 if status in {"blocked", "uncertain"} else 2
        draw.ellipse(
            (
                endpoint[0] - radius,
                endpoint[1] - radius,
                endpoint[0] + radius,
                endpoint[1] + radius,
            ),
            outline=color,
            width=2,
        )

    summaries, reference_records = _reprojection_errors(
        scene, row, calibration, draw, config
    )
    font = ImageFont.load_default()
    legend = (
        f"decision={assessment.decision}  "
        "ray: green=clear red=blocked orange=uncertain purple=edge"
    )
    draw.rectangle((0, 0, min(image.width, 690), 18), fill=(0, 0, 0, 190))
    draw.text((5, 4), legend, fill=(255, 255, 255, 255), font=font)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    image_output = destination / f"{stem}.png"
    report_output = destination / f"{stem}.json"
    image.convert("RGB").save(image_output)
    report: dict[str, Any] = {
        "schema_version": "camera-observability-visualization/1.0",
        "image_output": str(image_output),
        "report_output": str(report_output),
        "source_image": str(source),
        "assessment": assessment.to_dict(),
        "debug_ray_count": len(trace),
        "lidar_projected_count": int(np.sum(lidar_valid)),
        "reprojection_error_by_zone": summaries,
        "reprojection_references": reference_records,
        "fov_zone_boundaries_deg": {
            "nominal_half": config.nominal_half_fov_deg,
            "reliable_half": config.reliable_half_fov_deg,
            "edge_unreliable_start": config.edge_unreliable_start_deg,
        },
        "notes": [
            "FOV boundaries use the declared calibrated projection model.",
            "Rays are the debug trace from the same Camera-use assessment.",
            "No Camera semantic detector was called.",
        ],
    }
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def render_scene_file(
    scene_path: str | Path,
    *,
    image_path: str | Path | None = None,
    output_dir: str | Path = "outputs/camera_observability_validation",
    stem: str | None = None,
) -> dict[str, Any]:
    source = Path(scene_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("scene JSON must contain an object")
    return render_camera_observability(
        payload,
        scene_dir=source.parent,
        image_path=image_path,
        output_dir=output_dir,
        stem=stem or source.stem,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--image", type=Path, help="override scene camera_image_path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/camera_observability_validation"),
    )
    parser.add_argument("--stem")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = render_scene_file(
        args.scene,
        image_path=args.image,
        output_dir=args.output_dir,
        stem=args.stem,
    )
    print(
        json.dumps(
            {
                "image_output": report["image_output"],
                "report_output": report["report_output"],
                "decision": report["assessment"]["decision"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
