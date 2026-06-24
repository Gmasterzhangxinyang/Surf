from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .datatypes import SlotHypothesis


@dataclass
class CarlaActors:
    """Actors created by the CARLA demo script."""

    ego: Any
    lidar: Any
    camera: Any | None
    parked_vehicles: list[Any]
    lidar_queue: queue.Queue


def import_carla():
    """Import CARLA lazily so the repository still works without CARLA installed."""
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is not installed. Install the matching client package, e.g. "
            "`python3 -m pip install carla==0.9.16`, or install the wheel from "
            "`PythonAPI/carla/dist` inside your CARLA package."
        ) from exc
    return carla


def connect_carla(host: str, port: int, timeout_s: float):
    """Connect to a running CARLA server."""
    carla = import_carla()
    client = carla.Client(host, port)
    client.set_timeout(timeout_s)
    return client


def make_transform(spec: dict):
    """Create a CARLA transform from a YAML-friendly dict."""
    carla = import_carla()
    location = spec.get("location", {})
    rotation = spec.get("rotation", {})
    return carla.Transform(
        carla.Location(
            x=float(location.get("x", 0.0)),
            y=float(location.get("y", 0.0)),
            z=float(location.get("z", 0.0)),
        ),
        carla.Rotation(
            pitch=float(rotation.get("pitch", 0.0)),
            yaw=float(rotation.get("yaw", 0.0)),
            roll=float(rotation.get("roll", 0.0)),
        ),
    )


def configure_synchronous_world(world, fixed_delta_seconds: float):
    """Enable deterministic ticking for repeatable CARLA data capture."""
    settings = world.get_settings()
    previous = {
        "synchronous_mode": settings.synchronous_mode,
        "fixed_delta_seconds": settings.fixed_delta_seconds,
    }
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)
    return previous


def restore_world_settings(world, previous: dict) -> None:
    """Restore CARLA world settings after a run."""
    settings = world.get_settings()
    settings.synchronous_mode = previous["synchronous_mode"]
    settings.fixed_delta_seconds = previous["fixed_delta_seconds"]
    world.apply_settings(settings)


def _choose_blueprint(blueprint_library, filter_name: str, preferred_id: str | None = None):
    if preferred_id:
        candidates = blueprint_library.filter(preferred_id)
        if candidates:
            return candidates[0]
    candidates = blueprint_library.filter(filter_name)
    if not candidates:
        raise RuntimeError(f"No CARLA blueprint matched {filter_name!r}.")
    return candidates[0]


def spawn_carla_scene(world, cfg: dict) -> CarlaActors:
    """Spawn ego, LiDAR, optional camera, and parked vehicles from config."""
    carla = import_carla()
    bp_lib = world.get_blueprint_library()
    scene_cfg = cfg["carla"]

    ego_bp = _choose_blueprint(bp_lib, "vehicle.*", scene_cfg.get("ego_blueprint"))
    ego = world.try_spawn_actor(ego_bp, make_transform(scene_cfg["ego_spawn"]))
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle. Try another ego_spawn transform.")
    ego.set_autopilot(False)

    parked_vehicles = []
    for item in scene_cfg.get("parked_vehicles", []):
        bp = _choose_blueprint(bp_lib, "vehicle.*", item.get("blueprint"))
        actor = world.try_spawn_actor(bp, make_transform(item["transform"]))
        if actor is not None:
            actor.set_autopilot(False)
            parked_vehicles.append(actor)

    lidar_cfg = scene_cfg["lidar"]
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    for attr, value in lidar_cfg.get("attributes", {}).items():
        lidar_bp.set_attribute(attr, str(value))
    lidar_queue: queue.Queue = queue.Queue()
    lidar = world.spawn_actor(
        lidar_bp,
        make_transform(lidar_cfg["transform"]),
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid,
    )
    lidar.listen(lidar_queue.put)

    camera = None
    camera_cfg = scene_cfg.get("camera")
    if camera_cfg and camera_cfg.get("enabled", False):
        camera_bp = bp_lib.find("sensor.camera.rgb")
        for attr, value in camera_cfg.get("attributes", {}).items():
            camera_bp.set_attribute(attr, str(value))
        camera = world.spawn_actor(
            camera_bp,
            make_transform(camera_cfg["transform"]),
            attach_to=ego,
            attachment_type=carla.AttachmentType.Rigid,
        )

    return CarlaActors(
        ego=ego,
        lidar=lidar,
        camera=camera,
        parked_vehicles=parked_vehicles,
        lidar_queue=lidar_queue,
    )


def destroy_carla_actors(actors: CarlaActors) -> None:
    """Destroy actors created for the demo."""
    for actor in [actors.camera, actors.lidar, *actors.parked_vehicles, actors.ego]:
        if actor is not None:
            actor.destroy()


def tick_and_get_lidar(world, lidar_queue: queue.Queue, warmup_frames: int, timeout_s: float):
    """Tick CARLA and return the latest LiDAR measurement."""
    measurement = None
    for _ in range(max(1, warmup_frames)):
        world.tick()
        try:
            measurement = lidar_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise RuntimeError("Timed out waiting for CARLA LiDAR data.") from exc
    return measurement


def lidar_measurement_to_ego_points(measurement, sensor_transform_cfg: dict) -> np.ndarray:
    """
    Convert CARLA LiDAR points into DASP-Park ego BEV coordinates.

    CARLA uses x-forward, y-right, z-up in the sensor frame. DASP-Park uses
    x-forward, y-left, z-up, so y is negated. This minimal adapter assumes the
    LiDAR sensor is mounted with zero pitch/yaw/roll, which is the default config.
    """
    raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
    if raw.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    points = raw.reshape((-1, 4))[:, :3].copy()
    location = sensor_transform_cfg.get("location", {})
    sensor_x = float(location.get("x", 0.0))
    sensor_y = float(location.get("y", 0.0))
    sensor_z = float(location.get("z", 0.0))
    x_forward = points[:, 0] + sensor_x
    y_left = -(points[:, 1] + sensor_y)
    z_up = points[:, 2] + sensor_z
    return np.column_stack([x_forward, y_left, z_up]).astype(np.float32)


def load_slot_map_from_config(cfg: dict) -> list[SlotHypothesis]:
    """Load known parking slot map in DASP-Park ego coordinates."""
    slots = []
    for item in cfg["slot_map"]["slots"]:
        polygon = np.array(item["polygon_xy"], dtype=np.float32)
        entrance = np.array(item["entrance_xy"], dtype=np.float32)
        slots.append(
            SlotHypothesis(
                slot_id=item["slot_id"],
                polygon_xy=polygon,
                entrance_xy=entrance,
                metadata={
                    "center": item.get("center", polygon.mean(axis=0).tolist()),
                    "row": item.get("row", "carla"),
                    "preference_bonus": float(item.get("preference_bonus", 0.0)),
                },
            )
        )
    return slots


def wait_for_server(host: str, port: int, timeout_s: float):
    """Small helper used by users to check whether CARLA is running."""
    start = time.time()
    last_error = None
    while time.time() - start < timeout_s:
        try:
            return connect_carla(host, port, timeout_s=2.0)
        except RuntimeError as exc:
            raise exc
        except Exception as exc:  # CARLA raises transport-specific errors.
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Could not connect to CARLA at {host}:{port}: {last_error}")
