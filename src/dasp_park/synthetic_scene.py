import cv2
import numpy as np

from .datatypes import FREE, OCCUPIED, SlotHypothesis, SyntheticSample
from .diagnostics import mark_world_box
from .occupancy import build_occupancy_from_lidar, grid_shape


def create_front_camera_placeholders() -> dict[str, np.ndarray]:
    """Create three simple RGB front-view placeholder images."""
    images = {}
    names = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
    for name in names:
        img = np.full((480, 640, 3), (62, 66, 68), dtype=np.uint8)
        cv2.rectangle(img, (0, 310), (640, 480), (86, 88, 88), -1)
        for x in range(80, 620, 110):
            cv2.line(img, (x, 450), (x + 45, 310), (230, 230, 190), 2)
        cv2.rectangle(img, (300, 250), (338, 410), (120, 120, 125), -1)
        cv2.rectangle(img, (390, 345), (500, 410), (58, 80, 140), -1)
        cv2.putText(img, name, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
        cv2.putText(
            img,
            "Target slot cue",
            (225, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (205, 160, 255),
            2,
        )
        images[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return images


def create_parking_slots(
    grid_cfg: dict,
    target_slot_id: str | None = None,
    preferred_slot_id: str | None = None,
    preference_bonus: float = 0.0,
) -> list[SlotHypothesis]:
    """Create two rows of synthetic parking slots."""
    del grid_cfg
    slots = []
    x_centers = [8.0, 14.0, 20.0, 26.0, 32.0]
    rows = [("L", 8.0, 9.5, 6.5), ("R", -8.0, -6.5, -9.5)]
    for row_name, y_center, entrance_y, back_y in rows:
        for idx, x_center in enumerate(x_centers, start=1):
            x0, x1 = x_center - 2.5, x_center + 2.5
            y0, y1 = min(back_y, entrance_y), max(back_y, entrance_y)
            polygon = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            entrance = np.array([[x0, entrance_y], [x1, entrance_y]], dtype=np.float32)
            slot_id = f"slot_{row_name}{idx}"
            slots.append(
                SlotHypothesis(
                    slot_id=slot_id,
                    polygon_xy=polygon,
                    entrance_xy=entrance,
                    is_target_candidate=slot_id == target_slot_id,
                    metadata={
                        "center": [x_center, y_center],
                        "row": row_name,
                        "preference_bonus": preference_bonus if slot_id == preferred_slot_id else 0.0,
                    },
                )
            )
    return slots


def sample_ground_points(rng, grid_cfg: dict, n: int, noise_std: float) -> np.ndarray:
    """Sample ground plane points."""
    x = rng.uniform(grid_cfg["x_min"], grid_cfg["x_max"], n)
    y = rng.uniform(grid_cfg["y_min"], grid_cfg["y_max"], n)
    z = rng.normal(0.0, noise_std, n)
    return np.column_stack([x, y, z]).astype(np.float32)


def sample_box_points(
    rng,
    center_xy: tuple[float, float],
    length: float,
    width: float,
    height: float,
    n: int,
    noise_std: float,
) -> np.ndarray:
    """Sample points on and inside a vehicle-like box."""
    cx, cy = center_xy
    face = rng.integers(0, 6, n)
    x = rng.uniform(cx - length / 2, cx + length / 2, n)
    y = rng.uniform(cy - width / 2, cy + width / 2, n)
    z = rng.uniform(0.05, height, n)
    x[face == 0] = cx - length / 2
    x[face == 1] = cx + length / 2
    y[face == 2] = cy - width / 2
    y[face == 3] = cy + width / 2
    z[face == 4] = height
    z[face == 5] = rng.uniform(0.05, height * 0.45, np.count_nonzero(face == 5))
    pts = np.column_stack([x, y, z])
    pts += rng.normal(0.0, noise_std, pts.shape)
    pts[:, 2] = np.clip(pts[:, 2], 0.0, None)
    return pts.astype(np.float32)


def sample_cylinder_points(
    rng,
    center_xy: tuple[float, float],
    radius: float,
    height: float,
    n: int,
    noise_std: float,
) -> np.ndarray:
    """Sample points on a pillar-like cylinder."""
    cx, cy = center_xy
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    shell = rng.random(n) < 0.75
    r[shell] = radius
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z = rng.uniform(0.0, height, n)
    pts = np.column_stack([x, y, z])
    pts += rng.normal(0.0, noise_std, pts.shape)
    pts[:, 2] = np.clip(pts[:, 2], 0.0, None)
    return pts.astype(np.float32)


def degrade_lidar_observation(
    rng,
    points_full: np.ndarray,
    grid_cfg: dict,
    pillar_center: tuple[float, float],
    hidden_obstacles: list[dict] | None = None,
) -> np.ndarray:
    """Create sparse observed LiDAR from full points with occlusion and density degradation."""
    x, y, z = points_full[:, 0], points_full[:, 1], points_full[:, 2]
    distance = np.hypot(x, y)
    base_prob = np.where(z > 0.25, 0.52, 0.25)
    range_decay = np.clip(1.0 - distance / 55.0, 0.15, 1.0)
    keep_prob = base_prob * range_decay

    px, py = pillar_center
    behind_pillar = (x > px) & (y >= -4.0) & (y <= 2.0)
    keep_prob[behind_pillar] *= 0.08

    target_entrance = (x >= 17.5) & (x <= 22.5) & (y >= -6.8) & (y <= -5.2) & (z < 0.25)
    keep_prob[target_entrance] *= 0.12

    for obstacle in hidden_obstacles or []:
        cx, cy = obstacle["center"]
        margin = obstacle.get("observed_suppression_margin", 0.4)
        in_hidden_obstacle = (
            (x >= cx - obstacle["length"] / 2 - margin)
            & (x <= cx + obstacle["length"] / 2 + margin)
            & (y >= cy - obstacle["width"] / 2 - margin)
            & (y <= cy + obstacle["width"] / 2 + margin)
            & (z > 0.25)
        )
        keep_prob[in_hidden_obstacle] *= obstacle.get("observed_keep_multiplier", 0.02)

    near_obstacle = z > 0.25
    keep_prob[near_obstacle & behind_pillar] *= 2.5
    keep_prob = np.clip(keep_prob, 0.0, 0.9)
    return points_full[rng.random(points_full.shape[0]) < keep_prob]


def _world_box_mask(grid_cfg: dict, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    shape = grid_shape(grid_cfg)
    xs = grid_cfg["x_min"] + (np.arange(shape[0]) + 0.5) * grid_cfg["resolution"]
    ys = grid_cfg["y_min"] + (np.arange(shape[1]) + 0.5) * grid_cfg["resolution"]
    return ((xs[:, None] >= x_min) & (xs[:, None] <= x_max) & (ys[None, :] >= y_min) & (ys[None, :] <= y_max))


def create_synthetic_parking_scene(cfg: dict) -> SyntheticSample:
    """
    Create deterministic synthetic parking scene with observed data, ground truth, and slots.
    """
    seed = cfg.get("random_seed", 42)
    rng = np.random.default_rng(seed)
    grid_cfg = cfg["grid"]
    scene_cfg = cfg["synthetic_scene"]
    noise_std = scene_cfg["noise_std"] if scene_cfg.get("add_noise", True) else 0.0

    ground = sample_ground_points(rng, grid_cfg, scene_cfg["num_ground_points_full"], noise_std)
    vehicles = [
        sample_box_points(
            rng,
            tuple(vehicle["center"]),
            vehicle["length"],
            vehicle["width"],
            vehicle["height"],
            scene_cfg["num_vehicle_points_per_vehicle"],
            noise_std,
        )
        for vehicle in scene_cfg["vehicles"]
    ]
    pillar_cfg = scene_cfg["pillar"]
    pillar = sample_cylinder_points(
        rng,
        tuple(pillar_cfg["center"]),
        pillar_cfg["radius"],
        pillar_cfg["height"],
        scene_cfg["num_pillar_points"],
        noise_std,
    )
    hidden_obstacles_cfg = scene_cfg.get("hidden_obstacles", [])
    hidden_obstacles = [
        sample_box_points(
            rng,
            tuple(obstacle["center"]),
            obstacle["length"],
            obstacle["width"],
            obstacle["height"],
            obstacle["points"],
            noise_std,
        )
        for obstacle in hidden_obstacles_cfg
    ]
    lidar_points_full = np.vstack([ground, *vehicles, pillar, *hidden_obstacles]).astype(np.float32)
    lidar_points_observed = degrade_lidar_observation(
        rng,
        lidar_points_full,
        grid_cfg,
        tuple(pillar_cfg["center"]),
        hidden_obstacles_cfg,
    )

    preferred_slot_cfg = scene_cfg.get("target_slot", {})
    fake_slots = create_parking_slots(
        grid_cfg,
        target_slot_id=None,
        preferred_slot_id=preferred_slot_cfg.get("slot_id"),
        preference_bonus=preferred_slot_cfg.get("preference_bonus", 0.0),
    )
    gt_occupancy, _ = build_occupancy_from_lidar(lidar_points_full, grid_cfg)
    gt_occupancy[gt_occupancy != OCCUPIED] = FREE
    for vehicle in scene_cfg["vehicles"]:
        cx, cy = vehicle["center"]
        vehicle_mask = _world_box_mask(
            grid_cfg,
            cx - vehicle["length"] / 2,
            cx + vehicle["length"] / 2,
            cy - vehicle["width"] / 2,
            cy + vehicle["width"] / 2,
        )
        gt_occupancy[vehicle_mask] = OCCUPIED
    pillar_mask = _world_box_mask(
        grid_cfg,
        pillar_cfg["center"][0] - pillar_cfg["radius"],
        pillar_cfg["center"][0] + pillar_cfg["radius"],
        pillar_cfg["center"][1] - pillar_cfg["radius"],
        pillar_cfg["center"][1] + pillar_cfg["radius"],
    )
    gt_occupancy[pillar_mask] = OCCUPIED
    for obstacle in hidden_obstacles_cfg:
        cx, cy = obstacle["center"]
        obstacle_mask = _world_box_mask(
            grid_cfg,
            cx - obstacle["length"] / 2,
            cx + obstacle["length"] / 2,
            cy - obstacle["width"] / 2,
            cy + obstacle["width"] / 2,
        )
        gt_occupancy[obstacle_mask] = OCCUPIED

    shape = grid_shape(grid_cfg)
    gt_risk_zones = np.zeros(shape, dtype=np.float32)
    gt_occlusion_zones = np.zeros(shape, dtype=np.float32)
    target_slot_mask = np.zeros(shape, dtype=np.float32)

    mark_world_box(gt_risk_zones, grid_cfg, 0.0, 25.0, -3.0, 3.0, 0.35)
    mark_world_box(gt_risk_zones, grid_cfg, 18.0, 34.0, -4.0, 2.0, 0.75)
    mark_world_box(gt_occlusion_zones, grid_cfg, 18.0, 34.0, -4.0, 2.0, 1.0)

    for slot in fake_slots:
        x0, y0 = slot.polygon_xy.min(axis=0)
        x1, y1 = slot.polygon_xy.max(axis=0)
        mark_world_box(gt_risk_zones, grid_cfg, float(x0), float(x1), float(y0), float(y1), 0.45)
        configured_target = slot.slot_id == scene_cfg["target_slot"]["slot_id"]
        if configured_target:
            mark_world_box(gt_risk_zones, grid_cfg, float(x0), float(x1), float(y0), float(y1), 1.0)
            mark_world_box(gt_risk_zones, grid_cfg, 17.5, 22.5, -6.8, -5.2, 1.0)
            mark_world_box(target_slot_mask, grid_cfg, float(x0), float(x1), float(y0), float(y1), 1.0)

    return SyntheticSample(
        sample_id="vibe_demo_000",
        front_images=create_front_camera_placeholders(),
        lidar_points_full=lidar_points_full,
        lidar_points_observed=lidar_points_observed,
        fake_slots=fake_slots,
        gt_occupancy=gt_occupancy,
        gt_risk_zones=gt_risk_zones,
        gt_occlusion_zones=gt_occlusion_zones,
        target_slot_mask=target_slot_mask,
        metadata={"vehicles": scene_cfg["vehicles"], "pillar": pillar_cfg, "hidden_obstacles": hidden_obstacles_cfg},
    )
