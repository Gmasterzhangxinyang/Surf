#!/usr/bin/env python3
"""Render the full pose trajectory on the semantic GLTF parking map.

This uses dataset poses as the localization source. The pose-world trajectory
is mapped to the GLTF map frame with a documented 2D similarity transform for
visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}") from exc

try:
    from PIL import Image

    HAS_PIL = True
except Exception:
    HAS_PIL = False

from gltf_lidar_ndt import draw_gltf_map, load_gltf_map, normalize_angle, yaw_to_rot


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
MATCH_FILE = Path("/home/ParkingAgent/dataset/dataset/matched_frames.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw pose-truth route on the semantic GLTF map")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--match-file", type=Path, default=MATCH_FILE)
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pose_gltf_route"))
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=0, help="0 means all matched LiDAR frames")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--map-sample-step", type=float, default=0.06)
    parser.add_argument("--fit-margin", type=float, default=0.55)
    parser.add_argument("--html-step", type=int, default=20)
    parser.add_argument("--camera-width", type=int, default=900)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pose_rows(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    with path.open("r") as handle:
        for line in handle:
            vals = np.fromstring(line, sep=" ")
            if vals.size != 12:
                continue
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = vals.reshape(3, 4)
            poses.append(mat)
    return poses


def pose_to_xyyaw(mat: np.ndarray) -> np.ndarray:
    return np.array([mat[0, 3], mat[1, 3], math.atan2(float(mat[1, 0]), float(mat[0, 0]))], dtype=np.float64)


def load_matches(path: Path) -> dict[int, dict[str, int]]:
    matches: dict[int, dict[str, int]] = {}
    with path.open("r") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = [int(x) for x in line.split()]
            if len(fields) < 19:
                continue
            lidar_idx = fields[0]
            image_idx = fields[1] if fields[1] >= 0 else -1
            pose_candidates = [idx for idx in fields[14:19] if idx >= 0]
            if pose_candidates:
                matches[lidar_idx] = {"image": image_idx, "pose": pose_candidates[0]}
    return matches


def principal_yaw(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigval, eigvec = np.linalg.eigh(cov)
    axis = eigvec[:, int(np.argmax(eigval))]
    return math.atan2(float(axis[1]), float(axis[0]))


def transform_points(points: np.ndarray, scale: float, yaw: float, trans: np.ndarray) -> np.ndarray:
    return scale * (points @ yaw_to_rot(yaw).T) + trans


def fit_pose_to_gltf(points_world: np.ndarray, map_min: np.ndarray, map_max: np.ndarray, margin: float) -> dict:
    usable_min = map_min + margin
    usable_max = map_max - margin
    if np.any(usable_max <= usable_min):
        usable_min = map_min
        usable_max = map_max
    dst_span = usable_max - usable_min

    src_pca = principal_yaw(points_world)
    candidates = [
        -src_pca,
        -src_pca + math.pi,
        -src_pca + math.pi / 2.0,
        -src_pca - math.pi / 2.0,
        0.0,
        math.pi / 2.0,
        math.pi,
        -math.pi / 2.0,
    ]

    best: Optional[dict] = None
    for yaw in candidates:
        rotated = points_world @ yaw_to_rot(yaw).T
        rot_min = rotated.min(axis=0)
        rot_max = rotated.max(axis=0)
        span = rot_max - rot_min
        scale = float(min(dst_span[0] / max(span[0], 1e-6), dst_span[1] / max(span[1], 1e-6)))
        fitted_span = scale * span
        trans = usable_min + (dst_span - fitted_span) * 0.5 - scale * rot_min
        mapped = scale * rotated + trans
        inside = np.all((mapped >= map_min - 1e-9) & (mapped <= map_max + 1e-9), axis=1)
        area_fill = float(np.prod((mapped.max(axis=0) - mapped.min(axis=0)) / np.maximum(map_max - map_min, 1e-6)))
        # Prefer transforms that use more map area while keeping the whole route inside.
        score = float(inside.mean() * 10.0 + area_fill)
        item = {
            "scale": scale,
            "yaw": normalize_angle(float(yaw)),
            "translation": trans.tolist(),
            "inside_ratio": float(inside.mean()),
            "area_fill": area_fill,
            "score": score,
        }
        if best is None or score > best["score"]:
            best = item

    assert best is not None
    return best


def map_poses(poses_world: np.ndarray, alignment: dict) -> np.ndarray:
    scale = float(alignment["scale"])
    yaw = float(alignment["yaw"])
    trans = np.asarray(alignment["translation"], dtype=np.float64)
    out = np.zeros_like(poses_world)
    out[:, :2] = transform_points(poses_world[:, :2], scale, yaw, trans)
    out[:, 2] = [normalize_angle(float(v + yaw)) for v in poses_world[:, 2]]
    return out


def image_path_for(dataset_root: Path, image_idx: int) -> Optional[Path]:
    if image_idx < 0:
        return None
    candidates = [
        dataset_root / "image" / f"left{image_idx:06d}.png",
        dataset_root / "image" / f"{image_idx:06d}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def save_camera_copy(src: Optional[Path], dst: Path, width: int) -> Optional[str]:
    if src is None or not src.exists():
        return None
    ensure_dir(dst.parent)
    if HAS_PIL:
        img = Image.open(src)
        if width > 0 and img.width > width:
            h = int(round(img.height * width / img.width))
            img = img.resize((width, h))
        img.convert("RGB").save(dst, quality=90)
    else:
        shutil.copyfile(src, dst)
    return dst.name


def car_outline(pose: np.ndarray, length: float = 0.95, width: float = 0.42) -> np.ndarray:
    pts = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]], dtype=np.float64)
    return pts @ yaw_to_rot(float(pose[2])).T + pose[:2]


def draw_car(ax: plt.Axes, pose: np.ndarray, color: str, label: str) -> None:
    body = car_outline(pose)
    closed = np.vstack([body, body[0]])
    ax.fill(body[:, 0], body[:, 1], color=color, alpha=0.28, zorder=9)
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.7, zorder=10)
    heading = np.array([math.cos(pose[2]), math.sin(pose[2])])
    ax.arrow(pose[0], pose[1], heading[0] * 0.8, heading[1] * 0.8, color=color, width=0.025, zorder=11)
    ax.scatter([pose[0]], [pose[1]], s=28, c=color, zorder=12, label=label)


def render_static_route(output: Path, gltf_map, mapped: np.ndarray, frame_ids: Sequence[int]) -> None:
    fig, ax = plt.subplots(figsize=(11, 13), dpi=240)
    draw_gltf_map(ax, gltf_map)
    ax.plot(mapped[:, 0], mapped[:, 1], color="#ef4444", linewidth=2.2, alpha=0.92, zorder=8, label="pose trajectory")
    ax.scatter(mapped[:: max(1, len(mapped) // 180), 0], mapped[:: max(1, len(mapped) // 180), 1], s=3, c="#7f1d1d", alpha=0.5, zorder=8)
    draw_car(ax, mapped[0], "#16a34a", f"start {frame_ids[0]:06d}")
    draw_car(ax, mapped[-1], "#dc2626", f"end {frame_ids[-1]:06d}")
    ax.set_title("Full Pose Route on Semantic GLTF Parking Map")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def render_map_background(output: Path, gltf_map) -> None:
    fig, ax = plt.subplots(figsize=(11, 13), dpi=180)
    draw_gltf_map(ax, gltf_map)
    ax.set_title("Semantic GLTF Parking Map")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_html(output: Path, rows: list[dict], map_bounds: dict, start_photo: Optional[str], end_photo: Optional[str]) -> None:
    payload = json.dumps(rows, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>Pose Route on GLTF Map</title>
<style>
body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background:#f7f4ec; color:#111827; }}
.wrap {{ display:grid; grid-template-columns: minmax(360px, 1fr) 420px; gap:16px; padding:16px; }}
.panel {{ background:white; border:1px solid #d6d3ca; border-radius:12px; padding:14px; box-shadow:0 10px 24px rgba(17,24,39,.08); }}
canvas {{ width:100%; height:auto; background:#fbfaf6; border-radius:10px; }}
button,input {{ font:inherit; }}
.photos img {{ max-width:100%; border-radius:8px; border:1px solid #ddd6c8; margin-top:8px; }}
.metric {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px; white-space:pre-wrap; }}
@media (max-width: 900px) {{ .wrap {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="panel">
    <canvas id="map" width="1100" height="1300"></canvas>
    <div style="display:flex; gap:10px; align-items:center; margin-top:10px;">
      <button id="play">Play</button>
      <input id="slider" type="range" min="0" max="0" value="0" style="flex:1" />
      <select id="speed"><option value="1">1x</option><option value="3" selected>3x</option><option value="8">8x</option></select>
    </div>
  </div>
  <div class="panel">
    <h2>Pose 真值路线</h2>
    <div id="info" class="metric"></div>
    <div class="photos">
      <h3>起点照片</h3>
      {'<img src="' + start_photo + '">' if start_photo else '<p>missing</p>'}
      <h3>终点照片</h3>
      {'<img src="' + end_photo + '">' if end_photo else '<p>missing</p>'}
    </div>
  </div>
</div>
<script>
const frames = {payload};
const bounds = {json.dumps(map_bounds)};
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('slider');
const play = document.getElementById('play');
const speed = document.getElementById('speed');
const info = document.getElementById('info');
slider.max = String(frames.length - 1);
let idx = 0, timer = null;
function sx(x) {{ return (x - bounds.minX) / (bounds.maxX - bounds.minX) * canvas.width; }}
function sy(y) {{ return canvas.height - (y - bounds.minY) / (bounds.maxY - bounds.minY) * canvas.height; }}
function drawCar(f) {{
  const x = sx(f.map_x), y = sy(f.map_y), yaw = -f.map_yaw;
  ctx.save(); ctx.translate(x,y); ctx.rotate(yaw);
  ctx.fillStyle = 'rgba(220,38,38,.35)'; ctx.strokeStyle = '#991b1b'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.rect(-34,-15,68,30); ctx.fill(); ctx.stroke();
  ctx.strokeStyle = '#dc2626'; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(54,0); ctx.stroke();
  ctx.restore();
}}
function draw() {{
  const f = frames[idx];
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = '#fbfaf6'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = '#d0c8b8'; ctx.lineWidth = 1;
  for (let x=Math.floor(bounds.minX); x<=Math.ceil(bounds.maxX); x++) {{ ctx.beginPath(); ctx.moveTo(sx(x),0); ctx.lineTo(sx(x),canvas.height); ctx.stroke(); }}
  for (let y=Math.floor(bounds.minY); y<=Math.ceil(bounds.maxY); y++) {{ ctx.beginPath(); ctx.moveTo(0,sy(y)); ctx.lineTo(canvas.width,sy(y)); ctx.stroke(); }}
  ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 4; ctx.beginPath();
  for (let i=0; i<=idx; i++) {{
    const px = sx(frames[i].map_x), py = sy(frames[i].map_y);
    if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();
  ctx.fillStyle = '#16a34a'; ctx.beginPath(); ctx.arc(sx(frames[0].map_x), sy(frames[0].map_y), 8, 0, Math.PI*2); ctx.fill();
  drawCar(f);
  info.textContent = `frame: ${{f.frame}}\\npose index: ${{f.pose_index}}\\nmap x/y/yaw: ${{f.map_x.toFixed(3)}}, ${{f.map_y.toFixed(3)}}, ${{f.map_yaw.toFixed(3)}}\\nworld x/y/yaw: ${{f.world_x.toFixed(3)}}, ${{f.world_y.toFixed(3)}}, ${{f.world_yaw.toFixed(3)}}`;
}}
slider.oninput = () => {{ idx = Number(slider.value); draw(); }};
play.onclick = () => {{
  if (timer) {{ clearInterval(timer); timer=null; play.textContent='Play'; return; }}
  play.textContent='Pause';
  timer = setInterval(() => {{
    idx = Math.min(frames.length - 1, idx + Number(speed.value));
    slider.value = String(idx); draw();
    if (idx >= frames.length - 1) {{ clearInterval(timer); timer=null; play.textContent='Play'; }}
  }}, 40);
}};
draw();
</script>
</body>
</html>"""
    output.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    gltf_map = load_gltf_map(args.gltf, args.map_sample_step)
    poses = load_pose_rows(args.dataset_root / "pose" / "poses.txt")
    matches = load_matches(args.match_file)

    frame_ids = sorted(k for k in matches if k >= args.start_frame)
    if args.num_frames > 0:
        frame_ids = frame_ids[: args.num_frames * args.step]
    frame_ids = frame_ids[:: args.step]
    if not frame_ids:
        raise SystemExit("No matched frames selected")

    world_rows: list[np.ndarray] = []
    pose_indices: list[int] = []
    image_indices: list[int] = []
    for frame_id in frame_ids:
        pose_idx = matches[frame_id]["pose"]
        if pose_idx >= len(poses):
            continue
        world_rows.append(pose_to_xyyaw(poses[pose_idx]))
        pose_indices.append(pose_idx)
        image_indices.append(matches[frame_id]["image"])
    world = np.asarray(world_rows, dtype=np.float64)
    frame_ids = frame_ids[: len(world)]

    alignment = fit_pose_to_gltf(world[:, :2], gltf_map.bounds_min, gltf_map.bounds_max, args.fit_margin)
    mapped = map_poses(world, alignment)

    alignment_out = {
        "method": "pose_world_to_gltf_similarity_fit_for_visualization",
        "warning": "No dataset-provided world->GLTF extrinsic was found. This transform fits the pose trajectory into the GLTF map for visualization.",
        **alignment,
    }
    (args.output_dir / "alignment.json").write_text(json.dumps(alignment_out, indent=2), encoding="utf-8")

    rows: list[dict] = []
    csv_path = args.output_dir / "pose_gltf_route.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = ["frame", "pose_index", "image_index", "world_x", "world_y", "world_yaw", "map_x", "map_y", "map_yaw"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_id, pose_idx, image_idx, w, m in zip(frame_ids, pose_indices, image_indices, world, mapped):
            row = {
                "frame": int(frame_id),
                "pose_index": int(pose_idx),
                "image_index": int(image_idx),
                "world_x": float(w[0]),
                "world_y": float(w[1]),
                "world_yaw": float(w[2]),
                "map_x": float(m[0]),
                "map_y": float(m[1]),
                "map_yaw": float(m[2]),
            }
            writer.writerow(row)
            if len(rows) % max(1, args.html_step // max(1, args.step)) == 0:
                rows.append(row)

    render_static_route(args.output_dir / "pose_route_on_gltf.png", gltf_map, mapped, frame_ids)
    render_map_background(args.output_dir / "gltf_map_background.png", gltf_map)

    start_photo = save_camera_copy(image_path_for(args.dataset_root, image_indices[0]), args.output_dir / "start_photo.jpg", args.camera_width)
    end_photo = save_camera_copy(image_path_for(args.dataset_root, image_indices[-1]), args.output_dir / "end_photo.jpg", args.camera_width)
    map_bounds = {
        "minX": float(gltf_map.bounds_min[0]),
        "minY": float(gltf_map.bounds_min[1]),
        "maxX": float(gltf_map.bounds_max[0]),
        "maxY": float(gltf_map.bounds_max[1]),
    }
    write_html(args.output_dir / "route_player.html", rows, map_bounds, start_photo, end_photo)

    summary = {
        "frames": len(frame_ids),
        "start_frame": int(frame_ids[0]),
        "end_frame": int(frame_ids[-1]),
        "world_bounds_min": world[:, :2].min(axis=0).tolist(),
        "world_bounds_max": world[:, :2].max(axis=0).tolist(),
        "map_bounds_min": gltf_map.bounds_min.tolist(),
        "map_bounds_max": gltf_map.bounds_max.tolist(),
        "alignment": alignment_out,
        "outputs": {
            "route_png": str(args.output_dir / "pose_route_on_gltf.png"),
            "route_csv": str(csv_path),
            "route_html": str(args.output_dir / "route_player.html"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[route] {args.output_dir / 'pose_route_on_gltf.png'}")
    print(f"[html] {args.output_dir / 'route_player.html'}")
    print(f"[csv] {csv_path}")
    print(f"[alignment] {args.output_dir / 'alignment.json'}")
    print(f"[summary] {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
