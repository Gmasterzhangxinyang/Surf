#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image, ImageDraw


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
MATCH_FILE = Path("/home/ParkingAgent/dataset/dataset/matched_frames.txt")


@dataclass
class MatchedFrame:
    lidar_idx: int
    image_idx: int
    pose_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a 2D live-localization HTML player")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--match-file", type=Path, default=MATCH_FILE)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=11678)
    parser.add_argument("--step", type=int, default=20, help="camera/playback frame sampling")
    parser.add_argument("--map-accumulate-every", type=int, default=20)
    parser.add_argument("--range-max", type=float, default=45.0)
    parser.add_argument("--z-min", type=float, default=-2.2)
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--max-points-per-frame", type=int, default=10000)
    parser.add_argument("--max-camera-width", type=int, default=720)
    parser.add_argument("--map-width", type=int, default=1200)
    parser.add_argument("--map-height", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pose_truth_html_player"))
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def yaw_from_pose(mat: np.ndarray) -> float:
    return math.atan2(float(mat[1, 0]), float(mat[0, 0]))


def load_pose_rows(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    mats = np.tile(np.eye(4, dtype=np.float64), (poses.shape[0], 1, 1))
    mats[:, :3, :4] = poses.reshape(-1, 3, 4)
    return mats


def load_matched_frames(path: Path) -> List[MatchedFrame]:
    frames: List[MatchedFrame] = []
    with path.open("r") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = [int(x) for x in line.split()]
            if len(fields) < 19:
                continue
            lidar_idx = fields[0]
            if lidar_idx < 0:
                continue
            image_idx = next((idx for idx in fields[1:4] if idx >= 0), -1)
            pose_idx = next((idx for idx in fields[14:19] if idx >= 0), -1)
            if image_idx >= 0 and pose_idx >= 0:
                frames.append(MatchedFrame(lidar_idx=lidar_idx, image_idx=image_idx, pose_idx=pose_idx))
    return frames


def load_lidar_points(path: Path, z_min: float, z_max: float, range_max: float, max_points: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raw = raw[: raw.size - raw.size % 4]
    pts = raw.reshape(-1, 4)[:, :3].astype(np.float64)
    mask = np.isfinite(pts).all(axis=1)
    mask &= pts[:, 2] >= z_min
    mask &= pts[:, 2] <= z_max
    mask &= np.linalg.norm(pts[:, :2], axis=1) <= range_max
    pts = pts[mask]
    if len(pts) > max_points:
        pts = pts[:: int(math.ceil(len(pts) / max_points))]
    return pts


def transform_points(mat: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    return (mat[:3, :3] @ points_xyz.T).T + mat[:3, 3]


def calculate_view_bounds(points_xy: np.ndarray, traj_xy: np.ndarray, width: int, height: int) -> dict:
    min_xy = np.minimum(points_xy.min(axis=0), traj_xy.min(axis=0)) if len(points_xy) else traj_xy.min(axis=0)
    max_xy = np.maximum(points_xy.max(axis=0), traj_xy.max(axis=0)) if len(points_xy) else traj_xy.max(axis=0)
    span = max_xy - min_xy
    pad = max(8.0, float(max(span)) * 0.05)
    min_xy = min_xy - pad
    max_xy = max_xy + pad

    world_w = float(max_xy[0] - min_xy[0])
    world_h = float(max_xy[1] - min_xy[1])
    target_aspect = width / height
    world_aspect = world_w / world_h
    if world_aspect > target_aspect:
        new_h = world_w / target_aspect
        extra = (new_h - world_h) * 0.5
        min_xy[1] -= extra
        max_xy[1] += extra
    else:
        new_w = world_h * target_aspect
        extra = (new_w - world_w) * 0.5
        min_xy[0] -= extra
        max_xy[0] += extra

    return {
        "minX": float(min_xy[0]),
        "maxX": float(max_xy[0]),
        "minY": float(min_xy[1]),
        "maxY": float(max_xy[1]),
        "width": int(width),
        "height": int(height),
    }


def world_to_pixel(points_xy: np.ndarray, bounds: dict) -> np.ndarray:
    x = (points_xy[:, 0] - bounds["minX"]) / (bounds["maxX"] - bounds["minX"]) * bounds["width"]
    y = (bounds["maxY"] - points_xy[:, 1]) / (bounds["maxY"] - bounds["minY"]) * bounds["height"]
    return np.column_stack([x, y])


def render_map_background(path: Path, points_xy: np.ndarray, bounds: dict) -> None:
    width = bounds["width"]
    height = bounds["height"]
    counts = np.zeros((height, width), dtype=np.uint16)
    if len(points_xy):
        px = world_to_pixel(points_xy, bounds).astype(np.int32)
        valid = (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
        px = px[valid]
        np.add.at(counts, (px[:, 1], px[:, 0]), 1)

    if counts.max() > 0:
        density = np.log1p(counts.astype(np.float32))
        density = density / density.max()
        base = (246 - density * 112).astype(np.uint8)
    else:
        base = np.full((height, width), 246, dtype=np.uint8)

    rgb = np.dstack([base, base, base])
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    grid_step = 20.0
    x0 = math.floor(bounds["minX"] / grid_step) * grid_step
    x = x0
    while x <= bounds["maxX"]:
        px = world_to_pixel(np.array([[x, bounds["minY"]]], dtype=np.float64), bounds)[0, 0]
        draw.line([(px, 0), (px, height)], fill=(120, 132, 150, 65), width=1)
        x += grid_step
    y0 = math.floor(bounds["minY"] / grid_step) * grid_step
    y = y0
    while y <= bounds["maxY"]:
        py = world_to_pixel(np.array([[bounds["minX"], y]], dtype=np.float64), bounds)[0, 1]
        draw.line([(0, py), (width, py)], fill=(120, 132, 150, 65), width=1)
        y += grid_step

    ensure_dir(path.parent)
    image.save(path)


def export_camera_images(frames: Sequence[MatchedFrame], dataset_root: Path, asset_dir: Path, max_width: int) -> List[dict]:
    ensure_dir(asset_dir)
    image_dir = dataset_root / "image"
    exported = []
    for frame in frames:
        src = image_dir / f"left{frame.image_idx:06d}.png"
        rel = f"assets/camera_{frame.lidar_idx:06d}.jpg"
        dst = asset_dir / f"camera_{frame.lidar_idx:06d}.jpg"
        if src.exists() and not dst.exists():
            image = Image.open(src).convert("RGB")
            if image.width > max_width:
                scale = max_width / image.width
                image = image.resize((max_width, int(image.height * scale)), Image.Resampling.LANCZOS)
            image.save(dst, quality=82, optimize=True)
        exported.append({"lidar_idx": frame.lidar_idx, "image": rel})
    return exported


def build_html(path: Path, data: dict) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>2D Vehicle Localization Player</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #637083;
      --line: #d8dee8;
      --panel: #f7f8f6;
      --accent: #d97706;
      --car: #dc2626;
      --start: #16a34a;
      --paper: #fffefa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #e9ece7;
      color: var(--ink);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(520px, 1.35fr) minmax(360px, 0.9fr);
      gap: 14px;
      padding: 14px;
    }}
    .stage, .side {{
      background: var(--paper);
      border: 1px solid #cfd6df;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 14px 28px rgba(20, 30, 40, 0.08);
    }}
    .stage {{
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: calc(100vh - 28px);
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #f5f2ea;
    }}
    h1 {{
      margin: 0;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    .metric {{
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
    }}
    .metric strong {{
      display: block;
      color: var(--ink);
      font-size: 13px;
      margin-top: 2px;
      font-variant-numeric: tabular-nums;
    }}
    .map-wrap {{
      position: relative;
      background: #dfe4df;
      min-height: 0;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto 1fr auto auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      background: #f8f8f4;
    }}
    button {{
      width: 42px;
      height: 34px;
      border: 1px solid #b8c0ca;
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      font-size: 15px;
      cursor: pointer;
    }}
    input[type="range"] {{ width: 100%; }}
    select {{
      height: 34px;
      border: 1px solid #b8c0ca;
      border-radius: 6px;
      background: #fff;
    }}
    .side {{
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: calc(100vh - 28px);
    }}
    .camera {{
      background: #111827;
      display: grid;
      place-items: center;
      min-height: 0;
    }}
    .camera img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }}
    .legend {{
      display: grid;
      gap: 8px;
      padding: 12px;
      border-top: 1px solid var(--line);
      font-size: 13px;
      color: var(--muted);
    }}
    .legend span {{
      display: inline-block;
      width: 24px;
      height: 3px;
      margin-right: 8px;
      vertical-align: middle;
    }}
    @media (max-width: 980px) {{
      .app {{ grid-template-columns: 1fr; }}
      .stage, .side {{ min-height: 560px; }}
    }}
  </style>
</head>
<body>
  <main class="app">
    <section class="stage">
      <header>
        <h1>2D Map Localization</h1>
        <div id="frameLabel"></div>
      </header>
      <div class="map-wrap">
        <canvas id="mapCanvas"></canvas>
      </div>
      <div class="controls">
        <button id="playBtn" title="play/pause">▶</button>
        <input id="timeline" type="range" min="0" max="0" value="0" />
        <select id="speed">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
        <button id="resetBtn" title="reset">↺</button>
      </div>
    </section>
    <aside class="side">
      <header>
        <h1>Camera</h1>
      </header>
      <div class="camera">
        <img id="cameraImage" alt="camera frame" />
      </div>
      <div class="legend">
        <div class="metrics">
          <div class="metric">LiDAR<strong id="lidarValue"></strong></div>
          <div class="metric">Image<strong id="imageValue"></strong></div>
          <div class="metric">X / Y<strong id="xyValue"></strong></div>
          <div class="metric">Yaw<strong id="yawValue"></strong></div>
        </div>
        <div><span style="background: var(--accent)"></span>completed trajectory</div>
        <div><span style="background: var(--car)"></span>current vehicle pose</div>
        <div><span style="background: var(--start)"></span>start position</div>
      </div>
    </aside>
  </main>

  <script>
    const DATA = {json.dumps(data, separators=(",", ":"))};
    const canvas = document.getElementById("mapCanvas");
    const ctx = canvas.getContext("2d");
    const bg = new Image();
    bg.src = DATA.mapImage;
    const cameraImage = document.getElementById("cameraImage");
    const timeline = document.getElementById("timeline");
    const playBtn = document.getElementById("playBtn");
    const resetBtn = document.getElementById("resetBtn");
    const speed = document.getElementById("speed");
    const frameLabel = document.getElementById("frameLabel");
    const lidarValue = document.getElementById("lidarValue");
    const imageValue = document.getElementById("imageValue");
    const xyValue = document.getElementById("xyValue");
    const yawValue = document.getElementById("yawValue");

    let index = 0;
    let playing = false;
    let lastTick = 0;
    timeline.max = String(DATA.frames.length - 1);

    function worldToCanvas(x, y) {{
      const b = DATA.bounds;
      return [
        ((x - b.minX) / (b.maxX - b.minX)) * canvas.width,
        ((b.maxY - y) / (b.maxY - b.minY)) * canvas.height
      ];
    }}

    function resizeCanvas() {{
      const rect = canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      render();
    }}

    function drawPolyline(points, color, width, alpha) {{
      if (points.length < 2) return;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      let p = worldToCanvas(points[0][0], points[0][1]);
      ctx.moveTo(p[0], p[1]);
      for (let i = 1; i < points.length; i++) {{
        p = worldToCanvas(points[i][0], points[i][1]);
        ctx.lineTo(p[0], p[1]);
      }}
      ctx.stroke();
      ctx.restore();
    }}

    function drawCar(frame) {{
      const p = worldToCanvas(frame.x, frame.y);
      const b = DATA.bounds;
      const metersToPx = canvas.width / (b.maxX - b.minX);
      const length = 4.6 * metersToPx;
      const width = 1.9 * metersToPx;
      ctx.save();
      ctx.translate(p[0], p[1]);
      ctx.rotate(-frame.yaw);
      ctx.fillStyle = "rgba(220,38,38,0.38)";
      ctx.strokeStyle = "#991b1b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.rect(-length / 2, -width / 2, length, width);
      ctx.fill();
      ctx.stroke();
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(length * 0.75, 0);
      ctx.stroke();
      ctx.restore();
    }}

    function render() {{
      if (!canvas.width || !canvas.height) return;
      const frame = DATA.frames[index];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (bg.complete) ctx.drawImage(bg, 0, 0, canvas.width, canvas.height);
      drawPolyline(DATA.trajectory, "#475569", 1.2, 0.30);
      drawPolyline(DATA.frames.slice(0, index + 1).map(f => [f.x, f.y]), "#d97706", 3.2, 0.95);
      const start = DATA.frames[0];
      const startPt = worldToCanvas(start.x, start.y);
      ctx.fillStyle = "#16a34a";
      ctx.beginPath();
      ctx.arc(startPt[0], startPt[1], 7, 0, Math.PI * 2);
      ctx.fill();
      drawCar(frame);

      cameraImage.src = frame.image;
      timeline.value = String(index);
      frameLabel.textContent = `${{index + 1}} / ${{DATA.frames.length}}`;
      lidarValue.textContent = String(frame.lidar_idx).padStart(6, "0");
      imageValue.textContent = String(frame.image_idx).padStart(6, "0");
      xyValue.textContent = `${{frame.x.toFixed(2)}}, ${{frame.y.toFixed(2)}}`;
      yawValue.textContent = `${{frame.yaw.toFixed(3)}} rad`;
    }}

    function tick(ts) {{
      if (!playing) return;
      const interval = 120 / Number(speed.value);
      if (!lastTick || ts - lastTick >= interval) {{
        index = (index + 1) % DATA.frames.length;
        render();
        lastTick = ts;
      }}
      requestAnimationFrame(tick);
    }}

    playBtn.addEventListener("click", () => {{
      playing = !playing;
      playBtn.textContent = playing ? "Ⅱ" : "▶";
      lastTick = 0;
      if (playing) requestAnimationFrame(tick);
    }});
    resetBtn.addEventListener("click", () => {{
      index = 0;
      render();
    }});
    timeline.addEventListener("input", () => {{
      index = Number(timeline.value);
      render();
    }});
    window.addEventListener("resize", resizeCanvas);
    bg.onload = resizeCanvas;
    resizeCanvas();
  </script>
</body>
</html>
"""
    ensure_dir(path.parent)
    path.write_text(html)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    asset_dir = args.output_dir / "assets"
    ensure_dir(asset_dir)

    pose_rows = load_pose_rows(args.dataset_root / "pose" / "poses.txt")
    matched = load_matched_frames(args.match_file)
    selected_all = [frame for frame in matched if frame.lidar_idx >= args.start_index]
    if args.num_frames > 0:
        selected_all = selected_all[: args.num_frames]
    playback_frames = selected_all[:: max(1, args.step)]

    lidar_dir = args.dataset_root / "velodyne"
    traj = []
    map_parts = []
    for i, frame in enumerate(selected_all):
        pose = pose_rows[frame.pose_idx]
        traj.append([float(pose[0, 3]), float(pose[1, 3]), float(yaw_from_pose(pose))])
        if i % max(1, args.map_accumulate_every) == 0:
            lidar_path = lidar_dir / f"{frame.lidar_idx:06d}.bin"
            if lidar_path.exists():
                pts = load_lidar_points(lidar_path, args.z_min, args.z_max, args.range_max, args.max_points_per_frame)
                world = transform_points(pose, pts)
                map_parts.append(world[:, :2])

    traj_arr = np.asarray(traj, dtype=np.float64)
    map_xy = np.vstack(map_parts) if map_parts else np.empty((0, 2), dtype=np.float64)
    bounds = calculate_view_bounds(map_xy, traj_arr[:, :2], args.map_width, args.map_height)
    map_image_path = args.output_dir / "map_background.png"
    render_map_background(map_image_path, map_xy, bounds)

    exported = export_camera_images(playback_frames, args.dataset_root, asset_dir, args.max_camera_width)
    image_by_lidar = {item["lidar_idx"]: item["image"] for item in exported}
    frames = []
    for frame in playback_frames:
        pose = pose_rows[frame.pose_idx]
        frames.append(
            {
                "lidar_idx": frame.lidar_idx,
                "image_idx": frame.image_idx,
                "pose_idx": frame.pose_idx,
                "x": float(pose[0, 3]),
                "y": float(pose[1, 3]),
                "yaw": float(yaw_from_pose(pose)),
                "image": image_by_lidar[frame.lidar_idx],
            }
        )

    data = {
        "bounds": bounds,
        "mapImage": "map_background.png",
        "trajectory": [[float(x), float(y)] for x, y in traj_arr[:, :2]],
        "frames": frames,
    }
    build_html(args.output_dir / "index.html", data)
    print(f"[done] wrote {len(frames)} playback frames to {args.output_dir / 'index.html'}")
    print(f"[source] full trajectory poses: {len(traj_arr)}")


if __name__ == "__main__":
    main()
