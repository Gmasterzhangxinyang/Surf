#!/usr/bin/env python3
"""Build a prediction-blind Camera-visible GT annotation pack for frame 6681."""
from __future__ import annotations

import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import shutil
from typing import Any

from PIL import Image, ImageDraw

from parking_slot_agent_v2.fov import evaluate_horizontal_projection_fov

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "Nature_ParkingAgent_实验报告_20260728"
EXPERIMENT = REPORT / "random_midroute_experiment"
OUTPUT = REPORT / "Camera可见子集人工GT_frame6681"
LOCAL_MAP = EXPERIMENT / "part1_w30k5/local_map.json"
FRAMES = EXPERIMENT / "part1_w30k5/local_frame_manifest.json"
EXPECTED_IDS = ("slot_0942", "slot_0943", "slot_0944", "slot_0945")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def project(frame: dict[str, Any], slot: dict[str, Any], scale: float):
    return evaluate_horizontal_projection_fov(
        ego_map_xy=[frame["map_x"], frame["map_y"]],
        ego_yaw_rad=frame["map_yaw"],
        target_polygon_map=slot["polygon_map"],
        target_center_map=slot["center_map"],
        map_units_per_meter=scale,
        camera_frame_id=frame["camera_frame"],
    )


def neutral_map_svg(local_map: dict[str, Any], frames: list[dict[str, Any]], target: dict[str, Any]) -> str:
    ego = local_map["anchor_pose"]["map_xy"]
    points = [tuple(ego), tuple(target["center_map"])]
    for row in frames:
        points.append((float(row["map_x"]), float(row["map_y"])))
    for point in target["polygon_map"]:
        points.append(tuple(point))
    min_x, max_x = min(x for x, _ in points) - 0.25, max(x for x, _ in points) + 0.25
    min_y, max_y = min(y for _, y in points) - 0.25, max(y for _, y in points) + 0.25
    width, height, pad = 1000, 720, 60
    sx = (width - 2 * pad) / max(max_x - min_x, 1e-6)
    sy = (height - 2 * pad) / max(max_y - min_y, 1e-6)
    scale = min(sx, sy)
    def px(point):
        x, y = point
        return pad + (x - min_x) * scale, height - pad - (y - min_y) * scale
    nearby = []
    tc = target["center_map"]
    for slot in local_map["slots"]:
        cx, cy = slot["center_map"]
        if math.hypot(cx - tc[0], cy - tc[1]) <= 0.65:
            nearby.append(slot)
    polygons = []
    for slot in nearby:
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(px, slot["polygon_map"]))
        if slot["slot_id"] == target["slot_id"]:
            polygons.append(f'<polygon points="{coords}" fill="#d946ef55" stroke="#f0abfc" stroke-width="5"/>')
        else:
            polygons.append(f'<polygon points="{coords}" fill="#33415533" stroke="#64748b" stroke-width="2"/>')
    trajectory = " ".join(f"{x:.1f},{y:.1f}" for x, y in (px((f["map_x"], f["map_y"])) for f in frames))
    ex, ey = px(ego)
    tx, ty = px(target["center_map"])
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="prediction blind map for {escape(target['slot_id'])}">
<rect width="100%" height="100%" fill="#07101c"/><polyline points="{trajectory}" fill="none" stroke="#38bdf8" stroke-width="5"/>
{''.join(polygons)}<circle cx="{ex:.1f}" cy="{ey:.1f}" r="12" fill="#38bdf8"/><line x1="{ex:.1f}" y1="{ey:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="#f0abfc" stroke-width="3" stroke-dasharray="9 8"/>
<text x="{ex+16:.1f}" y="{ey-10:.1f}" fill="#7dd3fc" font-size="22">EGO t0</text><text x="{tx+15:.1f}" y="{ty-12:.1f}" fill="#f5d0fe" font-size="24">TARGET {escape(target['slot_id'])}</text>
<text x="50" y="45" fill="#e2e8f0" font-size="24">Neutral map geometry · no Part1/Agent state</text></svg>'''


def overlay_guide(source: Path, destination: Path, result) -> dict[str, Any]:
    image = Image.open(source).convert("RGB")
    rows = result.details["sample_horizontal_projection"]
    projected = [float(row["pixel_u"]) for row in rows if row["pixel_u"] is not None]
    center = rows[0]["pixel_u"]
    left = max(0.0, min(projected)) if projected else 0.0
    right = min(float(image.width - 1), max(projected)) if projected else 0.0
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if right > left:
        draw.rectangle((left, 0, right, image.height - 1), fill=(217, 70, 239, 38), outline=(240, 171, 252, 210), width=3)
    if center is not None and 0 <= float(center) < image.width:
        draw.line((float(center), 0, float(center), image.height - 1), fill=(255, 255, 0, 230), width=4)
    combined = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    combined.save(destination, quality=94)
    return {
        "center_u": center,
        "band_left_u": round(left, 3),
        "band_right_u": round(right, 3),
        "visibility": result.visibility.value,
    }


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assets = OUTPUT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    local_map = read_json(LOCAL_MAP)
    frames = read_json(FRAMES)["frames"]
    slots = {row["slot_id"]: row for row in local_map["slots"]}
    scale = float(local_map["map_units_per_meter"])
    anchor = frames[-1]
    selected = []
    for slot in local_map["slots"]:
        result = project(anchor, slot, scale)
        if result.visibility.value in {"visible", "partially_visible"}:
            selected.append((slot, result))
    selected_ids = tuple(sorted(slot["slot_id"] for slot, _ in selected))
    if selected_ids != EXPECTED_IDS:
        raise RuntimeError(f"unexpected Camera-visible subset: {selected_ids}")

    raw_assets = []
    for frame in frames:
        source = Path(frame["camera_image_path"])
        destination = assets / f"raw_lidar_{frame['frame_id']:06d}_camera_{frame['camera_frame']:06d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        raw_assets.append(destination)

    cases = []
    for slot, anchor_result in selected:
        sid = slot["slot_id"]
        map_path = assets / f"{sid}_neutral_map.svg"
        map_path.write_text(neutral_map_svg(local_map, frames, slot), encoding="utf-8")
        views = []
        for frame, raw_path in zip(frames, raw_assets):
            result = project(frame, slot, scale)
            guide_path = assets / f"{sid}_guide_lidar_{frame['frame_id']:06d}_camera_{frame['camera_frame']:06d}.jpg"
            guide = overlay_guide(Path(frame["camera_image_path"]), guide_path, result)
            views.append({
                "lidar_frame": frame["frame_id"],
                "camera_frame": frame["camera_frame"],
                "camera_lidar_dt_sec": frame["camera_lidar_dt_sec"],
                "raw_image": raw_path.name,
                "guide_image": guide_path.name,
                **guide,
            })
        cases.append({
            "slot_id": sid,
            "anchor_visibility": anchor_result.visibility.value,
            "anchor_center_u": anchor_result.details["target_center_pixel_u"],
            "anchor_inside_fraction": anchor_result.details["horizontal_inside_fraction"],
            "neutral_map": map_path.name,
            "views": views,
        })

    csv_path = OUTPUT / "camera_visible_gt_blank.csv"
    fields = ["slot_id", "anchor_lidar_frame", "anchor_camera_frame", "identity_verified", "gt_state", "gt_camera_observability", "state_confidence", "annotator", "evidence_frames", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({"slot_id": case["slot_id"], "anchor_lidar_frame": 6681, "anchor_camera_frame": 20019})

    case_html = []
    for index, case in enumerate(cases, 1):
        galleries = []
        for view in case["views"]:
            galleries.append(f'''<article class="frame"><h4>LiDAR {view['lidar_frame']} / Camera {view['camera_frame']} · Δt={1000*view['camera_lidar_dt_sec']:.1f}ms</h4><div class="pair"><figure><img src="assets/{escape(view['raw_image'])}"><figcaption>原始Camera</figcaption></figure><figure><img src="assets/{escape(view['guide_image'])}"><figcaption>水平定位辅助：紫色范围+黄色中心线，不是车位框</figcaption></figure></div></article>''')
        sid = case["slot_id"]
        case_html.append(f'''<section class="case" data-slot="{sid}"><span class="eyebrow">Blind case {index}/4</span><h2>{sid}</h2><div class="facts"><div>t0中心像素 u=<b>{float(case['anchor_center_u']):.1f}</b></div><div>横向覆盖=<b>{100*float(case['anchor_inside_fraction']):.0f}%</b></div><div>Camera门=<b>{case['anchor_visibility']}</b></div></div><figure class="map"><img src="assets/{case['neutral_map']}"><figcaption>中立地图：只显示几何、EGO和目标，不含任何预测状态。</figcaption></figure>{''.join(galleries)}<div class="form"><label>目标身份确认<select data-field="identity_verified"><option value="">未标</option><option value="yes">是</option><option value="no">否</option><option value="uncertain">不确定</option></select></label><label>GT状态<select data-field="gt_state"><option value="">未标</option><option value="free">Free</option><option value="occupied">Occupied</option><option value="unknown">Unknown</option></select></label><label>Camera可观测性<select data-field="gt_camera_observability"><option value="">未标</option><option value="clear">清楚</option><option value="partial">部分可见</option><option value="occluded">遮挡</option><option value="unidentifiable">无法定位目标</option></select></label><label>置信度<select data-field="state_confidence"><option value="">未标</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select></label><label>标注者<input data-field="annotator"></label><label>主要证据帧<input data-field="evidence_frames" placeholder="例如 6674,6681"></label><label class="wide">备注<textarea data-field="notes" rows="3"></textarea></label></div></section>''')

    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>frame6681 Camera可见子集盲标</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#07101c;color:#e7eef8;font:15px/1.6 system-ui}}main{{max-width:1450px;margin:auto;padding:28px}}header,.case{{padding:28px;margin:20px 0;background:#0d1a2b;border:1px solid #334155;border-radius:16px}}h1{{font-size:42px}}h2{{font-size:30px}}.eyebrow{{color:#38bdf8;font-weight:800;letter-spacing:.14em}}.warning{{padding:16px;background:#2a1b09;border:1px solid #92400e;border-radius:10px}}.facts,.form{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}.facts div,label{{padding:12px;background:#122238;border-radius:9px}}.map img{{width:100%;max-height:650px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}img{{width:100%;background:#030812;border:1px solid #334155;border-radius:9px}}figcaption{{color:#9dacbf}}select,input,textarea{{display:block;width:100%;margin-top:7px;padding:9px;background:#07101c;color:#e7eef8;border:1px solid #475569;border-radius:6px}}.wide{{grid-column:1/-1}}button{{padding:12px 18px;margin-right:10px;border:0;border-radius:8px;background:#0284c7;color:white;font-weight:800}}@media(max-width:800px){{.pair{{grid-template-columns:1fr}}}}</style></head><body><main><header><span class="eyebrow">Prediction-blind annotation · frame6681</span><h1>Camera可见子集人工GT</h1><p>仅标注4个中心像素进入Camera的车位。页面不包含Part1状态、Agent结论、最终分数或占用算法输出。</p><div class="warning"><b>标注顺序：</b>先确认目标身份，再判断状态。画面不能明确定位或被遮挡时标Unknown，不要根据空白区域猜Free。紫色带仅表示状态无关的横向投影范围，不是车位像素框。</div><p><button onclick="saveLocal()">暂存浏览器</button><button onclick="exportJson()">导出标注JSON</button></p></header>{''.join(case_html)}</main><script>const fields=['identity_verified','gt_state','gt_camera_observability','state_confidence','annotator','evidence_frames','notes'];function collect(){{return [...document.querySelectorAll('.case')].map(c=>Object.assign({{slot_id:c.dataset.slot,anchor_lidar_frame:6681,anchor_camera_frame:20019}},Object.fromEntries(fields.map(f=>[f,c.querySelector(`[data-field="${{f}}"]`).value]))))}}function saveLocal(){{localStorage.setItem('frame6681_camera_visible_gt',JSON.stringify(collect()));alert('已暂存')}}function restore(){{const x=JSON.parse(localStorage.getItem('frame6681_camera_visible_gt')||'null');if(!x)return;x.forEach(r=>{{const c=document.querySelector(`[data-slot="${{r.slot_id}}"]`);if(c)fields.forEach(f=>c.querySelector(`[data-field="${{f}}"]`).value=r[f]||'')}})}}function exportJson(){{const b=new Blob([JSON.stringify({{schema_version:'camera-visible-human-gt/1.0',prediction_blind:true,rows:collect()}},null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='frame6681_camera_visible_gt.json';a.click();URL.revokeObjectURL(a.href)}}restore();</script></body></html>'''
    (OUTPUT / "index.html").write_text(html, encoding="utf-8")
    manifest = {
        "schema_version": "camera-visible-prediction-blind-annotation-pack/1.0",
        "anchor_lidar_frame": 6681,
        "anchor_camera_frame": 20019,
        "prediction_blind": True,
        "scope": "Camera-visible subset only; not overall Agent or LiDAR accuracy",
        "selection_rule": "horizontal pixel projection center reachability",
        "selected_slot_ids": list(selected_ids),
        "excluded_prediction_fields": ["part1_state", "agent_final_state", "confidence_scores", "terminal_geometry_gate"],
        "allowed_annotation_aids": ["neutral map geometry", "raw synchronized Camera", "state-independent horizontal projection guide"],
        "source_sha256": {str(LOCAL_MAP.relative_to(ROOT)): sha256(LOCAL_MAP), str(FRAMES.relative_to(ROOT)): sha256(FRAMES)},
        "blank_csv": csv_path.name,
        "html": "index.html",
        "cases": cases,
    }
    write_json(OUTPUT / "manifest.json", manifest)


if __name__ == "__main__":
    build()
