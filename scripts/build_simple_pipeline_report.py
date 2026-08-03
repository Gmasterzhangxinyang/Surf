#!/usr/bin/env python3
"""Build the concise, slot-by-slot Part1→Part2 audit page requested by the user."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "Nature_ParkingAgent_实验报告_20260728"
V5 = REPORT / "artifacts/frame_009277_v5_static_semantic_w30k5"
GT = ROOT / "outputs/parking_slot_agent_v2_frame_9277/GT独立几何全集_frame9277_v3_r25"
ASSETS = REPORT / "simple_pipeline_assets"
AGENT = V5 / "agent/run_with_tools"


def j(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value):
    return html.escape(str(value))


def short(values, n=4):
    values = list(values or [])
    if not values:
        return "—"
    return ", ".join(map(str, values[:n])) + (f" …(+{len(values)-n})" if len(values) > n else "")


def copy(source: Path, name: str):
    target = ASSETS / name
    shutil.copy2(source, target)
    return f"simple_pipeline_assets/{name}"


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def draw_front_map(local_map, queue, output: Path):
    queued = {item["slot_id"] for item in queue["items"]}
    colors = {"free": "#36b37e", "occupied": "#ef5350", "unknown": "#aab3c0"}
    fig, ax = plt.subplots(figsize=(10, 9))
    anchor = np.asarray(local_map["anchor_pose"]["map_xy"], dtype=float)
    yaw = float(local_map["anchor_pose"]["map_yaw_rad"])
    radius = float(local_map["lidar_coverage"]["nominal_radius_map_units"])
    wedge = Wedge(anchor, radius, np.degrees(yaw)-90, np.degrees(yaw)+90,
                  facecolor="#dceaff", edgecolor="#2563eb", alpha=.42, lw=2)
    ax.add_patch(wedge)
    coverage = np.asarray(local_map["lidar_coverage"]["polygon_map"], dtype=float)
    ax.add_patch(Polygon(coverage, closed=True, facecolor="#7dd3fc", edgecolor="#0284c7", alpha=.16, lw=1))
    for slot in local_map["slots"]:
        poly = np.asarray(slot["polygon_map"], dtype=float)
        sid = slot["slot_id"]
        is_queue = sid in queued
        ax.add_patch(Polygon(poly, closed=True, facecolor=colors[slot["state"]], alpha=.78,
                             edgecolor="#f59e0b" if is_queue else "#334155",
                             lw=4 if is_queue else 1.1))
        center = np.asarray(slot["center_map"], dtype=float)
        ax.text(center[0], center[1], sid.replace("slot_", ""), ha="center", va="center",
                fontsize=8, fontweight="bold" if is_queue else "normal",
                color="#7c2d12" if is_queue else "#172033")
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    ax.arrow(anchor[0], anchor[1], *(forward*radius*.42), width=.006, head_width=.045,
             color="#111827", length_includes_head=True, zorder=20)
    ax.scatter(*anchor, marker="o", s=80, color="#111827", zorder=21)
    ax.text(anchor[0], anchor[1]-.035, "ego / forward", ha="center", va="top", fontsize=10)
    ax.set_aspect("equal"); ax.autoscale(); ax.margins(.08); ax.axis("off")
    ax.set_title("Part2 closed front 180 deg | orange = 5 queued Unknown slots", fontsize=14)
    fig.tight_layout(); fig.savefig(output, dpi=190, bbox_inches="tight"); plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    summary = j(V5 / "summary.json")
    local_map = j(V5 / "local_map.json")
    frame_manifest = j(V5 / "local_frame_manifest.json")
    camera_capability = j(V5 / "camera_capability.json")
    decisions = j(V5 / "slot_decisions.json")["decisions"]
    v3 = {d["slot_id"]: d for d in j(REPORT / "artifacts/frame_009277_v3_final/slot_decisions.json")["decisions"]}
    v5 = {d["slot_id"]: d for d in decisions}
    queue = j(V5 / "unknown_agent_queue.json")
    agent_summary = j(AGENT / "summary.json")
    agent_final = {d["slot_id"]: d for d in j(AGENT / "final_route_states.json")["decisions"]}
    ablation = j(GT / "history_ablation_independent_gt_static_v2.json")
    accepted = list(csv.DictReader((GT / "accepted_terminal_labels_v1.csv").open(encoding="utf-8-sig")))
    gt_all = {r["slot_id"]: r for r in csv.DictReader((GT / "gt_adjudicated_high_confidence_v1.csv").open(encoding="utf-8-sig"))}

    front_map = ASSETS / "part2_front180_slots.png"
    draw_front_map(local_map, queue, front_map)
    part1_map = copy(V5 / "local_map.png", "part1_v5_all_slots.png")
    gt_map = copy(GT / "geometry_universe_neutral_map.png", "gt_independent_universe.png")

    input_cards = []
    input_rows = []
    for row in frame_manifest["frames"]:
        raw = Path(row["lidar_path"])
        point_count = raw.stat().st_size // 16 if raw.exists() else "missing"
        img_src = Path(row["camera_image_path"])
        asset = copy(img_src, f"input_camera_{row['frame_id']:06d}.png") if img_src.exists() else None
        if asset:
            input_cards.append(f"<figure><img src='{asset}'><figcaption>LiDAR {row['frame_id']} ↔ Camera {row['camera_frame']} · Δt={1000*row['camera_lidar_dt_sec']:.2f}ms</figcaption></figure>")
        input_rows.append(
            f"<tr><td>{row['frame_id']}</td><td>{point_count}</td><td><code>{esc(row['lidar_path'])}</code></td>"
            f"<td><code>{esc(row['map_points_path'])}</code></td><td>{row['camera_frame']}</td>"
            f"<td>{1000*row['camera_lidar_dt_sec']:.2f}ms</td><td>({row['map_x']:.6f}, {row['map_y']:.6f}, {row['map_yaw']:.6f})</td></tr>"
        )

    part1_rows = []
    for d in decisions:
        occ, free = d["occupied_evidence"], d["free_evidence"]
        gt = gt_all.get(d["slot_id"], {})
        gt_state = gt.get("gt_state", "—")
        gt_display = f"<span class='{gt_state}'>{gt_state}</span>" if gt_state in {"free","occupied","unknown"} else "—"
        if gt_state in {"free","occupied"}:
            verdict = "正确" if d["state"] == gt_state else ("弃权" if d["state"] == "unknown" else "错误")
        else:
            verdict = "GT未给终态"
        part1_rows.append(
            f"<tr><td>{d['slot_id']}</td><td><span class='{d['state']}'>{d['state']}</span></td>"
            f"<td>{d['decision_reason']}</td><td>{occ['strength']:.3f}<br><small>{esc(short(occ.get('failures')))}</small></td>"
            f"<td>{free['strength']:.3f}<br><small>{esc(short(free.get('failures')))}</small></td>"
            f"<td>{gt_display}</td><td>{verdict}</td></tr>"
        )

    gt_rows = []
    for r in accepted:
        sid = r["slot_id"]
        old = v3.get(sid, {}).get("state", "out_of_scope")
        new = v5.get(sid, {}).get("state", "out_of_scope")
        verdict = "正确" if new == r["gt_state"] else ("Unknown弃权" if new == "unknown" else "错误/范围外")
        gt_rows.append(
            f"<tr><td>{sid}</td><td><span class='free'>{r['gt_state']}</span></td><td>{r['gt_observability']}</td>"
            f"<td>{float(r['state_confidence']):.2f}</td><td><span class='{old}'>{old}</span></td>"
            f"<td><span class='{new}'>{new}</span></td><td>{verdict}</td><td>{esc(r['evidence_basis'])}</td></tr>"
        )

    queue_rows, agent_cards, final_rows = [], [], []
    unique_tool_slots = set()
    tool_attempt_count = 0
    for p in sorted((AGENT / "tool_artifacts").glob("*.json")):
        attempt = j(p)["attempt"]
        if attempt.get("tool_name") == "inspect_lidar_map" and attempt.get("executed"):
            tool_attempt_count += 1
            unique_tool_slots.add(attempt["data"]["slot_id"])
    camera_dir = GT / "camera_slot_identity_review"
    lidar_dir = GT / "static_aware_lidar_review/slot_views"
    for item in queue["items"]:
        sid = item["slot_id"]
        bearing = item["audit"]["candidate_relative_bearing_deg"]
        queue_rows.append(
            f"<tr><td>{sid}</td><td>{bearing:+.1f}°</td><td>{item['occupied_evidence']['strength']:.3f}</td>"
            f"<td>{item['free_evidence']['strength']:.3f}</td><td>{esc(short(item['unknown_reasons'],5))}</td>"
            f"<td>{', '.join(item['available_modalities'])}</td></tr>"
        )
        lidar_asset = copy(lidar_dir / f"{sid}.png", f"agent_{sid}_lidar.png")
        camera_path = camera_dir / f"{sid}.jpg"
        camera_html = "<div class='no-camera'>该车位没有可用的前视Camera投影视图</div>"
        if camera_path.exists():
            camera_asset = copy(camera_path, f"agent_{sid}_camera.jpg")
            camera_html = f"<img src='{camera_asset}'><p>Camera诊断标注（仅交叉复核，不是本次Agent终态依据）</p>"
        gt_state = gt_all.get(sid, {}).get("gt_state", "unknown")
        final_state = agent_final[sid]["state"]
        agent_cards.append(
            f"<article class='case'><h3>{sid} · bearing {bearing:+.1f}° · Part1 Unknown → Agent {final_state}</h3>"
            f"<div class='evidence'><div><img src='{lidar_asset}'><p><b>Agent实际工具：</b>inspect_lidar_map；检查点云形态、静态解释与车位归属。</p></div><div>{camera_html}</div></div>"
            f"<p class='reason'><b>最终原因：</b>LiDAR仍不能同时消除 {esc(short(item['unknown_reasons'],5))}；可信Camera target correspondence缺失，因此保持Unknown。</p></article>"
        )
        if gt_state in {"free","occupied"}:
            compare = "正确" if final_state == gt_state else ("安全弃权，但未完成" if final_state == "unknown" else "错误")
        else:
            compare = "GT本身Unknown，不计准确率"
        final_rows.append(
            f"<tr><td>{sid}</td><td><span class='unknown'>unknown</span></td><td><span class='{final_state}'>{final_state}</span></td>"
            f"<td><span class='{gt_state}'>{gt_state}</span></td><td>{compare}</td></tr>"
        )

    ablation_rows = []
    for r in ablation["runs"]:
        best = r["history_span_W"] == 30 and r["sample_count_K"] == 5
        ablation_rows.append(
            f"<tr class={'best' if best else ''}><td>{r['history_span_W']}</td><td>{r['sample_count_K']}</td>"
            f"<td>{r['selected_frame_ids']}</td><td>{r['correct_terminal_count']}</td><td>{r['false_occupied_count']}</td>"
            f"<td>{r['terminal_count']}</td><td>{r['runtime_seconds']:.2f}s</td><td>{'当前采用' if best else ''}</td></tr>"
        )

    html_text = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ParkingAgent Part1→Part2完整结果</title><style>
:root{--ink:#162033;--muted:#637086;--blue:#2563eb;--green:#12865f;--red:#d74747;--amber:#d88918;--line:#dce3ed;--bg:#f5f7fa}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,'Microsoft YaHei',sans-serif}nav{position:sticky;top:0;z-index:5;background:#132039;padding:11px 4vw;display:flex;gap:18px;overflow:auto}nav a{color:#dbe7fb;text-decoration:none;white-space:nowrap}.hero{background:#fff;padding:50px 5vw;border-bottom:1px solid var(--line)}h1{font-size:clamp(35px,5vw,64px);line-height:1.05;margin:8px 0 15px}.hero p{max-width:980px;color:var(--muted);font-size:18px}.summary{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:24px}.summary div{border:1px solid var(--line);border-radius:12px;padding:14px}.summary b{font-size:25px;display:block}.wrap{max-width:1500px;margin:auto;padding:22px 4vw 80px}.step{background:white;border:1px solid var(--line);border-radius:16px;padding:26px;margin:22px 0}.step h2{font-size:28px;margin:0 0 7px}.desc{color:var(--muted)}.sources{background:#eef4ff;border-left:4px solid var(--blue);padding:13px 16px}.rule{background:#111827;color:#eaf1ff;border-radius:10px;padding:15px;font:13px/1.7 monospace}.table{overflow:auto;max-height:620px;border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{position:sticky;top:0;background:#eaf0f8;z-index:1}code{font-size:11px}.free{color:var(--green);font-weight:800}.occupied{color:var(--red);font-weight:800}.unknown{color:#7b8798;font-weight:800}.out_of_scope{color:#9aa3af}.figure{border:1px solid var(--line);border-radius:11px;overflow:hidden;margin:15px 0}.figure img{display:block;width:100%}.figure p{padding:0 13px;color:var(--muted)}.gallery{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}.gallery figure{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden}.gallery img{width:100%;display:block}.gallery figcaption{padding:7px;font-size:11px}.case{border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}.case h3{margin:0 0 10px}.evidence{display:grid;grid-template-columns:1fr 1fr;gap:12px}.evidence img{width:100%;border:1px solid var(--line)}.evidence p,.reason{font-size:13px;color:var(--muted)}.no-camera{height:180px;display:grid;place-items:center;background:#eef1f5;color:var(--muted)}.best{background:#e8f8f1}.warning{background:#fff6e8;border-left:4px solid var(--amber);padding:13px}.result{background:#eaf9f3;border-left:4px solid var(--green);padding:13px}.small{font-size:12px;color:var(--muted)}@media(max-width:950px){.summary{grid-template-columns:repeat(2,1fr)}.gallery{grid-template-columns:repeat(2,1fr)}.evidence{grid-template-columns:1fr}}@media(max-width:560px){.summary,.gallery{grid-template-columns:1fr}.step{padding:16px}}</style></head><body>
<nav><a href='#input'>1 输入</a><a href='#part1'>2 Part1逐车位</a><a href='#gt'>3 对GT</a><a href='#front'>4 前方180°</a><a href='#agent'>5 Agent工具与标注</a><a href='#final'>6 最终对GT</a><a href='#ablation'>7 消融</a></nav>
<header class='hero'><div style='color:#2563eb;font-weight:800'>FRAME 9277 · V5 W30/K5</div><h1>Part1到Part2，按车位看完整过程</h1><p>页面只保留必要内容：输入文件、Part1每个车位状态、GT验证、前方180°候选、Agent实际工具与标注证据、最终结果和历史帧消融。</p><div class='summary'><div><b>30→5</b>历史池→实际LiDAR帧</div><div><b>22</b>Part1有状态车位</div><div><b>3 / 0 / 19</b>Free / Occupied / Unknown</div><div><b>5</b>进入Part2 Agent</div><div><b>5→5 Unknown</b>Agent前→Agent后</div></div></header><main class='wrap'>
<section class='step' id='input'><h2>1. 输入数据完整清单</h2><p class='desc'>历史池是anchor 9277之前连续30帧（9248–9277），均匀且包含anchor地取5帧：[9248, 9255, 9262, 9270, 9277]。</p><div class='sources'><b>固定输入：</b><br>frames：<code>outputs/frame_map_dataset_pose_corrected_final/frames.csv</code><br>1397车位几何：<code>outputs/full_icpark_allframes_vehicle_cluster/slot_database.json</code><br>地图点云：<code>outputs/frame_map_dataset_pose_corrected_final/map_points/</code><br>静态语义：<code>file.gltf</code>中的wall/elevator/arrester，共632,482采样点<br>Camera：只提供RGB诊断，depth_capability=__DEPTH__，metric_depth=__METRIC__。</div><div class='table'><table><thead><tr><th>LiDAR帧</th><th>原始点数</th><th>原始.bin</th><th>校正map点</th><th>Camera帧</th><th>时间差</th><th>校正位姿(x,y,yaw)</th></tr></thead><tbody>__INPUT_ROWS__</tbody></table></div><h3>对应的5张RGB输入</h3><div class='gallery'>__INPUT_CARDS__</div></section>
<section class='step' id='part1'><h2>2. Part1输出：22个车位逐车位状态</h2><p class='desc'>验证原理不是“车位里有点就是Occupied”。每个车位同时跑两条证据链，再做确定性三态融合。</p><div class='rule'>Occupied：车体真实短边≥0.75m、低层覆盖≥0.05、至少2个高度层、框外残差≤0.35、7/7位姿扰动稳定、静态语义不解释为墙/柱。<br>Free：至少5个ray帧、2个独立视角、视角差≥10°、体积/近地覆盖≥0.70、遮挡≤0.20、没有core hit冲突。<br>任一强证据链不完整或冲突 → Unknown，并保存具体原因。</div><div class='figure'><img src='__PART1_MAP__'><p>绿色Free、灰色Unknown；当前不输出Occupied。图中的A/B只是旧debug显示，不是Part2正式队列。</p></div><div class='table'><table><thead><tr><th>车位</th><th>Part1状态</th><th>主原因</th><th>Occupied强度/失败门</th><th>Free强度/失败门</th><th>GT</th><th>验证</th></tr></thead><tbody>__PART1_ROWS__</tbody></table></div></section>
<section class='step' id='gt'><h2>3. Part1和独立GT比较，是否准确</h2><p class='desc'>GT车位全集先由1397车位几何和60帧路线距离选出36个，不读取Part1状态。首轮高置信终态只有9个，全部Free；其余27个保持Unknown。</p><div class='figure'><img src='__GT_MAP__'><p>独立GT几何全集；旧GT只覆盖20/36，所以旧24候选GT不能再当总体真值。</p></div><div class='table'><table><thead><tr><th>车位</th><th>GT</th><th>GT可观测依据</th><th>置信度</th><th>旧v3</th><th>当前v5</th><th>结果</th><th>验证证据</th></tr></thead><tbody>__GT_ROWS__</tbody></table></div><div class='result'><b>结论：</b>同一GT交集8个车位中，旧v3为2正确Free、2错误Occupied、4Unknown；当前v5为3正确Free、0错误Occupied、5Unknown。错误Occupied从2降为0，选择性准确率100%，但终态覆盖率只有37.5%。因为这个anchor没有确认Occupied，不能计算9277的Occupied recall。</div></section>
<section class='step' id='front'><h2>4. 从Part1进入Part2：只保留车头前方180°</h2><div class='rule'>bearing = atan2(left, forward)<br>进入Part2 = Part1状态Unknown ∧ agent_observable ∧ |bearing|≤90°</div><div class='figure'><img src='simple_pipeline_assets/part2_front180_slots.png'><p>橙色粗框为正式Part2队列：1250、1253、1254、1255、1256。后方Unknown全部排除。</p></div><div class='table'><table><thead><tr><th>车位</th><th>相对方位</th><th>Occ强度</th><th>Free强度</th><th>为什么还是Unknown</th><th>可用模态</th></tr></thead><tbody>__QUEUE_ROWS__</tbody></table></div></section>
<section class='step' id='agent'><h2>5. Agent模块干了什么、用了什么工具</h2><p class='desc'>正式Agent把5个车位按相邻/共享证据分成3个冲突组，执行__TOOL_CALLS__次 <code>inspect_lidar_map</code>，覆盖5个唯一车位证据包。工具逐次校验NPZ哈希并返回can_assess_occupied能力；Camera因为没有通过测量级target correspondence，不允许作为终态工具。</p><div class='warning'><b>区分：</b>下面左侧LiDAR图是Agent实际检查的证据类型；右侧Camera图是独立诊断标注，用于人看车位身份和GT交叉复核，不是本次Agent终态依据。</div>__AGENT_CARDS__</section>
<section class='step' id='final'><h2>6. Agent后的最终结果，与GT比较</h2><div class='table'><table><thead><tr><th>车位</th><th>Part1</th><th>Agent后</th><th>GT</th><th>比较</th></tr></thead><tbody>__FINAL_ROWS__</tbody></table></div><div class='result'><b>最终效果：</b>5个Part2车位全部保持Unknown，0个错误终态、0个终态覆盖。在其中4个高置信GT Free车位上，Agent没有误判Occupied，但也没有成功输出Free。因此Agent当前效果是“安全但没有提升覆盖”，不能写成Agent提高了准确率。</div></section>
<section class='step' id='ablation'><h2>7. 历史帧消融：原来多少帧，现在改成多少</h2><p><b>原始正式Part1：</b>最近15个连续帧。<br><b>此前扩展实验：</b>历史60帧中取15帧（W60/K15）。<br><b>现在正式v5：</b>历史30帧中均匀取5帧（W30/K5），帧为[9248,9255,9262,9270,9277]。</p><p class='desc'>选择规则：先要求0个错误终态，再最大化正确覆盖，最后选择更低计算量。W30/K5和W60/K15都是3正确、0假Occupied，但W30/K5约7.00秒，W60/K15约13.61秒；W100所有K都会重新出现slot_1249假Occupied。</p><div class='table'><table><thead><tr><th>W</th><th>K</th><th>实际帧</th><th>正确终态</th><th>假Occupied</th><th>终态数</th><th>时间</th><th>选择</th></tr></thead><tbody>__ABLATION_ROWS__</tbody></table></div></section>
</main></body></html>"""
    replacements = {
        "__DEPTH__": camera_capability["depth_capability"], "__METRIC__": str(camera_capability["metric_depth"]).lower(),
        "__INPUT_ROWS__": "".join(input_rows), "__INPUT_CARDS__": "".join(input_cards),
        "__PART1_MAP__": part1_map, "__PART1_ROWS__": "".join(part1_rows),
        "__GT_MAP__": gt_map, "__GT_ROWS__": "".join(gt_rows), "__QUEUE_ROWS__": "".join(queue_rows),
        "__TOOL_CALLS__": str(tool_attempt_count), "__AGENT_CARDS__": "".join(agent_cards),
        "__FINAL_ROWS__": "".join(final_rows), "__ABLATION_ROWS__": "".join(ablation_rows),
    }
    for key, value in replacements.items():
        html_text = html_text.replace(key, value)
    (REPORT / "index.html").write_text(html_text, encoding="utf-8")
    audit = {
        "schema_version": "parkingagent-simple-pipeline-report/1.0",
        "inputs": {"history_span_W":30,"sample_count_K":5,"selected_frames":local_map["lidar_window"]["frame_ids"]},
        "part1_counts": summary,
        "part2": {"queue_slots":[x["slot_id"] for x in queue["items"]],"tool_calls":tool_attempt_count,"unique_tool_slots":sorted(unique_tool_slots),"final_counts":agent_summary["state_counts"]},
        "gt": {"high_confidence_count":len(accepted),"formal":False,"part2_terminal_gt_count":sum(gt_all.get(x["slot_id"],{}).get("gt_state") in {"free","occupied"} for x in queue["items"])},
    }
    (REPORT / "results/simple_pipeline_report.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    final_study_path = REPORT / "results/final_v5_study.json"
    if final_study_path.exists():
        final_study = j(final_study_path)
        final_study["frame9277"]["part2"] = agent_summary
        final_study["simple_report"] = audit
        final_study_path.write_text(json.dumps(final_study,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    files = [REPORT/"index.html", REPORT/"README.md", REPORT/"REPRODUCE.md", REPORT/"results/simple_pipeline_report.json", final_study_path, front_map, ASSETS/"part1_v5_all_slots.png", ASSETS/"gt_independent_universe.png"] + sorted(p for p in ASSETS.iterdir() if p.name.startswith(("input_","agent_")))
    manifest = "\n".join(f"{sha(p)}  {p.relative_to(ROOT)}" for p in files)+"\n"
    (REPORT / "SIMPLE_REPORT_MANIFEST.sha256").write_text(manifest,encoding="utf-8")
    (REPORT / "FINAL_V5_MANIFEST.sha256").write_text(manifest,encoding="utf-8")
    print(json.dumps({"index":str(REPORT/"index.html"),"part1_slots":len(decisions),"part2_slots":len(queue["items"]),"tool_calls":tool_attempt_count,"assets":len(list(ASSETS.iterdir()))},ensure_ascii=False))


if __name__ == "__main__":
    main()
