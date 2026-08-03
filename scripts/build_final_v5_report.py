#!/usr/bin/env python3
"""Build the final data-driven v5 Part1-to-Part2 visual report."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "Nature_ParkingAgent_实验报告_20260728"
ASSETS = REPORT / "final_v5_assets"
GT = ROOT / "outputs/parking_slot_agent_v2_frame_9277/GT独立几何全集_frame9277_v3_r25"
V5 = REPORT / "artifacts/frame_009277_v5_static_semantic_w30k5"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_asset(source: Path, name: str):
    target = ASSETS / name
    shutil.copy2(source, target)
    return f"final_v5_assets/{name}"


def build_charts(ablation, comparison, frozen_old, frozen_v5, queue):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"correct": "#16a36a", "false": "#e95353", "unknown": "#8a96a8"}

    models = [row["model"] for row in comparison["results"]]
    correct = [row["correct_terminal_count"] for row in comparison["results"]]
    false = [row["false_occupied_count"] + row["false_free_count"] for row in comparison["results"]]
    unknown = [row["state_counts_on_gt_overlap"]["unknown"] for row in comparison["results"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(models))
    axes[0].bar(x, correct, color=colors["correct"], label="correct terminal")
    axes[0].bar(x, false, bottom=correct, color=colors["false"], label="wrong terminal")
    axes[0].bar(x, unknown, bottom=np.array(correct)+np.array(false), color=colors["unknown"], label="Unknown")
    axes[0].set_xticks(x, models)
    axes[0].set_title("Frame 9277 · independent high-confidence GT overlap")
    axes[0].set_ylabel("slot count")
    axes[0].legend(frameon=False)
    names = ["old frozen", "v5 W30/K5"]
    coverage = [frozen_old["coverage"], frozen_v5["coverage"]]
    recall = [frozen_old["occupied_recall"], frozen_v5["occupied_recall"]]
    contradiction = [frozen_old["terminal_contradiction_count"] / 6, (frozen_v5["false_occupied_count"] + frozen_v5["false_free_count"]) / 6]
    xx = np.arange(2); width = .24
    axes[1].bar(xx-width, coverage, width, label="terminal coverage", color="#2f6fe4")
    axes[1].bar(xx, recall, width, label="Occupied recall", color="#f29d38")
    axes[1].bar(xx+width, contradiction, width, label="contradiction / all GT", color="#e95353")
    axes[1].set_xticks(xx, names); axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Frozen 6-case challenge · safety/coverage trade-off")
    axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(ASSETS / "final_model_comparison.png", dpi=180); plt.close(fig)

    runs = ablation["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for w in sorted({r["history_span_W"] for r in runs}):
        rows = [r for r in runs if r["history_span_W"] == w]
        axes[0].plot([r["sample_count_K"] for r in rows], [r["correct_terminal_count"] for r in rows], marker="o", label=f"W={w}")
        axes[1].plot([r["sample_count_K"] for r in rows], [r["false_occupied_count"] for r in rows], marker="o", label=f"W={w}")
    axes[0].set_title("Correct terminal labels on 9277 GT"); axes[0].set_xlabel("sample count K"); axes[0].set_ylabel("correct")
    axes[1].set_title("False Occupied (lower is safer)"); axes[1].set_xlabel("sample count K"); axes[1].set_ylabel("false occupied")
    for ax in axes: ax.set_xticks([5,10,15,20]); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.suptitle("15 causal W/K combinations · semantic-static veto enabled")
    fig.tight_layout(); fig.savefig(ASSETS / "wk_ablation_v5.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    theta = np.linspace(-np.pi/2, np.pi/2, 200)
    ax.fill_between(np.cos(theta)*18, 0, np.sin(theta)*18, color="#e9f1ff", alpha=.9)
    ax.plot([0,18],[0,0],color="#2f6fe4",lw=2); ax.arrow(0,0,4,0,width=.12,head_width=.8,color="#111827")
    for item in queue["items"]:
        bearing = np.deg2rad(item["audit"]["candidate_relative_bearing_deg"])
        r = 8 + 0.35 * len(item["slot_id"])
        x, y = r*np.cos(bearing), r*np.sin(bearing)
        ax.scatter(x,y,s=90,color="#f4a83a",edgecolor="#8b5a00")
        ax.text(x+.3,y+.3,item["slot_id"].replace("slot_",""),fontsize=9)
    ax.text(7.5,16,"left +90°",ha="center",color="#2f6fe4"); ax.text(7.5,-16,"right −90°",ha="center",color="#2f6fe4")
    ax.set_xlim(-2,20); ax.set_ylim(-19,19); ax.axis("off"); ax.set_title("Part2 queue: all 5 candidates are inside the closed front 180°")
    fig.tight_layout(); fig.savefig(ASSETS / "front180_v5_queue.png", dpi=180); plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    comparison = load(GT / "part1_independent_gt_comparison_v5.json")
    ablation = load(GT / "history_ablation_independent_gt_static_v2.json")
    gt_manifest = load(GT / "gt_adjudicated_high_confidence_v1_manifest.json")
    frozen_score = load(REPORT / "results/frozen_challenge_v5_w30k5_score.json")
    frozen_all = load(REPORT / "results/frozen_human_eval_score_v2.json")
    frozen_old = frozen_all["metrics"]["evidence_aligned"]["frozen_v1_width_0p10"]
    v5_summary = load(V5 / "summary.json")
    queue = load(V5 / "unknown_agent_queue.json")
    agent = load(V5 / "agent/run/summary.json")
    decisions = load(V5 / "slot_decisions.json")["decisions"]
    build_charts(ablation, comparison, frozen_old, frozen_score["metrics"], queue)

    images = {
        "inputs": "full_pipeline_assets/01_inputs_and_sync.png",
        "pose": "full_pipeline_assets/02A_localization_pose_and_residual.png",
        "extract": "full_pipeline_assets/04_slot_evidence_extraction.png",
        "occupied": "full_pipeline_assets/05_occupied_branch.png",
        "free": "full_pipeline_assets/06_free_ray_branch.png",
        "fusion": "full_pipeline_assets/07_three_state_fusion.png",
        "oldmap": copy_asset(REPORT / "artifacts/frame_009277_v3_final/local_map.png", "part1_v3_local_map.png"),
        "v5map": copy_asset(V5 / "local_map.png", "part1_v5_local_map.png"),
        "gtmap": copy_asset(GT / "geometry_universe_neutral_map.png", "independent_gt_universe.png"),
        "static1252": copy_asset(GT / "static_aware_lidar_review/slot_views/slot_1252.png", "static_slot_1252.png"),
        "static1256": copy_asset(GT / "static_aware_lidar_review/slot_views/slot_1256.png", "static_slot_1256.png"),
        "contact": copy_asset(GT / "static_aware_lidar_review/contact_sheets/lidar_neutral_contact_sheet_04.jpg", "static_review_contact_sheet.jpg"),
    }
    qrows = "".join(
        f"<tr><td>{x['slot_id']}</td><td>{x['audit']['candidate_relative_bearing_deg']:+.1f}°</td><td>{x['occupied_evidence']['strength']:.3f}</td><td>{x['free_evidence']['strength']:.3f}</td><td>{', '.join(x['unknown_reasons'][:3])}</td><td>Unknown</td></tr>"
        for x in queue["items"]
    )
    frozen_rows = "".join(
        f"<tr><td>{r['anchor_frame']}</td><td>{r['slot_id']}</td><td class='{r['gt_state']}'>{r['gt_state']}</td><td class='{r['prediction']}'>{r['prediction']}</td><td>{r['decision_reason']}</td></tr>"
        for r in frozen_score["rows"]
    )
    abrows = "".join(
        f"<tr class={'best' if r['history_span_W']==30 and r['sample_count_K']==5 else ''}><td>{r['history_span_W']}</td><td>{r['sample_count_K']}</td><td>{r['correct_terminal_count']}</td><td>{r['false_occupied_count']}</td><td>{r['terminal_count']}</td><td>{r['runtime_seconds']:.2f}s</td></tr>"
        for r in ablation["runs"]
    )
    drows = "".join(
        f"<tr><td>{d['slot_id']}</td><td class='{d['state']}'>{d['state']}</td><td>{d['decision_reason']}</td><td>{', '.join(d.get('unknown_reasons',[])[:3]) or '—'}</td></tr>"
        for d in decisions
    )
    summary = {
        "schema_version": "parkingagent-final-v5-study/1.0",
        "created_date": "2026-07-29",
        "operating_point": {"model":"v5_static_semantic", "W":30, "K":5, "part2_front_total_fov_deg":180},
        "frame9277": {"gt": {"free":9,"occupied":0,"unknown":27,"formal":False}, "part1": v5_summary, "part2": agent},
        "frozen_challenge": {"old": frozen_old, "v5": frozen_score["metrics"], "status": frozen_score["evaluation_status"]},
        "claims": {"false_occupied_reduced_on_9277": "2 to 0", "occupied_recall_evaluable_on_9277": False, "nature_ready": False},
        "source_hashes": {
            "gt": gt_manifest.get("output",{}).get("sha256", gt_manifest.get("gt_csv_sha256")),
            "v5_decisions": sha(V5 / "slot_decisions.json"),
            "ablation": sha(GT / "history_ablation_independent_gt_static_v2.json"),
            "part2_queue": sha(V5 / "unknown_agent_queue.json"),
        },
    }
    (REPORT / "results/final_v5_study.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    html = """<!doctype html><html lang=zh-CN><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>ParkingAgent v5 · Part1→Part2完整实验</title>
<style>:root{--ink:#152033;--muted:#617087;--blue:#2563eb;--green:#159768;--red:#dc4c4c;--amber:#e79a2d;--line:#dde4ee;--bg:#f3f6fa}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,'Microsoft YaHei',sans-serif}nav{position:sticky;top:0;z-index:5;background:#111b2d;padding:12px 4vw;display:flex;gap:15px;overflow:auto}nav a{color:#c9d4e7;text-decoration:none;white-space:nowrap}.hero{padding:66px 5vw 50px;background:linear-gradient(125deg,#fff,#e6efff)}h1{font-size:clamp(38px,6vw,72px);line-height:1.03;margin:10px 0}.lead{max-width:1000px;color:var(--muted);font-size:18px}.metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:28px}.metric,.card{background:white;border:1px solid var(--line);border-radius:14px;padding:16px}.metric b{font-size:27px;display:block}.warn{background:#fff7e8;border-left:5px solid var(--amber);padding:15px;margin-top:22px}.wrap{max-width:1450px;margin:auto;padding:24px 4vw 80px}.stage{background:#fff;border:1px solid var(--line);border-radius:18px;padding:28px;margin:22px 0}.stage h2{font-size:28px;margin:0 0 6px}.sub{color:var(--muted)}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}.flow .card small{display:block;color:var(--blue);font-weight:800}.figure{border:1px solid var(--line);border-radius:12px;overflow:hidden;margin:18px 0;background:#f8fafc}.figure img{display:block;width:100%}.cap{padding:11px 15px;background:white;color:var(--muted)}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:14px}.formula{background:#111827;color:#e8efff;border-radius:11px;padding:16px;font:13px/1.7 monospace;overflow:auto}.good{border-left:5px solid var(--green);padding:13px;background:#ebfaf4}.bad{border-left:5px solid var(--red);padding:13px;background:#fff0f0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line)}th{position:sticky;top:45px;background:#eef3fa}.free{color:var(--green);font-weight:800}.occupied{color:var(--red);font-weight:800}.unknown{color:#7c899b;font-weight:800}.best{background:#eaf8f2}.scroll{max-height:560px;overflow:auto}.tag{display:inline-block;padding:3px 8px;border-radius:20px;background:#e9eef6;margin:2px;font-size:12px}.links a{display:inline-block;margin:4px 10px 4px 0}.foot{font-size:12px;color:var(--muted)}@media(max-width:900px){.metrics,.flow{grid-template-columns:repeat(2,1fr)}.twocol{grid-template-columns:1fr}}@media(max-width:550px){.metrics,.flow{grid-template-columns:1fr}.stage{padding:17px}}</style></head><body>
<nav><a href=#result>结论</a><a href=#gt>独立GT</a><a href=#input>输入/定位</a><a href=#part1>Part1算法</a><a href=#static>静态语义</a><a href=#wk>W/K消融</a><a href=#front>前方180°</a><a href=#part2>Part2 Agent</a><a href=#experiment>原始vs改进</a><a href=#slots>逐车位</a><a href=#limits>边界</a></nav>
<header class=hero><div style='color:#2563eb;font-weight:800'>FINAL V5 DATA-DRIVEN REPORT · 2026-07-29</div><h1>从原始点云到 Part2 Agent<br>每一步都有数据、算法和效果</h1><p class=lead>正式 operating point：因果历史 W=30 中均匀取 K=5；Part1 用真实3D足迹、Free-ray、姿态稳定性与 glTF 静态语义硬否决；Part2 只接收车头前方闭合180°内的 Unknown。</p><div class=metrics><div class=metric><b>1397→22</b>全库→当前有状态</div><div class=metric><b>3/0/19</b>Free/Occupied/Unknown</div><div class=metric><b>2→0</b>9277错误Occupied</div><div class=metric><b>30/5</b>最佳W/K</div><div class=metric><b>5</b>前向Part2队列</div><div class=metric><b>510</b>测试通过</div></div><div class=warn><b>效果不是“所有车位都判出来”：</b>误报被压到0，但冻结Occupied挑战集召回从75%降到25%。这是高精度安全模式的真实代价，不隐藏Unknown。</div></header>
<main class=wrap>
<section class=stage id=result><h2>0. 最终结论</h2><div class=twocol><div class=good><b>解决了什么</b><br>slot_1252/1256等立柱、墙体或窄残差不再直接变成Occupied；9277独立GT交集上错误Occupied由2降至0。W30/K5在零终态错误约束下得到最高正确覆盖。</div><div class=bad><b>仍然是什么</b><br>系统目前是precision-first而非recall-first。9277没有确认Occupied，不能计算该帧Occupied recall；冻结6例只是回顾性挑战集，不是新独立holdout。</div></div><div class=figure><img src='final_v5_assets/final_model_comparison.png'><div class=cap>左：9277同一独立高置信GT交集；右：冻结6例显示安全性提高、覆盖和Occupied recall下降。</div></div></section>
<section class=stage id=gt><h2>1. GT先独立于Part1建立</h2><p class=sub>旧GT先用旧Part1筛24个候选再标注，存在selection bias。新版先从1397车位几何与60帧因果路线取25m全集36个，再盲看原始LiDAR、静态语义和诊断支持的Camera。</p><div class=flow><div class=card><small>DATA</small>1397车位几何<br>9218–9277路线</div><div class=card><small>SELECTION</small>距因果路线≤25m<br>不读Part1/旧GT</div><div class=card><small>REVIEW</small>60帧LiDAR + glTF静态层<br>Camera只作辅助</div><div class=card><small>OUTPUT</small>9 Free / 0 Occupied / 27 Unknown</div></div><div class=figure><img src='final_v5_assets/independent_gt_universe.png'><div class=cap>36车位独立几何全集；该图不显示模型预测，防止标注者被Part1状态锚定。</div></div><div class=warn>formal_gt=false：仍缺第二名真人盲标者与冲突仲裁。当前9个终态可用于高置信子集评估，但不能冒充论文正式总体GT。</div></section>
<section class=stage id=input><h2>2. 输入、同步、定位与历史采样</h2><div class=flow><div class=card><small>INPUT</small>map-frame LiDAR NPZ<br>pose/timestamp</div><div class=card><small>CAUSAL</small>anchor=9277<br>pool=9248–9277</div><div class=card><small>SAMPLE</small>uniform inclusive anchor<br>[9248,9255,9262,9270,9277]</div><div class=card><small>ALIGN</small>按位姿变换到同一地图坐标<br>不使用未来帧</div></div><div class=figure><img src='full_pipeline_assets/01_inputs_and_sync.png'><div class=cap>真实LiDAR点数、相机时间匹配；8-bit depth PNG不可逆为metric depth。</div></div><div class=figure><img src='full_pipeline_assets/02A_localization_pose_and_residual.png'><div class=cap>原始/校正轨迹、相邻点云残差与局部ICP诊断。历史越长并不一定越准，漂移会扩大窄结构残差。</div></div></section>
<section class=stage id=part1><h2>3. Part1三态算法：不是“有点=Occupied”</h2><div class=flow><div class=card><small>SCOPE</small>真实ray/hit进入局部范围</div><div class=card><small>OCCUPIED BRANCH</small>车体3D足迹、层高、时序、ownership</div><div class=card><small>FREE BRANCH</small>射线穿越、视角分离、体积/近地覆盖、遮挡</div><div class=card><small>FUSION</small>强Occupied→O<br>强Free且无冲突→F<br>否则Unknown+原因</div></div><div class=formula>Occupied = all(valid_frames≥3, points≥40, support≥3, short_extent≥0.75m, low_BEV≥0.05, layers≥2, outside_residual≤0.35, stability=7/7, semantic_static_veto_pass)<br>Free = ray_frames≥5 ∧ viewpoints≥2 ∧ separation≥10° ∧ volume≥0.70 ∧ near_ground≥0.70 ∧ occlusion≤0.20 ∧ no_core_hit<br>else = Unknown(reason_codes)</div><div class=twocol><div class=figure><img src='full_pipeline_assets/05_occupied_branch.png'><div class=cap>Occupied分支逐硬门；任一失败只能Unknown。</div></div><div class=figure><img src='full_pipeline_assets/06_free_ray_branch.png'><div class=cap>Free-ray明确区分穿越、hit、occluded、unobserved。</div></div></div><div class=twocol><div class=figure><img src='final_v5_assets/part1_v3_local_map.png'><div class=cap>旧v3：2 Free / 2 Occupied / 17 Unknown，包含2个假Occupied。</div></div><div class=figure><img src='final_v5_assets/part1_v5_local_map.png'><div class=cap>v5 W30/K5：3 Free / 0 Occupied / 19 Unknown。</div></div></div></section>
<section class=stage id=static><h2>4. 立柱/墙体修复：静态语义Occupied硬否决</h2><p>只对准备输出Occupied的强候选做安全复核：点与glTF wall/elevator/arrester在0.35m内关联；静态解释比例≥0.50，或去掉静态点后的真实残差短边&lt;0.75m，则禁止Occupied。</p><div class=formula>if static_explained_ratio ≥ 0.50 → Unknown(semantic_static_explained_ratio)<br>else if residual_short_extent &lt; 0.75m → Unknown(semantic_static_residual_too_narrow)<br>else → allow original Occupied gates</div><div class=twocol><div class=figure><img src='final_v5_assets/static_slot_1252.png'><div class=cap>slot_1252：用户确认Free；真实点足迹/框外残差不足以支持车体。</div></div><div class=figure><img src='final_v5_assets/static_slot_1256.png'><div class=cap>slot_1256：69.4%目标点可由静态地图解释，因此从假Occupied降为Unknown。</div></div></div><div class=figure><img src='final_v5_assets/static_review_contact_sheet.jpg'><div class=cap>36车位静态语义复核接触表的一页；静态解释点与未解释残差分色显示。</div></div></section>
<section class=stage id=wk><h2>5. 历史帧消融：15组，不是拍脑袋选30/5</h2><p>目标按顺序固定为：零终态错误 → 最大正确终态覆盖 → 更小K/W/运行时间。W15证据不足；W30与W60同为3正确0错误；W30/K5成本最低；W100所有K均重新产生slot_1249假Occupied。</p><div class=figure><img src='final_v5_assets/wk_ablation_v5.png'><div class=cap>语义静态v2消融：W=100受长历史定位漂移影响，不能因为“帧更多”就选100/20。</div></div><div class=scroll><table><thead><tr><th>W</th><th>K</th><th>正确终态</th><th>假Occupied</th><th>终态数</th><th>时间</th></tr></thead><tbody>__ABROWS__</tbody></table></div></section>
<section class=stage id=front><h2>6. Candidate进入Part2：严格车头前方闭合180°</h2><p>Part1所有Unknown先计算相对方位 bearing=atan2(left,forward)。只有agent_observable且|bearing|≤90°进入完整Part2队列。Free可作为停车候选，但不需要Agent重新判断；后方Unknown不会送入Part2。</p><div class=formula>forward = Δp · [cos(yaw), sin(yaw)]<br>left = Δp · [−sin(yaw), cos(yaw)]<br>queued = state==Unknown ∧ agent_observable ∧ abs(atan2(left,forward))≤90°</div><div class=figure><img src='final_v5_assets/front180_v5_queue.png'><div class=cap>实际5个Unknown的方位为+26.1°、+30.2°、+39.7°、+49.1°、+72.3°，没有后方车位。</div></div><div class=scroll><table><thead><tr><th>slot</th><th>bearing</th><th>Occ strength</th><th>Free strength</th><th>主要Unknown原因</th><th>Part2后</th></tr></thead><tbody>__QROWS__</tbody></table></div><div class=warn><b>“最多展示两个临时候选”不是Part2策略：</b>local_map中的A/B仅是debug visualization shortlist，consumed_by_part2=false。正式Part2队列是上表5个。</div></section>
<section class=stage id=part2><h2>7. Part2 Agent：证据不足时必须保持Unknown</h2><div class=flow><div class=card><small>QUEUE</small>5个前向Unknown<br>3个冲突组</div><div class=card><small>TOOLS</small>每slot有LiDAR evidence pack<br>Camera target correspondence未通过</div><div class=card><small>GATE</small>不能用歧义LiDAR制造Free/Occupied终态</div><div class=card><small>RESULT</small>0 Free / 0 Occupied / 5 Unknown<br>0 terminal override</div></div><div class=good>Part2没有“为了有结果”覆盖Part1。当前回放是严格fail-closed控制：没有经过哈希审计的目标相机证据时，保留Unknown。</div><div class=bad>这同时说明Agent目前没有带来覆盖提升。原始vs Agent对比必须如实报告为5→5 Unknown，而不是引用旧队列上的历史OpenAI动作。</div></section>
<section class=stage id=experiment><h2>8. 原始、v5与Agent后的实验效果</h2><div class=twocol><div class=card><h3>9277独立高置信Free子集</h3><p>v3：2正确、2假Occupied、4Unknown（交集8）<br>v4：2正确、0假Occupied、6Unknown<br>v5：3正确、0假Occupied、5Unknown<br>Agent：5个送入Part2仍Unknown</p></div><div class=card><h3>冻结6例挑战集</h3><p>旧版：选择性准确率75%，覆盖66.7%，Occupied recall75%，1个终态矛盾<br>v5：选择性准确率100%，覆盖16.7%，Occupied recall25%，0终态矛盾</p></div></div><div class=scroll><table><thead><tr><th>anchor</th><th>slot</th><th>GT</th><th>v5</th><th>原因</th></tr></thead><tbody>__FROZEN_ROWS__</tbody></table></div><p class=foot>冻结挑战集标签在回放前已存在，但样本来自历史证据筛选，因此只作回顾性挑战，不与9277几何全集合并成“15例独立GT”。</p></section>
<section class=stage id=slots><h2>9. v5全部22个当前Part1状态</h2><div class=scroll><table><thead><tr><th>slot</th><th>state</th><th>主原因</th><th>前三个Unknown原因</th></tr></thead><tbody>__DROWS__</tbody></table></div></section>
<section class=stage id=limits><h2>10. 论文级边界与可复现入口</h2><div class=bad><b>不能声称：</b>Nature-ready、总体准确率100%、9277 Occupied recall、相机测量级标定、Agent已提升覆盖。</div><div class=good><b>可以声称：</b>在9277独立高置信Free子集上，v5把已观察到的2个假Occupied降到0；15组因果W/K中W30/K5满足预注册安全优先目标；Part2严格前方180°且在证据不足时fail-closed。</div><div class=links><a href='results/final_v5_study.json'>最终机器结果JSON</a><a href='results/frozen_challenge_v5_w30k5_rows.csv'>冻结挑战逐例CSV</a><a href='../outputs/parking_slot_agent_v2_frame_9277/GT独立几何全集_frame9277_v3_r25/gt_adjudicated_high_confidence_v1.csv'>36车位GT三态表</a><a href='../outputs/parking_slot_agent_v2_frame_9277/GT独立几何全集_frame9277_v3_r25/history_ablation_independent_gt_static_v2.csv'>W/K消融CSV</a><a href='artifacts/frame_009277_v5_static_semantic_w30k5/report.html'>原始v5运行报告</a></div><p class=foot>完整正式GT仍需要第二名真人盲标者和冲突仲裁；这是外部人工流程，代码不能伪造。所有其余工程、回放、消融、Part2队列和报告产物均已生成。</p></section>
</main></body></html>"""
    html = html.replace("__ABROWS__", abrows).replace("__QROWS__", qrows).replace("__FROZEN_ROWS__", frozen_rows).replace("__DROWS__", drows)
    archive = REPORT / "index_v4_archive.html"
    if not archive.exists() and (REPORT / "index.html").exists(): shutil.copy2(REPORT / "index.html", archive)
    (REPORT / "index.html").write_text(html,encoding="utf-8")
    manifest_files = [
        REPORT / "index.html",
        REPORT / "README.md",
        REPORT / "最终实验结论_v5.md",
        REPORT / "REPRODUCE.md",
        REPORT / "results/final_v5_study.json",
        REPORT / "results/frozen_challenge_v5_w30k5_score.json",
        REPORT / "results/frozen_challenge_v5_w30k5_rows.csv",
        GT / "gt_adjudicated_high_confidence_v1.csv",
        GT / "history_ablation_independent_gt_static_v2.json",
        V5 / "slot_decisions.json",
        V5 / "unknown_agent_queue.json",
        V5 / "agent/run/summary.json",
    ] + sorted(ASSETS.iterdir())
    manifest = "\n".join(
        f"{sha(path)}  {path.relative_to(ROOT)}" for path in manifest_files
    ) + "\n"
    (REPORT / "FINAL_V5_MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    print(json.dumps({"index":str(REPORT/'index.html'),"summary":str(REPORT/'results/final_v5_study.json'),"manifest":str(REPORT/'FINAL_V5_MANIFEST.sha256'),"assets":len(list(ASSETS.iterdir()))},ensure_ascii=False))


if __name__ == "__main__":
    main()
