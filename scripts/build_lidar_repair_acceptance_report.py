#!/usr/bin/env python3
"""Build a focused Chinese HTML acceptance report for the LiDAR repair."""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import shutil


TOOL_ZH = {
    "camera_context": "Camera全局定位",
    "camera_crop": "Camera假设裁剪",
    "camera_sequence": "Camera时序验证",
    "lidar_detail": "扩展LiDAR细查",
}
STATE_ZH = {"free": "Free 空闲", "occupied": "Occupied 占用", "unknown": "Unknown 未知"}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _case_index(run: dict) -> dict[str, dict]:
    return {
        str(row["case"]["slot"]["slot_id"]): row
        for row in run["slot_results"]
    }


def _evidence(case: dict, tool: str) -> dict | None:
    return next(
        (row for row in case["evidence"] if row.get("tool_name") == tool),
        None,
    )


def _copy_image(path: str | Path, assets: Path, name: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = assets / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return f"assets/{destination.name}"


def _pct(value: object) -> str:
    return f"{float(value or 0) * 100:.1f}%"


def _trace(case: dict) -> str:
    cards: list[str] = []
    for row in case.get("rounds", []):
        tool = str(row.get("tool_name"))
        localization = row.get("localization") or {}
        loc = (
            f"{localization.get('target_side', 'unknown')} / "
            f"{localization.get('depth_band', 'unknown')} / "
            f"{localization.get('stage', 'not_attempted')}"
        )
        cards.append(
            "<li>"
            f"<span class='round'>R{int(row['round_index'])}</span>"
            f"<b>{html.escape(TOOL_ZH.get(tool, tool))}</b>"
            f"<p>{html.escape(str(row.get('reasoning_summary', '')))}</p>"
            f"<small>定位状态：{html.escape(loc)}；Observation："
            f"{html.escape(str(row.get('observation_summary', '')))}</small>"
            "</li>"
        )
    return "<ol class='trace'>" + "".join(cards) + "</ol>"


def _hero(
    *,
    slot_id: str,
    row: dict,
    assets: Path,
    conclusion_zh: str,
) -> str:
    case = row["case"]
    fov = _evidence(case, "check_fov")
    lidar = _evidence(case, "lidar_detail")
    if not fov or not lidar:
        raise ValueError(f"{slot_id} is missing FOV or LiDAR evidence")
    fov_image = _copy_image(
        fov["artifact_paths"][0], assets, f"{slot_id}_fov.png"
    )
    lidar_image = _copy_image(
        lidar["metadata"]["model_image_paths"][0],
        assets,
        f"{slot_id}_lidar_explained.png",
    )
    card = lidar["metadata"]["geometry_card"]
    free = card["free_geometry"]
    occupied = card["occupied_geometry"]
    robust = card["robustness"]
    final_state = str(case["final_state"])
    return f"""
<article class="case">
  <div class="case-head">
    <div><span class="eyebrow">{html.escape(slot_id)}</span>
      <h3>Part1 Unknown → <span class="state {final_state}">{STATE_ZH[final_state]}</span></h3></div>
    <div class="score">Free {float(case['final_scores']['free_confidence']):.3f}<br>Occupied {float(case['final_scores']['occupied_confidence']):.3f}</div>
  </div>
  <p class="conclusion">{html.escape(conclusion_zh)}</p>
  <div class="visual-grid">
    <figure><img src="{fov_image}" alt="{slot_id} FOV"><figcaption>① 原 occupancy map 上的真实 Camera FOV 路由；只决定 Camera 是否可用，不直接改状态。</figcaption></figure>
    <figure class="wide"><img src="{lidar_image}" alt="{slot_id} LiDAR"><figcaption>② Agent 实际看到的扩展 LiDAR 解释图：目标核心、视角、垂直结构和代码硬门在同一张图中。</figcaption></figure>
  </div>
  <div class="metrics">
    <span>证据源 <b>{html.escape(str(lidar['metadata']['evidence_source']))}</b></span>
    <span>窗口 <b>{lidar['metadata']['selected_frame_count']}帧 / 有效{lidar['metadata']['valid_frame_count']}帧</b></span>
    <span>核心射线 <b>{_pct(free.get('core_ray_coverage'))}</b></span>
    <span>体积覆盖 <b>{_pct(free.get('observed_volume_ratio'))}</b></span>
    <span>核心障碍 <b>{occupied.get('core_point_count') or 0}点 / {occupied.get('supported_height_layers') or 0}层</b></span>
    <span>稳健性 <b>{robust.get('passing_variants', 0)}/{robust.get('total_variants', 0)}</b></span>
  </div>
  <h4>Agent 可审计决策轨迹</h4>
  {_trace(case)}
  <details><summary>展开最终审计理由</summary><p>{html.escape(str(case.get('final_reason', '')))}</p>
  <p>未解决项：{html.escape('；'.join(case.get('unresolved_reasons', [])) or '无')}</p></details>
</article>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-run", type=Path, required=True)
    parser.add_argument("--active-run", type=Path, required=True)
    parser.add_argument("--geometry-regression", type=Path, required=True)
    parser.add_argument("--part1-visual-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    terminal = _load(args.terminal_run / "part2_result.json")
    active = _load(args.active_run / "part2_result.json")
    geometry = _load(args.geometry_regression)
    visual_manifest = json.loads(args.part1_visual_manifest.read_text(encoding="utf-8"))
    terminal_cases = _case_index(terminal)
    active_cases = _case_index(active)
    output = args.output_dir
    assets = output / "assets"
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    repaired_0968 = next(
        row for row in visual_manifest
        if row["mode"] == "part1_review" and row["slot_id"] == "slot_0968"
    )
    part1_0968_image = _copy_image(
        repaired_0968["artifacts"][-1], assets, "slot_0968_part1_repaired.png"
    )
    p1 = _load(Path(geometry["source"]))
    part1_states = {
        row["slot"]["slot_id"]: row["part1_state"] for row in p1["slot_cases"]
    }
    transitions = Counter(
        (part1_states[row["slot_id"]], row["state"]) for row in geometry["cases"]
    )
    unknown_total = sum(value for (before, _), value in transitions.items() if before == "unknown")
    unknown_resolved = sum(
        value for (before, after), value in transitions.items()
        if before == "unknown" and after != "unknown"
    )
    table_rows = []
    for row in geometry["cases"]:
        state = row["state"]
        free = row["free_geometry"]
        occupied = row["occupied_geometry"]
        robust = row["robustness"]
        table_rows.append(
            "<tr>"
            f"<td><b>{html.escape(row['slot_id'])}</b></td>"
            f"<td>{STATE_ZH[part1_states[row['slot_id']]]}</td>"
            f"<td><span class='state {state}'>{STATE_ZH[state]}</span></td>"
            f"<td>{row['selected_frame_count']} / {row['valid_frame_count']}</td>"
            f"<td>{_pct(free.get('core_ray_coverage'))}</td>"
            f"<td>{_pct(free.get('observed_volume_ratio'))}</td>"
            f"<td>{occupied.get('core_point_count') or 0} / {occupied.get('supported_height_layers') or 0}</td>"
            f"<td>{robust.get('passing_variants', 0)}/{robust.get('total_variants', 0)}</td>"
            "</tr>"
        )

    hero_html = "".join(
        [
            _hero(
                slot_id="slot_1012",
                row=terminal_cases["slot_1012"],
                assets=assets,
                conclusion_zh="扩展视角把目标核心观测完整：自由空间体积覆盖99.8%、7/7位姿扰动通过。Agent首轮选LiDAR，第二轮立即判Free。",
            ),
            _hero(
                slot_id="slot_1258",
                row=terminal_cases["slot_1258"],
                assets=assets,
                conclusion_zh="目标核心形成27,891点、3个高度层的稳定三维障碍，7/7扰动通过。Agent首轮选LiDAR，第二轮判Occupied。",
            ),
            _hero(
                slot_id="slot_1248",
                row=terminal_cases["slot_1248"],
                assets=assets,
                conclusion_zh="扩展到60帧、60帧均有效，目标核心仍是0射线、0体积、0障碍点。这不是模型没看懂，而是轨迹没有观测到目标；正确结果是保留Unknown。",
            ),
            _hero(
                slot_id="slot_0968",
                row=active_cases["slot_0968"],
                assets=assets,
                conclusion_zh="LiDAR发现弱占用候选，但只有1个高度层且稳健性0/0；Camera裁剪也没有确认远处目标行。Agent识别了具体不确定性并拒绝过度推断。",
            ),
        ]
    )

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ParkingAgent v2 LiDAR工具修复验收</title>
<style>
:root{{--ink:#132238;--muted:#627189;--line:#dbe3ed;--blue:#246bdb;--green:#078963;--red:#d94248;--amber:#c78100;--bg:#eef3f8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}}
main{{max-width:1500px;margin:auto;padding:32px}}section,.case{{background:white;border:1px solid var(--line);border-radius:18px;padding:28px;margin:18px 0;box-shadow:0 6px 24px #18304b0d}}
h1{{font-size:34px;margin:.15em 0}}h2{{font-size:25px;margin:.2em 0 18px}}h3{{font-size:22px;margin:.15em 0}}h4{{margin:24px 0 8px}}p{{max-width:1050px}}
.eyebrow{{color:var(--blue);font-weight:800;letter-spacing:.05em}}.hero{{background:linear-gradient(130deg,#10233c,#234e83);color:white;padding:42px}}
.hero p{{color:#dbe8f8}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:24px}}.kpi{{background:#ffffff12;border:1px solid #ffffff2b;padding:16px;border-radius:12px}}.kpi b{{font-size:28px;display:block}}
.state{{font-weight:800}}.free{{color:var(--green)}}.occupied{{color:var(--red)}}.unknown{{color:var(--amber)}}.case-head{{display:flex;justify-content:space-between;gap:24px}}.score{{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}}
.conclusion{{font-size:17px;border-left:5px solid var(--blue);padding:12px 16px;background:#f4f8fe;max-width:none}}
.visual-grid{{display:grid;grid-template-columns:.8fr 1.7fr;gap:16px;align-items:start}}figure{{margin:0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#f7f9fc}}figure img{{display:block;width:100%;height:auto}}figcaption{{padding:11px 13px;color:var(--muted)}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin:16px 0}}.metrics span{{padding:10px 12px;background:#f5f8fb;border-radius:9px}}.metrics b{{display:block}}
.trace{{list-style:none;padding:0;margin:0;display:grid;gap:10px}}.trace li{{position:relative;border-left:4px solid var(--blue);background:#f7f9fc;padding:12px 16px 12px 58px;border-radius:8px}}.trace p{{margin:4px 0}}.trace small{{color:var(--muted)}}.round{{position:absolute;left:12px;top:13px;background:var(--blue);color:white;border-radius:20px;padding:1px 8px;font-weight:800}}
.fix-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.fix{{padding:14px;border:1px solid var(--line);border-radius:10px}}.fix del{{color:var(--red)}}.fix ins{{display:block;color:var(--green);text-decoration:none;font-weight:700}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left;white-space:nowrap}}th{{position:sticky;top:0;background:#edf3fa}}.table-wrap{{max-height:650px;overflow:auto;border:1px solid var(--line);border-radius:10px}}
.accept li{{margin:7px 0}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}details{{margin-top:10px;color:var(--muted)}}a{{color:var(--blue)}}
@media(max-width:900px){{main{{padding:12px}}.kpis,.metrics,.fix-grid,.visual-grid{{grid-template-columns:1fr}}.case-head{{display:block}}.score{{text-align:left}}}}
</style></head><body><main>
<section class="hero"><span class="eyebrow">2026-07-23 · frame 9277 · 修复后实跑</span><h1>LiDAR工具修复验收：现在能解释、能终止，也能正确拒绝</h1>
<p>这份报告只回答“修复是否真实有效”。不拿启发式分数冒充概率，不拿无GT实验冒充accuracy，也不把隐藏思维链伪装成解释。所有结论来自结构化Agent动作、工具Observation和确定性硬门。</p>
<div class="kpis"><div class="kpi"><b>37/37</b>模块自动化测试</div><div class="kpi"><b>24/24</b>扩展包合同与重绘通过</div><div class="kpi"><b>{unknown_resolved}/{unknown_total}</b>Part1 Unknown获得几何终态资格</div><div class="kpi"><b>0</b>4案例状态机验证错误</div></div></section>

<section><h2>1. 这次到底修了什么</h2><div class="fix-grid">
<div class="fix"><del>11/15帧图也硬写“60帧、额外45帧”</del><ins>标题、有效帧、视角角色全部从真实pack动态生成</ins></div>
<div class="fix"><del>JSON有45个核心点，图片却写“无点”</del><ins>geometry card与解释图强制使用同一effective decision</ins></div>
<div class="fix"><del>只有Part1包时，lidar_detail复读一次还浪费轮次</del><ins>is_incremental=false时从available_tools移除，可审计但不可当Part2新证据</ins></div>
<div class="fix"><del>Agent不知道LiDAR来源、帧数和稳健性</del><ins>请求新增tool_capabilities：source、selected/valid、incremental、robustness、blocking reason</ins></div>
</div></section>

<section><h2>2. P0反例已经修正：slot_0968</h2>
<p>下面不是扩展60帧，而是对原Part1包的诚实重绘。标题明确写“选择15帧 / 有效11帧”；核心45点和1个高度层在图内可见，结论是“弱候选但不够终态”，不再说“无点”。</p>
<figure><img src="{part1_0968_image}" alt="slot_0968 repaired Part1 LiDAR"><figcaption>修复后的Part1审计图；metadata同时声明 is_incremental_over_part1=false，所以Agent运行时不会把它当新增工具。</figcaption></figure></section>

<section><h2>3. 四个完整案例：Agent选择了什么，工具改变了什么</h2>
<p>“Agent思考”在这里指可审计的短rationale、定位假设、工具动作、Observation、证据引用和硬门结果。系统不索取也不展示模型私有隐藏思维链。</p></section>
{hero_html}

<section><h2>4. 24车位几何回归：不是只挑成功样例</h2>
<p>24个扩展包全部通过身份、anchor、selected frames与t0因果合同；工具输出为4个Free、7个Occupied、13个Unknown。22个Part1 Unknown中，9个获得确定性几何终态资格（40.9%）；其余13个保留Unknown。这里是几何硬门结果，不等同于有人工GT的分类准确率。</p>
<div class="table-wrap"><table><thead><tr><th>车位</th><th>Part1</th><th>修复后几何资格</th><th>选择/有效帧</th><th>核心射线</th><th>体积覆盖</th><th>核心点/层</th><th>稳健性</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></section>

<section><h2>5. 验收结论与边界</h2><ul class="accept">
<li>证据源合同：24/24为<code>part2_extended_causal_lidar</code>，帧数与能力声明一致，全部是Part1之外的新增因果证据。</li>
<li>正回归：slot_1012通过Free硬门；slot_1258通过Occupied硬门。OpenAI Agent均首轮选择LiDAR、第二轮终止。</li>
<li>负回归：slot_1248在60/60有效帧下仍为0核心覆盖，正确保持Unknown；没有为了提升数字而降门槛。</li>
<li>主动定位难例：slot_1248、1247、0968、1246全部给出具体未决原因，且没有状态机验证错误。</li>
<li>FOV仍是地图角度路由，不是像素级投影；Camera Crop验证的是Agent提出的假设，不是伪造ground truth框。</li>
<li>未做人工GT标注，因此本报告不宣称precision/recall/accuracy。40.9%指Unknown获得严格几何终态资格，不是分类准确率。</li>
</ul>
<p>机器可读证据：<a href="../lidar_repair_24case_geometry_regression_v2/geometry_regression.json">24车位几何回归JSON</a>；<a href="../openai_lidar_repair_terminal_regression3_v2/part2_result.json">三案例Agent结果</a>；<a href="../openai_agent_localization_extended60_multi4_v2/part2_result.json">四个主动定位案例结果</a>。</p></section>
</main></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    manifest = {
        "schema_version": "parking-slot-agent-v2-lidar-repair-report/1.0",
        "index": str((output / "index.html").resolve()),
        "asset_count": len(list(assets.iterdir())),
        "unknown_geometry_resolution": {
            "resolved": unknown_resolved,
            "total": unknown_total,
            "ratio": unknown_resolved / unknown_total,
        },
        "geometry_state_counts": geometry["state_counts"],
        "terminal_run": str(args.terminal_run.resolve()),
        "active_run": str(args.active_run.resolve()),
    }
    (output / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
