"""Build a clear Chinese HTML report spanning active traces and a full batch."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .reporting_agent_replay import build_agent_replay
from .reporting_localization_trace import build_localization_trace_report


STATE_ZH = {"free": "空闲", "occupied": "占用", "unknown": "未知"}
FOV_ZH = {
    "visible": "可见",
    "partially_visible": "部分可见",
    "uncertain": "不确定",
    "not_visible": "不可见",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _complete_active_cases(
    specs: Sequence[tuple[Path, Path, str]],
    output: Path,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for run_dir, part1_path, slot_id in specs:
        case_path = run_dir / "slot_cases" / f"{slot_id}.json"
        if not case_path.is_file():
            raise ValueError(f"active case result is incomplete: {case_path}")
        case_output = output / "active_cases" / slot_id
        result = build_localization_trace_report(
            run_dir=run_dir,
            part1_path=part1_path,
            output_dir=case_output,
            slot_id=slot_id,
        )
        trace = _read(Path(result["trace"]))
        trace["report_paths"] = {
            key: _relative(value, output) for key, value in result.items()
        }
        trace["assets"] = {
            key: f"active_cases/{slot_id}/{value}"
            for key, value in trace.get("assets", {}).items()
        }
        cases.append(trace)
    return cases


def _markdown(payload: Mapping[str, Any]) -> str:
    overview = payload["batch"]["overview"]
    rows = overview["all_cases"]
    lines = [
        "# Frame 9277 · Part2 Agent完整可视化报告", "",
        "## 结论", "",
        f"批量实验处理{overview['total_cases']}个车位，Part1 Unknown为{overview['unknown_inputs']}个，其中{overview['resolved_unknown']}个通过确定性硬门转成Free或Occupied，门控消歧率为{overview['unknown_resolution_rate']*100:.1f}%。", "",
        f"主动Map↔Camera协议另外验证{len(payload['active_cases'])}个Camera案例；这些案例用于检查FOV、目标定位、工具选择和验证效果，不与旧批量结果混算准确率。", "",
        "> 没有人工Ground Truth，因此“门控消歧率”不是分类准确率。", "",
        "## 为什么1248有时是Sequence、有时是Crop", "",
        "这是两次不同版本的真实运行，不是同一次轨迹前后矛盾：", "",
        "- 冻结60帧批量旧协议：Runtime先自动放入LiDAR证据，Agent随后选择Camera Context → Camera Sequence → Final Unknown。",
        "- 当前主动定位协议：Runtime只强制FOV，Agent自主选择LiDAR → Camera Context → Camera Crop → Final Unknown。",
        "- 当前协议要求显式记录定位假设、bbox和supported/refuted/ambiguous，因此Agent在看到语义地图后选择Crop反证候选区域。", "",
        "## 主动定位案例", "",
    ]
    for case in payload["active_cases"]:
        summary = case["summary"]
        lines.extend([
            f"### {case['slot_id']}", "",
            f"- 状态：{STATE_ZH.get(case['part1_state'], case['part1_state'])} → {STATE_ZH.get(case['final_state'], case['final_state'])}",
            f"- 方位变化：{summary['direction_change']}",
            f"- 定位置信度：{summary['initial_confidence']:.2f} → {summary['final_confidence']:.2f}",
            f"- 验证阶段：{summary['final_stage']}", "",
        ])
    lines.extend([
        "## 24个车位全部结果", "",
        "| 车位 | Part1 | Part2 | FOV | 模型轮次 | 最终置信度 | Agent动作路径 |",
        "|---|---|---|---|---:|---:|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['slot_id']} | {STATE_ZH.get(row['part1_state'], row['part1_state'])} | "
            f"{STATE_ZH.get(row['final_state'], row['final_state'])} | {FOV_ZH.get(row['fov'], row['fov'])} | "
            f"{row['model_turns']} | {row['final_confidence']*100:.0f}% | {row['action_path']} |"
        )
    lines.extend([
        "", "## 审计边界", "",
        "- 主动定位案例使用当前结构化定位协议；24车位批量统计来自冻结的60帧批量运行。两者在HTML中分区展示。",
        "- 报告展示模型保存的结构化rationale、localization和action，不展示或伪造隐藏思维链。",
        "- Agent置信度尚未概率校准，不能解释为真实准确率。", "",
    ])
    return "\n".join(lines)


def _document(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ParkingAgent · 完整Agent可视化报告</title><style>
:root{{--ink:#122137;--muted:#62758b;--line:#d8e2ed;--bg:#f3f6fa;--blue:#2472d8;--purple:#7653ca;--green:#13936e;--red:#de5155;--amber:#e5a01c}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,system-ui,"Microsoft YaHei",sans-serif}}nav{{position:sticky;top:0;z-index:10;background:#102039f2;color:#fff;padding:11px 24px;display:flex;gap:22px;align-items:center}}nav b{{margin-right:auto}}nav a{{color:#cfe3ff;text-decoration:none}}main{{max-width:1540px;margin:auto;padding:30px}}h1{{font-size:35px;margin:0 0 5px}}h2{{font-size:25px;margin:34px 0 13px}}h3{{margin:3px 0 8px}}.lead{{color:var(--muted);max-width:1050px}}.banner{{background:#fff7e4;border:1px solid #ebc76b;border-left:6px solid var(--amber);padding:13px 16px;border-radius:11px;margin:18px 0}}.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}.metric{{background:white;border:1px solid var(--line);border-radius:14px;padding:15px}}.metric b{{display:block;font-size:27px;line-height:1.2}}.metric span{{font-size:12px;color:var(--muted)}}.workflow{{display:grid;grid-template-columns:repeat(5,1fr);gap:22px}}.flow{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:15px;position:relative}}.flow:not(:last-child):after{{content:'→';position:absolute;right:-18px;top:42%;color:var(--blue);font-size:23px}}.flow em{{display:block;color:var(--blue);font-style:normal;font-size:12px;font-weight:800}}.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}}button{{font:inherit}}.tab{{border:1px solid var(--line);background:#fff;border-radius:10px;padding:9px 13px;cursor:pointer}}.tab.active{{background:var(--blue);border-color:var(--blue);color:#fff;font-weight:800}}.case-head{{display:grid;grid-template-columns:1.1fr 1fr;gap:12px;margin-bottom:12px}}.panel{{background:#fff;border:1px solid var(--line);border-radius:15px;padding:17px}}.case-metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}.mini{{background:#f1f5fa;border-radius:10px;padding:10px}}.mini b{{display:block;font-size:19px}}.finding{{margin:7px 0;padding:9px 11px;background:#f5f2ff;border-left:4px solid var(--purple)}}.board{{width:100%;display:block;background:#fff;border:1px solid var(--line);border-radius:15px}}.turn-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}}.turn{{background:#102039;color:#e7eef7;border-radius:13px;padding:14px;min-height:220px}}.turn .k{{color:#5fb1ff;font-size:11px;font-weight:800}}.turn .action{{background:#f4ad23;color:#102039;font-weight:800;padding:8px 10px;border-radius:8px;margin:10px 0}}.turn .loc{{color:#c0d0e2;font-size:12px}}.turn details pre{{max-height:260px}}.gallery{{display:grid;grid-template-columns:1fr 1.6fr 1fr;gap:10px;margin-top:12px}}figure{{margin:0;background:#fff;border:1px solid var(--line);border-radius:13px;padding:10px}}figure img{{width:100%;height:330px;object-fit:contain;background:#102039}}figcaption{{padding:8px;color:var(--muted)}}.batch-boards{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.batch-boards img{{width:100%;height:310px;object-fit:contain;background:#0b1627;border-radius:10px}}.tools{{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}}input,select{{border:1px solid var(--line);background:#fff;border-radius:9px;padding:9px 11px}}.table-wrap{{overflow:auto;max-height:720px;background:#fff;border:1px solid var(--line);border-radius:14px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th{{position:sticky;top:0;background:#eaf1f8;z-index:2;text-align:left}}th,td{{padding:9px 10px;border-bottom:1px solid #e5ebf2;vertical-align:top}}td.path{{min-width:340px}}.state{{font-weight:800}}.free{{color:var(--green)}}.occupied{{color:var(--red)}}.unknown{{color:var(--amber)}}pre{{white-space:pre-wrap;overflow:auto;background:#0c1727;color:#d7e4f2;padding:11px;border-radius:8px;font-size:11px}}.foot{{color:var(--muted);font-size:13px;margin-top:30px}}@media(max-width:1050px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.workflow,.turn-grid,.batch-boards{{grid-template-columns:1fr 1fr}}.case-head,.gallery{{grid-template-columns:1fr}}}}@media(max-width:650px){{main{{padding:14px}}nav a{{display:none}}.workflow,.turn-grid,.batch-boards{{grid-template-columns:1fr}}}}
.batch-boards{{grid-template-columns:1fr}}.batch-boards img{{height:auto;display:block}}
</style></head><body><nav><b>ParkingAgent · frame 9277</b><a href="#summary">总结果</a><a href="#version">1248版本对照</a><a href="#active">主动定位案例</a><a href="#all">全部24车位</a><a href="#audit">审计边界</a></nav><main>
<section id="summary"><h1>Part2 Agent完整可视化与结果报告</h1><p class="lead">先回答“Agent做了什么、工具产生了什么变化”，再展示当前协议的多个主动定位案例，最后给出冻结批量实验的24个车位完整结果。所有数字都绑定到真实运行目录。</p><div class="banner"><b>实验版本必须分开看：</b>24车位总体统计来自冻结的60帧批量旧运行；主动Map↔Camera定位来自当前协议的独立Camera案例。前者回答总体门控消歧，后者回答FOV、目标定位、工具选择与验证过程，二者不是同一次执行。</div>
<div class="metrics"><div class="metric"><b id="total">—</b><span>批量车位</span></div><div class="metric"><b id="unknownIn">—</b><span>Part1 Unknown</span></div><div class="metric"><b id="resolved">—</b><span>通过硬门消歧</span></div><div class="metric"><b id="rate">—</b><span>门控消歧率</span></div><div class="metric"><b id="dist">—</b><span>Free / Occupied / Unknown</span></div><div class="metric"><b id="activeN">—</b><span>当前主动定位案例</span></div></div></section>
<h2>统一工作流</h2><section class="workflow"><div class="flow"><em>0 · PART1</em><b>读取原SlotCase</b><p>保留原状态、分数和Unknown原因。</p></div><div class="flow"><em>1 · FOV</em><b>强制几何路由</b><p>只判断Camera是否合法，不伪装像素投影。</p></div><div class="flow"><em>2 · AGENT</em><b>主动选择工具</b><p>LiDAR、全景、Crop或因果序列。</p></div><div class="flow"><em>3 · VERIFY</em><b>支持或反证假设</b><p>记录方位、行序、bbox、地标与歧义。</p></div><div class="flow"><em>4 · VALIDATOR</em><b>硬门决定写回</b><p>Agent提议，代码核验证据和阈值。</p></div></section>
<section id="version"><h2>1248为什么一会Sequence、一会Crop？</h2><div class="panel"><p><b>因为这是两次不同版本的真实运行。</b>旧批量运行和当前主动定位运行不能拼成一条轨迹。</p><div class="case-metrics"><div class="mini"><span>冻结60帧批量旧协议</span><b>预加载LiDAR</b><p>Agent：Camera Context → Camera Sequence → Final Unknown</p></div><div class="mini"><span>当前主动定位协议</span><b>只预执行FOV</b><p>Agent：LiDAR → Camera Context → Camera Crop → Final Unknown</p></div><div class="mini"><span>动作变化原因</span><b>定位合同改变</b><p>新协议要求显式bbox与验证状态，因此选择Crop反证同一帧候选区。</p></div></div></div></section>
<section id="active"><h2>当前协议：多个主动定位案例</h2><div id="activeOverview" class="banner"></div><div id="activeTabs" class="tabs"></div><div class="case-head"><div class="panel"><h3 id="caseTitle"></h3><div id="caseMetrics" class="case-metrics"></div><div id="findings"></div></div><div class="panel"><h3>这例应当如何解释</h3><p id="interpret"></p><p><a id="caseHtml" target="_blank">打开该案例独立HTML和完整原始action</a></p></div></div><img id="activeBoard" class="board"><div id="turns" class="turn-grid"></div><div id="gallery" class="gallery"></div></section>
<section id="all"><h2>冻结60帧批量旧协议：24个车位全部结果</h2><div class="banner"><b>路径解释：</b>本表中的LiDAR由旧Runtime在模型首轮前自动提供，不等于当前协议中Agent主动调用LiDAR。表格仅用于报告冻结批量的总体门控结果。</div><div class="tools"><input id="search" placeholder="搜索车位，例如 1248"><select id="stateFilter"><option value="all">全部终态</option><option value="free">Free</option><option value="occupied">Occupied</option><option value="unknown">Unknown</option></select><select id="transitionFilter"><option value="all">全部变化</option><option value="resolved">Unknown→终态</option><option value="unresolved">Unknown→Unknown</option><option value="retained">原终态保持</option></select><span id="visibleCount"></span></div><div class="table-wrap"><table><thead><tr><th>车位</th><th>Part1</th><th>Part2</th><th>变化</th><th>FOV</th><th>模型轮次</th><th>最终置信度</th><th>旧协议动作路径</th><th>仍未解决</th></tr></thead><tbody id="resultBody"></tbody></table></div></section>
<section id="audit"><h2>审计边界与结论</h2><div class="panel"><ul><li><b>已证明：</b>Agent动作、定位假设、工具返回和Validator结果均可追踪；批量22个Unknown中9个通过内部硬门消歧。</li><li><b>未证明：</b>没有冻结人工Ground Truth，40.9%是门控消歧率，不是准确率。</li><li><b>定位实验：</b>工具可以提高、降低或修正定位置信度；降低同样是有效反证，不能只展示成功加分。</li><li><b>思考展示：</b>仅展示结构化rationale/localization/action，不伪造不可审计的隐藏思维链。</li></ul></div></section><p class="foot">机器可读总清单：complete_results.json；中文静态报告：REPORT.md；每个主动案例均有独立HTML、PNG与trace.json。</p>
</main><script id="data" type="application/json">{data}</script><script>
const D=JSON.parse(document.getElementById('data').textContent),zh={{free:'空闲',occupied:'占用',unknown:'未知'}},fovZh={{visible:'可见',partially_visible:'部分可见',uncertain:'不确定',not_visible:'不可见'}};let ai=0;const $=x=>document.getElementById(x);const esc=x=>String(x??'').replace(/[&<>\"]/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[m]));
function metrics(){{const o=D.batch.overview,a=D.active_overview;$('total').textContent=o.total_cases;$('unknownIn').textContent=o.unknown_inputs;$('resolved').textContent=o.resolved_unknown;$('rate').textContent=(o.unknown_resolution_rate*100).toFixed(1)+'%';$('dist').textContent=`${{o.final_counts.free}} / ${{o.final_counts.occupied}} / ${{o.final_counts.unknown}}`;$('activeN').textContent=D.active_cases.length;$('activeOverview').innerHTML=`<b>${{a.completed}}例主动定位总体：</b>全部完成3次Agent工具调用；最终定位阶段 ${{esc(JSON.stringify(a.stage_counts))}}；方位修正 ${{a.direction_revisions}}例；定位置信度上升/下降/不变 = ${{a.confidence_up}} / ${{a.confidence_down}} / ${{a.confidence_same}}；最终Free/Occupied/Unknown = ${{a.final_counts.free}} / ${{a.final_counts.occupied}} / ${{a.final_counts.unknown}}。`}}
function renderActive(){{const c=D.active_cases[ai],s=c.summary;$('activeTabs').innerHTML=D.active_cases.map((x,i)=>`<button class="tab ${{i===ai?'active':''}}" onclick="pickActive(${{i}})">${{x.slot_id}} · ${{zh[x.final_state]}}</button>`).join('');$('caseTitle').textContent=`${{c.slot_id}}：${{zh[c.part1_state]}} → ${{zh[c.final_state]}}`;$('caseMetrics').innerHTML=`<div class="mini"><span>方位变化</span><b>${{esc(s.direction_change)}}</b></div><div class="mini"><span>定位置信度</span><b>${{s.initial_confidence.toFixed(2)}} → ${{s.final_confidence.toFixed(2)}}</b></div><div class="mini"><span>验证阶段</span><b>${{esc(s.final_stage)}}</b></div>`;$('findings').innerHTML=c.findings.map(x=>`<div class="finding">${{esc(x)}}</div>`).join('');$('interpret').textContent=c.final_state==='unknown'?'工具未把证据推到终态硬门；如果定位置信度下降，表示Crop/Sequence反证了先前假设，并非工具失败。':'Agent终态提议仍必须通过代码硬门，定位增益本身不能替代占用证据。';$('caseHtml').href=c.report_paths.html;$('activeBoard').src=c.report_paths.board;
$('turns').innerHTML=c.turns.map(t=>{{const l=t.localization||{{}};return `<article class="turn"><div class="k">OPENAI TURN ${{t.turn}}</div><h3>${{esc(t.decision)}}</h3><div class="loc">阶段：${{esc(l.stage)}}<br>方位：${{esc(l.target_side)}} · 深度：${{esc(l.depth_band)}}<br>定位置信度：${{Number(l.confidence_after||0).toFixed(2)}}<br>假设：${{esc(l.hypothesis_id||'—')}}</div><div class="action">${{esc(t.decision)}}</div><p>${{esc(t.rationale)}}</p><details><summary>原始结构化action</summary><pre>${{esc(JSON.stringify(t.raw_action,null,2))}}</pre></details></article>`}}).join('');const panels=[['semantic_map','语义地图：目标方位与周围车位行'],['hypothesis_overlay','Agent提出的bbox；不是Ground Truth'],['crop','Crop/验证结果']];$('gallery').innerHTML=panels.map(([k,l])=>`<figure><img src="${{c.assets[k]}}"><figcaption>${{l}}</figcaption></figure>`).join('')}}function pickActive(i){{ai=i;renderActive()}}
function transition(r){{if(r.part1_state==='unknown'&&r.final_state!=='unknown')return'resolved';if(r.part1_state==='unknown'&&r.final_state==='unknown')return'unresolved';return'retained'}}function renderTable(){{const q=$('search').value.trim().toLowerCase(),sf=$('stateFilter').value,tf=$('transitionFilter').value;const rows=D.batch.overview.all_cases.filter(r=>(!q||r.slot_id.toLowerCase().includes(q))&&(sf==='all'||r.final_state===sf)&&(tf==='all'||transition(r)===tf));$('visibleCount').textContent=`显示 ${{rows.length}} / ${{D.batch.overview.all_cases.length}}`;$('resultBody').innerHTML=rows.map(r=>`<tr><td><b>${{r.slot_id}}</b></td><td class="state ${{r.part1_state}}">${{zh[r.part1_state]}}</td><td class="state ${{r.final_state}}">${{zh[r.final_state]}}</td><td>${{transition(r)==='resolved'?'Unknown→终态':transition(r)==='unresolved'?'仍未解决':'保持终态'}}</td><td>${{fovZh[r.fov]||r.fov}}</td><td>${{r.model_turns}}</td><td>${{(r.final_confidence*100).toFixed(0)}}%</td><td class="path">${{esc(r.action_path)}}</td><td>${{esc((r.unresolved||[]).join('；')||'—')}}</td></tr>`).join('')}}
$('search').oninput=renderTable;$('stateFilter').onchange=renderTable;$('transitionFilter').onchange=renderTable;metrics();renderActive();renderTable();
</script></body></html>"""


def build_complete_agent_report(
    *,
    batch_run_dir: str | Path,
    active_specs: Sequence[tuple[str | Path, str | Path, str]],
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    batch_output = output / "batch_replay"
    build_agent_replay(run_dir=batch_run_dir, output_dir=batch_output)
    batch = _read(batch_output / "replay_data.json")
    for case in batch["cases"]:
        case["decision_board"] = f"batch_replay/{case['decision_board']}"
    active = _complete_active_cases(
        [(Path(run).resolve(), Path(part1).resolve(), slot) for run, part1, slot in active_specs],
        output,
    )
    active_overview = {
        "completed": len(active),
        "stage_counts": {
            stage: sum(case["final_localization"].get("stage") == stage for case in active)
            for stage in ("supported", "refuted", "ambiguous", "not_visible", "not_attempted")
        },
        "direction_revisions": sum(
            "→" in str(case["summary"]["direction_change"]) for case in active
        ),
        "confidence_up": sum(
            case["summary"]["final_confidence"] > case["summary"]["initial_confidence"]
            for case in active
        ),
        "confidence_down": sum(
            case["summary"]["final_confidence"] < case["summary"]["initial_confidence"]
            for case in active
        ),
        "confidence_same": sum(
            case["summary"]["final_confidence"] == case["summary"]["initial_confidence"]
            for case in active
        ),
        "final_counts": {
            state: sum(case["final_state"] == state for case in active)
            for state in ("free", "occupied", "unknown")
        },
    }
    payload = {
        "schema_version": "parking-agent-complete-report/1.0",
        "batch": batch,
        "active_cases": active,
        "active_overview": active_overview,
        "provenance": {
            "batch_run": str(Path(batch_run_dir).resolve()),
            "active_runs": sorted({str(Path(run).resolve()) for run, _, _ in active_specs}),
        },
    }
    html_path = output / "index.html"
    html_path.write_text(_document(payload), encoding="utf-8")
    data_path = output / "complete_results.json"
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = output / "REPORT.md"
    report_path.write_text(_markdown(payload), encoding="utf-8")
    return {
        "html": str(html_path), "data": str(data_path), "report": str(report_path),
        "active_case_count": len(active),
        "batch_case_count": int(batch["overview"]["total_cases"]),
    }


__all__ = ["build_complete_agent_report"]
