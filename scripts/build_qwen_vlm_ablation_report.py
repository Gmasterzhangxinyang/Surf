#!/usr/bin/env python3
"""Build the frame6241 same-Agent VLM model-size ablation report."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "Nature_ParkingAgent_实验报告_20260728"
FRAME = REPORT / "frame6241_blind"
OUT = FRAME / "qwen_ablation"
DEPLOYMENT = OUT / "deployment_manifest.json"

SPECS = [
    {
        "key": "openai",
        "label": "OpenAI gpt-5.6-terra",
        "size": "API reference",
        "run": FRAME / "pose_tuned_run/camera_first_final_full11_w45k3_gt_scope",
        "metrics": FRAME / "locked_gt_experiments/final_camera_first_agent_metrics.json",
    },
    {
        "key": "qwen2b",
        "label": "Qwen3-VL-2B-Instruct",
        "size": "2B · local BF16",
        "run": OUT / "qwen3_vl_2b_instruct_full11",
        "metrics": OUT / "qwen3_vl_2b_instruct_full11/metrics.json",
    },
    {
        "key": "qwen08b",
        "label": "Qwen3.5-0.8B",
        "size": "0.8B · local BF16 (0.7B class)",
        "run": OUT / "qwen3_5_0_8b_strict_full11",
        "metrics": OUT / "qwen3_5_0_8b_strict_full11/metrics.json",
    },
]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPORT.resolve()).as_posix()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(value: str) -> str:
    value = value.lower()
    return f"<span class='state {esc(value)}'>{esc(value.title())}</span>"


def tool_counts(trace: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for slot in trace.get("slots", []):
        for step in slot.get("steps", []):
            execute = step.get("execute") or {}
            tool = execute.get("tool")
            if tool:
                counts[str(tool)] += 1
    return dict(sorted(counts.items()))


def raw_action_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "recorded_calls": None,
            "calls_with_images": None,
            "raw_action_types": {},
            "raw_final_states": {},
        }
    calls = load(path)
    action_types: Counter[str] = Counter()
    final_states: Counter[str] = Counter()
    with_images = 0
    for call in calls:
        action = call.get("action") or {}
        action_type = str(action.get("type", "missing"))
        action_types[action_type] += 1
        if action_type == "final":
            final_states[str(action.get("state", "missing"))] += 1
        if int(call.get("image_count", 0)) > 0:
            with_images += 1
    return {
        "recorded_calls": len(calls),
        "calls_with_images": with_images,
        "raw_action_types": dict(sorted(action_types.items())),
        "raw_final_states": dict(sorted(final_states.items())),
    }


def collect(spec: dict[str, Any]) -> dict[str, Any]:
    run = spec["run"]
    metrics = load(spec["metrics"])
    summary = load(run / "summary.json")
    trace = load(run / "agent_plan_execute_observe.json")
    prompt_path = run / "system_prompt.txt"
    method = metrics["methods"]["best_part1_w45k3_plus_camera_first_agent"]
    processed = metrics["subsets"]["agent_processed_11"]
    visible = metrics["subsets"]["camera_visible_processed"]
    raw = raw_action_stats(run / "local_model_calls.json")
    return {
        "key": spec["key"],
        "label": spec["label"],
        "size": spec["size"],
        "model": summary.get("model"),
        "architecture": summary.get("architecture"),
        "prediction_blind": bool(summary.get("prediction_blind")),
        "gt_used": bool(summary.get("gt_used")),
        "processed": int(summary.get("processed", 0)),
        "replay_verified": bool(summary.get("replay_verified")),
        "prompt_sha256": digest(prompt_path),
        "metrics_path": rel(spec["metrics"]),
        "run_path": rel(run),
        "all14_accuracy": method["three_class_accuracy"],
        "all14_terminal_coverage": method["terminal_coverage"],
        "all14_terminal_selective_accuracy": method["terminal_selective_accuracy"],
        "agent11_accuracy": processed["three_class_accuracy"],
        "agent11_terminal_coverage": processed["terminal_coverage"],
        "camera_visible_accuracy": visible["three_class_accuracy"],
        "false_occupied": method["false_occupied_count"],
        "false_free": method["false_free_count"],
        "prediction_counts": method["prediction_counts"],
        "first_tool_camera_count": int(metrics.get("first_tool_camera_count", 0)),
        "camera_first_compliance": metrics.get("camera_first_policy_compliance_rate"),
        "accepted_tool_counts": summary.get("tool_counts", tool_counts(trace)),
        "proposed_tool_counts": tool_counts(trace),
        "model_calls": summary.get("model_calls"),
        "mean_model_call_seconds": summary.get("mean_model_call_seconds"),
        "wall_seconds": summary.get("wall_seconds"),
        "validation_error_count": int(summary.get("validation_error_count", 0)),
        "rows": method["rows"],
        **raw,
    }


def main() -> None:
    required = [DEPLOYMENT]
    for spec in SPECS:
        required.extend([
            spec["metrics"],
            spec["run"] / "summary.json",
            spec["run"] / "agent_plan_execute_observe.json",
            spec["run"] / "system_prompt.txt",
        ])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing ablation inputs: " + "; ".join(missing))

    models = [collect(spec) for spec in SPECS]
    qwen_rows = [row for row in models if row["key"].startswith("qwen")]
    qwen_prompt_identical = len({row["prompt_sha256"] for row in qwen_rows}) == 1
    openai_reference_same_prompt = (
        models[0]["prompt_sha256"] == qwen_rows[0]["prompt_sha256"]
    )
    result = {
        "schema_version": "frame6241-vlm-model-size-ablation/1.0",
        "experimental_control": {
            "same_anchor": 6241,
            "same_gt": "frame6241_gt_locked.csv (6 Free / 2 Occupied / 6 Unknown)",
            "same_part1": "W45/K3",
            "same_scope": "11 front-180 GT IDs; merge evaluated on 14 locked GT IDs",
            "same_agent_loop": "bounded Plan-Execute-Observe",
            "same_tools": ["camera_context", "camera_crop", "lidar_detail"],
            "same_threshold": 0.6,
            "qwen_pair_prompt_identical": qwen_prompt_identical,
            "openai_reference_same_prompt": openai_reference_same_prompt,
            "openai_prompt_difference": (
                "The frozen OpenAI reference predates the camera_crop additions; "
                "it is contextual, not part of the strict 2B-vs-0.8B model-only pair."
            ),
            "gt_visible_to_models": False,
        },
        "models": [{k: v for k, v in row.items() if k != "rows"} for row in models],
    }
    output_json = OUT / "qwen_vlm_ablation_comparison.json"
    output_csv = OUT / "qwen_vlm_ablation_rows.csv"
    output_html = REPORT / "frame6241_qwen_vlm_ablation.html"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    by_model = {row["key"]: {item["slot_id"]: item for item in row["rows"]} for row in models}
    slot_ids = [item["slot_id"] for item in models[0]["rows"]]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slot_id", "gt_state", "gt_observability", "openai", "qwen_2b", "qwen_0_8b"])
        for sid in slot_ids:
            base = by_model["openai"][sid]
            writer.writerow([
                sid,
                base["gt_state"],
                base["gt_observability"],
                by_model["openai"][sid]["prediction"],
                by_model["qwen2b"][sid]["prediction"],
                by_model["qwen08b"][sid]["prediction"],
            ])

    metric_rows = []
    for row in models:
        raw_final = row["raw_final_states"] or "API audit见原始轨迹"
        latency = "—" if row["mean_model_call_seconds"] is None else f"{row['mean_model_call_seconds']:.2f}s"
        metric_rows.append(
            "<tr>"
            f"<td><b>{esc(row['label'])}</b><br><small>{esc(row['size'])}</small></td>"
            f"<td>{pct(row['all14_accuracy'])}</td><td>{pct(row['agent11_accuracy'])}</td>"
            f"<td>{pct(row['camera_visible_accuracy'])}</td><td>{pct(row['all14_terminal_coverage'])}</td>"
            f"<td>{row['false_occupied']} / {row['false_free']}</td>"
            f"<td>{row['first_tool_camera_count']}/11</td><td>{esc(row['accepted_tool_counts'])}</td>"
            f"<td>{esc(raw_final)}</td><td>{row['validation_error_count']}</td>"
            f"<td>{esc(latency)}</td><td>{str(row['replay_verified']).lower()}</td></tr>"
        )

    per_slot_rows = []
    for sid in slot_ids:
        base = by_model["openai"][sid]
        cells = []
        for key in ("openai", "qwen2b", "qwen08b"):
            item = by_model[key][sid]
            mark = "✓" if item["correct"] else "✗"
            cells.append(f"<td>{state(item['prediction'])} <b class='{'ok' if item['correct'] else 'bad'}'>{mark}</b></td>")
        per_slot_rows.append(
            f"<tr><td>{esc(sid)}</td><td>{state(base['gt_state'])}</td>"
            f"<td>{esc(base['gt_observability'])}</td>{''.join(cells)}</tr>"
        )

    q08 = next(row for row in models if row["key"] == "qwen08b")
    q2 = next(row for row in models if row["key"] == "qwen2b")
    html_text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Frame6241 · Qwen VLM模型消融</title>
<style>:root{{--bg:#f5f7fb;--ink:#172033;--muted:#667085;--line:#dce3ed;--blue:#2563eb;--green:#12865f;--red:#d74747}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,system-ui,"Microsoft YaHei",sans-serif}}
header,main{{max-width:1450px;margin:auto;padding:30px 4vw}}h1{{font-size:44px;line-height:1.1}}section{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px;margin:16px 0}}.note{{padding:13px;background:#eef4ff;border-left:4px solid var(--blue)}}.warn{{background:#fff5e7;border-left-color:#d98b13}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#eaf0f8;position:sticky;top:0}}.state{{font-weight:800}}.free,.ok{{color:var(--green)}}.occupied,.bad{{color:var(--red)}}.unknown{{color:#7a8494}}.links{{display:flex;gap:8px;flex-wrap:wrap}}.links a{{text-decoration:none;border:1px solid var(--line);padding:8px 10px;border-radius:8px}}</style></head><body>
<header><div style='color:#2563eb;font-weight:800'>FRAME 6241 · SAME AGENT / SAME TOOLS / MODEL SWAP ONLY</div>
<h1>自部署Qwen VLM<br>2B与0.8B消融</h1><p>Qwen 2B与0.8B严格冻结Part1、11个Part2输入、系统提示、工具合同、0.6终态阈值和人工GT，仅替换中心VLM。Qwen 0.8B是用户所说“0.7B档”的当前官方近似规格。OpenAI列作为既有参考，提示词版本早于Crop增补，不计入严格模型大小归因。</p></header><main>
<section><h2>结论</h2><div class='note'><b>2B：</b>14-GT准确率 {pct(q2['all14_accuracy'])}，Agent 11例准确率 {pct(q2['agent11_accuracy'])}，Camera-visible {pct(q2['camera_visible_accuracy'])}，首工具Camera {q2['first_tool_camera_count']}/11。</div>
<div class='note warn'><b>0.8B的关键失败不是“阈值过高”：</b>它在尚未调用Camera时原始提出 {esc(q08['raw_final_states'])}，其中模型输入图像数为0；运行时正确拒绝这些无证据Occupied，最终11例均回退Unknown。若降低终态阈值，只会把幻觉放进系统。</div></section>
<section><h2>严格受控对比</h2><p>Qwen pair prompt_identical={str(qwen_prompt_identical).lower()}；OpenAI reference same_prompt={str(openai_reference_same_prompt).lower()}；prediction_blind=true；GT仅在运行完成后进入评测。0.8B标记为0.7B class，不虚构一个不存在的“Qwen 0.7B VLM”型号。</p><div class='scroll'><table><thead><tr><th>模型</th><th>14-GT准确率</th><th>Agent 11准确率</th><th>Camera-visible</th><th>终态覆盖</th><th>假O/F</th><th>首Camera</th><th>接受工具调用</th><th>原始Final提议</th><th>契约错误</th><th>均次延迟</th><th>Replay</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div></section>
<section><h2>逐车位结果</h2><div class='scroll'><table><thead><tr><th>slot</th><th>GT</th><th>可观测性</th><th>OpenAI</th><th>Qwen 2B</th><th>Qwen 0.8B</th></tr></thead><tbody>{''.join(per_slot_rows)}</tbody></table></div></section>
<section><h2>如何解释</h2><p>这项实验同时测“视觉分类能力”和“能否遵守Agent协议”。模型必须先根据结构化available_tools自主选择Camera/Crop/LiDAR，再基于工具Observation输出F/O/U。安全契约属于同一架构，不能为某个小模型关闭。</p><p>单anchor仅能作为开发性消融；不能据此宣称跨路线泛化或Nature级统计显著性。这里的准确率只针对锁定的frame6241 GT。</p></section>
<section><h2>审计产物</h2><div class='links'><a href='{esc(rel(output_json))}'>完整对比JSON</a><a href='{esc(rel(output_csv))}'>逐车位CSV</a><a href='{esc(rel(DEPLOYMENT))}'>本地部署与权重校验</a><a href='{esc(q2['run_path'])}/agent_plan_execute_observe.json'>2B PEO</a><a href='{esc(q2['run_path'])}/local_model_calls.json'>2B模型动作</a><a href='{esc(q08['run_path'])}/agent_plan_execute_observe.json'>0.8B PEO</a><a href='{esc(q08['run_path'])}/local_model_calls.json'>0.8B模型动作</a></div></section>
</main></body></html>"""
    output_html.write_text(html_text, encoding="utf-8")
    print(json.dumps({
        "html": str(output_html),
        "json": str(output_json),
        "csv": str(output_csv),
        "qwen_pair_prompt_identical": qwen_prompt_identical,
        "openai_reference_same_prompt": openai_reference_same_prompt,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
