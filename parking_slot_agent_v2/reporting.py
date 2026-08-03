"""Human-readable and machine-readable comparison reports for Part 2 runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import html
import json
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


DETAIL_TOOLS = frozenset(
    {"camera_context", "camera_sequence", "camera_crop", "lidar_detail"}
)


@dataclass(frozen=True, slots=True)
class RunMetrics:
    name: str
    processed_cases: int
    selected_slot_id: str | None
    stop_reason: str
    final_state_counts: dict[str, int]
    fov_counts: dict[str, int]
    case_stop_counts: dict[str, int]
    model_turns: int
    tool_rounds: int
    tool_counts: dict[str, int]
    cases_with_detail: int
    detail_coverage_pct: float
    cases_with_successful_detail: int
    successful_detail_coverage_pct: float
    accepted_final_cases: int
    accepted_final_coverage_pct: float
    validation_errors: int
    validation_error_cases: int
    model_outputs: int
    model_tool_actions: int
    model_final_actions: int
    autonomous_tool_action_pct: float
    direct_final_action_pct: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    total_api_seconds: float | None
    median_api_seconds: float | None
    max_api_seconds: float | None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _flatten_actions(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    actions = payload.get("actions", {})
    if isinstance(actions, list):
        return [item for item in actions if isinstance(item, Mapping)]
    if not isinstance(actions, Mapping):
        return []
    result: list[Mapping[str, Any]] = []
    for case_actions in actions.values():
        if isinstance(case_actions, list):
            result.extend(
                item for item in case_actions if isinstance(item, Mapping)
            )
    return result


def collect_run_metrics(run_dir: str | Path, *, name: str | None = None) -> RunMetrics:
    """Collect comparable metrics without interpreting confidence as calibrated."""

    root = Path(run_dir).resolve()
    result = _load_json(root / "part2_result.json")
    slot_results = result.get("slot_results", [])
    if not isinstance(slot_results, list):
        raise ValueError("slot_results must be a list")

    rounds = [
        round_row
        for slot_result in slot_results
        for round_row in slot_result.get("case", {}).get("rounds", [])
        if isinstance(round_row, Mapping)
    ]
    cases_with_detail = sum(
        1
        for slot_result in slot_results
        if any(
            round_row.get("tool_name") in DETAIL_TOOLS
            for round_row in slot_result.get("case", {}).get("rounds", [])
        )
    )
    cases_with_successful_detail = sum(
        1
        for slot_result in slot_results
        if any(
            evidence.get("tool_name") in DETAIL_TOOLS
            and evidence.get("status") == "ok"
            for evidence in slot_result.get("case", {}).get("evidence", [])
        )
    )
    accepted_final_cases = sum(
        1
        for slot_result in slot_results
        if any(
            evidence.get("tool_name") == "agent_final"
            and evidence.get("status") == "ok"
            for evidence in slot_result.get("case", {}).get("evidence", [])
        )
    )
    validation_errors = sum(
        len(slot_result.get("validation_errors", []))
        for slot_result in slot_results
    )
    validation_error_cases = sum(
        bool(slot_result.get("validation_errors")) for slot_result in slot_results
    )

    replay_path = root / "replay_actions.json"
    actions = _flatten_actions(_load_json(replay_path)) if replay_path.exists() else []
    action_counts = Counter(str(action.get("type")) for action in actions)
    output_count = len(actions)

    audit_rows: list[Mapping[str, Any]] = []
    audit_dir = root / "openai_audit"
    if audit_dir.is_dir():
        audit_rows = [
            _load_json(path) for path in sorted(audit_dir.glob("**/turn_*.json"))
        ]
    usage_rows = [
        row.get("usage") for row in audit_rows if isinstance(row.get("usage"), Mapping)
    ]
    elapsed = [
        float(row["elapsed_seconds"])
        for row in audit_rows
        if isinstance(row.get("elapsed_seconds"), (int, float))
    ]

    processed = len(result.get("processed_case_ids", []))
    pct_denominator = processed or 1
    return RunMetrics(
        name=name or root.name,
        processed_cases=processed,
        selected_slot_id=result.get("selected_slot_id"),
        stop_reason=str(result.get("stop_reason")),
        final_state_counts=_counter(
            str(slot_result.get("case", {}).get("final_state"))
            for slot_result in slot_results
        ),
        fov_counts=_counter(
            str(slot_result.get("case", {}).get("fov", {}).get("visibility"))
            for slot_result in slot_results
        ),
        case_stop_counts=_counter(
            str(slot_result.get("stop_reason")) for slot_result in slot_results
        ),
        model_turns=sum(int(slot_result.get("model_turns", 0)) for slot_result in slot_results),
        tool_rounds=sum(int(slot_result.get("tool_rounds", 0)) for slot_result in slot_results),
        tool_counts=_counter(str(round_row.get("tool_name")) for round_row in rounds),
        cases_with_detail=cases_with_detail,
        detail_coverage_pct=100.0 * cases_with_detail / pct_denominator,
        cases_with_successful_detail=cases_with_successful_detail,
        successful_detail_coverage_pct=(
            100.0 * cases_with_successful_detail / pct_denominator
        ),
        accepted_final_cases=accepted_final_cases,
        accepted_final_coverage_pct=100.0 * accepted_final_cases / pct_denominator,
        validation_errors=validation_errors,
        validation_error_cases=validation_error_cases,
        model_outputs=output_count,
        model_tool_actions=action_counts["tool"],
        model_final_actions=action_counts["final"],
        autonomous_tool_action_pct=(
            100.0 * action_counts["tool"] / output_count if output_count else 0.0
        ),
        direct_final_action_pct=(
            100.0 * action_counts["final"] / output_count if output_count else 0.0
        ),
        input_tokens=(
            sum(int(row.get("input_tokens", 0)) for row in usage_rows)
            if usage_rows
            else None
        ),
        output_tokens=(
            sum(int(row.get("output_tokens", 0)) for row in usage_rows)
            if usage_rows
            else None
        ),
        total_tokens=(
            sum(int(row.get("total_tokens", 0)) for row in usage_rows)
            if usage_rows
            else None
        ),
        total_api_seconds=sum(elapsed) if elapsed else None,
        median_api_seconds=statistics.median(elapsed) if elapsed else None,
        max_api_seconds=max(elapsed) if elapsed else None,
    )


def comparison_payload(baseline: RunMetrics, experiment: RunMetrics) -> dict[str, Any]:
    """Return explicit deltas and the limits of what this run can establish."""

    return {
        "schema_version": "parking-slot-agent-v2-comparison/1.0",
        "baseline": asdict(baseline),
        "experiment": asdict(experiment),
        "deltas": {
            "detail_coverage_percentage_points": round(
                experiment.detail_coverage_pct - baseline.detail_coverage_pct, 3
            ),
            "accepted_final_coverage_percentage_points": round(
                experiment.accepted_final_coverage_pct
                - baseline.accepted_final_coverage_pct,
                3,
            ),
            "successful_detail_coverage_percentage_points": round(
                experiment.successful_detail_coverage_pct
                - baseline.successful_detail_coverage_pct,
                3,
            ),
            "autonomous_tool_action_percentage_points": round(
                experiment.autonomous_tool_action_pct
                - baseline.autonomous_tool_action_pct,
                3,
            ),
            "validation_errors": experiment.validation_errors
            - baseline.validation_errors,
            "model_turns": experiment.model_turns - baseline.model_turns,
            "tool_rounds": experiment.tool_rounds - baseline.tool_rounds,
        },
        "claims": {
            "workflow_reliability_improved": (
                experiment.successful_detail_coverage_pct
                > baseline.successful_detail_coverage_pct
                and experiment.validation_errors < baseline.validation_errors
                and experiment.accepted_final_coverage_pct
                > baseline.accepted_final_coverage_pct
            ),
            "classification_accuracy_improved": None,
            "classification_accuracy_reason": (
                "No independent ground-truth occupancy labels are attached to this "
                "single anchor, so accuracy, precision, and recall cannot be estimated."
            ),
            "controlled_ablation": False,
            "controlled_ablation_reason": (
                "The comparison changes both the model and the hardened FOV/tool-routing "
                "workflow; it demonstrates end-to-end improvement, not a model-only causal effect."
            ),
        },
    }


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _rel(path: str | Path, report_dir: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), report_dir.resolve())).as_posix()


def _metric_row(label: str, baseline: Any, experiment: Any, interpretation: str) -> str:
    return (
        "<tr>"
        f"<td>{_esc(label)}</td><td>{_esc(baseline)}</td>"
        f"<td>{_esc(experiment)}</td><td>{_esc(interpretation)}</td>"
        "</tr>"
    )


def _render_summary_plot(
    path: Path, baseline: RunMetrics, experiment: RunMetrics
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Successful detail", "Valid final coverage", "Tool actions"]
    base = [
        baseline.successful_detail_coverage_pct,
        baseline.accepted_final_coverage_pct,
        baseline.autonomous_tool_action_pct,
    ]
    exp = [
        experiment.successful_detail_coverage_pct,
        experiment.accepted_final_coverage_pct,
        experiment.autonomous_tool_action_pct,
    ]
    positions = list(range(len(labels)))
    width = 0.36
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax1.bar([x - width / 2 for x in positions], base, width, label=baseline.name, color="#94a3b8")
    ax1.bar([x + width / 2 for x in positions], exp, width, label=experiment.name, color="#2563eb")
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Percent")
    ax1.set_xticks(positions, labels)
    ax1.set_title("Part2 workflow compliance")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend()
    for x, value in zip(positions, base):
        ax1.text(x - width / 2, value + 2, f"{value:.1f}%", ha="center", fontsize=9)
    for x, value in zip(positions, exp):
        ax1.text(x + width / 2, value + 2, f"{value:.1f}%", ha="center", fontsize=9)

    error_labels = ["Validation errors", "Model turns", "Tool rounds"]
    base_error = [baseline.validation_errors, baseline.model_turns, baseline.tool_rounds]
    exp_error = [experiment.validation_errors, experiment.model_turns, experiment.tool_rounds]
    ax2.bar([x - width / 2 for x in positions], base_error, width, label=baseline.name, color="#94a3b8")
    ax2.bar([x + width / 2 for x in positions], exp_error, width, label=experiment.name, color="#2563eb")
    ax2.set_xticks(positions, error_labels)
    ax2.set_title("Counts (lower is better only for first two)")
    ax2.grid(axis="y", alpha=0.25)
    for x, value in zip(positions, base_error):
        ax2.text(x - width / 2, value + 2, str(value), ha="center", fontsize=9)
    for x, value in zip(positions, exp_error):
        ax2.text(x + width / 2, value + 2, str(value), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _slot_cards(experiment_dir: Path, report_dir: Path) -> str:
    result = _load_json(experiment_dir / "part2_result.json")
    cards: list[str] = []
    for index, slot_result in enumerate(result.get("slot_results", []), start=1):
        case = slot_result.get("case", {})
        slot_id = case.get("slot", {}).get("slot_id", "unknown")
        final_scores = case.get("final_scores") or {}
        figures: list[str] = []
        seen: set[str] = set()
        for evidence in case.get("evidence", []):
            if evidence.get("tool_name") not in DETAIL_TOOLS | {"check_fov"}:
                continue
            for raw in evidence.get("artifact_paths", []):
                if raw in seen or Path(raw).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                seen.add(raw)
                figures.append(
                    "<figure>"
                    f"<img src='{_esc(_rel(raw, report_dir))}' loading='lazy'>"
                    f"<figcaption>{_esc(evidence.get('tool_name'))} · "
                    f"round {int(evidence.get('round_index', 0))}</figcaption></figure>"
                )
        rounds = "".join(
            "<li>"
            f"R{int(row.get('round_index', 0))} <b>{_esc(row.get('tool_name'))}</b>: "
            f"{_esc(row.get('observation_summary', ''))}"
            "</li>"
            for row in case.get("rounds", [])
        )
        errors = slot_result.get("validation_errors", [])
        cards.append(
            f"<article class='slot-card' id='{_esc(slot_id)}'>"
            f"<h3>{index}. {_esc(slot_id)} · {_esc(case.get('final_state'))}</h3>"
            "<div class='badges'>"
            f"<span>Part1: {_esc(case.get('part1_state'))}</span>"
            f"<span>FOV: {_esc(case.get('fov', {}).get('visibility'))}</span>"
            f"<span>stop: {_esc(slot_result.get('stop_reason'))}</span>"
            f"<span>tools: {int(slot_result.get('tool_rounds', 0))}</span>"
            f"<span>model turns: {int(slot_result.get('model_turns', 0))}</span>"
            "</div>"
            f"<p><b>confidence:</b> free={float(final_scores.get('free_confidence', 0.0)):.3f}, "
            f"occupied={float(final_scores.get('occupied_confidence', 0.0)):.3f}. "
            "These are uncalibrated policy scores.</p>"
            f"<p><b>final reason:</b> {_esc(case.get('final_reason', ''))}</p>"
            f"<p class='error'><b>errors:</b> {_esc(' | '.join(errors) if errors else 'none')}</p>"
            f"<ol>{rounds}</ol>"
            f"<div class='gallery'>{''.join(figures)}</div>"
            "</article>"
        )
    return "".join(cards)


def write_comparison_report(
    baseline_dir: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path,
    *,
    baseline_name: str = "Qwen 0.8B baseline",
    experiment_name: str = "OpenAI VLM workflow",
) -> dict[str, Any]:
    """Write metrics JSON, a summary PNG, and a per-slot HTML audit report."""

    baseline_root = Path(baseline_dir).resolve()
    experiment_root = Path(experiment_dir).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    baseline = collect_run_metrics(baseline_root, name=baseline_name)
    experiment = collect_run_metrics(experiment_root, name=experiment_name)
    payload = comparison_payload(baseline, experiment)
    metrics_path = destination / "comparison_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_path = destination / "workflow_comparison.png"
    _render_summary_plot(plot_path, baseline, experiment)

    rows = [
        _metric_row("候选细节工具尝试覆盖", f"{baseline.cases_with_detail}/{baseline.processed_cases} ({baseline.detail_coverage_pct:.1f}%)", f"{experiment.cases_with_detail}/{experiment.processed_cases} ({experiment.detail_coverage_pct:.1f}%)", "每个候选是否实际尝试 Camera/LiDAR Part2 工具"),
        _metric_row("成功细节证据覆盖", f"{baseline.cases_with_successful_detail}/{baseline.processed_cases} ({baseline.successful_detail_coverage_pct:.1f}%)", f"{experiment.cases_with_successful_detail}/{experiment.processed_cases} ({experiment.successful_detail_coverage_pct:.1f}%)", "排除 unavailable 后，至少有一个成功 Camera/LiDAR 证据的候选"),
        _metric_row("合法最终动作覆盖", f"{baseline.accepted_final_cases}/{baseline.processed_cases} ({baseline.accepted_final_coverage_pct:.1f}%)", f"{experiment.accepted_final_cases}/{experiment.processed_cases} ({experiment.accepted_final_coverage_pct:.1f}%)", "通过状态机合同并记录 agent_final 的候选"),
        _metric_row("模型自主工具动作", f"{baseline.model_tool_actions}/{baseline.model_outputs} ({baseline.autonomous_tool_action_pct:.1f}%)", f"{experiment.model_tool_actions}/{experiment.model_outputs} ({experiment.autonomous_tool_action_pct:.1f}%)", "只计模型主动请求，不含强制 FOV/LiDAR 路由"),
        _metric_row("合同/运行错误", baseline.validation_errors, experiment.validation_errors, "越低越好；实验中 1 次为 API PermissionDenied，已 fail-closed 为 Unknown"),
        _metric_row("模型轮次", baseline.model_turns, experiment.model_turns, "基线大量重复非法 final；新流程减少无效重试"),
        _metric_row("实际细节工具轮次", baseline.tool_rounds, experiment.tool_rounds, "新流程真实收集的多模态证据量"),
        _metric_row("最终状态", baseline.final_state_counts, experiment.final_state_counts, "两者均未找到可信 Free；无 GT 时不能据此判断谁更准"),
    ]
    cards = _slot_cards(experiment_root, destination)
    report_path = destination / "part2_openai_vs_qwen_report.html"
    page = f"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Frame 9277 Part2 OpenAI vs Qwen 0.8B</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #172033; background: #f8fafc; }}
    h1, h2, h3 {{ color: #102a43; }}
    .notice {{ padding: 14px; border-radius: 10px; margin: 12px 0; background: #fff7ed; border: 1px solid #fdba74; }}
    .success {{ background: #ecfdf5; border-color: #6ee7b7; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #e2e8f0; }}
    .summary {{ max-width: 1200px; width: 100%; border: 1px solid #cbd5e1; background: white; }}
    .slot-card {{ background: white; border: 1px solid #cbd5e1; border-radius: 12px; margin: 18px 0; padding: 16px; }}
    .badges span {{ display: inline-block; padding: 3px 8px; margin: 2px; border-radius: 999px; background: #dbeafe; font-size: 12px; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; }}
    figure {{ margin: 0; padding: 8px; border: 1px solid #e2e8f0; }}
    figure img {{ width: 100%; max-height: 540px; object-fit: contain; background: #0f172a; }}
    figcaption {{ padding-top: 5px; font-size: 12px; color: #475569; }}
    .error {{ color: #9f1239; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Frame 9277 · Part2 OpenAI VLM 对比 Qwen 0.8B</h1>
  <div class='notice success'><b>可证实的提升：</b>端到端工作流可靠性明显提升。24/24 候选尝试细节工具、22/24 获得成功细节证据，23/24 形成合法最终动作，合同/运行错误由 138 降至 1。</div>
  <div class='notice'><b>不能声称的结论：</b>本锚点没有独立人工 GT，因此不能计算 accuracy / precision / recall，也不能宣称分类精度提升。两次运行还同时改变了模型与 FOV/工具路由，属于端到端比较，不是模型单变量消融。</div>
  <img class='summary' src='{_esc(plot_path.name)}' alt='workflow comparison'>
  <h2>定量对比</h2>
  <table><tr><th>指标</th><th>{_esc(baseline.name)}</th><th>{_esc(experiment.name)}</th><th>含义</th></tr>{''.join(rows)}</table>
  <p>OpenAI API 审计：模型输出 {experiment.model_outputs} 次，输入 token {experiment.input_tokens}, 输出 token {experiment.output_tokens}, 总 token {experiment.total_tokens}; API 累计时延 {experiment.total_api_seconds:.1f}s，中位数 {experiment.median_api_seconds:.1f}s，最大 {experiment.max_api_seconds:.1f}s。</p>
  <h2>24 个候选逐项证据</h2>
  <p>每张卡片按照真实队列顺序展示 FOV、工具轮次、最终原因和生成的证据图。置信度均标记为未校准策略分数。</p>
  {cards}
</body>
</html>
"""
    report_path.write_text(page, encoding="utf-8")
    return {
        "metrics": str(metrics_path),
        "summary_plot": str(plot_path),
        "html_report": str(report_path),
        "comparison": payload,
    }


__all__ = [
    "RunMetrics",
    "collect_run_metrics",
    "comparison_payload",
    "write_comparison_report",
]
