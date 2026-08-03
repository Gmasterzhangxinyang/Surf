"""Comprehensive Chinese research report for a Part2 VLM experiment."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .reporting import DETAIL_TOOLS, collect_run_metrics, comparison_payload


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
FOV_ORDER = ("visible", "partially_visible", "uncertain", "not_visible")
TOOL_ORDER = (
    "check_fov",
    "lidar_detail",
    "camera_context",
    "camera_sequence",
    "camera_crop",
)
FOV_ZH = {
    "visible": "可见",
    "partially_visible": "部分可见",
    "uncertain": "不确定",
    "not_visible": "不可见",
}
STATE_ZH = {"free": "空闲", "occupied": "占用", "unknown": "未知"}
TOOL_ZH = {
    "check_fov": "强制 FOV 检查",
    "lidar_detail": "LiDAR 局部细查",
    "camera_context": "Camera 全图上下文",
    "camera_sequence": "Camera 时序帧",
    "camera_crop": "Camera 局部裁剪",
    "agent_final": "Agent 证据融合",
    "part1_15frame": "Part1 十五帧证据",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _rel(path: str | Path, base: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), base.resolve())).as_posix()


def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _dict(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _slot_id(slot_result: Mapping[str, Any]) -> str:
    return str(_dict(_dict(slot_result.get("case")).get("slot")).get("slot_id"))


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfdff",
        }
    )
    return plt


def _save_figure(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _plot_overview(
    path: Path, baseline: Any, experiment: Any
) -> None:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    width = 0.36
    colors = ("#94a3b8", "#2563eb")

    labels = ["Successful detail", "Valid final", "Model tool action"]
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
    x = list(range(3))
    axes[0, 0].bar([v - width / 2 for v in x], base, width, color=colors[0], label="Qwen3.5-0.8B baseline")
    axes[0, 0].bar([v + width / 2 for v in x], exp, width, color=colors[1], label="OpenAI gpt-5.6-terra")
    axes[0, 0].set_ylim(0, 108)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("Percent")
    axes[0, 0].set_title("Workflow compliance")
    axes[0, 0].legend(fontsize=8)
    for offset, values in ((-width / 2, base), (width / 2, exp)):
        for pos, value in zip(x, values):
            axes[0, 0].text(pos + offset, value + 2, f"{value:.1f}%", ha="center", fontsize=9)

    count_labels = ["Validation errors", "Model turns", "Tool rounds"]
    base_counts = [baseline.validation_errors, baseline.model_turns, baseline.tool_rounds]
    exp_counts = [experiment.validation_errors, experiment.model_turns, experiment.tool_rounds]
    axes[0, 1].bar([v - width / 2 for v in x], base_counts, width, color=colors[0])
    axes[0, 1].bar([v + width / 2 for v in x], exp_counts, width, color=colors[1])
    axes[0, 1].set_xticks(x, count_labels)
    axes[0, 1].set_title("Execution counts")
    for offset, values in ((-width / 2, base_counts), (width / 2, exp_counts)):
        for pos, value in zip(x, values):
            axes[0, 1].text(pos + offset, value + 2, str(value), ha="center", fontsize=9)

    fov_labels = list(FOV_ORDER)
    base_fov = [baseline.fov_counts.get(label, 0) for label in fov_labels]
    exp_fov = [experiment.fov_counts.get(label, 0) for label in fov_labels]
    xf = list(range(len(fov_labels)))
    axes[1, 0].bar([v - width / 2 for v in xf], base_fov, width, color=colors[0])
    axes[1, 0].bar([v + width / 2 for v in xf], exp_fov, width, color=colors[1])
    axes[1, 0].set_xticks(xf, ["visible", "partial", "uncertain", "not visible"])
    axes[1, 0].set_ylabel("Cases")
    axes[1, 0].set_title("FOV routing distribution (workflow versions differ)")

    tool_labels = ["LiDAR", "Context", "Sequence", "Crop"]
    tool_keys = ["lidar_detail", "camera_context", "camera_sequence", "camera_crop"]
    base_tools = [baseline.tool_counts.get(key, 0) for key in tool_keys]
    exp_tools = [experiment.tool_counts.get(key, 0) for key in tool_keys]
    axes[1, 1].bar([v - width / 2 for v in xf], base_tools, width, color=colors[0])
    axes[1, 1].bar([v + width / 2 for v in xf], exp_tools, width, color=colors[1])
    axes[1, 1].set_xticks(xf, tool_labels)
    axes[1, 1].set_ylabel("Executed rounds")
    axes[1, 1].set_title("Part2 evidence tools")
    fig.suptitle("Frame 9277 Part2 end-to-end comparison", fontsize=16, y=1.01)
    _save_figure(fig, path)


def _plot_score_matrix(
    path: Path,
    experiment_results: list[Mapping[str, Any]],
    baseline_by_slot: Mapping[str, Mapping[str, Any]],
) -> None:
    plt = _setup_matplotlib()
    import numpy as np

    slot_ids = [_slot_id(row) for row in experiment_results]
    matrix: list[list[float]] = []
    for row in experiment_results:
        slot_id = _slot_id(row)
        case = _dict(row.get("case"))
        baseline_case = _dict(_dict(baseline_by_slot.get(slot_id)).get("case"))
        matrix.append(
            [
                _float(_dict(case.get("part1_scores")).get("free_confidence")),
                _float(_dict(case.get("part1_scores")).get("occupied_confidence")),
                _float(_dict(baseline_case.get("final_scores")).get("free_confidence")),
                _float(_dict(baseline_case.get("final_scores")).get("occupied_confidence")),
                _float(_dict(case.get("final_scores")).get("free_confidence")),
                _float(_dict(case.get("final_scores")).get("occupied_confidence")),
            ]
        )
    values = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(11, 12))
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(slot_ids)), slot_ids, fontsize=8)
    ax.set_xticks(
        range(6),
        ["P1 Free", "P1 Occ", "Qwen Free", "Qwen Occ", "OpenAI Free", "OpenAI Occ"],
        rotation=25,
        ha="right",
    )
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            color = "white" if value < 0.35 or value > 0.75 else "black"
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title("Per-slot uncalibrated confidence matrix")
    fig.colorbar(image, ax=ax, label="Policy score")
    _save_figure(fig, path)


def _plot_tool_matrix(path: Path, experiment_results: list[Mapping[str, Any]]) -> None:
    plt = _setup_matplotlib()
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm

    slot_ids = [_slot_id(row) for row in experiment_results]
    matrix = np.zeros((len(slot_ids), len(TOOL_ORDER)), dtype=int)
    for row_index, row in enumerate(experiment_results):
        evidence = _list(_dict(row.get("case")).get("evidence"))
        for col_index, tool in enumerate(TOOL_ORDER):
            matches = [item for item in evidence if _dict(item).get("tool_name") == tool]
            if not matches:
                matrix[row_index, col_index] = 0
            elif any(_dict(item).get("status") == "ok" for item in matches):
                matrix[row_index, col_index] = 2
            else:
                matrix[row_index, col_index] = 1
    cmap = ListedColormap(["#e2e8f0", "#f59e0b", "#10b981"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(range(len(slot_ids)), slot_ids, fontsize=8)
    ax.set_xticks(range(len(TOOL_ORDER)), ["FOV", "LiDAR", "Camera context", "Sequence", "Crop"], rotation=20, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, {0: "—", 1: "NA", 2: "OK"}[int(matrix[i, j])], ha="center", va="center", fontsize=8)
    ax.set_title("Per-slot evidence tool execution matrix")
    _save_figure(fig, path)


def _plot_api(path: Path, audit_rows: list[Mapping[str, Any]]) -> None:
    plt = _setup_matplotlib()
    elapsed = [_float(row.get("elapsed_seconds")) for row in audit_rows]
    total_tokens = [int(_dict(row.get("usage")).get("total_tokens", 0)) for row in audit_rows]
    input_tokens = [int(_dict(row.get("usage")).get("input_tokens", 0)) for row in audit_rows]
    output_tokens = [int(_dict(row.get("usage")).get("output_tokens", 0)) for row in audit_rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    indices = list(range(1, len(audit_rows) + 1))
    axes[0, 0].plot(indices, elapsed, marker="o", markersize=2.8, linewidth=1, color="#2563eb")
    axes[0, 0].axhline(120, color="#f59e0b", linestyle="--", linewidth=1, label="configured timeout")
    axes[0, 0].set_title("Successful API call latency")
    axes[0, 0].set_xlabel("Successful call index")
    axes[0, 0].set_ylabel("Seconds")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].hist(elapsed, bins=min(18, max(5, len(elapsed) // 3)), color="#60a5fa", edgecolor="white")
    axes[0, 1].set_title("Latency distribution")
    axes[0, 1].set_xlabel("Seconds")
    axes[0, 1].set_ylabel("Calls")
    axes[1, 0].bar(indices, input_tokens, color="#94a3b8", label="input")
    axes[1, 0].bar(indices, output_tokens, bottom=input_tokens, color="#2563eb", label="output")
    axes[1, 0].set_title("Token usage per successful call")
    axes[1, 0].set_xlabel("Successful call index")
    axes[1, 0].set_ylabel("Tokens")
    axes[1, 0].legend(fontsize=8)
    buckets = Counter()
    for row in audit_rows:
        count = len(_list(row.get("images")))
        buckets[count] += 1
    keys = sorted(buckets)
    axes[1, 1].bar([str(key) for key in keys], [buckets[key] for key in keys], color="#10b981")
    axes[1, 1].set_title("Images supplied per model call")
    axes[1, 1].set_xlabel("Image count")
    axes[1, 1].set_ylabel("Calls")
    fig.suptitle(f"OpenAI API audit: {sum(total_tokens):,} total tokens across {len(audit_rows)} successful calls", fontsize=15, y=1.01)
    _save_figure(fig, path)


def _plot_sensor_timeline(path: Path, scene: Mapping[str, Any]) -> None:
    plt = _setup_matplotlib()
    frames = _list(scene.get("frames"))
    ids = [int(_dict(frame).get("frame_id", 0)) for frame in frames]
    dt_ms = [1000.0 * _float(_dict(frame).get("camera_lidar_dt_sec")) for frame in frames]
    yaw_deg = [math.degrees(_float(_dict(frame).get("map_yaw_rad"))) for frame in frames]
    x = list(range(len(frames)))
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    axes[0].plot(x, dt_ms, marker="o", color="#2563eb")
    axes[0].set_ylabel("Camera-LiDAR dt (ms)")
    axes[0].set_title("15-frame sensor synchronization")
    axes[1].plot(x, yaw_deg, marker="o", color="#10b981")
    axes[1].set_ylabel("Map yaw (deg)")
    axes[1].set_xlabel("LiDAR frame")
    axes[1].set_xticks(x, [str(frame_id) for frame_id in ids], rotation=45)
    _save_figure(fig, path)


def _write_workflow_svg(path: Path) -> None:
    boxes = [
        (40, 35, 230, 72, "候选队列", "Free 优先，Unknown 其次"),
        (320, 35, 230, 72, "强制 FOV", "可靠区 ±40° / 名义包络 ±50.5°"),
        (600, 35, 230, 72, "自动路由", "不可见→LiDAR；其余→Agent"),
        (880, 35, 230, 72, "单车位 Agent", "最多 3 个证据工具轮次"),
        (880, 170, 230, 72, "工具选择", "Camera context/sequence/crop 或 LiDAR"),
        (600, 170, 230, 72, "证据融合", "Free / Occupied / Unknown"),
        (320, 170, 230, 72, "合同验证", "终态 ≥0.90 且引用成功证据"),
        (40, 170, 230, 72, "队列终止", "可信 Free 早停，否则继续"),
    ]
    arrows = [
        (270, 71, 320, 71), (550, 71, 600, 71), (830, 71, 880, 71),
        (995, 107, 995, 170), (880, 206, 830, 206), (600, 206, 550, 206), (320, 206, 270, 206),
    ]
    box_svg = "".join(
        f"<g><rect x='{x}' y='{y}' width='{w}' height='{h}' rx='12' fill='#eff6ff' stroke='#2563eb' stroke-width='2'/>"
        f"<text x='{x+w/2}' y='{y+29}' text-anchor='middle' font-size='17' font-weight='700'>{_esc(title)}</text>"
        f"<text x='{x+w/2}' y='{y+52}' text-anchor='middle' font-size='12' fill='#475569'>{_esc(subtitle)}</text></g>"
        for x, y, w, h, title, subtitle in boxes
    )
    arrow_svg = "".join(
        f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#334155' stroke-width='2.2' marker-end='url(#arrow)'/>"
        for x1, y1, x2, y2 in arrows
    )
    path.write_text(
        f"""<svg xmlns='http://www.w3.org/2000/svg' width='1150' height='285' viewBox='0 0 1150 285'>
<defs><marker id='arrow' markerWidth='10' markerHeight='10' refX='8' refY='3' orient='auto'><path d='M0,0 L0,6 L9,3 z' fill='#334155'/></marker></defs>
<rect width='1150' height='285' fill='white'/>{box_svg}{arrow_svg}
<text x='575' y='270' text-anchor='middle' font-size='12' fill='#64748b'>所有置信度均为未校准策略分数；没有独立 GT 时不解释为统计概率。</text>
</svg>""",
        encoding="utf-8",
    )


def _evidence_images(
    case: Mapping[str, Any], report_dir: Path, *, include_part1: bool = False
) -> tuple[str, int, set[str]]:
    figures: list[str] = []
    references = 0
    unique: set[str] = set()
    for evidence_raw in _list(case.get("evidence")):
        evidence = _dict(evidence_raw)
        tool = str(evidence.get("tool_name"))
        if tool == "part1_15frame" and not include_part1:
            continue
        if tool not in DETAIL_TOOLS | {"check_fov", "part1_15frame"}:
            continue
        for raw in _list(evidence.get("artifact_paths")):
            path = Path(str(raw))
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            references += 1
            unique.add(str(path.resolve()))
            figures.append(
                "<figure>"
                f"<a href='{_esc(_rel(path, report_dir))}' target='_blank'>"
                f"<img src='{_esc(_rel(path, report_dir))}' loading='lazy' alt='{_esc(tool)}'></a>"
                f"<figcaption>{_esc(TOOL_ZH.get(tool, tool))} · round {int(evidence.get('round_index', 0))} · {_esc(path.name)}</figcaption>"
                "</figure>"
            )
    return "".join(figures), references, unique


def _round_table(case: Mapping[str, Any]) -> str:
    evidence_by_id = {
        str(_dict(item).get("evidence_id")): _dict(item)
        for item in _list(case.get("evidence"))
    }
    rows: list[str] = []
    for round_raw in _list(case.get("rounds")):
        row = _dict(round_raw)
        evidences = [evidence_by_id.get(str(value), {}) for value in _list(row.get("evidence_ids"))]
        evidence_status = ", ".join(
            f"{_dict(item).get('tool_name')}={_dict(item).get('status')}" for item in evidences
        )
        scores = _dict(row.get("scores_after"))
        rows.append(
            "<tr>"
            f"<td>{int(row.get('round_index', 0))}</td>"
            f"<td>{_esc(TOOL_ZH.get(str(row.get('tool_name')), row.get('tool_name')))}</td>"
            f"<td>{_esc(row.get('reasoning_summary', ''))}</td>"
            f"<td>{_esc(row.get('observation_summary', ''))}</td>"
            f"<td>{_esc(evidence_status)}</td>"
            f"<td>F={_float(scores.get('free_confidence')):.3f}<br>O={_float(scores.get('occupied_confidence')):.3f}<br>U={_float(scores.get('unknown_confidence')):.3f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _slot_cards(
    experiment_results: list[Mapping[str, Any]],
    baseline_by_slot: Mapping[str, Mapping[str, Any]],
    report_dir: Path,
) -> tuple[str, dict[str, Any]]:
    cards: list[str] = []
    visual_refs = 0
    visual_unique: set[str] = set()
    for index, slot_result in enumerate(experiment_results, start=1):
        case = _dict(slot_result.get("case"))
        slot = _dict(case.get("slot"))
        slot_id = str(slot.get("slot_id"))
        baseline_case = _dict(_dict(baseline_by_slot.get(slot_id)).get("case"))
        p1 = _dict(case.get("part1_scores"))
        final = _dict(case.get("final_scores"))
        baseline_final = _dict(baseline_case.get("final_scores"))
        fov = _dict(case.get("fov"))
        fov_details = _dict(fov.get("details"))
        images, refs, unique = _evidence_images(case, report_dir)
        visual_refs += refs
        visual_unique.update(unique)
        errors = [str(value) for value in _list(slot_result.get("validation_errors"))]
        evidence_summary = Counter(
            (str(_dict(item).get("tool_name")), str(_dict(item).get("status")))
            for item in _list(case.get("evidence"))
            if _dict(item).get("tool_name") not in {"part1_15frame", "agent_final", "check_fov"}
        )
        summary_text = ", ".join(
            f"{TOOL_ZH.get(tool, tool)}={status}×{count}"
            for (tool, status), count in sorted(evidence_summary.items())
        ) or "无细节证据"
        cards.append(
            f"<article class='slot-card' data-fov='{_esc(fov.get('visibility'))}' data-state='{_esc(case.get('final_state'))}' data-slot='{_esc(slot_id)}'>"
            f"<header><h3>{index:02d}. {_esc(slot_id)} · {STATE_ZH.get(str(case.get('final_state')), case.get('final_state'))}</h3>"
            f"<a class='toplink' href='#top'>返回顶部</a></header>"
            "<div class='badges'>"
            f"<span>Part1={STATE_ZH.get(str(case.get('part1_state')), case.get('part1_state'))}</span>"
            f"<span>FOV={FOV_ZH.get(str(fov.get('visibility')), fov.get('visibility'))}</span>"
            f"<span>距离={_float(slot.get('distance_to_anchor_m')):.2f}m</span>"
            f"<span>工具轮次={int(slot_result.get('tool_rounds', 0))}</span>"
            f"<span>模型轮次={int(slot_result.get('model_turns', 0))}</span>"
            f"<span>停止={_esc(slot_result.get('stop_reason'))}</span>"
            "</div>"
            "<div class='three-col'>"
            f"<section><h4>Part1 初始状态</h4><p>Free={_float(p1.get('free_confidence')):.3f}<br>Occupied={_float(p1.get('occupied_confidence')):.3f}<br>Unknown={_float(p1.get('unknown_confidence')):.3f}</p></section>"
            f"<section><h4>Qwen 0.8B 基线</h4><p>状态={STATE_ZH.get(str(baseline_case.get('final_state')), baseline_case.get('final_state'))}<br>Free={_float(baseline_final.get('free_confidence')):.3f}<br>Occupied={_float(baseline_final.get('occupied_confidence')):.3f}</p></section>"
            f"<section><h4>OpenAI 最终状态</h4><p>状态={STATE_ZH.get(str(case.get('final_state')), case.get('final_state'))}<br>Free={_float(final.get('free_confidence')):.3f}<br>Occupied={_float(final.get('occupied_confidence')):.3f}</p></section>"
            "</div>"
            "<h4>FOV 路由审计</h4>"
            f"<p>Camera frame={_esc(fov.get('camera_frame_id'))}；置信度={_float(fov.get('confidence')):.2f}；目标方位={_float(fov_details.get('target_bearing_deg')):.2f}°；可靠半视场={_float(fov_details.get('half_fov_deg')):.1f}°；yaw 不确定性={_float(fov_details.get('yaw_uncertainty_deg')):.1f}°。原因：{_esc(fov.get('reason'))}。</p>"
            f"<p><b>细节证据：</b>{_esc(summary_text)}</p>"
            "<h4>逐轮 ReAct 记录</h4>"
            "<div class='table-wrap'><table><tr><th>轮次</th><th>工具</th><th>Reason</th><th>Observation</th><th>证据状态</th><th>轮后分数</th></tr>"
            f"{_round_table(case)}</table></div>"
            f"<p><b>最终说明：</b>{_esc(case.get('final_reason', ''))}</p>"
            f"<p class='errors'><b>验证/运行错误：</b>{_esc(' | '.join(errors) if errors else '无')}</p>"
            f"<details><summary>查看 Unknown 原因与原始 FOV 细节</summary><pre>{_esc(json.dumps({'unknown_reasons': case.get('unknown_reasons'), 'unresolved_reasons': case.get('unresolved_reasons'), 'fov': fov}, ensure_ascii=False, indent=2))}</pre></details>"
            f"<div class='gallery'>{images}</div>"
            "</article>"
        )
    return "".join(cards), {
        "part2_visual_references": visual_refs,
        "part2_unique_visuals": len(visual_unique),
        "part2_unique_visual_paths": sorted(visual_unique),
    }


def _part1_visual_appendix(
    experiment_results: list[Mapping[str, Any]], report_dir: Path
) -> tuple[str, list[str]]:
    paths: set[str] = set()
    for result in experiment_results:
        case = _dict(result.get("case"))
        for evidence_raw in _list(case.get("evidence")):
            evidence = _dict(evidence_raw)
            if evidence.get("tool_name") != "part1_15frame":
                continue
            for raw in _list(evidence.get("artifact_paths")):
                path = Path(str(raw))
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    paths.add(str(path.resolve()))
    figures = "".join(
        "<figure>"
        f"<a href='{_esc(_rel(path, report_dir))}' target='_blank'><img src='{_esc(_rel(path, report_dir))}' loading='lazy'></a>"
        f"<figcaption>Part1/原始 Camera · {_esc(Path(path).name)}</figcaption></figure>"
        for path in sorted(paths)
    )
    return figures, sorted(paths)


def _render_markdown(
    path: Path,
    baseline: Any,
    experiment: Any,
    manifest: Mapping[str, Any],
    report_path: Path,
) -> None:
    content = f"""# Frame 9277 Part2 OpenAI 多模态 Agent 完整实验报告

生成时间：{manifest['generated_at_utc']}

## 结论

- 工作流可靠性有明确提升：成功细节证据覆盖从 {baseline.successful_detail_coverage_pct:.1f}% 提升到 {experiment.successful_detail_coverage_pct:.1f}%。
- 合法最终动作覆盖从 {baseline.accepted_final_coverage_pct:.1f}% 提升到 {experiment.accepted_final_coverage_pct:.1f}%。
- 验证/运行错误从 {baseline.validation_errors} 次降到 {experiment.validation_errors} 次。
- OpenAI 运行最终为 {experiment.final_state_counts}，没有找到可信 Free。
- 本锚点没有独立人工 GT，不能声称分类准确率提升。

## 可视化完整性

- Part2 图像引用：{manifest['visual_inventory']['part2_visual_references']}
- Part2 唯一图像：{manifest['visual_inventory']['part2_unique_visuals']}
- Part1 唯一源图：{manifest['visual_inventory']['part1_unique_visuals']}
- 缺失图像：{manifest['visual_inventory']['missing_visuals']}

完整逐车位图文内容请打开：`{report_path.name}`。
"""
    path.write_text(content, encoding="utf-8")


def build_comprehensive_chinese_report(
    *,
    part1_path: str | Path,
    baseline_dir: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path,
    baseline_name: str = "Qwen3.5-0.8B 基线",
    experiment_name: str = "OpenAI gpt-5.6-terra Agent",
) -> dict[str, Any]:
    """Generate a complete local-only Chinese report and validate every image."""

    part1_file = Path(part1_path).resolve()
    baseline_root = Path(baseline_dir).resolve()
    experiment_root = Path(experiment_dir).resolve()
    destination = Path(output_dir).resolve()
    assets = destination / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    part1 = _load(part1_file)
    scene = _dict(part1.get("scene"))
    baseline_result = _load(baseline_root / "part2_result.json")
    experiment_result = _load(experiment_root / "part2_result.json")
    baseline_results = [
        _dict(item) for item in _list(baseline_result.get("slot_results"))
    ]
    experiment_results = [
        _dict(item) for item in _list(experiment_result.get("slot_results"))
    ]
    baseline_by_slot = {_slot_id(item): item for item in baseline_results}
    baseline = collect_run_metrics(baseline_root, name=baseline_name)
    experiment = collect_run_metrics(experiment_root, name=experiment_name)
    comparison = comparison_payload(baseline, experiment)

    audit_rows = [
        _dict(_load(path))
        for path in sorted((experiment_root / "openai_audit").glob("**/turn_*.json"))
    ]
    overview_path = assets / "01_overview_dashboard.png"
    scores_path = assets / "02_per_slot_confidence_matrix.png"
    tools_path = assets / "03_per_slot_tool_matrix.png"
    api_path = assets / "04_openai_api_audit.png"
    sensor_path = assets / "05_sensor_timeline.png"
    workflow_path = assets / "00_workflow.svg"
    _write_workflow_svg(workflow_path)
    _plot_overview(overview_path, baseline, experiment)
    _plot_score_matrix(scores_path, experiment_results, baseline_by_slot)
    _plot_tool_matrix(tools_path, experiment_results)
    _plot_api(api_path, audit_rows)
    _plot_sensor_timeline(sensor_path, scene)

    cards, visual_inventory = _slot_cards(
        experiment_results, baseline_by_slot, destination
    )
    part1_figures, part1_paths = _part1_visual_appendix(
        experiment_results, destination
    )
    all_visual_paths = [
        *visual_inventory["part2_unique_visual_paths"],
        *part1_paths,
        str(overview_path),
        str(scores_path),
        str(tools_path),
        str(api_path),
        str(sensor_path),
        str(workflow_path),
    ]
    missing = sorted(path for path in all_visual_paths if not Path(path).is_file())
    final_counter = Counter(
        str(_dict(item.get("case")).get("final_state")) for item in experiment_results
    )
    stop_counter = Counter(str(item.get("stop_reason")) for item in experiment_results)
    successful_calls = len(audit_rows)
    usage = {
        key: sum(int(_dict(row.get("usage")).get(key, 0)) for row in audit_rows)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    reasoning_tokens = sum(
        int(_dict(_dict(row.get("usage")).get("output_tokens_details")).get("reasoning_tokens", 0))
        for row in audit_rows
    )
    cached_tokens = sum(
        int(_dict(_dict(row.get("usage")).get("input_tokens_details")).get("cached_tokens", 0))
        for row in audit_rows
    )
    stability_path = destination / "stability_verification.json"
    preflight_path = destination / "preflight_verification.json"
    stability = _load(stability_path) if stability_path.is_file() else {"ok": None}
    preflight = _load(preflight_path) if preflight_path.is_file() else {"ok": None}
    manifest = {
        "schema_version": "parking-slot-agent-v2-comprehensive-zh-report/1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "part1_path": str(part1_file),
        "baseline_dir": str(baseline_root),
        "experiment_dir": str(experiment_root),
        "comparison": comparison,
        "visual_inventory": {
            **{key: value for key, value in visual_inventory.items() if key != "part2_unique_visual_paths"},
            "part1_unique_visuals": len(part1_paths),
            "report_generated_visuals": 6,
            "missing_visuals": len(missing),
            "missing_visual_paths": missing,
        },
        "openai_audit": {
            "successful_calls": successful_calls,
            **usage,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
        },
        "preflight_verification": preflight,
        "stability_verification": stability,
    }
    manifest_path = destination / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_path = destination / "frame_9277_part2_openai完整中文报告.html"
    queue_rows = "".join(
        "<tr>"
        f"<td>{index}</td><td><a href='#{_esc(_slot_id(item))}'>{_esc(_slot_id(item))}</a></td>"
        f"<td>{STATE_ZH.get(str(_dict(item.get('case')).get('part1_state')), _dict(item.get('case')).get('part1_state'))}</td>"
        f"<td>{FOV_ZH.get(str(_dict(_dict(item.get('case')).get('fov')).get('visibility')), _dict(_dict(item.get('case')).get('fov')).get('visibility'))}</td>"
        f"<td>{STATE_ZH.get(str(_dict(item.get('case')).get('final_state')), _dict(item.get('case')).get('final_state'))}</td>"
        f"<td>{int(item.get('model_turns', 0))}</td><td>{int(item.get('tool_rounds', 0))}</td><td>{_esc(item.get('stop_reason'))}</td>"
        "</tr>"
        for index, item in enumerate(experiment_results, start=1)
    )
    report_html = f"""<!doctype html>
<html lang='zh-CN'>
<head>
<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Frame 9277 Part2 OpenAI 多模态 Agent 完整中文报告</title>
<style>
:root {{--ink:#172033;--muted:#526273;--blue:#2563eb;--green:#047857;--orange:#c2410c;--line:#cbd5e1;--paper:#fff;--bg:#f1f5f9}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Microsoft YaHei","PingFang SC",sans-serif;line-height:1.65}}
main{{max-width:1500px;margin:auto;padding:26px}} h1,h2,h3,h4{{line-height:1.25;color:#102a43}} h1{{font-size:30px}} h2{{margin-top:42px;padding-bottom:8px;border-bottom:3px solid #bfdbfe}}
.toc{{background:#0f2942;color:white;padding:15px 22px;position:sticky;top:0;z-index:4;box-shadow:0 2px 8px #0003}} .toc a{{color:#dbeafe;margin-right:18px;text-decoration:none;font-size:14px}}
.hero,.panel,.slot-card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:20px;margin:18px 0;box-shadow:0 3px 12px #0f172a0d}}
.hero{{border-left:7px solid var(--blue)}} .claim{{padding:14px;border-radius:10px;margin:10px 0}} .proved{{background:#ecfdf5;border:1px solid #6ee7b7}} .limited{{background:#fff7ed;border:1px solid #fdba74}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}} .kpi{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:14px}} .kpi b{{display:block;font-size:27px;color:#1d4ed8}} .kpi small{{color:var(--muted)}}
.figure-wide{{width:100%;max-height:960px;object-fit:contain;background:white;border:1px solid var(--line)}}
.two-col{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}} .three-col{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}} .three-col section{{border:1px solid #dbe3ed;border-radius:9px;padding:10px;background:#f8fafc}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:white}} th,td{{border:1px solid var(--line);padding:7px;vertical-align:top}} th{{background:#e8eef6;position:sticky;top:50px}} .table-wrap{{overflow:auto;max-height:630px}}
.slot-card header{{display:flex;justify-content:space-between;align-items:center}} .toplink{{font-size:12px}} .badges span{{display:inline-block;background:#dbeafe;padding:3px 9px;border-radius:999px;margin:2px;font-size:12px}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px;margin-top:14px}} figure{{margin:0;border:1px solid var(--line);padding:8px;border-radius:8px;background:#f8fafc}} figure img{{width:100%;height:360px;object-fit:contain;background:#0f172a}} figcaption{{font-size:12px;color:var(--muted);padding-top:5px;overflow-wrap:anywhere}}
.errors{{color:#9f1239}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;max-height:430px;overflow:auto}}
.controls{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}} input,select{{padding:8px;border:1px solid #94a3b8;border-radius:7px;background:white}} code{{overflow-wrap:anywhere}} .footnote{{color:var(--muted);font-size:13px}}
@media print{{.toc,.controls,.toplink{{display:none}} body{{background:white}} main{{max-width:none;padding:0}} .slot-card{{break-inside:avoid;box-shadow:none}} figure img{{height:auto;max-height:380px}} details{{display:block}}}}
</style>
</head>
<body id='top'>
<nav class='toc'><a href='#summary'>摘要</a><a href='#scope'>实验设计</a><a href='#workflow'>工作流</a><a href='#quant'>定量结果</a><a href='#api'>API审计</a><a href='#stability'>稳定性</a><a href='#queue'>队列总表</a><a href='#cases'>逐车位证据</a><a href='#appendix'>附录</a></nav>
<main>
<section class='hero' id='summary'><h1>Frame 9277 Part2 OpenAI 多模态 Agent 完整实验报告</h1>
<p>报告对象是固定锚点 LiDAR frame 9277。Part1 使用 t0 及之前 15 帧 LiDAR 建立 30m 候选集合；Part2 对 24 个候选执行强制 FOV、Camera/LiDAR 细查与严格终态合同。</p>
<div class='claim proved'><b>已经证实：</b>端到端工作流可靠性明显提升。细节工具尝试覆盖从 4.2% 提升到 100%，成功细节证据覆盖从 4.2% 提升到 91.7%，合法最终动作覆盖从 4.2% 提升到 95.8%，验证/运行错误从 138 降到 1。</div>
<div class='claim limited'><b>尚不能证实：</b>没有独立人工 GT，无法计算 accuracy、precision、recall；同时模型、FOV 参数与工具集合均有变化，因此不是模型单变量 A/B。报告不声称识别准确率提升。</div>
<div class='kpis'><div class='kpi'><b>24/24</b><small>候选完成处理</small></div><div class='kpi'><b>22/24</b><small>至少一个成功细节证据</small></div><div class='kpi'><b>23/24</b><small>合法 agent_final</small></div><div class='kpi'><b>1 Occupied</b><small>slot_1012，O=0.92</small></div><div class='kpi'><b>0 Free</b><small>没有可信空闲车位</small></div><div class='kpi'><b>{usage['total_tokens']:,}</b><small>50 次成功 API 调用总 token</small></div></div>
</section>

<section id='scope'><h2>1. 实验范围与数据来源</h2><div class='panel'>
<table><tr><th>项目</th><th>值</th><th>解释</th></tr>
<tr><td>snapshot</td><td>{_esc(scene.get('snapshot_id'))}</td><td>固定单锚点实验，不是 30 锚点规模评测</td></tr>
<tr><td>anchor</td><td>LiDAR {int(scene.get('anchor_frame_id',0))}；timestamp {_float(scene.get('anchor_timestamp')):.6f}</td><td>所有证据禁止使用 t0 之后的帧</td></tr>
<tr><td>坐标系</td><td>{_esc(scene.get('coordinate_frame'))}</td><td>pose-corrected map</td></tr>
<tr><td>空间范围</td><td>{_float(scene.get('radius_m')):.1f} m；地图车位 {len(_list(scene.get('slots')))}</td><td>Part1 原始 occupancy map 与车位库</td></tr>
<tr><td>传感器窗口</td><td>{len(_list(scene.get('frames')))} 帧，LiDAR {int(_dict(_list(scene.get('frames'))[0]).get('frame_id',0))}–{int(_dict(_list(scene.get('frames'))[-1]).get('frame_id',0))}</td><td>Camera 与 LiDAR 时间匹配均记录在 SceneSnapshot</td></tr>
<tr><td>候选队列</td><td>24：Part1 Free=2，Unknown=22</td><td>Free 优先，Unknown 其次</td></tr>
<tr><td>OpenAI 配置</td><td>gpt-5.6-terra；reasoning=medium；image detail=original</td><td>Responses API，strict Structured Output，store=false</td></tr>
</table></div>
<img class='figure-wide' src='{_esc(_rel(sensor_path,destination))}' alt='sensor timeline'></section>

<section id='workflow'><h2>2. Part2 工作流与强制合同</h2><div class='panel'><img class='figure-wide' src='{_esc(_rel(workflow_path,destination))}' alt='workflow'>
<ol><li>每个 SlotCase 首先执行强制 FOV，并将结果写回原卡片。</li><li>FOV=not_visible 时禁止 Camera，自动获取一次 LiDAR 局部细节；其他状态由 Agent 自主选择 Camera/LiDAR。</li><li>最多 3 个证据工具轮次；Camera crop 必须在成功 Camera context 后执行。</li><li>Free/Occupied 必须达到 0.90，引用成功 Part2 证据，且不能存在相反终态置信度。</li><li>部分可见或 FOV 不确定时，Camera-only 终态还要求定位与占用置信度均达到 0.90。</li><li>不能消除不确定性时必须输出 Unknown；可信 Free 才能全局早停。</li></ol></div></section>

<section id='quant'><h2>3. 定量结果与可视化对比</h2>
<img class='figure-wide' src='{_esc(_rel(overview_path,destination))}' alt='overview'>
<div class='two-col'><div class='panel'><h3>核心指标</h3><table><tr><th>指标</th><th>Qwen 0.8B</th><th>OpenAI</th><th>变化</th></tr>
<tr><td>成功细节证据覆盖</td><td>1/24 (4.2%)</td><td>22/24 (91.7%)</td><td>+87.5 pp</td></tr><tr><td>合法最终动作覆盖</td><td>1/24 (4.2%)</td><td>23/24 (95.8%)</td><td>+91.7 pp</td></tr><tr><td>模型自主 ToolAction</td><td>0/139</td><td>27/50 (54.0%)</td><td>+54.0 pp</td></tr><tr><td>验证/运行错误</td><td>138</td><td>1</td><td>-137</td></tr><tr><td>模型轮次</td><td>139</td><td>51</td><td>-88</td></tr><tr><td>实际工具轮次</td><td>1</td><td>42</td><td>+41</td></tr></table></div>
<div class='panel'><h3>最终状态</h3><p>OpenAI：{_esc(dict(final_counter))}；总体停止原因：{_esc(dict(stop_counter))}。</p><p>唯一 Occupied 是 <b>slot_1012</b>，FOV 不可见，依靠成功的 15 帧目标局部 LiDAR 得到 occupied=0.92。Qwen 基线唯一 Occupied 是 slot_1258，两者不同；没有 GT 时不能判断哪一个正确。</p><p>OpenAI 唯一错误是 slot_1250 的 PermissionDeniedError，系统 fail-closed 为 Unknown，没有将异常转成伪高置信终态。</p></div></div>
<h3>24 个候选的未校准分数矩阵</h3><img class='figure-wide' src='{_esc(_rel(scores_path,destination))}' alt='score matrix'>
<h3>逐候选工具执行矩阵</h3><img class='figure-wide' src='{_esc(_rel(tools_path,destination))}' alt='tool matrix'>
<p class='footnote'>FOV 分布不能作为纯模型能力对比：旧 Qwen 结果使用旧 90° 路由，新流程使用可靠半视场 40°、名义物理包络约 50.5°，并增加 camera_sequence。</p></section>

<section id='api'><h2>4. OpenAI API 调用审计</h2><img class='figure-wide' src='{_esc(_rel(api_path,destination))}' alt='api audit'>
<div class='panel'><table><tr><th>指标</th><th>结果</th></tr><tr><td>成功调用</td><td>{successful_calls}</td></tr><tr><td>输入 / 输出 / 总 token</td><td>{usage['input_tokens']:,} / {usage['output_tokens']:,} / {usage['total_tokens']:,}</td></tr><tr><td>reasoning / cached token</td><td>{reasoning_tokens:,} / {cached_tokens:,}</td></tr><tr><td>成功调用累计时延</td><td>{experiment.total_api_seconds:.3f}s（不含失败调用和工具渲染）</td></tr><tr><td>中位数 / 最大时延</td><td>{experiment.median_api_seconds:.3f}s / {experiment.max_api_seconds:.3f}s</td></tr><tr><td>隐私</td><td>store=false；审计仅记录 key 来源与 api_key_recorded=false，不记录 Key 内容</td></tr></table></div></section>

<section id='stability'><h2>5. 稳定性与可重复性验收</h2><div class='panel'><table><tr><th>检查</th><th>结果</th><th>证据</th></tr>
<tr><td>运行前 fail-closed 预检</td><td>{'通过' if preflight.get('ok') is True else '未记录'}</td><td>固定 SDK、Key 600 权限、Part1 合同、15 帧媒体、输出隔离和磁盘空间</td></tr>
<tr><td>跨目录决策一致性</td><td>{'通过' if stability.get('cross_directory_decision_projection_identical') is True else '未通过/未记录'}</td><td>24 个车位的最终状态、F/O 分数、停止原因、模型轮次与工具轮次一致</td></tr>
<tr><td>同目录字节确定性</td><td>{'通过' if stability.get('same_directory_byte_deterministic') is True else '未通过/未记录'}</td><td>连续两次完整重放 SHA256：<code>{_esc(stability.get('first_sha256',''))}</code></td></tr>
<tr><td>队列与早停结果</td><td>{'通过' if stability.get('queue_identical') is True and stability.get('selected_slot_identical') is True else '未通过/未记录'}</td><td>队列顺序和 selected_slot_id 与线上结果一致</td></tr>
<tr><td>并发写保护</td><td>已实现</td><td>运行脚本使用原子 lock 目录，同一输出目录默认拒绝覆盖</td></tr>
<tr><td>API 重试边界</td><td>已固定</td><td>timeout=120s，max_retries=2；非法或异常结果 fail-closed 为 Unknown</td></tr>
</table><p class='footnote'>slot_1250 在线 PermissionDeniedError 在离线重放中表现为动作耗尽 RuntimeError；两者均按相同决策投影 fail-closed 为 Unknown，错误类型本身不宣称可重放。</p></div></section>

<section id='queue'><h2>6. 候选队列总表</h2><div class='panel table-wrap'><table><tr><th>#</th><th>slot</th><th>Part1</th><th>FOV</th><th>OpenAI 最终</th><th>模型轮次</th><th>工具轮次</th><th>停止原因</th></tr>{queue_rows}</table></div></section>

<section id='cases'><h2>7. 24 个候选逐项完整证据</h2><p>以下卡片按真实队列顺序展示。所有 FOV、LiDAR、Camera context、Camera crop、Camera sequence 结果均保留；点击图片可打开原分辨率文件。</p>
<div class='controls'><input id='search' placeholder='输入 slot，例如 1012'><select id='fovFilter'><option value=''>全部 FOV</option>{''.join(f"<option value='{key}'>{value}</option>" for key,value in FOV_ZH.items())}</select><select id='stateFilter'><option value=''>全部状态</option><option value='unknown'>未知</option><option value='occupied'>占用</option><option value='free'>空闲</option></select><span id='visibleCount'></span></div>
{cards}</section>

<section id='appendix'><h2>8. 附录：Part1 源图、完整性与复现</h2><div class='panel'><h3>Part1 唯一 Camera 源图</h3><div class='gallery'>{part1_figures}</div></div>
<div class='panel'><h3>可视化完整性</h3><ul><li>Part2 可视化引用：{visual_inventory['part2_visual_references']}；唯一图像：{visual_inventory['part2_unique_visuals']}。</li><li>Part1 唯一源图：{len(part1_paths)}。</li><li>报告生成图表/SVG：6。</li><li>缺失图像：{len(missing)}。</li></ul></div>
<div class='panel'><h3>关键文件</h3><ul><li><code>{_esc(experiment_root/'part2_result.json')}</code></li><li><code>{_esc(experiment_root/'openai_audit')}</code></li><li><code>{_esc(experiment_root/'replay_actions.json')}</code></li><li><code>{_esc(baseline_root/'part2_result.json')}</code></li><li><code>{_esc(manifest_path)}</code></li></ul><h3>复现命令</h3><pre>scripts/run_parking_slot_agent_v2_openai.sh
python3 scripts/build_parking_slot_agent_v2_report_zh.py</pre></div>
<div class='claim limited'><b>最终边界：</b>本报告证明的是端到端 Agent 工作流、工具使用和合同遵守的提升。要证明停车位识别效果提升，下一步必须对这 24 个候选建立独立人工 GT，并在相同 FOV、相同工具、相同终态合同下做受控模型消融。</div></section>
</main>
<script>
const cards=[...document.querySelectorAll('.slot-card')]; const search=document.getElementById('search'); const ff=document.getElementById('fovFilter'); const sf=document.getElementById('stateFilter'); const count=document.getElementById('visibleCount');
function filterCards(){{let n=0; const q=search.value.trim().toLowerCase(); cards.forEach(c=>{{const ok=(!q||c.dataset.slot.toLowerCase().includes(q))&&(!ff.value||c.dataset.fov===ff.value)&&(!sf.value||c.dataset.state===sf.value); c.style.display=ok?'block':'none'; if(ok)n++;}}); count.textContent=`显示 ${{n}} / ${{cards.length}}`;}} [search,ff,sf].forEach(x=>x.addEventListener('input',filterCards)); filterCards();
</script></body></html>"""
    report_path.write_text(report_html, encoding="utf-8")
    markdown_path = destination / "README_完整中文报告.md"
    _render_markdown(markdown_path, baseline, experiment, manifest, report_path)
    return {
        "html_report": str(report_path),
        "markdown_summary": str(markdown_path),
        "manifest": str(manifest_path),
        "visual_inventory": manifest["visual_inventory"],
        "comparison_claims": comparison["claims"],
    }


__all__ = ["build_comprehensive_chinese_report"]
