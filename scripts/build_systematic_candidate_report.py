#!/usr/bin/env python3
"""Build the root-level Chinese report, tables, figures, and hash manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _count_states(rows: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {"free": 0, "occupied": 0, "unknown": 0}
    for row in rows.values():
        counts[str(row["state"])] += 1
    return counts


def _decision(path: Path, slot_id: str) -> dict[str, Any]:
    payload = _load(path)
    return next(row for row in payload["decisions"] if row["slot_id"] == slot_id)


def _ego_xy(points: Any, snapshot: dict[str, Any]) -> Any:
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    anchor = np.asarray(snapshot["anchor_pose"]["map_xy"], dtype=np.float64)
    scale = float(snapshot["map_units_per_meter"])
    yaw = float(snapshot["anchor_pose"]["map_yaw_rad"])
    delta = (values - anchor) / scale
    forward = np.asarray([math.cos(yaw), math.sin(yaw)])
    left = np.asarray([-math.sin(yaw), math.cos(yaw)])
    return np.column_stack([delta @ forward, delta @ left])


def _method_plots(root: Path, study: dict[str, Any]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # Figure 0: end-to-end algorithm, not merely experiment statistics.
    fig, ax = plt.subplots(figsize=(15.5, 7.8), constrained_layout=True)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12", facecolor=color, edgecolor="#334155", linewidth=1.3))
        ax.text(x + 0.18, y + h - 0.32, title, fontsize=11, fontweight="bold", va="top")
        ax.text(x + 0.18, y + h - 0.72, body, fontsize=8.7, va="top", linespacing=1.35)

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, linewidth=1.35, color="#334155"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, fontsize=8, ha="center", color="#475569")

    box(0.2, 5.2, 2.2, 1.8, "1  Causal inputs", "15 records ending at t0\npose-corrected map points\n1,397 slot polygons", "#dbeafe")
    box(2.9, 5.2, 2.3, 1.8, "2  Observation scope", "near / crossing / hit\nactual ray traversal\ncore coverage + quality", "#e0f2fe")
    box(5.7, 5.2, 2.5, 1.8, "3  Occupied evidence", "vehicle-height points\nbox ownership + 3D voxels\nlegacy + footprint pillar veto", "#fee2e2")
    box(8.7, 5.2, 2.4, 1.8, "4  Free evidence", "ray-cleared volume\nnear-ground coverage\n2 views + no core hit", "#dcfce7")
    box(11.6, 5.2, 2.2, 1.8, "5  Robust decision", "7 pose perturbations\nall hard gates\nfree / occupied / unknown", "#fef3c7")
    box(0.8, 1.5, 2.7, 1.9, "6  Part2 candidate gate", "state = unknown\nagent_observable = true\n|relative bearing| <= 90 deg\nevidence resource exists", "#ede9fe")
    box(4.2, 1.5, 2.8, 1.9, "7  W/K causal history", "last W records, frame <= t0\nuniformly sample K incl. t0\nselected dev setting: 100 / 20", "#e0e7ff")
    box(7.7, 1.5, 2.8, 1.9, "8  Agent tool reasoning", "receives unknown_reasons\ninspect LiDAR / RGB if valid\nresolve each original blocker", "#fae8ff")
    box(11.2, 1.5, 3.1, 1.9, "9  Deterministic final gate", "model cannot bypass geometry\nfree requires all reasons resolved\notherwise keep unknown", "#f1f5f9")
    for left, right in ((2.4, 2.9), (5.2, 5.7), (8.2, 8.7), (11.1, 11.6)):
        arrow(left, 6.1, right, 6.1)
    arrow(12.7, 5.2, 2.15, 3.4, "unknown only")
    arrow(3.5, 2.45, 4.2, 2.45)
    arrow(7.0, 2.45, 7.7, 2.45)
    arrow(10.5, 2.45, 11.2, 2.45)
    ax.text(8.0, 7.65, "ParkingAgent algorithm: from causal LiDAR to an auditable Part2 decision", ha="center", fontsize=16, fontweight="bold")
    ax.text(8.0, 0.55, "Part1 never selects the final parking slot. Part2 only reasons over forward, observable Unknown slots and remains fail-closed.", ha="center", fontsize=10, color="#334155")
    path = figures / "Figure_0_complete_algorithm.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    base_root = root / "artifacts" / "frame_009277_baseline"
    final_root = root / "artifacts" / "frame_009277_agent_final"
    baseline_map = _load(base_root / "local_map.json")
    final_map = _load(final_root / "local_map.json")
    base_decision = _decision(base_root / "slot_decisions.json", "slot_1253")
    final_decision = _decision(final_root / "slot_decisions.json", "slot_1253")
    features = base_decision["occupied_evidence"]["features"]
    pca = float(features["robust_pca_linearity"])
    extent_x = float(features["robust_extent_x_m"])
    extent_y = float(features["robust_extent_y_m"])
    height = float(features["extent_z_m"])
    area = extent_x * extent_y
    aspect = height / max(extent_x, extent_y)
    fraction = float(features["robust_point_fraction"])

    # Figure 4: the actual 9277 slot and the physical gate conditions.
    fig = plt.figure(figsize=(15.2, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0), height_ratios=(1.0, 1.0))
    map_axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]))
    for ax, snapshot, state, title in (
        (map_axes[0], baseline_map, "occupied", "Legacy gates: slot_1253 = OCCUPIED"),
        (map_axes[1], final_map, "unknown", "Footprint-aware gates: slot_1253 = UNKNOWN"),
    ):
        target = next(row for row in snapshot["slots"] if row["slot_id"] == "slot_1253")
        target_center = _ego_xy([target["center_map"]], snapshot)[0]
        for row in snapshot["slots"]:
            polygon = _ego_xy(row["polygon_map"], snapshot)
            is_target = row["slot_id"] == "slot_1253"
            face = ("#ef4444" if state == "occupied" else "#f59e0b") if is_target else "#e2e8f0"
            alpha = 0.82 if is_target else 0.35
            ax.add_patch(Polygon(polygon, closed=True, facecolor=face, edgecolor="#334155", alpha=alpha, linewidth=2.2 if is_target else 0.6))
        ax.scatter([0], [0], marker=(3, 0, -90), s=120, color="#2563eb", label="ego @ 9277")
        ax.set_xlim(target_center[0] - 8, target_center[0] + 8)
        ax.set_ylim(target_center[1] - 6, target_center[1] + 6)
        ax.set_aspect("equal")
        ax.grid(alpha=0.18)
        ax.set_xlabel("forward (m)")
        ax.set_ylabel("left (m)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(target_center[0], target_center[1], "1253", ha="center", va="center", fontsize=9, fontweight="bold")

    ax = fig.add_subplot(grid[:, 1])
    names = ["PCA >= 0.90", "XY area <= 0.30", "height >= 0.80", "aspect >= 1.50", "inlier >= 0.70"]
    values = [pca / 0.90, 0.30 / area, height / 0.80, aspect / 1.50, fraction / 0.70]
    raw = [f"{pca:.3f}", f"{area:.3f} m2", f"{height:.3f} m", f"{aspect:.2f}", f"{fraction:.3f}"]
    y = np.arange(len(names))
    ax.barh(y, np.minimum(values, 2.8), color="#0f766e")
    ax.axvline(1.0, color="#dc2626", linewidth=1.8, linestyle="--", label="gate threshold")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 3.0)
    ax.set_xlabel("threshold-normalized margin (passes at >= 1)")
    ax.set_title("Why the legacy occupied hypothesis is pillar-like", fontsize=12, fontweight="bold")
    for index, (value, text_value) in enumerate(zip(values, raw)):
        ax.text(min(value, 2.75) + 0.04, index, text_value + "  PASS", va="center", fontsize=9, color="#065f46")
    ax.legend(loc="lower right")
    ax.text(0.02, -0.16, "Legacy rule missed it because PCA 0.915 < 0.95. The new conjunction also uses footprint area and vertical aspect.\nThe safe action is not FREE: strong Occupied closes, ownership/boundary remains unresolved, so Part1 returns UNKNOWN.", transform=ax.transAxes, fontsize=9.2, va="top", bbox={"facecolor": "#fff7ed", "edgecolor": "#fb923c", "pad": 8})
    fig.suptitle("Actual frame 9277 pillar-risk ablation: slot_1253", fontsize=15, fontweight="bold")
    path = figures / "Figure_4_pillar_case_slot1253.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    # Figure 5: actual forward-half-plane queue on the 9277 local map.
    queue = _load(final_root / "unknown_agent_queue.json")
    queue_bearing = {row["slot_id"]: float(row["audit"]["candidate_relative_bearing_deg"]) for row in queue["items"]}
    fig, ax = plt.subplots(figsize=(11.5, 8.0), constrained_layout=True)
    ax.axvspan(0, 22, color="#cffafe", alpha=0.42, label="eligible forward 180 deg")
    ax.axvspan(-22, 0, color="#f1f5f9", alpha=0.65, label="rear half-plane: excluded")
    ax.axvline(0, color="#0f172a", linestyle="--", linewidth=1.4)
    state_color = {"free": "#22c55e", "occupied": "#ef4444", "unknown": "#94a3b8"}
    for row in final_map["slots"]:
        polygon = _ego_xy(row["polygon_map"], final_map)
        slot_id = row["slot_id"]
        queued = slot_id in queue_bearing
        face = "#2563eb" if queued else state_color[row["state"]]
        ax.add_patch(Polygon(polygon, closed=True, facecolor=face, edgecolor="#1e293b", alpha=0.80 if queued else 0.40, linewidth=2.0 if queued else 0.7))
        center = polygon.mean(axis=0)
        if row["state"] == "unknown":
            if queued:
                label = f"{slot_id.replace('slot_', '')}\n{queue_bearing[slot_id]:+.1f} deg"
            else:
                bearing = math.degrees(math.atan2(float(center[1]), float(center[0])))
                label = f"{slot_id.replace('slot_', '')}\n{bearing:+.0f} deg"
            ax.text(center[0], center[1], label, ha="center", va="center", fontsize=7.2, fontweight="bold" if queued else "normal")
    ax.arrow(0, 0, 4.0, 0, width=0.15, head_width=0.8, head_length=1.0, color="#111827", length_includes_head=True)
    ax.text(0, -1.0, "ego t0=9277", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-20, 22)
    ax.set_ylim(-20, 20)
    ax.set_aspect("equal")
    ax.set_xlabel("forward relative to current yaw (m)")
    ax.set_ylabel("left (m)")
    ax.set_title("Actual Part2 candidate gate at frame 9277: 4 blue Unknown slots enter the queue", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left")
    path = figures / "Figure_5_actual_forward180_queue.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    # Figure 6: exact frame selection rather than an abstract W/K label.
    run = next(row for row in study["development_experiments"] if int(row["anchor_frame"]) == 9277 and int(row["history_span_W"]) == 100 and int(row["sample_count_K"]) == 20)
    selected_ids = [int(value) for value in run["selected_frame_ids"]]
    start = 9277 - 99
    all_ids = np.arange(start, 9278)
    fig, ax = plt.subplots(figsize=(14.5, 4.8), constrained_layout=True)
    ax.scatter(all_ids, np.zeros_like(all_ids), s=12, color="#cbd5e1", label="100-frame causal history pool")
    ax.scatter(selected_ids, np.zeros(len(selected_ids)), s=55, color="#4f46e5", zorder=3, label="20 uniformly sampled records")
    ax.axvspan(9277 - 14, 9277, color="#fbbf24", alpha=0.22, label="current Part1 15-record window")
    ax.axvline(9277, color="#dc2626", linewidth=2.0)
    ax.text(9277, 0.17, "t0 = 9277\nalways included", ha="right", color="#991b1b", fontweight="bold")
    for index, frame_id in enumerate(selected_ids):
        if index % 2 == 0 or frame_id == 9277:
            ax.text(frame_id, -0.10, str(frame_id), rotation=60, ha="right", va="top", fontsize=7)
    ax.set_ylim(-0.32, 0.34)
    ax.set_yticks([])
    ax.set_xlabel("LiDAR input record / frame id")
    ax.set_title("Exact causal sampling used by W=100, K=20 at development anchor 9277", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.text(0.5, 0.05, "index_j = round(j * (W-1)/(K-1)),  j=0...K-1;  no frame after t0 is allowed", transform=ax.transAxes, ha="center", fontsize=10, bbox={"facecolor": "white", "edgecolor": "#94a3b8", "pad": 5})
    path = figures / "Figure_6_exact_W100_K20_sampling.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)
    return outputs


def _plot(root: Path, study: dict[str, Any]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    aggregate = study["development_aggregate"]
    spans = sorted({int(row["history_span_W"]) for row in aggregate})
    samples = sorted({int(row["sample_count_K"]) for row in aggregate})
    resolution = np.full((len(samples), len(spans)), np.nan)
    agreement = np.full_like(resolution, np.nan)
    for row in aggregate:
        y = samples.index(int(row["sample_count_K"]))
        x = spans.index(int(row["history_span_W"]))
        resolution[y, x] = 100.0 * float(row["resolved_rate"])
        agreement[y, x] = 100.0 * float(row["exact_dense_agreement_rate"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for ax, matrix, title in zip(axes, (resolution, agreement), ("Resolved candidates (%)", "Exact dense-reference agreement (%)")):
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(spans)), spans)
        ax.set_yticks(range(len(samples)), samples)
        ax.set_xlabel("History span W")
        ax.set_ylabel("Sample count K")
        ax.set_title(title)
        for y in range(len(samples)):
            for x in range(len(spans)):
                value = matrix[y, x]
                ax.text(x, y, "-" if np.isnan(value) else f"{value:.0f}", ha="center", va="center", color="white" if not np.isnan(value) and value < 60 else "black", fontsize=9)
        fig.colorbar(image, ax=ax, shrink=0.82)
    path = figures / "Figure_1_WK_ablation.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    paired = study["paired_part1"]
    labels = [str(row["anchor_frame"]) for row in paired]
    full = [int(row["baseline_part2_candidates_360deg"]) for row in paired]
    same_front = [int(row["same_unknowns_in_forward_180deg"]) for row in paired]
    agent_front = [int(row["agent_part2_candidates_forward_180deg"]) for row in paired]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    width = 0.26
    ax.bar(x - width, full, width, label="Legacy unknown queue (360 deg)")
    ax.bar(x, same_front, width, label="Same legacy unknowns in front 180 deg")
    ax.bar(x + width, agent_front, width, label="Agent Part1 unknowns in front 180 deg")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Anchor frame")
    ax.set_ylabel("Candidate count")
    ax.set_title("Forward field-of-regard candidate filtering")
    ax.legend(fontsize=8)
    path = figures / "Figure_2_forward180_candidates.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    heldout = study["heldout_aggregate"]
    fig, ax = plt.subplots(figsize=(7.8, 4.5), constrained_layout=True)
    names = [f"{row['history_span_W']}/{row['sample_count_K']}" for row in heldout]
    rates = [100.0 * float(row["resolved_rate"]) for row in heldout]
    bars = ax.bar(names, rates, color=["#64748b", "#0f766e"][: len(names)])
    ax.set_ylim(0, max(5.0, max(rates, default=0.0) * 1.2))
    ax.set_ylabel("Resolved held-out candidates (%)")
    ax.set_xlabel("W/K")
    ax.set_title("Held-out validation: no resolution gain")
    for bar, value in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.1, f"{value:.1f}%", ha="center")
    path = figures / "Figure_3_heldout_validation.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)
    return outputs


def _table_html(rows: list[dict[str, Any]], fields: list[str]) -> str:
    head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    study = _load(args.study)
    if study.get("status") != "complete":
        raise ValueError("systematic study is not complete")

    paired_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for row in study["paired_part1"]:
        paired_rows.append({
            "anchor_frame": row["anchor_frame"],
            "baseline_free": row["baseline_counts"]["free"],
            "baseline_occupied": row["baseline_counts"]["occupied"],
            "baseline_unknown": row["baseline_counts"]["unknown"],
            "agent_free": row["agent_counts"]["free"],
            "agent_occupied": row["agent_counts"]["occupied"],
            "agent_unknown": row["agent_counts"]["unknown"],
            "legacy_queue_360": row["baseline_part2_candidates_360deg"],
            "same_unknowns_front180": row["same_unknowns_in_forward_180deg"],
            "agent_queue_front180": row["agent_part2_candidates_forward_180deg"],
            "baseline_seconds": round(float(row["baseline_seconds"]), 4),
            "agent_seconds": round(float(row["agent_seconds"]), 4),
        })
        for transition in row["state_transitions"]:
            transition_rows.append({
                "anchor_frame": row["anchor_frame"],
                "slot_id": transition["slot_id"],
                "baseline_state": transition["baseline_state"],
                "agent_state": transition["agent_state"],
                "agent_reason": transition["agent_record"].get("decision_reason", ""),
                "agent_unknown_reasons": "|".join(transition["agent_record"].get("unknown_reasons", [])),
            })
    _write_csv(root / "results" / "Table_1_part1_and_front180.csv", paired_rows, paired_rows[0].keys())
    _write_csv(root / "results" / "Table_2_state_transitions.csv", transition_rows, ("anchor_frame", "slot_id", "baseline_state", "agent_state", "agent_reason", "agent_unknown_reasons"))
    _write_csv(root / "results" / "Table_3_development_WK.csv", study["development_aggregate"], study["development_aggregate"][0].keys())
    _write_csv(root / "results" / "Table_4_heldout.csv", study["heldout_aggregate"], study["heldout_aggregate"][0].keys())

    dev_best = study["selected_best"]
    heldout_by_pair = {(int(row["history_span_W"]), int(row["sample_count_K"])): row for row in study["heldout_aggregate"]}
    best_pair = (int(dev_best["history_span_W"]), int(dev_best["sample_count_K"]))
    heldout_best = heldout_by_pair[best_pair]
    heldout_base = heldout_by_pair[(15, 15)]
    dev_ci = _wilson(int(dev_best["exact_dense_agreement_count"]), int(dev_best["case_count"]))
    hold_ci = _wilson(int(heldout_best["resolved_count"]), int(heldout_best["case_count"]))

    legacy_total = sum(int(row["legacy_queue_360"]) for row in paired_rows)
    legacy_front = sum(int(row["same_unknowns_front180"]) for row in paired_rows)
    agent_front = sum(int(row["agent_queue_front180"]) for row in paired_rows)
    pillar_transitions = [row for row in transition_rows if row["baseline_state"] == "occupied" and row["agent_state"] == "unknown"]
    method_figures = _method_plots(root, study)
    figures = _plot(root, study)

    report = f"""# ParkingAgent：Part1 柱体抑制、Part2 前向 180° 与历史采样消融

生成日期：2026-07-28
主实验状态：完成
GT 状态：当前 Hybrid3D 主实验没有充分人工 GT；100/100 是稠密因果参考，不是真值。

## 结论先行

- 9277 锚点的旧算法为 **2 free / 8 occupied / 11 unknown**；新柱体门后为 **2 / 7 / 12**。唯一 `occupied → unknown` 是 `{pillar_transitions[0]['slot_id'] if pillar_transitions else '无'}`，这是风险抑制，不是“已由 GT 证明的假阳性修复”。
- 正式 Part2 队列现在只接受相对当前车辆航向 **[-90°, +90°]** 的 Unknown。6 个锚点中，旧 360° 可观察 Unknown 共 {legacy_total} 个，其中仅 {legacy_front} 个位于同一批旧 Unknown 的前向 180°；新 Part1 状态变化后正式前向候选共 {agent_front} 个。
- 开发集选择出 **W={best_pair[0]}, K={best_pair[1]}**。其固定候选分辨率为 {int(dev_best['resolved_count'])}/{int(dev_best['case_count'])}，与 100/100 稠密参考精确一致 {int(dev_best['exact_dense_agreement_count'])}/{int(dev_best['case_count'])}（{100*float(dev_best['exact_dense_agreement_rate']):.1f}%，Wilson 95% CI {100*dev_ci[0]:.1f}–{100*dev_ci[1]:.1f}%），终态矛盾 {int(dev_best['terminal_contradiction_count'])}。
- 但留出锚点上，15/15 是 {int(heldout_base['resolved_count'])}/{int(heldout_base['case_count'])}，100/20 也是 {int(heldout_best['resolved_count'])}/{int(heldout_best['case_count'])}（Wilson 95% CI {100*hold_ci[0]:.1f}–{100*hold_ci[1]:.1f}%）。因此 **100/20 只是开发集最佳，未验证为可泛化最佳**。
- 当前主要瓶颈是遮挡、真实射线覆盖、视角分离和稳定性硬门；单纯增加历史跨度不会保证提升。

## 1. 当前主线与文件审计

项目文件指南记录 26,229 个文件。审计按“源代码/当前输入/当前主线产物/历史实验/受保护人工数据”分类进行，未把 24 个候选的旧 30 m Agent-v2 图当作当前局部 Part1 结果。当前正式输入是：

- `outputs/frame_map_dataset_pose_corrected_final/frames.csv` 与逐帧 map 点云；
- `outputs/full_icpark_allframes_vehicle_cluster/slot_database.json` 的 1,397 个几何车位；
- 当前 Part1 `parking_slot_hybrid_3d`；当前正式 Part2 队列 `parking_slot_part2`；
- 受保护人工标签仅 8 个，且属于旧 box-scoring 验证协议，不能直接提供本研究准确率。

## 2. 从 Part1 到 Part2 的完整逻辑

![完整算法流程](figures/Figure_0_complete_algorithm.png)

1. Part1 只取以当前帧 t0 结束的 3–15 个连续 LiDAR 记录，转到 pose-corrected map 坐标。
2. 对每个已知车位计算 near/crossing/hit、真实射线核心覆盖和点云质量；无局部证据的车位不赋状态。
3. Occupied 必须同时通过点数、支持帧、高度、体素、核心归属、邻位冲突、边界、形态和 7 个姿态扰动稳定性门。
4. Free 必须由实际射线穿越给出体积覆盖、近地覆盖、至少两视点及足够视角分离，并且不能有未解决核心命中。
5. 其余为 Unknown，且 `unknown_reasons` 随队列交给 Agent；Part1 不选择最终停车位。
6. Part2 候选合同为 `Unknown ∧ agent_observable ∧ forward_180 ∧ evidence_resource_available`。Agent 终态仍必须由确定性证据门复核。

## 3. 柱体误判修正

旧门只有在 `PCA≥0.95 ∧ robust max-XY≤0.75 m ∧ inlier≥0.80` 时认为是紧凑竖直结构。车辆候选框把柱旁杂点纳入后，PCA 或 inlier 很容易稍低，柱体便可通过 Occupied。

新增次级物理门要求以下条件同时成立：`PCA≥0.90`、稳健 XY 面积 `≤0.30 m²`、高度 `≥0.80 m`、`height/maxXY≥1.50`、至少 2 帧以及稳健点比例 `≥0.70`。命中时不是武断地判 free，而是关闭强 Occupied，降级为 Unknown 交给 Part2。旧门保留，便于受控消融。

![真实柱体案例](figures/Figure_4_pillar_case_slot1253.png)

## 4. 前向 180° 候选

对车位中心与当前车辆位置的向量计算 `bearing = wrap(atan2(dy,dx) - yaw)`；闭区间 `|bearing|≤90°` 才能进入 Part2。边界 ±90° 保留，正后方 ±180° 排除。这个门只改变 Part2 候选资格，不篡改 Part1 的 free/occupied/unknown 状态。

![前向候选]({figures[1].relative_to(root).as_posix()})

![9277真实前向队列](figures/Figure_5_actual_forward180_queue.png)

## 5. W/K 消融协议

- 开发锚点：{study['development_anchors']}；留出锚点：{study['heldout_anchors']}，两者不重叠。
- 候选身份冻结为每个锚点新 Part1 的 `Unknown ∧ observable ∧ forward180`；改变 W/K 不重新挑候选。
- 历史池严格因果，只含 `frame≤t0`。在最后 W 个输入记录上用包含首尾的等距索引均匀抽取 K 帧，保证 t0 必选。
- 网格：W∈{{15,30,60,100}}，K∈{{5,10,15,20}} 且 K≤W。
- 100/100 只用于开发选择和留出审计的稠密因果参考，不参与候选生成，也不称 GT。
- 选择规则为字典序：终态矛盾最少 → 与稠密参考相同的终态最多 → 全状态精确一致最多 → K 更小 → W 更小 → 运行时间更短。

![W/K 消融]({figures[0].relative_to(root).as_posix()})

![100/20实际取帧](figures/Figure_6_exact_W100_K20_sampling.png)

## 6. 原始版与 Agent 证据阶段对比

这里的“Agent 后”严格指：新柱体门 + 前向候选合同 + 扩展历史证据阶段经确定性终态门后的结果；本轮没有把 VLM 自报置信度当准确率，也没有把旧 24 候选 OpenAI replay 混入主表。

{_table_html(paired_rows, ['anchor_frame','baseline_free','baseline_occupied','baseline_unknown','agent_free','agent_occupied','agent_unknown','legacy_queue_360','same_unknowns_front180','agent_queue_front180'])}

## 7. 留出集与失败分析

![留出集]({figures[2].relative_to(root).as_posix()})

留出集 17 个固定前向 Unknown 上，100/20 没有新增终态。该负结果说明开发集的 10/25 提升依赖场景；历史跨度无法弥补以下信息缺口：

- 车位核心被前景车辆/墙/柱遮挡，射线没有穿越待判体积；
- 多帧仍来自近似同一视点，视角分离不足；
- 核心中存在弱命中，但不足以归属车辆，也不足以证明 free；
- 采样改变候选点云和 box hypothesis，导致非单调的稳定性门结果。

## 8. 能宣称与不能宣称

可以宣称：实现了可审计的柱体风险门和正式前向 180° 队列；开发集的最佳工程组合是 100/20；完整留出实验未发现覆盖提升。

不能宣称：柱体假阳性率下降多少、系统占用准确率、Nature 级统计显著性或 100/20 普适最优。原因是当前主协议没有足量、独立、盲法人工 GT，且只有单路线数据。

## 9. 投稿前必须补齐

1. 对当前 Hybrid3D 协议分层抽取至少数百个 slot-anchor，双人盲标并报告一致性；柱体/墙/车辆/遮挡必须单独分层。
2. 以 anchor/route 为聚类单位做 bootstrap CI、风险—覆盖曲线和配对检验。
3. 增加独立停车场、天气、昼夜、传感器或仿真域；留出集不得参与阈值选择。
4. 将 no-Agent、15/15、连续 60/60、均匀 60/15、100/20、Camera/LiDAR 消融放在同一 GT 上。
5. 校准相机投影后才允许 RGB 独立支持 free；否则继续 fail-closed。

## 10. 复现与产物

- 原始机器结果：`results/systematic_study.json`
- 主表：`results/Table_1_part1_and_front180.csv` 至 `Table_4_heldout.csv`
- 9277 完整基线和新算法产物：`artifacts/frame_009277_baseline/`、`artifacts/frame_009277_agent_final/`
- 代码：`parking_slot_hybrid_3d/config.py`、`occupied.py`、`reporting.py`、`scripts/run_systematic_candidate_experiments.py`
- 报告产物哈希：`MANIFEST.sha256`；本轮源代码哈希：`SOURCE_CODE.sha256`

完整复现命令见 `REPRODUCE.md`。所有旧输出和受保护人工数据均未覆盖。
"""
    (root / "完整研究报告.md").write_text(report, encoding="utf-8")

    html_report = """<!doctype html><html><head><meta charset='utf-8'><title>ParkingAgent algorithm and systematic study</title><style>
:root{--ink:#172033;--teal:#0f4c5c;--line:#cbd5e1;--soft:#f8fafc;--blue:#1d4ed8}*{box-sizing:border-box}body{font-family:system-ui,'Noto Sans CJK SC',sans-serif;max-width:1260px;margin:0 auto;padding:0 28px 80px;line-height:1.68;color:var(--ink);background:#fff}header{margin:0 -28px 32px;padding:44px 44px 34px;background:linear-gradient(125deg,#0f3d3e,#164e63);color:white}header h1{margin:0 0 10px;font-size:34px}header p{max-width:940px;margin:4px 0;color:#dbeafe}.nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}.nav a{color:white;border:1px solid #94a3b8;border-radius:18px;padding:5px 12px;text-decoration:none;font-size:13px}h2{color:var(--teal);font-size:25px;margin-top:44px;border-bottom:2px solid #dbeafe;padding-bottom:7px}h3{color:#155e75}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{border:1px solid var(--line);border-radius:10px;padding:15px;background:var(--soft)}.card .n{font-size:25px;font-weight:750;color:var(--blue)}.warning{border-left:5px solid #f97316;background:#fff7ed;padding:14px 18px}.method{border-left:5px solid #0ea5e9;background:#f0f9ff;padding:14px 18px}table{border-collapse:collapse;width:100%;font-size:12.5px;display:block;overflow:auto}th,td{border:1px solid var(--line);padding:6px;text-align:left;white-space:nowrap}th{background:#e2e8f0}figure{margin:24px 0}figure img{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:white}figcaption{font-size:13px;color:#475569;margin-top:7px}code,pre{background:#f1f5f9}code{padding:2px 4px}pre{padding:16px;border:1px solid var(--line);overflow:auto;line-height:1.5}.formula{font-family:ui-monospace,monospace;background:#f8fafc;border:1px solid var(--line);padding:12px 16px;border-radius:7px}.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}}</style></head><body>"""
    html_report += f"""<header><h1>ParkingAgent：算法、真实案例与系统性实验</h1><p>不是只展示结果：本页从 Part1 的真实射线证据开始，逐步说明柱体抑制、前向 180° 候选、W/K 历史采样、Agent 工具推理和最终硬门。</p><div class='nav'><a href='#pipeline'>完整流程</a><a href='#part1'>Part1算法</a><a href='#pillar'>柱体案例</a><a href='#candidate'>前向候选</a><a href='#history'>历史采样</a><a href='#results'>实验结果</a></div></header>
<div class='grid'><div class='card'><div class='n'>2/8/11</div><div>9277 旧 Part1<br>free / occupied / unknown</div></div><div class='card'><div class='n'>2/7/12</div><div>9277 新柱体门后</div></div><div class='card'><div class='n'>11 → 4</div><div>9277 队列：360°旧 Unknown → 正式前向候选</div></div><div class='card'><div class='n'>100 / 20</div><div>开发集选择的 W/K；留出集未复现增益</div></div></div>
<section id='pipeline'><h2>1. 系统到底用了什么算法</h2><figure><img src='{method_figures[0].relative_to(root).as_posix()}'><figcaption>完整数据流。Part1 给每个局部车位状态和 Unknown 原因；只有前向、可观测的 Unknown 才进入 Part2。模型不能越过确定性几何门。</figcaption></figure></section>
<section id='part1'><h2>2. Part1：15 帧因果 Hybrid-3D 证据</h2><div class='method'><b>输入：</b>以当前帧 t0 结束的 3–15 个 LiDAR 记录、pose-corrected map 点云、1,397 个车位 polygon。没有使用全停车场实时占用 GT。</div>
<div class='two'><div><h3>2.1 有没有有效 LiDAR 观测</h3><ol><li>车辆轨迹距车位足够近，得到 <code>near_frames</code>。</li><li>用每个真实回波从雷达原点到终点做射线穿越，得到 <code>crossing_frames</code>。</li><li>回波落入车位 prism 得到 <code>hit_frames</code>。</li><li>核心射线覆盖不足时为 partial/out-of-route，不用“ROI里没点”冒充 free。</li></ol></div><div><h3>2.2 三态决策</h3><ul><li><b>Occupied：</b>车辆高度点、跨帧支持、3D体素、核心归属、邻位/边界、形态和姿态稳定性全部通过。</li><li><b>Free：</b>核心体积被真实射线穿越、近地覆盖充分、至少两视点且无未解决核心命中。</li><li><b>Unknown：</b>任何证据不足、遮挡、弱障碍、归属冲突或姿态不稳定；原因写入 <code>unknown_reasons</code>。</li></ul></div></div>
<pre>for each local slot:
    scope = evaluate(near, actual_ray_crossing, hits, core_coverage)
    occupied = vehicle_height_points -> box hypotheses -> 3D/ownership/pillar gates
    free = actual_ray_traversal -> volume/ground/viewpoint/core-hit gates
    if occupied.strong and pose_stable: state = OCCUPIED
    elif free.strong and pose_stable and no_weak_obstacle: state = FREE
    else: state = UNKNOWN + exact reason codes</pre></section>
<section id='pillar'><h2>3. 为什么柱子以前会变成 Occupied，现在哪里修了</h2><div class='formula'>旧门：PCA ≥ 0.95 ∧ max(XY) ≤ 0.75 m ∧ inlier ≥ 0.80<br>新次级门：PCA ≥ 0.90 ∧ XY面积 ≤ 0.30 m² ∧ 高度 ≥ 0.80 m ∧ 高度/maxXY ≥ 1.50 ∧ inlier ≥ 0.70</div><p>旧门只认“极端线性”的柱体。柱旁杂点被车辆框包进去以后，PCA 从 0.95 以下漏过去。新门必须由小占地、足够高度、竖直长宽比和跨帧支持共同确认；命中后只关闭强 Occupied，<b>不会直接判 Free</b>。</p><figure><img src='{method_figures[1].relative_to(root).as_posix()}'><figcaption>真实 9277 / slot_1253。旧候选 PCA=0.915、XY面积=0.090 m²、高度=1.318 m、竖直比=2.60，全部满足新柱体风险门。最终从 occupied 降为 unknown，仍保留边界/归属不确定性给 Agent。</figcaption></figure></section>
<section id='candidate'><h2>4. Part2 candidate 怎么得到</h2><div class='formula'>candidate = (Part1 state = Unknown) ∧ agent_observable ∧ |wrap(atan2(dy,dx) − ego_yaw)| ≤ 90° ∧ evidence resource available</div><p>角度使用<b>当前帧 t0 的车辆姿态</b>，不是车位自身方向，也不是旧的相机 ±80° 预检查。±90° 边界保留，后半平面排除。这个门只决定是否进 Agent 队列，不修改 Part1 状态。</p><figure><img src='{method_figures[2].relative_to(root).as_posix()}'><figcaption>真实 9277 局部图。蓝色 4 个 Unknown 分别位于 +72.3°、+26.1°、+30.2°、+39.7°，进入队列；其余后方 Unknown 不进入。</figcaption></figure></section>
<section id='history'><h2>5. 历史 W/K 是怎么取的</h2><p>W 是 t0 之前最后 W 个输入记录的因果历史跨度，K 是其中实际加载的帧数。均匀抽样包含历史首端和 t0：</p><div class='formula'>index(j) = round(j × (W−1)/(K−1)), j=0…K−1；所有 frame ≤ t0，t0 必选</div><figure><img src='{method_figures[3].relative_to(root).as_posix()}'><figcaption>9277 的实际 100/20 帧号，而不是概念示意。黄色区域是原始 Part1 的最近 15 帧。</figcaption></figure><p class='warning'><b>重要：</b>100/20 是三个开发锚点按安全优先规则选出的组合；三个留出锚点 17 个候选上没有新增终态，因此当前不能把它叫成“已经验证的最佳参数”。</p></section>
<section id='agent'><h2>6. Agent 拿到什么、能做什么</h2><ul><li>收到 slot_id、原始 <code>unknown_reasons</code>、Occupied/Free 数值证据和绑定到该车位的 LiDAR/RGB资源。</li><li>工具检查扩展 LiDAR、有效相机帧和遮挡；无有效标定的 RGB 不能单独证明 Free。</li><li>若输出 Free，必须逐项解决全部原始 Unknown 原因且无 blocker；非法状态、缺字段或几何硬门失败都保持 Unknown。</li><li>因此 Agent 是证据调度与原因消解器，不是可以随意覆盖 Part1 的分类器。</li></ul></section>
<section id='results'><h2>7. 原始版、修正版和消融结果</h2>{_table_html(paired_rows, ['anchor_frame','baseline_free','baseline_occupied','baseline_unknown','agent_free','agent_occupied','agent_unknown','legacy_queue_360','same_unknowns_front180','agent_queue_front180'])}<figure><img src='{figures[0].relative_to(root).as_posix()}'><figcaption>开发集 W/K 网格：左为解决比例，右为与 100/100 稠密因果参考的一致率。</figcaption></figure><figure><img src='{figures[1].relative_to(root).as_posix()}'><figcaption>6 个锚点上 360° Unknown、同批前向 Unknown 和新算法正式候选数量。</figcaption></figure><figure><img src='{figures[2].relative_to(root).as_posix()}'><figcaption>留出实验是负结果：100/20 没有比 15/15 新解决候选。</figcaption></figure></section>
<section><h2>8. 科学边界</h2><p>本轮证明的是算法实现、候选合同、证据覆盖和稠密参考一致性。当前 Hybrid3D 协议没有足量人工 GT，所以不能声称柱体 FPR、占用准确率或 Nature 级统计显著性。完整限制见 <a href='Nature投稿就绪度.md'>Nature投稿就绪度.md</a>，逐项方法与数字见 <a href='完整研究报告.md'>完整研究报告.md</a>。</p></section>"""
    html_report += "</body></html>"
    (root / "index.html").write_text(html_report, encoding="utf-8")

    algorithm_doc = f"""# ParkingAgent 算法与实现说明

## 一句话定义

系统不是在地图中强制挑一个车位，而是先用 15 帧严格因果 LiDAR 给局部车位生成 `free / occupied / unknown`，再把当前车辆前方 180° 内、有工具证据的 Unknown 交给 Agent；Agent 的结论仍须通过确定性几何门。

## 完整流程图

![完整算法](figures/{method_figures[0].name})

## Part1 观测范围

有效观测不是“车位 ROI 内没有点”。必须存在实际雷达射线：原点为该帧雷达位姿，终点为真实回波；射线穿越车位核心才增加 coverage。near/crossing/hit、缺帧、地面拟合质量共同决定 scope。

## Occupied

候选点限制在离地 0.30–2.20 m。强 Occupied 需要至少 40 点、3 个支持帧、时序支持 0.20、z95≥0.60 m、高度跨度≥0.35 m、核心重叠≥0.45、邻位重叠<0.35、边界比例<0.50、至少 8 个 3D 体素和 2 个高度层，并通过形态、柱体与 7 个姿态扰动门。

## Free

Free 是正证据：至少 5 个射线帧、2 个视点、视角分离≥10°、体积覆盖≥0.70、近地 BEV 覆盖≥0.70、未观测最大连通分量≤0.20、遮挡≤0.20，且无未解决核心 hit/弱障碍。

## 柱体门

![柱体真实案例](figures/{method_figures[1].name})

新次级门：`PCA≥0.90 ∧ XY面积≤0.30 m² ∧ 高度≥0.80 m ∧ 高度/maxXY≥1.50 ∧ robust frames≥2 ∧ inlier≥0.70`。命中只会阻止 Occupied，不会把柱子所在车位判 Free。

## Part2 前向候选

![前向候选](figures/{method_figures[2].name})

`candidate = Unknown ∧ observable ∧ |relative_bearing|≤90° ∧ resource_available`。relative bearing 使用 t0 车辆 yaw。9277 正式队列为 slot_1250、slot_1253、slot_1254、slot_1255。

## W/K 扩展证据

![采样](figures/{method_figures[3].name})

最后 W 个因果记录上，按 `round(j(W−1)/(K−1))` 取 K 个索引。候选身份在 Part1 后冻结，消融不能因 W/K 改变而重新选“更容易”的车位。

## Agent 终态合同

Agent 接收全部原始 Unknown 原因。Free 必须精确解决全部原因且没有 blocker；Occupied/Free 都要通过绑定到 slot/anchor/config/hash 的证据硬门。失败、缺资源或模型格式错误均保持 Unknown。
"""
    (root / "算法与实现说明.md").write_text(algorithm_doc, encoding="utf-8")

    readiness = """# Nature 投稿就绪度审计

## 已完成

- 当前主线/历史实验/受保护数据分层审计。
- 柱体风险门、前向 180° 正式队列及回归测试。
- 固定候选、严格因果的 W/K 开发消融和不重叠留出实验。
- 机器可读逐车位结果、统计表、图和 SHA-256 manifest。

## 尚未达到投稿门槛

- 当前协议缺少足量人工 GT，不能计算 sensitivity/specificity/FPR/FNR。
- 留出集未复现 100/20 的覆盖提升。
- 只有单路线/单传感器域，外部有效性不足。
- 相机投影没有独立标定通过，RGB 不能单独给 free 终态。
- 未完成双人盲标、聚类 bootstrap、功效分析和跨域复现。

结论：工程闭环和可复现实验已完成，但科学证据不足以诚实声称“Nature-ready”。
"""
    (root / "Nature投稿就绪度.md").write_text(readiness, encoding="utf-8")

    reproduce = """# 复现命令

在项目首页 `/home/ParkingAgent/ParkingAgent` 执行：

```bash
python3 -m unittest -q tests.test_hybrid_3d_occupied tests.test_hybrid_3d_pipeline

python3 scripts/run_systematic_candidate_experiments.py \\
  --frames-csv outputs/frame_map_dataset_pose_corrected_final/frames.csv \\
  --slot-db outputs/full_icpark_allframes_vehicle_cluster/slot_database.json \\
  --map-points-dir outputs/frame_map_dataset_pose_corrected_final/map_points \\
  --output Nature_ParkingAgent_实验报告_20260728/results/systematic_study.json \\
  --development-anchors 5283 7605 9277 \\
  --heldout-anchors 234 3160 9443 \\
  --history-spans 15 30 60 100 --sample-counts 5 10 15 20 \\
  --dense-reference-frames 100 --cache-size 128

python3 scripts/build_systematic_candidate_report.py \\
  --study Nature_ParkingAgent_实验报告_20260728/results/systematic_study.json \\
  --root Nature_ParkingAgent_实验报告_20260728
```

注意：实验脚本会原子更新 JSON checkpoint；若要完全重跑，使用一个新的输出文件名，不要覆盖受保护数据或历史输出。
"""
    (root / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")

    source_paths = (
        PROJECT_ROOT / "parking_slot_hybrid_3d" / "config.py",
        PROJECT_ROOT / "parking_slot_hybrid_3d" / "occupied.py",
        PROJECT_ROOT / "parking_slot_hybrid_3d" / "reporting.py",
        PROJECT_ROOT / "tests" / "test_hybrid_3d_occupied.py",
        PROJECT_ROOT / "tests" / "test_hybrid_3d_pipeline.py",
        PROJECT_ROOT / "scripts" / "run_systematic_candidate_experiments.py",
        PROJECT_ROOT / "scripts" / "build_systematic_candidate_report.py",
    )
    source_lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(PROJECT_ROOT).as_posix()}"
        for path in source_paths
    ]
    (root / "SOURCE_CODE.sha256").write_text(
        "\n".join(source_lines) + "\n", encoding="utf-8"
    )

    manifest_path = root / "MANIFEST.sha256"
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"root": str(root), "best_W": best_pair[0], "best_K": best_pair[1], "manifest_files": len(lines)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
