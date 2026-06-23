from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

from .datatypes import OCCLUDED_UNKNOWN, OCCUPIED, RegionProposal, SlotHypothesis, SyntheticSample, ToolResult


def bev_extent(grid_cfg: dict) -> list[float]:
    """
    Return matplotlib extent as [y_min, y_max, x_max, x_min] so forward x appears upward.
    """
    return [grid_cfg["y_min"], grid_cfg["y_max"], grid_cfg["x_max"], grid_cfg["x_min"]]


def _savefig(path: str | Path, dpi: int = 160) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _slot_bounds(slot: SlotHypothesis) -> tuple[float, float, float, float]:
    x_min, y_min = slot.polygon_xy.min(axis=0)
    x_max, y_max = slot.polygon_xy.max(axis=0)
    return float(x_min), float(x_max), float(y_min), float(y_max)


def _draw_slots(ax, slots: list[SlotHypothesis] | None) -> None:
    if not slots:
        return
    for slot in slots:
        x_min, x_max, y_min, y_max = _slot_bounds(slot)
        color = "#8a2be2" if slot.is_target_candidate else "#6b7280"
        lw = 2.0 if slot.is_target_candidate else 0.8
        ax.add_patch(Rectangle((y_min, x_min), y_max - y_min, x_max - x_min, fill=False, ec=color, lw=lw))
        if slot.is_target_candidate:
            ax.text((y_min + y_max) / 2, x_min - 0.5, "Target", color=color, ha="center", va="top", fontsize=8)


def _draw_regions(ax, selected_regions: list[RegionProposal] | None, grid_cfg: dict) -> None:
    if not selected_regions:
        return
    res = grid_cfg["resolution"]
    for region in selected_regions:
        i_min, j_min, i_max, j_max = region.bbox_cells
        x_min = grid_cfg["x_min"] + i_min * res
        x_max = grid_cfg["x_min"] + i_max * res
        y_min = grid_cfg["y_min"] + j_min * res
        y_max = grid_cfg["y_min"] + j_max * res
        ax.add_patch(Rectangle((y_min, x_min), y_max - y_min, x_max - x_min, fill=False, ec="#2563eb", lw=1.8))
        ax.text(y_min, x_min, region.region_id, color="#1d4ed8", fontsize=9, weight="bold", va="bottom")


def _draw_ego(ax) -> None:
    ax.scatter([0.0], [0.0], marker="^", s=90, c="#111827", label="Ego", zorder=5)
    ax.text(0.6, 1.0, "Ego", color="#111827", fontsize=8)


def _setup_bev_axis(ax, title: str, grid_cfg: dict) -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Left y (m)")
    ax.set_ylabel("Forward x (m)")
    ax.set_xlim(grid_cfg["y_min"], grid_cfg["y_max"])
    ax.set_ylim(grid_cfg["x_max"], grid_cfg["x_min"])
    ax.grid(color="#d1d5db", lw=0.35, alpha=0.5)


def save_map(
    arr: np.ndarray,
    title: str,
    output_path: str,
    grid_cfg: dict,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    slots: list | None = None,
    selected_regions: list | None = None,
    target_slot_mask: np.ndarray | None = None,
) -> None:
    """Save a BEV map with consistent coordinate system."""
    del target_slot_mask
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr, extent=bev_extent(grid_cfg), aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    _setup_bev_axis(ax, title, grid_cfg)
    _draw_slots(ax, slots)
    _draw_regions(ax, selected_regions, grid_cfg)
    _draw_ego(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    _savefig(output_path)


def save_occupancy_map(
    occupancy: np.ndarray,
    title: str,
    output_path: str,
    grid_cfg: dict,
    slots: list | None = None,
    selected_regions: list | None = None,
) -> None:
    """Save occupancy map with fixed semantic colors."""
    cmap = ListedColormap(["#d9d9d9", "#b7e4c7", "#3f0d12", "#f59e0b"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(occupancy, extent=bev_extent(grid_cfg), aspect="equal", cmap=cmap, norm=norm)
    _setup_bev_axis(ax, title, grid_cfg)
    _draw_slots(ax, slots)
    _draw_regions(ax, selected_regions, grid_cfg)
    _draw_ego(ax)
    legend_items = [
        Patch(facecolor="#d9d9d9", label="Unknown"),
        Patch(facecolor="#b7e4c7", label="Free"),
        Patch(facecolor="#3f0d12", label="Occupied"),
        Patch(facecolor="#f59e0b", label="Occluded unknown"),
        Patch(facecolor="none", edgecolor="#8a2be2", label="Target slot"),
        Patch(facecolor="none", edgecolor="#2563eb", label="Selected region"),
        Patch(facecolor="#111827", label="Ego"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.92)
    _savefig(output_path)


def save_error_map(
    error_map: np.ndarray,
    title: str,
    output_path: str,
    grid_cfg: dict,
    selected_regions: list | None = None,
    slots: list | None = None,
) -> None:
    """Save error map with fixed colors."""
    cmap = ListedColormap(["#ffffff", "#dc2626", "#2563eb", "#9ca3af"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(error_map, extent=bev_extent(grid_cfg), aspect="equal", cmap=cmap, norm=norm)
    _setup_bev_axis(ax, title, grid_cfg)
    _draw_slots(ax, slots)
    _draw_regions(ax, selected_regions, grid_cfg)
    _draw_ego(ax)
    legend_items = [
        Patch(facecolor="#ffffff", edgecolor="#d1d5db", label="Correct / ignored"),
        Patch(facecolor="#dc2626", label="False free"),
        Patch(facecolor="#2563eb", label="False occupied"),
        Patch(facecolor="#9ca3af", label="Unresolved unknown"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.92)
    _savefig(output_path)


def save_scene_overview(
    sample: SyntheticSample,
    output_path: str,
    grid_cfg: dict,
) -> None:
    """Save top-down synthetic scene overview."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#f3f4f6")
    _setup_bev_axis(ax, "Synthetic Parking Scene Overview", grid_cfg)
    _draw_slots(ax, sample.fake_slots)
    _draw_ego(ax)

    for vehicle in sample.metadata["vehicles"]:
        cx, cy = vehicle["center"]
        ax.add_patch(
            Rectangle(
                (cy - vehicle["width"] / 2, cx - vehicle["length"] / 2),
                vehicle["width"],
                vehicle["length"],
                facecolor="#7f1d1d",
                edgecolor="#3f0d12",
                alpha=0.85,
            )
        )
    for obstacle in sample.metadata.get("hidden_obstacles", []):
        cx, cy = obstacle["center"]
        ax.add_patch(
            Rectangle(
                (cy - obstacle["width"] / 2, cx - obstacle["length"] / 2),
                obstacle["width"],
                obstacle["length"],
                facecolor="#581c87",
                edgecolor="#2e1065",
                alpha=0.9,
            )
        )
        ax.text(cy + 0.8, cx, "Hidden obstacle", color="#581c87", fontsize=8, weight="bold")
    pillar = sample.metadata["pillar"]
    ax.scatter([pillar["center"][1]], [pillar["center"][0]], s=220, c="#374151", marker="s")
    ax.add_patch(Rectangle((-4.0, 18.0), 6.0, 16.0, facecolor="#f59e0b", alpha=0.18, edgecolor="#f59e0b"))
    ax.text(-2.4, 25.0, "Pillar creates occlusion", color="#92400e", fontsize=9, weight="bold")
    ax.text(7.8, 13.0, "Occupied slots", color="#7f1d1d", fontsize=9, weight="bold")
    ax.text(-8.0, 19.4, "Target slot", color="#8a2be2", fontsize=9, weight="bold", ha="center")
    ax.text(1.0, 1.0, "Ego vehicle", color="#111827", fontsize=9, weight="bold")
    _savefig(output_path)


def save_lidar_bev(
    points: np.ndarray,
    title: str,
    output_path: str,
    grid_cfg: dict,
    slots: list | None = None,
    obstacle_z_threshold: float = 0.25,
) -> None:
    """Save observed LiDAR BEV scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    if points.size:
        ground_mask = points[:, 2] <= obstacle_z_threshold
        obstacle_mask = ~ground_mask

        if np.any(ground_mask):
            ax.scatter(
                points[ground_mask, 1],
                points[ground_mask, 0],
                c="#9ca3af",
                s=0.7,
                alpha=0.18,
                linewidths=0,
                label="ground / free-return",
            )
        if np.any(obstacle_mask):
            sc = ax.scatter(
                points[obstacle_mask, 1],
                points[obstacle_mask, 0],
                c=points[obstacle_mask, 2],
                s=5.0,
                cmap="magma",
                vmin=obstacle_z_threshold,
                vmax=2.5,
                alpha=0.92,
                linewidths=0,
                label="obstacle-height return",
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("obstacle point z height (m)")
    _setup_bev_axis(ax, title, grid_cfg)
    _draw_slots(ax, slots)
    _draw_ego(ax)
    if points.size:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    _savefig(output_path)


def save_slot_selection_panel(slot_scores: list[dict], output_path: str) -> None:
    """Save a panel explaining target slot selection from known slot map."""
    top_scores = slot_scores[:10]
    labels = [item["slot_id"] for item in top_scores][::-1]
    values = [item["score"] for item in top_scores][::-1]
    colors = ["#7c3aed" if idx == len(values) - 1 else "#64748b" for idx in range(len(values))]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(labels, values, color=colors)
    ax.set_title("Known Slot Map: Candidate Slot Selection Scores", fontsize=15, weight="bold")
    ax.set_xlabel("slot score from current belief")
    ax.grid(axis="x", color="#d1d5db", lw=0.5, alpha=0.8)

    best = top_scores[0]
    explanation = (
        f"Selected target: {best['slot_id']}\n"
        f"free_ratio={best['free_ratio']:.2f}, occupied_ratio={best['occupied_ratio']:.2f}, "
        f"unknown_ratio={best['unknown_ratio']:.2f}\n"
        f"occlusion_mean={best['occlusion_mean']:.2f}, entrance_unknown={best['entrance_unknown']:.2f}\n\n"
        "Known slot map provides geometry only. Current LiDAR-derived belief estimates whether each slot is safe/clear."
    )
    ax.text(
        0.02,
        0.04,
        explanation,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )
    _savefig(output_path)


def save_selected_regions_map(
    base_map: np.ndarray,
    title: str,
    output_path: str,
    grid_cfg: dict,
    regions: list[RegionProposal],
    slots: list | None = None,
    cmap: str = "inferno",
) -> None:
    """Save map with selected region boxes and labels."""
    save_map(
        base_map,
        title,
        output_path,
        grid_cfg,
        cmap=cmap,
        vmin=float(np.nanmin(base_map)),
        vmax=float(np.nanmax(base_map)),
        colorbar_label="score",
        slots=slots,
        selected_regions=regions,
    )


def save_tool_evidence_panel(
    regions: list[RegionProposal],
    tool_results: list[ToolResult],
    output_path: str,
) -> None:
    """Save text panel explaining tool calls and evidence."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    ax.set_title("Active Tool Evidence", fontsize=16, weight="bold", loc="left", pad=16)
    y = 0.94
    for region in regions:
        results = [r for r in tool_results if r.region_id == region.region_id]
        lines = [f"{region.region_id} | issue: {region.issue_type} | priority: {region.priority:.3f}"]
        for result in results:
            update = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in result.updates.items())
            lines.append(f"tools: {result.tool_name}")
            lines.append(f"evidence: {update}")
            lines.append(f"summary: {result.summary}")
        text = "\n".join(fill(line, width=120) for line in lines)
        ax.text(
            0.02,
            y,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
        )
        y -= 0.17
    _savefig(output_path)


def save_agent_reasoning_panel(reasoning: dict, output_path: str) -> None:
    """Save a visual panel that makes the active perception agent explicit."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.25], height_ratios=[0.9, 1.1])
    ax_goal = fig.add_subplot(gs[0, 0])
    ax_slot = fig.add_subplot(gs[1, 0])
    ax_actions = fig.add_subplot(gs[:, 1])
    for ax in [ax_goal, ax_slot, ax_actions]:
        ax.axis("off")

    fig.suptitle("Decision-Aware Active Perception Agent", fontsize=18, weight="bold", y=0.98)

    goal_lines = [
        "Goal",
        fill(reasoning.get("goal", ""), width=58),
        "",
        "Agent type",
        fill(reasoning.get("agent_type", ""), width=58),
        "",
        "Belief inputs",
    ]
    goal_lines += [f"- {item}" for item in reasoning.get("belief_inputs", [])]
    ax_goal.text(
        0.02,
        0.98,
        "\n".join(goal_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )

    target = reasoning.get("target_slot_selection", {})
    best = target.get("best_score", {}) or {}
    improvement = reasoning.get("observed_improvement", {})
    slot_lines = [
        "Target slot selection",
        f"selected_slot: {target.get('selected_slot_id', 'n/a')}",
        fill(target.get("reason", ""), width=58),
        "",
        "Best slot evidence",
        f"score: {best.get('score', 0.0):.3f}",
        f"free_ratio: {best.get('free_ratio', 0.0):.2f}",
        f"occupied_ratio: {best.get('occupied_ratio', 0.0):.2f}",
        f"unknown_ratio: {best.get('unknown_ratio', 0.0):.2f}",
        f"occlusion_mean: {best.get('occlusion_mean', 0.0):.2f}",
        "",
        "Observed improvement",
        f"occupied_iou: {improvement.get('occupied_iou_before', 0.0):.3f} -> {improvement.get('occupied_iou_after', 0.0):.3f}",
        "target_unknown: "
        f"{improvement.get('target_slot_unknown_ratio_before', 0.0):.3f} -> "
        f"{improvement.get('target_slot_unknown_ratio_after', 0.0):.3f}",
        f"false_free: {improvement.get('false_free_before', 0)} -> {improvement.get('false_free_after', 0)}",
    ]
    rounds = reasoning.get("rounds", [])
    if rounds:
        slot_lines.extend(["", "Round progress"])
        for round_info in rounds:
            before = round_info.get("observation", {}).get("target_slot_unknown_ratio_before", 0.0)
            after = round_info.get("target_slot_unknown_ratio_after", 0.0)
            target_before = round_info.get("observation", {}).get("target_slot_before", "n/a")
            target_after = round_info.get("target_slot_after", target_before)
            switch = " switch" if round_info.get("switched_target") else ""
            regions = ",".join(round_info.get("selected_regions", []))
            slot_lines.append(
                f"round {round_info.get('round')}: {target_before}->{target_after}{switch} | "
                f"unknown {before:.3f}->{after:.3f} | {regions}"
            )
            advice = round_info.get("ai_policy_advice")
            if advice:
                slot_lines.append(
                    f"  ai_policy: {advice.get('mode')} -> {','.join(round_info.get('validated_action_ids', []))}"
                )
    ax_slot.text(
        0.02,
        0.98,
        "\n".join(slot_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )

    action_blocks = ["Active questions"]
    action_blocks += [f"- {q}" for q in reasoning.get("active_questions", [])]
    action_blocks.append("")
    action_blocks.append("Selected actions")
    for action in reasoning.get("actions", []):
        tools = ", ".join(action.get("selected_tools", [])) or "none"
        relevance = action.get("parking_relevance", {})
        action_blocks.extend(
            [
                f"{action.get('region_id')} | {action.get('issue_type')}",
                fill(f"question: {action.get('active_question')}", width=78),
                fill(f"why: {action.get('why_selected')}", width=78),
                f"priority={relevance.get('priority', 0.0):.3f}, "
                f"decision={relevance.get('mean_decision_impact', 0.0):.2f}, "
                f"uncertainty={relevance.get('mean_uncertainty', 0.0):.2f}, "
                f"occlusion={relevance.get('mean_occlusion', 0.0):.2f}",
                fill(f"tool: {tools} | {action.get('chosen_tool_reason')}", width=78),
                fill(f"update: {action.get('belief_update_summary')}", width=78),
                "",
            ]
        )

    ax_actions.text(
        0.02,
        0.98,
        "\n".join(action_blocks),
        va="top",
        ha="left",
        fontsize=8.7,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    _savefig(output_path)


def save_ai_agent_briefing_panel(briefing: dict, output_path: str) -> None:
    """Save a demo-friendly AI agent briefing panel."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off")
    ax.set_title("AI Agent Briefing Layer", fontsize=18, weight="bold", pad=18)

    left_lines = [
        briefing.get("headline", "AI Agent Briefing"),
        "",
        f"mode: {briefing.get('mode', 'unknown')}",
        f"initial_target: {briefing.get('initial_target', 'n/a')}",
        f"final_target: {briefing.get('final_target', 'n/a')}",
        "",
        "Recommended decision",
        fill(str(briefing.get("recommended_decision", "")), width=58),
        "",
        "Key finding",
        fill(str(briefing.get("key_finding", "")), width=58),
        "",
        "Demo takeaway",
        fill(str(briefing.get("demo_takeaway", "")), width=58),
    ]

    right_lines = ["Round-by-round evidence"]
    right_lines += [fill(f"- {item}", width=74) for item in briefing.get("round_by_round", [])]
    right_lines += ["", "Metrics"]
    right_lines += [fill(f"- {item}", width=74) for item in briefing.get("metric_summary", [])]
    right_lines += ["", "Limitations"]
    right_lines += [fill(f"- {item}", width=74) for item in briefing.get("limitations", [])]

    ax.text(
        0.03,
        0.95,
        "\n".join(left_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )
    ax.text(
        0.52,
        0.95,
        "\n".join(right_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.6,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )
    _savefig(output_path)


def _imshow_panel(ax, arr: np.ndarray, title: str, grid_cfg: dict, cmap, norm=None, vmin=None, vmax=None) -> None:
    ax.imshow(arr, extent=bev_extent(grid_cfg), aspect="equal", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    _setup_bev_axis(ax, title, grid_cfg)


def save_effectiveness_panel(
    gt_occupancy: np.ndarray,
    occupancy_before: np.ndarray,
    error_before: np.ndarray,
    priority: np.ndarray,
    occupancy_after: np.ndarray,
    error_after: np.ndarray,
    metrics: dict,
    output_path: str,
    grid_cfg: dict,
    regions: list[RegionProposal],
    slots: list[SlotHypothesis],
) -> None:
    """Save final 2x3 summary panel."""
    occ_cmap = ListedColormap(["#d9d9d9", "#b7e4c7", "#3f0d12", "#f59e0b"])
    occ_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], occ_cmap.N)
    err_cmap = ListedColormap(["#ffffff", "#dc2626", "#2563eb", "#9ca3af"])
    err_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], err_cmap.N)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    _imshow_panel(axes[0, 0], gt_occupancy, "GT occupancy", grid_cfg, occ_cmap, occ_norm)
    _imshow_panel(axes[0, 1], occupancy_before, "Initial occupancy from sparse observed LiDAR", grid_cfg, occ_cmap, occ_norm)
    _imshow_panel(axes[0, 2], error_before, "Error before active refinement", grid_cfg, err_cmap, err_norm)
    _imshow_panel(axes[1, 0], priority, "Decision-aware priority + selected regions", grid_cfg, "inferno", vmin=0, vmax=1)
    _imshow_panel(axes[1, 1], occupancy_after, "Refined occupancy after local tools", grid_cfg, occ_cmap, occ_norm)
    _imshow_panel(axes[1, 2], error_after, "Error after active refinement", grid_cfg, err_cmap, err_norm)

    for ax in axes.flat:
        _draw_slots(ax, slots)
        _draw_regions(ax, regions, grid_cfg)
        _draw_ego(ax)

    metrics_text = (
        f"False-free: {metrics['false_free_before']} -> {metrics['false_free_after']}\n"
        f"Occupied IoU: {metrics['occupied_iou_before']:.3f} -> {metrics['occupied_iou_after']:.3f}\n"
        f"Unresolved unknown: {metrics['unresolved_unknown_before']} -> {metrics['unresolved_unknown_after']}\n"
        f"Target slot unknown: {metrics['target_slot_unknown_ratio_before']:.2f} -> "
        f"{metrics['target_slot_unknown_ratio_after']:.2f}\n"
        f"Selected/error overlap: {metrics['selected_region_error_overlap']:.2f}\n"
        f"Tool calls: {metrics['num_tool_calls']}"
    )
    fig.text(
        0.5,
        0.02,
        metrics_text,
        ha="center",
        va="bottom",
        fontsize=10.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )
    fig.suptitle("DASP-Park Controlled Synthetic Active Perception Loop", fontsize=18, weight="bold", y=0.995)
    fig.tight_layout(rect=[0.0, 0.14, 1.0, 0.97])
    _savefig(output_path)
