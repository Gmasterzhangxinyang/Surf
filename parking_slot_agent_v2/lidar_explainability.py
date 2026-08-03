"""Human-readable rendering of the exact evidence used by the LiDAR hard gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from parking_slot_part2.media import LidarEvidencePack


_FONT_PATH = Path(
    "/home/ParkingAgent/DASP/simulation_carla/CARLA_0.9.16/Engine/Content/Slate/Fonts/DroidSansFallback.ttf"
)
FONT = FontProperties(fname=str(_FONT_PATH)) if _FONT_PATH.is_file() else FontProperties()
plt.rcParams["axes.unicode_minus"] = False


def _fp(size: float, *, bold: bool = False) -> FontProperties:
    value = FONT.copy()
    value.set_size(size)
    value.set_weight("bold" if bold else "normal")
    return value


def _candidate_box(decision: Mapping[str, Any]) -> dict[str, float] | None:
    raw = decision.get("occupied_evidence", {}).get("best_box", [])
    if not isinstance(raw, list):
        return None
    try:
        value = {str(key): float(number) for key, number in raw}
    except (TypeError, ValueError):
        return None
    required = {"center_x_m", "center_y_m", "yaw_rad", "length_m", "width_m"}
    return value if required.issubset(value) else None


def _box_corners(box: Mapping[str, float]) -> np.ndarray:
    half = np.array([box["length_m"] / 2.0, box["width_m"] / 2.0])
    local = np.array([[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]])
    yaw = box["yaw_rad"]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rotation.T + np.array([box["center_x_m"], box["center_y_m"]])


def _inside_box(points: np.ndarray, box: Mapping[str, float]) -> np.ndarray:
    centered = points[:, :2] - np.array([box["center_x_m"], box["center_y_m"]])
    yaw = -box["yaw_rad"]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    local = centered @ rotation.T
    return (np.abs(local[:, 0]) <= box["length_m"] / 2.0) & (np.abs(local[:, 1]) <= box["width_m"] / 2.0)


def _sample(mask: np.ndarray, limit: int = 18_000) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if len(indices) <= limit:
        return indices
    return indices[np.linspace(0, len(indices) - 1, limit, dtype=np.int64)]


def _free_xy(decision: Mapping[str, Any]) -> np.ndarray:
    details = decision.get("free_evidence", {}).get("details") or {}
    voxels = details.get("free_voxels", [])
    if not voxels:
        return np.empty((0, 2), dtype=float)
    array = np.asarray(voxels, dtype=float)
    xy = np.unique(array[:, :2], axis=0)
    return (xy + 0.5) * 0.25


def _metric(card: Mapping[str, Any], group: str, key: str, default: Any = 0) -> Any:
    value = card.get(group, {})
    return value.get(key, default) if isinstance(value, Mapping) else default


def _float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else 0.0


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and np.isfinite(value) else 0


def _gate_story(card: Mapping[str, Any], state: str) -> list[tuple[bool, str, str]]:
    free = card.get("free_geometry", {})
    occ = card.get("occupied_geometry", {})
    robust = card.get("robustness", {})
    if state == "free":
        return [
            (True, "核心自由体素", f"目标体积覆盖 {_float(free.get('observed_volume_ratio'))*100:.1f}%"),
            (True, "近地面射线", f"覆盖 {_float(free.get('near_ground_bev_coverage'))*100:.1f}%"),
            (False, "占用候选", f"边界比例 {_float(occ.get('boundary_ratio'))*100:.0f}% → 否决"),
            (True, "位姿稳定性", f"{robust.get('passing_variants', 0)}/{robust.get('total_variants', 0)} 通过"),
        ]
    if state == "occupied":
        return [
            (True, "车位核心障碍", f"{_int(occ.get('core_point_count')):,} 点"),
            (True, "三维形状", f"{_int(occ.get('supported_height_layers'))}个高度层，跨度 {_float(occ.get('height_span_m')):.2f} m"),
            (False, "自由空间", f"体积覆盖仅 {_float(free.get('observed_volume_ratio'))*100:.1f}%"),
            (True, "位姿稳定性", f"{robust.get('passing_variants', 0)}/{robust.get('total_variants', 0)} 通过"),
        ]
    return [
        (False, "核心射线", f"覆盖 {_float(free.get('core_ray_coverage'))*100:.1f}%"),
        (False, "自由空间", f"体积覆盖 {_float(free.get('observed_volume_ratio'))*100:.1f}%"),
        (False, "占用核心", f"{_int(occ.get('core_point_count'))} 点"),
        (False, "位姿稳定性", f"{robust.get('passing_variants', 0)}/{robust.get('total_variants', 0)} 通过"),
    ]


def _unknown_headline(card: Mapping[str, Any]) -> str:
    """Describe the measured Unknown mode instead of collapsing it to no data."""

    free = card.get("free_geometry", {})
    occupied = card.get("occupied_geometry", {})
    robust = card.get("robustness", {})
    core_ray = _float(free.get("core_ray_coverage"))
    volume = _float(free.get("observed_volume_ratio"))
    core_points = _int(occupied.get("core_point_count"))
    layers = _int(occupied.get("supported_height_layers"))
    boundary = _float(occupied.get("boundary_ratio"))
    passing = _int(robust.get("passing_variants"))
    total = _int(robust.get("total_variants"))
    if core_ray == 0.0 and volume == 0.0 and core_points == 0:
        return "目标核心完全未观测\n无自由射线，也无占用候选"
    if core_points > 0:
        return (
            f"弱占用候选：核心{core_points:,}点，{layers}个高度层\n"
            f"覆盖不足且稳定性{passing}/{total}，不能判占用"
        )
    if boundary >= 0.5:
        return (
            f"回波以车位边界为主：边界比例{boundary*100:.0f}%\n"
            "不能把静态边界当作车辆"
        )
    if core_ray > 0.0 or volume > 0.0:
        return (
            f"只有稀疏观测：核心射线{core_ray*100:.1f}%\n"
            f"体积覆盖{volume*100:.1f}%，不足以判空闲"
        )
    return "几何证据未通过终态门\n保留Unknown并报告具体阻断项"


def describe_lidar_render(
    pack: LidarEvidencePack,
    geometry_card: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact human-facing semantics used by the renderer.

    Keeping these labels in a testable structure prevents frame-window claims
    in the bitmap from drifting away from the attached evidence pack.
    """

    selected_frames = tuple(int(value) for value in pack.selected_frames)
    valid_frames = tuple(int(value) for value in pack.valid_frames)
    selected_count = len(selected_frames)
    valid_count = len(valid_frames)
    evidence_source = str(
        geometry_card.get("decision_context", {}).get(
            "evidence_source",
            "unknown_lidar_source",
        )
    )
    tail_ids = set(selected_frames[-min(15, selected_count):])
    tail_count = sum(value in tail_ids for value in valid_frames)
    early_count = valid_count - tail_count
    labels: list[str] = []
    if early_count:
        labels.append(f"扩展有效视角 {early_count}")
    if tail_count:
        labels.append(
            f"Part1尾部有效视角 {tail_count}"
            if selected_count > 15
            else f"当前证据有效视角 {tail_count}"
        )
    return {
        "selected_frame_count": selected_count,
        "valid_frame_count": valid_count,
        "evidence_source": evidence_source,
        "subtitle": (
            f"真实证据窗口：选择{selected_count}帧 / 有效{valid_count}帧 / "
            f"{evidence_source}；位置、证据归属、反证与代码门"
        ),
        "viewpoint_title": f"3  {selected_count}帧证据从哪些有效位置观察？",
        "viewpoint_labels": labels,
        "unknown_headline": _unknown_headline(geometry_card),
    }


def render_explainable_lidar(
    pack: LidarEvidencePack,
    geometry_card: Mapping[str, Any],
    output_path: str | Path,
    *,
    decision: Mapping[str, Any] | None = None,
) -> Path:
    """Render target-local LiDAR as spatial evidence and explicit hard gates."""

    decision = decision or {}
    state = (
        "free" if geometry_card.get("terminal_geometry_gate", {}).get("free_eligible")
        else "occupied" if geometry_card.get("terminal_geometry_gate", {}).get("occupied_eligible")
        else "unknown"
    )
    colors = {"free": "#0b9b72", "occupied": "#e14d50", "unknown": "#d99712"}
    state_zh = {"free": "空闲", "occupied": "占用", "unknown": "未知"}[state]
    points = np.asarray(pack.points_local_xyzi)
    selected_frames = np.asarray(pack.selected_frames, dtype=np.int64)
    valid_frames = np.asarray(pack.valid_frames, dtype=np.int64)
    semantics = describe_lidar_render(pack, geometry_card)
    selected_count = int(semantics["selected_frame_count"])
    box = _candidate_box(decision)
    box_mask = _inside_box(points, box) if box is not None else np.zeros(len(points), dtype=bool)
    free_xy = _free_xy(decision)

    figure = plt.figure(figsize=(18, 10), dpi=120, facecolor="#f4f7fb")
    grid = figure.add_gridspec(2, 3, width_ratios=(1.45, 0.95, 0.95), height_ratios=(1.0, 1.0), wspace=0.25, hspace=0.31)
    bev = figure.add_subplot(grid[:, 0]); side = figure.add_subplot(grid[0, 1]); viewpoints = figure.add_subplot(grid[0, 2]); gates = figure.add_subplot(grid[1, 1:])
    figure.suptitle(f"LiDAR工具到底检查了什么？  {geometry_card.get('slot_id', pack.metadata.get('slot_id', ''))}", x=0.055, y=.975, ha="left", fontproperties=_fp(22, bold=True), color="#142238")
    figure.text(
        .055,
        .935,
        str(semantics["subtitle"]),
        fontproperties=_fp(11),
        color="#66768c",
    )

    # Panel A: target-local evidence ownership.
    margin = np.asarray(pack.margin_polygon_local_m); target = np.asarray(pack.polygon_local_m); core = np.asarray(pack.core_polygon_local_m)
    extent = np.vstack((margin, target, core))
    x0, y0 = extent.min(axis=0) - .75; x1, y1 = extent.max(axis=0) + .75
    local_mask = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    background = _sample(local_mask & ~box_mask, 13_000)
    candidate = _sample(local_mask & box_mask & (points[:, 2] >= .30), 15_000)
    if len(background):
        bev.scatter(points[background, 0], points[background, 1], s=2, c="#aab7c7", alpha=.23, linewidths=0, rasterized=True)
    band_color = "#fff2d8" if state == "free" else "#f8fafc"
    bev.add_patch(Polygon(margin, closed=True, facecolor=band_color, edgecolor="#d99712", linewidth=1.5, linestyle="--", zorder=1))
    bev.add_patch(Polygon(target, closed=True, facecolor="#ffffffcc", edgecolor="#2a6fdb", linewidth=3, zorder=2))
    core_fill = {"free": "#d7f5e9", "occupied": "#fee2e2", "unknown": "#edf1f6"}[state]
    bev.add_patch(Polygon(core, closed=True, facecolor=core_fill, edgecolor="#142238", linewidth=2.2, zorder=3))
    if len(free_xy):
        inside = (free_xy[:, 0] >= x0) & (free_xy[:, 0] <= x1) & (free_xy[:, 1] >= y0) & (free_xy[:, 1] <= y1)
        bev.scatter(free_xy[inside, 0], free_xy[inside, 1], marker="s", s=72, c="#30c795", alpha=.66, linewidths=0, label="射线确认的自由体素", zorder=4)
    candidate_color = "#f59e0b" if state == "free" else "#e14d50"
    if len(candidate):
        bev.scatter(points[candidate, 0], points[candidate, 1], s=4.5, c=candidate_color, alpha=.72, linewidths=0, rasterized=True, zorder=5)
    if box is not None:
        corners = _box_corners(box)
        bev.add_patch(Polygon(corners, closed=True, facecolor="none", edgecolor=candidate_color, linewidth=3.2, linestyle="--" if state == "free" else "-", zorder=6))
        if state == "free":
            center = corners.mean(axis=0)
            bev.plot([center[0]-.5, center[0]+.5], [center[1]-.5, center[1]+.5], c="#e14d50", lw=4, zorder=7)
            bev.plot([center[0]-.5, center[0]+.5], [center[1]+.5, center[1]-.5], c="#e14d50", lw=4, zorder=7)
    if not np.any(local_mask):
        bev.text(0, 0, "目标车位内没有LiDAR回波", ha="center", va="center", fontproperties=_fp(17, bold=True), color="#e14d50", bbox=dict(boxstyle="round,pad=.5", fc="white", ec="#e14d50", lw=2), zorder=8)
    if state == "free":
        headline = f"绿色覆盖核心：{_float(_metric(geometry_card,'free_geometry','observed_volume_ratio'))*100:.1f}%\n橙色候选触碰边界，被否决"
    elif state == "occupied":
        headline = f"红色三维核心：{_int(_metric(geometry_card,'occupied_geometry','core_point_count')):,}点\n连续{_int(_metric(geometry_card,'occupied_geometry','support_frame_count'))}帧支持"
    else:
        headline = str(semantics["unknown_headline"])
    bev.text(.03, .97, headline, transform=bev.transAxes, va="top", fontproperties=_fp(14, bold=True), color="#142238", bbox=dict(boxstyle="round,pad=.5", fc="white", ec=colors[state], lw=2), zorder=9)
    bev.set_xlim(x0, x1); bev.set_ylim(y0, y1); bev.set_aspect("equal")
    bev.set_title("1  车位内部到底有什么？", loc="left", fontproperties=_fp(16, bold=True), color="#142238", pad=12)
    bev.set_xlabel("车位长度方向 x (m)", fontproperties=_fp(10)); bev.set_ylabel("车位宽度方向 y (m)", fontproperties=_fp(10)); bev.grid(color="#dce4ee", lw=.7)
    for label in bev.get_xticklabels()+bev.get_yticklabels(): label.set_fontproperties(_fp(8))
    legend = [
        Line2D([0], [0], color="#2a6fdb", lw=3, label="完整车位边界"),
        Line2D([0], [0], color="#142238", lw=3, label="必须被解释的核心区"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#30c795", markersize=10, label="射线确认的自由体素"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=candidate_color, markersize=8, label="障碍候选点"),
    ]
    bev.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False, prop=_fp(9))

    # Panel B: vertical structure of only the fitted candidate.
    side.axhspan(0, .3, color="#edf1f6"); side.axhline(.3, color="#66768c", linestyle="--", lw=1.5)
    if len(candidate):
        heights = points[candidate, 2]
        side.scatter(points[candidate, 0], heights, s=4, c=heights, cmap="autumn_r", vmin=.3, vmax=2.2, alpha=.7, linewidths=0, rasterized=True)
    else:
        side.text(.5, .52, "没有可拟合的三维候选", transform=side.transAxes, ha="center", fontproperties=_fp(13, bold=True), color="#e14d50")
    side.set_xlim(float(target[:,0].min())-.3, float(target[:,0].max())+.3); side.set_ylim(-.1, 2.5)
    side.set_title("2  是否形成车体高度？", loc="left", fontproperties=_fp(14, bold=True), color="#142238")
    side.set_xlabel("车位长度 x (m)", fontproperties=_fp(9)); side.set_ylabel("离地高度 z (m)", fontproperties=_fp(9)); side.grid(color="#dce4ee", lw=.7)
    side.text(.02, .12, "0.30 m障碍阈值", transform=side.transAxes, fontproperties=_fp(8), color="#66768c")
    for label in side.get_xticklabels()+side.get_yticklabels(): label.set_fontproperties(_fp(8))

    # Panel C: show the actual selected/valid frame history.  The last up-to-15
    # selected frame IDs are the Part1 tail only when an extended pack exists.
    origins = np.asarray(pack.observation_origins_local_xyz)[:, :2]
    if len(origins):
        viewpoints.plot(origins[:,0], origins[:,1], color="#2a6fdb", lw=2, alpha=.7)
        tail_frame_ids = set(
            int(value) for value in selected_frames[-min(15, selected_count):]
        )
        tail_mask = np.asarray(
            [int(value) in tail_frame_ids for value in valid_frames],
            dtype=bool,
        )
        early_mask = ~tail_mask
        if np.any(early_mask):
            viewpoints.scatter(
                origins[early_mask,0],
                origins[early_mask,1],
                s=24,
                c="#2a6fdb",
                label=f"扩展有效视角 {int(np.sum(early_mask))}",
            )
        if np.any(tail_mask):
            tail_label = (
                f"Part1尾部有效视角 {int(np.sum(tail_mask))}"
                if selected_count > 15
                else f"当前证据有效视角 {int(np.sum(tail_mask))}"
            )
            viewpoints.scatter(
                origins[tail_mask,0],
                origins[tail_mask,1],
                s=35,
                c="#f59e0b",
                label=tail_label,
            )
        choices=np.linspace(0,len(origins)-1,min(3,len(origins)),dtype=int)
        for idx in choices: viewpoints.plot([origins[idx,0],0],[origins[idx,1],0],c="#8fa9c6",lw=.8,ls="--")
    viewpoints.add_patch(Rectangle((target[:,0].min(),target[:,1].min()),np.ptp(target[:,0]),np.ptp(target[:,1]),facecolor="#d7f5e9",edgecolor="#142238",lw=2))
    viewpoints.scatter([0],[0],marker="x",s=70,c="#142238",zorder=5)
    viewpoints.set_aspect("equal", adjustable="datalim"); viewpoints.grid(color="#dce4ee", lw=.7)
    viewpoints.set_title(
        str(semantics["viewpoint_title"]),
        loc="left",
        fontproperties=_fp(14, bold=True),
        color="#142238",
    )
    if len(origins):
        viewpoints.legend(loc="best", frameon=False, prop=_fp(8))
    viewpoints.set_xlabel("x (m)",fontproperties=_fp(8)); viewpoints.set_ylabel("y (m)",fontproperties=_fp(8))
    for label in viewpoints.get_xticklabels()+viewpoints.get_yticklabels(): label.set_fontproperties(_fp(7))

    # Panel D: convert measurements into a visible decision chain.
    gates.axis("off"); gates.set_title("4  代码门如何裁决？", loc="left", fontproperties=_fp(16, bold=True), color="#142238", pad=8)
    story = _gate_story(geometry_card, state)
    for index, (passed, name, value) in enumerate(story):
        y=.82-index*.18; color="#0b9b72" if passed else "#e14d50"
        gates.text(.03,y,"✓" if passed else "×",ha="center",va="center",fontproperties=_fp(13,bold=True),color="white",bbox=dict(boxstyle="circle,pad=.23",fc=color,ec="none"))
        gates.text(.08,y,name,va="center",fontproperties=_fp(12,bold=True),color="#142238")
        gates.text(.38,y,value,va="center",fontproperties=_fp(11),color="#66768c")
    gates.text(.73,.48,"最终",ha="center",fontproperties=_fp(10),color="#66768c")
    gates.text(.73,.31,state_zh,ha="center",va="center",fontproperties=_fp(24,bold=True),color="white",bbox=dict(boxstyle="round,pad=.5",fc=colors[state],ec="none"))
    gates.text(.73,.12,"模型建议 + 确定性硬门",ha="center",fontproperties=_fp(9),color="#66768c")
    gates.annotate("",xy=(.65,.31),xytext=(.52,.31),arrowprops=dict(arrowstyle="->",lw=2,color="#2a6fdb"))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


__all__ = ["describe_lidar_render", "render_explainable_lidar"]
