from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STATE_COLORS = {True: "#15803d", False: "#9ca3af"}


def write_drift_plots(
    output_dir: Path,
    keyframes: list[dict[str, Any]],
    original_poses: np.ndarray,
    corrected_poses: np.ndarray,
    map_layers: dict[str, np.ndarray],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    distance = np.asarray([float(row["travel_distance_m"]) for row in keyframes])
    accepted = np.asarray([bool(row["accepted"]) for row in keyframes])
    before = np.asarray([float(row["residual_before_m"]) for row in keyframes])
    after = np.asarray([float(row["residual_after_m"]) for row in keyframes])
    smoothed = np.asarray([float(row["smoothed_residual_m"]) for row in keyframes])
    dx = np.asarray([float(row["smoothed_dx_m"]) for row in keyframes])
    dy = np.asarray([float(row["smoothed_dy_m"]) for row in keyframes])
    dyaw = np.asarray([float(row["smoothed_dyaw_deg"]) for row in keyframes])

    residual_path = output_dir / "structural_residual_before_after.png"
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.plot(distance, before, color="#b91c1c", linewidth=1.2, label="before")
    ax.plot(distance, after, color="#86a98d", linewidth=0.8, alpha=0.7, label="per-frame local optimum")
    ax.plot(distance, smoothed, color="#15803d", linewidth=1.2, label="smoothed trajectory")
    ax.scatter(distance[accepted], smoothed[accepted], color="#15803d", s=12, label="accepted constraint")
    ax.set(xlabel="Cumulative odometry distance [m]", ylabel="Trimmed structural residual [m]")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(residual_path, dpi=150)
    plt.close(fig)

    correction_path = output_dir / "pose_correction_over_distance.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(distance, dx, label="dx", color="#2563eb")
    axes[0].plot(distance, dy, label="dy", color="#dc2626")
    axes[0].set_ylabel("Smoothed correction [m]")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(distance, dyaw, color="#7c3aed")
    axes[1].set(xlabel="Cumulative odometry distance [m]", ylabel="Yaw correction [deg]")
    axes[1].grid(alpha=0.25)
    fig.savefig(correction_path, dpi=150)
    plt.close(fig)

    trajectory_path = output_dir / "trajectory_before_after_on_map.png"
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    for layer_name, points in map_layers.items():
        if len(points):
            sample_step = max(1, len(points) // 100000)
            sampled = points[::sample_step]
            ax.scatter(sampled[:, 0], sampled[:, 1], s=0.25, color="#d1d5db", label=layer_name)
    ax.plot(original_poses[:, 0], original_poses[:, 1], color="#dc2626", linewidth=1.0, label="original")
    ax.plot(corrected_poses[:, 0], corrected_poses[:, 1], color="#2563eb", linewidth=1.0, label="corrected")
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="map x", ylabel="map y")
    ax.legend(markerscale=8)
    ax.grid(alpha=0.15)
    fig.savefig(trajectory_path, dpi=170)
    plt.close(fig)
    return {
        "residual_plot": residual_path.name,
        "correction_plot": correction_path.name,
        "trajectory_plot": trajectory_path.name,
    }


def write_drift_report(
    path: Path,
    summary: dict[str, Any],
    keyframes: list[dict[str, Any]],
    plots: dict[str, str],
) -> None:
    rows = []
    for row in keyframes:
        color = STATE_COLORS[bool(row["accepted"])]
        rows.append(
            "<tr>"
            f"<td>{int(row['frame'])}</td>"
            f"<td>{float(row['travel_distance_m']):.1f}</td>"
            f"<td style='color:{color};font-weight:700'>{'accepted' if row['accepted'] else 'rejected'}</td>"
            f"<td>{float(row['residual_before_m']):.3f}</td>"
            f"<td>{float(row['residual_after_m']):.3f}</td>"
            f"<td>{float(row['smoothed_residual_m']):.3f}</td>"
            f"<td>{float(row['inlier_ratio']):.3f}</td>"
            f"<td>{int(row['matched_points'])}</td>"
            f"<td>{float(row['raw_dx_m']):.2f}, {float(row['raw_dy_m']):.2f}, {float(row['raw_dyaw_deg']):.2f}</td>"
            f"<td>{float(row['smoothed_dx_m']):.2f}, {float(row['smoothed_dy_m']):.2f}, {float(row['smoothed_dyaw_deg']):.2f}</td>"
            f"<td>{html.escape(str(row['rejection_reason']))}</td>"
            "</tr>"
        )
    summary_json = html.escape(json.dumps(summary, indent=2, ensure_ascii=False))
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Pose drift correction report</title>
<style>
body{{font-family:Arial,"Noto Sans CJK SC",sans-serif;margin:24px;color:#172033;background:#f6f8fb}}
.panel{{background:white;border:1px solid #d8dee9;border-radius:6px;padding:18px;margin:16px 0}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(170px,1fr));gap:12px}}
.metric{{border-left:4px solid #2563eb;background:#eef4ff;padding:12px}}
img{{max-width:100%;height:auto;border:1px solid #d8dee9}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #d8dee9;padding:6px;text-align:left}}th{{background:#e9eef5;position:sticky;top:0}}
pre{{white-space:pre-wrap;font-size:12px}}
</style></head><body>
<h1>Pose drift correction report</h1>
<p>Static-map registration diagnostics. Occupancy states are not changed by this report.</p>
<section class="metrics">
<div class="metric"><b>frames</b><br>{summary['frame_count']}</div>
<div class="metric"><b>keyframes</b><br>{summary['keyframe_count']}</div>
<div class="metric"><b>accepted map constraints</b><br>{summary['accepted_keyframes']}</div>
<div class="metric"><b>camera dt p95</b><br>{summary['camera_sync']['abs_p95_sec']:.4f} s</div>
</section>
<section class="panel"><h2>Trajectory on global map</h2><img src="{plots['trajectory_plot']}"></section>
<section class="panel"><h2>Structural residual</h2><img src="{plots['residual_plot']}"></section>
<section class="panel"><h2>Smoothed correction</h2><img src="{plots['correction_plot']}"></section>
<section class="panel"><h2>Keyframe audit</h2><table><thead><tr>
<th>frame</th><th>distance m</th><th>gate</th><th>before m</th><th>local ICP m</th><th>smoothed m</th><th>inlier</th><th>matched</th><th>raw dx/dy/yaw</th><th>smooth dx/dy/yaw</th><th>reason</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class="panel"><h2>Machine summary</h2><pre>{summary_json}</pre></section>
</body></html>"""
    path.write_text(content, encoding="utf-8")
