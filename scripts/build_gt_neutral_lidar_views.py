#!/usr/bin/env python3
"""Render prediction-blind per-slot LiDAR views for manual GT annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_CJK_FONT_PATH = Path(
    "/home/ParkingAgent/DASP/simulation_carla/CARLA_0.9.16/Engine/Content/Slate/Fonts/DroidSansFallback.ttf"
)
if _CJK_FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(_CJK_FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=str(_CJK_FONT_PATH)
    ).get_name()
plt.rcParams["axes.unicode_minus"] = False


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = _CJK_FONT_PATH if _CJK_FONT_PATH.is_file() else Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _sample(indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    return indices[np.linspace(0, len(indices) - 1, limit, dtype=np.int64)]


def _render(pack_path: Path, slot_id: str, output: Path) -> dict:
    with np.load(pack_path, allow_pickle=False) as archive:
        points = np.asarray(archive["points_local_xyzi"], dtype=np.float32)
        target = np.asarray(archive["polygon_local_m"], dtype=np.float32)
        core = np.asarray(archive["core_polygon_local_m"], dtype=np.float32)
        margin = np.asarray(archive["margin_polygon_local_m"], dtype=np.float32)
        selected = np.asarray(archive["selected_frames"], dtype=np.int64)
        valid = np.asarray(archive["valid_frames"], dtype=np.int64)
    extent = np.vstack((target, margin))
    x0, y0 = extent.min(axis=0) - 0.7
    x1, y1 = extent.max(axis=0) + 0.7
    local = (
        (points[:, 0] >= x0) & (points[:, 0] <= x1)
        & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    )
    floor = _sample(np.flatnonzero(local & (points[:, 2] <= 0.30)), 18_000)
    obstacle = _sample(np.flatnonzero(local & (points[:, 2] > 0.30)), 24_000)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.4), dpi=130)
    bev, side = axes
    if len(floor):
        bev.scatter(
            points[floor, 0], points[floor, 1], s=2, c="#9aa8b8",
            alpha=0.24, linewidths=0, rasterized=True,
        )
    if len(obstacle):
        bev.scatter(
            points[obstacle, 0], points[obstacle, 1], s=3,
            c=np.clip(points[obstacle, 2], 0.3, 2.5), cmap="turbo",
            vmin=0.3, vmax=2.5, alpha=0.68, linewidths=0, rasterized=True,
        )
        side.scatter(
            points[obstacle, 0], points[obstacle, 2], s=3,
            c=np.clip(points[obstacle, 2], 0.3, 2.5), cmap="turbo",
            vmin=0.3, vmax=2.5, alpha=0.68, linewidths=0, rasterized=True,
        )
    bev.add_patch(Polygon(margin, closed=True, fill=False, edgecolor="#d18b00", linestyle="--", linewidth=1.3))
    bev.add_patch(Polygon(target, closed=True, fill=False, edgecolor="#246bdb", linewidth=3))
    bev.add_patch(Polygon(core, closed=True, fill=False, edgecolor="#132238", linewidth=2))
    bev.set_xlim(x0, x1)
    bev.set_ylim(y0, y1)
    bev.set_aspect("equal")
    bev.grid(color="#dbe3ed", linewidth=0.6)
    bev.set_title("俯视：原始累积点与车位边界")
    bev.set_xlabel("车位局部 x (m)")
    bev.set_ylabel("车位局部 y (m)")
    bev.legend(
        handles=[
            Line2D([0], [0], color="#246bdb", lw=3, label="目标车位"),
            Line2D([0], [0], color="#132238", lw=2, label="核心区"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#f97316", markersize=7, label="离地>0.30m回波"),
        ],
        loc="lower center",
        frameon=False,
        ncol=3,
    )
    side.axhspan(-0.1, 0.3, color="#e8eef5")
    side.axhline(0.3, color="#64748b", linestyle="--", linewidth=1.2)
    side.set_xlim(x0, x1)
    side.set_ylim(-0.1, 2.8)
    side.grid(color="#dbe3ed", linewidth=0.6)
    side.set_title("侧视：离地高度结构（不拟合车辆框）")
    side.set_xlabel("车位局部 x (m)")
    side.set_ylabel("z (m)")
    figure.suptitle(
        f"{slot_id} 盲标LiDAR视图｜选择{len(selected)}帧，有效{len(valid)}帧｜不显示任何模型预测",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return {
        "slot_id": slot_id,
        "pack_path": str(pack_path),
        "image": str(output),
        "selected_frame_count": int(len(selected)),
        "valid_frame_count": int(len(valid)),
        "local_floor_point_count": int(np.sum(local & (points[:, 2] <= 0.30))),
        "local_above_ground_point_count": int(np.sum(local & (points[:, 2] > 0.30))),
    }


def _contact_sheets(rows: list[dict], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[str] = []
    per_sheet, columns = 6, 2
    for sheet_index, start in enumerate(range(0, len(rows), per_sheet), start=1):
        batch = rows[start:start + per_sheet]
        canvas = Image.new("RGB", (1800, 3 * 930 + 65), "#eef3f8")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (22, 14),
            "Prediction-blind LiDAR GT review — no model states or fitted boxes",
            font=_font(26, bold=True),
            fill="#132238",
        )
        for index, row in enumerate(batch):
            image = Image.open(row["image"]).convert("RGB")
            image.thumbnail((880, 850), Image.Resampling.LANCZOS)
            x = (index % columns) * 900 + 10
            y = (index // columns) * 930 + 65
            canvas.paste(image, (x, y))
            draw.text(
                (x + 12, y + 855),
                f"{row['slot_id']} | valid {row['valid_frame_count']}/{row['selected_frame_count']}",
                font=_font(20, bold=True),
                fill="#25364d",
            )
        path = output_dir / f"lidar_neutral_contact_sheet_{sheet_index:02d}.jpg"
        canvas.save(path, quality=90)
        result.append(str(path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--gt-pack-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.part1.read_text(encoding="utf-8"))
    output_dir = args.gt_pack_dir / "06_annotation" / "lidar_neutral_60"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    rows = []
    for case in payload["slot_cases"]:
        slot_id = str(case["slot"]["slot_id"])
        pack = Path(case["resources"]["extended_lidar_evidence_path"])
        rows.append(_render(pack, slot_id, output_dir / f"{slot_id}.png"))
    sheets = _contact_sheets(
        rows,
        args.gt_pack_dir / "05_contact_sheets" / "lidar_neutral",
    )
    manifest = {
        "schema_version": "parking-agent-gt-neutral-lidar/1.0",
        "prediction_blind": True,
        "case_count": len(rows),
        "cases": rows,
        "contact_sheets": sheets,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
