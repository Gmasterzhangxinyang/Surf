#!/usr/bin/env python3
"""Copy an inclusive Camera frame window and build review contact sheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont


def _timestamps(path: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected frame timestamp")
            result[int(fields[0])] = float(fields[1])
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(
            "/home/ParkingAgent/DASP/simulation_carla/CARLA_0.9.16/"
            "Engine/Content/Slate/Fonts/DroidSansFallback.ttf"
        ),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _contact_sheets(rows: list[dict], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns, rows_per_sheet = 5, 4
    cell_width, cell_height = 330, 225
    per_sheet = columns * rows_per_sheet
    result: list[str] = []
    for sheet_index, start in enumerate(range(0, len(rows), per_sheet), start=1):
        batch = rows[start:start + per_sheet]
        canvas = Image.new(
            "RGB",
            (columns * cell_width, 58 + rows_per_sheet * cell_height),
            "#edf2f7",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (18, 13),
            (
                f"Camera window | {batch[0]['camera_frame']:06d}–"
                f"{batch[-1]['camera_frame']:06d} | "
                f"{batch[0]['offset_sec']:+.2f}s–{batch[-1]['offset_sec']:+.2f}s"
            ),
            font=_font(24, bold=True),
            fill="#132238",
        )
        for index, row in enumerate(batch):
            image = Image.open(row["absolute_path"]).convert("RGB")
            image.thumbnail((320, 180), Image.Resampling.LANCZOS)
            x = (index % columns) * cell_width + 5
            y = (index // columns) * cell_height + 60
            canvas.paste(image, (x, y))
            color = "#d12f38" if row["offset_frame"] == 0 else "#25364d"
            draw.text(
                (x + 3, y + 184),
                f"left{row['camera_frame']:06d}  {row['offset_sec']:+.2f}s",
                font=_font(17, bold=row["offset_frame"] == 0),
                fill=color,
            )
        destination = output_dir / f"camera_contact_sheet_{sheet_index:02d}.jpg"
        canvas.save(destination, quality=90)
        result.append(str(destination))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--before", type=int, default=150)
    parser.add_argument("--after", type=int, default=150)
    parser.add_argument(
        "--exclude-anchor",
        action="store_true",
        help="exclude the anchor from the main image count and copy it separately",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.before < 0 or args.after < 0:
        raise ValueError("before/after must be non-negative")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    timestamps = _timestamps(args.timestamps)
    if args.anchor_frame not in timestamps:
        raise ValueError("anchor Camera frame is missing from timestamps")
    first = args.anchor_frame - args.before
    last = args.anchor_frame + args.after
    frames = [
        frame
        for frame in range(first, last + 1)
        if not (args.exclude_anchor and frame == args.anchor_frame)
    ]
    missing_timestamps = [frame for frame in frames if frame not in timestamps]
    missing_images = [
        frame for frame in frames
        if not (args.image_dir / f"left{frame:06d}.png").is_file()
    ]
    if missing_timestamps or missing_images:
        raise FileNotFoundError(
            f"window incomplete: timestamp_missing={missing_timestamps[:5]}, "
            f"image_missing={missing_images[:5]}"
        )

    root = args.output_dir
    image_output = root / "images"
    image_output.mkdir(parents=True)
    anchor_timestamp = timestamps[args.anchor_frame]
    rows: list[dict] = []
    for frame in frames:
        source = args.image_dir / f"left{frame:06d}.png"
        destination = image_output / source.name
        shutil.copy2(source, destination)
        offset = frame - args.anchor_frame
        rows.append(
            {
                "camera_frame": frame,
                "camera_timestamp": timestamps[frame],
                "offset_frame": offset,
                "offset_sec": timestamps[frame] - anchor_timestamp,
                "position": "anchor" if offset == 0 else "before" if offset < 0 else "after",
                "image_file": str(destination.relative_to(root)),
                "absolute_path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    if args.exclude_anchor:
        anchor_source = args.image_dir / f"left{args.anchor_frame:06d}.png"
        anchor_output = root / "anchor_reference"
        anchor_output.mkdir(parents=True)
        shutil.copy2(anchor_source, anchor_output / anchor_source.name)
    sheets = _contact_sheets(rows, root / "contact_sheets")

    with (root / "frame_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = [
            "camera_frame", "camera_timestamp", "offset_frame", "offset_sec",
            "position", "image_file", "bytes", "sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in fields}
            for row in rows
        )
    manifest = {
        "schema_version": "parking-agent-camera-window/1.0",
        "anchor_camera_frame": args.anchor_frame,
        "anchor_camera_timestamp": anchor_timestamp,
        "before_frame_count": args.before,
        "after_frame_count": args.after,
        "anchor_excluded_from_main_image_count": bool(args.exclude_anchor),
        "anchor_reference_file": (
            f"anchor_reference/left{args.anchor_frame:06d}.png"
            if args.exclude_anchor else f"images/left{args.anchor_frame:06d}.png"
        ),
        "first_camera_frame": first,
        "last_camera_frame": last,
        "image_count": len(rows),
        "first_offset_sec": rows[0]["offset_sec"],
        "last_offset_sec": rows[-1]["offset_sec"],
        "all_source_images_present": True,
        "images_are_original_resolution_copies": True,
        "contact_sheets": [
            str(Path(value).relative_to(root)) for value in sheets
        ],
        "total_image_bytes": sum(row["bytes"] for row in rows),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        f"""# Camera前后帧提取包

- 锚点：`left{args.anchor_frame:06d}.png`
- 范围：`left{first:06d}.png` 至 `left{last:06d}.png`
- 主图片数量：`{len(rows)}` 张
- 计数方式：{
            f"前{args.before} + 后{args.after}；锚点另存，不计入主图片数量"
            if args.exclude_anchor
            else f"前{args.before} + 锚点1 + 后{args.after}"
        }
- 时间范围：相对锚点 `{rows[0]['offset_sec']:+.3f}s` 至
  `{rows[-1]['offset_sec']:+.3f}s`

`images/` 是原始分辨率Camera副本；`contact_sheets/` 只用于快速浏览；
`frame_manifest.csv` 保存帧号、时间戳、相对时间和SHA-256。
此文件夹不包含LiDAR、模型预测或自动标签。
""",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
