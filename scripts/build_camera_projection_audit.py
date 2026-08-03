#!/usr/bin/env python3
"""Build a strict, hash-bound camera audit from real pixel references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_hybrid_3d.projection_audit import (
    ProjectionAuditThresholds,
    build_projection_audit,
    write_projection_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-total-references", type=int, default=15)
    parser.add_argument("--minimum-references-per-zone", type=int, default=5)
    parser.add_argument("--maximum-rmse-px", type=float, default=3.0)
    parser.add_argument("--maximum-error-px", type=float, default=8.0)
    parser.add_argument(
        "--maximum-camera-lidar-delta-sec",
        type=float,
        default=0.04,
    )
    parser.add_argument("--default-image-width-px", type=int, default=1280)
    parser.add_argument("--default-image-height-px", type=int, default=720)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        thresholds = ProjectionAuditThresholds(
            minimum_total_references=args.minimum_total_references,
            minimum_references_per_zone=args.minimum_references_per_zone,
            maximum_rmse_px=args.maximum_rmse_px,
            maximum_error_px=args.maximum_error_px,
            maximum_camera_lidar_delta_sec=args.maximum_camera_lidar_delta_sec,
        )
        audit = build_projection_audit(
            args.calibration,
            args.references,
            dataset_id=args.dataset_id,
            thresholds=thresholds,
            default_image_size_px=(
                args.default_image_width_px,
                args.default_image_height_px,
            ),
        )
        output_path = write_projection_audit(args.output, audit)
    except (OSError, ValueError) as exc:
        print(f"camera projection audit error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output": str(output_path),
                "image_count": audit["image_count"],
                "image_registry_sha256": audit["image_registry_sha256"],
                "point_source_count": audit["point_source_count"],
                "point_source_registry_sha256": audit[
                    "point_source_registry_sha256"
                ],
                "reference_set_uri": audit["reference_set_uri"],
                "status": audit["status"],
                "failures": audit["failures"],
                "metrics": audit["metrics"],
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    # A generated-but-untrusted audit is still persisted for diagnosis, while
    # CI and shell pipelines get a failing gate.
    return 0 if audit["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
