#!/usr/bin/env python3
"""Retrospectively audit terminal Occupied decisions with strict safety gates.

This is a counterfactual rejection audit over saved features, not a replacement
for rerunning the complete point-cloud pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _items(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        result: list[dict[str, Any]] = []
        for item_path in sorted(path.glob("*.json")):
            payload = json.loads(item_path.read_text(encoding="utf-8"))
            item = payload.get("decision", payload)
            if "slot_id" in item:
                result.append(item)
        return result
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("decisions", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    thresholds = {
        "robust_short_extent_m": float(
            config["occupied_min_robust_short_extent_m"]
        ),
        "low_bev_coverage": float(config["occupied_min_low_bev_coverage"]),
        "upper_height_spread_ratio": float(
            config["occupied_min_upper_height_spread_ratio"]
        ),
        "outside_residual_ratio": float(
            config["occupied_max_outside_residual"]
        ),
        "stability_pass_ratio": float(config["stability_min_pass_ratio"]),
    }

    rows: list[dict[str, Any]] = []
    for decision in _items(args.decisions):
        if str(decision.get("state", "")).lower() != "occupied":
            continue
        evidence = decision.get("occupied_evidence") or {}
        features = evidence.get("features") or {}
        stability = decision.get("stability") or {}
        short_extent = min(
            float(features.get("robust_extent_x_m", 0.0)),
            float(features.get("robust_extent_y_m", 0.0)),
        )
        height_span = max(float(features.get("height_span_m", 0.0)), 1e-9)
        upper_spread = max(
            float(features.get("z95_m", 0.0))
            - float(features.get("z75_m", 0.0)),
            0.0,
        ) / height_span
        low_coverage = float(features.get("low_bev_coverage", 0.0))
        outside = float(features.get("outside_residual_ratio", 1.0))
        pass_ratio = float(stability.get("pass_ratio", 0.0))
        failures: list[str] = []
        if short_extent < thresholds["robust_short_extent_m"]:
            failures.append("insufficient_vehicle_footprint")
        if low_coverage < thresholds["low_bev_coverage"]:
            failures.append("insufficient_lower_body_coverage")
        if upper_spread < thresholds["upper_height_spread_ratio"]:
            failures.append("horizontal_cap_structure")
        if outside > thresholds["outside_residual_ratio"]:
            failures.append("outside_residual_conflict")
        if pass_ratio < thresholds["stability_pass_ratio"]:
            failures.append("pose_unstable")
        rows.append(
            {
                "slot_id": decision["slot_id"],
                "historical_state": "occupied",
                "robust_short_extent_m": short_extent,
                "low_bev_coverage": low_coverage,
                "upper_height_spread_ratio": upper_spread,
                "outside_residual_ratio": outside,
                "stability_pass_ratio": pass_ratio,
                "strict_added_gates_pass": not failures,
                "strict_rejection_reasons": ";".join(failures),
            }
        )

    payload = {
        "schema_version": "high-precision-occupied-counterfactual-audit/1.0",
        "audit_semantics": (
            "saved-feature rejection audit; a failed added gate proves the old "
            "Occupied decision cannot survive v4, but this is not a full rerun"
        ),
        "decisions": str(args.decisions),
        "config": str(args.config),
        "thresholds": thresholds,
        "historical_occupied_count": len(rows),
        "surviving_count": sum(row["strict_added_gates_pass"] for row in rows),
        "rejected_count": sum(not row["strict_added_gates_pass"] for row in rows),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 历史 Occupied 高精度门回放审计",
        "",
        "这是保存特征上的反事实否决审计，不冒充完整点云重跑。",
        "",
        f"- 历史 Occupied：{len(rows)}",
        f"- 被 v4 新增硬门明确拒绝：{payload['rejected_count']}",
        f"- 仍可能保留：{payload['surviving_count']}",
        "",
        "| slot | 短边(m) | 低层覆盖 | 上层展宽比 | 框外残差 | 稳定率 | v4新增门 | 原因 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {slot_id} | {robust_short_extent_m:.3f} | "
            "{low_bev_coverage:.3f} | {upper_height_spread_ratio:.3f} | "
            "{outside_residual_ratio:.3f} | {stability_pass_ratio:.3f} | "
            "{status} | {reasons} |".format(
                status="PASS" if row["strict_added_gates_pass"] else "REJECT",
                reasons=row["strict_rejection_reasons"] or "—",
                **row,
            )
        )
    lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
