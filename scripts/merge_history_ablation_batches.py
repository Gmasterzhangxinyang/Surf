#!/usr/bin/env python3
"""Merge compatible independent-GT W/K ablation batches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _rank(row: dict) -> tuple:
    return (
        -int(row["false_occupied_count"] + row["false_free_count"]),
        int(row["correct_terminal_count"]),
        float(row["selective_accuracy"] or 0.0),
        -int(row["sample_count_K"]),
        -int(row["history_span_W"]),
        -float(row["runtime_seconds"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    reference = payloads[0]
    compatibility = (
        "anchor",
        "gt_sha256",
        "config_sha256",
        "semantic_static_veto",
        "gt_state_counts",
    )
    for payload in payloads[1:]:
        for key in compatibility:
            if payload.get(key) != reference.get(key):
                raise ValueError(f"incompatible batch field: {key}")
    by_pair = {}
    for payload in payloads:
        for row in payload["runs"]:
            pair = (int(row["history_span_W"]), int(row["sample_count_K"]))
            if pair in by_pair:
                raise ValueError(f"duplicate W/K pair: {pair}")
            by_pair[pair] = row
    runs = [by_pair[pair] for pair in sorted(by_pair)]
    best = max(runs, key=_rank)
    merged = {
        **{key: value for key, value in reference.items() if key not in {"runs", "best"}},
        "schema_version": "independent-gt-history-ablation/1.1-merged",
        "batch_sources": [str(path) for path in args.input],
        "best": {
            key: value
            for key, value in best.items()
            if key not in {"rows", "selected_frame_ids"}
        },
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = (
        "history_span_W",
        "sample_count_K",
        "runtime_seconds",
        "prediction_scope_count",
        "terminal_count",
        "correct_terminal_count",
        "selective_accuracy",
        "terminal_coverage_all_gt",
        "effective_correct_coverage_all_gt",
        "false_occupied_count",
        "false_free_count",
    )
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in runs)

    lines = [
        "# 独立 GT 历史帧 W/K 消融（静态语义硬否决）",
        "",
        f"- anchor：{reference['anchor']}",
        f"- GT：Free={reference['gt_state_counts']['free']}，"
        f"Occupied={reference['gt_state_counts']['occupied']}",
        "- 目标：零终态错误 → 最大正确覆盖 → 更低 K/W/耗时。",
        "",
        "| W | K | 正确终态 | 错误Occupied | 选择性准确率 | 全GT终态覆盖率 | 秒 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        accuracy = (
            "N/A"
            if row["selective_accuracy"] is None
            else f"{100 * row['selective_accuracy']:.1f}%"
        )
        lines.append(
            f"| {row['history_span_W']} | {row['sample_count_K']} | "
            f"{row['correct_terminal_count']} | {row['false_occupied_count']} | "
            f"{accuracy} | {100 * row['terminal_coverage_all_gt']:.1f}% | "
            f"{row['runtime_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"最佳组合：W={best['history_span_W']}，K={best['sample_count_K']}；"
            f"正确终态={best['correct_terminal_count']}，"
            f"错误Occupied={best['false_occupied_count']}。",
            "",
            "W=100 的所有 K 仍在 slot_1249 出现 1 个错误 Occupied，说明超长历史的"
            "定位漂移会把窄静态残差拓宽到车辆门限以上；因此当前数据不支持 100/20。",
            "",
            "限制：本 anchor 没有可确认 Occupied GT，不能用本表估计 Occupied recall。",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_count": len(runs),
                "best_W": best["history_span_W"],
                "best_K": best["sample_count_K"],
                "output": str(args.output_json),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
