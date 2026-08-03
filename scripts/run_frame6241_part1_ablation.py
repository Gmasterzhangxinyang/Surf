#!/usr/bin/env python3
"""Run a locked-GT W/K history ablation for frame 6241 Part1."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_frame6241_locked_gt import evaluate, merged_system, part1_predictions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--slot-db", type=Path, required=True)
    p.add_argument("--map-points-dir", type=Path, required=True)
    p.add_argument("--gltf", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with a.gt.open(newline="", encoding="utf-8-sig") as f:
        gt = {row["slot_id"]: row["gt_state"].lower() for row in csv.DictReader(f)}

    combinations = [
        (w, k)
        for w in (15, 30, 45, 60)
        for k in (3, 5, 10, 15)
        if k <= w
    ]
    results = []
    for span, count in combinations:
        output = a.output_dir / f"w{span}_k{count}"
        decisions = output / "slot_decisions.json"
        if not decisions.exists():
            command = [
                sys.executable,
                str(ROOT / "scripts/run_hybrid_3d_slot_evidence.py"),
                "--frames-csv",
                str(a.frames),
                "--slot-db",
                str(a.slot_db),
                "--map-points-dir",
                str(a.map_points_dir),
                "--output-dir",
                str(output),
                "--phase",
                "full",
                "--anchor-frame",
                "6241",
                "--history-span",
                str(span),
                "--history-sample-count",
                str(count),
                "--gltf-static-map",
                str(a.gltf),
                "--overwrite",
            ]
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False
            )
            (a.output_dir / f"w{span}_k{count}.stdout.log").write_text(
                completed.stdout, encoding="utf-8"
            )
            (a.output_dir / f"w{span}_k{count}.stderr.log").write_text(
                completed.stderr, encoding="utf-8"
            )
            if completed.returncode:
                results.append(
                    {
                        "history_span": span,
                        "history_sample_count": count,
                        "status": "failed",
                        "returncode": completed.returncode,
                    }
                )
                continue
        prediction = merged_system(list(gt), part1_predictions(decisions))
        metrics = evaluate(gt, prediction)
        results.append(
            {
                "history_span": span,
                "history_sample_count": count,
                "status": "completed",
                "three_class_accuracy": metrics["three_class_accuracy"],
                "terminal_coverage": metrics["terminal_coverage"],
                "terminal_selective_accuracy": metrics[
                    "terminal_selective_accuracy"
                ],
                "determinate_gt_accuracy_unknown_counted_wrong": metrics[
                    "determinate_gt_accuracy_unknown_counted_wrong"
                ],
                "unknown_false_resolution_rate": metrics[
                    "unknown_false_resolution_rate"
                ],
                "prediction_counts": metrics["prediction_counts"],
                "output_dir": str(output),
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    completed_rows = [row for row in results if row["status"] == "completed"]
    ranked = sorted(
        completed_rows,
        key=lambda row: (
            row["terminal_selective_accuracy"]
            if row["terminal_selective_accuracy"] is not None
            else -1.0,
            -row["unknown_false_resolution_rate"],
            row["three_class_accuracy"],
            row["terminal_coverage"],
        ),
        reverse=True,
    )
    payload = {
        "schema_version": "frame6241-part1-history-ablation/1.0",
        "anchor_frame": 6241,
        "gt_path": str(a.gt),
        "exploratory_single_anchor": True,
        "selection_policy": (
            "rank terminal selective accuracy, then minimize Unknown false resolution, "
            "then maximize three-class accuracy and coverage"
        ),
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
        "all": results,
    }
    (a.output_dir / "ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "history_span",
        "history_sample_count",
        "status",
        "three_class_accuracy",
        "terminal_coverage",
        "terminal_selective_accuracy",
        "determinate_gt_accuracy_unknown_counted_wrong",
        "unknown_false_resolution_rate",
        "prediction_counts",
        "output_dir",
    ]
    with (a.output_dir / "ablation_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            item = dict(row)
            if "prediction_counts" in item:
                item["prediction_counts"] = json.dumps(
                    item["prediction_counts"], ensure_ascii=False, sort_keys=True
                )
            writer.writerow(item)
    print(json.dumps({"best": payload["best"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
