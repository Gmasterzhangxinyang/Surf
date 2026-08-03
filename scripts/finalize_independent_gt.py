#!/usr/bin/env python3
"""Finalize a conservative, prediction-independent three-state GT table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--accepted-labels", type=Path, required=True)
    parser.add_argument("--lidar-manifest", type=Path, required=True)
    parser.add_argument("--static-manifest", type=Path, required=True)
    parser.add_argument("--camera-manifest", type=Path, required=True)
    parser.add_argument("--legacy-recheck", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    lidar = json.loads(args.lidar_manifest.read_text(encoding="utf-8"))
    static = json.loads(args.static_manifest.read_text(encoding="utf-8"))
    camera = json.loads(args.camera_manifest.read_text(encoding="utf-8"))
    accepted_rows = _read_csv(args.accepted_labels)
    legacy_rows = _read_csv(args.legacy_recheck)

    ids = [str(row["slot_id"]) for row in universe["slots"]]
    accepted = {row["slot_id"]: row for row in accepted_rows}
    if len(accepted) != len(accepted_rows):
        raise ValueError("accepted labels contain duplicate slot IDs")
    unknown_ids = sorted(set(accepted) - set(ids))
    if unknown_ids:
        raise ValueError(f"accepted labels outside universe: {unknown_ids}")
    allowed_states = {"free", "occupied"}
    if any(row["gt_state"].lower() not in allowed_states for row in accepted_rows):
        raise ValueError("accepted labels may contain only terminal free/occupied states")

    lidar_by_id = {row["slot_id"]: row for row in lidar["cases"]}
    static_by_id = {row["slot_id"]: row for row in static["rows"]}
    camera_by_id = {row["slot_id"]: row for row in camera["rows"]}
    legacy_by_id = {row["slot_id"]: row for row in legacy_rows}
    output = []
    for item in universe["slots"]:
        slot_id = str(item["slot_id"])
        lidar_row = lidar_by_id[slot_id]
        static_row = static_by_id[slot_id]
        camera_row = camera_by_id[slot_id]
        accepted_row = accepted.get(slot_id)
        legacy_row = legacy_by_id.get(slot_id, {})
        if accepted_row:
            state = accepted_row["gt_state"].lower()
            observability = accepted_row["gt_observability"]
            identity = "yes"
            confidence = accepted_row["state_confidence"]
            adjudication = "accepted_terminal"
            annotator = accepted_row["annotator"]
            evidence_frames = accepted_row["evidence_frames"]
            basis = accepted_row["evidence_basis"]
            notes = accepted_row["notes"]
        else:
            state = "unknown"
            view_count = int(camera_row["view_count"])
            valid_count = int(lidar_row["valid_frame_count"])
            if valid_count == 0 and view_count == 0:
                observability = "unobserved"
            elif view_count > 0:
                observability = "camera_visible_but_terminal_unresolved"
            else:
                observability = "lidar_partial_or_structurally_ambiguous"
            identity = "not_terminally_verified"
            confidence = ""
            adjudication = "excluded_from_terminal_metrics"
            annotator = "independent_sensor_review_v1"
            evidence_frames = ""
            basis = "prediction_blind_evidence_insufficient_for_terminal_state"
            if legacy_row.get("legacy_gt_state_unverified"):
                notes = (
                    "legacy label retained only for audit and deliberately not copied; "
                    f"legacy={legacy_row['legacy_gt_state_unverified']}"
                )
            else:
                notes = "no terminal claim"
        output.append(
            {
                "slot_id": slot_id,
                "gt_state": state,
                "gt_observability": observability,
                "identity_verified": identity,
                "state_confidence": confidence,
                "adjudication_status": adjudication,
                "annotator": annotator,
                "evidence_basis": basis,
                "evidence_frames": evidence_frames,
                "min_route_distance_m": f"{float(item['min_route_distance_m']):.6f}",
                "lidar_valid_frames": int(lidar_row["valid_frame_count"]),
                "camera_review_views": int(camera_row["view_count"]),
                "static_explained_ratio": f"{float(static_row['static_explained_ratio']):.6f}",
                "unexplained_short_extent_m": (
                    f"{float(static_row['unexplained_short_extent_m']):.6f}"
                ),
                "legacy_gt_state_unverified": legacy_row.get(
                    "legacy_gt_state_unverified", ""
                ),
                "notes": notes,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_csv, output)
    counts = {
        state: sum(row["gt_state"] == state for row in output)
        for state in ("free", "occupied", "unknown")
    }
    inputs = [
        args.universe,
        args.accepted_labels,
        args.lidar_manifest,
        args.static_manifest,
        args.camera_manifest,
        args.legacy_recheck,
    ]
    manifest = {
        "schema_version": "parking-slot-independent-adjudicated-gt/1.0",
        "selection_is_prediction_independent": True,
        "part1_state_used_for_labels": False,
        "legacy_labels_automatically_copied": False,
        "formal_gt": False,
        "formal_gt_blocker": (
            "single independent adjudication pass only; a blinded second annotator "
            "and conflict arbitration are still required for a publication-grade formal GT"
        ),
        "evaluation_ready_high_confidence_subset": True,
        "terminal_metric_policy": (
            "use only adjudication_status=accepted_terminal and identity_verified=yes"
        ),
        "slot_count": len(output),
        "state_counts": counts,
        "terminal_label_count": counts["free"] + counts["occupied"],
        "occupied_metric_ready": counts["occupied"] > 0,
        "source_sha256": {str(path): _sha256(path) for path in inputs},
        "output_csv": str(args.output_csv),
        "output_csv_sha256": _sha256(args.output_csv),
    }
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
