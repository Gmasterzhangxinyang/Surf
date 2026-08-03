#!/usr/bin/env python3
"""Fail-closed audit for the Nature-style report and its source experiments."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.contracts import Part1Output
from parking_slot_hybrid_3d.io import write_json_atomic
from parking_slot_part2.media import load_lidar_evidence_pack


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    paper_root = PROJECT_ROOT / "outputs/parking_slot_agent_v2_nature_paper"
    report_root = paper_root / "Nature风格中文研究稿"
    frame_root = PROJECT_ROOT / "outputs/parking_slot_agent_v2_frame_9277"
    errors: list[str] = []
    checks: list[str] = []

    window = _load(paper_root / "window_ablation_frame_9277.json")
    window_counts = [int(row["resolved_count"]) for row in sorted(window["experiments"], key=lambda row: int(row["window_frames"]))]
    if window_counts != [0, 6, 8, 9, 9]:
        errors.append(f"unexpected window ablation counts: {window_counts}")
    else:
        checks.append("window ablation is exactly 0/6/8/9/9 for 15/30/45/60/75 frames")

    baseline = _load(paper_root / "cross_anchor_baseline15_systematic6.json")
    extended = _load(paper_root / "cross_anchor_extended60_systematic6.json")
    base_by_anchor = {int(row["anchor_frame"]): row for row in baseline["experiments"]}
    ext_by_anchor = {int(row["anchor_frame"]): row for row in extended["experiments"]}
    if set(base_by_anchor) != set(ext_by_anchor):
        errors.append("cross-anchor baseline and extended anchor sets differ")
    for anchor in sorted(set(base_by_anchor) & set(ext_by_anchor)):
        base_rows = {row["slot_id"] for row in base_by_anchor[anchor]["rows"]}
        ext_rows = {row["slot_id"] for row in ext_by_anchor[anchor]["rows"]}
        if base_rows != ext_rows:
            errors.append(f"anchor {anchor} candidate identity mismatch")
        if int(base_by_anchor[anchor]["resolved_count"]) != 0:
            errors.append(f"anchor {anchor} baseline unexpectedly resolved a case")
    pooled_total = sum(int(row["part1_unknown_count"]) for row in ext_by_anchor.values())
    pooled_resolved = sum(int(row["resolved_count"]) for row in ext_by_anchor.values())
    if (pooled_resolved, pooled_total) != (34, 208):
        errors.append(f"unexpected cross-anchor pooled result: {pooled_resolved}/{pooled_total}")
    else:
        checks.append("matched systematic six-anchor comparison is 0/208 versus 34/208")

    result = _load(frame_root / "openai_extended60_exhaustive_v1/part2_result.json")
    unknown_rows = [row for row in result["slot_results"] if row["case"]["part1_state"] == "unknown"]
    states = Counter(row["case"]["final_state"] for row in unknown_rows)
    if states != Counter({"unknown": 13, "occupied": 7, "free": 2}):
        errors.append(f"unexpected OpenAI terminal states: {dict(states)}")
    for row in unknown_rows:
        case = row["case"]
        if row["validation_errors"]:
            errors.append(f"{case['slot']['slot_id']} has validation errors")
        lidar = next(item for item in case["evidence"] if item["tool_name"] == "lidar_detail")
        gate = lidar["metadata"]["geometry_card"]["terminal_geometry_gate"]
        state = case["final_state"]
        if state in {"free", "occupied"} and not gate[f"{state}_eligible"]:
            errors.append(f"{case['slot']['slot_id']} terminal state bypassed geometry gate")
        if state == "unknown" and (gate["free_eligible"] or gate["occupied_eligible"]):
            errors.append(f"{case['slot']['slot_id']} stayed Unknown despite a terminal gate")
    if not errors:
        checks.append("all 22 OpenAI final states agree with deterministic hard-gate eligibility")

    part1 = Part1Output.from_dict(_load(frame_root / "part1_optimized_v3_extended60/part1_output.json"))
    for case in part1.slot_cases:
        pack = load_lidar_evidence_pack(case.resources["extended_lidar_evidence_path"], expected_slot_id=case.slot_id)
        frames = [int(value) for value in pack.selected_frames]
        if len(frames) != 60 or max(frames) > part1.scene.anchor_frame_id:
            errors.append(f"{case.slot_id} extended pack violates 60-frame causal contract")
    if len(part1.slot_cases) == 24 and not any("causal contract" in error for error in errors):
        checks.append("24/24 extended packs contain exactly 60 frames and no post-t0 frame")

    html_path = report_root / "manuscript.html"
    html_text = html_path.read_text(encoding="utf-8")
    refs = re.findall(r"(?:src|href)='([^']+)'", html_text)
    missing = [ref for ref in refs if not ref.startswith(("http:", "https:", "#")) and not (report_root / ref).is_file()]
    if missing:
        errors.append("missing HTML references: " + ", ".join(missing))
    else:
        checks.append(f"all {len(refs)} local HTML references exist")
    pdf_path = report_root / "manuscript.pdf"
    if not pdf_path.is_file() or pdf_path.stat().st_size < 100_000 or not pdf_path.read_bytes().startswith(b"%PDF-"):
        errors.append("PDF is missing or invalid")
    else:
        checks.append(f"PDF exists and is non-trivial ({pdf_path.stat().st_size} bytes)")

    audit = {
        "schema_version": "parking-slot-agent-v2-nature-report-audit/1.0",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "checks": checks,
    }
    write_json_atomic(report_root / "validation_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
