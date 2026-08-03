#!/usr/bin/env python3
"""Fail-closed validation for the complete Agent HTML report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.report_dir.resolve()
    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    data_path = root / "complete_results.json"
    html_path = root / "index.html"
    report_path = root / "REPORT.md"
    check("required_files", all(path.is_file() for path in (data_path, html_path, report_path)), "json/html/markdown")
    if not data_path.is_file() or not html_path.is_file():
        payload = {"ok": False, "checks": checks}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    data = json.loads(data_path.read_text(encoding="utf-8"))
    overview = data.get("batch", {}).get("overview", {})
    active = data.get("active_cases", [])
    rows = overview.get("all_cases", [])
    check("schema", data.get("schema_version") == "parking-agent-complete-report/1.0", str(data.get("schema_version")))
    check("batch_scope", overview.get("total_cases") == 24 and len(rows) == 24, f"declared={overview.get('total_cases')}, rows={len(rows)}")
    check("batch_metrics", overview.get("unknown_inputs") == 22 and overview.get("resolved_unknown") == 9, f"unknown={overview.get('unknown_inputs')}, resolved={overview.get('resolved_unknown')}")
    check("batch_unique_slots", len({row.get("slot_id") for row in rows}) == len(rows), "slot ids unique")
    check("active_scope", len(active) == 4, f"active_cases={len(active)}")
    check("active_turns", all(len(case.get("turns", [])) == 4 for case in active), "all active cases have 3 tool calls + final")
    check("active_verification", all(case.get("final_localization", {}).get("stage") in {"supported", "refuted", "ambiguous"} for case in active), "all final stages are verification outcomes")

    referenced: list[Path] = []
    for case in active:
        referenced.extend(root / value for value in case.get("report_paths", {}).values())
        referenced.extend(root / value for value in case.get("assets", {}).values())
    for case in data.get("batch", {}).get("cases", []):
        referenced.append(root / str(case.get("decision_board", "")))
    missing = [str(path) for path in referenced if not path.is_file()]
    check("referenced_assets", not missing, f"references={len(referenced)}, missing={len(missing)}")

    page = html_path.read_text(encoding="utf-8")
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.DOTALL)
    embedded_ok = False
    if match:
        embedded = json.loads(match.group(1))
        embedded_ok = (
            embedded.get("batch", {}).get("overview", {}).get("total_cases") == 24
            and len(embedded.get("active_cases", [])) == 4
        )
    check("embedded_json", embedded_ok, "HTML payload parses with 24 batch + 4 active cases")
    check("interactive_controls", all(token in page for token in ('id="activeTabs"', 'id="search"', 'id="stateFilter"', 'id="resultBody"')), "tabs/search/filters/table")
    forbidden = "sk-proj-"
    leaked = any(forbidden in path.read_text(encoding="utf-8", errors="ignore") for path in (data_path, html_path, report_path))
    check("secret_absent", not leaked, "no API-key prefix in report text")

    result = {"schema_version": "parking-agent-complete-report-validation/1.0", "ok": all(item["ok"] for item in checks), "checks": checks}
    audit_path = root / "validation_audit.json"
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
