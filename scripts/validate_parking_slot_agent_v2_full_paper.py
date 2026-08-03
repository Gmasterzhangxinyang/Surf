#!/usr/bin/env python3
"""Fail-closed validation for the detailed bilingual paper package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.reporting_full_paper import REFERENCES
from parking_slot_hybrid_3d.io import write_json_atomic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    report = PROJECT_ROOT / "outputs/parking_slot_agent_v2_nature_paper/CVPR完整论文包"
    errors: list[str] = []
    checks: list[str] = []

    required = [
        "paper_zh_detailed.md", "paper_zh_detailed.html", "paper_zh_detailed.pdf",
        "paper_en_cvpr_draft.md", "paper_en_cvpr_draft.tex", "supplementary_zh.md", "supplementary_en.md",
        "references.bib", "references_verified.json", "paper_manifest.json", "README.md",
        "visual_results_report.html",
    ]
    missing = [name for name in required if not (report / name).is_file()]
    if missing:
        errors.append("missing required files: " + ", ".join(missing))
    else:
        checks.append(f"all {len(required)} required paper files exist")

    zh = (report / "paper_zh_detailed.md").read_text(encoding="utf-8")
    en = (report / "paper_en_cvpr_draft.md").read_text(encoding="utf-8")
    required_zh = ["摘要", "相关工作", "问题定义", "Occupied 几何门", "Free 几何门", "实验设置", "局限性与威胁", "安全、伦理与隐私", "参考文献"]
    absent_zh = [item for item in required_zh if item not in zh]
    if absent_zh or len(zh) < 18_000:
        errors.append(f"Chinese manuscript incomplete: length={len(zh)}, absent={absent_zh}")
    else:
        checks.append(f"detailed Chinese manuscript has {len(zh)} characters and all required sections")
    required_en = ["Abstract", "Related Work", "Problem Formulation", "Experimental Protocol", "Discussion and Limitations", "References"]
    absent_en = [item for item in required_en if item not in en]
    if absent_en or len(en) < 8_000:
        errors.append(f"English manuscript incomplete: length={len(en)}, absent={absent_en}")
    else:
        checks.append(f"English CVPR draft has {len(en)} characters and all required sections")
    supp_en = (report / "supplementary_en.md").read_text(encoding="utf-8")
    if len(supp_en) < 5_000 or "Complete matched cross-anchor results" not in supp_en or "Statistical interpretation" not in supp_en:
        errors.append(f"English supplementary material incomplete: length={len(supp_en)}")
    else:
        checks.append(f"English supplementary material is complete ({len(supp_en)} characters)")

    factual_tokens = ["0/22", "9/22", "0/208", "34/208", "2 Free", "7 Occupied", "13 Unknown", "22.0%"]
    for token in factual_tokens:
        if token not in zh:
            errors.append(f"Chinese manuscript missing audited claim: {token}")
    if not any(error.startswith("Chinese manuscript missing audited claim") for error in errors):
        checks.append("all primary audited numerical claims appear in the manuscript")

    refs = json.loads((report / "references_verified.json").read_text(encoding="utf-8"))
    keys = [row["key"] for row in refs["references"]]
    urls = [row["url"] for row in refs["references"]]
    bib = (report / "references.bib").read_text(encoding="utf-8")
    if refs.get("count") != len(REFERENCES) or len(set(keys)) != len(keys) or len(set(urls)) != len(urls):
        errors.append("reference registry count/uniqueness mismatch")
    absent_bib = [key for key in keys if "{" + key + "," not in bib]
    if absent_bib:
        errors.append("BibTeX missing keys: " + ", ".join(absent_bib))
    if not any("reference" in error.lower() or "bibtex" in error.lower() for error in errors):
        checks.append(f"{len(keys)} unique verified references are present in JSON and BibTeX")

    html_text = (report / "paper_zh_detailed.html").read_text(encoding="utf-8")
    image_refs = re.findall(r"src='([^']+)'", html_text)
    missing_images = [path for path in image_refs if not (report / path).is_file()]
    if len(image_refs) != 7 or missing_images:
        errors.append(f"HTML figure set invalid: count={len(image_refs)}, missing={missing_images}")
    else:
        checks.append("HTML embeds all 7 redesigned local main/extended figures")

    visual_text = (report / "visual_results_report.html").read_text(encoding="utf-8")
    visual_refs = re.findall(r"src='([^']+)'", visual_text)
    missing_visual = [path for path in visual_refs if not (report / path).is_file()]
    if len(visual_refs) != 7 or missing_visual:
        errors.append(f"visual report invalid: count={len(visual_refs)}, missing={missing_visual}")
    else:
        checks.append("visual-first report embeds the complete redesigned figure suite")

    threshold_table = report / "tables/Table_4_hard_gate_thresholds.csv"
    threshold_rows = threshold_table.read_text(encoding="utf-8").strip().splitlines()
    if len(threshold_rows) != 21:
        errors.append(f"hard-gate table expected 20 rows, got {len(threshold_rows)-1}")
    else:
        checks.append("hard-gate table contains all 20 auditable criteria")

    tex = (report / "paper_en_cvpr_draft.tex").read_text(encoding="utf-8")
    if "\\usepackage[review]{cvpr}" not in tex or "Anonymous CVPR submission" not in tex:
        errors.append("CVPR LaTeX source is not anonymized review-mode source")
    else:
        checks.append("CVPR LaTeX source uses anonymous review mode")

    pdf = report / "paper_zh_detailed.pdf"
    if not pdf.read_bytes().startswith(b"%PDF-") or pdf.stat().st_size < 1_000_000:
        errors.append("detailed PDF missing, invalid, or implausibly small")
    else:
        checks.append(f"detailed PDF is valid and non-trivial ({pdf.stat().st_size} bytes)")

    manifest = json.loads((report / "paper_manifest.json").read_text(encoding="utf-8"))
    for row in manifest.get("files", []):
        path = Path(row["path"])
        if not path.is_file() or _sha256(path) != row["sha256"]:
            errors.append(f"manifest hash mismatch: {path}")
    if not any("manifest hash" in error for error in errors):
        checks.append(f"all {len(manifest.get('files', []))} generated-file hashes match")

    audit = {
        "schema_version": "parking-slot-agent-v2-full-paper-audit/1.0",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "checks": checks,
    }
    write_json_atomic(report / "paper_validation_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
