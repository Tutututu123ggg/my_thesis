#!/usr/bin/env python
from __future__ import annotations

"""Summarize one or more extraction-audit summary CSV files into a simple gate report.

Works with:
- targeted_extraction_summary.csv from run_targeted_extraction_audit4.py
- unbiased20_extraction_summary.csv from run_unbiased_extraction_audit20.py

It does not call LLM and does not touch Neo4j/Qdrant.
"""

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PASS_LIKE = {"PASS", "PASS_REVIEW"}
WARN_LIKE = {"WARN_REVIEW", "WARN_RECALL"}
BAD_LIKE = {"FAIL", "FAIL_NO_RELATIONS", "WARN_EVIDENCE", "WARN_ENDPOINT", "ERROR"}


@dataclass
class AuditRow:
    source_csv: str
    split: str
    status: str
    article: str
    relations: int
    useful_relations: int
    evidence_issues: int
    endpoint_issues: int
    keyword_issues: int
    low_relation_windows: int
    orphan_measures: int
    orphan_anchors: int
    raw: dict[str, Any]


def as_int(value: Any) -> int:
    try:
        text = str(value if value is not None else "").strip()
        if text == "":
            return 0
        return int(float(text))
    except Exception:
        return 0


def normalize_row(row: dict[str, str], *, source_csv: str, split: str) -> AuditRow:
    status = (row.get("status") or "").strip() or "UNKNOWN"
    article = (
        row.get("article_title")
        or row.get("article")
        or row.get("target")
        or row.get("group_label")
        or ""
    )
    relations = as_int(row.get("relation_count") or row.get("relations"))
    useful = as_int(row.get("useful_relation_count") or row.get("useful_relations"))
    evidence = as_int(row.get("suspicious_evidence_count") or row.get("suspicious_evidence") or row.get("evidence_issues"))
    endpoint = as_int(row.get("endpoint_issue_count") or row.get("endpoint_issues"))
    keyword = as_int(row.get("keyword_issue_count") or row.get("keyword_issues"))
    low_windows = as_int(row.get("low_relation_window_count") or row.get("low_relation_windows"))
    orphan_measures = as_int(row.get("orphan_measure_count") or row.get("orphan_measures"))
    orphan_anchors = as_int(row.get("orphan_anchor_count") or row.get("orphan_anchors"))
    return AuditRow(
        source_csv=source_csv,
        split=split,
        status=status,
        article=article,
        relations=relations,
        useful_relations=useful,
        evidence_issues=evidence,
        endpoint_issues=endpoint,
        keyword_issues=keyword,
        low_relation_windows=low_windows,
        orphan_measures=orphan_measures,
        orphan_anchors=orphan_anchors,
        raw=row,
    )


def read_rows(path: Path, split: str) -> list[AuditRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [normalize_row(row, source_csv=str(path), split=split) for row in reader]


def pct(num: float, den: float) -> str:
    if den <= 0:
        return "0.0%"
    return f"{num / den:.1%}"


def short(text: str, limit: int = 100) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def classify(status: str) -> str:
    if status in PASS_LIKE:
        return "pass"
    if status in WARN_LIKE:
        return "warn"
    if status in BAD_LIKE:
        return "bad"
    if status.startswith("PASS"):
        return "pass"
    if status.startswith("WARN"):
        return "warn"
    if status.startswith("FAIL") or status == "ERROR":
        return "bad"
    return "warn"


def compute_gate(rows: list[AuditRow]) -> dict[str, Any]:
    total = len(rows)
    pass_count = sum(1 for r in rows if classify(r.status) == "pass")
    bad_count = sum(1 for r in rows if classify(r.status) == "bad")
    warn_count = total - pass_count - bad_count
    relations = sum(r.relations for r in rows)
    useful = sum(r.useful_relations for r in rows)
    evidence = sum(r.evidence_issues for r in rows)
    endpoint = sum(r.endpoint_issues for r in rows)
    keyword = sum(r.keyword_issues for r in rows)
    low_windows = sum(r.low_relation_windows for r in rows)
    orphan_measures = sum(r.orphan_measures for r in rows)

    # Conservative gate for deciding whether to run bulk extraction-only.
    pass_ok = pass_count >= max(1, int(0.75 * total))
    evidence_ok = evidence == 0 or evidence <= max(1, int(0.02 * max(1, relations)))
    endpoint_ok = endpoint <= max(2, int(0.05 * max(1, relations)))
    useful_ok = useful / max(1, relations) >= 0.70
    low_window_ok = low_windows <= max(2, int(0.15 * max(1, total)))
    bad_ok = bad_count == 0
    gate = pass_ok and evidence_ok and endpoint_ok and useful_ok and low_window_ok and bad_ok

    return {
        "total_articles": total,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "bad_count": bad_count,
        "relation_count": relations,
        "useful_relation_count": useful,
        "useful_ratio": useful / max(1, relations),
        "evidence_issues": evidence,
        "endpoint_issues": endpoint,
        "keyword_issues": keyword,
        "low_relation_windows": low_windows,
        "orphan_measures": orphan_measures,
        "checks": {
            "pass_rate_ok": pass_ok,
            "no_bad_status_ok": bad_ok,
            "evidence_ok": evidence_ok,
            "endpoint_ok": endpoint_ok,
            "useful_ratio_ok": useful_ok,
            "low_relation_windows_ok": low_window_ok,
        },
        "bulk_extraction_gate": "PASS" if gate else "REVIEW_BEFORE_BULK",
    }


def write_csv(path: Path, rows: list[AuditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "status", "article", "relations", "useful_relations",
        "evidence_issues", "endpoint_issues", "keyword_issues",
        "low_relation_windows", "orphan_measures", "orphan_anchors", "source_csv",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "split": r.split,
                "status": r.status,
                "article": r.article,
                "relations": r.relations,
                "useful_relations": r.useful_relations,
                "evidence_issues": r.evidence_issues,
                "endpoint_issues": r.endpoint_issues,
                "keyword_issues": r.keyword_issues,
                "low_relation_windows": r.low_relation_windows,
                "orphan_measures": r.orphan_measures,
                "orphan_anchors": r.orphan_anchors,
                "source_csv": r.source_csv,
            })


def write_md(path: Path, rows: list[AuditRow], gate: dict[str, Any]) -> None:
    by_split = Counter(r.split for r in rows)
    by_status = Counter(r.status for r in rows)
    lines: list[str] = []
    lines.append("# Combined extraction audit gate")
    lines.append("")
    lines.append("Báo cáo này chỉ tổng hợp CSV audit đã có. Không gọi LLM, không ghi Neo4j/Qdrant.")
    lines.append("")
    lines.append("## Gate result")
    lines.append("")
    lines.append(f"- **Bulk extraction gate:** `{gate['bulk_extraction_gate']}`")
    lines.append(f"- Articles: {gate['total_articles']}")
    lines.append(f"- PASS-like / WARN-like / BAD-like: {gate['pass_count']} / {gate['warn_count']} / {gate['bad_count']}")
    lines.append(f"- Relations: {gate['relation_count']}")
    lines.append(f"- Useful relations: {gate['useful_relation_count']} ({gate['useful_ratio']:.1%})")
    lines.append(f"- Evidence issues: {gate['evidence_issues']}")
    lines.append(f"- Endpoint issues: {gate['endpoint_issues']}")
    lines.append(f"- Keyword issues: {gate['keyword_issues']}")
    lines.append(f"- Low-relation windows: {gate['low_relation_windows']}")
    lines.append(f"- Orphan measures: {gate['orphan_measures']}")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    for name, ok in gate["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if ok else 'REVIEW'}")
    lines.append("")
    lines.append("## Split distribution")
    lines.append("")
    for split, count in by_split.most_common():
        lines.append(f"- `{split}`: {count}")
    lines.append("")
    lines.append("## Status distribution")
    lines.append("")
    for status, count in by_status.most_common():
        lines.append(f"- `{status}`: {count}")
    lines.append("")
    lines.append("## Article rows")
    lines.append("")
    lines.append("| split | status | article | relations | useful | evidence | endpoint | keyword | low_windows | orphan_measures |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.split} | `{r.status}` | {short(r.article).replace('|', '\\|')} | {r.relations} | {r.useful_relations} | "
            f"{r.evidence_issues} | {r.endpoint_issues} | {r.keyword_issues} | {r.low_relation_windows} | {r.orphan_measures} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_csv_arg(value: str) -> tuple[Path, str]:
    # format: split_name=path or just path
    if "=" in value:
        split, path = value.split("=", 1)
        return Path(path), split.strip() or Path(path).stem
    p = Path(value)
    return p, p.stem


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize extraction audit summary CSVs into a combined gate report.")
    ap.add_argument("--summary-csv", action="append", required=True, help="Either split=path.csv or path.csv. Repeatable.")
    ap.add_argument("--out-dir", default="data/debug/extraction_audit_gate")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[AuditRow] = []
    for item in args.summary_csv:
        path, split = parse_csv_arg(item)
        if not path.exists():
            raise SystemExit(f"Summary CSV not found: {path}")
        rows.extend(read_rows(path, split=split))
    if not rows:
        raise SystemExit("No rows loaded.")

    gate = compute_gate(rows)
    write_csv(out_dir / "combined_extraction_gate.csv", rows)
    write_md(out_dir / "combined_extraction_gate.md", rows, gate)
    (out_dir / "combined_extraction_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE]")
    print(f"gate: {gate['bulk_extraction_gate']}")
    print(f"md  : {out_dir / 'combined_extraction_gate.md'}")
    print(f"csv : {out_dir / 'combined_extraction_gate.csv'}")
    print(f"json: {out_dir / 'combined_extraction_gate.json'}")


if __name__ == "__main__":
    main()
