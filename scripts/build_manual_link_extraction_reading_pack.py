#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a small manual reading pack comparing article source windows with extracted
entities / relations / keywords.

Read-only: no Neo4j, no Qdrant, no DB write.

Typical usage:
  python scripts/build_manual_link_extraction_reading_pack.py `
    --raw-jsonl data/debug/unbiased20_current_full_audit/raw_and_validated_chunk_extractions.jsonl `
    --summary-csv data/debug/unbiased20_current_full_audit/unbiased20_extraction_summary.csv `
    --entities-csv data/debug/unbiased20_current_full_audit/unbiased20_entities.csv `
    --relations-csv data/debug/unbiased20_current_full_audit/unbiased20_relations.csv `
    --out-dir data/debug/manual_link_reading_pack_3 `
    --count 3 `
    --seed 20260625
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

SUCCESS_STATUSES = {"PASS", "PASS_REVIEW", "WARN_REVIEW", "WARN_RECALL"}
MEASURE_TYPES = {"CAN_THIEP_Y_TE", "THUOC_VA_HOAT_CHAT", "PHUONG_PHAP_CHAN_DOAN", "YEU_TO_NGOAI_SINH"}
ANCHOR_TYPES = {"BENH_LY", "BIEU_HIEN_LAM_SANG", "YEU_TO_BENH_SINH", "YEU_TO_NGOAI_SINH", "THUOC_VA_HOAT_CHAT", "CAN_THIEP_Y_TE"}


def read_csv(path: Optional[Path]) -> List[Dict[str, str]]:
    if not path:
        return []
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def norm_url(url: str) -> str:
    return (url or "").strip().rstrip("/")


def article_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (norm_url(str(row.get("url", ""))), str(row.get("article_title") or row.get("title") or "").strip())


def safe_slug(s: str, max_len: int = 80) -> str:
    s = (s or "article").strip().lower()
    # keep Vietnamese letters, ascii letters, digits; convert separators to dash
    s = re.sub(r"[\\/:*?\"<>|]+", " ", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return (s[:max_len].strip("-") or "article")


def md_escape(s: Any) -> str:
    return str(s or "").replace("\r\n", "\n").replace("\r", "\n")


def truncate(s: str, n: int = 220) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def split_keywords(s: str) -> List[str]:
    if not s:
        return []
    raw = s.strip()
    # common formats: JSON list, semicolon-separated, pipe-separated
    if raw.startswith("[") and raw.endswith("]"):
        try:
            arr = json.loads(raw)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    parts = re.split(r"\s*[;|]\s*", raw)
    if len(parts) == 1:
        # Do not aggressively split comma because a keyword may contain comma in source, but flag later.
        return [raw]
    return [p for p in parts if p]


def text_contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    return term.lower() in (text or "").lower()


def extract_candidate_lines(text: str) -> List[str]:
    """Heuristic list-like / fact-like lines for manual recall review."""
    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^([-*•]|\d+[.)]|[a-zA-Z][.)])\s+", line):
            lines.append(line)
        elif re.match(r"^#{1,6}\s+", line):
            lines.append(line)
        elif any(marker in line.lower() for marker in ["triệu chứng", "nguyên nhân", "điều trị", "phòng ngừa", "chẩn đoán", "biến chứng", "cần tránh", "lưu ý"]):
            lines.append(line)
    # de-dup while preserving order
    seen = set()
    out = []
    for line in lines:
        key = re.sub(r"\s+", " ", line.lower())
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out


def extract_bold_terms(text: str) -> List[str]:
    terms: List[str] = []
    for pat in [r"\*\*([^*]{2,80})\*\*", r"__([^_]{2,80})__", r"\(([^()]{2,80})\)"]:
        for m in re.finditer(pat, text or ""):
            term = re.sub(r"\s+", " ", m.group(1)).strip(" .,:;!?")
            if 2 <= len(term) <= 80:
                terms.append(term)
    seen = set(); out=[]
    for t in terms:
        k=t.lower()
        if k not in seen:
            seen.add(k); out.append(t)
    return out


def group_by_article(rows: Iterable[Dict[str, Any]]) -> DefaultDict[Tuple[str, str], List[Dict[str, Any]]]:
    d: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d[article_key(r)].append(r)
    return d


def choose_articles(summary_rows: List[Dict[str, str]], raw_rows: List[Dict[str, Any]], count: int, seed: int, include_statuses: Sequence[str], prefer_warn: bool) -> List[Tuple[str, str]]:
    allowed = set(include_statuses) if include_statuses else SUCCESS_STATUSES
    keys_with_raw = set(group_by_article(raw_rows).keys())
    candidates: List[Dict[str, str]] = []
    for r in summary_rows:
        key = article_key(r)
        status = (r.get("status") or "").strip()
        if key in keys_with_raw and status in allowed and int(float(r.get("relation_count") or 0)) > 0:
            candidates.append(r)
    if not candidates:
        # fallback to raw keys if no summary was provided
        keys = sorted(keys_with_raw, key=lambda x: x[1])
        random.Random(seed).shuffle(keys)
        return keys[:count]

    rng = random.Random(seed)
    if prefer_warn:
        warn = [r for r in candidates if (r.get("status") or "").startswith("WARN")]
        ok = [r for r in candidates if not (r.get("status") or "").startswith("WARN")]
        rng.shuffle(warn); rng.shuffle(ok)
        picked = warn[:1] + ok[: max(0, count - min(1, len(warn)))]
        if len(picked) < count:
            rest = [r for r in candidates if r not in picked]
            rng.shuffle(rest)
            picked += rest[: count - len(picked)]
    else:
        picked = candidates[:]
        rng.shuffle(picked)
        picked = picked[:count]
    return [article_key(r) for r in picked]


def load_article_full_text(md_path: str, project_root: Optional[Path]) -> Optional[str]:
    if not md_path:
        return None
    p = Path(md_path)
    candidates = []
    if p.exists():
        candidates.append(p)
    if project_root:
        # Try path as-is relative to root
        candidates.append(project_root / md_path)
        # Windows absolute path often includes project root; keep suffix after data/raw if present
        m = re.search(r"(data[\\/].+)$", md_path)
        if m:
            candidates.append(project_root / m.group(1).replace("\\", "/"))
    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return c.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
    return None


def make_window_table(raws: List[Dict[str, Any]]) -> str:
    lines = ["| # | chunk_id | section | text chars | entities | relations |", "|---:|---|---|---:|---:|---:|"]
    for i, r in enumerate(raws, 1):
        ents = r.get("validated_entities") or (r.get("raw_graph") or {}).get("entities") or []
        rels = r.get("validated_relations") or (r.get("raw_graph") or {}).get("relations") or []
        lines.append(
            f"| {i} | `{md_escape(r.get('chunk_id'))}` | {truncate(md_escape(r.get('section')), 90)} | {len(md_escape(r.get('text')))} | {len(ents)} | {len(rels)} |"
        )
    return "\n".join(lines)


def markdown_for_article(
    key: Tuple[str, str],
    raw_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]],
    ent_by_key: Dict[Tuple[str, str], List[Dict[str, str]]],
    rel_by_key: Dict[Tuple[str, str], List[Dict[str, str]]],
    summary_by_key: Dict[Tuple[str, str], Dict[str, str]],
    project_root: Optional[Path],
) -> str:
    raws = sorted(raw_by_key.get(key, []), key=lambda r: (int(r.get("window_index") or 0), str(r.get("chunk_id") or "")))
    ents = ent_by_key.get(key, [])
    rels = rel_by_key.get(key, [])
    summary = summary_by_key.get(key, {})
    first = raws[0] if raws else {}
    title = summary.get("article_title") or first.get("title") or key[1]
    url = summary.get("url") or first.get("url") or key[0]
    md_path = summary.get("md_path") or first.get("md_path") or ""
    status = summary.get("status", "")

    md: List[str] = []
    md.append(f"# Manual link-vs-extraction reading pack: {title}\n")
    md.append("## 1. Article metadata\n")
    md.append(f"- **Status:** `{status}`")
    md.append(f"- **Group label:** `{summary.get('group_label') or first.get('group_label') or ''}`")
    md.append(f"- **URL:** {url}")
    md.append(f"- **Markdown path:** `{md_path}`")
    md.append(f"- **Windows:** {len(raws)}")
    md.append(f"- **Entities:** {summary.get('entity_count') or len(ents)}")
    md.append(f"- **Relations:** {summary.get('relation_count') or len(rels)}")
    md.append(f"- **Useful relations:** {summary.get('useful_relation_count', '')}")
    md.append(f"- **Evidence issues:** {summary.get('suspicious_evidence_count') or summary.get('evidence_issues') or ''}")
    md.append(f"- **Endpoint issues:** {summary.get('endpoint_issue_count') or summary.get('endpoint_issues') or ''}")
    md.append(f"- **Keyword issues:** {summary.get('keyword_issue_count') or summary.get('keyword_issues') or ''}")
    md.append("")

    md.append("## 2. Manual verdict checklist\n")
    md.append("Điền tay sau khi đọc source ở mục 5 và đối chiếu mục 6–8.\n")
    md.append("| Criterion | Verdict | Notes |")
    md.append("|---|---|---|")
    md.append("| Entity recall for main disease/symptom/treatment/factor | ☐ High ☐ Medium ☐ Low | |")
    md.append("| Relation recall for symptoms/factors/treatments | ☐ High ☐ Medium ☐ Low | |")
    md.append("| Relation recall for care/prevention/bullet lists | ☐ High ☐ Medium ☐ Low | |")
    md.append("| Precision / hallucination risk | ☐ Clean ☐ Minor issue ☐ Serious issue | |")
    md.append("| Keyword usefulness for retrieval | ☐ Useful ☐ Mixed ☐ Poor | |")
    md.append("| Final decision | ☐ PASS_INGEST ☐ PASS_WITH_NOTES ☐ RETRY/PATCH | |")
    md.append("")

    md.append("## 3. Extraction summary from audit\n")
    rel_type_counts = Counter(r.get("relation_type") for r in rels if r.get("relation_type"))
    ent_type_counts = Counter(e.get("entity_type") for e in ents if e.get("entity_type"))
    md.append("### Entity type counts\n")
    for k, v in ent_type_counts.most_common():
        md.append(f"- `{k}`: {v}")
    md.append("\n### Relation type counts\n")
    for k, v in rel_type_counts.most_common():
        md.append(f"- `{k}`: {v}")
    if summary.get("orphan_measures"):
        md.append("\n### Orphan measures from summary\n")
        md.append(summary.get("orphan_measures", ""))
    if summary.get("orphan_anchors"):
        md.append("\n### Orphan anchors from summary\n")
        md.append(summary.get("orphan_anchors", ""))
    md.append("")

    md.append("## 4. Window overview\n")
    md.append(make_window_table(raws))
    md.append("")

    md.append("## 5. Source text by extraction window\n")
    for i, r in enumerate(raws, 1):
        md.append(f"\n### Window {i}: `{md_escape(r.get('chunk_id'))}`")
        md.append(f"- **Section:** {md_escape(r.get('section'))}")
        md.append(f"- **Text chars:** {len(md_escape(r.get('text')))}")
        md.append("\n```markdown")
        md.append(md_escape(r.get("text")))
        md.append("```")

    full_text = load_article_full_text(md_path, project_root)
    if full_text and all((r.get("text") or "") not in full_text for r in raws[:1]):
        # Only include full article if source windows do not appear to be enough / are not contiguous.
        md.append("\n## 5b. Full markdown source found on disk\n")
        md.append("```markdown")
        md.append(full_text[:30000] + ("\n\n...[TRUNCATED BY SCRIPT]..." if len(full_text) > 30000 else ""))
        md.append("```")

    md.append("\n## 6. Extracted entities grouped by type\n")
    by_type: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    for e in ents:
        by_type[e.get("entity_type", "UNKNOWN")].append(e)
    for etype, group in sorted(by_type.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        md.append(f"\n### `{etype}` ({len(group)})")
        for e in sorted(group, key=lambda x: x.get("name", "").lower()):
            evidence = truncate(e.get("evidence_text", ""), 180)
            anchor = " anchor" if str(e.get("is_anchor_candidate", "")).lower() in {"true", "1", "yes"} else ""
            md.append(f"- **{e.get('name','')}**{anchor} — evidence: {evidence}")

    md.append("\n## 7. Extracted relations grouped by type\n")
    rel_by_type: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rels:
        rel_by_type[r.get("relation_type", "UNKNOWN")].append(r)
    for rtype, group in sorted(rel_by_type.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        md.append(f"\n### `{rtype}` ({len(group)})")
        for r in group:
            subj = r.get("subject", "")
            obj = r.get("object", "")
            ev = truncate(r.get("evidence_text", ""), 220)
            kw = r.get("keywords", "")
            support = r.get("evidence_supported_by_chunk", "")
            endpoint = r.get("endpoint_covered_by_chunk", "")
            md.append(f"- `{subj} [{r.get('subject_type','')}] --{rtype}--> {obj} [{r.get('object_type','')}]`")
            md.append(f"  - evidence: {ev}")
            md.append(f"  - keywords: {kw}")
            if support or endpoint:
                md.append(f"  - checks: evidence_supported={support}; endpoint_covered={endpoint}")

    md.append("\n## 8. Recall helper: source lines that deserve manual checking\n")
    md.append("Các dòng dưới đây là heading/bullet/fact-like lines. Script đánh dấu `covered_by_endpoint_or_evidence` nếu có entity/relation endpoint/evidence chứa một phần nội dung. Dòng `NO_CLEAR_HIT` cần đọc tay kỹ vì có thể là missed relation.\n")
    all_endpoint_terms = []
    for e in ents:
        all_endpoint_terms.append(e.get("name", ""))
    for r in rels:
        all_endpoint_terms.extend([r.get("subject", ""), r.get("object", ""), r.get("evidence_text", "")])
    for i, r in enumerate(raws, 1):
        lines = extract_candidate_lines(md_escape(r.get("text")))
        if not lines:
            continue
        md.append(f"\n### Candidate lines — window {i}")
        for line in lines[:80]:
            compact = truncate(line, 260)
            low = compact.lower()
            hit_terms = [t for t in all_endpoint_terms if t and len(t) >= 4 and (t.lower() in low or low in t.lower())]
            status_line = "covered_by_endpoint_or_evidence" if hit_terms else "NO_CLEAR_HIT"
            hits = "; ".join(sorted(set(hit_terms), key=len)[:5])
            md.append(f"- **{status_line}** — {compact}" + (f"  _(hits: {hits})_" if hits else ""))

    md.append("\n## 9. Recall helper: bold/parenthesized terms not obviously extracted\n")
    extracted_names = {e.get("name", "").lower() for e in ents}
    for i, r in enumerate(raws, 1):
        terms = extract_bold_terms(md_escape(r.get("text")))
        missing_like = [t for t in terms if t.lower() not in extracted_names and not any(t.lower() in n or n in t.lower() for n in extracted_names if n)]
        if missing_like:
            md.append(f"\n### Window {i}")
            for t in missing_like[:50]:
                md.append(f"- `{t}`")

    md.append("\n## 10. Keyword quality helper\n")
    bad_kw_rows = []
    for r in rels:
        kws = split_keywords(r.get("keywords", ""))
        for kw in kws:
            word_count = len(re.findall(r"\w+", kw, flags=re.UNICODE))
            has_comma = "," in kw
            if word_count > 6 or has_comma or len(kw) > 70:
                bad_kw_rows.append((r, kw, word_count, has_comma))
    if not bad_kw_rows:
        md.append("Không phát hiện keyword quá dài / nhét nhiều keyword bằng dấu phẩy theo heuristic.")
    else:
        md.append("Các keyword nên soi lại:")
        for r, kw, wc, comma in bad_kw_rows[:100]:
            md.append(f"- `{kw}` — words={wc}, comma={comma}; relation `{r.get('subject')} --{r.get('relation_type')}--> {r.get('object')}`")

    md.append("\n## 11. Manual notes\n")
    md.append("- Missing entities:")
    md.append("- Missing relations:")
    md.append("- Wrong/weak relation types:")
    md.append("- Evidence problems:")
    md.append("- Keyword problems:")
    md.append("- Final verdict:")
    md.append("")
    return "\n".join(md)


def write_summary_csv(path: Path, selected_keys: Sequence[Tuple[str, str]], summary_by_key: Dict[Tuple[str, str], Dict[str, str]], ent_by_key, rel_by_key) -> None:
    fieldnames = [
        "article_title", "url", "status", "entity_count", "relation_count", "useful_relation_count",
        "evidence_issues", "endpoint_issues", "keyword_issues", "orphan_measure_count", "orphan_anchor_count", "report_file",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for idx, key in enumerate(selected_keys, 1):
            s = summary_by_key.get(key, {})
            title = s.get("article_title") or key[1]
            report_file = f"{idx:02d}_{safe_slug(title)}.md"
            w.writerow({
                "article_title": title,
                "url": s.get("url") or key[0],
                "status": s.get("status", ""),
                "entity_count": s.get("entity_count") or len(ent_by_key.get(key, [])),
                "relation_count": s.get("relation_count") or len(rel_by_key.get(key, [])),
                "useful_relation_count": s.get("useful_relation_count", ""),
                "evidence_issues": s.get("suspicious_evidence_count") or s.get("evidence_issues", ""),
                "endpoint_issues": s.get("endpoint_issue_count") or s.get("endpoint_issues", ""),
                "keyword_issues": s.get("keyword_issue_count") or s.get("keyword_issues", ""),
                "orphan_measure_count": s.get("orphan_measure_count", ""),
                "orphan_anchor_count": s.get("orphan_anchor_count", ""),
                "report_file": report_file,
            })


def build_index_page(path: Path, selected_keys: Sequence[Tuple[str, str]], summary_by_key, out_files: Sequence[Path], seed: int) -> None:
    lines = ["# Manual link-vs-extraction reading pack index", ""]
    lines.append(f"- Seed: `{seed}`")
    lines.append(f"- Article count: `{len(selected_keys)}`")
    lines.append("")
    lines.append("## Selected articles")
    for key, f in zip(selected_keys, out_files):
        s = summary_by_key.get(key, {})
        lines.append(f"- [{s.get('article_title') or key[1]}]({f.name}) — `{s.get('status','')}`, entities={s.get('entity_count','')}, relations={s.get('relation_count','')}, useful={s.get('useful_relation_count','')}")
    lines.append("")
    lines.append("## How to audit")
    lines.append("1. Đọc mục 5 source text trước, không nhìn extraction quá sớm.")
    lines.append("2. Ghi ra các fact y khoa chính: bệnh, triệu chứng, nguyên nhân/yếu tố, điều trị, phòng ngừa, chẩn đoán, biến chứng.")
    lines.append("3. Đối chiếu mục 6 entity và mục 7 relation.")
    lines.append("4. Dùng mục 8–10 để soi các dòng/bold term/keyword nghi ngờ bị miss hoặc bẩn.")
    lines.append("5. Điền checklist mục 2 và notes mục 11.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build manual article-vs-extraction reading packs.")
    ap.add_argument("--raw-jsonl", required=True, type=Path, help="raw_and_validated_chunk_extractions.jsonl")
    ap.add_argument("--summary-csv", type=Path, default=None, help="extraction summary CSV")
    ap.add_argument("--entities-csv", type=Path, default=None, help="entities CSV")
    ap.add_argument("--relations-csv", type=Path, default=None, help="relations CSV")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260625)
    ap.add_argument("--include-status", default="PASS_REVIEW,WARN_RECALL,WARN_REVIEW,PASS", help="Comma-separated statuses to sample from")
    ap.add_argument("--prefer-warn", action="store_true", help="Include one WARN_* article first if available, then random PASS articles")
    ap.add_argument("--project-root", type=Path, default=Path("."), help="Repo root, used to locate full markdown source when possible")
    args = ap.parse_args(argv)

    raw_rows = read_jsonl(args.raw_jsonl)
    summary_rows = read_csv(args.summary_csv)
    entity_rows = read_csv(args.entities_csv)
    relation_rows = read_csv(args.relations_csv)

    raw_by_key = group_by_article(raw_rows)
    summary_by_key = {article_key(r): r for r in summary_rows}
    ent_by_key = group_by_article(entity_rows)
    rel_by_key = group_by_article(relation_rows)

    statuses = [s.strip() for s in args.include_status.split(",") if s.strip()]
    selected = choose_articles(summary_rows, raw_rows, args.count, args.seed, statuses, args.prefer_warn)
    if not selected:
        print("No selected articles. Check input paths/status filters.", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_files: List[Path] = []
    for idx, key in enumerate(selected, 1):
        title = (summary_by_key.get(key, {}) or {}).get("article_title") or key[1]
        out_path = args.out_dir / f"{idx:02d}_{safe_slug(title)}.md"
        out_path.write_text(
            markdown_for_article(key, raw_by_key, ent_by_key, rel_by_key, summary_by_key, args.project_root),
            encoding="utf-8",
        )
        out_files.append(out_path)

    write_summary_csv(args.out_dir / "manual_reading_pack_summary.csv", selected, summary_by_key, ent_by_key, rel_by_key)
    build_index_page(args.out_dir / "README_MANUAL_READING_PACK.md", selected, summary_by_key, out_files, args.seed)

    print(f"Wrote {len(out_files)} article reading packs to: {args.out_dir}")
    for p in out_files:
        print(f"- {p}")
    print(f"- {args.out_dir / 'manual_reading_pack_summary.csv'}")
    print(f"- {args.out_dir / 'README_MANUAL_READING_PACK.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
