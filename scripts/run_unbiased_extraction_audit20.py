#!/usr/bin/env python
from __future__ import annotations

"""
Run a read-only extraction audit on 20 dermatology articles that are intentionally
kept away from prompt few-shot / previously targeted topics.

Why this script exists:
- The earlier targeted audit used articles that later became prompt examples.
- This script chooses 4 different articles and reports extraction quality without
  hard-coding expected endpoint terms from the prompt.

Safety:
- No Neo4j writes.
- No Qdrant writes.
- No ingest/build step.
- No --fresh/delete-output behavior.
- Existing files in --out-dir may be overwritten, but the directory is not deleted.

Outputs:
- unbiased20_extraction_audit.md: readable audit + manual review sections
- unbiased20_extraction_summary.csv: one row per selected article
- unbiased20_relations.csv: flattened validated relations
- unbiased20_entities.csv: flattened validated entities
- unbiased20_chunks.csv: one row per extraction window/chunk
- raw_and_validated_chunk_extractions.jsonl: raw + validated dumps
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from app.ingestion.extraction.entity.medical_extractor import MedicalExtractor
from app.ingestion.extraction.keyword.keyword_profiler import ArticleKeywordProfiler
from app.ingestion.extraction.pipelines.article_extraction_pipeline import (
    ArticleExtractionPipeline,
    ArticleExtractionResult,
)
from app.ingestion.extraction.relation.extraction_validator import ExtractionValidator
from app.ingestion.processing.chunker import chunk_article
from app.ingestion.processing.ingestion_models import ArticleDocument, ChunkDocument
from app.ingestion.processing.markdown_loader import load_markdown_article, parse_frontmatter


PROMPT_LEAK_BLACKLIST = (
    # Previous targeted audit topics.
    "viêm da dị ứng",
    "kien ba khoang",
    "kiến ba khoang",
    "mụn cóc",
    "mun coc",
    "trẻ sơ sinh",
    "tre so sinh",
    "mẩn đỏ ở cổ",
    "man do o co",
    # Explicit examples that appeared in prompt/few-shot discussion.
    "chườm lạnh",
    "chườm mát",
    "tắm nước mát",
    "nha đam",
    "lô hội",
    "human papilloma",
    "virus hpv",
    "hpv",
    "vảy nến thể giọt",
    "tiểu đường còn gọi là",
    "đái tháo đường",
    # Keyword-profiler examples that may still be present in some local versions.
    "lang ben",
    "lang-ben",
    "nấm lang ben",
    "nấm da đầu",
    "nấm móng",
    "itraconazole",
    "đèn wood",
    "den wood",
)

# These are not expected-answer checks. They are just search groups for choosing
# diverse articles. If a group is not found, the script falls back to another group.
DEFAULT_QUERY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mụn trứng cá", ("mụn trứng cá", "mun trung ca", "mụn acne", "acne")),
    ("viêm da tiết bã", ("viêm da tiết bã", "viem da tiet ba", "seborrheic")),
    ("mề đay", ("mề đay", "me day", "mày đay", "may day", "urticaria")),
    ("zona thần kinh", ("zona", "giời leo", "herpes zoster")),
    ("ghẻ", ("bệnh ghẻ", "cái ghẻ", "ghẻ ngứa")),
    ("rôm sảy", ("rôm sảy", "rom say", "rôm ở trẻ", "phát ban nhiệt")),
    ("chàm sữa", ("chàm sữa", "cham sua")),
    ("mụn nhọt", ("mụn nhọt", "mun nhot", "nhọt da")),
)

MEASURE_ENTITY_TYPES = {
    "THUOC_VA_HOAT_CHAT",
    "CAN_THIEP_Y_TE",
    "PHUONG_PHAP_CHAN_DOAN",
    "YEU_TO_NGOAI_SINH",
}
ANCHOR_IMPORTANT_TYPES = {
    "BENH_LY",
    "THUOC_VA_HOAT_CHAT",
    "CAN_THIEP_Y_TE",
    "PHUONG_PHAP_CHAN_DOAN",
    "YEU_TO_BENH_SINH",
    "YEU_TO_NGOAI_SINH",
}
USEFUL_RELATION_TYPES = {
    "DIEU_TRI_BANG",
    "PHONG_NGUA_BANG",
    "CHAN_DOAN_BANG",
    "CAN_TRANH",
    "CO_BIEU_HIEN",
    "LIEN_QUAN_YEU_TO",
    "GAY_BIEN_CHUNG",
}


@dataclass(frozen=True)
class QueryGroup:
    label: str
    aliases: tuple[str, ...]


@dataclass
class CandidateArticle:
    path: Path
    title: str = ""
    url: str = ""
    source: str = ""
    row: dict[str, str] = field(default_factory=dict)


@dataclass
class ArticleAudit:
    group_label: str
    candidate: CandidateArticle
    article: ArticleDocument | None = None
    result: ArticleExtractionResult | None = None
    error: str = ""
    entity_count: int = 0
    relation_count: int = 0
    useful_relation_count: int = 0
    suspicious_evidence_count: int = 0
    endpoint_issue_count: int = 0
    keyword_issue_count: int = 0
    empty_window_count: int = 0
    low_relation_window_count: int = 0
    orphan_measure_count: int = 0
    orphan_anchor_count: int = 0
    relation_type_counts: Counter[str] = field(default_factory=Counter)
    entity_type_counts: Counter[str] = field(default_factory=Counter)
    orphan_measure_names: list[str] = field(default_factory=list)
    orphan_anchor_names: list[str] = field(default_factory=list)
    evidence_issue_examples: list[str] = field(default_factory=list)
    endpoint_issue_examples: list[str] = field(default_factory=list)
    keyword_issue_examples: list[str] = field(default_factory=list)
    empty_window_examples: list[str] = field(default_factory=list)


def normalize(text: str | None) -> str:
    value = str(text or "").strip().lower().replace("đ", "d")
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens(text: str | None) -> list[str]:
    return [t for t in normalize(text).split() if t]


def loose_contains(needle: str | None, haystack: str | None, *, min_token_coverage: float = 0.8) -> bool:
    n = normalize(needle)
    h = normalize(haystack)
    if not n:
        return True
    if not h:
        return False
    if n in h:
        return True
    ntoks = tokens(needle)
    htoks = set(tokens(haystack))
    if not ntoks:
        return True
    hit = sum(1 for tok in ntoks if tok in htoks)
    return hit / max(1, len(ntoks)) >= min_token_coverage


def evidence_supported(evidence: str, source_text: str) -> bool:
    if not str(evidence or "").strip():
        return False
    return loose_contains(evidence, source_text, min_token_coverage=0.78)


def md_escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", "<br>")


def short(value: Any, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_md_metadata(path: Path) -> tuple[str, str, str, str]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        metadata, body = parse_frontmatter(raw)
        title = str(metadata.get("title") or path.stem)
        url = str(metadata.get("url") or "")
        source = str(metadata.get("source") or "")
        return title, url, source, body
    except Exception:
        return path.stem, "", "", ""


def candidates_from_csv(input_csv: Path) -> list[CandidateArticle]:
    rows = read_csv_rows(input_csv)
    out: list[CandidateArticle] = []
    for row in rows:
        raw_path = row.get("md_path") or row.get("markdown_path") or row.get("path") or ""
        if not raw_path:
            continue
        path = resolve_path(raw_path)
        if not path.exists():
            continue
        title = row.get("title") or row.get("article_title") or ""
        url = row.get("canonical_url") or row.get("url") or ""
        source = row.get("source") or ""
        if not title or not url or not source:
            md_title, md_url, md_source, _body = read_md_metadata(path)
            title = title or md_title
            url = url or md_url
            source = source or md_source
        out.append(CandidateArticle(path=path, title=title, url=url, source=source, row=row))
    return out


def candidates_from_markdown(md_root: Path, patterns: list[str]) -> list[CandidateArticle]:
    seen: set[Path] = set()
    out: list[CandidateArticle] = []
    for pattern in patterns:
        for path in sorted(md_root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            title, url, source, _body = read_md_metadata(path)
            out.append(CandidateArticle(path=path, title=title, url=url, source=source))
    return out


def candidate_text(candidate: CandidateArticle, *, include_body: bool = False) -> str:
    parts = [candidate.title, candidate.url, str(candidate.path), candidate.source]
    if include_body:
        _title, _url, _source, body = read_md_metadata(candidate.path)
        parts.append(body[:12000])
    return "\n".join(str(x or "") for x in parts)


def is_blacklisted(candidate: CandidateArticle, extra_blacklist: list[str] | None = None) -> bool:
    text_norm = normalize(candidate_text(candidate, include_body=False))
    body_norm = ""
    blacklist = list(PROMPT_LEAK_BLACKLIST) + list(extra_blacklist or [])
    for term in blacklist:
        t = normalize(term)
        if t and t in text_norm:
            return True
    # Only read the body if title/path/url did not settle it. This keeps dry-run fast.
    body_norm = normalize(candidate_text(candidate, include_body=True))
    return any((normalize(term) and normalize(term) in body_norm) for term in blacklist)


def score_candidate(candidate: CandidateArticle, group: QueryGroup, *, include_body: bool) -> int:
    text_norm = normalize(candidate_text(candidate, include_body=include_body))
    title_norm = normalize(candidate.title)
    path_norm = normalize(str(candidate.path))
    score = 0
    for alias in group.aliases:
        a = normalize(alias)
        if not a:
            continue
        if a in title_norm:
            score += 120
        elif a in path_norm:
            score += 75
        elif a in text_norm:
            score += 35
        else:
            alias_tokens = [tok for tok in a.split() if len(tok) >= 2]
            if alias_tokens:
                title_hits = sum(1 for tok in alias_tokens if tok in title_norm)
                body_hits = sum(1 for tok in alias_tokens if tok in text_norm)
                if title_hits >= max(1, len(alias_tokens) - 1):
                    score += 45
                elif body_hits >= max(1, len(alias_tokens) - 1):
                    score += 15
    return score


def parse_query_groups(raw: str | None) -> list[QueryGroup]:
    if not raw:
        return [QueryGroup(label, aliases) for label, aliases in DEFAULT_QUERY_GROUPS]
    groups: list[QueryGroup] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split("|") if p.strip()]
        if not parts:
            continue
        groups.append(QueryGroup(parts[0], tuple(parts)))
    return groups


def choose_diverse_candidates(
    candidates: list[CandidateArticle],
    groups: list[QueryGroup],
    *,
    count: int,
    extra_blacklist: list[str] | None,
) -> list[tuple[str, CandidateArticle]]:
    usable = [c for c in candidates if not is_blacklisted(c, extra_blacklist)]
    selected: list[tuple[str, CandidateArticle]] = []
    used_paths: set[Path] = set()
    used_title_tokens: set[str] = set()

    for group in groups:
        scored = [(score_candidate(c, group, include_body=False), c) for c in usable if c.path not in used_paths]
        scored = [(s, c) for s, c in scored if s > 0]
        if not scored:
            scored = [(score_candidate(c, group, include_body=True), c) for c in usable if c.path not in used_paths]
            scored = [(s, c) for s, c in scored if s > 0]
        if not scored:
            continue
        # Prefer good query match, then diverse title tokens.
        rescored: list[tuple[int, CandidateArticle]] = []
        for s, c in scored:
            title_toks = {tok for tok in tokens(c.title) if len(tok) >= 3}
            overlap_penalty = 25 * len(title_toks & used_title_tokens)
            rescored.append((s - overlap_penalty, c))
        rescored.sort(key=lambda item: (item[0], -len(str(item[1].path))), reverse=True)
        picked = rescored[0][1]
        selected.append((group.label, picked))
        used_paths.add(picked.path)
        used_title_tokens.update(tok for tok in tokens(picked.title) if len(tok) >= 3)
        if len(selected) >= count:
            return selected

    # Fallback: pick non-blacklisted articles by title/path diversity even if none
    # matches the query groups. This is better than silently reusing biased topics.
    for c in usable:
        if c.path in used_paths:
            continue
        selected.append(("fallback_diverse", c))
        used_paths.add(c.path)
        if len(selected) >= count:
            return selected
    return selected


def collect_entities(result: ArticleExtractionResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window, graph in zip(result.extraction_windows, result.validated_graphs, strict=True):
        for entity in graph.entities:
            rows.append(
                {
                    "chunk_id": window.chunk_id,
                    "section": window.section or "",
                    "heading_context": (window.metadata or {}).get("heading_path_context", ""),
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "aliases": "; ".join(entity.aliases or []),
                    "surface_forms": "; ".join(entity.surface_forms or []),
                    "evidence_text": entity.evidence_text,
                    "is_anchor_candidate": entity.is_anchor_candidate,
                    "anchor_reason": entity.anchor_reason,
                }
            )
    return rows


def collect_relations(result: ArticleExtractionResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window, graph in zip(result.extraction_windows, result.validated_graphs, strict=True):
        source_context = "\n".join([
            window.title or "",
            window.section or "",
            (window.metadata or {}).get("heading_path_context", ""),
            window.text or "",
        ])
        for relation in graph.relations:
            rows.append(
                {
                    "chunk_id": window.chunk_id,
                    "section": window.section or "",
                    "heading_context": (window.metadata or {}).get("heading_path_context", ""),
                    "subject": relation.subject,
                    "subject_type": relation.subject_type,
                    "relation_type": relation.relation_type,
                    "object": relation.object,
                    "object_type": relation.object_type,
                    "description": relation.description,
                    "keywords": "; ".join(relation.keywords or []),
                    "evidence_text": relation.evidence_text,
                    "confidence": relation.confidence,
                    "evidence_supported_by_chunk": evidence_supported(relation.evidence_text, source_context),
                    "endpoint_covered_by_chunk": loose_contains(relation.subject, source_context) and loose_contains(relation.object, source_context),
                }
            )
    return rows


def collect_chunks(result: ArticleExtractionResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (window, graph) in enumerate(zip(result.extraction_windows, result.validated_graphs, strict=True), start=1):
        rows.append(
            {
                "window_index": index,
                "chunk_id": window.chunk_id,
                "section": window.section or "",
                "heading_context": (window.metadata or {}).get("heading_path_context", ""),
                "text_excerpt": short(window.text, 500),
                "entity_count": len(graph.entities),
                "relation_count": len(graph.relations),
                "should_ingest_graph": graph.should_ingest_graph,
                "skip_reason": graph.skip_reason,
                "entity_names": "; ".join(e.name for e in graph.entities[:30]),
                "relation_preview": " | ".join(relation_display_model(r) for r in graph.relations[:20]),
            }
        )
    return rows


def relation_display(row: dict[str, Any]) -> str:
    return (
        f"{row['subject']} [{row['subject_type']}] "
        f"--{row['relation_type']}--> {row['object']} [{row['object_type']}]"
    )


def relation_display_model(relation: Any) -> str:
    return (
        f"{relation.subject} [{relation.subject_type}] "
        f"--{relation.relation_type}--> {relation.object} [{relation.object_type}]"
    )


def keyword_issues(relation_rows: list[dict[str, Any]], limit: int = 8) -> tuple[int, list[str]]:
    count = 0
    examples: list[str] = []
    for row in relation_rows:
        kws = [kw.strip() for kw in str(row.get("keywords") or "").split(";") if kw.strip()]
        for kw in kws:
            word_count = len(re.findall(r"\S+", kw))
            bad = False
            reason = ""
            if len(kw) > 80 or word_count > 7:
                bad = True
                reason = "too_long"
            elif "," in kw:
                bad = True
                reason = "comma_fragment"
            elif len(kw) >= 36 and loose_contains(kw, row.get("evidence_text", ""), min_token_coverage=0.95):
                bad = True
                reason = "looks_like_evidence_copy"
            if bad:
                count += 1
                if len(examples) < limit:
                    examples.append(f"{reason}: {kw} | {relation_display(row)}")
    return count, examples


def orphan_entities(
    entity_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    *,
    allowed_types: set[str],
    only_anchor: bool = False,
    limit: int = 12,
) -> tuple[int, list[str]]:
    endpoints = {normalize(row.get("subject")) for row in relation_rows} | {normalize(row.get("object")) for row in relation_rows}
    seen: set[tuple[str, str]] = set()
    orphans: list[str] = []
    for row in entity_rows:
        entity_type = str(row.get("entity_type") or "")
        if entity_type not in allowed_types:
            continue
        if only_anchor and str(row.get("is_anchor_candidate")).lower() not in {"true", "1"}:
            continue
        key = (normalize(row.get("name")), entity_type)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        if key[0] not in endpoints:
            orphans.append(f"{row.get('name')} [{entity_type}]")
    return len(orphans), orphans[:limit]


def evaluate_article(group_label: str, candidate: CandidateArticle, article: ArticleDocument, result: ArticleExtractionResult) -> ArticleAudit:
    entity_rows = collect_entities(result)
    relation_rows = collect_relations(result)
    chunk_rows = collect_chunks(result)
    relation_type_counts = Counter(str(row.get("relation_type") or "") for row in relation_rows)
    entity_type_counts = Counter(str(row.get("entity_type") or "") for row in entity_rows)
    useful_relation_count = sum(1 for row in relation_rows if row.get("relation_type") in USEFUL_RELATION_TYPES)

    evidence_bad = [row for row in relation_rows if not row.get("evidence_supported_by_chunk")]
    endpoint_bad = [row for row in relation_rows if not row.get("endpoint_covered_by_chunk")]
    kw_issue_count, kw_examples = keyword_issues(relation_rows)
    orphan_measure_count, orphan_measure_names = orphan_entities(
        entity_rows, relation_rows, allowed_types=MEASURE_ENTITY_TYPES, only_anchor=False
    )
    orphan_anchor_count, orphan_anchor_names = orphan_entities(
        entity_rows, relation_rows, allowed_types=ANCHOR_IMPORTANT_TYPES, only_anchor=True
    )
    empty_windows = [r for r in chunk_rows if int(r["entity_count"]) > 0 and int(r["relation_count"]) == 0]
    low_relation_windows = [r for r in chunk_rows if int(r["entity_count"]) >= 5 and int(r["relation_count"]) <= 1]

    return ArticleAudit(
        group_label=group_label,
        candidate=candidate,
        article=article,
        result=result,
        entity_count=len(entity_rows),
        relation_count=len(relation_rows),
        useful_relation_count=useful_relation_count,
        suspicious_evidence_count=len(evidence_bad),
        endpoint_issue_count=len(endpoint_bad),
        keyword_issue_count=kw_issue_count,
        empty_window_count=len(empty_windows),
        low_relation_window_count=len(low_relation_windows),
        orphan_measure_count=orphan_measure_count,
        orphan_anchor_count=orphan_anchor_count,
        relation_type_counts=relation_type_counts,
        entity_type_counts=entity_type_counts,
        orphan_measure_names=orphan_measure_names,
        orphan_anchor_names=orphan_anchor_names,
        evidence_issue_examples=[relation_display(row) for row in evidence_bad[:8]],
        endpoint_issue_examples=[relation_display(row) for row in endpoint_bad[:8]],
        keyword_issue_examples=kw_examples,
        empty_window_examples=[
            f"{r['chunk_id']} | entities={r['entity_count']} relations={r['relation_count']} | {short(r['section'] or r['heading_context'], 120)}"
            for r in low_relation_windows[:10]
        ],
    )


def overall_status(audit: ArticleAudit) -> str:
    if audit.error:
        return "ERROR"
    if audit.relation_count == 0 and audit.entity_count > 0:
        return "FAIL_NO_RELATIONS"
    if audit.suspicious_evidence_count > max(2, audit.relation_count // 5):
        return "WARN_EVIDENCE"
    if audit.endpoint_issue_count > max(2, audit.relation_count // 5):
        return "WARN_ENDPOINT"
    if audit.low_relation_window_count >= 3 or audit.orphan_measure_count >= 6:
        return "WARN_RECALL"
    if audit.relation_count >= max(4, audit.entity_count // 3):
        return "PASS_REVIEW"
    return "WARN_REVIEW"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, audits: list[ArticleAudit]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for audit in audits:
            if not audit.result or not audit.article:
                continue
            for idx, (window, raw_graph, validated_graph) in enumerate(
                zip(audit.result.extraction_windows, audit.result.raw_graphs, audit.result.validated_graphs, strict=True),
                start=1,
            ):
                row = {
                    "group_label": audit.group_label,
                    "title": audit.article.title,
                    "url": audit.article.url,
                    "md_path": str(audit.candidate.path),
                    "window_index": idx,
                    "chunk_id": window.chunk_id,
                    "section": window.section,
                    "heading_context": (window.metadata or {}).get("heading_path_context", ""),
                    "text": window.text,
                    "raw_graph": raw_graph.model_dump(mode="json"),
                    "validated_entities": [e.model_dump(mode="json") for e in validated_graph.entities],
                    "validated_relations": [r.model_dump(mode="json") for r in validated_graph.relations],
                    "graph_write": {
                        "should_ingest_graph": validated_graph.should_ingest_graph,
                        "skip_reason": validated_graph.skip_reason,
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, audits: list[ArticleAudit], args: argparse.Namespace) -> None:
    lines: list[str] = []
    lines.append("# Unbiased extraction audit — 20 dermatology articles")
    lines.append("")
    lines.append("Audit này tránh các chủ đề đã dùng trong targeted audit/few-shot prompt, nên phù hợp hơn để kiểm tra bias sau khi sửa prompt.")
    lines.append("")
    lines.append("## Run config")
    lines.append("")
    lines.append(f"- input_csv: `{args.input_csv}`")
    lines.append(f"- md_root: `{args.md_root}`")
    lines.append(f"- model: `{args.model or 'env OPENAI_MODEL / default in MedicalExtractor'}`")
    lines.append(f"- keyword_mode: `{'LLM' if args.keyword_llm else 'deterministic_only'}`")
    lines.append(f"- chunk_max_tokens: `{args.max_tokens}`")
    lines.append(f"- fact_window_max_tokens: `{args.fact_window_max_tokens}`")
    lines.append(f"- max_windows_per_article: `{args.max_windows_per_article}`")
    lines.append(f"- selected_count: `{len(audits)}`")
    lines.append("")
    lines.append("## Selection guardrails")
    lines.append("")
    lines.append("Excluded prompt-leak topics include: viêm da dị ứng, kiến ba khoang, mụn cóc, trẻ sơ sinh mẩn đỏ ở cổ, chườm lạnh/tắm nước mát/nha đam, HPV, lang ben/nấm da đầu/nấm móng, and related explicit example terms.")
    if args.extra_blacklist:
        lines.append(f"Extra blacklist: `{args.extra_blacklist}`")
    lines.append("")

    lines.append("## Overall summary")
    lines.append("")
    lines.append("| group | status | article | entities | relations | useful_relations | evidence_issues | endpoint_issues | keyword_issues | low_relation_windows | orphan_measures | orphan_anchors |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for audit in audits:
        title = audit.article.title if audit.article else audit.candidate.title
        lines.append(
            f"| {md_escape(audit.group_label)} | `{overall_status(audit)}` | {md_escape(short(title, 80))} "
            f"| {audit.entity_count} | {audit.relation_count} | {audit.useful_relation_count} "
            f"| {audit.suspicious_evidence_count} | {audit.endpoint_issue_count} | {audit.keyword_issue_count} "
            f"| {audit.low_relation_window_count} | {audit.orphan_measure_count} | {audit.orphan_anchor_count} |"
        )
    lines.append("")

    lines.append("## Cách đọc kết quả")
    lines.append("")
    lines.append("- `PASS_REVIEW`: đủ sạch để review thủ công nhanh, không có nghĩa là perfect.")
    lines.append("- `WARN_RECALL`: nên soi kỹ vì có nhiều window ít relation hoặc nhiều measure/anchor orphan.")
    lines.append("- `WARN_EVIDENCE` / `WARN_ENDPOINT`: ưu tiên soi precision trước khi bulk.")
    lines.append("- Với LightRAG/HippoRAG, relation_type lệch nhẹ không quá nghiêm trọng; cần ưu tiên endpoint + evidence + relation recall.")
    lines.append("")

    for audit in audits:
        lines.append(f"## {audit.group_label}")
        lines.append("")
        if audit.error:
            lines.append("- **Status:** ERROR")
            lines.append(f"- **Error:** `{audit.error}`")
            lines.append("")
            continue
        assert audit.article is not None and audit.result is not None
        lines.append(f"- **Status:** `{overall_status(audit)}`")
        lines.append(f"- **Article:** {audit.article.title}")
        lines.append(f"- **URL:** {audit.article.url}")
        lines.append(f"- **Markdown path:** `{audit.candidate.path}`")
        lines.append(f"- **Chunks/windows:** {len(audit.result.retrieval_chunks)} retrieval chunks / {len(audit.result.extraction_windows)} extraction windows")
        lines.append(f"- **Entities / relations:** {audit.entity_count} / {audit.relation_count}")
        lines.append(f"- **Useful relations:** {audit.useful_relation_count}")
        lines.append("")

        lines.append("### Entity type distribution")
        lines.append("")
        for entity_type, count in audit.entity_type_counts.most_common():
            lines.append(f"- `{entity_type}`: {count}")
        if not audit.entity_type_counts:
            lines.append("- No entities.")
        lines.append("")

        lines.append("### Relation type distribution")
        lines.append("")
        for rel_type, count in audit.relation_type_counts.most_common():
            lines.append(f"- `{rel_type}`: {count}")
        if not audit.relation_type_counts:
            lines.append("- No relations.")
        lines.append("")

        if audit.orphan_measure_names:
            lines.append("### Orphan measure/entity review")
            lines.append("")
            lines.append("Các entity dạng thuốc/can thiệp/chẩn đoán/yếu tố ngoài sinh chưa xuất hiện ở endpoint relation:")
            for name in audit.orphan_measure_names:
                lines.append(f"- {name}")
            lines.append("")

        if audit.orphan_anchor_names:
            lines.append("### Orphan important anchor review")
            lines.append("")
            lines.append("Các anchor quan trọng chưa xuất hiện ở endpoint relation:")
            for name in audit.orphan_anchor_names:
                lines.append(f"- {name}")
            lines.append("")

        if audit.empty_window_examples:
            lines.append("### Low-relation window review")
            lines.append("")
            lines.append("Các window có nhiều entity nhưng ít relation; thường là dấu hiệu hụt recall:")
            for item in audit.empty_window_examples:
                lines.append(f"- {md_escape(item)}")
            lines.append("")

        if audit.evidence_issue_examples:
            lines.append("### Evidence issue examples")
            lines.append("")
            for item in audit.evidence_issue_examples:
                lines.append(f"- {md_escape(item)}")
            lines.append("")

        if audit.endpoint_issue_examples:
            lines.append("### Endpoint coverage issue examples")
            lines.append("")
            for item in audit.endpoint_issue_examples:
                lines.append(f"- {md_escape(item)}")
            lines.append("")

        if audit.keyword_issue_examples:
            lines.append("### Keyword issues")
            lines.append("")
            for item in audit.keyword_issue_examples:
                lines.append(f"- {md_escape(item)}")
            lines.append("")

        relation_rows = collect_relations(audit.result)
        lines.append("### Useful relation examples")
        lines.append("")
        good_rows = [row for row in relation_rows if row.get("relation_type") in USEFUL_RELATION_TYPES]
        if good_rows:
            for row in good_rows[:25]:
                lines.append(f"- `{relation_display(row)}`")
                if row.get("evidence_text"):
                    lines.append(f"  - evidence: {short(row.get('evidence_text'), 240)}")
                if row.get("keywords"):
                    lines.append(f"  - keywords: {short(row.get('keywords'), 160)}")
        else:
            lines.append("- No useful relation examples.")
        lines.append("")

        lines.append("### Chunk/window extraction overview")
        lines.append("")
        lines.append("| chunk | section | entities | relations | preview |")
        lines.append("|---|---|---:|---:|---|")
        for row in collect_chunks(audit.result):
            lines.append(
                f"| `{row['chunk_id']}` | {md_escape(short(row['section'] or row['heading_context'], 80))} "
                f"| {row['entity_count']} | {row['relation_count']} | {md_escape(short(row['relation_preview'], 180))} |"
            )
        lines.append("")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_pipeline(args: argparse.Namespace) -> ArticleExtractionPipeline:
    extractor = MedicalExtractor(
        model=args.model or None,
        entity_max_tokens=args.entity_max_tokens,
        relation_max_tokens=args.relation_max_tokens,
    )
    validator = ExtractionValidator()
    keyword_profiler = ArticleKeywordProfiler(model=(args.keyword_model or None) if args.keyword_llm else "")
    return ArticleExtractionPipeline(
        extractor=extractor,
        validator=validator,
        keyword_profiler=keyword_profiler,
        use_fact_windows=not args.no_fact_windows,
        fact_window_max_tokens=args.fact_window_max_tokens,
        fact_window_min_tokens=args.fact_window_min_tokens,
        fact_window_overlap_sentences=args.fact_window_overlap_sentences,
    )


def run_one(group_label: str, candidate: CandidateArticle, pipeline: ArticleExtractionPipeline, args: argparse.Namespace) -> ArticleAudit:
    try:
        article = load_markdown_article(candidate.path)
        chunks = chunk_article(article, max_tokens=args.max_tokens)
        if args.max_chunks_per_article and args.max_chunks_per_article > 0:
            chunks = chunks[: args.max_chunks_per_article]

        def progress(i: int, total: int, window: ChunkDocument) -> None:
            print(f"[{group_label}] window {i}/{total}: {window.chunk_id} | {short(window.section, 90)}")

        result = pipeline.extract_article(
            article=article,
            retrieval_chunks=chunks,
            max_windows=args.max_windows_per_article,
            progress_callback=progress,
        )
        return evaluate_article(group_label, candidate, article, result)
    except Exception as exc:
        return ArticleAudit(group_label=group_label, candidate=candidate, error=f"{type(exc).__name__}: {exc}")


def build_candidates(args: argparse.Namespace) -> list[CandidateArticle]:
    input_csv = resolve_path(args.input_csv)
    if input_csv.exists():
        print(f"[LOAD] candidates from CSV: {input_csv}")
        return candidates_from_csv(input_csv)

    md_root = resolve_path(args.md_root)
    print(f"[LOAD] CSV not found; scanning markdown under: {md_root}")
    patterns = [p.strip() for p in args.md_glob.split(";") if p.strip()]
    return candidates_from_markdown(md_root, patterns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unbiased 20-article extraction audit and write MD/CSV/JSONL reports.")
    parser.add_argument("--input-csv", default="data/final_selection/articles_final_selected_v1.csv")
    parser.add_argument("--md-root", default=".")
    parser.add_argument("--md-glob", default="data/**/*.md;*.md")
    parser.add_argument("--out-dir", default="data/debug/unbiased_extraction_audit20_readonly")
    parser.add_argument("--model", default=None, help="LLM model for entity/relation stages. Defaults to MedicalExtractor/env.")
    parser.add_argument("--entity-max-tokens", type=int, default=2500)
    parser.add_argument("--relation-max-tokens", type=int, default=4000)
    parser.add_argument("--keyword-llm", action="store_true", help="Use LLM keyword profiler. Default is deterministic keyword enrichment only.")
    parser.add_argument("--keyword-model", default=None, help="Keyword profiler model if --keyword-llm is set.")
    parser.add_argument("--max-tokens", type=int, default=900, help="Retrieval chunk max tokens before fact-window split.")
    parser.add_argument("--no-fact-windows", action="store_true")
    parser.add_argument("--fact-window-max-tokens", type=int, default=450)
    parser.add_argument("--fact-window-min-tokens", type=int, default=80)
    parser.add_argument("--fact-window-overlap-sentences", type=int, default=1)
    parser.add_argument("--max-chunks-per-article", type=int, default=0, help="Debug only. 0 means all chunks.")
    parser.add_argument("--max-windows-per-article", type=int, default=0, help="Debug only. 0 means all extraction windows.")
    parser.add_argument("--dry-run", action="store_true", help="Only select markdown files; do not call LLM.")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--query-groups",
        default="",
        help=(
            "Optional groups separated by ';'. Aliases inside a group separated by '|'. "
            "Example: 'mụn trứng cá|acne;zona|giời leo'."
        ),
    )
    parser.add_argument(
        "--extra-blacklist",
        default="",
        help="Optional extra blacklist terms separated by ';'. Articles containing these are excluded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(args)
    if not candidates:
        raise SystemExit("No markdown candidates found. Check --input-csv, --md-root, or --md-glob.")
    print(f"[CANDIDATES] {len(candidates)}")

    groups = parse_query_groups(args.query_groups)
    extra_blacklist = [x.strip() for x in args.extra_blacklist.split(";") if x.strip()]
    selected = choose_diverse_candidates(candidates, groups, count=args.count, extra_blacklist=extra_blacklist)
    if not selected:
        raise SystemExit("No non-blacklisted articles selected. Relax --extra-blacklist or change --query-groups.")

    for group_label, candidate in selected:
        print(f"[SELECT] {group_label} -> {candidate.title} | {candidate.path}")

    if args.dry_run:
        lines = ["# Unbiased extraction audit dry run", ""]
        lines.append("Selected files only; no LLM call was made.")
        lines.append("")
        for group_label, candidate in selected:
            lines.append(f"- **{group_label}** -> `{candidate.path}` — {candidate.title}")
        (out_dir / "unbiased20_extraction_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[DRY RUN DONE] {out_dir / 'unbiased20_extraction_audit.md'}")
        return

    pipeline = build_pipeline(args)
    audits = [run_one(group_label, candidate, pipeline, args) for group_label, candidate in selected]

    all_entity_rows: list[dict[str, Any]] = []
    all_relation_rows: list[dict[str, Any]] = []
    all_chunk_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for audit in audits:
        if audit.result and audit.article:
            for row in collect_entities(audit.result):
                all_entity_rows.append(
                    {
                        "group_label": audit.group_label,
                        "article_title": audit.article.title,
                        "url": audit.article.url,
                        "md_path": str(audit.candidate.path),
                        **row,
                    }
                )
            for row in collect_relations(audit.result):
                all_relation_rows.append(
                    {
                        "group_label": audit.group_label,
                        "article_title": audit.article.title,
                        "url": audit.article.url,
                        "md_path": str(audit.candidate.path),
                        **row,
                    }
                )
            for row in collect_chunks(audit.result):
                all_chunk_rows.append(
                    {
                        "group_label": audit.group_label,
                        "article_title": audit.article.title,
                        "url": audit.article.url,
                        "md_path": str(audit.candidate.path),
                        **row,
                    }
                )
        summary_rows.append(
            {
                "group_label": audit.group_label,
                "status": overall_status(audit),
                "article_title": audit.article.title if audit.article else audit.candidate.title,
                "url": audit.article.url if audit.article else audit.candidate.url,
                "md_path": str(audit.candidate.path),
                "error": audit.error,
                "entity_count": audit.entity_count,
                "relation_count": audit.relation_count,
                "useful_relation_count": audit.useful_relation_count,
                "suspicious_evidence_count": audit.suspicious_evidence_count,
                "endpoint_issue_count": audit.endpoint_issue_count,
                "keyword_issue_count": audit.keyword_issue_count,
                "empty_window_count": audit.empty_window_count,
                "low_relation_window_count": audit.low_relation_window_count,
                "orphan_measure_count": audit.orphan_measure_count,
                "orphan_anchor_count": audit.orphan_anchor_count,
                "relation_type_counts": json.dumps(dict(audit.relation_type_counts), ensure_ascii=False),
                "entity_type_counts": json.dumps(dict(audit.entity_type_counts), ensure_ascii=False),
                "orphan_measures": "; ".join(audit.orphan_measure_names),
                "orphan_anchors": "; ".join(audit.orphan_anchor_names),
            }
        )

    write_csv(
        out_dir / "unbiased20_extraction_summary.csv",
        summary_rows,
        [
            "group_label",
            "status",
            "article_title",
            "url",
            "md_path",
            "error",
            "entity_count",
            "relation_count",
            "useful_relation_count",
            "suspicious_evidence_count",
            "endpoint_issue_count",
            "keyword_issue_count",
            "empty_window_count",
            "low_relation_window_count",
            "orphan_measure_count",
            "orphan_anchor_count",
            "relation_type_counts",
            "entity_type_counts",
            "orphan_measures",
            "orphan_anchors",
        ],
    )
    write_csv(
        out_dir / "unbiased20_relations.csv",
        all_relation_rows,
        [
            "group_label",
            "article_title",
            "url",
            "md_path",
            "chunk_id",
            "section",
            "heading_context",
            "subject",
            "subject_type",
            "relation_type",
            "object",
            "object_type",
            "description",
            "keywords",
            "evidence_text",
            "confidence",
            "evidence_supported_by_chunk",
            "endpoint_covered_by_chunk",
        ],
    )
    write_csv(
        out_dir / "unbiased20_entities.csv",
        all_entity_rows,
        [
            "group_label",
            "article_title",
            "url",
            "md_path",
            "chunk_id",
            "section",
            "heading_context",
            "name",
            "entity_type",
            "aliases",
            "surface_forms",
            "evidence_text",
            "is_anchor_candidate",
            "anchor_reason",
        ],
    )
    write_csv(
        out_dir / "unbiased20_chunks.csv",
        all_chunk_rows,
        [
            "group_label",
            "article_title",
            "url",
            "md_path",
            "window_index",
            "chunk_id",
            "section",
            "heading_context",
            "text_excerpt",
            "entity_count",
            "relation_count",
            "should_ingest_graph",
            "skip_reason",
            "entity_names",
            "relation_preview",
        ],
    )
    write_jsonl(out_dir / "raw_and_validated_chunk_extractions.jsonl", audits)
    write_markdown(out_dir / "unbiased20_extraction_audit.md", audits, args)

    print("[DONE]")
    print(f"out_md       : {out_dir / 'unbiased20_extraction_audit.md'}")
    print(f"summary_csv  : {out_dir / 'unbiased20_extraction_summary.csv'}")
    print(f"relations_csv: {out_dir / 'unbiased20_relations.csv'}")
    print(f"entities_csv : {out_dir / 'unbiased20_entities.csv'}")
    print(f"chunks_csv   : {out_dir / 'unbiased20_chunks.csv'}")
    print(f"jsonl        : {out_dir / 'raw_and_validated_chunk_extractions.jsonl'}")


if __name__ == "__main__":
    main()
