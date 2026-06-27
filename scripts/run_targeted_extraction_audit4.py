#!/usr/bin/env python
from __future__ import annotations

"""
Run a small targeted extraction audit for the 4 hard dermatology articles.

This script is intentionally self-contained and does NOT depend on the older
one-off scripts that used deprecated compatibility imports.

Safety:
- No Neo4j writes.
- No Qdrant writes.
- No ingest/build step.
- No --fresh/delete-output behavior.

Outputs:
- targeted_extraction_audit.md: readable summary + automatic checks
- targeted_extraction_summary.csv: one row per target/article
- targeted_relations.csv: flattened validated relations
- targeted_entities.csv: flattened validated entities
- raw_and_validated_chunk_extractions.jsonl: chunk/window-level raw + validated dumps
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
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional for this script
    load_dotenv = None

from app.ingestion.extraction.entity.medical_extractor import MedicalExtractor
from app.ingestion.extraction.keyword.keyword_profiler import ArticleKeywordProfiler
from app.ingestion.extraction.pipelines.article_extraction_pipeline import (
    ArticleExtractionPipeline,
    ArticleExtractionResult,
)
from app.ingestion.extraction.relation.extraction_validator import ExtractionValidator
from app.ingestion.processing.chunker import chunk_article
from app.ingestion.processing.markdown_loader import load_markdown_article, parse_frontmatter
from app.ingestion.processing.ingestion_models import ArticleDocument, ChunkDocument


MEASURE_ENTITY_TYPES = {
    "THUOC_VA_HOAT_CHAT",
    "CAN_THIEP_Y_TE",
    "PHUONG_PHAP_CHAN_DOAN",
    "YEU_TO_NGOAI_SINH",
}
USEFUL_RELATION_TYPES = {
    "DIEU_TRI_BANG",
    "PHONG_NGUA_BANG",
    "CHAN_DOAN_BANG",
    "CAN_TRANH",
}


@dataclass(frozen=True)
class ExpectedTerm:
    label: str
    aliases: tuple[str, ...]
    relation_types: frozenset[str] = frozenset()


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    label: str
    search_terms: tuple[str, ...]
    required_terms: tuple[ExpectedTerm, ...]
    min_useful_relations: int = 0
    notes: str = ""


TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        target_id="viem_da_di_ung_tai_nha",
        label="viêm da dị ứng tại nhà",
        search_terms=("viêm da dị ứng", "tại nhà"),
        min_useful_relations=4,
        notes="Kỳ vọng có home-care treatment edges: chườm lạnh, tắm nước mát, uống nước, nha đam.",
        required_terms=(
            ExpectedTerm("chườm lạnh", ("chườm lạnh", "chườm mát"), frozenset({"DIEU_TRI_BANG"})),
            ExpectedTerm("tắm nước mát", ("tắm nước mát", "nước mát"), frozenset({"DIEU_TRI_BANG"})),
            ExpectedTerm("uống nước", ("uống nước", "bổ sung nước"), frozenset({"DIEU_TRI_BANG"})),
            ExpectedTerm("nha đam", ("nha đam", "lô hội"), frozenset({"DIEU_TRI_BANG", "PHONG_NGUA_BANG"})),
        ),
    ),
    TargetSpec(
        target_id="kien_ba_khoang",
        label="kiến ba khoang",
        search_terms=("kiến ba khoang",),
        min_useful_relations=3,
        notes="Kỳ vọng prevention/avoidance bullets thành PHONG_NGUA_BANG hoặc CAN_TRANH.",
        required_terms=(
            ExpectedTerm("lưới chống côn trùng", ("lưới chống côn trùng", "lưới chống muỗi"), frozenset({"PHONG_NGUA_BANG", "CAN_TRANH"})),
            ExpectedTerm("ngủ trong mùng", ("ngủ trong mùng", "mắc màn", "mùng"), frozenset({"PHONG_NGUA_BANG", "CAN_TRANH"})),
            ExpectedTerm("kiểm tra chăn gối, quần áo", ("kiểm tra chăn", "kiểm tra chăn gối", "kiểm tra quần áo", "chăn gối"), frozenset({"PHONG_NGUA_BANG", "CAN_TRANH"})),
        ),
    ),
    TargetSpec(
        target_id="mun_coc_duoi_ban_chan",
        label="mụn cóc dưới bàn chân",
        search_terms=("mụn cóc dưới bàn chân", "mụn cóc bàn chân"),
        min_useful_relations=3,
        notes="Kỳ vọng treatment recall không tụt; ưu tiên DIEU_TRI_BANG cho các biện pháp điều trị.",
        required_terms=(
            ExpectedTerm("axit salicylic", ("axit salicylic", "salicylic"), frozenset({"DIEU_TRI_BANG"})),
            ExpectedTerm("áp lạnh", ("áp lạnh", "liệu pháp lạnh", "nitơ lỏng"), frozenset({"DIEU_TRI_BANG"})),
            ExpectedTerm("laser / thủ thuật", ("laser", "phẫu thuật", "đốt", "thủ thuật"), frozenset({"DIEU_TRI_BANG"})),
        ),
    ),
    TargetSpec(
        target_id="tre_so_sinh_man_do_o_co",
        label="trẻ sơ sinh nổi mẩn đỏ ở cổ",
        search_terms=("trẻ sơ sinh", "mẩn đỏ", "cổ"),
        min_useful_relations=3,
        notes="Kỳ vọng care/prevention edges, bớt orphan intervention.",
        required_terms=(
            ExpectedTerm("giữ khô thoáng", ("khô thoáng", "giữ khô", "thoáng mát"), frozenset({"DIEU_TRI_BANG", "PHONG_NGUA_BANG"})),
            ExpectedTerm("vệ sinh / tắm", ("vệ sinh", "tắm", "lau sạch"), frozenset({"DIEU_TRI_BANG", "PHONG_NGUA_BANG"})),
            ExpectedTerm("quần áo thoáng", ("quần áo", "vải cotton", "thoáng khí"), frozenset({"DIEU_TRI_BANG", "PHONG_NGUA_BANG", "CAN_TRANH"})),
        ),
    ),
)


@dataclass
class CandidateArticle:
    path: Path
    title: str = ""
    url: str = ""
    source: str = ""
    row: dict[str, str] = field(default_factory=dict)


@dataclass
class TermHit:
    label: str
    status: str
    matched_relation: str = ""
    matched_type: str = ""
    matched_scope: str = ""


@dataclass
class ArticleAudit:
    target: TargetSpec
    candidate: CandidateArticle
    article: ArticleDocument | None = None
    result: ArticleExtractionResult | None = None
    error: str = ""
    term_hits: list[TermHit] = field(default_factory=list)
    relation_type_counts: Counter[str] = field(default_factory=Counter)
    entity_type_counts: Counter[str] = field(default_factory=Counter)
    useful_relation_count: int = 0
    relation_count: int = 0
    entity_count: int = 0
    orphan_measure_count: int = 0
    orphan_measure_names: list[str] = field(default_factory=list)
    suspicious_evidence_count: int = 0
    keyword_issue_count: int = 0
    keyword_issue_examples: list[str] = field(default_factory=list)
    evidence_issue_examples: list[str] = field(default_factory=list)


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
    text = str(value or "")
    text = text.replace("|", "\\|").replace("\n", "<br>")
    return text


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


def candidate_score(candidate: CandidateArticle, target: TargetSpec, *, include_body: bool) -> int:
    title_norm = normalize(candidate.title)
    path_norm = normalize(str(candidate.path))
    url_norm = normalize(candidate.url)
    body_norm = ""
    if include_body:
        _title, _url, _source, body = read_md_metadata(candidate.path)
        body_norm = normalize(body[:9000])

    score = 0
    matched_any = False
    for term in target.search_terms:
        t = normalize(term)
        if not t:
            continue
        if t in title_norm:
            score += 100
            matched_any = True
        elif t in path_norm or t in url_norm:
            score += 50
            matched_any = True
        elif body_norm and t in body_norm:
            score += 25
            matched_any = True
        else:
            # Token-level partial match for target like: trẻ sơ sinh + mẩn đỏ + cổ.
            term_tokens = [x for x in t.split() if len(x) >= 2]
            if term_tokens:
                title_hits = sum(1 for tok in term_tokens if tok in title_norm)
                body_hits = sum(1 for tok in term_tokens if body_norm and tok in body_norm)
                if title_hits >= max(1, len(term_tokens) - 1):
                    score += 40
                    matched_any = True
                elif body_hits >= max(1, len(term_tokens) - 1):
                    score += 15
                    matched_any = True
    return score if matched_any else 0


def choose_candidate(candidates: list[CandidateArticle], target: TargetSpec) -> CandidateArticle | None:
    scored = [(candidate_score(c, target, include_body=False), c) for c in candidates]
    scored = [(s, c) for s, c in scored if s > 0]
    if not scored:
        scored = [(candidate_score(c, target, include_body=True), c) for c in candidates]
        scored = [(s, c) for s, c in scored if s > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], -len(str(item[1].path))), reverse=True)
    return scored[0][1]


def collect_entities(result: ArticleExtractionResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window, graph in zip(result.extraction_windows, result.validated_graphs, strict=True):
        for entity in graph.entities:
            rows.append(
                {
                    "chunk_id": window.chunk_id,
                    "section": window.section or "",
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "aliases": "; ".join(entity.aliases or []),
                    "surface_forms": "; ".join(entity.surface_forms or []),
                    "evidence_text": entity.evidence_text,
                    "is_anchor_candidate": entity.is_anchor_candidate,
                }
            )
    return rows


def collect_relations(result: ArticleExtractionResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window, graph in zip(result.extraction_windows, result.validated_graphs, strict=True):
        source_context = "\n".join([window.title or "", window.section or "", window.text or ""])
        for relation in graph.relations:
            rows.append(
                {
                    "chunk_id": window.chunk_id,
                    "section": window.section or "",
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


def relation_display(row: dict[str, Any]) -> str:
    return (
        f"{row['subject']} [{row['subject_type']}] "
        f"--{row['relation_type']}--> {row['object']} [{row['object_type']}]"
    )


def match_expected_terms(target: TargetSpec, relation_rows: list[dict[str, Any]]) -> list[TermHit]:
    hits: list[TermHit] = []
    for expected in target.required_terms:
        best: TermHit | None = None
        for row in relation_rows:
            endpoint_text = f"{row['subject']} {row['object']}"
            evidence_text = f"{row.get('evidence_text', '')} {row.get('keywords', '')}"
            relation_type = str(row.get("relation_type") or "")
            type_ok = not expected.relation_types or relation_type in expected.relation_types

            endpoint_hit = any(loose_contains(alias, endpoint_text, min_token_coverage=0.75) for alias in expected.aliases)
            evidence_hit = any(loose_contains(alias, evidence_text, min_token_coverage=0.75) for alias in expected.aliases)
            if endpoint_hit and type_ok:
                best = TermHit(expected.label, "PASS", relation_display(row), relation_type, "endpoint+type")
                break
            if endpoint_hit and best is None:
                best = TermHit(expected.label, "WARN_TYPE", relation_display(row), relation_type, "endpoint_wrong_type")
            elif evidence_hit and best is None:
                best = TermHit(expected.label, "WARN_EVIDENCE_ONLY", relation_display(row), relation_type, "evidence_or_keyword_only")
        hits.append(best or TermHit(expected.label, "MISS"))
    return hits


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


def orphan_measures(entity_rows: list[dict[str, Any]], relation_rows: list[dict[str, Any]], limit: int = 12) -> tuple[int, list[str]]:
    endpoints = {normalize(row.get("subject")) for row in relation_rows} | {normalize(row.get("object")) for row in relation_rows}
    seen: set[tuple[str, str]] = set()
    orphans: list[str] = []
    for row in entity_rows:
        entity_type = str(row.get("entity_type") or "")
        if entity_type not in MEASURE_ENTITY_TYPES:
            continue
        key = (normalize(row.get("name")), entity_type)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        if key[0] not in endpoints:
            orphans.append(f"{row.get('name')} [{entity_type}]")
    return len(orphans), orphans[:limit]


def evaluate_article(target: TargetSpec, candidate: CandidateArticle, article: ArticleDocument, result: ArticleExtractionResult) -> ArticleAudit:
    entity_rows = collect_entities(result)
    relation_rows = collect_relations(result)
    relation_type_counts = Counter(str(row.get("relation_type") or "") for row in relation_rows)
    entity_type_counts = Counter(str(row.get("entity_type") or "") for row in entity_rows)
    useful_relation_count = sum(1 for row in relation_rows if row.get("relation_type") in USEFUL_RELATION_TYPES)
    suspicious_rows = [row for row in relation_rows if not row.get("evidence_supported_by_chunk") or not row.get("endpoint_covered_by_chunk")]
    kw_issue_count, kw_examples = keyword_issues(relation_rows)
    orphan_count, orphan_names = orphan_measures(entity_rows, relation_rows)

    return ArticleAudit(
        target=target,
        candidate=candidate,
        article=article,
        result=result,
        term_hits=match_expected_terms(target, relation_rows),
        relation_type_counts=relation_type_counts,
        entity_type_counts=entity_type_counts,
        useful_relation_count=useful_relation_count,
        relation_count=len(relation_rows),
        entity_count=len(entity_rows),
        orphan_measure_count=orphan_count,
        orphan_measure_names=orphan_names,
        suspicious_evidence_count=len(suspicious_rows),
        keyword_issue_count=kw_issue_count,
        keyword_issue_examples=kw_examples,
        evidence_issue_examples=[relation_display(row) for row in suspicious_rows[:8]],
    )


def overall_status(audit: ArticleAudit) -> str:
    if audit.error:
        return "ERROR"
    miss_count = sum(1 for hit in audit.term_hits if hit.status == "MISS")
    pass_count = sum(1 for hit in audit.term_hits if hit.status == "PASS")
    if miss_count == 0 and audit.useful_relation_count >= audit.target.min_useful_relations and audit.suspicious_evidence_count <= max(2, audit.relation_count // 5):
        return "PASS"
    if pass_count >= max(1, len(audit.term_hits) // 2) or audit.useful_relation_count >= audit.target.min_useful_relations:
        return "WARN_REVIEW"
    return "FAIL"


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
                    "target_id": audit.target.target_id,
                    "target_label": audit.target.label,
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
    lines.append("# Targeted extraction audit — 4 hard dermatology articles")
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
    lines.append("")

    lines.append("## Overall summary")
    lines.append("")
    lines.append("| target | status | article | relations | useful_relations | suspicious_evidence | keyword_issues | orphan_measures | expected_hits |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for audit in audits:
        expected_hits = ", ".join(f"{h.label}:{h.status}" for h in audit.term_hits) if audit.term_hits else ""
        title = audit.article.title if audit.article else audit.candidate.title
        lines.append(
            f"| {md_escape(audit.target.label)} | {overall_status(audit)} | {md_escape(short(title, 80))} "
            f"| {audit.relation_count} | {audit.useful_relation_count} | {audit.suspicious_evidence_count} "
            f"| {audit.keyword_issue_count} | {audit.orphan_measure_count} | {md_escape(expected_hits)} |"
        )
    lines.append("")

    for audit in audits:
        lines.append(f"## {audit.target.label}")
        lines.append("")
        if audit.error:
            lines.append(f"- **Status:** ERROR")
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
        lines.append(f"- **Useful relation count:** {audit.useful_relation_count} / min expected {audit.target.min_useful_relations}")
        lines.append(f"- **Note:** {audit.target.notes}")
        lines.append("")

        lines.append("### Relation type distribution")
        lines.append("")
        if audit.relation_type_counts:
            for rel_type, count in audit.relation_type_counts.most_common():
                lines.append(f"- `{rel_type}`: {count}")
        else:
            lines.append("- No relations.")
        lines.append("")

        lines.append("### Expected-term checks")
        lines.append("")
        lines.append("| expected | status | matched relation | scope |")
        lines.append("|---|---|---|---|")
        for hit in audit.term_hits:
            lines.append(
                f"| {md_escape(hit.label)} | `{hit.status}` | {md_escape(short(hit.matched_relation, 180))} | {md_escape(hit.matched_scope)} |"
            )
        lines.append("")

        if audit.orphan_measure_names:
            lines.append("### Orphan measure/entity review")
            lines.append("")
            lines.append("Các entity dạng thuốc/can thiệp/chẩn đoán/yếu tố ngoài sinh chưa xuất hiện ở endpoint relation:")
            for name in audit.orphan_measure_names:
                lines.append(f"- {name}")
            lines.append("")

        if audit.evidence_issue_examples:
            lines.append("### Evidence/support suspicious examples")
            lines.append("")
            lines.append("Automatic heuristic only; review manually before concluding hallucination.")
            for item in audit.evidence_issue_examples:
                lines.append(f"- {item}")
            lines.append("")

        if audit.keyword_issue_examples:
            lines.append("### Keyword issues")
            lines.append("")
            for item in audit.keyword_issue_examples:
                lines.append(f"- {md_escape(item)}")
            lines.append("")

        relation_rows = collect_relations(audit.result)
        good_rows = [row for row in relation_rows if row.get("relation_type") in USEFUL_RELATION_TYPES]
        lines.append("### Useful relation examples")
        lines.append("")
        if good_rows:
            for row in good_rows[:20]:
                kws = str(row.get("keywords") or "")
                lines.append(f"- `{relation_display(row)}`")
                if row.get("evidence_text"):
                    lines.append(f"  - evidence: {short(row.get('evidence_text'), 240)}")
                if kws:
                    lines.append(f"  - keywords: {short(kws, 160)}")
        else:
            lines.append("- No useful relation examples.")
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


def run_one(target: TargetSpec, candidate: CandidateArticle, pipeline: ArticleExtractionPipeline, args: argparse.Namespace) -> ArticleAudit:
    try:
        article = load_markdown_article(candidate.path)
        chunks = chunk_article(article, max_tokens=args.max_tokens)
        if args.max_chunks_per_article and args.max_chunks_per_article > 0:
            chunks = chunks[: args.max_chunks_per_article]

        def progress(i: int, total: int, window: ChunkDocument) -> None:
            print(f"[{target.target_id}] window {i}/{total}: {window.chunk_id} | {short(window.section, 90)}")

        result = pipeline.extract_article(
            article=article,
            retrieval_chunks=chunks,
            max_windows=args.max_windows_per_article,
            progress_callback=progress,
        )
        return evaluate_article(target, candidate, article, result)
    except Exception as exc:  # keep the rest of the audit running
        return ArticleAudit(target=target, candidate=candidate, error=f"{type(exc).__name__}: {exc}")


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
    parser = argparse.ArgumentParser(description="Run targeted 4-article extraction audit and write MD/CSV/JSONL reports.")
    parser.add_argument("--input-csv", default="data/final_selection/articles_final_selected_v1.csv")
    parser.add_argument("--md-root", default=".")
    parser.add_argument("--md-glob", default="data/**/*.md;*.md")
    parser.add_argument("--out-dir", default="data/debug/targeted_extraction_audit4_readonly")
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
    parser.add_argument("--dry-run", action="store_true", help="Only select target markdown files; do not call LLM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    out_dir = resolve_path(args.out_dir)
    # Read-only audit with respect to databases: this script never writes Neo4j/Qdrant.
    # It also never deletes the output directory; existing report files may be overwritten.
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(args)
    if not candidates:
        raise SystemExit("No markdown candidates found. Check --input-csv, --md-root, or --md-glob.")
    print(f"[CANDIDATES] {len(candidates)}")

    selected: list[tuple[TargetSpec, CandidateArticle]] = []
    for target in TARGETS:
        candidate = choose_candidate(candidates, target)
        if candidate is None:
            print(f"[MISS] target={target.label} | no matching markdown found")
            selected.append((target, CandidateArticle(path=Path("NOT_FOUND"), title="NOT_FOUND")))
        else:
            print(f"[SELECT] {target.label} -> {candidate.title} | {candidate.path}")
            selected.append((target, candidate))

    if args.dry_run:
        lines = ["# Targeted extraction audit dry run", ""]
        for target, candidate in selected:
            lines.append(f"- **{target.label}** -> `{candidate.path}` — {candidate.title}")
        (out_dir / "targeted_extraction_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[DRY RUN DONE] {out_dir / 'targeted_extraction_audit.md'}")
        return

    pipeline = build_pipeline(args)
    audits: list[ArticleAudit] = []
    for target, candidate in selected:
        if candidate.title == "NOT_FOUND":
            audits.append(ArticleAudit(target=target, candidate=candidate, error="No matching markdown found"))
            continue
        audits.append(run_one(target, candidate, pipeline, args))

    all_entity_rows: list[dict[str, Any]] = []
    all_relation_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for audit in audits:
        if audit.result and audit.article:
            entity_rows = collect_entities(audit.result)
            relation_rows = collect_relations(audit.result)
            for row in entity_rows:
                all_entity_rows.append(
                    {
                        "target_id": audit.target.target_id,
                        "target_label": audit.target.label,
                        "article_title": audit.article.title,
                        "url": audit.article.url,
                        "md_path": str(audit.candidate.path),
                        **row,
                    }
                )
            for row in relation_rows:
                all_relation_rows.append(
                    {
                        "target_id": audit.target.target_id,
                        "target_label": audit.target.label,
                        "article_title": audit.article.title,
                        "url": audit.article.url,
                        "md_path": str(audit.candidate.path),
                        **row,
                    }
                )
        summary_rows.append(
            {
                "target_id": audit.target.target_id,
                "target_label": audit.target.label,
                "status": overall_status(audit),
                "article_title": audit.article.title if audit.article else audit.candidate.title,
                "url": audit.article.url if audit.article else audit.candidate.url,
                "md_path": str(audit.candidate.path),
                "error": audit.error,
                "entity_count": audit.entity_count,
                "relation_count": audit.relation_count,
                "useful_relation_count": audit.useful_relation_count,
                "suspicious_evidence_count": audit.suspicious_evidence_count,
                "keyword_issue_count": audit.keyword_issue_count,
                "orphan_measure_count": audit.orphan_measure_count,
                "relation_type_counts": json.dumps(dict(audit.relation_type_counts), ensure_ascii=False),
                "expected_term_hits": json.dumps([hit.__dict__ for hit in audit.term_hits], ensure_ascii=False),
            }
        )

    write_csv(
        out_dir / "targeted_extraction_summary.csv",
        summary_rows,
        [
            "target_id",
            "target_label",
            "status",
            "article_title",
            "url",
            "md_path",
            "error",
            "entity_count",
            "relation_count",
            "useful_relation_count",
            "suspicious_evidence_count",
            "keyword_issue_count",
            "orphan_measure_count",
            "relation_type_counts",
            "expected_term_hits",
        ],
    )
    write_csv(
        out_dir / "targeted_relations.csv",
        all_relation_rows,
        [
            "target_id",
            "target_label",
            "article_title",
            "url",
            "md_path",
            "chunk_id",
            "section",
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
        out_dir / "targeted_entities.csv",
        all_entity_rows,
        [
            "target_id",
            "target_label",
            "article_title",
            "url",
            "md_path",
            "chunk_id",
            "section",
            "name",
            "entity_type",
            "aliases",
            "surface_forms",
            "evidence_text",
            "is_anchor_candidate",
        ],
    )
    write_jsonl(out_dir / "raw_and_validated_chunk_extractions.jsonl", audits)
    write_markdown(out_dir / "targeted_extraction_audit.md", audits, args)

    print("[DONE]")
    print(f"out_md       : {out_dir / 'targeted_extraction_audit.md'}")
    print(f"summary_csv  : {out_dir / 'targeted_extraction_summary.csv'}")
    print(f"relations_csv: {out_dir / 'targeted_relations.csv'}")
    print(f"entities_csv : {out_dir / 'targeted_entities.csv'}")
    print(f"jsonl        : {out_dir / 'raw_and_validated_chunk_extractions.jsonl'}")


if __name__ == "__main__":
    main()
