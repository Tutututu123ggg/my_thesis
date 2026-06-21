from __future__ import annotations

# Important on Windows + sentence-transformers / MKL / OpenMP.
# Put this before importing project modules.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import dataclasses
import faulthandler
import gc
import json
import re
import sys
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.datasets import load_evaluation_questions
from app.evaluation.retriever_factory import RetrieverBuildConfig, build_retriever_bundle


# =========================
# Utility
# =========================

def norm_text(text: str) -> str:
    text = (text or "").strip().lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_url(url: str) -> str:
    text = str(url or "").strip().strip("<>\"'").lower()
    if not text:
        return ""
    text = text.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    text = re.sub(r"^https?://", "", text)
    if text.startswith("www."):
        text = text[4:]
    text = re.sub(r"/amp$", "", text)
    text = re.sub(r"/+", "/", text)
    return text.rstrip("/")


def url_slug(url: str) -> str:
    u = norm_url(url)
    return u.rsplit("/", 1)[-1] if u else ""


def urls_match(a: str, b: str) -> bool:
    a = norm_url(a)
    b = norm_url(b)
    if not a or not b:
        return False
    if a == b or a.endswith(b) or b.endswith(a):
        return True
    return bool(url_slug(a) and url_slug(a) == url_slug(b))


def short(text: Any, n: int = 360) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def score(x: Any) -> float:
    try:
        return float(x or 0.0)
    except Exception:
        return 0.0


def jsonable(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return {k: jsonable(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def getv(obj: Any, key: str, default: Any = "") -> Any:
    """Read a field from either dataclass/object chunks or compact dict chunks."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        fh.flush()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def compact_chunk(chunk: Any, expected_terms: list[str], expected_sources: list[str]) -> dict[str, Any]:
    """Keep only fields needed for reporting; drop heavy framework objects early."""
    metadata = dict(getv(chunk, "metadata", {}) or {})
    text = str(getv(chunk, "text", "") or getv(chunk, "contextualized_text", "") or "")
    contextualized = str(getv(chunk, "contextualized_text", "") or "")
    return {
        "chunk_id": getv(chunk, "chunk_id", ""),
        "score": score(getv(chunk, "score", 0.0)),
        "source": getv(chunk, "source", ""),
        "article_id": getv(chunk, "article_id", ""),
        "title": getv(chunk, "title", ""),
        "section": getv(chunk, "section", ""),
        "subsection": getv(chunk, "subsection", ""),
        "source_url": getv(chunk, "source_url", ""),
        "text": short(text or contextualized, 1200),
        "contextualized_text": short(contextualized, 1200),
        "metadata": {
            k: metadata.get(k)
            for k in [
                "mixed_sources",
                "evidence_expansion",
                "evidence_expansion_reasons",
                "evidence_expansion_intent",
        "evidence_expansion_raw_score",
        "evidence_expansion_article_base",
        "evidence_expansion_article_anchor",
        "evidence_expansion_bonus",
        "evidence_expansion_adjacent_bonus",
        "evidence_expansion_score_cap",
        "evidence_expansion_breakdown",
                "evidence_expansion_raw_score",
                "evidence_expansion_article_base",
                "evidence_expansion_article_anchor",
                "evidence_expansion_bonus",
                "evidence_expansion_adjacent_bonus",
                "evidence_expansion_score_cap",
                "evidence_expansion_breakdown",
                "supporting_entity_ids",
                "supporting_relation_ids",
                "query_anchor_original_score",
                "query_anchor_boost",
                "query_anchor_matches",
                "query_slot_matches",
                "query_term_matches",
                "hipporag_original_chunk_score",
                "hipporag_query_evidence_boost",
                "hipporag_query_term_matches",
                "hipporag_slot_term_matches",
                "hipporag_anchor_term_matches",
                "hipporag_passage_integration",
                "vector_original_score",
                "passage_added_score",
                "lightrag_original_chunk_score",
                "lightrag_disease_candidate_boost",
                "lightrag_matched_diseases",
                "cross_encoder_score",
                "cross_encoder_score_norm",
                "cross_encoder_rank",
                "pre_rerank_score",
                "pre_rerank_rank",
                "pre_rerank_source",
                "retrieval_score_norm",
                "final_combined_score",
                "final_rerank_rank",
                "rerank_model_name",
                "rerank_method_name",
                "cross_encoder_available",
                "cross_encoder_error",
                "cross_encoder_weight",
                "retrieval_weight",
                "rerank_preserved_by_backbone",
                "rerank_preserve_reason",
                "rerank_preserve_top_k",
                "rerank_preserve_at_top",
                "rerank_preserve_at_top_qualified",
                "rerank_preserve_at_top_max_ce_rank",
                "rerank_preserve_not_forced_reason",
                "source_url",
                "article_url",
                "url",
            ]
            if k in metadata
        },
        "term_hits": matched_expected_terms(chunk, expected_terms),
        "source_hits": source_hit(chunk, expected_sources),
    }




def chunk_score_breakdown(chunk: Any) -> dict[str, Any]:
    meta = dict(getv(chunk, "metadata", {}) or {})
    keys = [
        "supporting_entity_ids",
        "supporting_relation_ids",
        "query_anchor_original_score",
        "query_anchor_boost",
        "query_anchor_matches",
        "query_slot_matches",
        "query_term_matches",
        "hipporag_original_chunk_score",
        "hipporag_query_evidence_boost",
        "hipporag_query_term_matches",
        "hipporag_slot_term_matches",
        "hipporag_anchor_term_matches",
        "hipporag_passage_integration",
        "vector_original_score",
        "passage_added_score",
        "lightrag_original_chunk_score",
        "lightrag_disease_candidate_boost",
        "lightrag_matched_diseases",
        "cross_encoder_score",
        "cross_encoder_score_norm",
        "cross_encoder_rank",
        "pre_rerank_score",
        "pre_rerank_rank",
        "pre_rerank_source",
        "retrieval_score_norm",
        "final_combined_score",
        "final_rerank_rank",
        "rerank_model_name",
        "rerank_method_name",
        "cross_encoder_available",
        "cross_encoder_error",
        "cross_encoder_weight",
        "retrieval_weight",
        "rerank_preserved_by_backbone",
        "rerank_preserve_reason",
        "rerank_preserve_top_k",
        "rerank_preserve_at_top",
        "rerank_preserve_at_top_qualified",
        "rerank_preserve_at_top_max_ce_rank",
        "rerank_preserve_not_forced_reason",
        "evidence_expansion",
        "evidence_expansion_reasons",
        "evidence_expansion_intent",
    ]
    return {k: jsonable(meta.get(k)) for k in keys if k in meta}


def chunk_trace_rows(
    *,
    question_id: str,
    method: str,
    stage: str,
    chunks: list[Any],
    expected_terms: list[str],
    expected_sources: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, chunk in enumerate(chunks[:top_n], start=1):
        rows.append({
            "question_id": question_id,
            "method": method,
            "stage": stage,
            "rank": rank,
            "chunk_id": getv(chunk, "chunk_id", ""),
            "score": score(getv(chunk, "score", 0.0)),
            "source": getv(chunk, "source", ""),
            "article_id": getv(chunk, "article_id", ""),
            "title": getv(chunk, "title", ""),
            "section": getv(chunk, "section", ""),
            "subsection": getv(chunk, "subsection", ""),
            "source_url": getv(chunk, "source_url", ""),
            "source_slug": url_slug(getv(chunk, "source_url", "")),
            "term_hits": matched_expected_terms(chunk, expected_terms),
            "source_hits": source_hit(chunk, expected_sources),
            "score_breakdown": chunk_score_breakdown(chunk),
            "preview": short(getv(chunk, "text", "") or getv(chunk, "contextualized_text", ""), 500),
        })
    return rows


def append_chunk_stage_trace(
    path: Path,
    *,
    question_id: str,
    method: str,
    stage: str,
    chunks: list[Any],
    expected_terms: list[str],
    expected_sources: list[str],
    top_n: int,
) -> None:
    for row in chunk_trace_rows(
        question_id=question_id,
        method=method,
        stage=stage,
        chunks=chunks,
        expected_terms=expected_terms,
        expected_sources=expected_sources,
        top_n=top_n,
    ):
        append_jsonl(path, row)


def entity_trace(entity: Any, rank: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "rank": rank,
        "entity_id": getv(entity, "entity_id", ""),
        "name": getv(entity, "name", ""),
        "normalized_name": getv(entity, "normalized_name", ""),
        "entity_type": getv(entity, "entity_type", ""),
        "score": score(getv(entity, "score", 0.0)),
        "source": getv(entity, "source", ""),
        "metadata": jsonable(getv(entity, "metadata", {}) or {}),
    }
    if extra:
        row.update(extra)
    return row


def relation_trace(relation: Any, rank: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "rank": rank,
        "relation_id": getv(relation, "relation_id", ""),
        "relation_type": getv(relation, "relation_type", ""),
        "subject_entity_id": getv(relation, "subject_entity_id", ""),
        "subject_name": getv(relation, "subject_name", ""),
        "subject_type": getv(relation, "subject_type", ""),
        "object_entity_id": getv(relation, "object_entity_id", ""),
        "object_name": getv(relation, "object_name", ""),
        "object_type": getv(relation, "object_type", ""),
        "score": score(getv(relation, "score", 0.0)),
        "source_url": getv(relation, "source_url", ""),
        "evidence_chunk_ids": jsonable(getv(relation, "evidence_chunk_ids", []) or []),
        "metadata": jsonable(getv(relation, "metadata", {}) or {}),
    }
    if extra:
        row.update(extra)
    return row

def compact_error(method: str, exc: BaseException) -> dict[str, Any]:
    return {
        "error": True,
        "method": method,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def mem() -> str:
    try:
        import psutil  # type: ignore
        import os as _os
        return f"{psutil.Process(_os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB"
    except Exception:
        return "N/A"


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("w", encoding="utf-8", buffering=1)

    def p(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    @contextmanager
    def step(self, name: str):
        t0 = time.perf_counter()
        self.p(f"\n[STEP_BEGIN] {name} | mem={mem()}")
        try:
            yield
        except BaseException as exc:
            self.p(f"[STEP_ERROR] {name}: {type(exc).__name__}: {exc}")
            self.p(traceback.format_exc())
            raise
        finally:
            self.p(f"[STEP_END] {name} | elapsed_ms={(time.perf_counter()-t0)*1000:.2f} | mem={mem()}")

    def close(self):
        self.fh.close()


# =========================
# Matching / ranking analysis
# =========================

def chunk_text_blob(chunk: Any) -> str:
    parts = [
        getv(chunk, "title", ""),
        getv(chunk, "section", ""),
        getv(chunk, "subsection", ""),
        getv(chunk, "text", ""),
        getv(chunk, "contextualized_text", ""),
    ]
    return "\n".join(str(p or "") for p in parts)


def matched_expected_terms(chunk: Any, expected_terms: list[str]) -> list[str]:
    cached = getv(chunk, "term_hits", None)
    if cached is not None:
        return list(cached)
    hay = norm_text(chunk_text_blob(chunk))
    matched = []
    for term in expected_terms:
        t = norm_text(term)
        if t and t in hay:
            matched.append(term)
    return matched


def chunk_source_candidates(chunk: Any) -> list[str]:
    values: list[str] = []
    meta = dict(getv(chunk, "metadata", {}) or {})

    def add(v: Any):
        if v is None:
            return
        if isinstance(v, (list, tuple, set)):
            for item in v:
                add(item)
            return
        s = str(v).strip()
        if s and s not in values:
            values.append(s)

    add(getv(chunk, "source_url", ""))
    for k in ["source_url", "url", "article_url", "canonical_url", "source_link", "source_links", "source"]:
        add(meta.get(k))
    return values


def source_hit(chunk: Any, expected_sources: list[str]) -> list[str]:
    cached = getv(chunk, "source_hits", None)
    if cached is not None:
        return list(cached)
    candidates = chunk_source_candidates(chunk)
    hits = []
    for expected in expected_sources:
        if any(urls_match(c, expected) for c in candidates):
            hits.append(expected)
    return hits


def first_hit_rank(chunks: list[Any], expected_terms: list[str], expected_sources: list[str]) -> dict[str, Any]:
    first_term_rank = None
    first_source_rank = None
    all_terms = set()
    all_sources = set()

    for i, c in enumerate(chunks, start=1):
        mt = matched_expected_terms(c, expected_terms)
        ms = source_hit(c, expected_sources)
        all_terms.update(mt)
        all_sources.update(ms)
        if mt and first_term_rank is None:
            first_term_rank = i
        if ms and first_source_rank is None:
            first_source_rank = i

    return {
        "term_hit_rank": first_term_rank,
        "source_hit_rank": first_source_rank,
        "term_hit_at_5": first_term_rank is not None and first_term_rank <= 5,
        "source_hit_at_5": first_source_rank is not None and first_source_rank <= 5,
        "matched_terms": sorted(all_terms),
        "matched_sources": sorted(all_sources),
    }


def article_distribution(chunks: list[Any]) -> list[dict[str, Any]]:
    acc: dict[str, dict[str, Any]] = {}
    for i, c in enumerate(chunks, start=1):
        key = norm_url(getv(c, "source_url", "")) or str(getv(c, "article_id", "") or "")
        if key not in acc:
            acc[key] = {
                "source_key": key,
                "title": getv(c, "title", ""),
                "url": getv(c, "source_url", ""),
                "count": 0,
                "best_rank": i,
                "best_score": score(getattr(c, "score", 0)),
            }
        acc[key]["count"] += 1
        acc[key]["best_rank"] = min(acc[key]["best_rank"], i)
        acc[key]["best_score"] = max(acc[key]["best_score"], score(getattr(c, "score", 0)))
    return sorted(acc.values(), key=lambda x: (x["best_rank"], -x["count"]))


def branch_sources_from_mixed(chunk: Any) -> list[str]:
    meta = dict(getv(chunk, "metadata", {}) or {})
    src = meta.get("mixed_sources") or []
    if isinstance(src, str):
        return [src]
    return [str(x) for x in src]


# =========================
# Rendering report
# =========================

def render_chunk_table(chunks: list[Any], expected_terms: list[str], expected_sources: list[str], max_rows: int) -> str:
    rows = [
        "| rank | score | ce trace | source | title | source hit | term hits | section | preview |",
        "|---:|---:|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(chunks[:max_rows], start=1):
        mt = matched_expected_terms(c, expected_terms)
        ms = source_hit(c, expected_sources)
        source = getv(c, "source", "")
        mixed = branch_sources_from_mixed(c)
        if mixed:
            source += f" ({','.join(mixed)})"
        title = short(getattr(c, "title", ""), 70).replace("|", "\\|")
        section = short(" / ".join(x for x in [getattr(c, "section", ""), getattr(c, "subsection", "")] if x), 60).replace("|", "\\|")
        preview = short(getattr(c, "text", "") or getattr(c, "contextualized_text", ""), 160).replace("|", "\\|")
        ce_trace = ce_score_line(c).replace("|", "\\|") or "—"
        rows.append(
            f"| {i} | {score(getv(c, 'score', 0)):.4f} | {ce_trace} | {source} | {title} | "
            f"{'✅' if ms else '—'} | {', '.join(mt) if mt else '—'} | {section} | {preview} |"
        )
    return "\n".join(rows)


def explain_why_vector_wins(method_rows: dict[str, dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    vec = method_rows.get("vector_naive")
    if not vec:
        return ["Không có kết quả vector_naive để so sánh."]

    v_sum = vec["summary"]
    for method in ["lightrag", "hipporag", "mixed"]:
        cur = method_rows.get(method)
        if not cur:
            continue
        s = cur["summary"]
        prefix = f"**{method}**:"
        reasons = []

        if v_sum["source_hit_rank"] and (not s["source_hit_rank"] or v_sum["source_hit_rank"] < s["source_hit_rank"]):
            reasons.append(
                f"Vector tìm đúng nguồn sớm hơn: rank {v_sum['source_hit_rank']} vs {s['source_hit_rank'] or 'miss'}."
            )
        if v_sum["term_hit_rank"] and (not s["term_hit_rank"] or v_sum["term_hit_rank"] < s["term_hit_rank"]):
            reasons.append(
                f"Vector chạm expected terms sớm hơn: rank {v_sum['term_hit_rank']} vs {s['term_hit_rank'] or 'miss'}."
            )

        v_articles = [a["source_key"] for a in vec["article_distribution"][:3]]
        g_articles = [a["source_key"] for a in cur["article_distribution"][:3]]
        if v_articles and g_articles and v_articles[0] != g_articles[0]:
            reasons.append(
                f"Top article lệch: vector top=`{v_articles[0]}`, {method} top=`{g_articles[0]}`."
            )

        if method == "mixed":
            # mixed can be pulled by bad graph branches
            top_chunks = cur.get("chunks", [])
            graph_pulled = [
                (i + 1, branch_sources_from_mixed(c), getv(c, "title", ""))
                for i, c in enumerate(top_chunks[:5])
                if any(src in {"lightrag", "hipporag"} for src in branch_sources_from_mixed(c))
                and not source_hit(c, vec["expected_sources"])
            ]
            if graph_pulled:
                r, srcs, title = graph_pulled[0]
                reasons.append(
                    f"Mixed có chunk top-{r} từ nhánh graph `{srcs}` nhưng không khớp expected source: `{short(title, 80)}`."
                )

        if not reasons:
            reasons.append("Không thấy vector thắng rõ bằng term/source rank; khác biệt có thể nằm ở answer generation hoặc faithfulness.")
        notes.append(prefix + " " + " ".join(reasons))
    if "vector_ce" in method_rows:
        ce = method_rows["vector_ce"]["summary"]
        reasons = []
        if ce["source_hit_rank"] and v_sum["source_hit_rank"]:
            if ce["source_hit_rank"] < v_sum["source_hit_rank"]:
                reasons.append(f"vector_ce cải thiện source rank: {v_sum['source_hit_rank']} → {ce['source_hit_rank']}.")
            elif ce["source_hit_rank"] > v_sum["source_hit_rank"]:
                reasons.append(f"vector_ce làm tụt source rank: {v_sum['source_hit_rank']} → {ce['source_hit_rank']}; cần xem rerank_trace.jsonl.")
        elif ce["source_hit_rank"] and not v_sum["source_hit_rank"]:
            reasons.append(f"vector_ce cứu được source miss của vector_naive ở rank {ce['source_hit_rank']}.")
        elif not ce["source_hit_rank"] and v_sum["source_hit_rank"]:
            reasons.append(f"vector_ce làm mất expected source mà vector_naive có ở rank {v_sum['source_hit_rank']}.")
        if reasons:
            notes.append("**vector_ce**: " + " ".join(reasons))

    if "mixed_ce" in method_rows:
        ce = method_rows["mixed_ce"]["summary"]
        base = method_rows.get("mixed", {}).get("summary", {})
        reasons = []
        if base:
            if ce["source_hit_rank"] and base.get("source_hit_rank"):
                if ce["source_hit_rank"] < base["source_hit_rank"]:
                    reasons.append(f"mixed_ce cải thiện source rank so với mixed: {base['source_hit_rank']} → {ce['source_hit_rank']}.")
                elif ce["source_hit_rank"] > base["source_hit_rank"]:
                    reasons.append(f"mixed_ce làm tụt source rank so với mixed: {base['source_hit_rank']} → {ce['source_hit_rank']}.")
            elif ce["source_hit_rank"] and not base.get("source_hit_rank"):
                reasons.append(f"mixed_ce cứu được source miss của mixed ở rank {ce['source_hit_rank']}.")
            elif not ce["source_hit_rank"] and base.get("source_hit_rank"):
                reasons.append(f"mixed_ce làm mất expected source mà mixed có ở rank {base['source_hit_rank']}.")
        if reasons:
            notes.append("**mixed_ce**: " + " ".join(reasons))

    return notes


def render_method_section(method: str, data: dict[str, Any], top_n: int) -> str:
    s = data["summary"]
    parts = []
    parts.append(f"### {method}")
    parts.append(
        f"- term_hit_rank: `{s['term_hit_rank']}` | source_hit_rank: `{s['source_hit_rank']}` | "
        f"term@5: `{s['term_hit_at_5']}` | source@5: `{s['source_hit_at_5']}`"
    )
    parts.append(f"- matched_terms: `{s['matched_terms']}`")
    parts.append(f"- matched_sources: `{s['matched_sources']}`")
    rerank_debug = data.get("rerank_debug") or {}
    if rerank_debug:
        parts.append(
            "- rerank: "
            f"model=`{rerank_debug.get('model_name')}` | available=`{rerank_debug.get('available')}` | "
            f"fallback=`{rerank_debug.get('fallback')}` | candidates=`{rerank_debug.get('candidate_count')}` | "
            f"weights=CE:{rerank_debug.get('cross_encoder_weight')}/RET:{rerank_debug.get('retrieval_weight')}`"
        )
    parts.append("- article distribution:")
    for a in data["article_distribution"][:6]:
        parts.append(
            f"  - rank {a['best_rank']}, count={a['count']}, score={a['best_score']:.4f}: "
            f"{a['title']} — `{a['url']}`"
        )
    parts.append("")
    parts.append(render_chunk_table(data["chunks"], data["expected_terms"], data["expected_sources"], top_n))
    return "\n".join(parts)


# =========================
# Deep probes
# =========================

def run_method_retrieve(log: Logger, retriever: Any, method: str, query: str, chunk_top_k: int) -> tuple[Any | None, dict[str, Any] | None]:
    try:
        with log.step(f"{method}.retrieve"):
            if method == "vector_naive":
                return retriever.retrieve(query, top_k=chunk_top_k), None
            return retriever.retrieve(query), None
    except Exception as exc:
        err = compact_error(method, exc)
        log.p(f"[METHOD_ERROR] {method}: {err['exception_type']}: {err['message']}")
        # Keep the script alive; the error is persisted to method_rows.jsonl.
        return None, err


def safe_probe(log: Logger, name: str, fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return fn(log, *args, **kwargs)
    except Exception as exc:
        err = compact_error(name, exc)
        log.p(f"[PROBE_ERROR] {name}: {err['exception_type']}: {err['message']}")
        return err


def probe_lightrag_internals(log: Logger, light: Any, vector: Any, query: str, top_n: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with log.step("lightrag.keyword_extract/analyze"):
        raw_kw = light.keyword_extractor.extract(query)
        kw = light._clean_keywords(raw_kw) if hasattr(light, "_clean_keywords") else raw_kw
        analysis = light.query_analyzer.analyze(query, kw)
        rel_types = light._infer_relation_types(query, kw, analysis=analysis)
        out["raw_keywords"] = jsonable(raw_kw)
        out["clean_keywords"] = jsonable(kw)
        out["analysis"] = jsonable(analysis)
        out["allowed_relation_types"] = rel_types
        log.p(json.dumps(out, ensure_ascii=False, indent=2, default=str))

    low_query = kw.low_query_text() or query
    high_query = kw.high_query_text() or query
    intent = getattr(analysis, "intent", "")
    if intent == "symptom_diagnosis":
        high_query = "; ".join(x for x in [kw.high_query_text(), "gợi ý bệnh từ triệu chứng", "biểu hiện lâm sàng", "yếu tố liên quan"] if x)
    elif intent == "treatment_care":
        high_query = "; ".join(x for x in [kw.high_query_text(), "điều trị", "giảm triệu chứng", "chăm sóc da", "dưỡng ẩm", "thuốc bôi", "khi nào cần gặp bác sĩ"] if x)

    out["low_query"] = low_query
    out["high_query"] = high_query

    with log.step("lightrag.raw_vector_entity_hits"):
        ents = vector.search_entities(low_query, top_k=top_n)
        out["raw_entity_hits"] = [jsonable(e) for e in ents]
        for i, e in enumerate(ents[:top_n], start=1):
            log.p(f"{i:02d}. {getattr(e,'name','')} ({getattr(e,'entity_type','')}) score={score(getattr(e,'score',0)):.4f}")

    with log.step("lightrag.raw_vector_relation_hits"):
        rels = vector.search_relations(high_query, top_k=top_n)
        out["raw_relation_hits"] = [jsonable(r) for r in rels]
        for i, r in enumerate(rels[:top_n], start=1):
            log.p(
                f"{i:02d}. {getattr(r,'subject_name','')} -[{getattr(r,'relation_type','')}]-> "
                f"{getattr(r,'object_name','')} score={score(getattr(r,'score',0)):.4f} url={getattr(r,'source_url','')}"
            )

    return out


def probe_hippo_internals(log: Logger, hippo: Any, vector: Any, query: str, qid: str, expected_terms: list[str], expected_sources: list[str], top_n: int) -> dict[str, Any]:
    out: dict[str, Any] = {}

    with log.step("hippo.keyword_extract/analyze/seed_terms"):
        kw = hippo.keyword_extractor.extract(query)
        analysis = hippo.query_analyzer.analyze(query, kw)
        seed_query = hippo._build_seed_query(query, kw)
        seed_terms = hippo._seed_terms(query=query, seed_query=seed_query, keywords=kw)
        out["keywords"] = jsonable(kw)
        out["analysis"] = jsonable(analysis)
        out["seed_query"] = seed_query
        out["seed_terms"] = seed_terms
        log.p(json.dumps({k: out[k] for k in ["keywords", "analysis", "seed_query", "seed_terms"]}, ensure_ascii=False, indent=2, default=str))

    lexical = {}
    for term in seed_terms[: getattr(hippo.config, "max_seed_terms", 12)]:
        with log.step(f"hippo.lexical_seed term={term!r}"):
            hits = hippo.graph_repo.find_seed_entities(query_text=term, limit=max(top_n, hippo.config.seed_top_k))
            lexical[term] = [jsonable(e) for e in hits]
            for i, e in enumerate(hits[:top_n], start=1):
                log.p(f"{i:02d}. {getattr(e,'name','')} ({getattr(e,'entity_type','')}) score={score(getattr(e,'score',0)):.4f}")
    out["lexical_seed_hits"] = lexical

    with log.step("hippo.vector_seed_hits"):
        vector_seed_hits = {}
        for vq in list(dict.fromkeys([seed_query, query])):
            hits = vector.search_entities(vq, top_k=max(top_n, hippo.config.vector_seed_top_k))
            vector_seed_hits[vq] = [jsonable(e) for e in hits]
            log.p(f"\nvector seed query={vq!r}")
            for i, e in enumerate(hits[:top_n], start=1):
                log.p(f"{i:02d}. {getattr(e,'name','')} ({getattr(e,'entity_type','')}) score={score(getattr(e,'score',0)):.4f}")
        out["vector_seed_hits"] = vector_seed_hits

    with log.step("hippo.collect_seed_scores"):
        raw_seed = hippo._collect_seed_scores(query=query, seed_query=seed_query, keywords=kw)
        seed_scores, node_spec = hippo._apply_node_specificity(dict(raw_seed))
        out["raw_seed_scores"] = raw_seed
        out["seed_scores_after_specificity"] = seed_scores
        out["node_specificity"] = node_spec
        log.p("raw_seed_scores=" + json.dumps(raw_seed, ensure_ascii=False, indent=2, default=str))
        log.p("seed_scores_after_specificity=" + json.dumps(seed_scores, ensure_ascii=False, indent=2, default=str))

    with log.step("hippo.adjacency_and_ppr"):
        seed_ids = list(seed_scores.keys())
        adjacency = hippo.graph_repo.get_entity_adjacency_around_seeds(
            seed_entity_ids=seed_ids,
            max_hops=hippo.config.adjacency_hops,
            limit=hippo.config.adjacency_limit,
        )
        if not adjacency:
            adjacency = hippo.graph_repo.get_entity_adjacency(limit=hippo.config.adjacency_limit)
        rel_counts = dict(Counter(getattr(e, "relation_type", "") for e in adjacency).most_common())
        ppr_output = hippo._personalized_pagerank(
            seed_scores=seed_scores,
            adjacency=adjacency,
            relation_type_boosts=getattr(analysis, "relation_type_boosts", None) if hippo.config.use_query_aware_edge_weights else None,
        )
        ppr_debug = {}
        if isinstance(ppr_output, tuple):
            ppr_scores = ppr_output[0] if len(ppr_output) >= 1 else {}
            if len(ppr_output) >= 2 and isinstance(ppr_output[1], dict):
                ppr_debug = ppr_output[1]
        else:
            ppr_scores = ppr_output
        if not isinstance(ppr_scores, dict):
            ppr_scores = {}

        ranked_ids = [eid for eid, s in sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True) if s > 0.0][: hippo.config.ppr_top_k_entities]
        entities = hippo._load_entities(ranked_ids, ppr_scores)
        out["adjacency_edge_count"] = len(adjacency)
        out["adjacency_relation_type_counts"] = rel_counts
        out["ppr_debug"] = ppr_debug
        out["top_ppr_entity_ids"] = [{"entity_id": eid, "score": ppr_scores.get(eid), "is_seed": eid in set(seed_ids), "node_specificity": node_spec.get(eid)} for eid in ranked_ids[:top_n]]
        out["top_ppr_entities"] = [entity_trace(e, rank=i) for i, e in enumerate(entities[:top_n], start=1)]
        log.p(f"adjacency_edges={len(adjacency)} relation_counts={json.dumps(rel_counts, ensure_ascii=False)}")
        log.p("\nTop PPR entities:")
        for i, e in enumerate(entities[:top_n], start=1):
            eid = getattr(e, 'entity_id', '')
            log.p(f"{i:02d}. {getattr(e,'name','')} ({getattr(e,'entity_type','')}) score={score(getattr(e,'score',0)):.6f} id={eid} is_seed={eid in set(seed_ids)} spec={node_spec.get(eid)}")

    # Reconstruct Hippo chunk pipeline stage by stage. These calls mirror
    # HippoRAGRetriever.retrieve(), but store compact per-stage traces so we can
    # see exactly where a wrong article overtakes the expected one.
    with log.step("hippo.chunk_pipeline_trace"):
        relations = hippo._load_supporting_relations(ranked_ids, ppr_scores)
        chunks_graph = hippo._load_ranked_chunks(ranked_ids, ppr_scores, relations)
        out["top_supporting_relations"] = [relation_trace(r, rank=i) for i, r in enumerate(relations[:top_n], start=1)]
        out["chunk_stages"] = {
            "after_ppr_chunk_mapping": [compact_chunk(c, expected_terms, expected_sources) for c in chunks_graph[:top_n]],
        }
        log.p("[HIPPO_STAGE] after_ppr_chunk_mapping")
        for row in chunk_trace_rows(question_id=qid, method="hipporag", stage="after_ppr_chunk_mapping", chunks=chunks_graph, expected_terms=expected_terms, expected_sources=expected_sources, top_n=top_n):
            log.p(json.dumps(row, ensure_ascii=False, default=str))

        chunks_current = chunks_graph
        if getattr(hippo.config, "use_passage_integration", False):
            chunks_current = hippo._merge_passage_integrated_chunks(query=query, graph_chunks=chunks_current)
            out["chunk_stages"]["after_passage_integration"] = [compact_chunk(c, expected_terms, expected_sources) for c in chunks_current[:top_n]]
            log.p("[HIPPO_STAGE] after_passage_integration")
            for row in chunk_trace_rows(question_id=qid, method="hipporag", stage="after_passage_integration", chunks=chunks_current, expected_terms=expected_terms, expected_sources=expected_sources, top_n=top_n):
                log.p(json.dumps(row, ensure_ascii=False, default=str))

        if getattr(hippo.config, "enable_chunk_query_rerank", False):
            chunks_current = hippo._rerank_chunks_by_query_evidence(query=query, keywords=kw, analysis=analysis, chunks=chunks_current, relations=relations)
            out["chunk_stages"]["after_query_evidence_rerank"] = [compact_chunk(c, expected_terms, expected_sources) for c in chunks_current[:top_n]]
            log.p("[HIPPO_STAGE] after_query_evidence_rerank")
            for row in chunk_trace_rows(question_id=qid, method="hipporag", stage="after_query_evidence_rerank", chunks=chunks_current, expected_terms=expected_terms, expected_sources=expected_sources, top_n=top_n):
                log.p(json.dumps(row, ensure_ascii=False, default=str))

        if getattr(hippo.config, "enable_evidence_expansion", False):
            chunks_expanded, exp_debug = hippo.evidence_expander.expand_chunks(query=query, chunks=chunks_current, entities=entities, relations=relations, analysis=analysis)
            out["evidence_expansion_debug_reconstructed"] = jsonable(exp_debug)
            out["chunk_stages"]["after_evidence_expansion"] = [compact_chunk(c, expected_terms, expected_sources) for c in chunks_expanded[:top_n]]
            log.p("[HIPPO_STAGE] after_evidence_expansion")
            for row in chunk_trace_rows(question_id=qid, method="hipporag", stage="after_evidence_expansion", chunks=chunks_expanded, expected_terms=expected_terms, expected_sources=expected_sources, top_n=top_n):
                log.p(json.dumps(row, ensure_ascii=False, default=str))
    return out




def append_stage_snapshots_from_debug(
    path: Path,
    *,
    question_id: str,
    method: str,
    result_debug: dict[str, Any],
    expected_terms: list[str],
    expected_sources: list[str],
    top_n: int,
) -> None:
    """Persist retriever-produced per-stage chunk snapshots to pipeline_trace.jsonl.

    HippoRAG now exposes debug["chunk_stages"] from the actual retrieve() path,
    not only from reconstructed probes. This lets us see whether the final run
    diverges at PPR mapping, passage integration, query rerank, or expansion.
    """
    stages = (result_debug or {}).get("chunk_stages") or {}
    if not isinstance(stages, dict):
        return
    for stage, chunks in stages.items():
        if not isinstance(chunks, list):
            continue
        append_chunk_stage_trace(
            path,
            question_id=question_id,
            method=method,
            stage=str(stage),
            chunks=chunks,
            expected_terms=expected_terms,
            expected_sources=expected_sources,
            top_n=top_n,
        )



def compact_rerank_debug(result_debug: dict[str, Any]) -> dict[str, Any]:
    """Extract the cross-encoder rerank block from a RetrievalResult debug dict.

    For vector_ce/mixed_ce the patched retriever stores rerank info at
    debug["rerank"]. This compact copy is duplicated into method_rows.jsonl so
    we can inspect the effective model/config without opening large nested debug.
    """
    rerank = (result_debug or {}).get("rerank") or {}
    if not isinstance(rerank, dict):
        return {}
    keep = [
        "enabled",
        "available",
        "fallback",
        "reason",
        "method_name",
        "model_name",
        "candidate_count",
        "returned_count",
        "candidate_top_k",
        "final_top_k",
        "batch_size",
        "max_text_chars",
        "cross_encoder_weight",
        "retrieval_weight",
        "score_calibration",
        "source_diversity",
        "pre_rerank_preservation",
    ]
    out = {k: jsonable(rerank.get(k)) for k in keep if k in rerank}
    top = rerank.get("top_candidates")
    if isinstance(top, list):
        out["top_candidates"] = jsonable(top[:20])
    return out


def ce_score_line(chunk: Any) -> str:
    meta = dict(getv(chunk, "metadata", {}) or {})
    if "cross_encoder_score" not in meta and "final_combined_score" not in meta:
        return ""
    ce = meta.get("cross_encoder_score")
    ce_norm = meta.get("cross_encoder_score_norm")
    pre = meta.get("pre_rerank_rank")
    final = meta.get("final_combined_score")
    avail = meta.get("cross_encoder_available")
    err = meta.get("cross_encoder_error")
    parts = []
    if pre is not None:
        parts.append(f"pre_rank={pre}")
    if ce is not None:
        parts.append(f"ce={score(ce):.4f}")
    if ce_norm is not None:
        parts.append(f"ce_norm={score(ce_norm):.3f}")
    if final is not None:
        parts.append(f"final={score(final):.4f}")
    if meta.get("rerank_preserved_by_backbone"):
        if meta.get("rerank_preserve_at_top"):
            parts.append("PRESERVED_TOP")
        else:
            parts.append("PRESERVED_WINDOW")
    if avail is False:
        parts.append("CE_FALLBACK")
    if err:
        parts.append(f"err={short(err, 90)}")
    return "; ".join(parts)


def append_rerank_trace(
    path: Path,
    *,
    question_id: str,
    method: str,
    result_debug: dict[str, Any],
    chunks: list[Any],
    expected_terms: list[str],
    expected_sources: list[str],
    top_n: int,
) -> None:
    """Write CE-specific trace rows for vector_ce/mixed_ce.

    This file is intentionally separate from pipeline_trace.jsonl so we can grep
    cross_encoder_score / pre_rerank_rank / final_combined_score quickly.
    """
    rerank = compact_rerank_debug(result_debug)
    if not rerank:
        return
    append_jsonl(path, {
        "question_id": question_id,
        "method": method,
        "event": "rerank_debug",
        "rerank": rerank,
    })
    for rank, chunk in enumerate(chunks[:top_n], start=1):
        meta = dict(getv(chunk, "metadata", {}) or {})
        append_jsonl(path, {
            "question_id": question_id,
            "method": method,
            "event": "final_reranked_chunk",
            "rank": rank,
            "chunk_id": getv(chunk, "chunk_id", ""),
            "title": getv(chunk, "title", ""),
            "section": getv(chunk, "section", ""),
            "source_url": getv(chunk, "source_url", ""),
            "source_hit": source_hit(chunk, expected_sources),
            "term_hits": matched_expected_terms(chunk, expected_terms),
            "score": score(getv(chunk, "score", 0.0)),
            "pre_rerank_rank": meta.get("pre_rerank_rank"),
            "pre_rerank_score": meta.get("pre_rerank_score"),
            "pre_rerank_source": meta.get("pre_rerank_source"),
            "cross_encoder_score": meta.get("cross_encoder_score"),
            "cross_encoder_score_norm": meta.get("cross_encoder_score_norm"),
            "cross_encoder_rank": meta.get("cross_encoder_rank"),
            "retrieval_score_norm": meta.get("retrieval_score_norm"),
            "final_combined_score": meta.get("final_combined_score"),
            "final_rerank_rank": meta.get("final_rerank_rank"),
            "rerank_model_name": meta.get("rerank_model_name"),
            "cross_encoder_available": meta.get("cross_encoder_available"),
            "cross_encoder_error": meta.get("cross_encoder_error"),
            "rerank_preserved_by_backbone": meta.get("rerank_preserved_by_backbone"),
            "rerank_preserve_reason": meta.get("rerank_preserve_reason"),
            "rerank_preserve_top_k": meta.get("rerank_preserve_top_k"),
            "rerank_preserve_at_top": meta.get("rerank_preserve_at_top"),
            "rerank_preserve_at_top_qualified": meta.get("rerank_preserve_at_top_qualified"),
            "rerank_preserve_at_top_max_ce_rank": meta.get("rerank_preserve_at_top_max_ce_rank"),
            "rerank_preserve_not_forced_reason": meta.get("rerank_preserve_not_forced_reason"),
            "preview": short(getv(chunk, "text", "") or getv(chunk, "contextualized_text", ""), 500),
        })

def summarize_result(method: str, result: Any, expected_terms: list[str], expected_sources: list[str], error: dict[str, Any] | None = None) -> dict[str, Any]:
    if error is not None or result is None:
        return {
            "method": method,
            "expected_terms": expected_terms,
            "expected_sources": expected_sources,
            "summary": {
                "term_hit_rank": None,
                "source_hit_rank": None,
                "term_hit_at_5": False,
                "source_hit_at_5": False,
                "matched_terms": [],
                "matched_sources": [],
            },
            "article_distribution": [],
            "chunks": [],
            "result_debug": {},
            "rerank_debug": {},
            "keywords": None,
            "error": error or {"error": True, "method": method, "message": "retrieve_failed"},
        }

    chunks_raw = list(getattr(result, "chunks", []) or [])
    summary = first_hit_rank(chunks_raw, expected_terms, expected_sources)
    result_debug = jsonable(getattr(result, "debug", {}))
    row = {
        "method": method,
        "expected_terms": expected_terms,
        "expected_sources": expected_sources,
        "summary": summary,
        "article_distribution": article_distribution(chunks_raw),
        "chunks": [compact_chunk(c, expected_terms, expected_sources) for c in chunks_raw],
        "result_debug": result_debug,
        "rerank_debug": compact_rerank_debug(result_debug),
        "keywords": jsonable(getattr(result, "keywords", None)),
        "error": None,
    }
    return row


def normalize_question_ids(raw: Any) -> list[str]:
    if not raw:
        return []
    values: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
            return
        for part in str(value).replace(",", " ").split():
            part = part.strip()
            if part:
                values.append(part)

    add(raw)
    return values


def resolve_dataset_path(dataset_file: str) -> Path:
    raw = Path(dataset_file)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def available_benchmark_files(limit: int = 30) -> list[str]:
    bench_dir = PROJECT_ROOT / "data" / "evaluation" / "benchmarks"
    if not bench_dir.exists():
        return []
    files = sorted(bench_dir.glob("*.json"), key=lambda path: path.name.lower())
    return [str(path.relative_to(PROJECT_ROOT)) for path in files[:limit]]


def load_questions(args: argparse.Namespace) -> list[Any]:
    dataset_path = resolve_dataset_path(args.dataset_file)
    if not dataset_path.exists():
        available = available_benchmark_files()
        hint = ""
        if available:
            hint = "\nAvailable benchmark JSON files:\n" + "\n".join(f"  - {item}" for item in available)
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            f"Run PowerShell: Get-ChildItem .\data\evaluation\benchmarks -Filter *.json{hint}"
        )

    dataset = load_evaluation_questions(dataset_path)

    question_ids = normalize_question_ids(args.question_id)
    if question_ids:
        wanted = {x.strip() for x in question_ids if x.strip()}
        selected = []
        for idx, item in enumerate(dataset, start=1):
            meta = dict(item.metadata or {})
            qid = str(meta.get("id") or f"q{idx:03d}")
            aliases = {qid, f"q{idx:03d}", str(idx)}
            if wanted & aliases:
                selected.append((qid, item))
        return selected

    difficulty = (args.difficulty or "").strip().lower()
    selected = []
    for idx, item in enumerate(dataset, start=1):
        if difficulty and str(item.difficulty).strip().lower() != difficulty:
            continue
        meta = dict(item.metadata or {})
        qid = str(meta.get("id") or f"q{idx:03d}")
        selected.append((qid, item))
        if len(selected) >= args.max_questions:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain why vector_naive outperforms graph retrievers on selected hard questions.")
    parser.add_argument("--dataset-file", default="data/evaluation/benchmarks/vietnamese_medical_rag_benchmark_100_fixed.json")
    parser.add_argument(
        "--question-id",
        nargs="+",
        action="append",
        default=None,
        help="Question ids. Examples: --question-id q001 q002 or repeated --question-id q001 --question-id q002.",
    )
    parser.add_argument("--difficulty", default="hard", help="Used when --question-id is not provided.")
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument("--output-dir", default="data/evaluation/why_vector_wins")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["vector_naive", "lightrag", "hipporag", "mixed"],
        choices=["vector_naive", "vector_ce", "lightrag", "hipporag", "mixed", "mixed_ce"],
    )
    parser.add_argument("--no-llm-keywords", action="store_true")
    parser.add_argument("--chunk-top-k", type=int, default=5)
    parser.add_argument("--entity-top-k", type=int, default=5)
    parser.add_argument("--relation-top-k", type=int, default=5)
    parser.add_argument("--hippo-seed-top-k", type=int, default=5)
    parser.add_argument("--hippo-ppr-entities", type=int, default=40)
    parser.add_argument("--hippo-hops", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=5)
    parser.add_argument("--cross-encoder-model", default="", help="Model for vector_ce/mixed_ce. Defaults to CROSS_ENCODER_MODEL or BAAI/bge-reranker-v2-m3.")
    parser.add_argument("--rerank-candidate-top-k", type=int, default=50)
    parser.add_argument("--rerank-final-top-k", type=int, default=0, help="0 means use --max-chunks.")
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-max-text-chars", type=int, default=2400)
    parser.add_argument("--rerank-cross-encoder-weight", type=float, default=0.35)
    parser.add_argument("--rerank-retrieval-weight", type=float, default=0.65)
    parser.add_argument("--rerank-max-chunks-per-source", type=int, default=3)
    parser.add_argument("--rerank-preserve-pre-top-k", type=int, default=2)
    parser.add_argument("--no-rerank-preserve-pre-at-top", action="store_true", help="Keep protected pre-rerank chunks in final window but do not force them to top ranks.")
    parser.add_argument("--rerank-preserve-at-top-max-ce-rank", type=int, default=5, help="Only force preserved chunks to top ranks if CE rank is at most this value. Use 0 to always force.")
    parser.add_argument("--no-rerank-fail-open", action="store_true", help="Raise if the CE model cannot load instead of falling back to pre-rerank order.")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--skip-internals", action="store_true", help="Skip expensive LightRAG/HippoRAG internal probes.")
    parser.add_argument("--continue-on-error", action="store_true", default=True, help="Keep running if one method/probe fails.")
    parser.add_argument("--no-continue-on-error", dest="continue_on_error", action="store_false")
    parser.add_argument("--gc-after-method", action="store_true", default=True)
    parser.add_argument("--gc-after-question", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "why_vector_wins_full.log")
    fatal_fh = (out_dir / "fatal_fault.log").open("w", encoding="utf-8", buffering=1)
    faulthandler.enable(file=fatal_fh, all_threads=True)

    bundle = None
    report_path = out_dir / "why_vector_wins_report.md"
    raw_jsonl_path = out_dir / "why_vector_wins_raw.jsonl"
    method_jsonl_path = out_dir / "method_rows.jsonl"
    trace_jsonl_path = out_dir / "pipeline_trace.jsonl"
    rerank_jsonl_path = out_dir / "rerank_trace.jsonl"
    status_path = out_dir / "run_status.json"

    # Start fresh for this run. Everything below is append/streaming, so a crash
    # still leaves usable partial files on disk.
    report_path.write_text("", encoding="utf-8")
    raw_jsonl_path.write_text("", encoding="utf-8")
    method_jsonl_path.write_text("", encoding="utf-8")
    trace_jsonl_path.write_text("", encoding="utf-8")
    rerank_jsonl_path.write_text("", encoding="utf-8")

    try:
        logger.p("[WHY_VECTOR_WINS] Retrieval-only diagnostic. No answer generation, no judge.")
        logger.p("[LOWMEM] Streaming mode: write after each method/question, no all_rows/md_parts accumulation.")
        logger.p(f"[CONFIG] {json.dumps(vars(args), ensure_ascii=False, indent=2)}")
        load_dotenv(PROJECT_ROOT / ".env")
        effective_ce_model = args.cross_encoder_model or os.getenv("CROSS_ENCODER_MODEL") or "BAAI/bge-reranker-v2-m3"
        logger.p(
            "[CROSS_ENCODER_CONFIG] "
            + json.dumps(
                {
                    "effective_model": effective_ce_model,
                    "cli_model": args.cross_encoder_model,
                    "env_CROSS_ENCODER_MODEL": os.getenv("CROSS_ENCODER_MODEL", ""),
                    "candidate_top_k": args.rerank_candidate_top_k,
                    "final_top_k": args.rerank_final_top_k or args.max_chunks,
                    "batch_size": args.rerank_batch_size,
                    "max_text_chars": args.rerank_max_text_chars,
                    "cross_encoder_weight": args.rerank_cross_encoder_weight,
                    "retrieval_weight": args.rerank_retrieval_weight,
                    "max_chunks_per_source": args.rerank_max_chunks_per_source,
                    "preserve_pre_rerank_top_k": args.rerank_preserve_pre_top_k,
                    "preserve_pre_rerank_at_top": not args.no_rerank_preserve_pre_at_top,
                    "preserve_at_top_max_cross_encoder_rank": args.rerank_preserve_at_top_max_ce_rank,
                    "fail_open": not args.no_rerank_fail_open,
                },
                ensure_ascii=False,
            )
        )

        with logger.step("load questions"):
            questions = load_questions(args)
            logger.p(f"[DATASET] {resolve_dataset_path(args.dataset_file)}")
            logger.p(f"[QUESTIONS] selected={len(questions)}")
            if not questions:
                requested_ids = normalize_question_ids(args.question_id)
                logger.p(f"[WARN] No questions selected. requested_ids={requested_ids} difficulty={args.difficulty!r}")

        with logger.step("build retriever bundle"):
            bundle = build_retriever_bundle(
                RetrieverBuildConfig(
                    no_llm_keywords=args.no_llm_keywords,
                    chunk_top_k=args.chunk_top_k,
                    entity_top_k=args.entity_top_k,
                    relation_top_k=args.relation_top_k,
                    hippo_seed_top_k=args.hippo_seed_top_k,
                    hippo_ppr_entities=args.hippo_ppr_entities,
                    hippo_hops=args.hippo_hops,
                    max_chunks=args.max_chunks,
                    cross_encoder_model=args.cross_encoder_model,
                    rerank_candidate_top_k=args.rerank_candidate_top_k,
                    rerank_final_top_k=args.rerank_final_top_k,
                    rerank_batch_size=args.rerank_batch_size,
                    rerank_max_text_chars=args.rerank_max_text_chars,
                    rerank_cross_encoder_weight=args.rerank_cross_encoder_weight,
                    rerank_retrieval_weight=args.rerank_retrieval_weight,
                    rerank_max_chunks_per_source=args.rerank_max_chunks_per_source,
                    rerank_fail_open=not args.no_rerank_fail_open,
                    rerank_preserve_pre_top_k=args.rerank_preserve_pre_top_k,
                    rerank_preserve_pre_at_top=not args.no_rerank_preserve_pre_at_top,
                    rerank_preserve_at_top_max_ce_rank=args.rerank_preserve_at_top_max_ce_rank,
                )
            )

        retrievers = bundle.retrievers

        header = [
            "# Why vector_naive is outperforming graph retrievers",
            "",
            "Diagnostic này chỉ phân tích retrieval, không gọi answer generator/judge.",
            "",
            "Script đang chạy ở chế độ low-memory streaming: kết quả được ghi dần sau từng method/câu.",
            "",
            "## Config",
            "",
            "```json",
            json.dumps(vars(args), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Cross-encoder rerank config",
            "",
            f"- effective_model: `{args.cross_encoder_model or os.getenv('CROSS_ENCODER_MODEL') or 'BAAI/bge-reranker-v2-m3'}`",
            f"- candidate_top_k: `{args.rerank_candidate_top_k}` | final_top_k: `{args.rerank_final_top_k or args.max_chunks}` | batch_size: `{args.rerank_batch_size}`",
            f"- weights: CE=`{args.rerank_cross_encoder_weight}`, retrieval=`{args.rerank_retrieval_weight}` | max_chunks_per_source: `{args.rerank_max_chunks_per_source}`",
            f"- preserve_pre_rerank_top_k: `{args.rerank_preserve_pre_top_k}` | preserve_at_top: `{not args.no_rerank_preserve_pre_at_top}` | preserve_at_top_max_ce_rank: `{args.rerank_preserve_at_top_max_ce_rank}`",
            f"- fail_open: `{not args.no_rerank_fail_open}`",
            "",
        ]
        report_path.write_text("\n".join(header), encoding="utf-8")

        completed_questions = 0
        method_errors = 0
        probe_errors = 0

        for q_idx, (qid, item) in enumerate(questions, start=1):
            query = item.question
            expected_terms = list(item.expected_terms or [])
            expected_sources = list(item.source_links or [])

            logger.p("\n" + "#" * 140)
            logger.p(f"QUESTION {q_idx}/{len(questions)} id={qid} difficulty={item.difficulty}")
            logger.p(query)
            logger.p(f"expected_terms={expected_terms}")
            logger.p(f"expected_sources={expected_sources}")

            q_json: dict[str, Any] = {
                "id": qid,
                "difficulty": item.difficulty,
                "question": query,
                "expected_terms": expected_terms,
                "expected_sources": expected_sources,
                "methods": {},
                "internals": {},
            }
            method_rows: dict[str, dict[str, Any]] = {}

            for method in args.methods:
                retriever = retrievers[method]
                result, error = run_method_retrieve(logger, retriever, method, query, args.chunk_top_k)
                row = summarize_result(method, result, expected_terms, expected_sources, error=error)
                if error:
                    method_errors += 1
                    if not args.continue_on_error:
                        raise RuntimeError(f"{method} failed for {qid}: {error.get('message')}")

                method_rows[method] = row
                q_json["methods"][method] = {
                    "summary": row["summary"],
                    "article_distribution": row["article_distribution"],
                    "keywords": row["keywords"],
                    "debug": row["result_debug"],
                    "rerank_debug": row.get("rerank_debug", {}),
                    "chunks": row["chunks"][: args.top_n],
                    "error": row.get("error"),
                }

                append_jsonl(method_jsonl_path, {
                    "question_id": qid,
                    "difficulty": item.difficulty,
                    "question": query,
                    "method": method,
                    **q_json["methods"][method],
                })

                append_chunk_stage_trace(
                    trace_jsonl_path,
                    question_id=qid,
                    method=method,
                    stage="final_retrieval_result",
                    chunks=row.get("chunks", []),
                    expected_terms=expected_terms,
                    expected_sources=expected_sources,
                    top_n=args.top_n,
                )
                append_stage_snapshots_from_debug(
                    trace_jsonl_path,
                    question_id=qid,
                    method=method,
                    result_debug=row.get("result_debug", {}),
                    expected_terms=expected_terms,
                    expected_sources=expected_sources,
                    top_n=args.top_n,
                )
                append_rerank_trace(
                    rerank_jsonl_path,
                    question_id=qid,
                    method=method,
                    result_debug=row.get("result_debug", {}),
                    chunks=row.get("chunks", []),
                    expected_terms=expected_terms,
                    expected_sources=expected_sources,
                    top_n=args.top_n,
                )
                if row.get("rerank_debug"):
                    logger.p("[RERANK_DEBUG] " + json.dumps({"question_id": qid, "method": method, **row["rerank_debug"]}, ensure_ascii=False, default=str))

                # Drop heavy result objects immediately before the next method, especially before mixed.
                del result
                if args.gc_after_method:
                    gc.collect()
                    logger.p(f"[GC] after method={method} qid={qid} mem={mem()}")

            if not args.skip_internals:
                if "lightrag" in args.methods:
                    li = safe_probe(
                        logger,
                        "lightrag",
                        probe_lightrag_internals,
                        retrievers["lightrag"],
                        retrievers["vector_naive"],
                        query,
                        args.top_n,
                    )
                    q_json["internals"]["lightrag"] = li
                    if li.get("error"):
                        probe_errors += 1
                if "hipporag" in args.methods:
                    hi = safe_probe(
                        logger,
                        "hipporag",
                        probe_hippo_internals,
                        retrievers["hipporag"],
                        retrievers["vector_naive"],
                        query,
                        qid,
                        expected_terms,
                        expected_sources,
                        args.top_n,
                    )
                    q_json["internals"]["hipporag"] = hi
                    if hi.get("error"):
                        probe_errors += 1
                    else:
                        append_stage_snapshots_from_debug(
                            trace_jsonl_path,
                            question_id=qid,
                            method="hipporag_probe",
                            result_debug={"chunk_stages": hi.get("chunk_stages", {})},
                            expected_terms=expected_terms,
                            expected_sources=expected_sources,
                            top_n=args.top_n,
                        )

            q_md: list[str] = []
            q_md.append(f"\n---\n\n## Question {q_idx}: `{qid}` ({item.difficulty})\n")
            q_md.append(f"**Query:** {query}\n")
            q_md.append(f"**Expected terms:** `{expected_terms}`\n")
            q_md.append(f"**Expected sources:** `{expected_sources}`\n")
            q_md.append("## Quick diagnosis\n")
            for note in explain_why_vector_wins(method_rows):
                q_md.append(f"- {note}")
            q_md.append("")
            q_md.append("## Per-method retrieval summary\n")
            for method in args.methods:
                q_md.append(render_method_section(method, method_rows[method], args.top_n))
                if method_rows[method].get("error"):
                    err = method_rows[method]["error"]
                    q_md.append(f"\n**ERROR:** `{err.get('exception_type')}: {err.get('message')}`\n")
                q_md.append("")

            if "lightrag" in q_json["internals"]:
                li = q_json["internals"]["lightrag"]
                q_md.append("## LightRAG internals\n")
                if li.get("error"):
                    q_md.append(f"- ERROR: `{li.get('exception_type')}: {li.get('message')}`")
                else:
                    q_md.append(f"- raw_keywords: `{li.get('raw_keywords')}`")
                    q_md.append(f"- clean_keywords: `{li.get('clean_keywords')}`")
                    q_md.append(f"- analysis: `{li.get('analysis')}`")
                    q_md.append(f"- low_query: `{li.get('low_query')}`")
                    q_md.append(f"- high_query: `{li.get('high_query')}`")
                    q_md.append("- raw entity hits:")
                    for i, e in enumerate(li.get("raw_entity_hits", [])[: args.top_n], start=1):
                        q_md.append(f"  - {i}. {e.get('name')} ({e.get('entity_type')}) score={score(e.get('score')):.4f}")
                    q_md.append("- raw relation hits:")
                    for i, r in enumerate(li.get("raw_relation_hits", [])[: args.top_n], start=1):
                        q_md.append(
                            f"  - {i}. {r.get('subject_name')} -[{r.get('relation_type')}]-> {r.get('object_name')} "
                            f"score={score(r.get('score')):.4f} source={r.get('source_url')}"
                        )
                q_md.append("")

            if "hipporag" in q_json["internals"]:
                hi = q_json["internals"]["hipporag"]
                q_md.append("## HippoRAG internals\n")
                if hi.get("error"):
                    q_md.append(f"- ERROR: `{hi.get('exception_type')}: {hi.get('message')}`")
                else:
                    q_md.append(f"- keywords: `{hi.get('keywords')}`")
                    q_md.append(f"- analysis: `{hi.get('analysis')}`")
                    q_md.append(f"- seed_query: `{hi.get('seed_query')}`")
                    q_md.append(f"- seed_terms: `{hi.get('seed_terms')}`")
                    q_md.append(f"- adjacency_edge_count: `{hi.get('adjacency_edge_count')}`")
                    q_md.append(f"- adjacency_relation_type_counts: `{hi.get('adjacency_relation_type_counts')}`")
                    q_md.append("- top PPR entities:")
                    for i, e in enumerate(hi.get("top_ppr_entities", [])[: args.top_n], start=1):
                        q_md.append(f"  - {i}. {e.get('name')} ({e.get('entity_type')}) score={score(e.get('score')):.6f}")
                q_md.append("")

            with report_path.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(q_md) + "\n")
                fh.flush()

            write_json(out_dir / f"{qid}_diagnostic.json", q_json)
            append_jsonl(raw_jsonl_path, q_json)
            completed_questions += 1
            write_json(status_path, {
                "completed_questions": completed_questions,
                "total_questions": len(questions),
                "method_errors": method_errors,
                "probe_errors": probe_errors,
                "last_question_id": qid,
                "report_path": str(report_path),
                "raw_jsonl_path": str(raw_jsonl_path),
                "method_jsonl_path": str(method_jsonl_path),
                "trace_jsonl_path": str(trace_jsonl_path),
                "rerank_jsonl_path": str(rerank_jsonl_path),
            })

            logger.p(f"[QUESTION_DONE] id={qid}")
            del q_json, method_rows, q_md
            if args.gc_after_question:
                gc.collect()
                logger.p(f"[GC] after question={qid} mem={mem()}")

        logger.p(f"[SAVED] {report_path}")
        logger.p(f"[SAVED] {raw_jsonl_path}")
        logger.p(f"[SAVED] {method_jsonl_path}")
        logger.p(f"[SAVED] {trace_jsonl_path}")
        logger.p(f"[SAVED] {rerank_jsonl_path}")
        logger.p(f"[SAVED] {logger.path}")

    finally:
        if bundle is not None:
            with logger.step("bundle.close"):
                bundle.close()
        faulthandler.disable()
        fatal_fh.close()
        logger.close()


if __name__ == "__main__":
    main()