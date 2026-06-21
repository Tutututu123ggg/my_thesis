from __future__ import annotations

import asyncio
import csv
import dataclasses
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from app.evaluation.judge import RagasStyleEvaluator, normalize_text
from app.infrastructure.vector_database import NAIVE_CHUNK_COLLECTION
from app.evaluation.models import EvaluationConfig, EvaluationQuestion, EvaluationRow, RetrievalMetricResult
from app.reasoning import MedicalAnswerGenerator
from app.reasoning.answer_generator import build_clean_chunk_context
from app.retrieval import RetrievalResult


class MedicalRAGEvaluationService:
    """Run retrieval -> reasoning -> RAGAS-style LLM judge evaluation.

    Core evaluation lives in app/, not scripts/. For each question, all selected
    retrieval methods are evaluated concurrently up to max_concurrent_methods.
    """

    def __init__(
        self,
        retrievers: dict[str, Any],
        answer_generator: MedicalAnswerGenerator,
        evaluator: RagasStyleEvaluator,
        config: EvaluationConfig | None = None,
    ):
        self.retrievers = retrievers
        self.answer_generator = answer_generator
        self.evaluator = evaluator
        self.config = config or EvaluationConfig()
        self._method_semaphore = asyncio.Semaphore(max(1, self.config.max_concurrent_methods))

    async def run(self, questions: list[EvaluationQuestion]) -> list[EvaluationRow]:
        selected = questions[: self.config.max_questions or None]
        rows: list[EvaluationRow] = []
        for item in selected:
            tasks = [self._evaluate_one_guarded(item, method) for method in self.config.methods]
            rows.extend(await asyncio.gather(*tasks))
        return rows

    async def _evaluate_one_guarded(
        self,
        item: EvaluationQuestion,
        method: str,
    ) -> EvaluationRow:
        async with self._method_semaphore:
            return await self.evaluate_one(item, method)

    async def evaluate_one(self, item: EvaluationQuestion, method: str) -> EvaluationRow:
        started = time.perf_counter()
        result = await self._retrieve(method, item.question)
        self._assert_naive_vector_contract(method, result)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        generated = await asyncio.to_thread(
            self.answer_generator.generate,
            item.question,
            result,
        )
        # Use the exact context chunks seen by the answer generator when
        # sentence-level evidence selection is enabled. Retrieval metrics below
        # still use the original retrieval result, so Source@K/Term@K remain
        # comparable across methods.
        contexts = (generated.context_chunks or [])[: self.config.max_context_chunks]
        if not contexts:
            contexts = [
                build_clean_chunk_context([chunk], max_chunks=1)
                for chunk in result.chunks[: self.config.max_context_chunks]
            ]
        judgement = await asyncio.to_thread(
            self.evaluator.evaluate,
            item.question,
            item.reference,
            generated.answer,
            contexts,
            result,
            item.expected_answer_points,
        )
        retrieval_metrics = self._compute_retrieval_metrics(item, result)

        return EvaluationRow(
            difficulty=item.difficulty,
            method=result.method,
            question=item.question,
            reference=item.reference,
            answer=generated.answer,
            contexts=contexts,
            latency_ms=latency_ms,
            retrieval=self._result_to_compact_dict(result),
            evaluation=judgement,
            retrieval_metrics=retrieval_metrics,
        )

    async def _retrieve(self, method: str, question: str) -> RetrievalResult:
        retriever = self.retrievers[method]
        if method in {"vector", "vector_naive"}:
            return await asyncio.to_thread(retriever.retrieve, question, self.config.chunk_top_k)
        if method == "mixed" and self.config.async_mixed and hasattr(retriever, "retrieve_async"):
            return await retriever.retrieve_async(question)
        return await asyncio.to_thread(retriever.retrieve, question)

    def _assert_naive_vector_contract(self, requested_method: str, result: RetrievalResult) -> None:
        if requested_method in {"vector", "vector_naive", "vector_ce"}:
            debug = result.debug.get("base_debug") if requested_method == "vector_ce" else result.debug
            debug = debug or result.debug
            collection = debug.get("collection")
            mode = debug.get("vector_text_mode")
            if collection != NAIVE_CHUNK_COLLECTION or mode != "raw_chunk_text":
                raise RuntimeError(
                    "Naive vector contract violated: "
                    f"method={requested_method}, collection={collection}, mode={mode}. "
                    f"Expected collection={NAIVE_CHUNK_COLLECTION}, mode=raw_chunk_text."
                )

        if requested_method in {"mixed", "mixed_ce"}:
            base_debug = result.debug.get("base_debug") if requested_method == "mixed_ce" else result.debug
            branch_debug = (base_debug or result.debug).get("branch_debug") or {}
            vector_debug = branch_debug.get("vector") or {}
            collection = vector_debug.get("collection")
            mode = vector_debug.get("vector_text_mode")
            if collection != NAIVE_CHUNK_COLLECTION or mode != "raw_chunk_text":
                raise RuntimeError(
                    "Mixed retriever is not using naive vector branch: "
                    f"method={requested_method}, collection={collection}, mode={mode}. "
                    f"Expected collection={NAIVE_CHUNK_COLLECTION}, mode=raw_chunk_text."
                )

    def save(self, rows: list[EvaluationRow], output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        jsonl_path = output_path / "ragas_style_eval_rows.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.to_json_dict(), ensure_ascii=False) + "\n")

        scores_path = output_path / "ragas_style_scores.csv"
        with scores_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "difficulty",
                    "method",
                    "question",
                    "latency_ms",
                    "faithfulness",
                    "answer_relevance",
                    "context_relevance",
                    "ragas_style_overall",
                    "medical_safety",
                    "answer_correctness",
                    "answer_point_coverage",
                    "covered_answer_points",
                    "missing_answer_points",
                    "answer_correctness_rationale",
                    "term_hit_at_1",
                    "term_hit_at_3",
                    "term_hit_at_5",
                    "term_mrr",
                    "term_first_hit_rank",
                    "matched_terms",
                    "source_hit_at_1",
                    "source_hit_at_3",
                    "source_hit_at_5",
                    "source_mrr",
                    "source_first_hit_rank",
                    "matched_source_links",
                    "rationale",
                ],
            )
            writer.writeheader()
            for row in rows:
                eval_result = row.evaluation
                retrieval_metrics = row.retrieval_metrics
                writer.writerow({
                    "difficulty": row.difficulty,
                    "method": row.method,
                    "question": row.question,
                    "latency_ms": row.latency_ms,
                    "faithfulness": eval_result.faithfulness if eval_result else "",
                    "answer_relevance": eval_result.answer_relevance if eval_result else "",
                    "context_relevance": eval_result.context_relevance if eval_result else "",
                    "ragas_style_overall": eval_result.ragas_style_overall if eval_result else "",
                    "medical_safety": eval_result.medical_safety if eval_result else "",
                    "answer_correctness": eval_result.answer_correctness if eval_result else "",
                    "answer_point_coverage": eval_result.answer_point_coverage if eval_result else "",
                    "covered_answer_points": "; ".join(eval_result.covered_answer_points) if eval_result else "",
                    "missing_answer_points": "; ".join(eval_result.missing_answer_points) if eval_result else "",
                    "answer_correctness_rationale": eval_result.answer_correctness_rationale if eval_result else "",
                    "term_hit_at_1": retrieval_metrics.term_hit_at_1 if retrieval_metrics else "",
                    "term_hit_at_3": retrieval_metrics.term_hit_at_3 if retrieval_metrics else "",
                    "term_hit_at_5": retrieval_metrics.term_hit_at_5 if retrieval_metrics else "",
                    "term_mrr": retrieval_metrics.term_mrr if retrieval_metrics else "",
                    "term_first_hit_rank": retrieval_metrics.term_first_hit_rank if retrieval_metrics else "",
                    "matched_terms": "; ".join(retrieval_metrics.matched_terms) if retrieval_metrics else "",
                    "source_hit_at_1": retrieval_metrics.source_hit_at_1 if retrieval_metrics else "",
                    "source_hit_at_3": retrieval_metrics.source_hit_at_3 if retrieval_metrics else "",
                    "source_hit_at_5": retrieval_metrics.source_hit_at_5 if retrieval_metrics else "",
                    "source_mrr": retrieval_metrics.source_mrr if retrieval_metrics else "",
                    "source_first_hit_rank": retrieval_metrics.source_first_hit_rank if retrieval_metrics else "",
                    "matched_source_links": "; ".join(retrieval_metrics.matched_source_links) if retrieval_metrics else "",
                    "rationale": eval_result.rationale if eval_result else "",
                })

        summary = self._summarize(rows)
        summary_path = output_path / "ragas_style_summary_by_method.csv"
        with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "count",
                    "faithfulness",
                    "answer_relevance",
                    "context_relevance",
                    "ragas_style_overall",
                    "medical_safety",
                    "answer_correctness",
                    "answer_point_coverage",
                    "term_hit_at_1",
                    "term_hit_at_3",
                    "term_hit_at_5",
                    "term_mrr",
                    "source_hit_at_1",
                    "source_hit_at_3",
                    "source_hit_at_5",
                    "source_mrr",
                ],
            )
            writer.writeheader()
            writer.writerows(summary)

    def _summarize(self, rows: list[EvaluationRow]) -> list[dict[str, Any]]:
        grouped: dict[str, list[EvaluationRow]] = defaultdict(list)
        for row in rows:
            grouped[row.method].append(row)

        result: list[dict[str, Any]] = []
        metric_names = [
            "faithfulness",
            "answer_relevance",
            "context_relevance",
            "ragas_style_overall",
            "medical_safety",
            "answer_correctness",
            "answer_point_coverage",
        ]
        retrieval_metric_names = [
            "term_hit_at_1",
            "term_hit_at_3",
            "term_hit_at_5",
            "term_mrr",
            "source_hit_at_1",
            "source_hit_at_3",
            "source_hit_at_5",
            "source_mrr",
        ]
        for method, values in grouped.items():
            item: dict[str, Any] = {"method": method, "count": len(values)}
            for metric in metric_names:
                scores = [getattr(row.evaluation, metric) for row in values if row.evaluation is not None]
                item[metric] = round(sum(scores) / len(scores), 4) if scores else ""
            for metric in retrieval_metric_names:
                scores = [
                    getattr(row.retrieval_metrics, metric)
                    for row in values
                    if row.retrieval_metrics is not None
                ]
                item[metric] = round(sum(scores) / len(scores), 4) if scores else ""
            result.append(item)
        return sorted(result, key=lambda x: str(x["method"]))


    def _compute_retrieval_metrics(
        self,
        item: EvaluationQuestion,
        result: RetrievalResult,
    ) -> RetrievalMetricResult:
        expected_terms = [term for term in item.expected_terms if term and term.strip()]
        expected_source_links = [link for link in item.source_links if link and link.strip()]

        ranked_texts: list[str] = []
        ranked_sources: list[list[str]] = []
        for chunk in result.chunks[:10]:
            ranked_texts.append("\n".join([chunk.title, chunk.section, chunk.text, chunk.contextualized_text]))
            ranked_sources.append(self._chunk_source_candidates(chunk))

        term_rank, matched_terms = self._first_term_hit(expected_terms, ranked_texts)
        source_rank, matched_sources = self._first_source_hit(expected_source_links, ranked_sources)

        return RetrievalMetricResult(
            expected_terms=expected_terms,
            expected_source_links=expected_source_links,
            term_hit_at_1=1.0 if term_rank is not None and term_rank <= 1 else 0.0,
            term_hit_at_3=1.0 if term_rank is not None and term_rank <= 3 else 0.0,
            term_hit_at_5=1.0 if term_rank is not None and term_rank <= 5 else 0.0,
            term_mrr=round(1.0 / term_rank, 4) if term_rank else 0.0,
            term_first_hit_rank=term_rank,
            matched_terms=matched_terms,
            source_hit_at_1=1.0 if source_rank is not None and source_rank <= 1 else 0.0,
            source_hit_at_3=1.0 if source_rank is not None and source_rank <= 3 else 0.0,
            source_hit_at_5=1.0 if source_rank is not None and source_rank <= 5 else 0.0,
            source_mrr=round(1.0 / source_rank, 4) if source_rank else 0.0,
            source_first_hit_rank=source_rank,
            matched_source_links=matched_sources,
        )

    def _first_term_hit(
        self,
        expected_terms: list[str],
        ranked_texts: list[str],
    ) -> tuple[int | None, list[str]]:
        if not expected_terms:
            return None, []
        normalized_terms = [normalize_text(term) for term in expected_terms]
        first_hit_rank: int | None = None
        matched_terms: list[str] = []
        for rank, text in enumerate(ranked_texts, start=1):
            normalized = normalize_text(text)
            terms_in_rank = [
                original
                for original, normalized_term in zip(expected_terms, normalized_terms)
                if normalized_term and normalized_term in normalized
            ]
            if terms_in_rank:
                if first_hit_rank is None:
                    first_hit_rank = rank
                for term in terms_in_rank:
                    if term not in matched_terms:
                        matched_terms.append(term)
        return first_hit_rank, matched_terms

    def _first_source_hit(
        self,
        expected_source_links: list[str],
        ranked_sources: list[list[str]] | list[str],
    ) -> tuple[int | None, list[str]]:
        if not expected_source_links:
            return None, []

        expected_pairs = [
            (original, self._normalize_url(original))
            for original in expected_source_links
        ]
        first_hit_rank: int | None = None
        matched_links: list[str] = []

        for rank, raw_sources in enumerate(ranked_sources, start=1):
            if isinstance(raw_sources, str):
                candidate_sources = [raw_sources]
            else:
                candidate_sources = list(raw_sources or [])

            normalized_sources = [
                self._normalize_url(source)
                for source in candidate_sources
                if source
            ]
            normalized_sources = [source for source in normalized_sources if source]
            if not normalized_sources:
                continue

            for original, expected in expected_pairs:
                if not expected:
                    continue
                if any(self._urls_match(source, expected) for source in normalized_sources):
                    if first_hit_rank is None:
                        first_hit_rank = rank
                    if original not in matched_links:
                        matched_links.append(original)

        return first_hit_rank, matched_links

    def _chunk_source_candidates(self, chunk) -> list[str]:
        metadata = dict(getattr(chunk, "metadata", None) or {})
        values: list[str] = []

        def add(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    add(item)
                return
            text = str(value).strip()
            if text and text not in values:
                values.append(text)

        add(getattr(chunk, "source_url", ""))
        for key in (
            "source_url",
            "url",
            "article_url",
            "canonical_url",
            "source_link",
            "source_links",
            "source",
        ):
            add(metadata.get(key))

        return values

    def _urls_match(self, source: str, expected: str) -> bool:
        if not source or not expected:
            return False
        if source == expected:
            return True
        if source.endswith(expected) or expected.endswith(source):
            return True

        source_slug = self._url_slug(source)
        expected_slug = self._url_slug(expected)
        if source_slug and expected_slug and source_slug == expected_slug:
            return True

        return False

    def _url_slug(self, normalized_url: str) -> str:
        text = (normalized_url or "").rstrip("/")
        if not text:
            return ""
        return text.rsplit("/", 1)[-1]

    def _normalize_url(self, value: str) -> str:
        text = str(value or "").strip().strip("<>\"'").lower()
        if not text:
            return ""

        # Remove markdown-like wrappers and tracking fragments.
        text = text.split("#", 1)[0].split("?", 1)[0].strip().rstrip("/")
        text = re.sub(r"^\[(.*?)\]\((.*?)\)$", r"\2", text)

        parse_input = text
        if "://" not in parse_input and re.match(r"^[a-z0-9.-]+\.", parse_input):
            parse_input = "https://" + parse_input

        parsed = urlparse(parse_input)
        if parsed.netloc:
            host = parsed.netloc
            if host.startswith("www."):
                host = host[4:]
            path = unquote(parsed.path or "").rstrip("/")
            path = re.sub(r"/amp$", "", path)
            path = re.sub(r"/+", "/", path)
            return f"{host}{path}".rstrip("/")

        text = re.sub(r"^https?://", "", text)
        if text.startswith("www."):
            text = text[4:]
        text = unquote(text).rstrip("/")
        text = re.sub(r"/amp$", "", text)
        text = re.sub(r"/+", "/", text)
        return text

    def _result_to_compact_dict(self, result: RetrievalResult) -> dict[str, Any]:
        return {
            "query": result.query,
            "method": result.method,
            "keywords": dataclasses.asdict(result.keywords) if result.keywords else None,
            "top_chunk_ids": [chunk.chunk_id for chunk in result.chunks[:10]],
            "top_chunk_titles": [chunk.title for chunk in result.chunks[:10]],
            "top_chunk_source_urls": [chunk.source_url for chunk in result.chunks[:10]],
            "top_entities": [e.name for e in result.entities[:10]],
            "top_relations": [
                f"{r.subject_name} -[{r.relation_type}]-> {r.object_name}"
                for r in result.relations[:10]
            ],
            "debug": result.debug,
        }