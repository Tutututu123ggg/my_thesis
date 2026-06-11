from __future__ import annotations

import asyncio
import csv
import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

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
        if requested_method in {"vector", "vector_naive"}:
            collection = result.debug.get("collection")
            mode = result.debug.get("vector_text_mode")
            if collection != NAIVE_CHUNK_COLLECTION or mode != "raw_chunk_text":
                raise RuntimeError(
                    "Naive vector contract violated: "
                    f"method={requested_method}, collection={collection}, mode={mode}. "
                    f"Expected collection={NAIVE_CHUNK_COLLECTION}, mode=raw_chunk_text."
                )

        if requested_method == "mixed":
            branch_debug = result.debug.get("branch_debug") or {}
            vector_debug = branch_debug.get("vector") or {}
            collection = vector_debug.get("collection")
            mode = vector_debug.get("vector_text_mode")
            if collection != NAIVE_CHUNK_COLLECTION or mode != "raw_chunk_text":
                raise RuntimeError(
                    "Mixed retriever is not using naive vector branch: "
                    f"collection={collection}, mode={mode}. "
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
                    "retrieval_hit_at_1",
                    "retrieval_hit_at_3",
                    "retrieval_hit_at_5",
                    "retrieval_mrr",
                    "retrieval_first_hit_rank",
                    "retrieval_matched_terms",
                    "verdict",
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
                    "retrieval_hit_at_1": retrieval_metrics.hit_at_1 if retrieval_metrics else "",
                    "retrieval_hit_at_3": retrieval_metrics.hit_at_3 if retrieval_metrics else "",
                    "retrieval_hit_at_5": retrieval_metrics.hit_at_5 if retrieval_metrics else "",
                    "retrieval_mrr": retrieval_metrics.mrr if retrieval_metrics else "",
                    "retrieval_first_hit_rank": retrieval_metrics.first_hit_rank if retrieval_metrics else "",
                    "retrieval_matched_terms": "; ".join(retrieval_metrics.matched_terms) if retrieval_metrics else "",
                    "verdict": eval_result.verdict if eval_result else "",
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
                    "retrieval_hit_at_1",
                    "retrieval_hit_at_3",
                    "retrieval_hit_at_5",
                    "retrieval_mrr",
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
        ]
        retrieval_metric_names = [
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "mrr",
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
                item[f"retrieval_{metric}"] = round(sum(scores) / len(scores), 4) if scores else ""
            result.append(item)
        return sorted(result, key=lambda x: str(x["method"]))


    def _compute_retrieval_metrics(
        self,
        item: EvaluationQuestion,
        result: RetrievalResult,
    ) -> RetrievalMetricResult:
        expected_terms = [term for term in item.expected_terms if term and term.strip()]
        if not expected_terms:
            return RetrievalMetricResult()

        normalized_terms = [normalize_text(term) for term in expected_terms]
        first_hit_rank: int | None = None
        matched_terms: list[str] = []

        ranked_texts: list[str] = []
        for chunk in result.chunks[:10]:
            ranked_texts.append("\n".join([chunk.title, chunk.section, chunk.text, chunk.contextualized_text]))

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

        if first_hit_rank is None:
            return RetrievalMetricResult(
                expected_terms=expected_terms,
                hit_at_1=0.0,
                hit_at_3=0.0,
                hit_at_5=0.0,
                mrr=0.0,
                first_hit_rank=None,
                matched_terms=matched_terms,
            )

        return RetrievalMetricResult(
            expected_terms=expected_terms,
            hit_at_1=1.0 if first_hit_rank <= 1 else 0.0,
            hit_at_3=1.0 if first_hit_rank <= 3 else 0.0,
            hit_at_5=1.0 if first_hit_rank <= 5 else 0.0,
            mrr=round(1.0 / first_hit_rank, 4),
            first_hit_rank=first_hit_rank,
            matched_terms=matched_terms,
        )

    def _result_to_compact_dict(self, result: RetrievalResult) -> dict[str, Any]:
        return {
            "query": result.query,
            "method": result.method,
            "keywords": dataclasses.asdict(result.keywords) if result.keywords else None,
            "top_chunk_ids": [chunk.chunk_id for chunk in result.chunks[:10]],
            "top_chunk_titles": [chunk.title for chunk in result.chunks[:10]],
            "top_entities": [e.name for e in result.entities[:10]],
            "top_relations": [
                f"{r.subject_name} -[{r.relation_type}]-> {r.object_name}"
                for r in result.relations[:10]
            ],
            "debug": result.debug,
        }
