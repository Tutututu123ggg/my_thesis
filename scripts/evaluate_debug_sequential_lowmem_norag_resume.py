from __future__ import annotations

import argparse
import csv
import json
import math
import gc
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Allow running with: python -m scripts.evaluate_debug_sequential
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.datasets import SMALL_MEDICAL_EVAL_SET, load_evaluation_questions
from app.evaluation.embedding_clients import build_answer_relevance_embedding_client
from app.evaluation.judge import RagasStyleEvaluator
from app.evaluation.models import EvaluationConfig, EvaluationQuestion, EvaluationRow
from app.evaluation.retriever_factory import RetrieverBuildConfig, build_retriever_bundle
from app.evaluation.service import MedicalRAGEvaluationService
from app.reasoning import ChatLLMClient, GeneratedAnswer, MedicalAnswerConfig, MedicalAnswerGenerator
from app.reasoning.answer_generator import build_clean_chunk_context
from app.retrieval import RetrievalResult


def log_memory(label: str) -> None:
    """Best-effort RAM logger; works if psutil is installed."""
    try:
        import psutil

        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024
        print(f"[MEM] {label}: rss={rss_mb:.1f} MB", flush=True)
    except Exception:
        pass


NO_RAG_SYSTEM_PROMPT = """
Bạn là trợ lý y khoa tiếng Việt.
Bạn KHÔNG có retrieved context trong chế độ no_rag.
Hãy trả lời dựa trên kiến thức y khoa chung một cách thận trọng.
Không bịa nguồn, không nói rằng thông tin đến từ Tâm Anh nếu không được cung cấp.
Không chẩn đoán chắc chắn từ mô tả triệu chứng; chỉ nêu khả năng liên quan và khuyến nghị đi khám khi cần.
""".strip()


def build_no_rag_prompt(question: str) -> str:
    return f"""
Câu hỏi của người dùng:
{question}

Yêu cầu trả lời:
- Trả lời bằng tiếng Việt, ngắn gọn nhưng đủ ý.
- Đây là baseline no_rag: không có retrieved context, không có nguồn truy xuất.
- Không bịa dẫn chứng nguồn.
- Nếu câu hỏi cần thông tin rất đặc thù theo nguồn, hãy trả lời thận trọng và nêu rằng cần đối chiếu nguồn/bác sĩ.
- Nếu người dùng mô tả triệu chứng, chỉ nêu các khả năng liên quan, không chẩn đoán chắc chắn.
- Nhắc người dùng đi khám nếu triệu chứng kéo dài, nặng lên, khó thở, sốt, đau nhiều, chảy dịch/mủ, sưng nóng hoặc ảnh hưởng sinh hoạt.
""".strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def _hydrate_missing_benchmark_fields(item: EvaluationQuestion) -> None:
    """Make the runner robust even if datasets.py is still an older version.

    Older loader versions put source_links / expected_answer_points in metadata
    but did not copy them into EvaluationQuestion fields. This function mirrors
    them back so source_hit and answer_point_coverage can work.
    """
    metadata = dict(item.metadata or {})

    if not item.source_links:
        item.source_links = _as_list(metadata.get("source_links"))

    if not item.expected_answer_points:
        item.expected_answer_points = _as_list(metadata.get("expected_answer_points"))

    if not item.expected_terms:
        expected_terms = _as_list(metadata.get("expected_terms"))
        if not expected_terms:
            expected_terms = _as_list(metadata.get("expected_entities"))
        item.expected_terms = expected_terms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully sequential debug evaluator for medical-vietnamese-hybrid-rag."
    )

    parser.add_argument(
        "--dataset-file",
        default="data/evaluation/benchmarks/vietnamese_medical_rag_benchmark_10_topk5.json",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["no_rag", "vector_naive", "lightrag", "hipporag", "mixed"],
        choices=["no_rag", "vector", "vector_naive", "lightrag", "hipporag", "mixed"],
    )
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--output-dir", default="data/evaluation/debug_sequential")

    parser.add_argument("--no-llm-keywords", action="store_true")
    parser.add_argument("--answer-context-mode", default="chunks_only", choices=["chunks_only", "full_graph"])

    parser.add_argument("--reasoning-provider", default="openai")
    parser.add_argument("--reasoning-model", default="gpt-4o-mini")
    parser.add_argument("--judge-provider", default="openai")
    parser.add_argument("--judge-model", default="gpt-4o-mini")

    parser.add_argument("--answer-relevance-embedding-provider", default="local", choices=["local", "openai"])
    parser.add_argument("--answer-relevance-embedding-model", default="BAAI/bge-m3")

    parser.add_argument("--chunk-top-k", type=int, default=5)
    parser.add_argument("--entity-top-k", type=int, default=5)
    parser.add_argument("--relation-top-k", type=int, default=5)
    parser.add_argument("--hippo-seed-top-k", type=int, default=5)
    parser.add_argument("--hippo-ppr-entities", type=int, default=60)
    parser.add_argument("--hippo-hops", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=5)
    parser.add_argument("--max-contexts", type=int, default=5)

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running other rows if one method/question fails.",
    )
    parser.add_argument(
        "--save-after-each-row",
        action="store_true",
        default=True,
        help="Write partial CSV/JSONL after every completed row.",
    )
    parser.add_argument(
        "--reuse-retrieval-embedding-for-answer-relevance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When answer relevance embedding provider is local, reuse the BGE-M3 "
            "EmbeddingService already loaded by the retriever bundle. This avoids "
            "loading BGE-M3 twice and is strongly recommended for 100-question eval."
        ),
    )
    parser.add_argument(
        "--gc-after-row",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run gc.collect() after each completed row.",
    )
    parser.add_argument(
        "--save-full-rows-after-each-row",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, call service.save(rows, output_dir) after each row. "
            "This rewrites full CSV/JSONL repeatedly and keeps all rows in RAM. "
            "Default false writes partial JSONL/checkpoints only during the run, "
            "then writes final CSV at the end."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read output_dir/partial_rows.jsonl and skip already completed (question, method) jobs.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Do not run evaluation. Only rebuild CSV/summary from output_dir/partial_rows.jsonl.",
    )

    return parser.parse_args()


def build_components(args: argparse.Namespace):
    print("[INIT] Loading .env", flush=True)
    load_dotenv(PROJECT_ROOT / ".env")

    print("[INIT] Building retriever bundle...", flush=True)
    retriever_config = RetrieverBuildConfig(
        no_llm_keywords=args.no_llm_keywords,
        chunk_top_k=args.chunk_top_k,
        entity_top_k=args.entity_top_k,
        relation_top_k=args.relation_top_k,
        hippo_seed_top_k=args.hippo_seed_top_k,
        hippo_ppr_entities=args.hippo_ppr_entities,
        hippo_hops=args.hippo_hops,
        max_chunks=args.max_chunks,
    )
    bundle = build_retriever_bundle(config=retriever_config)

    print("[INIT] Building reasoning LLM...", flush=True)
    reasoning_llm = ChatLLMClient.from_env(
        prefix="REASONING",
        default_provider=args.reasoning_provider,
        default_model=args.reasoning_model,
    )
    reasoning_llm.config.provider = args.reasoning_provider
    reasoning_llm.config.model = args.reasoning_model

    print("[INIT] Building judge LLM...", flush=True)
    judge_llm = ChatLLMClient.from_env(
        prefix="JUDGE",
        default_provider=args.judge_provider,
        default_model=args.judge_model,
    )
    judge_llm.config.provider = args.judge_provider
    judge_llm.config.model = args.judge_model

    answer_generator = MedicalAnswerGenerator(
        llm_client=reasoning_llm,
        config=MedicalAnswerConfig(
            max_chunks=args.max_contexts,
            context_mode=args.answer_context_mode,  # type: ignore[arg-type]
        ),
    )

    print("[INIT] Building answer relevance embedding...", flush=True)
    embedding_client = None
    evaluator_embedding_service = None

    if (
        args.answer_relevance_embedding_provider == "local"
        and args.reuse_retrieval_embedding_for_answer_relevance
    ):
        vector_retriever = bundle.retrievers.get("vector_naive") or bundle.retrievers.get("vector")
        evaluator_embedding_service = getattr(vector_retriever, "embedding_service", None)
        if evaluator_embedding_service is None:
            print(
                "[INIT] Could not find retriever embedding service; loading a separate local embedding client.",
                flush=True,
            )
            embedding_client = build_answer_relevance_embedding_client(
                provider=args.answer_relevance_embedding_provider,
                model_name=args.answer_relevance_embedding_model,
            )
        else:
            model_name = getattr(evaluator_embedding_service, "model_name", None)
            print(
                f"[INIT] Reusing retriever embedding service for answer_relevance "
                f"(model={model_name or args.answer_relevance_embedding_model}).",
                flush=True,
            )
    else:
        embedding_client = build_answer_relevance_embedding_client(
            provider=args.answer_relevance_embedding_provider,
            model_name=args.answer_relevance_embedding_model,
        )

    evaluator = RagasStyleEvaluator(
        judge_llm,
        embedding_service=evaluator_embedding_service,
        embedding_client=embedding_client,
        generated_question_count=3,
    )
    log_memory("after component build")

    eval_config = EvaluationConfig(
        methods=args.methods,
        async_mixed=False,
        max_questions=args.max_questions,
        max_context_chunks=args.max_contexts,
        chunk_top_k=args.chunk_top_k,
        entity_top_k=args.entity_top_k,
        relation_top_k=args.relation_top_k,
        hippo_seed_top_k=args.hippo_seed_top_k,
        hippo_ppr_entities=args.hippo_ppr_entities,
        hippo_hops=args.hippo_hops,
        max_chunks=args.max_chunks,
        max_concurrent_methods=1,
        answer_relevance_embedding_provider=args.answer_relevance_embedding_provider,
        answer_relevance_embedding_model=args.answer_relevance_embedding_model,
        output_dir=args.output_dir,
    )

    service = MedicalRAGEvaluationService(
        retrievers=bundle.retrievers,
        answer_generator=answer_generator,
        evaluator=evaluator,
        config=eval_config,
    )

    return bundle, service


def load_dataset(args: argparse.Namespace) -> list[EvaluationQuestion]:
    if args.dataset_file:
        dataset_path = PROJECT_ROOT / args.dataset_file
        print(f"[DATASET] Loading {dataset_path}", flush=True)
        dataset = load_evaluation_questions(dataset_path)
    else:
        print("[DATASET] Loading SMALL_MEDICAL_EVAL_SET", flush=True)
        dataset = list(SMALL_MEDICAL_EVAL_SET)

    for item in dataset:
        _hydrate_missing_benchmark_fields(item)

    selected = dataset[: args.max_questions or None]
    print(f"[DATASET] total={len(dataset)} selected={len(selected)}", flush=True)
    if selected:
        first = selected[0]
        print(f"[DATASET] first_question={first.question}", flush=True)
        print(f"[DATASET] first_terms={first.expected_terms}", flush=True)
        print(f"[DATASET] first_sources={first.source_links}", flush=True)
        print(f"[DATASET] first_points={len(first.expected_answer_points)}", flush=True)

    if not selected:
        raise RuntimeError("Dataset is empty after applying max_questions.")

    log_memory("after dataset load")
    return selected


def retrieve_sync(service: MedicalRAGEvaluationService, method: str, question: str) -> RetrievalResult:
    retriever = service.retrievers[method]
    if method in {"vector", "vector_naive"}:
        return retriever.retrieve(question, top_k=service.config.chunk_top_k)
    return retriever.retrieve(question)


def generate_no_rag_answer(service: MedicalRAGEvaluationService, question: str) -> GeneratedAnswer:
    answer = service.answer_generator.llm_client.generate(
        system_prompt=NO_RAG_SYSTEM_PROMPT,
        user_prompt=build_no_rag_prompt(question),
    )
    return GeneratedAnswer(
        question=question,
        answer=(answer or "").strip(),
        method="no_rag",
        context="",
        context_chunks=[],
        metadata={"baseline": "no_rag", "retrieval": False},
    )


def mark_context_dependent_metrics_nan(judgement) -> None:
    """No-RAG has no retrieved context, so context-dependent RAG metrics are N/A."""
    judgement.faithfulness = math.nan
    judgement.context_relevance = math.nan
    judgement.ragas_style_overall = math.nan
    judgement.relevant_sentences = []
    judgement.total_context_sentences = 0
    # Keep answer_relevance, medical_safety, answer_point_coverage, answer_correctness.


def evaluate_one_sync(
    service: MedicalRAGEvaluationService,
    item: EvaluationQuestion,
    method: str,
) -> EvaluationRow:
    if method == "no_rag":
        print("[RETRIEVE] method=no_rag skipped", flush=True)
        started = time.perf_counter()

        print("[ANSWER] method=no_rag", flush=True)
        generated = generate_no_rag_answer(service, item.question)
        answer = generated.answer or ""
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        print(f"[ANSWER] chars={len(answer)} preview={answer[:180].replace(chr(10), ' ')}", flush=True)

        contexts: list[str] = []
        result = None
        print("[JUDGE] method=no_rag contexts=0", flush=True)
        judgement = service.evaluator.evaluate(
            item.question,
            item.reference,
            answer,
            contexts,
            None,
            item.expected_answer_points,
        )
        mark_context_dependent_metrics_nan(judgement)

        retrieval_metrics = None
        print(
            f"[METRICS] method=no_rag "
            f"correct={judgement.answer_correctness:.3f} "
            f"coverage={judgement.answer_point_coverage:.3f} "
            f"faith=N/A source@5=N/A term@5=N/A",
            flush=True,
        )

        return EvaluationRow(
            difficulty=item.difficulty,
            method="no_rag",
            question=item.question,
            reference=item.reference,
            answer=answer,
            contexts=[],
            latency_ms=latency_ms,
            retrieval={
                "method": "no_rag",
                "chunks": [],
                "entities": [],
                "relations": [],
                "debug": {"baseline": "no_rag", "retrieval": False},
            },
            evaluation=judgement,
            retrieval_metrics=retrieval_metrics,
        )

    print(f"[RETRIEVE] method={method}", flush=True)
    started = time.perf_counter()
    result = retrieve_sync(service, method, item.question)
    service._assert_naive_vector_contract(method, result)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    print(
        f"[RETRIEVE] done method={method} chunks={len(result.chunks)} "
        f"entities={len(result.entities)} relations={len(result.relations)} latency_ms={latency_ms}",
        flush=True,
    )

    for rank, chunk in enumerate(result.chunks[:3], start=1):
        print(
            f"  [CHUNK {rank}] score={chunk.score:.4f} title={chunk.title!r} url={chunk.source_url!r}",
            flush=True,
        )

    print(f"[ANSWER] method={method}", flush=True)
    generated = service.answer_generator.generate(item.question, result)
    answer = generated.answer or ""
    print(f"[ANSWER] chars={len(answer)} preview={answer[:180].replace(chr(10), ' ')}", flush=True)

    contexts = [
        build_clean_chunk_context([chunk], max_chunks=1)
        for chunk in result.chunks[: service.config.max_context_chunks]
    ]
    print(f"[JUDGE] method={method} contexts={len(contexts)}", flush=True)
    judgement = service.evaluator.evaluate(
        item.question,
        item.reference,
        answer,
        contexts,
        result,
        item.expected_answer_points,
    )

    retrieval_metrics = service._compute_retrieval_metrics(item, result)
    print(
        f"[METRICS] method={method} "
        f"correct={judgement.answer_correctness:.3f} "
        f"coverage={judgement.answer_point_coverage:.3f} "
        f"faith={judgement.faithfulness:.3f} "
        f"source@5={retrieval_metrics.source_hit_at_5:.0f} "
        f"term@5={retrieval_metrics.term_hit_at_5:.0f}",
        flush=True,
    )

    # Keep only compact retrieval/context in memory.
    # The judge already consumed the original contexts above.
    compact_contexts = [ctx[:1600] for ctx in contexts]
    compact_retrieval = service._result_to_compact_dict(result)

    return EvaluationRow(
        difficulty=item.difficulty,
        method=result.method,
        question=item.question,
        reference=item.reference,
        answer=answer,
        contexts=compact_contexts,
        latency_ms=latency_ms,
        retrieval=compact_retrieval,
        evaluation=judgement,
        retrieval_metrics=retrieval_metrics,
    )


def row_key_from_dict(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("question") or "").strip(), str(row.get("method") or "").strip())


def row_key_from_item(item: EvaluationQuestion, method: str) -> tuple[str, str]:
    return (str(item.question or "").strip(), str(method or "").strip())


def load_partial_row_dicts(partial_path: Path) -> list[dict[str, Any]]:
    """Load partial_rows.jsonl and deduplicate by (question, method).

    If duplicate rows exist because an older run was restarted, keep the last
    valid row for each key while preserving the first-seen order.
    """
    if not partial_path.exists():
        return []

    order: list[tuple[str, str]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}

    with partial_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                print(f"[RESUME] Skip broken JSONL line {line_no}: {exc}", flush=True)
                continue

            key = row_key_from_dict(row)
            if not key[0] or not key[1]:
                print(f"[RESUME] Skip row with empty key at line {line_no}", flush=True)
                continue

            if key not in by_key:
                order.append(key)
            by_key[key] = row

    return [by_key[key] for key in order]


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, list):
        return "; ".join(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def _metric(row: dict[str, Any], key: str) -> Any:
    ragas = row.get("ragas_style") or row.get("judge") or {}
    if isinstance(ragas, dict) and key in ragas:
        return ragas.get(key)
    retrieval_metrics = row.get("retrieval_metrics") or {}
    if isinstance(retrieval_metrics, dict) and key in retrieval_metrics:
        return retrieval_metrics.get(key)
    return ""


def _to_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        score = float(value)
    except Exception:
        return None
    if math.isnan(score):
        return None
    return score


def write_outputs_from_partial(output_dir: Path) -> None:
    """Rebuild final JSONL, scores CSV and method summary from partial_rows.jsonl."""
    partial_path = output_dir / "partial_rows.jsonl"
    rows = load_partial_row_dicts(partial_path)
    if not rows:
        print(f"[FINALIZE] No rows found in {partial_path}", flush=True)
        return

    eval_rows_path = output_dir / "ragas_style_eval_rows.jsonl"
    with eval_rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    columns = [
        "difficulty",
        "method",
        "question",
        "latency_ms",
        "answer_correctness",
        "answer_point_coverage",
        "faithfulness",
        "answer_relevance",
        "context_relevance",
        "ragas_style_overall",
        "medical_safety",
        "term_hit_at_1",
        "term_hit_at_3",
        "term_hit_at_5",
        "term_mrr",
        "source_hit_at_1",
        "source_hit_at_3",
        "source_hit_at_5",
        "source_mrr",
        "matched_terms",
        "matched_source_links",
        "answer",
    ]

    scores_path = output_dir / "ragas_style_scores.csv"
    with scores_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            retrieval_metrics = row.get("retrieval_metrics") or {}
            out = {
                "difficulty": row.get("difficulty", ""),
                "method": row.get("method", ""),
                "question": row.get("question", ""),
                "latency_ms": row.get("latency_ms", ""),
                "answer": row.get("answer", ""),
            }
            for col in columns:
                if col in out:
                    continue
                if col in {"matched_terms", "matched_source_links"} and isinstance(retrieval_metrics, dict):
                    out[col] = retrieval_metrics.get(col, "")
                else:
                    out[col] = _metric(row, col)
            writer.writerow({col: _csv_value(out.get(col, "")) for col in columns})

    numeric_cols = [
        "latency_ms",
        "answer_correctness",
        "answer_point_coverage",
        "faithfulness",
        "answer_relevance",
        "context_relevance",
        "ragas_style_overall",
        "medical_safety",
        "term_hit_at_1",
        "term_hit_at_3",
        "term_hit_at_5",
        "term_mrr",
        "source_hit_at_1",
        "source_hit_at_3",
        "source_hit_at_5",
        "source_mrr",
    ]
    grouped: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        method = str(row.get("method") or "")
        if not method:
            continue
        counts[method] = counts.get(method, 0) + 1
        grouped.setdefault(method, {col: [] for col in numeric_cols})
        for col in numeric_cols:
            value = row.get(col, "")
            if col != "latency_ms":
                value = _metric(row, col)
            num = _to_float_or_none(value)
            if num is not None:
                grouped[method][col].append(num)

    summary_path = output_dir / "ragas_style_summary_by_method.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["method", "rows", *numeric_cols]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in sorted(grouped):
            out = {"method": method, "rows": counts.get(method, 0)}
            for col in numeric_cols:
                values = grouped[method].get(col, [])
                out[col] = round(sum(values) / len(values), 6) if values else ""
            writer.writerow(out)

    print(
        f"[FINALIZE] rows={len(rows)} files={scores_path.name}, "
        f"{summary_path.name}, {eval_rows_path.name}",
        flush=True,
    )



def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "partial_rows.jsonl"

    if args.finalize_only:
        write_outputs_from_partial(output_dir)
        return

    bundle = None
    rows: list[EvaluationRow] = []

    existing_rows = load_partial_row_dicts(partial_path) if args.resume else []
    completed_keys = {row_key_from_dict(row) for row in existing_rows}
    if existing_rows:
        print(
            f"[RESUME] Loaded {len(existing_rows)} completed rows from {partial_path}",
            flush=True,
        )
    elif args.resume:
        print(f"[RESUME] No existing checkpoint found at {partial_path}", flush=True)

    try:
        dataset = load_dataset(args)
        bundle, service = build_components(args)

        total_jobs = len(dataset) * len(args.methods)
        done_jobs = 0
        skipped_jobs = 0

        for q_idx, item in enumerate(dataset, start=1):
            print("\n" + "=" * 100, flush=True)
            print(f"[QUESTION] {q_idx}/{len(dataset)} difficulty={item.difficulty}", flush=True)
            print(f"[QUESTION] {item.question}", flush=True)
            print(f"[EXPECTED] terms={item.expected_terms}", flush=True)
            print(f"[EXPECTED] sources={item.source_links}", flush=True)

            for method in args.methods:
                done_jobs += 1
                key = row_key_from_item(item, method)

                if args.resume and key in completed_keys:
                    skipped_jobs += 1
                    print("-" * 100, flush=True)
                    print(
                        f"[SKIP] {done_jobs}/{total_jobs} q={q_idx} method={method} already in checkpoint",
                        flush=True,
                    )
                    continue

                print("-" * 100, flush=True)
                print(f"[JOB] {done_jobs}/{total_jobs} q={q_idx} method={method}", flush=True)

                try:
                    row = evaluate_one_sync(service, item, method)
                    rows.append(row)
                    row_dict = row.to_json_dict()

                    if args.save_after_each_row:
                        with partial_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row_dict, ensure_ascii=False, default=str) + "\n")
                        completed_keys.add(row_key_from_dict(row_dict))
                        print(
                            f"[PARTIAL_JSONL] completed={len(completed_keys)} "
                            f"new_rows={len(rows)} file={partial_path}",
                            flush=True,
                        )

                        if args.save_full_rows_after_each_row:
                            # Rebuild CSV/summary from checkpoint, including old resumed rows.
                            write_outputs_from_partial(output_dir)

                    if args.gc_after_row:
                        gc.collect()
                        log_memory(f"after new row {len(rows)}")

                except Exception as exc:
                    print(f"[ERROR] q={q_idx} method={method} {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    write_outputs_from_partial(output_dir)
                    gc.collect()
                    log_memory("after error")
                    if not args.continue_on_error:
                        raise

        write_outputs_from_partial(output_dir)
        print("\n" + "=" * 100, flush=True)
        print(
            f"[SAVED] new_rows={len(rows)} skipped={skipped_jobs} "
            f"completed={len(completed_keys)} dir={output_dir}",
            flush=True,
        )
        log_memory("final")

    finally:
        if bundle is not None:
            bundle.close()
            print("[CLOSE] retriever bundle closed", flush=True)


if __name__ == "__main__":
    main()
