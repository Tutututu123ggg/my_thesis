from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from app.evaluation.datasets import SMALL_MEDICAL_EVAL_SET, load_evaluation_questions
from app.evaluation.embedding_clients import build_answer_relevance_embedding_client
from app.evaluation.judge import RagasStyleEvaluator
from app.evaluation.models import EvaluationConfig
from app.evaluation.retriever_factory import RetrieverBuildConfig, build_retriever_bundle
from app.evaluation.service import MedicalRAGEvaluationService
from app.reasoning import ChatLLMClient, MedicalAnswerConfig, MedicalAnswerGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Medical RAG reasoning + LLM judge evaluation.")
    parser.add_argument("--methods", nargs="+", choices=["vector", "vector_naive", "vector_ce", "lightrag", "hipporag", "mixed", "mixed_ce"], default=["vector_naive", "lightrag", "hipporag", "mixed"])
    parser.add_argument("--no-llm-keywords", action="store_true")
    parser.add_argument("--async-mixed", action="store_true")
    parser.add_argument("--dataset-file", default="", help="Path to a benchmark JSON file. If omitted, uses the built-in small set.")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--max-contexts", type=int, default=5)
    parser.add_argument("--chunk-top-k", type=int, default=10)
    parser.add_argument("--entity-top-k", type=int, default=20)
    parser.add_argument("--relation-top-k", type=int, default=20)
    parser.add_argument("--hippo-seed-top-k", type=int, default=12)
    parser.add_argument("--hippo-ppr-entities", type=int, default=40)
    parser.add_argument("--hippo-hops", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--cross-encoder-model", default="", help="Model for vector_ce/mixed_ce. Defaults to CROSS_ENCODER_MODEL or BAAI/bge-reranker-v2-m3.")
    parser.add_argument("--rerank-candidate-top-k", type=int, default=50)
    parser.add_argument("--rerank-final-top-k", type=int, default=0, help="0 means use --max-chunks.")
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-max-text-chars", type=int, default=2400)
    parser.add_argument("--rerank-cross-encoder-weight", type=float, default=0.35)
    parser.add_argument("--rerank-retrieval-weight", type=float, default=0.65)
    parser.add_argument("--rerank-max-chunks-per-source", type=int, default=3)
    parser.add_argument("--rerank-preserve-pre-top-k", type=int, default=2)
    parser.add_argument("--no-rerank-preserve-pre-at-top", action="store_true", help="Keep protected chunks in the final window but do not force them to the top positions.")
    parser.add_argument("--rerank-preserve-at-top-max-ce-rank", type=int, default=5, help="Only force preserved chunks to the top if their CE rank is at most this value. Use 0 to always force.")
    parser.add_argument("--no-rerank-fail-open", action="store_true", help="Raise if the CE model cannot load instead of falling back to pre-rerank order.")
    parser.add_argument("--max-concurrent-methods", type=int, default=4)
    parser.add_argument("--answer-context-mode", choices=["chunks_only", "full_graph"], default="chunks_only")
    parser.add_argument("--disable-evidence-selection", action="store_true", help="Disable sentence-level context compaction before answer generation.")
    parser.add_argument("--evidence-max-sentences-per-chunk", type=int, default=3)
    parser.add_argument("--evidence-sentence-window-size", type=int, default=1)
    parser.add_argument("--evidence-max-compact-chars-per-chunk", type=int, default=1400)
    parser.add_argument("--output-dir", default="data/evaluation/ragas_style_small")

    parser.add_argument("--reasoning-provider", choices=["openai", "ollama"], default=os.getenv("REASONING_PROVIDER", "ollama"))
    parser.add_argument("--reasoning-model", default=os.getenv("REASONING_MODEL", "qwen2.5:3b"))
    parser.add_argument("--judge-provider", choices=["openai", "ollama"], default=os.getenv("JUDGE_PROVIDER", "openai"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--answer-relevance-embedding-provider", choices=["openai", "local"], default=os.getenv("ANSWER_RELEVANCE_EMBEDDING_PROVIDER", "openai"))
    parser.add_argument("--answer-relevance-embedding-model", default=os.getenv("ANSWER_RELEVANCE_EMBEDDING_MODEL", "text-embedding-3-large"))
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    bundle = build_retriever_bundle(
        RetrieverBuildConfig(
            no_llm_keywords=args.no_llm_keywords,
            entity_top_k=args.entity_top_k,
            relation_top_k=args.relation_top_k,
            chunk_top_k=args.chunk_top_k,
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

    reasoning_llm = ChatLLMClient.from_env(
        prefix="REASONING",
        default_provider=args.reasoning_provider,
        default_model=args.reasoning_model,
    )
    # CLI flags should override env-derived defaults.
    reasoning_llm.config.provider = args.reasoning_provider
    reasoning_llm.config.model = args.reasoning_model

    judge_llm = ChatLLMClient.from_env(
        prefix="JUDGE",
        default_provider=args.judge_provider,
        default_model=args.judge_model,
    )
    judge_llm.config.provider = args.judge_provider
    judge_llm.config.model = args.judge_model

    config = EvaluationConfig(
        methods=args.methods,
        async_mixed=args.async_mixed,
        max_questions=args.max_questions,
        max_context_chunks=args.max_contexts,
        chunk_top_k=args.chunk_top_k,
        entity_top_k=args.entity_top_k,
        relation_top_k=args.relation_top_k,
        hippo_seed_top_k=args.hippo_seed_top_k,
        hippo_ppr_entities=args.hippo_ppr_entities,
        hippo_hops=args.hippo_hops,
        max_chunks=args.max_chunks,
        max_concurrent_methods=args.max_concurrent_methods,
        answer_relevance_embedding_provider=args.answer_relevance_embedding_provider,
        answer_relevance_embedding_model=args.answer_relevance_embedding_model,
        output_dir=args.output_dir,
    )
    answer_generator = MedicalAnswerGenerator(
        llm_client=reasoning_llm,
        config=MedicalAnswerConfig(
            max_chunks=args.max_contexts,
            context_mode=args.answer_context_mode,
            enable_evidence_selection=not args.disable_evidence_selection,
            evidence_max_sentences_per_chunk=args.evidence_max_sentences_per_chunk,
            evidence_sentence_window_size=args.evidence_sentence_window_size,
            evidence_max_compact_chars_per_chunk=args.evidence_max_compact_chars_per_chunk,
        ),
    )
    answer_relevance_embedding_client = build_answer_relevance_embedding_client(
        provider=args.answer_relevance_embedding_provider,
        model_name=args.answer_relevance_embedding_model,
    )
    evaluator = RagasStyleEvaluator(
        judge_llm,
        embedding_client=answer_relevance_embedding_client,
        generated_question_count=config.ragas_generated_questions,
    )
    service = MedicalRAGEvaluationService(
        retrievers=bundle.retrievers,
        answer_generator=answer_generator,
        evaluator=evaluator,
        config=config,
    )

    dataset = (
        load_evaluation_questions(PROJECT_ROOT / args.dataset_file)
        if args.dataset_file
        else SMALL_MEDICAL_EVAL_SET
    )

    try:
        rows = await service.run(dataset)
        service.save(rows, PROJECT_ROOT / args.output_dir)
        for row in rows:
            score = row.evaluation.ragas_style_overall if row.evaluation else 0
            coverage = row.evaluation.answer_point_coverage if row.evaluation else 0
            source_hit5 = row.retrieval_metrics.source_hit_at_5 if row.retrieval_metrics else 0
            print(
                f"[DONE] {row.difficulty} | {row.method} | "
                f"ragas={score:.2f} | answer_points={coverage:.2f} | "
                f"source@5={source_hit5:.0f} | {row.question}"
            )
        print(f"[SAVED] {PROJECT_ROOT / args.output_dir}")
    finally:
        bundle.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()