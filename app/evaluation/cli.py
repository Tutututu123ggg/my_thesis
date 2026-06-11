from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from app.evaluation.datasets import SMALL_MEDICAL_EVAL_SET, load_evaluation_questions
from app.evaluation.judge import RagasStyleEvaluator
from app.evaluation.models import EvaluationConfig
from app.evaluation.retriever_factory import RetrieverBuildConfig, build_retriever_bundle
from app.evaluation.service import MedicalRAGEvaluationService
from app.reasoning import ChatLLMClient, MedicalAnswerConfig, MedicalAnswerGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Medical RAG reasoning + LLM judge evaluation.")
    parser.add_argument("--methods", nargs="+", choices=["vector", "vector_naive", "lightrag", "hipporag", "mixed"], default=["vector_naive", "lightrag", "hipporag", "mixed"])
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
    parser.add_argument("--max-concurrent-methods", type=int, default=4)
    parser.add_argument("--answer-context-mode", choices=["chunks_only", "full_graph"], default="chunks_only")
    parser.add_argument("--output-dir", default="data/evaluation/ragas_style_small")

    parser.add_argument("--reasoning-provider", choices=["openai", "ollama"], default=os.getenv("REASONING_PROVIDER", "ollama"))
    parser.add_argument("--reasoning-model", default=os.getenv("REASONING_MODEL", "qwen2.5:3b"))
    parser.add_argument("--judge-provider", choices=["openai", "ollama"], default=os.getenv("JUDGE_PROVIDER", "openai"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
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
        output_dir=args.output_dir,
    )
    answer_generator = MedicalAnswerGenerator(
        llm_client=reasoning_llm,
        config=MedicalAnswerConfig(
            max_chunks=args.max_contexts,
            context_mode=args.answer_context_mode,
        ),
    )
    evaluator = RagasStyleEvaluator(
        judge_llm,
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
            hit5 = row.retrieval_metrics.hit_at_5 if row.retrieval_metrics else 0
            print(f"[DONE] {row.difficulty} | {row.method} | ragas={score:.2f} | hit@5={hit5:.0f} | {row.question}")
        print(f"[SAVED] {PROJECT_ROOT / args.output_dir}")
    finally:
        bundle.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
