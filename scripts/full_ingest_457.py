#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ingestion.extraction.pipelines.full_corpus_ingest_pipeline import (
    FullCorpusIngestConfig,
    run_full_corpus_ingest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe full ingest for 457 Vietnamese medical articles into Neo4j + Qdrant."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--md-root", default=".")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--fact-window-max-tokens", type=int, default=450)
    parser.add_argument("--fact-window-min-tokens", type=int, default=80)
    parser.add_argument("--fact-window-overlap-sentences", type=int, default=1)

    parser.add_argument("--entity-max-tokens", type=int, default=2500)
    parser.add_argument("--relation-max-tokens", type=int, default=4000)

    parser.add_argument("--keyword-llm", action="store_true")
    parser.add_argument("--keyword-model", default="gpt-4o-mini")

    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--llm-workers", type=int, default=3)
    parser.add_argument("--llm-retries", type=int, default=3)

    parser.add_argument("--checkpoint-jsonl", default="")

    parser.add_argument("--no-recreate", action="store_true", help="Default behavior. Never clears Neo4j/Qdrant.")
    parser.add_argument("--recreate", action="store_true", help="Clear Neo4j/Qdrant only with --confirm-clear token.")
    parser.add_argument("--confirm-clear", default="")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.recreate and args.no_recreate:
        raise SystemExit("Use either --recreate or --no-recreate, not both.")

    config = FullCorpusIngestConfig(
        input_csv=Path(args.input_csv),
        md_root=Path(args.md_root),
        out_dir=Path(args.out_dir),
        run_id=args.run_id,
        start_index=args.start_index,
        limit=args.limit,
        resume=args.resume,
        dry_run=args.dry_run,
        max_tokens=args.max_tokens,
        fact_window_max_tokens=args.fact_window_max_tokens,
        fact_window_min_tokens=args.fact_window_min_tokens,
        fact_window_overlap_sentences=args.fact_window_overlap_sentences,
        entity_max_tokens=args.entity_max_tokens,
        relation_max_tokens=args.relation_max_tokens,
        keyword_llm=args.keyword_llm,
        keyword_model=args.keyword_model,
        embedding_batch_size=args.embedding_batch_size,
        llm_workers=args.llm_workers,
        llm_retries=args.llm_retries,
        checkpoint_jsonl=Path(args.checkpoint_jsonl) if args.checkpoint_jsonl else None,
        recreate=args.recreate,
        confirm_clear=args.confirm_clear,
    )

    summary = run_full_corpus_ingest(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
