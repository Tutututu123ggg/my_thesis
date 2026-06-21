from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.graph_database import GraphRepository, Neo4jClient
from app.infrastructure.vector_database import QdrantClientWrapper, VectorRepository
from app.ingestion.builders import GraphBuilder, MedicalGraphBuilder, VectorGraphBuilder
from app.ingestion.extraction import (
    ArticleExtractionPipeline,
    ArticleKeywordProfiler,
    ExtractionValidator,
    MedicalExtractor,
)
from app.ingestion.processing import chunk_article, load_markdown_article
from app.ingestion.resolution import EntityResolver


def build_graph_repo() -> tuple[Neo4jClient, GraphRepository]:
    client = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "admin123456"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    return client, GraphRepository(client)


def build_vector_repo(embedder: EmbeddingService) -> tuple[QdrantClientWrapper, VectorRepository]:
    client = QdrantClientWrapper(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    repo = VectorRepository(
        client=client,
        vector_dim=embedder.vector_dim,
        distance=os.getenv("VECTOR_DISTANCE", "cosine"),
    )
    return client, repo


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_file(
    path: Path,
    *,
    args: argparse.Namespace,
    graph_builder: GraphBuilder,
    medical_graph_builder: MedicalGraphBuilder,
    vector_builder: VectorGraphBuilder,
    extraction_pipeline: ArticleExtractionPipeline,
    extraction_jsonl: Path,
) -> dict[str, Any]:
    article = load_markdown_article(path)
    retrieval_chunks = chunk_article(article, max_tokens=args.max_tokens)

    if not args.skip_graph_document:
        # Store retrieval chunks as normal Article/Chunk nodes.
        graph_builder.ingest_article_with_chunks(article, retrieval_chunks)

    chunk_vector_count = 0
    naive_chunk_vector_count = 0
    if not args.skip_vectors:
        # Vector RAG remains based on retrieval chunks, not tiny extraction windows.
        chunk_vector_count = vector_builder.upsert_chunks(retrieval_chunks)
        naive_chunk_vector_count = vector_builder.upsert_naive_chunks(retrieval_chunks)

    stats = {
        "file_path": str(path),
        "article_id": article.article_id,
        "title": article.title,
        "url": article.url,
        "chunks": len(retrieval_chunks),
        "extraction_windows": 0,
        "split_parent_chunks": 0,
        "chunk_vectors": chunk_vector_count,
        "naive_chunk_vectors": naive_chunk_vector_count,
        "medical_chunks": 0,
        "skipped_chunks": 0,
        "entities": 0,
        "relations": 0,
        "entity_writes": 0,
        "relation_writes": 0,
        "entity_vectors": 0,
        "relation_vectors": 0,
        "keyword_profile_llm_updates": 0,
    }

    if args.skip_extract:
        return stats

    def progress(i: int, total: int, window) -> None:
        print(f"[EXTRACT] {article.title} | window {i}/{total} {window.chunk_id}", flush=True)

    result = extraction_pipeline.extract_article(
        article=article,
        retrieval_chunks=retrieval_chunks,
        max_windows=args.max_extraction_windows,
        progress_callback=progress,
    )

    # If a retrieval chunk was split, store the fact windows too so evidence
    # chunk ids in relations point to concrete text nodes.
    if not args.skip_graph_document:
        extra_windows = result.extra_extraction_windows
        if extra_windows:
            graph_builder.upsert_chunks(extra_windows)

    stats["extraction_windows"] = len(result.extraction_windows)
    stats["split_parent_chunks"] = int(result.windowing_debug.get("split_parent_chunk_count", 0) or 0)
    stats["keyword_profile_llm_updates"] = int(result.keyword_profile_debug.get("llm_update_count", 0) or 0)

    for window, raw_graph, graph in zip(
        result.extraction_windows,
        result.raw_graphs,
        result.validated_graphs,
        strict=True,
    ):
        if graph.should_ingest_graph:
            stats["medical_chunks"] += 1
        else:
            stats["skipped_chunks"] += 1

        stats["entities"] += len(graph.entities)
        stats["relations"] += len(graph.relations)

        graph_write = medical_graph_builder.write_chunk_extraction(window, graph)
        vector_write = vector_builder.upsert_chunk_extraction(window, graph) if not args.skip_vectors else {}

        stats["entity_writes"] += int(graph_write.get("entity_write_count", 0) or 0)
        stats["relation_writes"] += int(graph_write.get("relation_write_count", 0) or 0)
        stats["entity_vectors"] += int(vector_write.get("entity_vector_count", 0) or 0)
        stats["relation_vectors"] += int(vector_write.get("relation_vector_count", 0) or 0)

        write_jsonl(
            extraction_jsonl,
            {
                "chunk_id": window.chunk_id,
                "article_id": window.article_id,
                "title": window.title,
                "section": window.section,
                "subsection": window.subsection,
                "source_url": window.source_url,
                "metadata": window.metadata,
                "is_extraction_window": bool(window.metadata.get("extraction_window")),
                "parent_chunk_id": window.metadata.get("parent_chunk_id"),
                "windowing_debug": result.windowing_debug,
                "is_medical_knowledge_chunk": graph.is_medical_knowledge_chunk,
                "should_ingest_graph": graph.should_ingest_graph,
                "skip_reason": graph.skip_reason,
                "raw_entities": [e.model_dump() for e in raw_graph.entities],
                "raw_relations": [r.model_dump() for r in raw_graph.relations],
                "entities": [e.model_dump() for e in graph.entities],
                "relations": [r.model_dump() for r in graph.relations],
                "keyword_profile_debug": result.keyword_profile_debug,
                "graph_write": graph_write,
                "vector_write": vector_write,
            },
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest final cleaned Vietnamese medical markdown corpus into Neo4j/Qdrant."
    )
    parser.add_argument("--input-dir", default="data/raw/articles_final_v1")
    parser.add_argument("--output-dir", default="data/processed/ingest_final_v1")
    parser.add_argument("--max-tokens", type=int, default=900, help="Retrieval/vector chunk size.")
    parser.add_argument("--extraction-window-tokens", type=int, default=450, help="Fact-window size for LLM extraction only.")
    parser.add_argument("--extraction-window-min-tokens", type=int, default=80)
    parser.add_argument("--extraction-window-overlap-sentences", type=int, default=1)
    parser.add_argument("--disable-extraction-windows", action="store_true")
    parser.add_argument("--max-extraction-windows", type=int, default=0, help="Debug cap per article. 0 = no cap.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-vectors", action="store_true")
    parser.add_argument("--skip-graph-document", action="store_true")
    parser.add_argument("--keyword-model", default=None, help="Optional article-level keyword profiler model. Empty = deterministic only.")
    parser.add_argument("--skip-keyword-profiler", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    extraction_jsonl = output_dir / "chunk_extractions.jsonl"
    article_stats_jsonl = output_dir / "article_stats.jsonl"

    files = sorted(input_dir.glob("*.md"))
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        raise FileNotFoundError(f"No markdown files found in {input_dir}")

    graph_client = None
    qdrant_client = None
    try:
        graph_client, graph_repo = build_graph_repo()
        if not graph_repo.health_check():
            raise RuntimeError("Neo4j health check failed")
        graph_repo.setup_schema()
        if args.fresh_start:
            graph_repo.delete_all_data()
            graph_repo.setup_schema()

        embedder = EmbeddingService()
        qdrant_client, vector_repo = build_vector_repo(embedder)
        if not vector_repo.health_check():
            raise RuntimeError("Qdrant health check failed")
        vector_builder = VectorGraphBuilder(
            vector_repo=vector_repo,
            embedder=embedder,
            resolver=EntityResolver(graph_repo),
            batch_size=32,
        )
        vector_builder.setup_collections(recreate=args.fresh_start)

        resolver = EntityResolver(graph_repo)
        graph_builder = GraphBuilder(graph_repo)
        medical_graph_builder = MedicalGraphBuilder(graph_repo, resolver)
        keyword_profiler = None if args.skip_keyword_profiler else ArticleKeywordProfiler(model=args.keyword_model)
        extraction_pipeline = ArticleExtractionPipeline(
            extractor=MedicalExtractor(model=args.model),
            validator=ExtractionValidator(),
            keyword_profiler=keyword_profiler,
            use_fact_windows=not args.disable_extraction_windows,
            fact_window_max_tokens=args.extraction_window_tokens,
            fact_window_min_tokens=args.extraction_window_min_tokens,
            fact_window_overlap_sentences=args.extraction_window_overlap_sentences,
        )

        totals: dict[str, int] = {
            "articles": 0,
            "chunks": 0,
            "extraction_windows": 0,
            "split_parent_chunks": 0,
            "medical_chunks": 0,
            "skipped_chunks": 0,
            "entities": 0,
            "relations": 0,
            "entity_writes": 0,
            "relation_writes": 0,
            "chunk_vectors": 0,
            "naive_chunk_vectors": 0,
            "entity_vectors": 0,
            "relation_vectors": 0,
            "keyword_profile_llm_updates": 0,
        }

        for index, path in enumerate(files, start=1):
            stats = process_file(
                path,
                args=args,
                graph_builder=graph_builder,
                medical_graph_builder=medical_graph_builder,
                vector_builder=vector_builder,
                extraction_pipeline=extraction_pipeline,
                extraction_jsonl=extraction_jsonl,
            )
            write_jsonl(article_stats_jsonl, stats)

            totals["articles"] += 1
            for key in totals:
                if key == "articles":
                    continue
                totals[key] += int(stats.get(key, 0) or 0)

            print(
                f"[{index}/{len(files)}] {stats['title']} | "
                f"chunks={stats['chunks']} windows={stats['extraction_windows']} "
                f"medical={stats['medical_chunks']} skip={stats['skipped_chunks']} "
                f"entities={stats['entities']} relations={stats['relations']}",
                flush=True,
            )

        summary = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "max_tokens": args.max_tokens,
            "extraction_window_tokens": args.extraction_window_tokens,
            "extraction_window_enabled": not args.disable_extraction_windows,
            "model": args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "keyword_model": (args.keyword_model or "deterministic_only") if not args.skip_keyword_profiler else "disabled",
            "embedding_model": embedder.model_name,
            "embedding_dim": embedder.vector_dim,
            "totals": totals,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    finally:
        if graph_client is not None:
            graph_client.close()
        if qdrant_client is not None:
            qdrant_client.close()


if __name__ == "__main__":
    main()
