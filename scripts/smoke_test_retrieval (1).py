from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.graph_database import GraphRepository, Neo4jClient
from app.infrastructure.vector_database import QdrantClientWrapper, VectorRepository
from app.retrieval import (
    GraphRetriever,
    HippoRAGRetrievalConfig,
    HippoRAGRetriever,
    HybridRetrievalConfig,
    HybridRetriever,
    LightRAGRetrievalConfig,
    LightRAGRetriever,
    QueryKeywordExtractor,
    RetrievalResult,
    VectorRetriever,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUERIES = [
    "viêm da cơ địa có triệu chứng gì",
    "chàm thể tạng có biểu hiện gì",
    "hen suyễn liên quan đến dị ứng như thế nào",
    "vảy nến có những dạng nào",
    "viêm mũi dị ứng có biến chứng gì",
]


def make_embedding_service() -> EmbeddingService:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    try:
        return EmbeddingService(model_name=model_name)
    except TypeError:
        try:
            return EmbeddingService(model=model_name)
        except TypeError:
            return EmbeddingService()


def get_embedding_dim(embedding: EmbeddingService) -> int:
    for attr in ["dim", "dimension", "embedding_dim", "vector_dim"]:
        value = getattr(embedding, attr, None)
        if value:
            return int(value)

    model = getattr(embedding, "model", None)
    if model is not None:
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())
        if hasattr(model, "get_sentence_embedding_dimension"):
            return int(model.get_sentence_embedding_dimension())

    return int(os.getenv("VECTOR_DIM", "1024"))


def make_graph_repo() -> tuple[Neo4jClient, GraphRepository]:
    client = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    repo = GraphRepository(client)
    if not repo.health_check():
        client.close()
        raise RuntimeError("Neo4j health check failed. Check Neo4j is running and .env is correct.")
    return client, repo


def make_vector_repo(embedding: EmbeddingService) -> tuple[QdrantClientWrapper, VectorRepository]:
    client = QdrantClientWrapper(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    repo = VectorRepository(
        client=client,
        vector_dim=get_embedding_dim(embedding),
        distance=os.getenv("VECTOR_DISTANCE", "cosine"),
    )
    if not repo.health_check():
        client.close()
        raise RuntimeError("Qdrant health check failed. Check Qdrant is running.")
    return client, repo


def print_result(result: RetrievalResult) -> None:
    print(f"\n========== {result.method.upper()} | {result.query} ==========")
    if result.keywords:
        print(f"low  = {result.keywords.low_level_keywords}")
        print(f"high = {result.keywords.high_level_keywords}")

    if result.debug:
        compact_debug = {
            key: value
            for key, value in result.debug.items()
            if key not in {"branch_debug"}
        }
        print("debug=", json.dumps(compact_debug, ensure_ascii=False))

    print("\n-- Top chunks --")
    for idx, c in enumerate(result.chunks[:8], start=1):
        print(f"{idx:02d}. score={c.score:.6f} | {c.title} | {c.section}")
        print(f"    chunk_id={c.chunk_id} | source={c.source}")
        text = (c.contextualized_text or c.text or "").replace("\n", " ")
        print(f"    {text[:260]}")

    print("\n-- Top entities --")
    for idx, e in enumerate(result.entities[:10], start=1):
        print(f"{idx:02d}. score={e.score:.6f} | {e.name} ({e.entity_type}) | {e.source}")

    print("\n-- Top relations --")
    for idx, r in enumerate(result.relations[:10], start=1):
        print(
            f"{idx:02d}. score={r.score:.6f} | "
            f"{r.subject_name} -[{r.relation_type}]-> {r.object_name} | {r.source}"
        )
        if r.evidence_text:
            print(f"    {r.evidence_text[:220]}")


def result_to_dict(result: RetrievalResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "method": result.method,
        "keywords": dataclasses.asdict(result.keywords) if result.keywords else None,
        "entities": [dataclasses.asdict(e) for e in result.entities],
        "relations": [dataclasses.asdict(r) for r in result.relations],
        "chunks": [dataclasses.asdict(c) for c in result.chunks],
        "debug": result.debug,
        "assembled_context_preview": result.assemble_context()[:6000],
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["vector", "lightrag", "hipporag", "mixed", "all"], default="all")
    parser.add_argument("--query", action="append", default=None, help="Repeatable custom query.")
    parser.add_argument("--no-llm-keywords", action="store_true", help="Use fallback keyword extractor.")
    parser.add_argument("--chunk-top-k", type=int, default=10)
    parser.add_argument("--entity-top-k", type=int, default=20)
    parser.add_argument("--relation-top-k", type=int, default=20)
    parser.add_argument("--hippo-seed-top-k", type=int, default=12)
    parser.add_argument("--hippo-ppr-entities", type=int, default=40)
    parser.add_argument("--hippo-hops", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--output", default="data/tmp/retrieval_smoke/retrieval_results.json")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    queries = args.query or DEFAULT_QUERIES

    print("[INIT] embedding")
    embedding = make_embedding_service()

    print("[INIT] Qdrant")
    qdrant_client, vector_repo = make_vector_repo(embedding)

    print("[INIT] Neo4j")
    neo4j_client, graph_repo = make_graph_repo()

    vector_retriever = VectorRetriever(vector_repo=vector_repo, embedding_service=embedding)
    graph_retriever = GraphRetriever(graph_repo=graph_repo)
    keyword_extractor = QueryKeywordExtractor(use_llm=not args.no_llm_keywords)

    lightrag_retriever = LightRAGRetriever(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        keyword_extractor=keyword_extractor,
        config=LightRAGRetrievalConfig(
            entity_top_k=args.entity_top_k,
            relation_top_k=args.relation_top_k,
            max_chunks=args.max_chunks,
        ),
    )
    hipporag_retriever = HippoRAGRetriever(
        graph_repo=graph_repo,
        vector_retriever=vector_retriever,
        keyword_extractor=keyword_extractor,
        config=HippoRAGRetrievalConfig(
            seed_top_k=args.hippo_seed_top_k,
            vector_seed_top_k=args.hippo_seed_top_k,
            ppr_top_k_entities=args.hippo_ppr_entities,
            adjacency_hops=args.hippo_hops,
            max_chunks=args.max_chunks,
        ),
    )
    mixed_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        lightrag_retriever=lightrag_retriever,
        hipporag_retriever=hipporag_retriever,
        config=HybridRetrievalConfig(
            vector_top_k=args.chunk_top_k,
            max_chunks=args.max_chunks,
        ),
    )

    records: list[dict[str, Any]] = []

    try:
        for query in queries:
            if args.method in {"vector", "all"}:
                result = vector_retriever.retrieve(query, top_k=args.chunk_top_k)
                print_result(result)
                records.append(result_to_dict(result))

            if args.method in {"lightrag", "all"}:
                result = lightrag_retriever.retrieve(query)
                print_result(result)
                records.append(result_to_dict(result))

            if args.method in {"hipporag", "all"}:
                result = hipporag_retriever.retrieve(query)
                print_result(result)
                records.append(result_to_dict(result))

            if args.method in {"mixed", "all"}:
                result = mixed_retriever.retrieve(query)
                print_result(result)
                records.append(result_to_dict(result))

        save_json(PROJECT_ROOT / args.output, records)
        print(f"\n[SAVED] {PROJECT_ROOT / args.output}")
    finally:
        qdrant_client.close()
        neo4j_client.close()


if __name__ == "__main__":
    main()
