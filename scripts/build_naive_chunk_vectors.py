from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.graph_database import GraphRepository, Neo4jClient
from app.infrastructure.vector_database import (
    ChunkVectorPayload,
    NAIVE_CHUNK_COLLECTION,
    QdrantClientWrapper,
    VectorPoint,
    VectorRepository,
    make_qdrant_point_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_embedding_service() -> EmbeddingService:
    return EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))


def get_embedding_dim(embedding: EmbeddingService) -> int:
    value = getattr(embedding, "vector_dim", None)
    if value:
        return int(value)
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
        raise RuntimeError("Neo4j health check failed. Check Neo4j and .env.")
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
        raise RuntimeError("Qdrant health check failed. Check Qdrant and .env.")
    return client, repo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build real naive chunk vectors from raw chunk.text into medical_chunks_naive."
    )
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    embedding = make_embedding_service()
    qdrant_client, vector_repo = make_vector_repo(embedding)
    neo4j_client, graph_repo = make_graph_repo()

    try:
        print(f"[SETUP] ensure collection={NAIVE_CHUNK_COLLECTION} recreate={args.recreate}")
        vector_repo.ensure_collection(NAIVE_CHUNK_COLLECTION, recreate=args.recreate)

        chunks = graph_repo.get_all_chunks(limit=args.limit)
        print(f"[LOAD] chunks={len(chunks)} from Neo4j")

        total = 0
        for start in range(0, len(chunks), args.batch_size):
            batch = chunks[start : start + args.batch_size]
            texts = [chunk.text or "" for chunk in batch]
            vectors = embedding.embed_texts(texts, batch_size=args.batch_size)

            points = []
            for chunk, vector in zip(batch, vectors, strict=True):
                if not chunk.text:
                    continue
                points.append(
                    VectorPoint(
                        point_id=make_qdrant_point_id(f"naive_chunk::{chunk.chunk_id}"),
                        vector=vector,
                        payload=ChunkVectorPayload(
                            chunk_id=chunk.chunk_id,
                            article_id=chunk.article_id,
                            source_url=chunk.source_url,
                            title=chunk.title,
                            section=chunk.section,
                            subsection=chunk.subsection,
                            chunk_index=chunk.chunk_index,
                            token_count=chunk.token_count,
                            text=chunk.text,
                            contextualized_text=chunk.contextualized_text,
                            metadata={
                                **(chunk.metadata or {}),
                                "vector_text_mode": "raw_chunk_text",
                                "vector_collection": NAIVE_CHUNK_COLLECTION,
                            },
                        ).model_dump(),
                    )
                )

            vector_repo.upsert_naive_chunk_vectors(points)
            total += len(points)
            print(f"[UPSERT] {total}/{len(chunks)}")

        print(f"[DONE] upserted {total} raw chunk vectors into {NAIVE_CHUNK_COLLECTION}")
    finally:
        qdrant_client.close()
        neo4j_client.close()


if __name__ == "__main__":
    main()
