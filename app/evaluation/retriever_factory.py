from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.graph_database import GraphRepository, Neo4jClient
from app.infrastructure.vector_database import NAIVE_CHUNK_COLLECTION, QdrantClientWrapper, VectorRepository
from app.retrieval import (
    GraphRetriever,
    HippoRAGRetrievalConfig,
    HippoRAGRetriever,
    HybridRetrievalConfig,
    HybridRetriever,
    LightRAGRetrievalConfig,
    LightRAGRetriever,
    QueryKeywordExtractor,
    VectorRetriever,
)


@dataclass
class RetrieverBuildConfig:
    no_llm_keywords: bool = False
    entity_top_k: int = 20
    relation_top_k: int = 20
    chunk_top_k: int = 10
    hippo_seed_top_k: int = 12
    hippo_ppr_entities: int = 40
    hippo_hops: int = 2
    max_chunks: int = 12


@dataclass
class RetrieverBundle:
    retrievers: dict[str, Any]
    clients: list[Any]

    def close(self) -> None:
        for client in self.clients:
            close = getattr(client, "close", None)
            if close:
                close()


def make_embedding_service() -> EmbeddingService:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    return EmbeddingService(model_name=model_name)


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


def build_retriever_bundle(config: RetrieverBuildConfig) -> RetrieverBundle:
    embedding = make_embedding_service()
    qdrant_client, vector_repo = make_vector_repo(embedding)
    neo4j_client, graph_repo = make_graph_repo()

    naive_count = vector_repo.count_points(NAIVE_CHUNK_COLLECTION)
    if naive_count <= 0:
        raise RuntimeError(
            f"Naive vector collection '{NAIVE_CHUNK_COLLECTION}' is empty or missing. "
            "Run: python -m scripts.build_naive_chunk_vectors --recreate"
        )

    vector_retriever = VectorRetriever(
        vector_repo=vector_repo,
        embedding_service=embedding,
        chunk_collection_name=NAIVE_CHUNK_COLLECTION,
        method_name="vector_naive",
        vector_text_mode="raw_chunk_text",
    )
    actual_collection = getattr(vector_retriever, "chunk_collection_name", None)
    actual_mode = getattr(vector_retriever, "vector_text_mode", None)
    if actual_collection != NAIVE_CHUNK_COLLECTION or actual_mode != "raw_chunk_text":
        raise RuntimeError(
            "VectorRetriever is not configured for real naive RAG. "
            "Make sure app/retrieval/vector_retriever.py is updated from the latest patch. "
            f"Got collection={actual_collection}, mode={actual_mode}; "
            f"expected collection={NAIVE_CHUNK_COLLECTION}, mode=raw_chunk_text."
        )
    graph_retriever = GraphRetriever(graph_repo=graph_repo)
    keyword_extractor = QueryKeywordExtractor(use_llm=not config.no_llm_keywords)

    lightrag_retriever = LightRAGRetriever(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        keyword_extractor=keyword_extractor,
        config=LightRAGRetrievalConfig(
            entity_top_k=config.entity_top_k,
            relation_top_k=config.relation_top_k,
            max_chunks=config.max_chunks,
        ),
    )
    hipporag_retriever = HippoRAGRetriever(
        graph_repo=graph_repo,
        vector_retriever=vector_retriever,
        keyword_extractor=keyword_extractor,
        config=HippoRAGRetrievalConfig(
            seed_top_k=config.hippo_seed_top_k,
            vector_seed_top_k=config.hippo_seed_top_k,
            ppr_top_k_entities=config.hippo_ppr_entities,
            adjacency_hops=config.hippo_hops,
            max_chunks=config.max_chunks,
        ),
    )
    mixed_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        lightrag_retriever=lightrag_retriever,
        hipporag_retriever=hipporag_retriever,
        config=HybridRetrievalConfig(
            vector_top_k=config.chunk_top_k,
            max_chunks=config.max_chunks,
        ),
    )

    return RetrieverBundle(
        retrievers={
            "vector": vector_retriever,
            "vector_naive": vector_retriever,
            "lightrag": lightrag_retriever,
            "hipporag": hipporag_retriever,
            "mixed": mixed_retriever,
        },
        clients=[qdrant_client, neo4j_client],
    )
