from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.graph_database import GraphRepository, Neo4jClient
from app.infrastructure.vector_database import NAIVE_CHUNK_COLLECTION, QdrantClientWrapper, VectorRepository
from app.retrieval import (
    CrossEncoderReranker,
    CrossEncoderRerankingConfig,
    GraphRetriever,
    HippoRAGRetrievalConfig,
    HippoRAGRetriever,
    HybridRetrievalConfig,
    HybridRetriever,
    LightRAGRetrievalConfig,
    LightRAGRetriever,
    QueryKeywordExtractor,
    RerankedRetriever,
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

    # Cross-encoder reranking is opt-in through vector_ce / mixed_ce. These
    # values can be overridden by env variables without changing code.
    cross_encoder_model: str = ""
    rerank_candidate_top_k: int = 50
    rerank_final_top_k: int = 0
    rerank_batch_size: int = 16
    rerank_max_text_chars: int = 2400
    rerank_cross_encoder_weight: float = 0.35
    rerank_retrieval_weight: float = 0.65
    rerank_max_chunks_per_source: int = 3
    rerank_fail_open: bool = True
    rerank_preserve_pre_top_k: int = 2
    rerank_preserve_pre_at_top: bool = True
    rerank_preserve_at_top_max_ce_rank: int = 5


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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def make_cross_encoder_config(config: RetrieverBuildConfig) -> CrossEncoderRerankingConfig:
    final_top_k = config.rerank_final_top_k or config.max_chunks
    return CrossEncoderRerankingConfig(
        model_name=os.getenv("CROSS_ENCODER_MODEL", config.cross_encoder_model),
        candidate_top_k=_env_int("RERANK_CANDIDATE_TOP_K", config.rerank_candidate_top_k),
        final_top_k=_env_int("RERANK_FINAL_TOP_K", final_top_k),
        batch_size=_env_int("RERANK_BATCH_SIZE", config.rerank_batch_size),
        max_text_chars=_env_int("RERANK_MAX_TEXT_CHARS", config.rerank_max_text_chars),
        cross_encoder_weight=_env_float("RERANK_CROSS_ENCODER_WEIGHT", config.rerank_cross_encoder_weight),
        retrieval_weight=_env_float("RERANK_RETRIEVAL_WEIGHT", config.rerank_retrieval_weight),
        max_chunks_per_source=_env_int("RERANK_MAX_CHUNKS_PER_SOURCE", config.rerank_max_chunks_per_source),
        fail_open=_env_bool("RERANK_FAIL_OPEN", config.rerank_fail_open),
        device=os.getenv("CROSS_ENCODER_DEVICE", ""),
        preserve_pre_rerank_top_k=_env_int("RERANK_PRESERVE_PRE_TOP_K", config.rerank_preserve_pre_top_k),
        preserve_pre_rerank_at_top=_env_bool("RERANK_PRESERVE_PRE_AT_TOP", config.rerank_preserve_pre_at_top),
        preserve_at_top_max_cross_encoder_rank=_env_int("RERANK_PRESERVE_AT_TOP_MAX_CE_RANK", config.rerank_preserve_at_top_max_ce_rank),
    )


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
            preserve_lightrag_top_k=2,
            preserve_hipporag_top_k=2,
        ),
    )

    rerank_config = make_cross_encoder_config(config)
    cross_encoder_reranker = CrossEncoderReranker(config=rerank_config)

    # Candidate-pool retrievers for CE methods. They reuse the same Qdrant,
    # Neo4j and embedding clients, but return more candidates before the final
    # cross-encoder stage. The original methods above remain unchanged.
    rerank_candidate_top_k = max(rerank_config.candidate_top_k, config.max_chunks)

    lightrag_candidate_retriever = LightRAGRetriever(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        keyword_extractor=keyword_extractor,
        config=LightRAGRetrievalConfig(
            entity_top_k=config.entity_top_k,
            relation_top_k=config.relation_top_k,
            max_chunks=rerank_candidate_top_k,
            graph_max_chunks=max(50, rerank_candidate_top_k),
        ),
    )
    hipporag_candidate_retriever = HippoRAGRetriever(
        graph_repo=graph_repo,
        vector_retriever=vector_retriever,
        keyword_extractor=keyword_extractor,
        config=HippoRAGRetrievalConfig(
            seed_top_k=config.hippo_seed_top_k,
            vector_seed_top_k=config.hippo_seed_top_k,
            ppr_top_k_entities=config.hippo_ppr_entities,
            adjacency_hops=config.hippo_hops,
            max_chunks=rerank_candidate_top_k,
        ),
    )
    mixed_candidate_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        lightrag_retriever=lightrag_candidate_retriever,
        hipporag_retriever=hipporag_candidate_retriever,
        config=HybridRetrievalConfig(
            vector_top_k=rerank_candidate_top_k,
            max_chunks=rerank_candidate_top_k,
            preserve_vector_top_k=2,
            preserve_lightrag_top_k=2,
            preserve_hipporag_top_k=2,
        ),
    )

    vector_ce_retriever = RerankedRetriever(
        base_retriever=vector_retriever,
        reranker=cross_encoder_reranker,
        method_name="vector_ce",
        config=rerank_config,
    )
    mixed_ce_retriever = RerankedRetriever(
        base_retriever=mixed_candidate_retriever,
        reranker=cross_encoder_reranker,
        method_name="mixed_ce",
        config=rerank_config,
    )

    return RetrieverBundle(
        retrievers={
            "vector": vector_retriever,
            "vector_naive": vector_retriever,
            "vector_ce": vector_ce_retriever,
            "lightrag": lightrag_retriever,
            "hipporag": hipporag_retriever,
            "mixed": mixed_retriever,
            "mixed_ce": mixed_ce_retriever,
        },
        clients=[qdrant_client, neo4j_client],
    )
