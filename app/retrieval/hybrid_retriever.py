from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

from app.retrieval.lightrag_retriever import LightRAGRetriever
from app.retrieval.retrieval_models import RetrievalResult, RetrievedChunk
from app.retrieval.vector_retriever import VectorRetriever


@dataclass
class HybridRetrievalConfig:
    vector_top_k: int = 10
    max_chunks: int = 12
    rrf_k: int = 60


class HybridRetriever:
    """
    Hybrid-basic retriever = Vector RAG + LightRAG.

    Khi HippoRAGRetriever có sau, mixed_full sẽ cộng thêm nhánh HippoRAG ở đây.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        lightrag_retriever: LightRAGRetriever,
        config: HybridRetrievalConfig | None = None,
    ):
        self.vector_retriever = vector_retriever
        self.lightrag_retriever = lightrag_retriever
        self.config = config or HybridRetrievalConfig()

    def retrieve(self, query: str) -> RetrievalResult:
        vector_chunks = self.vector_retriever.search_chunks(
            query,
            top_k=self.config.vector_top_k,
        )
        lightrag_result = self.lightrag_retriever.retrieve(query)

        fused_chunks = self._rrf_fuse_chunks([
            vector_chunks,
            lightrag_result.chunks,
        ])[: self.config.max_chunks]

        return RetrievalResult(
            query=query,
            method="hybrid_basic",
            keywords=lightrag_result.keywords,
            chunks=fused_chunks,
            entities=lightrag_result.entities,
            relations=lightrag_result.relations,
            debug={
                "vector_chunk_count": len(vector_chunks),
                "lightrag_chunk_count": len(lightrag_result.chunks),
                "rrf_k": self.config.rrf_k,
                "note": "hybrid_basic = vector + lightrag; mixed_full will add hipporag later.",
            },
        )

    def _rrf_fuse_chunks(self, ranked_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        scores: dict[str, float] = defaultdict(float)
        best: dict[str, RetrievedChunk] = {}

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                scores[chunk.chunk_id] += 1.0 / (self.config.rrf_k + rank)
                old = best.get(chunk.chunk_id)
                if old is None or chunk.score > old.score:
                    best[chunk.chunk_id] = chunk

        result: list[RetrievedChunk] = []
        for chunk_id, chunk in best.items():
            chunk.score = scores[chunk_id]
            chunk.source = "mixed"
            result.append(chunk)

        return sorted(result, key=lambda x: x.score, reverse=True)
