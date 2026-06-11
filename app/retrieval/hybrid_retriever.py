from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.retrieval.hipporag_retriever import HippoRAGRetriever
from app.retrieval.lightrag_retriever import LightRAGRetriever
from app.retrieval.retrieval_models import (
    QueryKeywords,
    RetrievalResult,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelation,
)
from app.retrieval.vector_retriever import VectorRetriever


@dataclass
class HybridRetrievalConfig:
    vector_top_k: int = 10
    max_chunks: int = 12
    max_entities: int = 40
    max_relations: int = 60
    rrf_k: int = 60


class HybridRetriever:
    """
    mixed retriever = Vector RAG + LightRAG + HippoRAG, fused at the end by RRF.

    The class intentionally requires a HippoRAGRetriever. Without the PPR branch,
    this is not exposed as mixed mode.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        lightrag_retriever: LightRAGRetriever,
        hipporag_retriever: HippoRAGRetriever,
        config: HybridRetrievalConfig | None = None,
    ):
        self.vector_retriever = vector_retriever
        self.lightrag_retriever = lightrag_retriever
        self.hipporag_retriever = hipporag_retriever
        self.config = config or HybridRetrievalConfig()

    def retrieve(self, query: str) -> RetrievalResult:
        vector_result = self.vector_retriever.retrieve(
            query,
            top_k=self.config.vector_top_k,
        )
        lightrag_result = self.lightrag_retriever.retrieve(query)
        hipporag_result = self.hipporag_retriever.retrieve(query)

        fused_chunks = self._rrf_fuse_chunks([
            vector_result.chunks,
            lightrag_result.chunks,
            hipporag_result.chunks,
        ])[: self.config.max_chunks]
        entities = self._merge_entities(lightrag_result.entities + hipporag_result.entities)
        relations = self._merge_relations(lightrag_result.relations + hipporag_result.relations)

        return RetrievalResult(
            query=query,
            method="mixed",
            keywords=self._merge_keywords(lightrag_result.keywords, hipporag_result.keywords),
            chunks=fused_chunks,
            entities=entities[: self.config.max_entities],
            relations=relations[: self.config.max_relations],
            debug={
                "vector_chunk_count": len(vector_result.chunks),
                "lightrag_chunk_count": len(lightrag_result.chunks),
                "hipporag_chunk_count": len(hipporag_result.chunks),
                "rrf_k": self.config.rrf_k,
                "fusion": "RRF over vector, lightrag, hipporag chunk rankings",
                "branch_debug": {
                    "vector": vector_result.debug,
                    "lightrag": lightrag_result.debug,
                    "hipporag": hipporag_result.debug,
                },
            },
        )

    def _rrf_fuse_chunks(self, ranked_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        scores: dict[str, float] = defaultdict(float)
        best: dict[str, RetrievedChunk] = {}
        sources: dict[str, list[str]] = defaultdict(list)

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                scores[chunk.chunk_id] += 1.0 / (self.config.rrf_k + rank)
                if chunk.source not in sources[chunk.chunk_id]:
                    sources[chunk.chunk_id].append(str(chunk.source))
                old = best.get(chunk.chunk_id)
                if old is None or chunk.score > old.score:
                    best[chunk.chunk_id] = chunk

        result: list[RetrievedChunk] = []
        for chunk_id, chunk in best.items():
            fused = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                score=scores[chunk_id],
                source="mixed",
                article_id=chunk.article_id,
                title=chunk.title,
                section=chunk.section,
                subsection=chunk.subsection,
                source_url=chunk.source_url,
                text=chunk.text,
                contextualized_text=chunk.contextualized_text,
                metadata={
                    **chunk.metadata,
                    "mixed_sources": sources[chunk_id],
                    "best_branch_score": chunk.score,
                },
            )
            result.append(fused)

        return sorted(result, key=lambda x: x.score, reverse=True)

    def _merge_entities(self, entities: list[RetrievedEntity]) -> list[RetrievedEntity]:
        by_id: dict[str, RetrievedEntity] = {}
        for entity in entities:
            old = by_id.get(entity.entity_id)
            if old is None or entity.score > old.score:
                by_id[entity.entity_id] = entity
        return sorted(by_id.values(), key=lambda item: item.score, reverse=True)

    def _merge_relations(self, relations: list[RetrievedRelation]) -> list[RetrievedRelation]:
        by_id: dict[str, RetrievedRelation] = {}
        for relation in relations:
            old = by_id.get(relation.relation_id)
            if old is None or relation.score > old.score:
                by_id[relation.relation_id] = relation
        return sorted(by_id.values(), key=lambda item: item.score, reverse=True)

    def _merge_keywords(
        self,
        left: QueryKeywords | None,
        right: QueryKeywords | None,
    ) -> QueryKeywords | None:
        if left is None and right is None:
            return None
        low: list[str] = []
        high: list[str] = []
        for keywords in [left, right]:
            if keywords is None:
                continue
            for item in keywords.low_level_keywords:
                if item not in low:
                    low.append(item)
            for item in keywords.high_level_keywords:
                if item not in high:
                    high.append(item)
        return QueryKeywords(low_level_keywords=low, high_level_keywords=high)
