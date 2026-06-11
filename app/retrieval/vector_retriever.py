from __future__ import annotations

from typing import Any

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.vector_database import VectorRepository
from app.infrastructure.vector_database.vector_models import VectorSearchResult
from app.retrieval.retrieval_models import (
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelation,
)


class VectorRetriever:
    """
    Thin wrapper quanh Qdrant.

    Dùng cho:
    - Vector RAG: search medical_chunks.
    - LightRAG: search medical_entities / medical_relations bằng low/high keywords.
    """

    def __init__(
        self,
        vector_repo: VectorRepository,
        embedding_service: EmbeddingService,
    ):
        self.vector_repo = vector_repo
        self.embedding_service = embedding_service

    def search_chunks(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        vector = self._embed(query)
        results = self.vector_repo.search_chunks(vector, limit=top_k, filters=filters)
        return [self._chunk_from_result(r) for r in results]

    def search_entities(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedEntity]:
        vector = self._embed(query)
        results = self.vector_repo.search_entities(vector, limit=top_k, filters=filters)
        return [self._entity_from_result(r) for r in results]

    def search_relations(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedRelation]:
        vector = self._embed(query)
        results = self.vector_repo.search_relations(vector, limit=top_k, filters=filters)
        return [self._relation_from_result(r) for r in results]

    def _embed(self, text: str) -> list[float]:
        if hasattr(self.embedding_service, "embed_text"):
            return self.embedding_service.embed_text(text)
        if hasattr(self.embedding_service, "embed_texts"):
            return self.embedding_service.embed_texts([text])[0]
        raise AttributeError("EmbeddingService must provide embed_text() or embed_texts().")

    def _chunk_from_result(self, result: VectorSearchResult) -> RetrievedChunk:
        p = result.payload
        return RetrievedChunk(
            chunk_id=str(p.get("chunk_id") or result.point_id),
            score=result.score,
            source="vector_chunk",
            article_id=str(p.get("article_id") or ""),
            title=str(p.get("title") or ""),
            section=str(p.get("section") or ""),
            subsection=str(p.get("subsection") or ""),
            source_url=str(p.get("source_url") or ""),
            text=str(p.get("text") or ""),
            contextualized_text=str(p.get("contextualized_text") or p.get("text") or ""),
            metadata=dict(p),
        )

    def _entity_from_result(self, result: VectorSearchResult) -> RetrievedEntity:
        p = result.payload
        return RetrievedEntity(
            entity_id=str(p.get("entity_id") or result.point_id),
            score=result.score,
            source="lightrag_entity",
            name=str(p.get("name") or ""),
            normalized_name=str(p.get("normalized_name") or ""),
            entity_type=str(p.get("entity_type") or ""),
            description=str(p.get("description") or ""),
            profile_text=str(p.get("profile_text") or p.get("description") or ""),
            aliases=list(p.get("aliases") or []),
            surface_forms=list(p.get("surface_forms") or []),
            metadata=dict(p),
        )

    def _relation_from_result(self, result: VectorSearchResult) -> RetrievedRelation:
        p = result.payload
        return RetrievedRelation(
            relation_id=str(p.get("relation_id") or result.point_id),
            score=result.score,
            source="lightrag_relation",
            relation_type=str(p.get("relation_type") or ""),
            subject_entity_id=str(p.get("subject_entity_id") or ""),
            subject_name=str(p.get("subject_name") or p.get("subject") or ""),
            subject_type=str(p.get("subject_type") or ""),
            object_entity_id=str(p.get("object_entity_id") or ""),
            object_name=str(p.get("object_name") or p.get("object") or ""),
            object_type=str(p.get("object_type") or ""),
            description=str(p.get("description") or ""),
            keywords=list(p.get("keywords") or []),
            evidence_text=str(p.get("evidence_text") or ""),
            evidence_chunk_ids=list(p.get("evidence_chunk_ids") or []),
            section=str(p.get("section") or ""),
            source_url=str(p.get("source_url") or ""),
            metadata=dict(p),
        )
