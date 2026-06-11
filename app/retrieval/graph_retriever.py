from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from app.infrastructure.graph_database import GraphRepository
from app.retrieval.retrieval_models import (
    RetrievalResult,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelation,
)


class GraphRetriever:
    """
    Neo4j graph expansion layer.

    Không làm vector search. Nhận seed entity/relation IDs từ Qdrant/LightRAG rồi:
    - lấy entity nodes thật từ Neo4j,
    - lấy relations quanh seeds,
    - lấy evidence chunks,
    - optional expand DONG_NGHIA_VOI nếu GraphRepository đã có method này.
    """

    def __init__(self, graph_repo: GraphRepository):
        self.graph_repo = graph_repo

    def retrieve_from_entity_ids(
        self,
        query: str,
        entity_ids: list[str],
        relation_types: list[str] | None = None,
        max_relations: int = 80,
        max_chunks: int = 50,
        expand_synonyms: bool = True,
    ) -> RetrievalResult:
        seed_ids = self._dedupe(entity_ids)
        all_entity_ids = list(seed_ids)

        if expand_synonyms:
            all_entity_ids.extend(self._get_synonym_entity_ids(seed_ids))
            all_entity_ids = self._dedupe(all_entity_ids)

        bundle = self.graph_repo.get_entity_context_bundle(
            entity_ids=all_entity_ids,
            relation_types=relation_types,
            max_relations=max_relations,
            max_chunks=max_chunks,
            expand_synonyms=False,
        )

        return RetrievalResult(
            query=query,
            method="graph_local",
            entities=[self._entity_from_node(e, score=1.0) for e in bundle.entities],
            relations=[self._relation_from_view(r, score=getattr(r, "score", 1.0)) for r in bundle.relations],
            chunks=[self._chunk_from_node(c, score=getattr(c, "score", 1.0), source="graph_expansion") for c in bundle.chunks],
            debug={
                "seed_entity_ids": seed_ids,
                "expanded_entity_ids": all_entity_ids,
                "expand_synonyms": expand_synonyms,
            },
        )

    def retrieve_from_relation_ids(
        self,
        query: str,
        relation_ids: list[str],
        max_chunks: int = 50,
        expand_endpoint_entities: bool = True,
        max_entity_relations: int = 60,
    ) -> RetrievalResult:
        relation_ids = self._dedupe(relation_ids)
        relations = self.graph_repo.get_relations_by_ids(relation_ids)
        chunks = self.graph_repo.get_chunks_by_relation_ids(relation_ids, limit=max_chunks)

        entities = []
        extra_relations = []
        if expand_endpoint_entities:
            endpoint_ids = []
            for r in relations:
                endpoint_ids.extend([r.subject_entity_id, r.object_entity_id])
            endpoint_ids = self._dedupe(endpoint_ids)
            entities = self.graph_repo.get_entities_by_ids(endpoint_ids)
            extra_relations = self.graph_repo.get_relations_by_entity_ids(
                entity_ids=endpoint_ids,
                limit=max_entity_relations,
            )

        return RetrievalResult(
            query=query,
            method="graph_relation",
            entities=[self._entity_from_node(e, score=1.0) for e in entities],
            relations=self._merge_relations(
                [self._relation_from_view(r, score=getattr(r, "score", 1.0)) for r in relations]
                + [self._relation_from_view(r, score=getattr(r, "score", 0.8)) for r in extra_relations]
            ),
            chunks=[self._chunk_from_node(c, score=getattr(c, "score", 1.0), source="graph_expansion") for c in chunks],
            debug={"seed_relation_ids": relation_ids},
        )

    def _get_synonym_entity_ids(self, entity_ids: list[str]) -> list[str]:
        if not hasattr(self.graph_repo, "get_synonym_neighbors") or not entity_ids:
            return []

        try:
            neighbors = self.graph_repo.get_synonym_neighbors(entity_ids=entity_ids, limit=30)
        except TypeError:
            neighbors = self.graph_repo.get_synonym_neighbors(entity_ids, 30)
        except Exception:
            neighbors = []

        return [n.entity_id for n in neighbors]

    def _entity_from_node(self, entity, score: float) -> RetrievedEntity:
        return RetrievedEntity(
            entity_id=entity.entity_id,
            score=score,
            source="graph_expansion",
            name=entity.name,
            normalized_name=entity.normalized_name,
            entity_type=entity.entity_type,
            description=entity.description,
            profile_text=entity.profile_text,
            aliases=list(entity.aliases or []),
            surface_forms=list(entity.surface_forms or []),
            metadata=dict(entity.metadata or {}),
        )

    def _relation_from_view(self, relation, score: float) -> RetrievedRelation:
        return RetrievedRelation(
            relation_id=relation.relation_id,
            score=float(score or 0.0),
            source="graph_expansion",
            relation_type=relation.relation_type,
            subject_entity_id=relation.subject_entity_id,
            subject_name=relation.subject_name,
            subject_type=relation.subject_type,
            object_entity_id=relation.object_entity_id,
            object_name=relation.object_name,
            object_type=relation.object_type,
            description=relation.description,
            keywords=list(relation.keywords or []),
            evidence_text=relation.evidence_text,
            evidence_chunk_ids=list(relation.evidence_chunk_ids or []),
            section=relation.section,
            source_url=relation.source_url,
            metadata=dict(relation.metadata or {}),
        )

    def _chunk_from_node(self, chunk, score: float, source: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            score=float(score or 0.0),
            source=source,  # type: ignore[arg-type]
            article_id=chunk.article_id,
            title=chunk.title,
            section=chunk.section or "",
            subsection=chunk.subsection or "",
            source_url=chunk.source_url,
            text=chunk.text,
            contextualized_text=chunk.contextualized_text or chunk.text,
            metadata=dict(chunk.metadata or {}),
        )

    def _merge_relations(self, relations: Iterable[RetrievedRelation]) -> list[RetrievedRelation]:
        by_id: dict[str, RetrievedRelation] = {}
        for relation in relations:
            old = by_id.get(relation.relation_id)
            if old is None or relation.score > old.score:
                by_id[relation.relation_id] = relation
        return sorted(by_id.values(), key=lambda x: x.score, reverse=True)

    def _dedupe(self, items: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result