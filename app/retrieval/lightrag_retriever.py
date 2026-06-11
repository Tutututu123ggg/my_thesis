from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.query_keywords import QueryKeywordExtractor
from app.retrieval.retrieval_models import (
    QueryKeywords,
    RetrievalResult,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelation,
)
from app.retrieval.vector_retriever import VectorRetriever


@dataclass
class LightRAGRetrievalConfig:
    entity_top_k: int = 20
    relation_top_k: int = 20
    max_entities: int = 30
    max_relations: int = 40
    max_chunks: int = 12
    graph_max_relations: int = 80
    graph_max_chunks: int = 50
    synonym_expand: bool = True


class LightRAGRetriever:
    """
    LightRAG-style dual-level retriever.

    Mapping theo paper:
    - QueryKeywordExtractor sinh low/high-level keywords.
    - low-level keywords -> Qdrant entity vectors.
    - high-level keywords -> Qdrant relation vectors.
    - entity/relation seeds -> Neo4j graph expansion 1-hop + evidence chunks.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        keyword_extractor: QueryKeywordExtractor | None = None,
        config: LightRAGRetrievalConfig | None = None,
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.keyword_extractor = keyword_extractor or QueryKeywordExtractor()
        self.config = config or LightRAGRetrievalConfig()

    def retrieve(self, query: str) -> RetrievalResult:
        keywords = self.keyword_extractor.extract(query)
        low_query = keywords.low_query_text() or query
        high_query = keywords.high_query_text() or query

        entity_hits = self.vector_retriever.search_entities(
            low_query,
            top_k=self.config.entity_top_k,
        )
        relation_hits = self.vector_retriever.search_relations(
            high_query,
            top_k=self.config.relation_top_k,
        )

        entity_seed_ids = [e.entity_id for e in entity_hits]
        relation_seed_ids = [r.relation_id for r in relation_hits]

        graph_from_entities = self.graph_retriever.retrieve_from_entity_ids(
            query=query,
            entity_ids=entity_seed_ids,
            max_relations=self.config.graph_max_relations,
            max_chunks=self.config.graph_max_chunks,
            expand_synonyms=self.config.synonym_expand,
        )
        graph_from_relations = self.graph_retriever.retrieve_from_relation_ids(
            query=query,
            relation_ids=relation_seed_ids,
            max_chunks=self.config.graph_max_chunks,
            expand_endpoint_entities=True,
            max_entity_relations=self.config.graph_max_relations,
        )

        entities = self._merge_entities(entity_hits + graph_from_entities.entities + graph_from_relations.entities)
        relations = self._merge_relations(relation_hits + graph_from_entities.relations + graph_from_relations.relations)
        chunks = self._merge_chunks(graph_from_entities.chunks + graph_from_relations.chunks)

        return RetrievalResult(
            query=query,
            method="lightrag",
            keywords=keywords,
            entities=entities[: self.config.max_entities],
            relations=relations[: self.config.max_relations],
            chunks=chunks[: self.config.max_chunks],
            debug={
                "low_query": low_query,
                "high_query": high_query,
                "entity_seed_count": len(entity_hits),
                "relation_seed_count": len(relation_hits),
                "synonym_expand": self.config.synonym_expand,
            },
        )

    def _merge_entities(self, entities: list[RetrievedEntity]) -> list[RetrievedEntity]:
        by_id: dict[str, RetrievedEntity] = {}
        for entity in entities:
            old = by_id.get(entity.entity_id)
            if old is None or entity.score > old.score:
                by_id[entity.entity_id] = entity
        return sorted(by_id.values(), key=lambda x: x.score, reverse=True)

    def _merge_relations(self, relations: list[RetrievedRelation]) -> list[RetrievedRelation]:
        by_id: dict[str, RetrievedRelation] = {}
        for relation in relations:
            old = by_id.get(relation.relation_id)
            if old is None or relation.score > old.score:
                by_id[relation.relation_id] = relation
        return sorted(by_id.values(), key=lambda x: x.score, reverse=True)

    def _merge_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        by_id: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            old = by_id.get(chunk.chunk_id)
            if old is None or chunk.score > old.score:
                by_id[chunk.chunk_id] = chunk
        return sorted(by_id.values(), key=lambda x: x.score, reverse=True)
