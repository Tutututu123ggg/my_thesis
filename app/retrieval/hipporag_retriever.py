from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from app.infrastructure.graph_database import GraphRepository
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
class HippoRAGRetrievalConfig:
    """
    HippoRAG-style associative retrieval.

    This implementation does not require Neo4j GDS. It exports a small entity
    graph from Neo4j and runs standard iterative Personalized PageRank in Python.
    """

    seed_top_k: int = 12
    vector_seed_top_k: int = 12
    ppr_top_k_entities: int = 40
    max_relations: int = 80
    max_chunks: int = 12
    adjacency_hops: int = 2
    adjacency_limit: int = 100_000
    chunk_link_limit: int = 100_000
    damping: float = 0.85
    tolerance: float = 1e-7
    max_iterations: int = 100
    use_vector_entity_seeds: bool = True
    treat_edges_as_undirected: bool = True

    # HippoRAG Section 2.3: node specificity downweights generic query nodes
    # that are mentioned by many passages/chunks before running PPR.
    use_node_specificity: bool = True
    node_specificity_min: float = 1e-6


class HippoRAGRetriever:
    """
    HippoRAG-style retriever.

    Pipeline:
    1. Get query/entity seeds from query keywords, Neo4j entity lookup and
       optionally the entity vector index.
    2. Export the entity graph around seeds.
    3. Run Personalized PageRank seeded by query/entity matches.
    4. Aggregate entity PPR scores back to chunks via MENTIONS and relation
       evidence_chunk_ids.
    5. Return top chunks plus supporting entities/relations.
    """

    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_retriever: VectorRetriever | None = None,
        keyword_extractor: QueryKeywordExtractor | None = None,
        config: HippoRAGRetrievalConfig | None = None,
    ):
        self.graph_repo = graph_repo
        self.vector_retriever = vector_retriever
        self.keyword_extractor = keyword_extractor or QueryKeywordExtractor()
        self.config = config or HippoRAGRetrievalConfig()

    def retrieve(self, query: str) -> RetrievalResult:
        keywords = self.keyword_extractor.extract(query)
        seed_query = keywords.low_query_text() or query
        seed_scores = self._collect_seed_scores(query=query, seed_query=seed_query)

        if not seed_scores:
            return RetrievalResult(
                query=query,
                method="hipporag",
                keywords=keywords,
                debug={
                    "seed_query": seed_query,
                    "seed_count": 0,
                    "note": "No entity seeds found; PPR was not run.",
                },
            )

        raw_seed_scores = dict(seed_scores)
        seed_scores, node_specificity = self._apply_node_specificity(seed_scores)

        seed_entity_ids = list(seed_scores.keys())
        adjacency = self.graph_repo.get_entity_adjacency_around_seeds(
            seed_entity_ids=seed_entity_ids,
            max_hops=self.config.adjacency_hops,
            limit=self.config.adjacency_limit,
        )
        if not adjacency:
            adjacency = self.graph_repo.get_entity_adjacency(limit=self.config.adjacency_limit)

        ppr_scores = self._personalized_pagerank(
            seed_scores=seed_scores,
            adjacency=adjacency,
        )
        ranked_entity_ids = [
            entity_id
            for entity_id, score in sorted(
                ppr_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if score > 0.0
        ][: self.config.ppr_top_k_entities]

        entities = self._load_entities(ranked_entity_ids, ppr_scores)
        relations = self._load_supporting_relations(ranked_entity_ids, ppr_scores)
        chunks = self._load_ranked_chunks(ranked_entity_ids, ppr_scores, relations)

        return RetrievalResult(
            query=query,
            method="hipporag",
            keywords=keywords,
            chunks=chunks[: self.config.max_chunks],
            entities=entities,
            relations=relations[: self.config.max_relations],
            debug={
                "seed_query": seed_query,
                "seed_entity_ids": seed_entity_ids,
                "seed_count": len(seed_entity_ids),
                "raw_seed_scores": raw_seed_scores,
                "node_specificity": node_specificity,
                "node_specificity_enabled": self.config.use_node_specificity,
                "adjacency_edge_count": len(adjacency),
                "ppr_entity_count": len(ppr_scores),
                "damping": self.config.damping,
                "adjacency_hops": self.config.adjacency_hops,
                "treat_edges_as_undirected": self.config.treat_edges_as_undirected,
                "note": "PPR is computed in Python because Neo4j GDS is not required.",
            },
        )

    # =====================
    # Seeds
    # =====================

    def _collect_seed_scores(self, query: str, seed_query: str) -> dict[str, float]:
        seed_scores: dict[str, float] = {}

        for entity in self.graph_repo.find_seed_entities(
            query_text=seed_query,
            limit=self.config.seed_top_k,
        ):
            seed_scores[entity.entity_id] = max(seed_scores.get(entity.entity_id, 0.0), float(entity.score or 1.0))

        if (
            self.config.use_vector_entity_seeds
            and self.vector_retriever is not None
            and self.config.vector_seed_top_k > 0
        ):
            for entity in self.vector_retriever.search_entities(
                seed_query or query,
                top_k=self.config.vector_seed_top_k,
            ):
                seed_scores[entity.entity_id] = max(seed_scores.get(entity.entity_id, 0.0), float(entity.score or 0.0))

        return {entity_id: score for entity_id, score in seed_scores.items() if entity_id and score > 0.0}


    def _apply_node_specificity(
        self,
        seed_scores: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Apply HippoRAG node specificity to query node probabilities.

        HippoRAG uses node specificity as an IDF-like signal based on how many
        passages contain a node. In this project, passages are chunks and the
        Chunk-[:MENTIONS]->Entity edges give the node document frequency.
        Specificity is multiplied into query seed probabilities before PPR.
        """
        if not self.config.use_node_specificity or not seed_scores:
            return seed_scores, {entity_id: 1.0 for entity_id in seed_scores}

        chunk_counts = self.graph_repo.get_entity_chunk_counts(list(seed_scores.keys()))
        specificity: dict[str, float] = {}
        adjusted: dict[str, float] = {}

        for entity_id, score in seed_scores.items():
            # Paper-faithful IDF-like specificity: 1 / |P_i|, where P_i is the
            # set of passages/chunks mentioning node i. Missing counts fall back
            # to 1 so unseen-but-linked seeds are not dropped.
            count = max(int(chunk_counts.get(entity_id, 1) or 1), 1)
            value = max(1.0 / float(count), self.config.node_specificity_min)
            specificity[entity_id] = value
            adjusted[entity_id] = float(score or 0.0) * value

        normalized = self._normalize(adjusted)
        if not normalized:
            return seed_scores, specificity
        return normalized, specificity

    # =====================
    # PPR
    # =====================

    def _personalized_pagerank(self, seed_scores: dict[str, float], adjacency) -> dict[str, float]:
        nodes: set[str] = set(seed_scores)
        outgoing: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for edge in adjacency:
            if not edge.source_entity_id or not edge.target_entity_id:
                continue
            weight = max(float(edge.weight or edge.confidence or 1.0), 0.0)
            if weight <= 0.0:
                continue

            nodes.add(edge.source_entity_id)
            nodes.add(edge.target_entity_id)
            outgoing[edge.source_entity_id][edge.target_entity_id] += weight

            if self.config.treat_edges_as_undirected:
                outgoing[edge.target_entity_id][edge.source_entity_id] += weight

        if not nodes:
            return {}

        personalization = self._normalize({node: seed_scores.get(node, 0.0) for node in nodes})
        if not personalization:
            return {}

        damping = min(max(float(self.config.damping), 0.0), 1.0)
        rank = dict(personalization)

        for _ in range(max(1, self.config.max_iterations)):
            next_rank = {node: (1.0 - damping) * personalization.get(node, 0.0) for node in nodes}
            sink_mass = 0.0

            for source in nodes:
                neighbors = outgoing.get(source) or {}
                if not neighbors:
                    sink_mass += rank.get(source, 0.0)
                    continue

                total_weight = sum(neighbors.values())
                if total_weight <= 0.0:
                    sink_mass += rank.get(source, 0.0)
                    continue

                source_rank = rank.get(source, 0.0)
                for target, weight in neighbors.items():
                    next_rank[target] = next_rank.get(target, 0.0) + damping * source_rank * (weight / total_weight)

            if sink_mass > 0.0:
                for node, p in personalization.items():
                    next_rank[node] = next_rank.get(node, 0.0) + damping * sink_mass * p

            delta = sum(abs(next_rank.get(node, 0.0) - rank.get(node, 0.0)) for node in nodes)
            rank = next_rank
            if delta <= self.config.tolerance:
                break

        return rank

    def _normalize(self, scores: dict[str, float]) -> dict[str, float]:
        positive = {key: max(float(value), 0.0) for key, value in scores.items() if key}
        total = sum(positive.values())
        if total <= 0.0:
            return {}
        return {key: value / total for key, value in positive.items() if value > 0.0}

    # =====================
    # Result assembly
    # =====================

    def _load_entities(self, entity_ids: list[str], ppr_scores: dict[str, float]) -> list[RetrievedEntity]:
        nodes = self.graph_repo.get_entities_by_ids(entity_ids)
        result = [self._entity_from_node(node, ppr_scores.get(node.entity_id, 0.0)) for node in nodes]
        return sorted(result, key=lambda item: item.score, reverse=True)

    def _load_supporting_relations(
        self,
        entity_ids: list[str],
        ppr_scores: dict[str, float],
    ) -> list[RetrievedRelation]:
        relation_views = self.graph_repo.get_relations_by_entity_ids(
            entity_ids=entity_ids,
            limit=self.config.max_relations,
        )
        relations = []
        for relation in relation_views:
            score = max(
                ppr_scores.get(relation.subject_entity_id, 0.0),
                ppr_scores.get(relation.object_entity_id, 0.0),
            ) * float(relation.confidence or 1.0)
            relations.append(self._relation_from_view(relation, score=score))
        return sorted(relations, key=lambda item: item.score, reverse=True)

    def _load_ranked_chunks(
        self,
        entity_ids: list[str],
        ppr_scores: dict[str, float],
        relations: list[RetrievedRelation],
    ) -> list[RetrievedChunk]:
        chunk_scores: dict[str, float] = defaultdict(float)
        chunk_support: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"entities": [], "relations": []})

        links = self.graph_repo.get_chunk_entity_links(
            entity_ids=entity_ids,
            limit=self.config.chunk_link_limit,
        )
        for link in links:
            entity_score = ppr_scores.get(link.entity_id, 0.0)
            if entity_score <= 0.0:
                continue
            chunk_scores[link.chunk_id] += entity_score * float(link.confidence or 1.0)
            chunk_support[link.chunk_id]["entities"].append(link.entity_id)

        for relation in relations:
            if relation.score <= 0.0:
                continue
            for chunk_id in relation.evidence_chunk_ids:
                if not chunk_id:
                    continue
                chunk_scores[chunk_id] += relation.score
                chunk_support[chunk_id]["relations"].append(relation.relation_id)

        chunk_ids = [
            chunk_id
            for chunk_id, _ in sorted(
                chunk_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ][: max(self.config.max_chunks * 3, self.config.max_chunks)]
        chunks = self.graph_repo.get_chunks_by_ids(chunk_ids)

        result = []
        for chunk in chunks:
            retrieved = self._chunk_from_node(chunk, score=chunk_scores.get(chunk.chunk_id, 0.0))
            retrieved.metadata = {
                **retrieved.metadata,
                "supporting_entity_ids": self._dedupe(chunk_support[chunk.chunk_id]["entities"]),
                "supporting_relation_ids": self._dedupe(chunk_support[chunk.chunk_id]["relations"]),
            }
            result.append(retrieved)

        return sorted(result, key=lambda item: item.score, reverse=True)

    def _entity_from_node(self, entity, score: float) -> RetrievedEntity:
        return RetrievedEntity(
            entity_id=entity.entity_id,
            score=float(score or 0.0),
            source="hipporag",
            name=entity.name,
            normalized_name=entity.normalized_name,
            entity_type=entity.entity_type,
            description=entity.description or "",
            profile_text=entity.profile_text or entity.description or "",
            aliases=list(entity.aliases or []),
            surface_forms=list(entity.surface_forms or []),
            metadata=dict(entity.metadata or {}),
        )

    def _relation_from_view(self, relation, score: float) -> RetrievedRelation:
        return RetrievedRelation(
            relation_id=relation.relation_id,
            score=float(score or 0.0),
            source="hipporag",
            relation_type=relation.relation_type,
            subject_entity_id=relation.subject_entity_id,
            subject_name=relation.subject_name,
            subject_type=relation.subject_type,
            object_entity_id=relation.object_entity_id,
            object_name=relation.object_name,
            object_type=relation.object_type,
            description=relation.description or "",
            keywords=list(relation.keywords or []),
            evidence_text=relation.evidence_text or "",
            evidence_chunk_ids=list(relation.evidence_chunk_ids or []),
            section=relation.section or "",
            source_url=relation.source_url or "",
            metadata=dict(relation.metadata or {}),
        )

    def _chunk_from_node(self, chunk, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            score=float(score or 0.0),
            source="hipporag",
            article_id=chunk.article_id,
            title=chunk.title,
            section=chunk.section or "",
            subsection=chunk.subsection or "",
            source_url=chunk.source_url,
            text=chunk.text,
            contextualized_text=chunk.contextualized_text or chunk.text,
            metadata=dict(chunk.metadata or {}),
        )

    def _dedupe(self, items: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result
