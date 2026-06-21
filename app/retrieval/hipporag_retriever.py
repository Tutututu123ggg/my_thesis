from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from app.infrastructure.graph_database import GraphRepository
from app.retrieval.evidence_expander import EvidenceExpansionConfig, GraphEvidenceExpander
from app.retrieval.graph_scoring import GraphQueryScorer
from app.retrieval.query_analyzer import QueryAnalyzer
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
    vector_seed_phrase_queries: int = 6
    ppr_top_k_entities: int = 60
    max_relations: int = 120
    max_chunks: int = 12
    adjacency_hops: int = 3
    adjacency_limit: int = 100_000
    chunk_link_limit: int = 100_000
    damping: float = 0.85
    tolerance: float = 1e-7
    max_iterations: int = 100
    use_vector_entity_seeds: bool = True
    treat_edges_as_undirected: bool = True

    # Seed lookup should use individual low-level entity terms, not one long
    # semicolon-joined query. Otherwise lexical seed search misses entities like
    # "viêm da tiết bã" inside a comparative question.
    max_seed_terms: int = 12
    lexical_seed_weight: float = 1.0
    vector_seed_weight: float = 0.8

    # Final chunk ranking remains PPR-based, but these light query/evidence
    # bonuses help answer-bearing chunks beat generic symptom chunks.
    enable_chunk_query_rerank: bool = True
    query_term_chunk_weight: float = 0.08
    slot_term_chunk_weight: float = 0.12
    relation_evidence_chunk_weight: float = 1.0
    anchor_term_chunk_weight: float = 0.18
    title_anchor_chunk_weight: float = 0.42
    generic_term_chunk_weight: float = 0.018
    slot_anchor_gate_floor: float = 0.0
    relation_anchor_gate_floor: float = 0.0
    relation_evidence_max_scale: float = 1.6
    score_scale_floor: float = 0.01

    # HippoRAG Section 2.3: node specificity downweights generic query nodes
    # that are mentioned by many passages/chunks before running PPR.
    use_node_specificity: bool = True
    node_specificity_min: float = 1e-6

    # Query-aware edge weighting is a light medical adaptation: PPR remains
    # standard, but relation types that match the query intent receive higher
    # transition weights before row-normalization.
    use_query_aware_edge_weights: bool = True
    use_transition_node_specificity: bool = True
    transition_specificity_power: float = 0.6
    final_specificity_power: float = 0.35
    seed_anchor_preservation: float = 1.25

    # Passage integration inspired by HippoRAG 2: graph PPR still drives the
    # branch, but a small chunk-level passage signal is allowed to rescue
    # source-local factual evidence. This is not a replacement for vector_naive.
    use_passage_integration: bool = True
    passage_vector_top_k: int = 5
    passage_vector_weight: float = 0.20

    # Same-article expansion is applied after PPR/chunk aggregation, not before.
    enable_evidence_expansion: bool = True
    evidence_expansion: EvidenceExpansionConfig | None = None

    # Final presentation guard for standalone HippoRAG. PPR can produce many
    # chunks from one drifted article; keep article diversity in the returned
    # top-k without changing the graph walk itself.
    enable_final_source_diversity: bool = True
    final_diversity_window: int = 5
    final_max_chunks_per_source: int = 2
    final_max_graph_expansion_per_source: int = 1
    final_preserve_passage_chunks: int = 2
    final_anchor_relax_threshold: float = 1.4


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
        query_analyzer: QueryAnalyzer | None = None,
        config: HippoRAGRetrievalConfig | None = None,
    ):
        self.graph_repo = graph_repo
        self.vector_retriever = vector_retriever
        self.keyword_extractor = keyword_extractor or QueryKeywordExtractor()
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.config = config or HippoRAGRetrievalConfig()
        self.graph_scorer = GraphQueryScorer()
        self.evidence_expander = GraphEvidenceExpander(
            graph_repo=self.graph_repo,
            config=self.config.evidence_expansion or EvidenceExpansionConfig(
                max_total_chunks=self.config.max_chunks,
            ),
        )

    def retrieve(self, query: str) -> RetrievalResult:
        keywords = self.keyword_extractor.extract(query)
        analysis = self.query_analyzer.analyze(query, keywords)
        seed_query = self._build_seed_query(query, keywords)
        seed_scores = self._collect_seed_scores(query=query, seed_query=seed_query, keywords=keywords)

        if not seed_scores:
            return RetrievalResult(
                query=query,
                method="hipporag",
                keywords=keywords,
                debug={
                    "seed_query": seed_query,
                    "intent": analysis.intent,
                    "intent_confidence": analysis.confidence,
                    "intent_cues": analysis.cues,
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

        ppr_scores, transition_specificity_debug = self._personalized_pagerank(
            seed_scores=seed_scores,
            adjacency=adjacency,
            relation_type_boosts=analysis.relation_type_boosts if self.config.use_query_aware_edge_weights else None,
        )
        chunk_stages: dict[str, list[dict]] = {}

        def snapshot_chunks(stage: str, values: list[RetrievedChunk]) -> None:
            chunk_stages[stage] = [self._debug_chunk_snapshot(chunk, rank=i) for i, chunk in enumerate(values[: self.config.max_chunks * 3], start=1)]
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
        snapshot_chunks("after_ppr_chunk_mapping", chunks)

        if self.config.use_passage_integration:
            chunks = self._merge_passage_integrated_chunks(query=query, graph_chunks=chunks)
            snapshot_chunks("after_passage_integration", chunks)

        if self.config.enable_chunk_query_rerank:
            chunks = self._rerank_chunks_by_query_evidence(
                query=query,
                keywords=keywords,
                analysis=analysis,
                chunks=chunks,
                relations=relations,
            )
            snapshot_chunks("after_query_evidence_rerank", chunks)

        expansion_debug = {"enabled": False}
        if self.config.enable_evidence_expansion:
            chunks, expansion_debug = self.evidence_expander.expand_chunks(
                query=query,
                chunks=chunks,
                entities=entities,
                relations=relations,
                analysis=analysis,
            )
            snapshot_chunks("after_evidence_expansion", chunks)

        if self.config.enable_final_source_diversity:
            chunks = self._apply_final_source_diversity(query=query, keywords=keywords, chunks=chunks)
            snapshot_chunks("after_final_source_diversity", chunks)

        return RetrievalResult(
            query=query,
            method="hipporag",
            keywords=keywords,
            chunks=chunks[: self.config.max_chunks],
            entities=entities,
            relations=relations[: self.config.max_relations],
            debug={
                "seed_query": seed_query,
                "intent": analysis.intent,
                "intent_confidence": analysis.confidence,
                "intent_cues": analysis.cues,
                "intent_relation_boosts": analysis.relation_type_boosts,
                "evidence_expansion": expansion_debug,
                "seed_entity_ids": seed_entity_ids,
                "seed_count": len(seed_entity_ids),
                "raw_seed_scores": raw_seed_scores,
                "node_specificity": node_specificity,
                "transition_node_specificity_top": dict(
                    sorted(transition_specificity_debug.items(), key=lambda item: item[1], reverse=True)[:40]
                ),
                "node_specificity_enabled": self.config.use_node_specificity,
                "adjacency_edge_count": len(adjacency),
                "ppr_entity_count": len(ppr_scores),
                "damping": self.config.damping,
                "adjacency_hops": self.config.adjacency_hops,
                "treat_edges_as_undirected": self.config.treat_edges_as_undirected,
                "query_aware_edge_weights": self.config.use_query_aware_edge_weights,
                "transition_specificity_enabled": self.config.use_transition_node_specificity,
                "passage_integration_enabled": self.config.use_passage_integration,
                "graph_confidence": self.graph_scorer.branch_confidence(chunks, query, keywords),
                "chunk_stages": chunk_stages,
                "note": "PPR is computed in Python because Neo4j GDS is not required.",
            },
        )

    # =====================
    # Seeds
    # =====================

    def _build_seed_query(self, query: str, keywords: QueryKeywords) -> str:
        return "; ".join(
            self._dedupe(
                [
                    *keywords.low_level_keywords,
                    query,
                ]
            )
        ) or query

    def _collect_seed_scores(
        self,
        query: str,
        seed_query: str,
        keywords: QueryKeywords,
    ) -> dict[str, float]:
        seed_scores: dict[str, float] = {}
        seed_debug_terms = self._seed_terms(query=query, seed_query=seed_query, keywords=keywords)

        # Lexical/entity lookup is run per term. A single combined query such as
        # "ngứa; bong tróc; viêm da tiết bã; viêm da cơ địa" will not be contained
        # in any entity name, so the old code often missed explicit disease seeds.
        for idx, term in enumerate(seed_debug_terms):
            term_weight = self.config.lexical_seed_weight / (1.0 + idx * 0.05)
            for entity in self.graph_repo.find_seed_entities(
                query_text=term,
                limit=max(3, self.config.seed_top_k),
            ):
                score = float(entity.score or 1.0) * term_weight
                anchor_multiplier, _ = self.graph_scorer.entity_anchor_multiplier(
                    self._entity_from_node(entity, score=0.0),
                    query,
                    keywords,
                )
                score *= anchor_multiplier
                seed_scores[entity.entity_id] = max(seed_scores.get(entity.entity_id, 0.0), score)

        if (
            self.config.use_vector_entity_seeds
            and self.vector_retriever is not None
            and self.config.vector_seed_top_k > 0
        ):
            profile = self.graph_scorer.build_query_anchor_profile(query, keywords, None)
            vector_queries = self._dedupe_norm([
                seed_query,
                query,
                *self.graph_scorer.select_search_phrases(
                    profile,
                    limit=max(0, int(self.config.vector_seed_phrase_queries)),
                ),
            ])
            for vector_query in vector_queries:
                for entity in self.vector_retriever.search_entities(
                    vector_query,
                    top_k=self.config.vector_seed_top_k,
                ):
                    score = float(entity.score or 0.0) * self.config.vector_seed_weight
                    anchor_multiplier, _ = self.graph_scorer.entity_anchor_multiplier(entity, query, keywords)
                    score *= anchor_multiplier
                    seed_scores[entity.entity_id] = max(seed_scores.get(entity.entity_id, 0.0), score)

        ranked = sorted(
            ((entity_id, score) for entity_id, score in seed_scores.items() if entity_id and score > 0.0),
            key=lambda item: item[1],
            reverse=True,
        )
        max_seed_count = max(self.config.seed_top_k * 3, self.config.seed_top_k)
        return dict(ranked[:max_seed_count])

    def _seed_terms(self, query: str, seed_query: str, keywords: QueryKeywords) -> list[str]:
        profile = self.graph_scorer.build_query_anchor_profile(query, keywords, None)
        values: list[str] = []
        values.extend(keywords.low_level_keywords)
        values.extend(
            self.graph_scorer.select_search_phrases(
                profile,
                limit=max(self.config.max_seed_terms * 3, self.config.max_seed_terms),
            )
        )

        # Keep raw query variants as fallbacks. There is no phrase/cue splitting
        # here; candidate phrases are mined mechanically by GraphQueryScorer.
        values.append(seed_query)
        values.append(query)
        return self._dedupe_norm([value for value in values if value])[: self.config.max_seed_terms]


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
        total_chunks = max(int(self.graph_repo.get_graph_stats().get("chunk_count", 0) or 0), 1)
        specificity: dict[str, float] = {}
        adjusted: dict[str, float] = {}

        for entity_id, score in seed_scores.items():
            # Smooth IDF-like specificity. Exact disease anchors are already
            # protected by anchor multipliers in seed collection; this prevents
            # broad nodes from dominating without deleting valid disease nodes.
            count = max(int(chunk_counts.get(entity_id, 1) or 1), 1)
            value = max(self.graph_scorer.node_specificity(count, total_chunks), self.config.node_specificity_min)
            specificity[entity_id] = value
            adjusted[entity_id] = float(score or 0.0) * value

        normalized = self._normalize(adjusted)
        if not normalized:
            return seed_scores, specificity
        return normalized, specificity

    # =====================
    # PPR
    # =====================

    def _personalized_pagerank(
        self,
        seed_scores: dict[str, float],
        adjacency,
        relation_type_boosts: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        nodes: set[str] = set(seed_scores)
        raw_edges: list[tuple[str, str, str, float]] = []

        for edge in adjacency:
            if not edge.source_entity_id or not edge.target_entity_id:
                continue
            weight = max(float(edge.weight or edge.confidence or 1.0), 0.0)
            if weight <= 0.0:
                continue
            nodes.add(edge.source_entity_id)
            nodes.add(edge.target_entity_id)
            raw_edges.append((edge.source_entity_id, edge.target_entity_id, edge.relation_type, weight))
            if self.config.treat_edges_as_undirected:
                raw_edges.append((edge.target_entity_id, edge.source_entity_id, edge.relation_type, weight))

        if not nodes:
            return {}, {}

        chunk_counts = self.graph_repo.get_entity_chunk_counts(list(nodes))
        total_chunks = max(int(self.graph_repo.get_graph_stats().get("chunk_count", 0) or 0), 1)
        node_specificity = {
            node: self.graph_scorer.node_specificity(chunk_counts.get(node, 1), total_chunks)
            for node in nodes
        }

        outgoing: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for source, target, relation_type, base_weight in raw_edges:
            weight = base_weight
            if relation_type_boosts:
                weight *= float(relation_type_boosts.get(relation_type, 0.70))
            if self.config.use_transition_node_specificity:
                target_spec = max(node_specificity.get(target, 1.0), self.config.node_specificity_min)
                weight *= target_spec ** max(float(self.config.transition_specificity_power), 0.0)
            if target in seed_scores:
                weight *= float(self.config.seed_anchor_preservation)
            if weight > 0.0:
                outgoing[source][target] += weight

        personalization = self._normalize({node: seed_scores.get(node, 0.0) for node in nodes})
        if not personalization:
            return {}, node_specificity

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

        if self.config.use_node_specificity and self.config.final_specificity_power > 0.0:
            final_rank: dict[str, float] = {}
            for node, score in rank.items():
                spec = max(node_specificity.get(node, 1.0), self.config.node_specificity_min)
                seed_bonus = self.config.seed_anchor_preservation if node in seed_scores else 1.0
                final_rank[node] = float(score or 0.0) * (spec ** self.config.final_specificity_power) * seed_bonus
            rank = self._normalize(final_rank) or final_rank

        return rank, node_specificity

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

    def _merge_passage_integrated_chunks(
        self,
        *,
        query: str,
        graph_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if self.vector_retriever is None or self.config.passage_vector_top_k <= 0:
            return graph_chunks

        vector_result = self.vector_retriever.retrieve(query, top_k=self.config.passage_vector_top_k)
        by_id: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in graph_chunks}

        for rank, vector_chunk in enumerate(vector_result.chunks, start=1):
            passage_score = float(vector_chunk.score or 0.0) * self.config.passage_vector_weight / (1.0 + 0.05 * rank)
            old = by_id.get(vector_chunk.chunk_id)
            if old is None:
                by_id[vector_chunk.chunk_id] = RetrievedChunk(
                    chunk_id=vector_chunk.chunk_id,
                    score=passage_score,
                    source="hipporag_passage",
                    article_id=vector_chunk.article_id,
                    title=vector_chunk.title,
                    section=vector_chunk.section,
                    subsection=vector_chunk.subsection,
                    source_url=vector_chunk.source_url,
                    text=vector_chunk.text,
                    contextualized_text=vector_chunk.contextualized_text,
                    metadata={
                        **dict(vector_chunk.metadata or {}),
                        "hipporag_passage_integration": True,
                        "vector_original_score": vector_chunk.score,
                    },
                )
            else:
                old.score = float(old.score or 0.0) + passage_score
                old.metadata = {
                    **dict(old.metadata or {}),
                    "hipporag_passage_integration": True,
                    "passage_added_score": passage_score,
                }

        return sorted(by_id.values(), key=lambda item: item.score, reverse=True)

    def _rerank_chunks_by_query_evidence(
        self,
        query: str,
        keywords: QueryKeywords,
        analysis,
        chunks: list[RetrievedChunk],
        relations: list[RetrievedRelation],
    ) -> list[RetrievedChunk]:
        """Anchor-gated chunk rerank after PPR/passage integration.

        This stage must not let generic evidence slots ("triệu chứng", "nguyên
        nhân", "biến chứng", ...) or relation evidence chunks override the PPR
        ranking unless the chunk also has query-anchor support. Generic lexical
        overlap is kept as a weak signal only.
        """
        if not chunks:
            return chunks

        profile = self.graph_scorer.build_query_anchor_profile(query, keywords, analysis)
        profile = self.graph_scorer.enrich_profile_with_corpus(
            profile,
            [" ".join([chunk.title or "", chunk.section or "", chunk.subsection or "", chunk.text[:2000] or ""]) for chunk in chunks],
        )
        score_scale, score_scale_debug = self._local_chunk_score_scale(chunks)

        relation_chunk_support: dict[str, float] = defaultdict(float)
        relation_type_boosts = getattr(analysis, "relation_type_boosts", {}) or {}
        for relation in relations:
            if relation.score <= 0.0:
                continue
            # Intent-compatible relations get extra support, but this support is
            # normalized and capped below instead of being added as raw score.
            type_weight = float(relation_type_boosts.get(relation.relation_type, 1.0) or 1.0)
            if relation.relation_type not in relation_type_boosts:
                type_weight = min(type_weight, 1.0)
            for chunk_id in relation.evidence_chunk_ids:
                if chunk_id:
                    relation_chunk_support[chunk_id] += float(relation.score or 0.0) * max(type_weight, 0.0)

        max_relation_support = max(relation_chunk_support.values(), default=0.0)
        max_relation_delta = score_scale * self.config.relation_evidence_max_scale

        ranked: list[RetrievedChunk] = []
        for chunk in chunks:
            text = " ".join([chunk.title, chunk.section, chunk.subsection, chunk.text[:1600]])
            breakdown = self.graph_scorer.score_text_against_profile(text, profile)
            title_breakdown = self.graph_scorer.score_text_against_profile(chunk.title, profile)
            anchor_score = float(breakdown.get("anchor_score", 0.0) or 0.0)
            title_anchor_score = float(title_breakdown.get("anchor_score", 0.0) or 0.0)
            anchor_gate = self.graph_scorer.anchor_gate(
                anchor_score=anchor_score,
                title_anchor_score=title_anchor_score,
                anchor_max_specificity=float(breakdown.get("anchor_max_specificity", 0.0) or 0.0),
            )

            original = float(chunk.score or 0.0)

            # Keep body-only lexical evidence calibrated to the current Hippo
            # score scale. Exact title anchors remain allowed to rescue a chunk
            # because they are much less likely to be generic symptom overlap.
            generic_raw = (
                float(breakdown.get("generic_score", 0.0) or 0.0)
                * self.config.generic_term_chunk_weight
                * (0.20 + anchor_gate)
            )
            generic_delta = min(generic_raw, score_scale * (1.0 + 1.5 * anchor_gate))

            anchor_raw = (
                anchor_score
                * self.config.anchor_term_chunk_weight
                * (0.30 + 0.70 * anchor_gate)
            )
            anchor_delta = min(anchor_raw, score_scale * (4.0 + 10.0 * anchor_gate))

            title_anchor_raw = title_anchor_score * self.config.title_anchor_chunk_weight
            if title_anchor_score >= 0.85:
                title_anchor_delta = min(title_anchor_raw, max(0.55, score_scale * 60.0))
            elif title_anchor_score > 0.0:
                title_anchor_delta = min(title_anchor_raw, score_scale * 8.0)
            else:
                title_anchor_delta = 0.0

            slot_raw = (
                float(breakdown.get("slot_score", 0.0) or 0.0)
                * self.config.slot_term_chunk_weight
                * max(anchor_gate, self.config.slot_anchor_gate_floor)
            )
            slot_delta = min(slot_raw, score_scale * (1.5 + 6.0 * anchor_gate))

            relation_support = float(relation_chunk_support.get(chunk.chunk_id, 0.0) or 0.0)
            relation_norm = relation_support / max_relation_support if max_relation_support > 0.0 else 0.0
            relation_delta = (
                relation_norm
                * max_relation_delta
                * self.config.relation_evidence_chunk_weight
                * max(anchor_gate, self.config.relation_anchor_gate_floor)
            )
            relation_delta = min(relation_delta, max_relation_delta)

            boost = generic_delta + anchor_delta + title_anchor_delta + slot_delta + relation_delta
            chunk.score = original + boost
            chunk.metadata = {
                **dict(chunk.metadata or {}),
                "hipporag_original_chunk_score": original,
                "hipporag_query_evidence_boost": boost,
                "hipporag_anchor_gate": anchor_gate,
                "hipporag_score_scale": score_scale,
                "hipporag_score_scale_debug": score_scale_debug,
                "hipporag_query_anchor_profile": self.graph_scorer.profile_debug(profile, max_terms=12),
                "hipporag_query_term_matches": breakdown.get("query_matches", [])[:8],
                "hipporag_generic_term_matches": breakdown.get("generic_matches", [])[:8],
                "hipporag_slot_term_matches": breakdown.get("slot_matches", [])[:8],
                "hipporag_anchor_term_matches": breakdown.get("anchor_matches", [])[:8],
                "hipporag_query_evidence_breakdown": {
                    **breakdown,
                    "title_anchor_matches": title_breakdown.get("anchor_matches", [])[:8],
                    "title_anchor_score": title_anchor_score,
                    "generic_raw": generic_raw,
                    "generic_delta": generic_delta,
                    "anchor_raw": anchor_raw,
                    "anchor_delta": anchor_delta,
                    "title_anchor_raw": title_anchor_raw,
                    "title_anchor_delta": title_anchor_delta,
                    "slot_raw": slot_raw,
                    "slot_delta": slot_delta,
                    "relation_support_raw": relation_support,
                    "relation_support_norm": relation_norm,
                    "relation_delta": relation_delta,
                    "relation_delta_cap": max_relation_delta,
                },
            }
            ranked.append(chunk)

        return sorted(ranked, key=lambda item: item.score, reverse=True)

    def _apply_final_source_diversity(
        self,
        *,
        query: str,
        keywords: QueryKeywords,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Final Hippo guard against expansion/source drift.

        This is not disease/source hard-coding. It uses only mined query anchors,
        chunk provenance and source/article counts. The goal is to prevent one
        drifted graph source, especially graph_expansion chunks, from occupying
        most of the returned top-k when passage/vector-supported alternatives
        exist.
        """
        if not chunks:
            return chunks

        profile = self.graph_scorer.build_query_anchor_profile(query, keywords, None)
        profile = self.graph_scorer.enrich_profile_with_corpus(
            profile,
            [" ".join([chunk.title or "", chunk.section or "", chunk.subsection or "", chunk.text[:1200] or ""]) for chunk in chunks],
        )
        window = max(1, int(self.config.final_diversity_window or 5))
        max_per_source = max(1, int(self.config.final_max_chunks_per_source or 2))
        max_expansion_per_source = max(1, int(self.config.final_max_graph_expansion_per_source or 1))
        preserve_passage = max(0, int(self.config.final_preserve_passage_chunks or 0))
        relax_threshold = float(self.config.final_anchor_relax_threshold or 1.4)

        sorted_chunks = sorted(chunks, key=lambda item: float(item.score or 0.0), reverse=True)
        selected: list[RetrievedChunk] = []
        used_ids: set[str] = set()
        source_counts: dict[str, int] = defaultdict(int)
        expansion_counts: dict[str, int] = defaultdict(int)

        def support_scores(chunk: RetrievedChunk) -> tuple[float, float]:
            body = self.graph_scorer.score_text_against_profile(
                " ".join([chunk.title or "", chunk.section or "", chunk.subsection or "", chunk.text[:1200] or ""]),
                profile,
            )
            title = self.graph_scorer.score_text_against_profile(chunk.title or "", profile)
            return float(body.get("anchor_score", 0.0) or 0.0), float(title.get("anchor_score", 0.0) or 0.0)

        def is_passage_supported(chunk: RetrievedChunk) -> bool:
            meta = dict(chunk.metadata or {})
            return (
                chunk.source == "hipporag_passage"
                or bool(meta.get("hipporag_passage_integration"))
                or "vector_original_score" in meta
                or "passage_added_score" in meta
            )

        def can_add(chunk: RetrievedChunk, *, strict_window: bool) -> bool:
            key = self._source_key(chunk)
            if not key:
                return True
            count = source_counts.get(key, 0)
            expansion_count = expansion_counts.get(key, 0)
            body_anchor, title_anchor = support_scores(chunk)
            strong_title_anchor = title_anchor >= relax_threshold

            if strict_window:
                # In the guarded top window, graph expansion is allowed as a
                # supplement, but not as repeated evidence from the same source
                # unless the article title itself strongly matches the mined
                # query anchors.
                if chunk.source == "graph_expansion" and expansion_count >= max_expansion_per_source and not strong_title_anchor:
                    return False
                if count >= max_per_source and not strong_title_anchor:
                    return False
                # Body-only broad anchors should not relax the source cap. They
                # still affect score inside the source, but the final context
                # must remain diverse.
                if count >= 1 and chunk.source == "graph_expansion" and title_anchor <= 0.0 and body_anchor < relax_threshold:
                    return False
            return True

        def add(chunk: RetrievedChunk) -> bool:
            if chunk.chunk_id in used_ids:
                return False
            key = self._source_key(chunk)
            selected.append(chunk)
            used_ids.add(chunk.chunk_id)
            if key:
                source_counts[key] += 1
                if chunk.source == "graph_expansion":
                    expansion_counts[key] += 1
            return True

        # Preserve a small number of passage/vector-supported chunks with mined
        # anchor support. These often represent source-local evidence that PPR
        # alone under-ranks, and they protect Hippo standalone from graph drift.
        passage_candidates = []
        for chunk in sorted_chunks:
            if not is_passage_supported(chunk):
                continue
            body_anchor, title_anchor = support_scores(chunk)
            if max(body_anchor, title_anchor) <= 0.0:
                continue
            passage_candidates.append((max(body_anchor, title_anchor), float(chunk.score or 0.0), chunk))
        passage_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, chunk in passage_candidates[:preserve_passage]:
            if len(selected) >= window:
                break
            if can_add(chunk, strict_window=True):
                add(chunk)

        # Fill the guarded window with source/expansion caps.
        for chunk in sorted_chunks:
            if len(selected) >= window:
                break
            if chunk.chunk_id in used_ids:
                continue
            if can_add(chunk, strict_window=True):
                add(chunk)

        # If the pool is too narrow, fill the rest of the requested window while
        # still avoiding exact duplicates. This avoids returning fewer chunks.
        for chunk in sorted_chunks:
            if len(selected) >= window:
                break
            if chunk.chunk_id not in used_ids:
                add(chunk)

        # Append the remaining chunks after the guarded window, preserving score
        # order. The caller still slices max_chunks after this stage.
        for chunk in sorted_chunks:
            if chunk.chunk_id not in used_ids:
                add(chunk)

        return selected

    def _source_key(self, chunk: RetrievedChunk) -> str:
        return self._norm_url(chunk.source_url) or chunk.article_id or chunk.title or ""

    def _norm_url(self, value: str | None) -> str:
        value = (value or "").strip().lower()
        value = value.replace("http://", "https://")
        return value.rstrip("/")

    def _local_chunk_score_scale(self, chunks: list[RetrievedChunk]) -> tuple[float, dict]:
        scores = sorted([float(chunk.score or 0.0) for chunk in chunks], reverse=True)
        if not scores:
            return self.config.score_scale_floor, {"reason": "empty"}
        top_score = abs(scores[0])
        gaps = [abs(scores[idx] - scores[idx + 1]) for idx in range(len(scores) - 1)]
        positive_gaps = [gap for gap in gaps if gap > 1e-12]
        median_gap = sorted(positive_gaps)[len(positive_gaps) // 2] if positive_gaps else 0.0
        fallback = max(top_score * 0.03, self.config.score_scale_floor)
        score_scale = max(median_gap, fallback)
        return score_scale, {
            "scores_top": scores[:10],
            "positive_gaps": positive_gaps[:10],
            "median_gap": median_gap,
            "fallback": fallback,
            "score_scale": score_scale,
        }

    def _important_terms(self, query: str) -> list[str]:
        # Backward-compatible wrapper. Keep extraction centralized in
        # GraphQueryScorer so we do not maintain separate medical keyword lists
        # inside HippoRAG.
        return self.graph_scorer.build_query_anchor_profile(query, None, None).query_terms[:16]

    def _debug_chunk_snapshot(self, chunk: RetrievedChunk, rank: int) -> dict:
        meta = dict(chunk.metadata or {})
        return {
            "rank": rank,
            "chunk_id": chunk.chunk_id,
            "score": float(chunk.score or 0.0),
            "source": chunk.source,
            "article_id": chunk.article_id,
            "title": chunk.title,
            "section": chunk.section,
            "subsection": chunk.subsection,
            "source_url": chunk.source_url,
            "metadata": {
                key: meta.get(key)
                for key in [
                    "hipporag_original_chunk_score",
                    "hipporag_query_evidence_boost",
                    "hipporag_query_term_matches",
                    "hipporag_slot_term_matches",
                    "hipporag_anchor_term_matches",
                    "hipporag_generic_term_matches",
                    "hipporag_anchor_gate",
                    "hipporag_score_scale",
                    "hipporag_query_anchor_profile",
                    "hipporag_query_evidence_breakdown",
                    "hipporag_passage_integration",
                    "passage_added_score",
                    "evidence_expansion",
                    "evidence_expansion_reasons",
                    "evidence_expansion_raw_score",
                    "evidence_expansion_article_base",
                    "evidence_expansion_article_anchor",
                    "evidence_expansion_bonus",
                    "evidence_expansion_adjacent_bonus",
                    "evidence_expansion_score_cap",
                    "evidence_expansion_breakdown",
                    "supporting_entity_ids",
                    "supporting_relation_ids",
                ]
                if key in meta
            },
        }

    def _norm(self, text: str) -> str:
        text = (text or "").strip().lower().replace("đ", "d")
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

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

    def _dedupe_norm(self, items: Iterable[str]) -> list[str]:
        """Dedupe seed query variants by the shared graph normalizer.

        V4 added mechanically mined query phrases to Hippo seed retrieval, but
        accidentally called this helper before defining it. Keeping the helper
        local avoids reaching into GraphQueryScorer private methods while still
        using the exact same normalization semantics.
        """
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = self.graph_scorer.norm(item or "")
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item.strip() if isinstance(item, str) else str(item))
        return result
