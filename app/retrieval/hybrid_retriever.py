from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.retrieval.graph_scoring import GraphQueryScorer
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
    vector_weight: float = 1.0
    lightrag_weight: float = 1.0
    hipporag_weight: float = 0.75
    enable_confidence_weighting: bool = True

    # Final guard: mixed should keep a strong vector backbone and should not let
    # one generic graph article/source occupy the whole final context window.
    preserve_vector_top_k: int = 2
    preserve_lightrag_top_k: int = 2
    preserve_hipporag_top_k: int = 2
    preserve_min_branch_confidence: float = 0.55
    max_chunks_per_article: int = 3
    max_graph_only_per_article: int = 2
    max_graph_expansion_chunks: int = 3
    article_cap_anchor_override: float = 0.85


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
        self.graph_scorer = GraphQueryScorer()

    def retrieve(self, query: str) -> RetrievalResult:
        vector_result = self.vector_retriever.retrieve(
            query,
            top_k=self.config.vector_top_k,
        )
        lightrag_result = self.lightrag_retriever.retrieve(query)
        hipporag_result = self.hipporag_retriever.retrieve(query)

        keywords = self._merge_keywords(lightrag_result.keywords, hipporag_result.keywords)
        branch_weights = self._branch_weights(query, vector_result, lightrag_result, hipporag_result)
        fused_all = self._rrf_fuse_chunks([
            (vector_result.chunks, branch_weights["vector"]),
            (lightrag_result.chunks, branch_weights["lightrag"]),
            (hipporag_result.chunks, branch_weights["hipporag"]),
        ])
        fused_chunks, guard_debug = self._apply_final_guard(
            query=query,
            keywords=keywords,
            fused_chunks=fused_all,
            vector_chunks=vector_result.chunks,
            lightrag_chunks=lightrag_result.chunks,
            hipporag_chunks=hipporag_result.chunks,
            lightrag_confidence=float(
                lightrag_result.debug.get("graph_confidence")
                or self.graph_scorer.branch_confidence(lightrag_result.chunks, query, keywords)
            ),
            hipporag_confidence=float(
                hipporag_result.debug.get("graph_confidence")
                or self.graph_scorer.branch_confidence(hipporag_result.chunks, query, keywords)
            ),
        )
        entities = self._merge_entities(lightrag_result.entities + hipporag_result.entities)
        relations = self._merge_relations(lightrag_result.relations + hipporag_result.relations)

        return RetrievalResult(
            query=query,
            method="mixed",
            keywords=keywords,
            chunks=fused_chunks,
            entities=entities[: self.config.max_entities],
            relations=relations[: self.config.max_relations],
            debug={
                "vector_chunk_count": len(vector_result.chunks),
                "lightrag_chunk_count": len(lightrag_result.chunks),
                "hipporag_chunk_count": len(hipporag_result.chunks),
                "rrf_k": self.config.rrf_k,
                "branch_weights": branch_weights,
                "fusion": "confidence-weighted RRF + final vector/source diversity guard",
                "fusion_guard": guard_debug,
                "branch_debug": {
                    "vector": vector_result.debug,
                    "lightrag": lightrag_result.debug,
                    "hipporag": hipporag_result.debug,
                },
            },
        )

    def _branch_weights(
        self,
        query: str,
        vector_result: RetrievalResult,
        lightrag_result: RetrievalResult,
        hipporag_result: RetrievalResult,
    ) -> dict[str, float]:
        weights = {
            "vector": self.config.vector_weight,
            "lightrag": self.config.lightrag_weight,
            "hipporag": self.config.hipporag_weight,
        }
        if not self.config.enable_confidence_weighting:
            return weights

        keywords = self._merge_keywords(lightrag_result.keywords, hipporag_result.keywords)
        light_conf = float(
            lightrag_result.debug.get("graph_confidence")
            or self.graph_scorer.branch_confidence(lightrag_result.chunks, query, keywords)
        )
        hippo_conf = float(
            hipporag_result.debug.get("graph_confidence")
            or self.graph_scorer.branch_confidence(hipporag_result.chunks, query, keywords)
        )
        weights["lightrag"] *= 0.55 + 0.70 * max(0.0, min(light_conf, 1.0))
        weights["hipporag"] *= 0.45 + 0.75 * max(0.0, min(hippo_conf, 1.0))
        return weights

    def _rrf_fuse_chunks(self, ranked_lists: list[tuple[list[RetrievedChunk], float]]) -> list[RetrievedChunk]:
        scores: dict[str, float] = defaultdict(float)
        best: dict[str, RetrievedChunk] = {}
        sources: dict[str, list[str]] = defaultdict(list)
        branch_ranks: dict[str, dict[str, int]] = defaultdict(dict)

        for ranked, weight in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                scores[chunk.chunk_id] += float(weight) / (self.config.rrf_k + rank)
                if chunk.source not in sources[chunk.chunk_id]:
                    sources[chunk.chunk_id].append(str(chunk.source))
                branch_ranks[chunk.chunk_id][str(chunk.source)] = rank
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
                    **dict(chunk.metadata or {}),
                    "mixed_sources": sources[chunk_id],
                    "mixed_branch_ranks": branch_ranks[chunk_id],
                    "best_branch_score": chunk.score,
                },
            )
            result.append(fused)

        return sorted(result, key=lambda x: x.score, reverse=True)

    def _apply_final_guard(
        self,
        *,
        query: str,
        keywords: QueryKeywords | None,
        fused_chunks: list[RetrievedChunk],
        vector_chunks: list[RetrievedChunk],
        lightrag_chunks: list[RetrievedChunk],
        hipporag_chunks: list[RetrievedChunk],
        lightrag_confidence: float,
        hipporag_confidence: float,
    ) -> tuple[list[RetrievedChunk], dict]:
        by_id = {chunk.chunk_id: chunk for chunk in fused_chunks}
        profile = self.graph_scorer.build_query_anchor_profile(query, keywords, None)
        selected: list[RetrievedChunk] = []
        selected_ids: set[str] = set()
        article_counts: dict[str, int] = defaultdict(int)
        graph_only_article_counts: dict[str, int] = defaultdict(int)
        expansion_count = 0
        guard_events: list[dict] = []

        vector_confidence = self._vector_confidence(vector_chunks)
        preserve_vector_count = min(self.config.preserve_vector_top_k, len(vector_chunks))
        if vector_confidence < self.config.preserve_min_branch_confidence and preserve_vector_count > 1:
            preserve_vector_count = 1

        def fused_for(chunk: RetrievedChunk) -> RetrievedChunk | None:
            return by_id.get(chunk.chunk_id)

        def add_chunk(chunk: RetrievedChunk, reason: str, bypass_caps: bool = False) -> bool:
            nonlocal expansion_count
            if chunk.chunk_id in selected_ids:
                return False
            key = self._source_key(chunk)
            anchor_support = self._chunk_anchor_support(chunk, profile)
            is_expansion = self._is_graph_expansion(chunk)
            graph_only = self._is_graph_only(chunk)

            if not bypass_caps:
                if is_expansion and expansion_count >= self.config.max_graph_expansion_chunks:
                    guard_events.append({
                        "action": "skip",
                        "reason": "expansion_cap",
                        "chunk_id": chunk.chunk_id,
                        "article_key": key,
                    })
                    return False
                if key and article_counts[key] >= self.config.max_chunks_per_article and anchor_support < self.config.article_cap_anchor_override:
                    guard_events.append({
                        "action": "skip",
                        "reason": "article_cap",
                        "chunk_id": chunk.chunk_id,
                        "article_key": key,
                        "anchor_support": anchor_support,
                    })
                    return False
                if (
                    graph_only
                    and key
                    and graph_only_article_counts[key] >= self.config.max_graph_only_per_article
                    and anchor_support < self.config.article_cap_anchor_override
                ):
                    guard_events.append({
                        "action": "skip",
                        "reason": "graph_only_article_cap",
                        "chunk_id": chunk.chunk_id,
                        "article_key": key,
                        "anchor_support": anchor_support,
                    })
                    return False

            chunk.metadata = {
                **dict(chunk.metadata or {}),
                "mixed_guard_selected_reason": reason,
                "mixed_guard_anchor_support": anchor_support,
            }
            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
            if key:
                article_counts[key] += 1
                if graph_only:
                    graph_only_article_counts[key] += 1
            if is_expansion:
                expansion_count += 1
            guard_events.append({
                "action": "select",
                "reason": reason,
                "chunk_id": chunk.chunk_id,
                "article_key": key,
                "anchor_support": anchor_support,
            })
            return True

        for chunk in vector_chunks[:preserve_vector_count]:
            fused = fused_for(chunk)
            if fused is not None:
                add_chunk(fused, reason="preserve_vector", bypass_caps=True)

        preserve_lightrag_count = 0
        if lightrag_confidence >= self.config.preserve_min_branch_confidence:
            for chunk in lightrag_chunks[: self.config.preserve_lightrag_top_k]:
                fused = fused_for(chunk)
                if fused is not None and add_chunk(fused, reason="preserve_lightrag", bypass_caps=True):
                    preserve_lightrag_count += 1

        preserve_hipporag_count = 0
        if hipporag_confidence >= self.config.preserve_min_branch_confidence:
            for chunk in hipporag_chunks[: self.config.preserve_hipporag_top_k]:
                fused = fused_for(chunk)
                if fused is not None and add_chunk(fused, reason="preserve_hipporag", bypass_caps=True):
                    preserve_hipporag_count += 1

        for chunk in fused_chunks:
            if len(selected) >= self.config.max_chunks:
                break
            add_chunk(chunk, reason="rrf_diversified", bypass_caps=False)

        # Backfill only if caps were too strict. Keep this last so trace clearly
        # shows which chunks bypassed the guard due to insufficient context.
        if len(selected) < self.config.max_chunks:
            for chunk in fused_chunks:
                if len(selected) >= self.config.max_chunks:
                    break
                add_chunk(chunk, reason="backfill_after_guard", bypass_caps=True)

        # Preserve/guard decides membership, but the final retrieval ranking
        # should still be ordered by fused score. Otherwise a preserved low-score
        # vector chunk can appear above stronger fused evidence, as seen in q076.
        final_selected = sorted(selected, key=lambda item: item.score, reverse=True)[: self.config.max_chunks]

        return final_selected, {
            "vector_confidence": vector_confidence,
            "lightrag_confidence": lightrag_confidence,
            "hipporag_confidence": hipporag_confidence,
            "preserve_vector_count": preserve_vector_count,
            "preserve_lightrag_count": preserve_lightrag_count,
            "preserve_hipporag_count": preserve_hipporag_count,
            "configured_preserve_lightrag_top_k": self.config.preserve_lightrag_top_k,
            "configured_preserve_hipporag_top_k": self.config.preserve_hipporag_top_k,
            "max_chunks_per_article": self.config.max_chunks_per_article,
            "max_graph_only_per_article": self.config.max_graph_only_per_article,
            "max_graph_expansion_chunks": self.config.max_graph_expansion_chunks,
            "final_article_counts": dict(article_counts),
            "final_graph_only_article_counts": dict(graph_only_article_counts),
            "final_expansion_count": expansion_count,
            "events_top": guard_events[:40],
            "lightrag_candidate_count": len(lightrag_chunks),
            "hipporag_candidate_count": len(hipporag_chunks),
            "branch_preservation": "preserve top candidates from vector, LightRAG and HippoRAG before diversified RRF fill",
            "final_sort": "selected_by_guard_then_sorted_by_fused_score",
        }

    def _vector_confidence(self, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        scores = [abs(float(chunk.score or 0.0)) for chunk in chunks[:5]]
        if len(scores) == 1:
            return 1.0
        top = scores[0]
        second = scores[1]
        total_top3 = sum(scores[:3]) or top or 1.0
        dominance = top / total_top3 if total_top3 > 0.0 else 0.0
        gap = (top - second) / top if top > 0.0 else 0.0
        return max(0.0, min(1.0, 0.65 * dominance + 0.35 * max(gap, 0.0)))

    def _chunk_anchor_support(self, chunk: RetrievedChunk, profile) -> float:
        text = " ".join([chunk.title or "", chunk.section or "", chunk.text[:900] or ""])
        breakdown = self.graph_scorer.score_text_against_profile(text, profile)
        return float(breakdown.get("anchor_score", 0.0) or 0.0)

    def _source_key(self, chunk: RetrievedChunk) -> str:
        return self._norm_url(chunk.source_url) or str(chunk.article_id or chunk.title or chunk.chunk_id)

    def _norm_url(self, value: str) -> str:
        value = (value or "").strip().lower()
        value = value.replace("http://", "https://")
        return value.rstrip("/")

    def _is_graph_expansion(self, chunk: RetrievedChunk) -> bool:
        meta = dict(chunk.metadata or {})
        sources = set(str(item) for item in meta.get("mixed_sources", []) or [])
        return bool(meta.get("evidence_expansion") or chunk.source == "graph_expansion" or "graph_expansion" in sources)

    def _is_graph_only(self, chunk: RetrievedChunk) -> bool:
        meta = dict(chunk.metadata or {})
        sources = set(str(item) for item in meta.get("mixed_sources", []) or [])
        if not sources:
            return chunk.source not in {"vector", "vector_naive"}
        return not any(source in sources for source in {"vector", "vector_naive"})

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