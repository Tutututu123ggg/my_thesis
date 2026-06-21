from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from app.infrastructure.graph_database import GraphRepository
from app.retrieval.graph_scoring import GraphQueryScorer
from app.retrieval.query_analyzer import QueryAnalysis
from app.retrieval.retrieval_models import RetrievedChunk, RetrievedEntity, RetrievedRelation


@dataclass
class EvidenceExpansionConfig:
    enabled: bool = True
    max_articles: int = 4
    max_candidate_chunks: int = 160
    max_extra_per_article: int = 3
    max_total_chunks: int = 12

    # Minimum final score after calibrating to the caller retriever score scale.
    min_extra_score: float = 0.15

    # Raw evidence-slot scoring. Raw scores are NEVER used directly as final
    # chunk scores. They are normalized inside the candidate pool and mapped
    # back to the score scale of the base retriever result.
    query_term_weight: float = 0.10
    slot_term_weight: float = 0.28
    anchor_term_weight: float = 0.45
    anchor_disease_weight: float = 0.75
    graph_disease_weight: float = 0.10
    section_weight: float = 0.18
    relation_source_weight: float = 0.08
    anchor_article_weight: float = 0.75
    title_anchor_weight: float = 0.55

    # Score calibration: expansion is a small delta on top of the best original
    # chunk from the same article. It must speak the same score language as the
    # caller retriever (HippoRAG/LightRAG), instead of inventing scores like 6.75.
    expansion_delta_scale: float = 0.35
    weak_anchor_delta_scale: float = 0.12
    adjacent_delta_scale: float = 0.12
    score_scale_floor: float = 0.05
    top_score_floor_ratio: float = 0.03
    generic_only_multiplier: float = 0.35
    weak_article_multiplier_floor: float = 0.25
    min_anchor_for_full_delta: float = 1.0

    # Adjacent chunks often contain complementary evidence from the same section
    # or neighboring section. The bonus is calibrated to the same local score
    # scale and gated by anchor support.
    enable_adjacent_chunk_expansion: bool = True
    adjacent_window: int = 2
    max_adjacent_seed_chunks: int = 12


class GraphEvidenceExpander:
    """Controlled same-article expansion for graph-based retrievers.

    The expander is a context supplement, not a new retriever. It first scores
    candidates with an internal raw evidence score, then normalizes that score
    within the candidate pool and converts it into a small delta on the same
    scale as the original retriever chunks. This prevents expanded chunks from
    resetting to a different scoring regime and dominating HippoRAG/LightRAG.
    """

    def __init__(self, graph_repo: GraphRepository, config: EvidenceExpansionConfig | None = None):
        self.graph_repo = graph_repo
        self.config = config or EvidenceExpansionConfig()
        self.graph_scorer = GraphQueryScorer()

    def expand_chunks(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        entities: list[RetrievedEntity],
        relations: list[RetrievedRelation],
        analysis: QueryAnalysis,
    ) -> tuple[list[RetrievedChunk], dict]:
        if not self.config.enabled or not analysis.enable_same_article_expansion:
            return chunks, {"enabled": False, "reason": "disabled_or_not_needed"}
        if not chunks:
            return chunks, {"enabled": False, "reason": "no_base_chunks"}

        anchor_terms = self._anchor_terms(query, entities)
        query_terms = self._important_terms(query)
        slot_terms = [self._norm(item) for item in analysis.evidence_slot_keywords if item]
        anchor_disease_terms = self._anchor_disease_terms(query, entities)
        graph_disease_terms = [
            term for term in self._disease_terms(entities, relations)
            if term not in set(anchor_disease_terms)
        ]
        relation_source_urls = {self._norm_url(r.source_url) for r in relations if r.source_url}

        article_ids = self._candidate_article_ids(
            chunks,
            query=query,
            entities=entities,
            relations=relations,
            analysis=analysis,
            anchor_terms=anchor_terms,
            anchor_disease_terms=anchor_disease_terms,
            max_articles=self.config.max_articles,
        )
        if not article_ids:
            return chunks, {"enabled": False, "reason": "no_article_ids"}

        candidates = self.graph_repo.get_chunks_by_article_ids(
            article_ids=article_ids,
            limit=self.config.max_candidate_chunks,
        )
        existing_ids = {chunk.chunk_id for chunk in chunks}
        base_article_scores = self._base_article_scores(chunks)
        base_article_anchor_scores = self._base_article_anchor_scores(chunks, anchor_terms, anchor_disease_terms)
        top_original_score = max(float(chunk.score or 0.0) for chunk in chunks)
        score_scale, score_scale_debug = self._local_score_scale(chunks)

        adjacent_candidate_ids: set[str] = set()
        adjacent_seed_ids = [chunk.chunk_id for chunk in chunks[: self.config.max_adjacent_seed_chunks] if chunk.chunk_id]
        if self.config.enable_adjacent_chunk_expansion and adjacent_seed_ids:
            adjacent_candidates = self.graph_repo.get_adjacent_chunks_by_chunk_ids(
                chunk_ids=adjacent_seed_ids,
                window=self.config.adjacent_window,
                limit=self.config.max_candidate_chunks,
            )
            adjacent_candidate_ids = {node.chunk_id for node in adjacent_candidates}
            candidates = self._merge_chunk_nodes([*candidates, *adjacent_candidates])

        if not candidates:
            return chunks, {
                "enabled": False,
                "reason": "no_same_article_or_adjacent_candidates",
                "article_ids": article_ids,
            }

        records: list[dict] = []
        candidate_debug: list[dict] = []

        for node in candidates:
            if node.chunk_id in existing_ids:
                continue
            raw_score, reasons, breakdown = self._score_candidate(
                node=node,
                query_terms=query_terms,
                slot_terms=slot_terms,
                anchor_terms=anchor_terms,
                anchor_disease_terms=anchor_disease_terms,
                graph_disease_terms=graph_disease_terms,
                relation_source_urls=relation_source_urls,
            )
            article_base = base_article_scores.get(node.article_id, 0.0)
            article_anchor = max(
                base_article_anchor_scores.get(node.article_id, 0.0),
                float(breakdown.get("anchor_score", 0.0)),
                float(breakdown.get("anchor_disease_score", 0.0)),
            )
            if raw_score <= 0.0 and article_anchor <= 0.0:
                continue
            records.append({
                "node": node,
                "raw_score": raw_score,
                "reasons": reasons,
                "breakdown": breakdown,
                "article_base": article_base,
                "article_anchor": article_anchor,
                "is_adjacent": node.chunk_id in adjacent_candidate_ids,
            })

        if not records:
            return chunks, {
                "enabled": False,
                "reason": "no_positive_expansion_candidates",
                "article_ids": article_ids,
                "query_terms": query_terms,
                "anchor_terms": anchor_terms,
                "anchor_disease_terms": anchor_disease_terms,
            }

        raw_values = [float(record["raw_score"] or 0.0) for record in records]
        raw_min = min(raw_values)
        raw_max = max(raw_values)
        sorted_raw = sorted(raw_values, reverse=True)
        raw_rank_lookup: dict[float, int] = {}
        for rank, value in enumerate(sorted_raw, start=1):
            raw_rank_lookup.setdefault(value, rank)

        extras_by_article: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for record in records:
            node = record["node"]
            raw_score = float(record["raw_score"] or 0.0)
            reasons = list(record["reasons"])
            breakdown = dict(record["breakdown"])
            article_base = float(record["article_base"] or 0.0)
            article_anchor = float(record["article_anchor"] or 0.0)
            is_adjacent = bool(record["is_adjacent"])

            raw_minmax = self._minmax_norm(raw_score, raw_min, raw_max)
            raw_rank = raw_rank_lookup.get(raw_score, len(records))
            raw_rank_norm = 1.0 / math.log2(raw_rank + 1.0) if raw_rank > 0 else 0.0
            raw_norm = 0.65 * raw_minmax + 0.35 * raw_rank_norm

            anchor_factor = self._anchor_factor(article_anchor)
            if breakdown.get("generic_only", False):
                anchor_factor *= self.config.generic_only_multiplier
                reasons.append("low_specificity_without_anchor_scaled_down")

            if article_base <= 0.0:
                # Adjacent candidates from articles not represented by the core
                # retrieval result are allowed only as weak supplements.
                anchor_factor *= self.config.weak_article_multiplier_floor
                reasons.append("no_base_article_score_scaled_down")

            delta_scale = self.config.expansion_delta_scale
            if article_anchor < self.config.min_anchor_for_full_delta:
                delta_scale = min(delta_scale, self.config.weak_anchor_delta_scale)
                reasons.append("weak_anchor_delta_scale")

            expansion_delta = raw_norm * score_scale * delta_scale * anchor_factor
            adjacent_delta = 0.0
            if is_adjacent:
                adjacent_delta = score_scale * self.config.adjacent_delta_scale * min(article_anchor, 1.0)
                if adjacent_delta > 0.0:
                    reasons.append("adjacent_delta_anchor_gated")

            score = article_base + expansion_delta + adjacent_delta
            if score < self.config.min_extra_score:
                candidate_debug.append({
                    "chunk_id": node.chunk_id,
                    "article_id": node.article_id,
                    "raw_score": raw_score,
                    "raw_norm": raw_norm,
                    "score_scale": score_scale,
                    "final_score": score,
                    "dropped": True,
                    "reasons": reasons,
                    "breakdown": breakdown,
                })
                continue

            chunk = RetrievedChunk(
                chunk_id=node.chunk_id,
                score=score,
                source="graph_expansion",
                article_id=node.article_id,
                title=node.title,
                section=node.section or "",
                subsection=node.subsection or "",
                source_url=node.source_url,
                text=node.text,
                contextualized_text=node.contextualized_text or node.text,
                metadata={
                    **dict(node.metadata or {}),
                    "evidence_expansion": True,
                    "evidence_expansion_reasons": reasons,
                    "evidence_expansion_intent": analysis.intent,
                    "evidence_expansion_raw_score": raw_score,
                    "evidence_expansion_raw_min": raw_min,
                    "evidence_expansion_raw_max": raw_max,
                    "evidence_expansion_raw_minmax": raw_minmax,
                    "evidence_expansion_raw_rank": raw_rank,
                    "evidence_expansion_raw_rank_norm": raw_rank_norm,
                    "evidence_expansion_raw_norm": raw_norm,
                    "evidence_expansion_score_scale": score_scale,
                    "evidence_expansion_delta_scale": delta_scale,
                    "evidence_expansion_anchor_factor": anchor_factor,
                    "evidence_expansion_article_base": article_base,
                    "evidence_expansion_article_anchor": article_anchor,
                    "evidence_expansion_delta": expansion_delta,
                    "evidence_expansion_adjacent_delta": adjacent_delta,
                    "evidence_expansion_breakdown": breakdown,
                },
            )
            extras_by_article[node.article_id].append(chunk)
            candidate_debug.append({
                "chunk_id": node.chunk_id,
                "article_id": node.article_id,
                "raw_score": raw_score,
                "raw_norm": raw_norm,
                "raw_rank": raw_rank,
                "score_scale": score_scale,
                "final_score": score,
                "article_base": article_base,
                "article_anchor": article_anchor,
                "anchor_factor": anchor_factor,
                "delta": expansion_delta,
                "adjacent_delta": adjacent_delta,
                "reasons": reasons,
                "breakdown": breakdown,
            })

        extras: list[RetrievedChunk] = []
        for article_id, article_chunks in extras_by_article.items():
            article_chunks.sort(key=lambda item: item.score, reverse=True)
            extras.extend(article_chunks[: self.config.max_extra_per_article])

        merged = self._merge_chunks(chunks + extras)
        if analysis.enable_source_diversity:
            merged = self._apply_source_diversity(merged, anchor_terms=anchor_terms, anchor_disease_terms=anchor_disease_terms)
        merged = sorted(merged, key=lambda item: item.score, reverse=True)[: self.config.max_total_chunks]

        return merged, {
            "enabled": True,
            "intent": analysis.intent,
            "article_ids": article_ids,
            "candidate_count": len(candidates),
            "adjacent_candidate_count": len(adjacent_candidate_ids),
            "adjacent_window": self.config.adjacent_window if self.config.enable_adjacent_chunk_expansion else 0,
            "extra_count": len(extras),
            "query_terms": query_terms,
            "anchor_terms": anchor_terms,
            "anchor_disease_terms": anchor_disease_terms,
            "graph_disease_terms": graph_disease_terms[:20],
            "slot_terms": analysis.evidence_slot_keywords,
            "base_article_scores": base_article_scores,
            "base_article_anchor_scores": base_article_anchor_scores,
            "top_original_score": top_original_score,
            "score_scale": score_scale,
            "score_scale_debug": score_scale_debug,
            "raw_score_min": raw_min,
            "raw_score_max": raw_max,
            "candidate_debug_top": sorted(candidate_debug, key=lambda item: item.get("final_score", 0.0), reverse=True)[:15],
        }

    def _merge_chunk_nodes(self, nodes: Iterable) -> list:
        by_id = {}
        for node in nodes:
            if not getattr(node, "chunk_id", None):
                continue
            if node.chunk_id not in by_id:
                by_id[node.chunk_id] = node
        return list(by_id.values())

    def _base_article_scores(self, chunks: list[RetrievedChunk]) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for chunk in chunks:
            if chunk.article_id:
                scores[chunk.article_id] = max(scores[chunk.article_id], float(chunk.score or 0.0))
        return dict(scores)

    def _base_article_anchor_scores(
        self,
        chunks: list[RetrievedChunk],
        anchor_terms: list[str],
        anchor_disease_terms: list[str],
    ) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for chunk in chunks:
            if not chunk.article_id:
                continue
            text = " ".join([chunk.title or "", chunk.section or "", chunk.subsection or "", chunk.text[:900] or ""])
            score = self.graph_scorer.text_anchor_score(text, anchor_terms)
            score += 1.5 * self.graph_scorer.text_anchor_score(text, anchor_disease_terms)
            scores[chunk.article_id] = max(scores[chunk.article_id], score)
        return dict(scores)

    def _candidate_article_ids(
        self,
        chunks: list[RetrievedChunk],
        *,
        query: str,
        entities: list[RetrievedEntity],
        relations: list[RetrievedRelation],
        analysis: QueryAnalysis,
        anchor_terms: list[str],
        anchor_disease_terms: list[str],
        max_articles: int,
    ) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        slot_terms = [self._norm(item) for item in analysis.evidence_slot_keywords if item]

        for rank, chunk in enumerate(chunks, start=1):
            if not chunk.article_id:
                continue
            article_score = max(float(chunk.score or 0.0), 0.0) + 1.0 / (rank + 1)
            title_text = " ".join([chunk.title or "", chunk.section or "", chunk.text[:800] or ""])
            anchor_score = self.graph_scorer.text_anchor_score(title_text, anchor_terms)
            disease_score = self.graph_scorer.text_anchor_score(title_text, anchor_disease_terms)
            slot_score = self.graph_scorer.text_anchor_score(title_text, slot_terms)
            article_score += anchor_score * self.config.anchor_article_weight
            article_score += disease_score * self.config.anchor_disease_weight
            article_score += slot_score * self.config.section_weight
            if self.graph_scorer.text_anchor_score(chunk.title or "", anchor_terms) > 0:
                article_score += self.config.title_anchor_weight
            scores[chunk.article_id] += article_score

        return [article_id for article_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_articles]]

    def _score_candidate(
        self,
        *,
        node,
        query_terms: list[str],
        slot_terms: list[str],
        anchor_terms: list[str],
        anchor_disease_terms: list[str],
        graph_disease_terms: list[str],
        relation_source_urls: set[str],
    ) -> tuple[float, list[str], dict[str, float | bool | list[str]]]:
        text = " ".join([node.title or "", node.section or "", node.subsection or "", node.text or ""])
        haystack = self._norm(text)
        score = 0.0
        reasons: list[str] = []

        q_matches = [term for term in query_terms if term and term in haystack]
        if q_matches:
            inc = min(len(q_matches), 4) * self.config.query_term_weight
            score += inc
            reasons.append(f"query_phrases:{','.join(q_matches[:4])}")

        slot_matches = [term for term in slot_terms if term and term in haystack]
        if slot_matches:
            inc = min(len(slot_matches), 4) * self.config.slot_term_weight
            score += inc
            reasons.append(f"slot_terms:{','.join(slot_matches[:4])}")

        anchor_matches = [term for term in anchor_terms if term and term in haystack]
        if anchor_matches:
            inc = min(len(anchor_matches), 4) * self.config.anchor_term_weight
            score += inc
            reasons.append(f"anchor_terms:{','.join(anchor_matches[:4])}")

        anchor_disease_matches = [term for term in anchor_disease_terms if term and term in haystack]
        if anchor_disease_matches:
            inc = min(len(anchor_disease_matches), 3) * self.config.anchor_disease_weight
            score += inc
            reasons.append(f"anchor_disease:{','.join(anchor_disease_matches[:3])}")

        graph_disease_matches = [term for term in graph_disease_terms if term and term in haystack]
        if graph_disease_matches:
            inc = min(len(graph_disease_matches), 3) * self.config.graph_disease_weight
            score += inc
            reasons.append(f"graph_disease_weak:{','.join(graph_disease_matches[:3])}")

        section_norm = self._norm(" ".join([node.section or "", node.subsection or ""]))
        section_cues = [term for term in slot_terms if term and term in section_norm]
        if section_cues:
            score += self.config.section_weight
            reasons.append("section_slot_match")

        relation_source_hit = self._norm_url(node.source_url) in relation_source_urls
        if relation_source_hit:
            score += self.config.relation_source_weight
            reasons.append("relation_source_url_weak")

        max_query_specificity = max((self.graph_scorer.term_specificity(term) for term in q_matches), default=0.0)
        generic_only = bool(
            (q_matches or slot_matches or graph_disease_matches or relation_source_hit)
            and not (anchor_matches or anchor_disease_matches)
            and max_query_specificity < 0.55
        )
        breakdown = {
            "query_score": min(len(q_matches), 4) * self.config.query_term_weight,
            "slot_score": min(len(slot_matches), 4) * self.config.slot_term_weight,
            "anchor_score": min(len(anchor_matches), 4) * self.config.anchor_term_weight,
            "anchor_disease_score": min(len(anchor_disease_matches), 3) * self.config.anchor_disease_weight,
            "graph_disease_score": min(len(graph_disease_matches), 3) * self.config.graph_disease_weight,
            "section_score": self.config.section_weight if section_cues else 0.0,
            "relation_source_score": self.config.relation_source_weight if relation_source_hit else 0.0,
            "max_query_specificity": max_query_specificity,
            "generic_only": generic_only,
            "query_matches": q_matches[:6],
            "slot_matches": slot_matches[:6],
            "anchor_matches": anchor_matches[:6],
            "anchor_disease_matches": anchor_disease_matches[:6],
            "graph_disease_matches": graph_disease_matches[:6],
        }
        return score, reasons, breakdown

    def _local_score_scale(self, chunks: list[RetrievedChunk]) -> tuple[float, dict]:
        scores = sorted([float(chunk.score or 0.0) for chunk in chunks], reverse=True)
        if not scores:
            return self.config.score_scale_floor, {"reason": "empty"}
        top_score = scores[0]
        gaps = [abs(scores[idx] - scores[idx + 1]) for idx in range(len(scores) - 1)]
        positive_gaps = [gap for gap in gaps if gap > 1e-9]
        median_gap = statistics.median(positive_gaps) if positive_gaps else 0.0
        fallback = max(abs(top_score) * self.config.top_score_floor_ratio, self.config.score_scale_floor)
        score_scale = max(median_gap, fallback)
        return score_scale, {
            "scores_top": scores[:10],
            "positive_gaps": positive_gaps[:10],
            "median_gap": median_gap,
            "fallback": fallback,
            "score_scale": score_scale,
        }

    def _minmax_norm(self, value: float, min_value: float, max_value: float) -> float:
        if max_value <= min_value:
            return 0.5 if value > 0.0 else 0.0
        return max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))

    def _anchor_factor(self, article_anchor: float) -> float:
        if article_anchor >= self.config.min_anchor_for_full_delta:
            return 1.0
        ratio = max(0.0, min(1.0, article_anchor / max(self.config.min_anchor_for_full_delta, 1e-6)))
        return self.config.weak_article_multiplier_floor + (1.0 - self.config.weak_article_multiplier_floor) * ratio

    def _apply_source_diversity(
        self,
        chunks: list[RetrievedChunk],
        *,
        anchor_terms: list[str],
        anchor_disease_terms: list[str],
    ) -> list[RetrievedChunk]:
        sorted_chunks = sorted(chunks, key=lambda item: item.score, reverse=True)
        selected: list[RetrievedChunk] = []
        used_ids: set[str] = set()
        source_counts: dict[str, int] = defaultdict(int)

        for chunk in sorted_chunks:
            key = self._norm_url(chunk.source_url) or chunk.article_id
            if not key:
                selected.append(chunk)
                used_ids.add(chunk.chunk_id)
                continue
            anchor_score = self.graph_scorer.text_anchor_score(
                " ".join([chunk.title or "", chunk.text[:800] or ""]),
                [*anchor_terms, *anchor_disease_terms],
            )
            if source_counts[key] == 0 or anchor_score > 0.0 or not chunk.metadata.get("evidence_expansion"):
                selected.append(chunk)
                used_ids.add(chunk.chunk_id)
                source_counts[key] += 1

        for chunk in sorted_chunks:
            if chunk.chunk_id not in used_ids:
                selected.append(chunk)
                used_ids.add(chunk.chunk_id)
        return selected

    def _merge_chunks(self, chunks: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
        by_id: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            old = by_id.get(chunk.chunk_id)
            if old is None or chunk.score > old.score:
                by_id[chunk.chunk_id] = chunk
        return sorted(by_id.values(), key=lambda item: item.score, reverse=True)

    def _anchor_terms(self, query: str, entities: list[RetrievedEntity]) -> list[str]:
        values: list[str] = []
        values.extend(self.graph_scorer.anchor_terms(query, None))
        for entity in entities[:20]:
            name = self._norm(entity.name)
            if name and (name in self._norm(query) or self._norm(query) in name):
                values.append(name)
        return self._dedupe([term for term in values if self._is_useful_phrase(term)])[:24]

    def _anchor_disease_terms(self, query: str, entities: list[RetrievedEntity]) -> list[str]:
        q = self._norm(query)
        terms: list[str] = []
        for entity in entities:
            if entity.entity_type != "BENH_LY" or not entity.name:
                continue
            candidates = [entity.name, entity.normalized_name, *(entity.aliases or []), *(entity.surface_forms or [])]
            for cand in candidates:
                term = self._norm(cand)
                if term and self._is_useful_phrase(term) and (self._contains_phrase(q, term) or term in q):
                    terms.append(term)
        return self._dedupe(terms)[:12]

    def _important_terms(self, query: str) -> list[str]:
        profile = self.graph_scorer.build_query_anchor_profile(query, None, None)
        # Prefer discriminative phrases. Short unigrams remain weak query terms in
        # GraphQueryScorer and should not drive evidence expansion candidates.
        return self._dedupe([*profile.anchor_phrases, *profile.operator_cues])[:18]

    def _is_useful_phrase(self, phrase: str) -> bool:
        phrase = self._norm(phrase)
        toks = phrase.split()
        if not toks or len(phrase) < 5:
            return False
        if len(toks) == 1:
            return self.graph_scorer.term_specificity(phrase) >= 0.55
        if len(toks) > 5:
            return False
        return self.graph_scorer.term_specificity(phrase) >= 0.45

    def _disease_terms(self, entities: list[RetrievedEntity], relations: list[RetrievedRelation]) -> list[str]:
        terms: list[str] = []
        for entity in entities:
            if entity.entity_type == "BENH_LY" and entity.name:
                terms.append(self._norm(entity.name))
        for relation in relations:
            if relation.subject_type == "BENH_LY" and relation.subject_name:
                terms.append(self._norm(relation.subject_name))
            if relation.object_type == "BENH_LY" and relation.object_name:
                terms.append(self._norm(relation.object_name))
        return self._dedupe([term for term in terms if self._is_useful_phrase(term)])[:20]

    def _dedupe(self, items: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = self._norm(item)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    def _contains_phrase(self, text: str, phrase: str) -> bool:
        text = self._norm(text)
        phrase = self._norm(phrase)
        if not phrase:
            return False
        return re.search(rf"(^|\s){re.escape(phrase)}($|\s)", text) is not None

    def _norm_url(self, value: str) -> str:
        value = (value or "").strip().lower()
        value = value.replace("http://", "https://")
        return value.rstrip("/")

    def _norm(self, text: str) -> str:
        text = (text or "").strip().lower().replace("đ", "d")
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()