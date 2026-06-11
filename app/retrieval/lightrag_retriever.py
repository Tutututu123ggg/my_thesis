from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
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

    # Retrieval-time guardrails. Genericness is learned from graph statistics,
    # not from a hand-written dictionary: entities/relations mentioned in too
    # many chunks are downweighted by an IDF-like node specificity signal.
    apply_node_specificity: bool = True
    node_specificity_min: float = 1e-6
    generic_chunk_ratio_threshold: float = 0.12
    constrain_relation_types_by_intent: bool = True
    constrain_relation_endpoints_to_entity_seeds: bool = True

    # Medical diagnosis queries should rank BENH_LY candidates first, then
    # map the best disease candidates to evidence chunks. This prevents
    # generic trigger nodes such as "dị ứng/phấn hoa/bụi" from dominating
    # the final context before a disease candidate is established.
    disease_candidate_first_for_symptom_queries: bool = True
    disease_candidate_top_k: int = 8
    disease_relation_weights: dict[str, float] | None = None


class LightRAGRetriever:
    """
    LightRAG-style dual-level retriever.

    Mapping theo paper:
    - QueryKeywordExtractor sinh low/high-level keywords.
    - low-level keywords -> Qdrant entity vectors.
    - high-level keywords -> Qdrant relation vectors.
    - entity/relation seeds -> Neo4j graph expansion 1-hop + evidence chunks.

    This implementation adds retrieval-time constraints so high-level relation
    search does not drift away from the low-level entity seeds.
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
        raw_keywords = self.keyword_extractor.extract(query)
        # Keyword genericness cannot be solved with a static dictionary. Keep
        # deduped keyword text here; dynamic genericness is applied after entity
        # linking using chunk mention frequency from Neo4j.
        keywords = self._clean_keywords(raw_keywords)
        intent = self._infer_intent(query, keywords)
        intent_relation_types = self._infer_relation_types(query, keywords, intent=intent)
        low_query = keywords.low_query_text() or query
        high_query = keywords.high_query_text() or query
        if intent == "symptom_diagnosis":
            # Do not let bad fallback high-level keywords such as "phân loại"
            # drive relation search for patient symptom questions. In LightRAG
            # terms, this keeps the global/relation query aligned with the
            # user's diagnostic intent instead of drifting to taxonomy edges.
            high_query = "triệu chứng; biểu hiện; yếu tố liên quan; dị ứng"

        entity_hits = self.vector_retriever.search_entities(
            low_query,
            top_k=self.config.entity_top_k,
        )
        entity_hits, entity_specificity_debug = self._apply_entity_specificity(
            self._dedupe_entities(entity_hits)
        )
        entity_seed_ids = [e.entity_id for e in entity_hits]

        relation_hits = self.vector_retriever.search_relations(
            high_query,
            top_k=self.config.relation_top_k,
        )
        relation_hits = self._filter_relations(
            relation_hits,
            allowed_relation_types=intent_relation_types,
            seed_entity_ids=set(entity_seed_ids),
        )
        relation_hits, relation_specificity_debug = self._apply_relation_endpoint_specificity(relation_hits)
        relation_seed_ids = [r.relation_id for r in relation_hits]

        graph_from_entities = self.graph_retriever.retrieve_from_entity_ids(
            query=query,
            entity_ids=entity_seed_ids,
            relation_types=intent_relation_types,
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

        base_relations = self._merge_relations(
            relation_hits + graph_from_entities.relations + graph_from_relations.relations
        )
        disease_candidates = self._rank_disease_candidates(
            relations=base_relations,
            seed_entity_ids=set(entity_seed_ids),
            intent=intent,
        )
        disease_ids = [entity_id for entity_id, _ in disease_candidates[: self.config.disease_candidate_top_k]]

        graph_from_diseases = RetrievalResult(query=query, method="graph_disease")
        if disease_ids and (
            intent == "symptom_diagnosis"
            and self.config.disease_candidate_first_for_symptom_queries
        ):
            graph_from_diseases = self.graph_retriever.retrieve_from_entity_ids(
                query=query,
                entity_ids=disease_ids,
                relation_types=intent_relation_types,
                max_relations=self.config.graph_max_relations,
                max_chunks=self.config.graph_max_chunks,
                expand_synonyms=True,
            )

        entities = self._merge_entities(
            entity_hits
            + graph_from_entities.entities
            + graph_from_relations.entities
            + graph_from_diseases.entities
        )
        relations = self._merge_relations(base_relations + graph_from_diseases.relations)
        chunks = self._merge_chunks(
            graph_from_entities.chunks
            + graph_from_relations.chunks
            + graph_from_diseases.chunks
        )
        if intent == "symptom_diagnosis" and disease_candidates:
            chunks = self._rerank_chunks_by_disease_candidates(chunks, relations, disease_candidates)

        return RetrievalResult(
            query=query,
            method="lightrag",
            keywords=keywords,
            entities=entities[: self.config.max_entities],
            relations=relations[: self.config.max_relations],
            chunks=chunks[: self.config.max_chunks],
            debug={
                "raw_low_keywords": raw_keywords.low_level_keywords,
                "raw_high_keywords": raw_keywords.high_level_keywords,
                "low_query": low_query,
                "high_query": high_query,
                "intent": intent,
                "intent_relation_types": intent_relation_types,
                "disease_candidate_first": bool(disease_candidates),
                "top_disease_candidates": [
                    {"entity_id": entity_id, "score": score}
                    for entity_id, score in disease_candidates[: self.config.disease_candidate_top_k]
                ],
                "entity_seed_count": len(entity_hits),
                "relation_seed_count": len(relation_hits),
                "entity_specificity": entity_specificity_debug,
                "relation_endpoint_specificity": relation_specificity_debug,
                "genericness_strategy": "dynamic_chunk_document_frequency",
                "synonym_expand": self.config.synonym_expand,
                "endpoint_constraint": self.config.constrain_relation_endpoints_to_entity_seeds,
            },
        )

    def _clean_keywords(self, keywords: QueryKeywords) -> QueryKeywords:
        return QueryKeywords(
            low_level_keywords=self._dedupe_keyword_text(keywords.low_level_keywords),
            high_level_keywords=self._dedupe_keyword_text(keywords.high_level_keywords),
        )

    def _dedupe_keyword_text(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = (value or "").strip()
            key = self._norm(cleaned)
            if not key or len(key) <= 1 or key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
        return result

    def _dedupe_entities(self, entities: list[RetrievedEntity]) -> list[RetrievedEntity]:
        result: list[RetrievedEntity] = []
        seen: set[str] = set()
        for entity in entities:
            if not entity.entity_id or entity.entity_id in seen:
                continue
            seen.add(entity.entity_id)
            result.append(entity)
        return result

    def _apply_entity_specificity(
        self,
        entities: list[RetrievedEntity],
    ) -> tuple[list[RetrievedEntity], dict[str, dict[str, float | int | bool]]]:
        if not self.config.apply_node_specificity or not entities:
            return entities, {}

        entity_ids = [entity.entity_id for entity in entities]
        chunk_counts = self.graph_retriever.graph_repo.get_entity_chunk_counts(entity_ids)
        total_chunks = max(int(self.graph_retriever.graph_repo.get_graph_stats().get("chunk_count", 0) or 0), 1)

        debug: dict[str, dict[str, float | int | bool]] = {}
        adjusted: list[RetrievedEntity] = []

        for entity in entities:
            chunk_count = max(int(chunk_counts.get(entity.entity_id, 1) or 1), 1)
            specificity = max(1.0 / float(chunk_count), self.config.node_specificity_min)
            chunk_ratio = float(chunk_count) / float(total_chunks)
            original_score = float(entity.score or 0.0)
            entity.score = original_score * specificity
            entity.metadata = {
                **dict(entity.metadata or {}),
                "original_score": original_score,
                "chunk_document_frequency": chunk_count,
                "chunk_document_ratio": chunk_ratio,
                "node_specificity": specificity,
                "frequency_generic": chunk_ratio >= self.config.generic_chunk_ratio_threshold,
            }
            debug[entity.entity_id] = {
                "chunk_count": chunk_count,
                "chunk_ratio": chunk_ratio,
                "specificity": specificity,
                "original_score": original_score,
                "adjusted_score": entity.score,
                "frequency_generic": chunk_ratio >= self.config.generic_chunk_ratio_threshold,
            }
            adjusted.append(entity)

        return sorted(adjusted, key=lambda item: item.score, reverse=True), debug

    def _apply_relation_endpoint_specificity(
        self,
        relations: list[RetrievedRelation],
    ) -> tuple[list[RetrievedRelation], dict[str, dict[str, float | int]]]:
        if not self.config.apply_node_specificity or not relations:
            return relations, {}

        endpoint_ids = list(
            dict.fromkeys(
                entity_id
                for relation in relations
                for entity_id in [relation.subject_entity_id, relation.object_entity_id]
                if entity_id
            )
        )
        chunk_counts = self.graph_retriever.graph_repo.get_entity_chunk_counts(endpoint_ids)

        def specificity(entity_id: str) -> float:
            count = max(int(chunk_counts.get(entity_id, 1) or 1), 1)
            return max(1.0 / float(count), self.config.node_specificity_min)

        debug: dict[str, dict[str, float | int]] = {}
        adjusted: list[RetrievedRelation] = []
        for relation in relations:
            subject_spec = specificity(relation.subject_entity_id)
            object_spec = specificity(relation.object_entity_id)
            endpoint_spec = max(subject_spec, object_spec)
            original_score = float(relation.score or 0.0)
            relation.score = original_score * endpoint_spec
            relation.metadata = {
                **dict(relation.metadata or {}),
                "original_score": original_score,
                "subject_node_specificity": subject_spec,
                "object_node_specificity": object_spec,
                "endpoint_node_specificity": endpoint_spec,
            }
            debug[relation.relation_id] = {
                "subject_chunk_count": int(chunk_counts.get(relation.subject_entity_id, 1) or 1),
                "object_chunk_count": int(chunk_counts.get(relation.object_entity_id, 1) or 1),
                "endpoint_specificity": endpoint_spec,
                "original_score": original_score,
                "adjusted_score": relation.score,
            }
            adjusted.append(relation)

        return sorted(adjusted, key=lambda item: item.score, reverse=True), debug

    def _filter_relations(
        self,
        relations: list[RetrievedRelation],
        allowed_relation_types: list[str] | None,
        seed_entity_ids: set[str],
    ) -> list[RetrievedRelation]:
        type_filtered = [
            r for r in relations
            if not allowed_relation_types or r.relation_type in allowed_relation_types
        ]

        if not self.config.constrain_relation_endpoints_to_entity_seeds or not seed_entity_ids:
            return type_filtered

        endpoint_filtered = [
            r for r in type_filtered
            if r.subject_entity_id in seed_entity_ids or r.object_entity_id in seed_entity_ids
        ]

        # Avoid catastrophic empty results when entity vector linking is weak.
        return endpoint_filtered or type_filtered

    def _infer_intent(self, query: str, keywords: QueryKeywords) -> str:
        text = self._norm(" ".join([query] + keywords.low_level_keywords + keywords.high_level_keywords))
        if any(
            cue in text
            for cue in [
                "co the la",
                "benh gi",
                "van de gi",
                "nen nghi",
                "nen kham",
                "khong biet",
                "goi y",
                "toi bi",
                "con toi",
            ]
        ):
            return "symptom_diagnosis"
        if any(cue in text for cue in ["phan biet", "de nham", "nham voi"]):
            return "differential"
        if any(cue in text for cue in ["bien chung", "dan toi", "gay ra", "nguy hiem"]):
            return "complication"
        if any(cue in text for cue in ["dang", "the", "loai", "phan loai"]):
            return "taxonomy"
        if any(cue in text for cue in ["con goi", "ten khac", "dong nghia", "tieng anh", "aka"]):
            return "synonym"
        return "general_fact"

    def _infer_relation_types(self, query: str, keywords: QueryKeywords, intent: str | None = None) -> list[str] | None:
        if not self.config.constrain_relation_types_by_intent:
            return None

        text = self._norm(" ".join([query] + keywords.low_level_keywords + keywords.high_level_keywords))
        relation_types: list[str] = []

        def add(*values: str) -> None:
            for value in values:
                if value not in relation_types:
                    relation_types.append(value)

        if intent == "symptom_diagnosis":
            # For patient symptom questions, keep LightRAG global retrieval on
            # disease-symptom/trigger edges. Do not allow taxonomy/treatment
            # relation types from noisy high-level keywords to dominate.
            add("CO_BIEU_HIEN", "LIEN_QUAN_YEU_TO", "ANH_HUONG_DEN", "CHAN_DOAN_PHAN_BIET_VOI")
            return relation_types

        if any(cue in text for cue in ["trieu chung", "bieu hien", "dau hieu", "ngua", "kho tho", "kho khe", "ho"]):
            add("CO_BIEU_HIEN", "ANH_HUONG_DEN")
        if any(cue in text for cue in ["nguyen nhan", "yeu to", "lien quan", "di ung", "bui", "phan hoa", "long thu", "mat bui"]):
            add("LIEN_QUAN_YEU_TO")
        if any(cue in text for cue in ["bien chung", "dan toi", "gay ra", "nguy hiem"]):
            add("GAY_BIEN_CHUNG")
        if any(cue in text for cue in ["phan biet", "de nham", "nham voi", "chan doan"]):
            add("CHAN_DOAN_PHAN_BIET_VOI", "CHAN_DOAN_BANG")
        if any(cue in text for cue in ["dieu tri", "chua", "thuoc", "boi", "uong"]):
            add("DIEU_TRI_BANG")
        if any(cue in text for cue in ["phong ngua", "phong tranh"]):
            add("PHONG_NGUA_BANG")
        if any(cue in text for cue in ["tranh", "kieng"]):
            add("CAN_TRANH")
        if any(cue in text for cue in ["dang", "the", "loai", "phan loai"]):
            add("LA_DANG_CUA")
        if any(cue in text for cue in ["con goi", "ten khac", "dong nghia", "tieng anh", "aka"]):
            add("DONG_NGHIA_VOI")

        return relation_types or None


    def _relation_weights(self) -> dict[str, float]:
        return self.config.disease_relation_weights or {
            "CO_BIEU_HIEN": 2.4,
            "LIEN_QUAN_YEU_TO": 1.2,
            "ANH_HUONG_DEN": 0.8,
            "CHAN_DOAN_PHAN_BIET_VOI": 0.6,
        }

    def _rank_disease_candidates(
        self,
        relations: list[RetrievedRelation],
        seed_entity_ids: set[str],
        intent: str,
    ) -> list[tuple[str, float]]:
        if intent != "symptom_diagnosis" or not relations:
            return []

        weights = self._relation_weights()
        scores: dict[str, float] = defaultdict(float)
        evidence_counts: dict[str, int] = defaultdict(int)

        for relation in relations:
            weight = weights.get(relation.relation_type)
            if weight is None:
                continue

            disease_id = ""
            matched_seed = False

            if relation.subject_type == "BENH_LY":
                disease_id = relation.subject_entity_id
                matched_seed = relation.object_entity_id in seed_entity_ids
            elif relation.object_type == "BENH_LY":
                disease_id = relation.object_entity_id
                matched_seed = relation.subject_entity_id in seed_entity_ids

            if not disease_id:
                continue

            base = max(float(relation.score or 0.0), 0.05)
            # Relations that explicitly touch a symptom/trigger seed should be
            # much stronger than broad disease-neighborhood relations.
            seed_multiplier = 1.8 if matched_seed else 0.65
            scores[disease_id] += base * weight * seed_multiplier
            evidence_counts[disease_id] += 1

        # Reward diseases supported by multiple matched relations. This helps
        # asthma beat allergic rhinitis when the query contains several lower
        # respiratory symptoms, not just generic allergens.
        for disease_id, count in evidence_counts.items():
            scores[disease_id] *= 1.0 + min(count, 4) * 0.12

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def _rerank_chunks_by_disease_candidates(
        self,
        chunks: list[RetrievedChunk],
        relations: list[RetrievedRelation],
        disease_candidates: list[tuple[str, float]],
    ) -> list[RetrievedChunk]:
        if not chunks or not disease_candidates:
            return chunks

        disease_rank_score = {entity_id: score for entity_id, score in disease_candidates}
        relation_chunk_boost: dict[str, float] = defaultdict(float)
        disease_name_by_id: dict[str, str] = {}

        for relation in relations:
            disease_id = ""
            disease_name = ""
            if relation.subject_entity_id in disease_rank_score and relation.subject_type == "BENH_LY":
                disease_id = relation.subject_entity_id
                disease_name = relation.subject_name
            elif relation.object_entity_id in disease_rank_score and relation.object_type == "BENH_LY":
                disease_id = relation.object_entity_id
                disease_name = relation.object_name
            if not disease_id:
                continue
            disease_name_by_id[disease_id] = disease_name
            boost = disease_rank_score[disease_id]
            for chunk_id in relation.evidence_chunk_ids:
                relation_chunk_boost[chunk_id] += boost

        ranked: list[RetrievedChunk] = []
        for chunk in chunks:
            text_key = self._norm(" ".join([chunk.title, chunk.section, chunk.text[:800]]))
            name_boost = 0.0
            matched_diseases: list[str] = []
            for disease_id, disease_score in disease_rank_score.items():
                name = disease_name_by_id.get(disease_id, "")
                if name and self._norm(name) in text_key:
                    name_boost += disease_score * 0.35
                    matched_diseases.append(name)

            original = float(chunk.score or 0.0)
            boost = relation_chunk_boost.get(chunk.chunk_id, 0.0) + name_boost
            chunk.score = original + boost
            chunk.metadata = {
                **dict(chunk.metadata or {}),
                "lightrag_original_chunk_score": original,
                "lightrag_disease_candidate_boost": boost,
                "lightrag_matched_diseases": matched_diseases,
            }
            ranked.append(chunk)

        return sorted(ranked, key=lambda item: item.score, reverse=True)


    def _norm(self, text: str) -> str:
        text = (text or "").strip().lower().replace("đ", "d")
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

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
