import re
import unicodedata

from app.infrastructure.graph_database import (
    GraphRepository,
    MedicalRelation,
)
from app.ingestion.extraction.common.extraction_models import (
    ExtractedEntity,
    ExtractedGraph,
    ExtractedRelation,
)
from app.ingestion.extraction.keyword.keyword_profiler import clean_retrieval_keywords
from app.ingestion.processing import ChunkDocument
from app.ingestion.resolution import (
    EntityResolver,
    ResolvedEntity,
    make_relation_id,
)
from app.retrieval.lightrag_keyword_profiles import KeywordProfileGenerator
from app.ingestion.policies.graph_write_policy import (
    entities_selected_for_graph_write,
    entity_key as shared_entity_key,
    is_keepable_anchor_entity,
)


def _normalize_key_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _entity_key(name: str, entity_type: str) -> tuple[str, str]:
    return shared_entity_key(name, entity_type)


def _is_keepable_anchor_entity(entity: ExtractedEntity) -> bool:
    return is_keepable_anchor_entity(entity)


def _entities_selected_for_write(extracted_graph: ExtractedGraph) -> tuple[list[ExtractedEntity], dict[str, int]]:
    return entities_selected_for_graph_write(extracted_graph)


class MedicalGraphBuilder:
    """
    Ghi entity/relation y khoa vào Neo4j sau khi LLM extraction + validation.

    Không ghi Article/Chunk.
    Article/Chunk vẫn do GraphBuilder trong graph_builder.py xử lý.
    """

    def __init__(
        self,
        graph_repo: GraphRepository,
        resolver: EntityResolver,
    ):
        self.graph_repo = graph_repo
        self.resolver = resolver
        # Deterministic LightRAG index-time keyword normalizer.
        # Relation keywords are global/theme keys; subject/object names are already stored separately.
        self.keyword_profiles = KeywordProfileGenerator(llm_client=None)

    def write_chunk_extraction(
        self,
        chunk: ChunkDocument,
        extracted_graph: ExtractedGraph,
    ) -> dict:
        """
        Ghi graph đã prune từ một chunk.

        Entity candidates có thể rộng để tăng recall, nhưng Neo4j chỉ nhận:
        - entity là endpoint của relation hợp lệ; hoặc
        - entity anchor có khả năng hữu ích độc lập.

        Return thống kê để print/debug.
        """
        if not extracted_graph.should_ingest_graph:
            return {
                "chunk_id": chunk.chunk_id,
                "entity_write_count": 0,
                "mention_write_count": 0,
                "relation_write_count": 0,
                "written_relation_ids": [],
                "skipped_graph_ingest": True,
                "skip_reason": extracted_graph.skip_reason,
            }

        resolved_by_key: dict[tuple[str, str], ResolvedEntity] = {}
        entities_to_write, prune_stats = _entities_selected_for_write(extracted_graph)

        entity_write_count = 0
        mention_write_count = 0

        for entity in entities_to_write:
            resolved = self.resolver.resolve(entity)

            # Dùng cả key gốc và normalized key để relation endpoint dễ tìm.
            resolved_by_key[(entity.name, entity.entity_type)] = resolved
            resolved_by_key[_entity_key(entity.name, entity.entity_type)] = resolved

            node = self.resolver.to_entity_node(
                resolved,
                mention_count=1,
                source_count=1,
            )

            was_written = self.graph_repo.upsert_entity(node)

            if was_written:
                entity_write_count += 1

                self.graph_repo.link_chunk_mentions_entity(
                    chunk_id=chunk.chunk_id,
                    entity_id=resolved.entity_id,
                    confidence=1.0,
                    evidence_text=entity.evidence_text,
                    section=chunk.section,
                )

                mention_write_count += 1

        relation_write_count = 0
        written_relation_ids: list[str] = []

        for relation in extracted_graph.relations:
            subject = self._resolve_relation_endpoint(
                relation=relation,
                is_subject=True,
                resolved_by_key=resolved_by_key,
                chunk=chunk,
            )

            object_ = self._resolve_relation_endpoint(
                relation=relation,
                is_subject=False,
                resolved_by_key=resolved_by_key,
                chunk=chunk,
            )

            if subject is None or object_ is None:
                continue

            relation_id = make_relation_id(
                subject_entity_id=subject.entity_id,
                relation_type=relation.relation_type,
                object_entity_id=object_.entity_id,
            )

            keyword_profile = self.keyword_profiles.generate_relation_profile(
                relation_type=relation.relation_type,
                subject_name=subject.name,
                subject_type=subject.entity_type,
                object_name=object_.name,
                object_type=object_.entity_type,
                description=relation.description,
                evidence_text=relation.evidence_text,
            )
            relation_keywords = clean_retrieval_keywords(relation.keywords or [], limit=8)
            if not relation_keywords:
                relation_keywords = clean_retrieval_keywords(keyword_profile.keywords, limit=8)

            medical_relation = MedicalRelation(
                relation_id=relation_id,
                subject_entity_id=subject.entity_id,
                object_entity_id=object_.entity_id,
                relation_type=relation.relation_type,
                description=relation.description,
                keywords=relation_keywords,
                evidence_text=relation.evidence_text,
                evidence_chunk_ids=[chunk.chunk_id],
                confidence=relation.confidence,
                section=chunk.section,
                source_url=chunk.source_url,
                metadata={
                    "subsection": chunk.subsection,
                    "title": chunk.title,
                    "original_llm_keywords": relation.keywords,
                    "keyword_policy": "article_llm_keywords_preferred_safe_fallback",
                },
            )

            ok = self.graph_repo.upsert_medical_relation(
                relation=medical_relation,
                subject_type=subject.entity_type,
                object_type=object_.entity_type,
            )

            if ok:
                relation_write_count += 1
                written_relation_ids.append(relation_id)

        return {
            "chunk_id": chunk.chunk_id,
            "entity_write_count": entity_write_count,
            "mention_write_count": mention_write_count,
            "relation_write_count": relation_write_count,
            "written_relation_ids": written_relation_ids,
            "skipped_graph_ingest": False,
            "skip_reason": None,
            **prune_stats,
        }

    def _resolve_relation_endpoint(
        self,
        relation: ExtractedRelation,
        is_subject: bool,
        resolved_by_key: dict[tuple[str, str], ResolvedEntity],
        chunk: ChunkDocument,
    ) -> ResolvedEntity | None:
        """
        Relation có subject/object.
        Ta cần biết subject/object đó map về canonical entity nào.

        Nếu LLM đã đưa subject/object vào entities list thì dùng lại.
        Nếu LLM quên entity nhưng relation có nhắc tới, tạo temp entity để resolve.
        """
        if is_subject:
            name = relation.subject
            entity_type = relation.subject_type
        else:
            name = relation.object
            entity_type = relation.object_type

        direct = resolved_by_key.get((name, entity_type)) or resolved_by_key.get(_entity_key(name, entity_type))
        if direct:
            return direct

        temp_entity = ExtractedEntity(
            name=name,
            entity_type=entity_type,
            aliases=[],
            surface_forms=[name],
            description="",
            evidence_text=relation.evidence_text,
            is_anchor_candidate=True,
            anchor_reason="relation_endpoint",
        )

        resolved = self.resolver.resolve(temp_entity)

        node = self.resolver.to_entity_node(
            resolved,
            mention_count=1,
            source_count=1,
        )

        was_written = self.graph_repo.upsert_entity(node)

        if not was_written:
            return None

        self.graph_repo.link_chunk_mentions_entity(
            chunk_id=chunk.chunk_id,
            entity_id=resolved.entity_id,
            confidence=1.0,
            evidence_text=relation.evidence_text,
            section=chunk.section,
        )

        resolved_by_key[(name, entity_type)] = resolved
        return resolved
