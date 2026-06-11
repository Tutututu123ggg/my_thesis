from app.infrastructure.graph_database import (
    GraphRepository,
    MedicalRelation,
)
from app.ingestion.extraction import (
    ExtractedEntity,
    ExtractedGraph,
    ExtractedRelation,
)
from app.ingestion.processing import ChunkDocument
from app.ingestion.resolution import (
    EntityResolver,
    ResolvedEntity,
    make_relation_id,
)


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

    def write_chunk_extraction(
        self,
        chunk: ChunkDocument,
        extracted_graph: ExtractedGraph,
    ) -> dict:
        """
        Ghi toàn bộ entity/relation được extract từ một chunk.

        Return thống kê để print/debug.
        """
        resolved_by_key: dict[tuple[str, str], ResolvedEntity] = {}

        entity_write_count = 0
        mention_write_count = 0

        for entity in extracted_graph.entities:
            resolved = self.resolver.resolve(entity)

            # Dùng cả key gốc và key normalized để relation endpoint dễ tìm.
            resolved_by_key[(entity.name, entity.entity_type)] = resolved

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

            medical_relation = MedicalRelation(
                relation_id=relation_id,
                subject_entity_id=subject.entity_id,
                object_entity_id=object_.entity_id,
                relation_type=relation.relation_type,
                description=relation.description,
                keywords=relation.keywords,
                evidence_text=relation.evidence_text,
                evidence_chunk_ids=[chunk.chunk_id],
                confidence=relation.confidence,
                section=chunk.section,
                source_url=chunk.source_url,
                metadata={
                    "subsection": chunk.subsection,
                    "title": chunk.title,
                },
            )

            ok = self.graph_repo.upsert_medical_relation(
                relation=medical_relation,
                subject_type=subject.entity_type,
                object_type=object_.entity_type,
            )

            if ok:
                relation_write_count += 1

        return {
            "chunk_id": chunk.chunk_id,
            "entity_write_count": entity_write_count,
            "mention_write_count": mention_write_count,
            "relation_write_count": relation_write_count,
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

        direct = resolved_by_key.get((name, entity_type))
        if direct:
            return direct

        temp_entity = ExtractedEntity(
            name=name,
            entity_type=entity_type,
            aliases=[],
            surface_forms=[name],
            description="",
            evidence_text=relation.evidence_text,
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