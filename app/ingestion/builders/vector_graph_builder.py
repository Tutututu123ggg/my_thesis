from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.vector_database import (
    ChunkVectorPayload,
    EntityVectorPayload,
    RelationVectorPayload,
    VectorPoint,
    VectorRepository,
    make_qdrant_point_id,
)
from app.ingestion.extraction.common.extraction_models import (
    ExtractedEntity,
    ExtractedGraph,
    ExtractedRelation,
)
from app.ingestion.processing import ChunkDocument
from app.ingestion.extraction.keyword.keyword_profiler import (
    clean_retrieval_keywords,
    deterministic_relation_keywords,
)
from app.ingestion.resolution import EntityResolver, ResolvedEntity, make_relation_id
from app.retrieval.lightrag_keyword_profiles import KeywordProfileGenerator
from app.ingestion.policies.graph_write_policy import graph_with_selected_entities


class VectorGraphBuilder:
    """
    Ghi vectors vào Qdrant cho Hybrid RAG.

    Collections:
    - medical_chunks: Vector RAG chunk retrieval.
    - medical_entities: LightRAG-style entity vector lookup.
    - medical_relations: LightRAG-style relation/global retrieval.

    Lưu ý entity resolution:
    - Builder này dùng cùng EntityResolver với graph builder để sinh entity_id ổn định.
    - EntityResolver policy cuối: hard merge chỉ theo normalized entity.name.
    - aliases/surface_forms có thể lưu payload, nhưng không được dùng để hard merge.
    """

    def __init__(
        self,
        vector_repo: VectorRepository,
        embedder: EmbeddingService,
        resolver: EntityResolver | None = None,
        batch_size: int = 32,
    ):
        self.vector_repo = vector_repo
        self.embedder = embedder
        self.resolver = resolver
        self.batch_size = batch_size
        self.keyword_profiles = KeywordProfileGenerator(llm_client=None)

    def setup_collections(self, recreate: bool = False) -> None:
        self.vector_repo.setup_collections(recreate=recreate)

    # =====================
    # Chunk vectors: standard Vector RAG
    # =====================

    def upsert_chunks(self, chunks: list[ChunkDocument]) -> int:
        if not chunks:
            return 0

        texts = [self._chunk_vector_text(chunk) for chunk in chunks]
        vectors = self.embedder.embed_texts(texts, batch_size=self.batch_size)

        points = [
            VectorPoint(
                point_id=make_qdrant_point_id(f"chunk::{chunk.chunk_id}"),
                vector=vector,
                payload=ChunkVectorPayload(
                    chunk_id=chunk.chunk_id,
                    article_id=chunk.article_id,
                    source_url=chunk.source_url,
                    title=chunk.title,
                    section=chunk.section,
                    subsection=chunk.subsection,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    text=chunk.text,
                    contextualized_text=chunk.contextualized_text,
                    metadata=chunk.metadata,
                ).model_dump(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        self.vector_repo.upsert_chunk_vectors(points)
        return len(points)

    def upsert_naive_chunks(self, chunks: list[ChunkDocument]) -> int:
        """Write the true vector_naive baseline: embeddings from raw chunk.text only."""
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedder.embed_texts(texts, batch_size=self.batch_size)

        points = [
            VectorPoint(
                point_id=make_qdrant_point_id(f"naive_chunk::{chunk.chunk_id}"),
                vector=vector,
                payload=ChunkVectorPayload(
                    chunk_id=chunk.chunk_id,
                    article_id=chunk.article_id,
                    source_url=chunk.source_url,
                    title=chunk.title,
                    section=chunk.section,
                    subsection=chunk.subsection,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    text=chunk.text,
                    contextualized_text=chunk.text,
                    metadata={
                        **dict(chunk.metadata or {}),
                        "vector_text_mode": "raw_chunk_text",
                    },
                ).model_dump(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        self.vector_repo.upsert_naive_chunk_vectors(points)
        return len(points)

    # =====================
    # Entity/relation vectors: LightRAG-style vector lookup
    # =====================

    def upsert_chunk_extraction(
        self,
        chunk: ChunkDocument,
        extracted_graph: ExtractedGraph,
        allowed_relation_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> dict:
        """
        Ghi entity/relation vectors sinh ra từ một chunk extraction.

        Hàm này không ghi Neo4j. Nếu cần ghi cả graph + vector, hãy gọi
        MedicalGraphBuilder.write_chunk_extraction() trước, rồi gọi hàm này.

        allowed_relation_ids dùng để giữ Qdrant medical_relations đồng bộ với Neo4j:
        chỉ vector hóa relation đã được Neo4j write policy chấp nhận.
        """
        if not extracted_graph.should_ingest_graph:
            return {
                "chunk_id": chunk.chunk_id,
                "entity_vector_count": 0,
                "relation_vector_count": 0,
                "skipped_graph_vector_ingest": True,
                "skip_reason": extracted_graph.skip_reason,
            }

        # Keep Qdrant medical_entities consistent with Neo4j: only vectorize
        # entities selected by the same graph-write policy. Relation endpoints
        # that are missing from the entity list are still resolved below for
        # relation vector payloads, but loose orphan candidates are not written.
        selected_graph = graph_with_selected_entities(extracted_graph)
        resolved_by_key = self._resolve_entities(selected_graph.entities)

        entity_points = self._build_entity_points(resolved_by_key.values())
        relation_points = self._build_relation_points(
            chunk=chunk,
            relations=selected_graph.relations,
            resolved_by_key=resolved_by_key,
            allowed_relation_ids=allowed_relation_ids,
        )

        if entity_points:
            self.vector_repo.upsert_entity_vectors(entity_points)

        if relation_points:
            self.vector_repo.upsert_relation_vectors(relation_points)

        return {
            "chunk_id": chunk.chunk_id,
            "entity_vector_count": len(entity_points),
            "relation_vector_count": len(relation_points),
            "skipped_graph_vector_ingest": False,
            "skip_reason": None,
        }

    def _resolve_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> dict[tuple[str, str], ResolvedEntity]:
        resolved_by_key: dict[tuple[str, str], ResolvedEntity] = {}

        for entity in entities:
            resolved = self._resolve_entity(entity)
            resolved_by_key[(entity.name, entity.entity_type)] = resolved

        return resolved_by_key

    def _resolve_entity(self, entity: ExtractedEntity) -> ResolvedEntity:
        if self.resolver is not None:
            return self.resolver.resolve(entity)

        # Fallback chỉ dùng khi test Qdrant độc lập, không có Neo4j resolver.
        from app.ingestion.resolution import clean_surface_text, make_entity_id, normalize_text

        name = clean_surface_text(entity.name)
        return ResolvedEntity(
            entity_id=make_entity_id(name, entity.entity_type),
            name=name,
            normalized_name=normalize_text(name),
            entity_type=entity.entity_type,
            aliases=entity.aliases,
            surface_forms=entity.surface_forms or [name],
            description=entity.description.strip(),
            merge_method="vector_fallback_exact_name",
        )

    def _build_entity_points(
        self,
        resolved_entities: list[ResolvedEntity] | tuple[ResolvedEntity, ...] | object,
    ) -> list[VectorPoint]:
        entities = list(resolved_entities)
        if not entities:
            return []

        texts = [self._entity_vector_text(entity) for entity in entities]
        vectors = self.embedder.embed_texts(texts, batch_size=self.batch_size)

        return [
            VectorPoint(
                point_id=make_qdrant_point_id(f"entity::{entity.entity_id}"),
                vector=vector,
                payload=EntityVectorPayload(
                    entity_id=entity.entity_id,
                    name=entity.name,
                    normalized_name=entity.normalized_name,
                    entity_type=entity.entity_type,
                    aliases=entity.aliases,
                    surface_forms=entity.surface_forms,
                    description=entity.description,
                    profile_text=self._entity_vector_text(entity),
                    local_keys=self.keyword_profiles.fallback_entity_profile(
                        name=entity.name,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        surface_forms=entity.surface_forms,
                    ).local_keys,
                    global_keys=self.keyword_profiles.fallback_entity_profile(
                        name=entity.name,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        surface_forms=entity.surface_forms,
                    ).global_keys,
                ).model_dump(),
            )
            for entity, vector in zip(entities, vectors, strict=True)
        ]

    def _build_relation_points(
        self,
        chunk: ChunkDocument,
        relations: list[ExtractedRelation],
        resolved_by_key: dict[tuple[str, str], ResolvedEntity],
        allowed_relation_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list[VectorPoint]:
        relation_payloads: list[RelationVectorPayload] = []
        allowed_ids = set(allowed_relation_ids) if allowed_relation_ids is not None else None

        for relation in relations:
            subject = self._get_or_resolve_endpoint(
                name=relation.subject,
                entity_type=relation.subject_type,
                evidence_text=relation.evidence_text,
                resolved_by_key=resolved_by_key,
            )
            object_ = self._get_or_resolve_endpoint(
                name=relation.object,
                entity_type=relation.object_type,
                evidence_text=relation.evidence_text,
                resolved_by_key=resolved_by_key,
            )

            relation_id = make_relation_id(
                subject_entity_id=subject.entity_id,
                relation_type=relation.relation_type,
                object_entity_id=object_.entity_id,
            )

            if allowed_ids is not None and relation_id not in allowed_ids:
                continue

            relation_keywords = clean_retrieval_keywords(relation.keywords, limit=8)
            if not relation_keywords:
                relation_keywords = deterministic_relation_keywords(relation)

            relation_payloads.append(
                RelationVectorPayload(
                    relation_id=relation_id,
                    relation_type=relation.relation_type,
                    subject_entity_id=subject.entity_id,
                    subject_name=subject.name,
                    subject_type=subject.entity_type,
                    object_entity_id=object_.entity_id,
                    object_name=object_.name,
                    object_type=object_.entity_type,
                    description=relation.description,
                    keywords=relation_keywords,
                    evidence_text=relation.evidence_text,
                    evidence_chunk_ids=[chunk.chunk_id],
                    chunk_id=chunk.chunk_id,
                    section=chunk.section,
                    source_url=chunk.source_url,
                )
            )

        if not relation_payloads:
            return []

        texts = [self._relation_vector_text(payload) for payload in relation_payloads]
        vectors = self.embedder.embed_texts(texts, batch_size=self.batch_size)

        return [
            VectorPoint(
                point_id=make_qdrant_point_id(f"relation::{payload.relation_id}::{payload.chunk_id}"),
                vector=vector,
                payload=payload.model_dump(),
            )
            for payload, vector in zip(relation_payloads, vectors, strict=True)
        ]

    def _get_or_resolve_endpoint(
        self,
        name: str,
        entity_type: str,
        evidence_text: str,
        resolved_by_key: dict[tuple[str, str], ResolvedEntity],
    ) -> ResolvedEntity:
        key = (name, entity_type)
        direct = resolved_by_key.get(key)
        if direct:
            return direct

        temp_entity = ExtractedEntity(
            name=name,
            entity_type=entity_type,
            aliases=[],
            surface_forms=[name],
            description="",
            evidence_text=evidence_text,
        )
        resolved = self._resolve_entity(temp_entity)
        resolved_by_key[key] = resolved
        return resolved

    # =====================
    # Text builders
    # =====================

    def _chunk_vector_text(self, chunk: ChunkDocument) -> str:
        return chunk.contextualized_text or chunk.text

    def _entity_vector_text(self, entity: ResolvedEntity) -> str:
        profile = self.keyword_profiles.fallback_entity_profile(
            name=entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
            surface_forms=entity.surface_forms,
        )
        parts = [profile.profile_text]

        if entity.surface_forms:
            parts.append("Cách gọi trong văn bản: " + "; ".join(entity.surface_forms[:8]))

        # aliases chỉ để enrich vector text/payload, không dùng hard merge.
        if entity.aliases:
            parts.append("Aliases metadata: " + "; ".join(entity.aliases[:8]))

        return "\n".join([p for p in parts if p])

    def _relation_vector_text(self, relation: RelationVectorPayload) -> str:
        parts = [
            f"Quan hệ: {relation.relation_type}",
            f"Subject: {relation.subject_name}",
            f"Object: {relation.object_name}",
        ]

        if relation.keywords:
            parts.append("Keywords: " + "; ".join(relation.keywords))

        if relation.description:
            parts.append(f"Mô tả: {relation.description}")

        if relation.evidence_text:
            parts.append(f"Bằng chứng: {relation.evidence_text}")

        return "\n".join(parts)
