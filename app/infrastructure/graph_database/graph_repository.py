import json
from typing import Any

from app.domain.medical_schema import (
    is_valid_entity_type,
    is_valid_relation_schema,
    looks_like_blocked_entity,
)
from app.infrastructure.graph_database import graph_queries as q
from app.infrastructure.graph_database.graph_models import (
    ArticleNode,
    ChunkNode,
    EntityNode,
    MedicalRelation,
    MedicalRelationView,
    EntityAdjacencyEdge,
    ChunkEntityLink,
    EntityContextBundle,
)
from app.infrastructure.graph_database.neo4j_client import Neo4jClient


class GraphRepository:
    """
    Cổng duy nhất để các module khác giao tiếp với Neo4j.

    Rule:
    - Không viết Cypher ngoài infrastructure/database.
    - Không trả raw Neo4j Record ra ngoài.
    - Không lưu dict trực tiếp vào Neo4j, luôn serialize thành metadata_json.
    """

    def __init__(self, client: Neo4jClient):
        self.client = client

    # =====================
    # Setup
    # =====================

    def setup_schema(self) -> None:
        with self.client.session() as session:
            for query in q.CREATE_CONSTRAINTS:
                session.run(query)

            for query in q.CREATE_INDEXES:
                session.run(query)

    def delete_all_data(self) -> None:
        with self.client.session() as session:
            session.run(q.DELETE_ALL_DATA)

    def health_check(self) -> bool:
        return self.client.health_check()

    # =====================
    # Write methods
    # =====================

    def upsert_article(self, article: ArticleNode) -> None:
        params = article.model_dump()
        params["metadata_json"] = self._to_json(article.metadata)
        params.pop("metadata", None)

        params["description"] = params.get("description") or ""
        params["author"] = params.get("author") or ""
        params["published_at"] = params.get("published_at") or ""
        params["updated_at"] = params.get("updated_at") or ""
        params["category"] = params.get("category") or ""
        params["crawled_at"] = params.get("crawled_at") or ""

        with self.client.session() as session:
            session.run(q.UPSERT_ARTICLE, params)

    def upsert_chunk(self, chunk: ChunkNode) -> None:
        params = chunk.model_dump()
        params["metadata_json"] = self._to_json(chunk.metadata)
        params.pop("metadata", None)

        params["section"] = params.get("section") or ""
        params["subsection"] = params.get("subsection") or ""
        params["contextualized_text"] = params.get("contextualized_text") or chunk.text
        params["token_count"] = params.get("token_count") or 0

        with self.client.session() as session:
            session.run(q.UPSERT_CHUNK, params)

    def upsert_entity(self, entity: EntityNode) -> bool:
        if not is_valid_entity_type(entity.entity_type):
            return False

        if looks_like_blocked_entity(entity.name):
            return False

        params = entity.model_dump()
        params["metadata_json"] = self._to_json(entity.metadata)
        params.pop("metadata", None)

        # Neo4j không lưu property null, nên ép default để tránh missing property warnings
        params["description"] = params.get("description") or ""
        params["profile_text"] = params.get("profile_text") or ""
        params["aliases"] = params.get("aliases") or []
        params["local_keys"] = params.get("local_keys") or []
        params["global_keys"] = params.get("global_keys") or []
        params["mention_count"] = params.get("mention_count") or 0
        params["source_count"] = params.get("source_count") or 0

        with self.client.session() as session:
            session.run(q.UPSERT_ENTITY, params)

        return True

    def link_chunk_mentions_entity(
        self,
        chunk_id: str,
        entity_id: str,
        confidence: float = 1.0,
        evidence_text: str | None = None,
        section: str | None = None,
    ) -> None:
        params = {
            "chunk_id": chunk_id,
            "entity_id": entity_id,
            "confidence": float(confidence),
            "evidence_text": evidence_text,
            "section": section,
        }

        with self.client.session() as session:
            session.run(q.LINK_CHUNK_MENTIONS_ENTITY, params)

    def upsert_medical_relation(
        self,
        relation: MedicalRelation,
        subject_type: str,
        object_type: str,
    ) -> bool:
        """
        Relation phải qua validator domain/range trước khi ghi.
        Return:
            True nếu ghi thành công.
            False nếu relation bị loại.
        """
        if not is_valid_relation_schema(
            relation_type=relation.relation_type,
            subject_type=subject_type,
            object_type=object_type,
        ):
            return False

        params = relation.model_dump()
        params["metadata_json"] = self._to_json(relation.metadata)
        params.pop("metadata", None)

        with self.client.session() as session:
            session.run(q.UPSERT_MEDICAL_RELATION, params)

        return True

    def upsert_synonym(
        self,
        entity_id_1: str,
        entity_id_2: str,
        score: float = 1.0,
        method: str = "alias",
    ) -> None:
        if entity_id_1 == entity_id_2:
            return

        params = {
            "entity_id_1": entity_id_1,
            "entity_id_2": entity_id_2,
            "score": float(score),
            "method": method,
        }

        with self.client.session() as session:
            session.run(q.UPSERT_SYNONYM, params)

    # =====================
    # Read methods
    # =====================

    def get_entity_by_id(self, entity_id: str) -> EntityNode | None:
        with self.client.session() as session:
            record = session.run(q.GET_ENTITY_BY_ID, {"entity_id": entity_id}).single()

        if not record:
            return None

        return self._record_to_entity(record)

    def find_entities_by_normalized_name(
        self,
        text: str,
        limit: int = 10,
    ) -> list[EntityNode]:
        params = {
            "text": text.strip().lower(),
            "limit": limit,
        }

        with self.client.session() as session:
            result = session.run(q.FIND_ENTITIES_BY_NORMALIZED_NAME, params)
            return [self._record_to_entity(record) for record in result]

    def get_chunks_by_entity_ids(
        self,
        entity_ids: list[str],
        limit: int = 20,
    ) -> list[ChunkNode]:
        if not entity_ids:
            return []

        params = {
            "entity_ids": entity_ids,
            "limit": limit,
        }

        with self.client.session() as session:
            result = session.run(q.GET_CHUNKS_BY_ENTITY_IDS, params)
            return [self._record_to_chunk(record) for record in result]

    def get_neighbor_entities(
        self,
        entity_ids: list[str],
        limit: int = 30,
    ) -> list[EntityNode]:
        if not entity_ids:
            return []

        params = {
            "entity_ids": entity_ids,
            "limit": limit,
        }

        with self.client.session() as session:
            result = session.run(q.GET_NEIGHBOR_ENTITIES, params)
            return [self._record_to_entity(record) for record in result]

    def get_graph_stats(self) -> dict[str, int]:
        with self.client.session() as session:
            record = session.run(q.GET_GRAPH_STATS).single()

        if not record:
            return {
                "article_count": 0,
                "chunk_count": 0,
                "entity_count": 0,
                "relation_count": 0,
            }

        return {
            "article_count": int(record["article_count"]),
            "chunk_count": int(record["chunk_count"]),
            "entity_count": int(record["entity_count"]),
            "relation_count": int(record["relation_count"]),
        }
        # =====================
    # Entity lookup / resolution
    # =====================

    def get_entities_by_ids(self, entity_ids: list[str]) -> list[EntityNode]:
        if not entity_ids:
            return []

        with self.client.session() as session:
            result = session.run(q.GET_ENTITIES_BY_IDS, {"entity_ids": entity_ids})
            return [self._record_to_entity(record) for record in result]

    def get_entities_by_normalized_names(
        self,
        normalized_names: list[str],
    ) -> list[EntityNode]:
        if not normalized_names:
            return []

        with self.client.session() as session:
            result = session.run(
                q.GET_ENTITIES_BY_NORMALIZED_NAMES,
                {"normalized_names": normalized_names},
            )
            return [self._record_to_entity(record) for record in result]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[ChunkNode]:
        if not chunk_ids:
            return []

        with self.client.session() as session:
            result = session.run(q.GET_CHUNKS_BY_IDS, {"chunk_ids": chunk_ids})
            return [self._record_to_chunk(record) for record in result]

    # =====================
    # LightRAG-style retrieval views
    # =====================

    def search_entities_for_lightrag(
        self,
        query_text: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[EntityNode]:
        params = {
            "query_text": query_text.strip().lower(),
            "entity_types": entity_types,
            "limit": limit,
        }

        with self.client.session() as session:
            result = session.run(q.SEARCH_ENTITIES_FOR_LIGHTRAG, params)
            return [self._record_to_entity(record) for record in result]

    def search_relations_for_lightrag(
        self,
        query_text: str,
        relation_types: list[str] | None = None,
        limit: int = 30,
    ) -> list[MedicalRelationView]:
        params = {
            "query_text": query_text.strip().lower(),
            "relation_types": relation_types,
            "limit": limit,
        }

        with self.client.session() as session:
            result = session.run(q.SEARCH_RELATIONS_FOR_LIGHTRAG, params)
            return [self._record_to_relation_view(record) for record in result]

    def get_relations_by_entity_ids(
        self,
        entity_ids: list[str],
        relation_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[MedicalRelationView]:
        if not entity_ids:
            return []

        params = {
            "entity_ids": entity_ids,
            "relation_types": relation_types,
            "limit": limit,
        }

        with self.client.session() as session:
            result = session.run(q.GET_RELATIONS_BY_ENTITY_IDS, params)
            return [self._record_to_relation_view(record) for record in result]

    def get_chunks_by_relation_ids(
        self,
        relation_ids: list[str],
        limit: int = 50,
    ) -> list[ChunkNode]:
        if not relation_ids:
            return []

        params = {
            "relation_ids": relation_ids,
            "limit": limit,
        }

        with self.client.session() as session:
            result = session.run(q.GET_CHUNKS_BY_RELATION_IDS, params)
            return [self._record_to_chunk(record) for record in result]

    def get_entity_context_bundle(
        self,
        entity_ids: list[str],
        relation_types: list[str] | None = None,
        max_relations: int = 80,
        max_chunks: int = 50,
    ) -> EntityContextBundle:
        """
        View gom context cho LightRAG:
        entity seeds + relations quanh seed + evidence chunks.
        """
        entities = self.get_entities_by_ids(entity_ids)

        relations = self.get_relations_by_entity_ids(
            entity_ids=entity_ids,
            relation_types=relation_types,
            limit=max_relations,
        )

        relation_ids = [r.relation_id for r in relations]
        chunks = self.get_chunks_by_relation_ids(
            relation_ids=relation_ids,
            limit=max_chunks,
        )

        return EntityContextBundle(
            entities=entities,
            relations=relations,
            chunks=chunks,
        )

    # =====================
    # HippoRAG-style retrieval views
    # =====================

    def find_seed_entities(
        self,
        query_text: str,
        limit: int = 10,
    ) -> list[EntityNode]:
        """
        Seed lookup cho HippoRAG.
        Cố tình chỉ search name/normalized_name/aliases, không search profile quá rộng.
        """
        params = {
            "query_text": query_text.strip().lower(),
            "limit": limit,
        }

        with self.client.session() as session:
            result = session.run(q.FIND_SEED_ENTITIES, params)
            return [self._record_to_entity(record) for record in result]

    def get_entity_adjacency(
        self,
        relation_types: list[str] | None = None,
        limit: int = 100_000,
    ) -> list[EntityAdjacencyEdge]:
        """
        Export adjacency toàn graph cho PPR.
        Dùng khi graph vừa/nhỏ.
        """
        params = {
            "relation_types": relation_types,
            "limit": limit,
        }

        with self.client.session() as session:
            result = session.run(q.GET_ENTITY_ADJACENCY, params)
            return [self._record_to_adjacency_edge(record) for record in result]

    def get_entity_adjacency_around_seeds(
        self,
        seed_entity_ids: list[str],
        max_hops: int = 2,
        relation_types: list[str] | None = None,
        limit: int = 100_000,
    ) -> list[EntityAdjacencyEdge]:
        """
        Export subgraph quanh seed cho HippoRAG/PPR.
        max_hops chỉ hỗ trợ 1, 2, 3 để tránh Cypher động.
        """
        if not seed_entity_ids:
            return []

        if max_hops <= 1:
            query = q.GET_ENTITY_ADJACENCY_AROUND_SEEDS_1HOP
        elif max_hops == 2:
            query = q.GET_ENTITY_ADJACENCY_AROUND_SEEDS_2HOP
        else:
            query = q.GET_ENTITY_ADJACENCY_AROUND_SEEDS_3HOP

        params = {
            "seed_entity_ids": seed_entity_ids,
            "relation_types": relation_types,
            "limit": limit,
        }

        with self.client.session() as session:
            result = session.run(query, params)
            return [self._record_to_adjacency_edge(record) for record in result]

    def get_chunk_entity_links(
        self,
        entity_ids: list[str],
        limit: int = 100_000,
    ) -> list[ChunkEntityLink]:
        """
        Mapping entity -> chunk để HippoRAG aggregate PPR score về chunks.
        """
        if not entity_ids:
            return []

        params = {
            "entity_ids": entity_ids,
            "limit": limit,
        }

        with self.client.session() as session:
            result = session.run(q.GET_CHUNK_ENTITY_LINKS, params)
            return [self._record_to_chunk_entity_link(record) for record in result]
    # =====================
    # Helpers
    # =====================

    def _to_json(self, data: dict[str, Any] | None) -> str:
        return json.dumps(data or {}, ensure_ascii=False)

    def _from_json(self, text: str | None) -> dict[str, Any]:
        if not text:
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _record_to_entity(self, record) -> EntityNode:
        return EntityNode(
            entity_id=record["entity_id"],
            name=record["name"],
            normalized_name=record["normalized_name"],
            entity_type=record["entity_type"],
            aliases=list(record["aliases"] or []),
            description=record["description"],
            profile_text=record["profile_text"],
            local_keys=list(record["local_keys"] or []),
            global_keys=list(record["global_keys"] or []),
            mention_count=int(record["mention_count"] or 0),
            source_count=int(record["source_count"] or 0),
            score=float(record.get("score", 0.0) or 0.0),
            metadata=self._from_json(record["metadata_json"]),
        )

    def _record_to_chunk(self, record) -> ChunkNode:
        return ChunkNode(
            chunk_id=record["chunk_id"],
            article_id=record["article_id"],
            source_url=record["source_url"],
            title=record["title"],
            section=record["section"],
            subsection=record["subsection"],
            text=record["text"],
            contextualized_text=record["contextualized_text"],
            chunk_index=int(record["chunk_index"] or 0),
            token_count=record["token_count"],
            score=float(record.get("score", 0.0) or 0.0),
            metadata=self._from_json(record["metadata_json"]),
        )
    
    def _record_to_relation_view(self, record) -> MedicalRelationView:
        return MedicalRelationView(
            relation_id=record["relation_id"],
            relation_type=record["relation_type"],

            subject_entity_id=record["subject_entity_id"],
            subject_name=record["subject_name"],
            subject_type=record["subject_type"],

            object_entity_id=record["object_entity_id"],
            object_name=record["object_name"],
            object_type=record["object_type"],

            evidence_text=record["evidence_text"],
            evidence_chunk_ids=list(record["evidence_chunk_ids"] or []),
            confidence=float(record["confidence"] or 1.0),
            section=record["section"],
            source_url=record["source_url"],
            score=float(record["score"] or 0.0),
            metadata=self._from_json(record["metadata_json"]),
        )

    def _record_to_adjacency_edge(self, record) -> EntityAdjacencyEdge:
        return EntityAdjacencyEdge(
            source_entity_id=record["source_entity_id"],
            target_entity_id=record["target_entity_id"],
            relation_id=record["relation_id"],
            relation_type=record["relation_type"],
            confidence=float(record["confidence"] or 1.0),
            weight=float(record["weight"] or 1.0),
        )

    def _record_to_chunk_entity_link(self, record) -> ChunkEntityLink:
        return ChunkEntityLink(
            chunk_id=record["chunk_id"],
            entity_id=record["entity_id"],
            entity_name=record["entity_name"],
            entity_type=record["entity_type"],
            confidence=float(record["confidence"] or 1.0),
            section=record["section"],
            evidence_text=record["evidence_text"],
        )
    
    def get_relations_by_ids(
        self,
        relation_ids: list[str],
    ) -> list[MedicalRelationView]:
        if not relation_ids:
            return []

        with self.client.session() as session:
            result = session.run(
                q.GET_RELATIONS_BY_IDS,
                {"relation_ids": relation_ids},
            )
            return [self._record_to_relation_view(record) for record in result]
        
    def get_relation_context_bundle(
        self,
        relation_ids: list[str],
        max_chunks: int = 50,
    ) -> EntityContextBundle:
        """
        Dùng sau khi Qdrant search medical_relations trả relation_ids.
        Lấy relations + subject/object entities + evidence chunks.
        """
        relations = self.get_relations_by_ids(relation_ids)

        entity_ids = []
        for r in relations:
            entity_ids.append(r.subject_entity_id)
            entity_ids.append(r.object_entity_id)

        entity_ids = list(dict.fromkeys(entity_ids))

        entities = self.get_entities_by_ids(entity_ids)

        chunks = self.get_chunks_by_relation_ids(
            relation_ids=relation_ids,
            limit=max_chunks,
        )

        return EntityContextBundle(
            entities=entities,
            relations=relations,
            chunks=chunks,
        )