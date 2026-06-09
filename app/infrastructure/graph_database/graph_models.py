from typing import Any

from pydantic import BaseModel, Field


class ArticleNode(BaseModel):
    article_id: str
    source: str
    url: str
    title: str

    description: str | None = None
    author: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    category: str | None = None
    crawled_at: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkNode(BaseModel):
    chunk_id: str
    article_id: str
    source_url: str
    title: str
    text: str

    section: str | None = None
    subsection: str | None = None
    contextualized_text: str | None = None
    chunk_index: int = 0
    token_count: int | None = None

    # Dùng cho retrieval result, không nhất thiết lưu vào Neo4j.
    score: float = 0.0

    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityNode(BaseModel):
    entity_id: str
    name: str
    normalized_name: str
    entity_type: str

    aliases: list[str] = Field(default_factory=list)
    description: str | None = None

    # Dành cho LightRAG-style.
    profile_text: str | None = None
    local_keys: list[str] = Field(default_factory=list)
    global_keys: list[str] = Field(default_factory=list)

    mention_count: int = 0
    source_count: int = 0

    # Dùng cho retrieval result, không nhất thiết lưu vào Neo4j.
    score: float = 0.0

    metadata: dict[str, Any] = Field(default_factory=dict)


class MedicalRelation(BaseModel):
    relation_id: str
    subject_entity_id: str
    object_entity_id: str
    relation_type: str

    evidence_text: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0

    section: str | None = None
    source_url: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class MedicalRelationView(BaseModel):
    relation_id: str
    relation_type: str

    subject_entity_id: str
    subject_name: str
    subject_type: str

    object_entity_id: str
    object_name: str
    object_type: str

    evidence_text: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    section: str | None = None
    source_url: str | None = None

    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityAdjacencyEdge(BaseModel):
    source_entity_id: str
    target_entity_id: str
    relation_id: str
    relation_type: str
    confidence: float = 1.0
    weight: float = 1.0


class ChunkEntityLink(BaseModel):
    chunk_id: str
    entity_id: str
    entity_name: str
    entity_type: str
    confidence: float = 1.0
    section: str | None = None
    evidence_text: str | None = None


class EntityContextBundle(BaseModel):
    entities: list[EntityNode] = Field(default_factory=list)
    relations: list[MedicalRelationView] = Field(default_factory=list)
    chunks: list[ChunkNode] = Field(default_factory=list)