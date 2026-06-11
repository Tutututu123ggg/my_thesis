import uuid
from typing import Any

from pydantic import BaseModel, Field


def make_qdrant_point_id(logical_id: str) -> str:
    """
    Qdrant chỉ nhận unsigned int hoặc UUID làm point id.
    Ta sinh UUID ổn định từ logical_id để upsert nhiều lần vẫn vào cùng point.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, logical_id))


class VectorPoint(BaseModel):
    point_id: str
    vector: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(BaseModel):
    point_id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class ChunkVectorPayload(BaseModel):
    chunk_id: str
    article_id: str
    source_url: str
    title: str
    section: str | None = None
    subsection: str | None = None
    chunk_index: int = 0
    token_count: int | None = None

    # text dùng để trả context cho Vector RAG.
    text: str | None = None
    contextualized_text: str | None = None

    # Metadata giữ nhẹ để filter/debug.
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityVectorPayload(BaseModel):
    entity_id: str
    name: str
    normalized_name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    surface_forms: list[str] = Field(default_factory=list)
    description: str | None = None
    profile_text: str | None = None


class RelationVectorPayload(BaseModel):
    relation_id: str
    relation_type: str
    subject_entity_id: str
    subject_name: str
    subject_type: str | None = None
    object_entity_id: str
    object_name: str
    object_type: str | None = None
    description: str | None = None
    keywords: list[str] = Field(default_factory=list)
    evidence_text: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    chunk_id: str | None = None
    section: str | None = None
    source_url: str | None = None
