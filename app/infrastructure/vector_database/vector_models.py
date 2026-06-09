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
    text: str | None = None


class EntityVectorPayload(BaseModel):
    entity_id: str
    name: str
    normalized_name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)


class RelationVectorPayload(BaseModel):
    relation_id: str
    relation_type: str
    subject_entity_id: str
    subject_name: str
    object_entity_id: str
    object_name: str
    evidence_text: str | None = None
    source_url: str | None = None