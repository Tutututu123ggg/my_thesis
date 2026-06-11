from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ArticleDocument(BaseModel):
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
    related_links: list[str] = Field(default_factory=list)

    body: str
    file_path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionDocument(BaseModel):
    article_id: str
    title: str
    source_url: str

    section_id: str
    section_title: str
    section_level: int
    parent_section_title: str | None = None

    text: str
    order: int


class ChunkDocument(BaseModel):
    chunk_id: str
    article_id: str
    source_url: str
    title: str

    section: str | None = None
    subsection: str | None = None
    chunk_index: int

    text: str
    contextualized_text: str
    token_count: int

    metadata: dict[str, Any] = Field(default_factory=dict)