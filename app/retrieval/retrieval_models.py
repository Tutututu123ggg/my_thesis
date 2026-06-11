from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RetrievalSource = Literal[
    "vector_chunk",
    "lightrag_entity",
    "lightrag_relation",
    "graph_expansion",
    "hipporag",
    "mixed",
]


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    source: RetrievalSource

    article_id: str = ""
    title: str = ""
    section: str = ""
    subsection: str = ""
    source_url: str = ""
    text: str = ""
    contextualized_text: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    def context_text(self) -> str:
        text = self.contextualized_text or self.text
        title = self.title or "Unknown title"
        section = self.section or ""
        subsection = self.subsection or ""
        header = f"[Chunk {self.chunk_id}] {title}"
        if section:
            header += f" | {section}"
        if subsection:
            header += f" | {subsection}"
        return f"{header}\n{text}".strip()


@dataclass
class RetrievedEntity:
    entity_id: str
    score: float
    source: RetrievalSource

    name: str = ""
    normalized_name: str = ""
    entity_type: str = ""
    description: str = ""
    profile_text: str = ""
    aliases: list[str] = field(default_factory=list)
    surface_forms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def context_text(self) -> str:
        desc = self.profile_text or self.description
        return (
            f"- {self.name} ({self.entity_type})"
            + (f": {desc}" if desc else "")
        ).strip()


@dataclass
class RetrievedRelation:
    relation_id: str
    score: float
    source: RetrievalSource

    relation_type: str = ""
    subject_entity_id: str = ""
    subject_name: str = ""
    subject_type: str = ""
    object_entity_id: str = ""
    object_name: str = ""
    object_type: str = ""

    description: str = ""
    keywords: list[str] = field(default_factory=list)
    evidence_text: str = ""
    evidence_chunk_ids: list[str] = field(default_factory=list)
    section: str = ""
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def context_text(self) -> str:
        head = (
            f"- {self.subject_name} -[{self.relation_type}]-> {self.object_name}"
        )
        parts = [head]
        if self.description:
            parts.append(f"  Mô tả: {self.description}")
        if self.keywords:
            parts.append(f"  Keywords: {', '.join(self.keywords)}")
        if self.evidence_text:
            parts.append(f"  Evidence: {self.evidence_text}")
        return "\n".join(parts)


@dataclass
class QueryKeywords:
    low_level_keywords: list[str] = field(default_factory=list)
    high_level_keywords: list[str] = field(default_factory=list)

    def low_query_text(self) -> str:
        return "; ".join(self.low_level_keywords)

    def high_query_text(self) -> str:
        return "; ".join(self.high_level_keywords)


@dataclass
class RetrievalResult:
    query: str
    method: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    entities: list[RetrievedEntity] = field(default_factory=list)
    relations: list[RetrievedRelation] = field(default_factory=list)
    keywords: QueryKeywords | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def assemble_context(
        self,
        max_entities: int = 30,
        max_relations: int = 40,
        max_chunks: int = 12,
    ) -> str:
        sections: list[str] = []

        if self.keywords:
            sections.append(
                "## Query keywords\n"
                f"Low-level: {', '.join(self.keywords.low_level_keywords) or '(none)'}\n"
                f"High-level: {', '.join(self.keywords.high_level_keywords) or '(none)'}"
            )

        if self.entities:
            body = "\n".join(e.context_text() for e in self.entities[:max_entities])
            sections.append(f"## Entities\n{body}")

        if self.relations:
            body = "\n".join(r.context_text() for r in self.relations[:max_relations])
            sections.append(f"## Relations\n{body}")

        if self.chunks:
            body = "\n\n".join(c.context_text() for c in self.chunks[:max_chunks])
            sections.append(f"## Source chunks\n{body}")

        return "\n\n".join(sections).strip()
