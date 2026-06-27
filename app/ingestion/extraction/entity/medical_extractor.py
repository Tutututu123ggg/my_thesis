import os
import re
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from app.ingestion.extraction.common.extraction_models import (
    ExtractedGraph,
    ExtractedRelation,
    LLMEntityExtractionResult,
    LLMRelationExtractionResult,
)
from app.ingestion.extraction.entity.extraction_prompt import (
    build_entity_system_prompt,
    build_entity_user_prompt,
)
from app.ingestion.extraction.relation.extraction_prompt import (
    build_relation_system_prompt,
    build_relation_user_prompt,
)
from app.ingestion.processing import ChunkDocument


TParsed = TypeVar("TParsed", bound=BaseModel)


def _strip_heading_marker(text: str) -> str:
    return re.sub(r"^\s*#{1,6}\s*", "", str(text or "").strip()).strip()


def _split_heading_context(heading_context: str) -> list[str]:
    return [
        _strip_heading_marker(part)
        for part in str(heading_context or "").split("|")
        if _strip_heading_marker(part)
    ]


def build_contextualized_markdown_for_extraction(chunk: ChunkDocument) -> str:
    """
    Tạo input markdown cho LLM theo đúng cấu trúc tài liệu:

    # Article title
    ## Heading context
    ### Raw chunk headings/content

    Vector retrieval vẫn có thể dùng raw chunk text ở pipeline khác;
    hàm này chỉ phục vụ extraction graph.
    """

    parts: list[str] = []

    title = _strip_heading_marker(chunk.title)
    if title:
        parts.append(f"# {title}")

    metadata = getattr(chunk, "metadata", {}) or {}
    heading_context = metadata.get("heading_path_context") or ""

    for index, heading in enumerate(_split_heading_context(heading_context), start=2):
        level = min(index, 6)
        parts.append(f"{'#' * level} {heading}")

    text = str(chunk.text or "").strip()
    if text:
        parts.append(text)

    return "\n\n".join(parts).strip()


def build_extraction_user_prompt(chunk: ChunkDocument) -> str:
    """Backward-compatible helper for old imports; entity stage uses this text."""

    extraction_markdown = build_contextualized_markdown_for_extraction(chunk)
    return build_entity_user_prompt(extraction_markdown)


def _relation_key(relation: ExtractedRelation) -> tuple[str, str, str, str, str]:
    return (
        relation.subject.strip().lower(),
        relation.subject_type.strip(),
        relation.relation_type.strip(),
        relation.object.strip().lower(),
        relation.object_type.strip(),
    )


def _dedupe_extracted_relations(relations: list[ExtractedRelation]) -> list[ExtractedRelation]:
    result: list[ExtractedRelation] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for relation in relations:
        key = _relation_key(relation)
        if key in seen:
            continue
        seen.add(key)
        result.append(relation)
    return result


class MedicalExtractor:
    """
    Multi-stage LLM extractor.

    LLM #1: entity extraction + local explicit synonym/alias detection.
    LLM #2: relation extraction using entity list from LLM #1.
    LLM #3: keyword profiler runs later in ArticleExtractionPipeline.

    This class only runs stages 1 and 2 and returns ExtractedGraph.
    It does not validate, merge, keyword-profile, or write database.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        prompt_mode: str = "strict_text",
        entity_max_tokens: int | None = None,
        relation_max_tokens: int | None = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.prompt_mode = prompt_mode
        self.entity_max_tokens = int(entity_max_tokens or os.getenv("OPENAI_ENTITY_MAX_TOKENS", "2500"))
        self.relation_max_tokens = int(relation_max_tokens or os.getenv("OPENAI_RELATION_MAX_TOKENS", "4000"))
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _parse_llm(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: type[TParsed],
        max_tokens: int,
    ) -> TParsed | None:
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "response_format": response_format,
        }

        # OpenAI chat completions still accepts max_tokens for gpt-4o-mini.
        # This matters for relation recall in dense bullet/list chunks.
        if max_tokens and max_tokens > 0:
            kwargs["max_tokens"] = int(max_tokens)

        completion = self.client.beta.chat.completions.parse(**kwargs)
        return completion.choices[0].message.parsed

    def extract_chunk(self, chunk: ChunkDocument) -> ExtractedGraph:
        markdown = build_contextualized_markdown_for_extraction(chunk)

        entity_result = self._parse_llm(
            system_prompt=build_entity_system_prompt(self.prompt_mode),
            user_prompt=build_entity_user_prompt(markdown),
            response_format=LLMEntityExtractionResult,
            max_tokens=self.entity_max_tokens,
        )

        if entity_result is None:
            return ExtractedGraph()

        if entity_result.is_noise_chunk:
            return ExtractedGraph(
                is_medical_knowledge_chunk=False,
                should_ingest_graph=False,
                skip_reason=entity_result.skip_reason or "noise_chunk",
                entities=[],
                relations=[],
            )

        entities = entity_result.to_internal_entities()
        relation_result = self._parse_llm(
            system_prompt=build_relation_system_prompt(self.prompt_mode),
            user_prompt=build_relation_user_prompt(
                markdown,
                [entity.model_dump() for entity in entities],
            ),
            response_format=LLMRelationExtractionResult,
            max_tokens=self.relation_max_tokens,
        )

        # Explicit synonym pairs belong to entity-level extraction. Convert them
        # in code instead of asking LLM #2 to recreate synonym edges; this keeps
        # relation extraction focused on factual medical edges and avoids duplicate
        # DONG_NGHIA_VOI outputs.
        relations: list[ExtractedRelation] = []
        relations.extend(entity_result.to_synonym_relations())
        if relation_result is not None:
            relations.extend(relation_result.to_internal_relations())
        relations = _dedupe_extracted_relations(relations)

        return ExtractedGraph(
            is_medical_knowledge_chunk=True,
            should_ingest_graph=bool(entities or relations),
            skip_reason=None if (entities or relations) else "no_medical_entities_or_relations",
            entities=entities,
            relations=relations,
        )