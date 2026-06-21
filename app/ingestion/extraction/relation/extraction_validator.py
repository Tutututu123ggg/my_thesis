import re
import unicodedata

from app.domain.medical_schema import (
    ENTITY_TYPES,
    RELATION_TYPES,
    get_all_relation_keyword_intents,
    get_relation_keyword_intent,
    is_valid_entity_type,
    is_valid_relation_by_schema_policy,
    looks_like_blocked_entity,
)
from app.ingestion.extraction.common.extraction_models import (
    ExtractedEntity,
    ExtractedGraph,
    ExtractedRelation,
)
from app.ingestion.extraction.keyword.keyword_profiler import (
    clean_retrieval_keywords,
    deterministic_relation_keywords,
)


# Validator philosophy:
# - LLM decides medical semantics.
# - Validator only normalizes output and removes format/invariant errors.
# - Do not drop/remap relations because their relation_type seems semantically imperfect.
# - Keep recall high for GraphRAG; final grounding still comes from evidence passages.

# If True, only entities that appear as endpoints of validated relations are kept.
# False is safer for recall/debug because medical chunks with entities but no relations
# will not become empty. Set True only when you intentionally want a strict triple-first KG.
KEEP_ONLY_RELATION_ENDPOINT_ENTITIES = False


GENERIC_KEYWORDS = {
    "triệu chứng",
    "biểu hiện",
    "điều trị",
    "biến chứng",
    "chẩn đoán",
    "nguyên nhân",
    "phòng ngừa",
    "yếu tố",
    "bệnh lý",
}


def normalize_name(name: str) -> str:
    name = str(name or "").strip().lower()
    name = name.replace("đ", "d").replace("Đ", "D")
    name = unicodedata.normalize("NFD", name)
    name = name.encode("ascii", "ignore").decode("utf-8")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def humanize_schema_token(value: str) -> str:
    return str(value or "").lower().replace("_", " ").strip()


def schema_keyword_keys() -> set[str]:
    values: set[str] = set()

    for item in ENTITY_TYPES:
        values.add(normalize_name(item))
        values.add(normalize_name(humanize_schema_token(item)))

    for item in RELATION_TYPES:
        values.add(normalize_name(item))
        values.add(normalize_name(humanize_schema_token(item)))

    for item in get_all_relation_keyword_intents():
        values.add(normalize_name(item))

    return {value for value in values if value}


SCHEMA_KEYWORD_KEYS = schema_keyword_keys()


def clean_entity_name(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"\s+", " ", name)

    # Bỏ markdown emphasis ở tên entity, nhưng không động vào evidence_text.
    name = name.replace("**", "")
    name = name.replace("__", "")
    name = name.strip(" -–—:;,.()[]{}")

    return name.strip()


def clean_short_text(text: str, max_chars: int = 500) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("**", "")
    text = text.replace("__", "")
    return text[:max_chars].strip()


def is_bad_entity_name(name: str) -> bool:
    if not name:
        return True

    if len(name) < 2:
        return True

    if len(name.split()) > 12:
        return True

    if looks_like_blocked_entity(name):
        return True

    return False


def is_schema_like_keyword(keyword: str) -> bool:
    key = normalize_name(keyword)
    if not key:
        return True

    if key in SCHEMA_KEYWORD_KEYS:
        return True

    # Drop generic facet if it stands alone. Anchored phrases such as
    # "triệu chứng của hen phế quản" are kept by clean_relation_keywords.
    if key in {normalize_name(item) for item in GENERIC_KEYWORDS}:
        return True

    return False


def clean_keywords(items: list[str], limit: int = 8) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in items or []:
        cleaned = clean_short_text(str(item), max_chars=80)
        key = normalize_name(cleaned)

        if not key or key in seen:
            continue

        if len(cleaned.split()) > 10:
            continue

        if looks_like_blocked_entity(cleaned):
            continue

        if is_schema_like_keyword(cleaned):
            continue

        seen.add(key)
        result.append(cleaned)

        if len(result) >= limit:
            break

    return result


def clean_relation_keywords(
    items: list[str],
    subject: str,
    object_: str,
    limit: int = 8,
) -> list[str]:
    """Clean LLM-proposed relation keywords conservatively.

    Agent 1 is no longer trusted to create final LightRAG keywords. We keep
    only short atomic retrieval terms and drop sentence/clause fragments. Final
    deterministic fallback and optional article-level Agent 2 run later.
    """
    return clean_retrieval_keywords(items or [], limit=limit)

def clamp_confidence(value: float | int | str | None) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5

    if confidence < 0.0:
        return 0.0

    if confidence > 1.0:
        return 1.0

    return confidence


def make_fallback_keywords(relation: ExtractedRelation) -> list[str]:
    """Safe deterministic keywords used when Agent 1 returns nothing usable."""
    return deterministic_relation_keywords(relation)


class ExtractionValidator:
    """
    Normalize and format-check LLM extraction output.

    This validator intentionally avoids semantic judging. In particular, it does
    NOT drop/remap a relation just because the chosen relation_type may be less
    ideal. For GraphRAG, keeping a useful edge with faithful evidence is usually
    better than dropping it and losing the passage bridge.

    Lưu ý:
    - aliases = deprecated/unused for new extraction; validator always sets [].
    - surface_forms = internal bookkeeping field; validator always sets [].
    - Không đọc surface_forms từ LLM output.
    - Không dùng surface_forms để giữ synonym/tên gọi khác.
    - relation.keywords = indexing-time relation keys cho LightRAG-style relation retrieval.
    - Validator không merge canonical entity; merge exact-name do EntityResolver xử lý sau.
    """

    def validate(self, graph: ExtractedGraph) -> ExtractedGraph:
        if not graph.is_medical_knowledge_chunk:
            return ExtractedGraph(
                is_medical_knowledge_chunk=False,
                should_ingest_graph=False,
                skip_reason=graph.skip_reason or "non_medical_knowledge_chunk",
                entities=[],
                relations=[],
            )

        entities = self._validate_entities(graph.entities)
        entity_map = {
            (normalize_name(e.name), e.entity_type): e
            for e in entities
        }

        relations = self._validate_relations(graph.relations)

        # Add endpoint entities if the LLM forgot to include them in entities[].
        endpoint_keys: list[tuple[str, str]] = []
        seen_endpoint_keys: set[tuple[str, str]] = set()

        def remember_endpoint(key: tuple[str, str]) -> None:
            if key in seen_endpoint_keys:
                return
            seen_endpoint_keys.add(key)
            endpoint_keys.append(key)

        for relation in relations:
            s_key = (normalize_name(relation.subject), relation.subject_type)
            o_key = (normalize_name(relation.object), relation.object_type)
            remember_endpoint(s_key)
            remember_endpoint(o_key)

            if s_key not in entity_map:
                entity = ExtractedEntity(
                    name=relation.subject,
                    entity_type=relation.subject_type,
                    aliases=[],
                    surface_forms=[],
                    description="",
                    evidence_text=relation.evidence_text,
                )
                entities.append(entity)
                entity_map[s_key] = entity

            if o_key not in entity_map:
                entity = ExtractedEntity(
                    name=relation.object,
                    entity_type=relation.object_type,
                    aliases=[],
                    surface_forms=[],
                    description="",
                    evidence_text=relation.evidence_text,
                )
                entities.append(entity)
                entity_map[o_key] = entity

        if KEEP_ONLY_RELATION_ENDPOINT_ENTITIES:
            entities = [entity_map[key] for key in endpoint_keys if key in entity_map]

        should_ingest_graph = bool(entities or relations)
        return ExtractedGraph(
            is_medical_knowledge_chunk=True,
            should_ingest_graph=should_ingest_graph,
            skip_reason=None if should_ingest_graph else "no_valid_medical_entities_or_relations",
            entities=entities,
            relations=relations,
        )

    def _validate_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        by_key: dict[tuple[str, str], ExtractedEntity] = {}

        for entity in entities or []:
            name = clean_entity_name(entity.name)
            entity_type = str(entity.entity_type or "").strip()

            if not is_valid_entity_type(entity_type):
                continue

            if is_bad_entity_name(name):
                continue

            description = clean_short_text(entity.description, max_chars=500)
            evidence_text = clean_short_text(entity.evidence_text, max_chars=800)

            key = (normalize_name(name), entity_type)

            if key in by_key:
                old = by_key[key]
                by_key[key] = ExtractedEntity(
                    name=old.name,
                    entity_type=old.entity_type,
                    aliases=[],
                    surface_forms=[],
                    description=old.description or description,
                    evidence_text=old.evidence_text or evidence_text,
                )
                continue

            by_key[key] = ExtractedEntity(
                name=name,
                entity_type=entity_type,
                aliases=[],
                surface_forms=[],
                description=description,
                evidence_text=evidence_text,
            )

        return list(by_key.values())

    def _validate_relations(
        self,
        relations: list[ExtractedRelation],
    ) -> list[ExtractedRelation]:
        result: list[ExtractedRelation] = []
        seen: set[tuple[str, str, str, str, str]] = set()

        for relation in relations or []:
            subject = clean_entity_name(relation.subject)
            object_ = clean_entity_name(relation.object)
            subject_type = str(relation.subject_type or "").strip()
            object_type = str(relation.object_type or "").strip()
            relation_type = str(relation.relation_type or "").strip()
            evidence_text = clean_short_text(relation.evidence_text, max_chars=1000)

            # Format/invariant checks only.
            if is_bad_entity_name(subject) or is_bad_entity_name(object_):
                continue

            if normalize_name(subject) == normalize_name(object_):
                continue

            if subject_type not in ENTITY_TYPES or object_type not in ENTITY_TYPES:
                continue

            if relation_type not in RELATION_TYPES:
                continue

            if not evidence_text:
                continue

            if not is_valid_relation_by_schema_policy(
                relation_type,
                subject_type,
                object_type,
            ):
                continue

            key = (
                normalize_name(subject),
                subject_type,
                relation_type,
                normalize_name(object_),
                object_type,
            )

            if key in seen:
                continue

            cleaned_keywords = clean_relation_keywords(
                relation.keywords,
                subject=subject,
                object_=object_,
                limit=8,
            )

            cleaned_relation = ExtractedRelation(
                subject=subject,
                subject_type=subject_type,
                relation_type=relation_type,
                object=object_,
                object_type=object_type,
                description=clean_short_text(relation.description, max_chars=500),
                keywords=cleaned_keywords,
                evidence_text=evidence_text,
                confidence=clamp_confidence(relation.confidence),
            )

            if not cleaned_relation.keywords:
                cleaned_relation.keywords = make_fallback_keywords(cleaned_relation)

            result.append(cleaned_relation)
            seen.add(key)

        return result