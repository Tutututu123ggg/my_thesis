import re
import unicodedata

from app.domain.medical_schema import (
    is_valid_entity_type,
    is_valid_relation_schema,
    looks_like_blocked_entity,
)
from app.ingestion.extraction.extraction_models import (
    ExtractedEntity,
    ExtractedGraph,
    ExtractedRelation,
)


SYNONYM_POSITIVE_CUES = [
    "còn gọi là",
    "còn được gọi là",
    "hay gọi là",
    "còn có tên là",
    "tên khác là",
    "tên gọi khác",
    "tiếng anh là",
    "tên tiếng anh",
    "viết tắt là",
    "aka",
    "also known as",
]

SYNONYM_NEGATIVE_CUES = [
    "là dạng của",
    "là một dạng",
    "là thể của",
    "là một thể",
    "là loại",
    "là một loại",
    "phân loại",
    "bao gồm các thể",
    "các dạng",
]

RELATION_INTENT_KEYWORDS = {
    "LA_DANG_CUA": "phân loại",
    "DONG_NGHIA_VOI": "tên gọi khác",
    "CHAN_DOAN_PHAN_BIET_VOI": "chẩn đoán phân biệt",
    "CO_BIEU_HIEN": "triệu chứng",
    "LIEN_QUAN_YEU_TO": "yếu tố liên quan",
    "GAY_BIEN_CHUNG": "biến chứng",
    "ANH_HUONG_DEN": "ảnh hưởng",
    "CHAN_DOAN_BANG": "chẩn đoán",
    "DIEU_TRI_BANG": "điều trị",
    "PHONG_NGUA_BANG": "phòng ngừa",
    "CAN_TRANH": "cần tránh",
    "CHONG_CHI_DINH": "chống chỉ định",
    "TUONG_TAC_VOI": "tương tác",
}


def normalize_name(name: str) -> str:
    name = name.strip().lower()
    name = name.replace("đ", "d").replace("Đ", "D")
    name = unicodedata.normalize("NFD", name)
    name = name.encode("ascii", "ignore").decode("utf-8")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def clean_entity_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", " ", name)

    # Bỏ markdown emphasis ở tên entity, nhưng không động vào evidence_text.
    name = name.replace("**", "")
    name = name.replace("__", "")
    name = name.strip(" -–—:;,.()[]{}")

    return name.strip()


def clean_short_text(text: str, max_chars: int = 500) -> str:
    text = text.strip()
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


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        cleaned = clean_entity_name(item)
        key = normalize_name(cleaned)

        if not key or key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return result


def clean_keywords(items: list[str], limit: int = 5) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in items:
        cleaned = clean_short_text(item, max_chars=80)
        key = normalize_name(cleaned)

        if not key or key in seen:
            continue

        if len(cleaned.split()) > 8:
            continue

        if looks_like_blocked_entity(cleaned):
            continue

        seen.add(key)
        result.append(cleaned)

        if len(result) >= limit:
            break

    return result


def contains_any_normalized(text: str, cues: list[str]) -> bool:
    normalized_text = normalize_name(text)
    return any(normalize_name(cue) in normalized_text for cue in cues)


def has_parenthetical_synonym_evidence(
    evidence_text: str,
    subject: str,
    object_: str,
) -> bool:
    """
    Accept simple patterns such as:
    - Bệnh chàm (Eczema)
    - Eczema (bệnh chàm)

    This is not a generic synonym detector. It only validates that both endpoint
    strings appear close together in a parenthetical naming pattern.
    """
    evidence = normalize_name(evidence_text)
    s = re.escape(normalize_name(subject))
    o = re.escape(normalize_name(object_))

    if not s or not o:
        return False

    patterns = [
        rf"{s}\s*\(\s*{o}\s*\)",
        rf"{o}\s*\(\s*{s}\s*\)",
    ]

    return any(re.search(pattern, evidence) for pattern in patterns)


def looks_like_question_only(evidence_text: str) -> bool:
    text = evidence_text.strip()
    if not text:
        return True

    lowered = text.lower()

    if "?" in text:
        return True

    question_starters = [
        "khi nào",
        "vì sao",
        "tại sao",
        "làm gì",
        "làm thế nào",
        "có nên",
        "có cần",
        "bao lâu",
        "bao nhiêu",
        "ai",
        "ở đâu",
    ]

    return any(lowered.startswith(starter) for starter in question_starters)


def make_fallback_keywords(relation: ExtractedRelation) -> list[str]:
    intent = RELATION_INTENT_KEYWORDS.get(
        relation.relation_type,
        relation.relation_type.lower(),
    )
    return clean_keywords([relation.subject, relation.object, intent], limit=5)


class ExtractionValidator:
    """
    Validate schema + lọc nhiễu cho LLM output.

    Lưu ý:
    - aliases = deprecated/unused for new extraction; validator always sets [].
    - surface_forms = internal bookkeeping field; validator always sets [].
    - Không đọc surface_forms từ LLM output.
    - Không dùng surface_forms để giữ synonym/tên gọi khác.
    - relation.keywords = indexing-time relation keys cho LightRAG-style relation retrieval.
    - Validator không merge canonical entity; merge exact-name do EntityResolver xử lý sau.
    """

    def validate(self, graph: ExtractedGraph) -> ExtractedGraph:
        entities = self._validate_entities(graph.entities)
        entity_map = {
            (normalize_name(e.name), e.entity_type): e
            for e in entities
        }

        relations = self._validate_relations(graph.relations)

        # Thêm entity bị relation nhắc đến nhưng LLM quên đưa vào entities.
        for relation in relations:
            s_key = (normalize_name(relation.subject), relation.subject_type)
            o_key = (normalize_name(relation.object), relation.object_type)

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

        return ExtractedGraph(
            entities=entities,
            relations=relations,
        )

    def _validate_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        by_key: dict[tuple[str, str], ExtractedEntity] = {}

        for entity in entities:
            name = clean_entity_name(entity.name)
            entity_type = entity.entity_type.strip()

            if not is_valid_entity_type(entity_type):
                continue

            if is_bad_entity_name(name):
                continue

            description = clean_short_text(entity.description, max_chars=500)
            evidence_text = entity.evidence_text.strip()

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

        for relation in relations:
            subject = clean_entity_name(relation.subject)
            object_ = clean_entity_name(relation.object)
            subject_type = relation.subject_type.strip()
            object_type = relation.object_type.strip()
            relation_type = relation.relation_type.strip()
            evidence_text = relation.evidence_text.strip()

            if is_bad_entity_name(subject) or is_bad_entity_name(object_):
                continue

            if not is_valid_relation_schema(
                relation_type=relation_type,
                subject_type=subject_type,
                object_type=object_type,
            ):
                continue

            if not evidence_text:
                continue

            if looks_like_question_only(evidence_text):
                continue

            if relation_type == "DONG_NGHIA_VOI" and not self._is_valid_synonym_relation(
                subject=subject,
                subject_type=subject_type,
                object_=object_,
                object_type=object_type,
                evidence_text=evidence_text,
            ):
                continue

            # Defensive rule. Schema should already enforce this after patching medical_schema.py.
            if relation_type == "CHAN_DOAN_BANG" and object_type != "PHUONG_PHAP_CHAN_DOAN":
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

            cleaned_relation = ExtractedRelation(
                subject=subject,
                subject_type=subject_type,
                relation_type=relation_type,
                object=object_,
                object_type=object_type,
                description=clean_short_text(relation.description, max_chars=500),
                keywords=clean_keywords(relation.keywords, limit=5),
                evidence_text=evidence_text,
                confidence=float(relation.confidence),
            )

            if not cleaned_relation.keywords:
                cleaned_relation.keywords = make_fallback_keywords(cleaned_relation)

            result.append(cleaned_relation)
            seen.add(key)

        return result

    def _is_valid_synonym_relation(
        self,
        subject: str,
        subject_type: str,
        object_: str,
        object_type: str,
        evidence_text: str,
    ) -> bool:
        if subject_type != object_type:
            return False

        if normalize_name(subject) == normalize_name(object_):
            return False

        if contains_any_normalized(evidence_text, SYNONYM_NEGATIVE_CUES):
            return False

        if contains_any_normalized(evidence_text, SYNONYM_POSITIVE_CUES):
            return True

        if has_parenthetical_synonym_evidence(evidence_text, subject, object_):
            return True

        return False