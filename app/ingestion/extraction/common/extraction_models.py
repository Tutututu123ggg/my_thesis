from pydantic import BaseModel, Field

from app.domain.medical_schema import should_flip_relation_direction


class LLMExtractedEntity(BaseModel):
    """
    Entity schema exposed to LLM.

    Important:
    - Keep this schema minimal.
    - Do not expose internal bookkeeping fields here.
    - Different names that are explicitly synonyms must be represented by
      DONG_NGHIA_VOI relations, not bundled inside one entity.
    """

    name: str = Field(
        description="Tên thực thể y khoa ngắn gọn, đúng như xuất hiện trong chunk."
    )
    entity_type: str = Field(
        description="Một trong các EntityType đã cho."
    )
    description: str = Field(
        default="",
        description="Mô tả ngắn về entity dựa trên chunk. Để rỗng nếu chunk không đủ thông tin.",
    )
    evidence_text: str = Field(
        description="Đoạn bằng chứng ngắn từ chunk chứng minh entity này xuất hiện."
    )
    is_anchor_candidate: bool = Field(
        default=False,
        description=(
            "True nếu entity là khái niệm y khoa có thể hữu ích độc lập hoặc "
            "có khả năng liên quan nhiều bệnh/tài liệu, ví dụ bệnh lý, thuốc, "
            "phương pháp chẩn đoán, tác nhân bệnh sinh, triệu chứng đặc hiệu."
        ),
    )
    anchor_reason: str = Field(
        default="",
        description="Lý do ngắn nếu is_anchor_candidate=true; để rỗng nếu không chắc.",
    )


class LLMExtractedRelation(BaseModel):
    """
    Binary relation schema exposed to LLM.
    """

    subject: str = Field(description="Tên subject entity.")
    subject_type: str = Field(description="Entity type của subject.")
    relation_type: str = Field(description="Một trong các RelationType đã cho.")

    object: str = Field(description="Tên object entity.")
    object_type: str = Field(description="Entity type của object.")

    description: str = Field(
        default="",
        description="Mô tả ngắn về relation dựa trên chunk. Để rỗng nếu không cần.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Metadata phụ; có thể để [] vì ArticleKeywordProfiler sẽ sinh keyword sạch sau validation.",
    )
    evidence_text: str = Field(
        description="Đoạn bằng chứng ngắn từ chunk chứng minh relation."
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class LLMRelationEndpoint(BaseModel):
    """A compact endpoint used by grouped LLM relations."""

    name: str = Field(description="Tên entity endpoint.")
    entity_type: str = Field(description="Entity type của endpoint.")


class LLMExtractedRelationGroup(BaseModel):
    """
    Grouped relation schema exposed to LLM.

    Chỉ dùng cho quan hệ 1-vs-many:
    - một subject tới nhiều objects; hoặc
    - nhiều subjects tới một object.

    Không dùng cho many-vs-many vì dễ nổ tổ hợp và sai ngữ nghĩa.
    Internal graph sẽ expand group thành các binary triples.
    """

    subjects: list[LLMRelationEndpoint] = Field(
        description="Một hoặc nhiều subject endpoints. Nếu nhiều subject thì objects chỉ nên có đúng 1 item."
    )
    relation_type: str = Field(description="Một trong các RelationType đã cho.")
    objects: list[LLMRelationEndpoint] = Field(
        description="Một hoặc nhiều object endpoints. Nếu nhiều object thì subjects chỉ nên có đúng 1 item."
    )

    description: str = Field(
        default="",
        description="Mô tả ngắn về relation group dựa trên chunk. Để rỗng nếu không cần.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Metadata phụ; có thể để [] vì ArticleKeywordProfiler sẽ sinh keyword sạch sau validation.",
    )
    evidence_text: str = Field(
        description="Đoạn bằng chứng ngắn từ chunk chứng minh relation group."
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)



def _normalize_relation_direction(relation: "ExtractedRelation") -> "ExtractedRelation":
    """Normalize obvious reversed triples using schema direction policy."""

    should_flip = should_flip_relation_direction(
        relation_type=relation.relation_type,
        subject_type=relation.subject_type,
        object_type=relation.object_type,
    )

    if not should_flip:
        return relation

    return ExtractedRelation(
        subject=relation.object,
        subject_type=relation.object_type,
        relation_type=relation.relation_type,
        object=relation.subject,
        object_type=relation.subject_type,
        description=relation.description,
        keywords=relation.keywords,
        evidence_text=relation.evidence_text,
        confidence=relation.confidence,
    )


def _relation_key(relation: "ExtractedRelation") -> tuple[str, str, str, str, str]:
    return (
        relation.subject.strip().lower(),
        relation.subject_type.strip(),
        relation.relation_type.strip(),
        relation.object.strip().lower(),
        relation.object_type.strip(),
    )


def _dedupe_relations(relations: list["ExtractedRelation"]) -> list["ExtractedRelation"]:
    result: list[ExtractedRelation] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for relation in relations:
        key = _relation_key(relation)
        if key in seen:
            continue
        seen.add(key)
        result.append(relation)

    return result


def _expand_relation_group(group: LLMExtractedRelationGroup) -> list["ExtractedRelation"]:
    """Expand only 1-vs-many or many-vs-1 groups.

    Allowed:
    - A --r--> [B, C, D]
    - [A, B, C] --r--> D

    Dropped:
    - [A, B] --r--> [C, D]
    """

    subjects = list(group.subjects or [])
    objects = list(group.objects or [])

    if not subjects or not objects:
        return []

    if len(subjects) > 1 and len(objects) > 1:
        return []

    expanded: list[ExtractedRelation] = []

    for subject in subjects:
        for object_ in objects:
            relation = ExtractedRelation(
                subject=subject.name,
                subject_type=subject.entity_type,
                relation_type=group.relation_type,
                object=object_.name,
                object_type=object_.entity_type,
                description=group.description,
                keywords=group.keywords,
                evidence_text=group.evidence_text,
                confidence=group.confidence,
            )
            expanded.append(_normalize_relation_direction(relation))

    return expanded


class LLMExtractedGraph(BaseModel):
    """
    Raw graph returned by LLM.

    This is intentionally not the same as ExtractedGraph.
    The internal graph can contain extra bookkeeping fields, but LLM should not see them.
    """

    is_medical_knowledge_chunk: bool = Field(
        default=True,
        description=(
            "True nếu chunk có tri thức y khoa thuộc schema; False nếu chunk chỉ là "
            "quảng bá, hành chính, hotline, đặt lịch, bài liên quan, nguồn tham khảo, "
            "thông tin bệnh viện/bác sĩ/app/dịch vụ."
        ),
    )
    skip_reason: str | None = Field(
        default=None,
        description=(
            "Lý do skip khi is_medical_knowledge_chunk=false. "
            "Ví dụ: administrative_or_promotional_content."
        ),
    )
    entities: list[LLMExtractedEntity] = Field(default_factory=list)
    relations: list[LLMExtractedRelation] = Field(
        default_factory=list,
        description="Binary relations. Use for single subject-object facts.",
    )
    relation_groups: list[LLMExtractedRelationGroup] = Field(
        default_factory=list,
        description=(
            "Grouped 1-vs-many or many-vs-1 relations. Do not use many-vs-many. "
            "The pipeline expands valid groups into binary triples."
        ),
    )

    def to_internal_graph(self) -> "ExtractedGraph":
        if not self.is_medical_knowledge_chunk:
            return ExtractedGraph(
                is_medical_knowledge_chunk=False,
                should_ingest_graph=False,
                skip_reason=self.skip_reason or "non_medical_knowledge_chunk",
                entities=[],
                relations=[],
            )

        internal_relations: list[ExtractedRelation] = []

        for relation in self.relations:
            internal_relations.append(
                _normalize_relation_direction(
                    ExtractedRelation(
                        subject=relation.subject,
                        subject_type=relation.subject_type,
                        relation_type=relation.relation_type,
                        object=relation.object,
                        object_type=relation.object_type,
                        description=relation.description,
                        keywords=relation.keywords,
                        evidence_text=relation.evidence_text,
                        confidence=relation.confidence,
                    )
                )
            )

        for group in self.relation_groups:
            internal_relations.extend(_expand_relation_group(group))

        return ExtractedGraph(
            is_medical_knowledge_chunk=True,
            should_ingest_graph=True,
            skip_reason=None,
            entities=[
                ExtractedEntity(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    aliases=[],
                    surface_forms=[],
                    description=entity.description,
                    evidence_text=entity.evidence_text,
                    is_anchor_candidate=entity.is_anchor_candidate,
                    anchor_reason=entity.anchor_reason,
                )
                for entity in self.entities
            ],
            relations=_dedupe_relations(internal_relations),
        )


class ExtractedEntity(BaseModel):
    """
    Internal extracted entity.

    aliases/surface_forms are internal bookkeeping fields.
    They are not exposed to the LLM.
    """

    name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    surface_forms: list[str] = Field(default_factory=list)
    description: str = ""
    evidence_text: str
    is_anchor_candidate: bool = False
    anchor_reason: str = ""


class ExtractedRelation(BaseModel):
    """
    Internal extracted relation/fact.
    """

    subject: str
    subject_type: str
    relation_type: str

    object: str
    object_type: str

    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    evidence_text: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractedGraph(BaseModel):
    """
    Internal extracted graph after converting from LLM output.
    """

    is_medical_knowledge_chunk: bool = True
    should_ingest_graph: bool = True
    skip_reason: str | None = None
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)