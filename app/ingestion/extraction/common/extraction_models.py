from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.domain.medical_schema import should_flip_relation_direction


class LLMExtractedEntity(BaseModel):
    """
    Entity schema exposed to the entity-extraction LLM only.

    Scope:
    - Extract entities and local mention variants.
    - Do not create relations.
    - Do not create retrieval keywords.
    - Do not infer synonyms from outside medical knowledge.
    """

    name: str = Field(
        description="Tên entity y khoa ngắn gọn, tự nhiên nhất trong chunk/title/heading."
    )
    entity_type: str = Field(
        description="Một trong các EntityType đã cho."
    )
    mentions: list[str] = Field(
        default_factory=list,
        description=(
            "Các surface forms/biến thể cách viết xuất hiện trực tiếp trong chunk/title/heading "
            "cho cùng entity này. Chỉ lấy biến thể có trong text."
        ),
    )
    explicit_aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Tên gọi khác/viết tắt được text nêu rõ là tương đương với entity này "
            "qua tín hiệu như 'còn gọi là', 'hay còn gọi là', 'viết tắt là'. "
            "Không suy từ kiến thức ngoài."
        ),
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


class LLMExplicitSynonymPair(BaseModel):
    """
    Explicit local synonym/alias pair detected by LLM #1.

    This is still entity-level output, not a relation-extraction task:
    the entity stage may only report equivalence that is directly stated
    inside the current title/heading/chunk.
    """

    name_a: str = Field(description="Tên entity/mention thứ nhất, đúng như trong text.")
    name_b: str = Field(description="Tên entity/mention thứ hai, đúng như trong text.")
    entity_type: str = Field(
        description=(
            "Entity type chung của cả hai tên. Chỉ dùng một type vì synonym/alias "
            "phải là hai tên của cùng một loại entity."
        )
    )
    evidence_text: str = Field(
        description=(
            "Evidence ngắn thể hiện rõ hai tên là tương đương/tên gọi khác/viết tắt. "
            "Ví dụ chứa 'còn gọi là', 'hay còn gọi là', 'viết tắt là', 'tên khoa học là'."
        )
    )
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _accept_previous_type_fields(cls, data):
        """Allow old cached/manual payloads that used type_a/type_b during transition."""

        if not isinstance(data, dict):
            return data

        if data.get("entity_type"):
            return data

        type_a = data.get("type_a")
        type_b = data.get("type_b")
        if type_a and type_b and type_a != type_b:
            # Entity-level synonym pairs should not cross entity types.
            # Prefer type_a only as a transition fallback; validator/prompt will
            # prevent new payloads from using this shape.
            data["entity_type"] = type_a
        elif type_a or type_b:
            data["entity_type"] = type_a or type_b

        return data


class LLMEntityExtractionResult(BaseModel):
    """Output of LLM #1: entity extraction + local explicit synonym detection only."""

    is_noise_chunk: bool = Field(
        default=False,
        description=(
            "True chỉ khi chunk gần như toàn quảng bá/hành chính/hotline/đặt lịch/"
            "navigation/bài liên quan/metadata, không có nội dung y khoa để extract."
        ),
    )
    skip_reason: str | None = Field(
        default=None,
        description="Lý do skip ngắn nếu is_noise_chunk=true.",
    )
    entities: list[LLMExtractedEntity] = Field(default_factory=list)
    explicit_synonym_pairs: list[LLMExplicitSynonymPair] = Field(default_factory=list)

    def to_internal_entities(self) -> list["ExtractedEntity"]:
        if self.is_noise_chunk:
            return []

        return [
            ExtractedEntity(
                name=entity.name,
                entity_type=entity.entity_type,
                aliases=list(entity.explicit_aliases or []),
                surface_forms=list(entity.mentions or []),
                description=entity.description,
                evidence_text=entity.evidence_text,
                is_anchor_candidate=entity.is_anchor_candidate,
                anchor_reason=entity.anchor_reason,
            )
            for entity in self.entities
        ]

    def to_synonym_relations(self) -> list["ExtractedRelation"]:
        if self.is_noise_chunk:
            return []

        relations: list[ExtractedRelation] = []
        for pair in self.explicit_synonym_pairs or []:
            if not pair.name_a or not pair.name_b:
                continue
            relations.append(
                ExtractedRelation(
                    subject=pair.name_a,
                    subject_type=pair.entity_type,
                    relation_type="DONG_NGHIA_VOI",
                    object=pair.name_b,
                    object_type=pair.entity_type,
                    description="Tên gọi/viết tắt đồng nghĩa được nêu trực tiếp trong text.",
                    keywords=[],
                    evidence_text=pair.evidence_text,
                    confidence=pair.confidence,
                )
            )
        return relations



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
        description="Để [] trong extraction; keywords được sinh bởi ArticleKeywordProfiler sau validation.",
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
        description="Để [] trong extraction; keywords được sinh bởi ArticleKeywordProfiler sau validation.",
    )
    evidence_text: str = Field(
        description="Đoạn bằng chứng ngắn từ chunk chứng minh relation group."
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class LLMRelationCandidate(BaseModel):
    """Output relation of LLM #2. No keyword field on purpose."""

    subject: str = Field(description="Tên subject entity, ưu tiên dùng name từ entity list đầu vào.")
    subject_type: str = Field(description="Entity type của subject.")
    relation_type: str = Field(description="Một trong các RelationType đã cho.")
    object: str = Field(description="Tên object entity, ưu tiên dùng name từ entity list đầu vào.")
    object_type: str = Field(description="Entity type của object.")
    description: str = Field(
        default="",
        description="Mô tả ngắn về relation dựa trên chunk. Để rỗng nếu không cần.",
    )
    evidence_text: str = Field(description="Đoạn evidence ngắn từ chunk/heading hỗ trợ relation.")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class LLMRelationGroupCandidate(BaseModel):
    """Grouped 1-vs-many or many-vs-1 relation of LLM #2. No keyword field."""

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
    evidence_text: str = Field(description="Đoạn evidence ngắn từ chunk/heading hỗ trợ relation group.")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class LLMRelationExtractionResult(BaseModel):
    """Output of LLM #2: relation extraction only."""

    relations: list[LLMRelationCandidate] = Field(
        default_factory=list,
        description="Binary relations. Use for single subject-object facts.",
    )
    relation_groups: list[LLMRelationGroupCandidate] = Field(
        default_factory=list,
        description=(
            "Grouped 1-vs-many or many-vs-1 relations. Do not use many-vs-many. "
            "The pipeline expands valid groups into binary triples."
        ),
    )

    def to_internal_relations(self) -> list["ExtractedRelation"]:
        internal: list[ExtractedRelation] = []

        for relation in self.relations or []:
            internal.append(
                _normalize_relation_direction(
                    ExtractedRelation(
                        subject=relation.subject,
                        subject_type=relation.subject_type,
                        relation_type=relation.relation_type,
                        object=relation.object,
                        object_type=relation.object_type,
                        description=relation.description,
                        keywords=[],
                        evidence_text=relation.evidence_text,
                        confidence=relation.confidence,
                    )
                )
            )

        for group in self.relation_groups or []:
            subjects = list(group.subjects or [])
            objects = list(group.objects or [])
            if not subjects or not objects:
                continue
            if len(subjects) > 1 and len(objects) > 1:
                continue
            for subject in subjects:
                for object_ in objects:
                    internal.append(
                        _normalize_relation_direction(
                            ExtractedRelation(
                                subject=subject.name,
                                subject_type=subject.entity_type,
                                relation_type=group.relation_type,
                                object=object_.name,
                                object_type=object_.entity_type,
                                description=group.description,
                                keywords=[],
                                evidence_text=group.evidence_text,
                                confidence=group.confidence,
                            )
                        )
                    )

        return _dedupe_relations(internal)



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
                    aliases=list(getattr(entity, "explicit_aliases", []) or []),
                    surface_forms=list(getattr(entity, "mentions", []) or []),
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