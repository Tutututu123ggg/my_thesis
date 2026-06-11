from pydantic import BaseModel, Field


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


class LLMExtractedRelation(BaseModel):
    """
    Relation schema exposed to LLM.
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
        description="4-8 LightRAG-style keywords, gồm cả low-level/local và high-level/global nếu evidence đủ thông tin.",
    )
    evidence_text: str = Field(
        description="Đoạn bằng chứng ngắn từ chunk chứng minh relation."
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class LLMExtractedGraph(BaseModel):
    """
    Raw graph returned by LLM.

    This is intentionally not the same as ExtractedGraph.
    The internal graph can contain extra bookkeeping fields, but LLM should not see them.
    """

    entities: list[LLMExtractedEntity] = Field(default_factory=list)
    relations: list[LLMExtractedRelation] = Field(default_factory=list)

    def to_internal_graph(self) -> "ExtractedGraph":
        return ExtractedGraph(
            entities=[
                ExtractedEntity(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    aliases=[],
                    surface_forms=[],
                    description=entity.description,
                    evidence_text=entity.evidence_text,
                )
                for entity in self.entities
            ],
            relations=[
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
                for relation in self.relations
            ],
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

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)