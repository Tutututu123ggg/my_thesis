from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.domain.medical_schema import get_relation_keyword_intent
from app.ingestion.extraction.common.extraction_models import ExtractedGraph, ExtractedRelation
from app.ingestion.processing import ChunkDocument


MAX_KEYWORD_WORDS = 5
MAX_KEYWORD_CHARS = 80


RELATION_INTENT_EXTRAS: dict[str, list[str]] = {
    "CO_BIEU_HIEN": ["triệu chứng", "biểu hiện"],
    "LIEN_QUAN_YEU_TO": ["yếu tố nguy cơ", "nguyên nhân", "cơ chế"],
    "GAY_BIEN_CHUNG": ["biến chứng"],
    "ANH_HUONG_DEN": ["vị trí ảnh hưởng", "đối tượng ảnh hưởng"],
    "CHAN_DOAN_BANG": ["chẩn đoán", "xét nghiệm"],
    "DIEU_TRI_BANG": ["điều trị", "thuốc điều trị"],
    "PHONG_NGUA_BANG": ["phòng ngừa", "giảm nguy cơ"],
    "CAN_TRANH": ["cần tránh"],
    "CHONG_CHI_DINH": ["chống chỉ định"],
    "TUONG_TAC_VOI": ["tương tác thuốc"],
    "DONG_NGHIA_VOI": ["tên gọi khác"],
    "LA_DANG_CUA": ["phân loại", "dạng bệnh"],
    "CHAN_DOAN_PHAN_BIET_VOI": ["chẩn đoán phân biệt"],
    "LIEN_QUAN_TRUC_TIEP_DEN": ["liên quan trực tiếp"],
}


BAD_KEYWORD_FRAGMENTS = {
    "nếu ",
    "neu ",
    "có thể",
    "co the",
    "không có",
    "khong co",
    "người bệnh",
    "nguoi benh",
    "bệnh nhân",
    "benh nhan",
    "được sử dụng",
    "duoc su dung",
    "các lựa chọn",
    "cac lua chon",
    "bao gồm",
    "bao gom",
    "chúng bao gồm",
    "chung bao gom",
    "không hiệu quả",
    "khong hieu qua",
}


@dataclass(frozen=True)
class RelationRef:
    """Stable local reference used only inside one article keyword pass."""

    relation_id: str
    chunk_index: int
    relation_index: int
    chunk_id: str


class LLMRelationKeywordUpdate(BaseModel):
    relation_id: str = Field(description="Relation id from the input pack, e.g. r0001.")
    keywords: list[str] = Field(default_factory=list)


class LLMEntityKeywordUpdate(BaseModel):
    entity_id: str = Field(description="Entity id from the input pack, e.g. e0001.")
    search_terms: list[str] = Field(default_factory=list)


class LLMArticleKeywordProfile(BaseModel):
    relation_keyword_updates: list[LLMRelationKeywordUpdate] = Field(default_factory=list)
    entity_keyword_updates: list[LLMEntityKeywordUpdate] = Field(default_factory=list)


def normalize_for_keyword_match(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_keyword_text(value: str) -> str:
    value = str(value or "")
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip(" -–—:;,.()[]{}\n\t")
    return value.strip()


def is_bad_retrieval_keyword(value: str) -> bool:
    text = clean_keyword_text(value)
    if not text:
        return True
    if len(text) > MAX_KEYWORD_CHARS:
        return True
    if len(text.split()) > MAX_KEYWORD_WORDS:
        return True

    norm = normalize_for_keyword_match(text)
    if not norm or len(norm) < 2:
        return True

    # Relation keywords should be atomic keys, not comma-separated copied clauses.
    if "," in text or ";" in text:
        return True

    if any(fragment in norm for fragment in BAD_KEYWORD_FRAGMENTS):
        return True

    # Drop obvious sentence fragments rather than medical terms.
    sentence_like_tokens = {"va", "hoac", "nhung", "tuy", "dong thoi", "ngoai ra"}
    if len(norm.split()) >= 4 and any(tok in norm for tok in sentence_like_tokens):
        return True

    return False


def clean_retrieval_keywords(values: list[str] | tuple[str, ...] | None, *, limit: int = 8) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values or []:
        text = clean_keyword_text(str(value))
        if is_bad_retrieval_keyword(text):
            continue
        key = normalize_for_keyword_match(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break

    return result


def deterministic_relation_keywords(relation: ExtractedRelation, *, extra_keywords: list[str] | None = None) -> list[str]:
    """Safe fallback for relation retrieval keywords.

    The extractor/LLM is no longer trusted to create final LightRAG relation
    keys. This fallback uses only relation endpoints and schema-level intent;
    it never mines arbitrary n-grams from evidence, which prevents clause
    fragments such as "neu griseofulvin khong" from leaking into Neo4j/Qdrant.
    """

    intent = get_relation_keyword_intent(relation.relation_type, default="")
    candidates: list[str] = [
        relation.subject,
        relation.object,
        intent,
        *RELATION_INTENT_EXTRAS.get(relation.relation_type, []),
    ]

    if relation.relation_type == "DIEU_TRI_BANG":
        if relation.object_type == "THUOC_VA_HOAT_CHAT":
            candidates.append("thuốc điều trị")
            candidates.append("thuốc y khoa")
        if relation.object_type == "CAN_THIEP_Y_TE":
            candidates.append("can thiệp điều trị")
    elif relation.relation_type == "LIEN_QUAN_YEU_TO":
        if relation.object_type == "YEU_TO_BENH_SINH":
            candidates.append("cơ chế bệnh sinh")
        if relation.object_type == "YEU_TO_NGOAI_SINH":
            candidates.append("yếu tố môi trường")
    elif relation.relation_type == "CO_BIEU_HIEN":
        candidates.append(f"triệu chứng {relation.subject}")
    elif relation.relation_type == "CHAN_DOAN_BANG":
        candidates.append(f"chẩn đoán {relation.subject}")

    candidates.extend(extra_keywords or [])
    return clean_retrieval_keywords(candidates, limit=8)


class ArticleKeywordProfiler:
    """Article-level Agent 2 for retrieval keywords only.

    Scope boundary:
    - does NOT create entities;
    - does NOT create relations;
    - does NOT create synonyms;
    - does NOT change subject/object/relation_type/evidence;
    - only updates relation.keywords and optionally entity search terms metadata.

    If no model is configured, the profiler still applies deterministic safe
    relation keywords. If a model is configured, it receives a compact article
    pack: title/headings/short summary + validated entities/relations, not raw
    full markdown.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_relations_per_call: int = 180,
    ):
        self.model = (model or os.getenv("OPENAI_KEYWORD_MODEL") or "").strip()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.max_relations_per_call = max(1, int(max_relations_per_call))

    def enrich_graphs(
        self,
        *,
        title: str,
        chunks: list[ChunkDocument],
        graphs: list[ExtractedGraph],
    ) -> dict[str, Any]:
        relation_refs = self._relation_refs(chunks, graphs)
        llm_updates: dict[str, list[str]] = {}
        llm_error = ""

        if self.model and relation_refs:
            try:
                llm_updates = self._profile_with_llm(title=title, chunks=chunks, graphs=graphs, refs=relation_refs)
            except Exception as exc:  # fail-open: deterministic keywords still apply
                llm_error = f"{type(exc).__name__}: {exc}"
                llm_updates = {}

        updated_count = 0
        for ref in relation_refs:
            relation = graphs[ref.chunk_index].relations[ref.relation_index]
            merged = deterministic_relation_keywords(
                relation,
                extra_keywords=[*(relation.keywords or []), *llm_updates.get(ref.relation_id, [])],
            )
            graphs[ref.chunk_index].relations[ref.relation_index] = relation.model_copy(update={"keywords": merged})
            updated_count += 1

        return {
            "enabled": True,
            "model": self.model or "deterministic_only",
            "relation_count": len(relation_refs),
            "updated_relation_count": updated_count,
            "llm_update_count": len(llm_updates),
            "llm_error": llm_error,
            "policy": "article_level_relation_keyword_metadata_only_no_synonyms",
        }

    def _relation_refs(self, chunks: list[ChunkDocument], graphs: list[ExtractedGraph]) -> list[RelationRef]:
        refs: list[RelationRef] = []
        counter = 1
        for chunk_index, (chunk, graph) in enumerate(zip(chunks, graphs, strict=True)):
            if not graph.should_ingest_graph:
                continue
            for relation_index, _relation in enumerate(graph.relations):
                refs.append(
                    RelationRef(
                        relation_id=f"r{counter:04d}",
                        chunk_index=chunk_index,
                        relation_index=relation_index,
                        chunk_id=chunk.chunk_id,
                    )
                )
                counter += 1
        return refs

    def _profile_with_llm(
        self,
        *,
        title: str,
        chunks: list[ChunkDocument],
        graphs: list[ExtractedGraph],
        refs: list[RelationRef],
    ) -> dict[str, list[str]]:
        from openai import OpenAI

        refs = refs[: self.max_relations_per_call]
        pack = self._build_article_pack(title=title, chunks=chunks, graphs=graphs, refs=refs)
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ARTICLE_KEYWORD_PROFILER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(pack, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        profile = LLMArticleKeywordProfile.model_validate(parsed)

        valid_ids = {ref.relation_id for ref in refs}
        updates: dict[str, list[str]] = {}
        for item in profile.relation_keyword_updates:
            if item.relation_id not in valid_ids:
                continue
            keywords = clean_retrieval_keywords(item.keywords, limit=8)
            if keywords:
                updates[item.relation_id] = keywords
        return updates

    def _build_article_pack(
        self,
        *,
        title: str,
        chunks: list[ChunkDocument],
        graphs: list[ExtractedGraph],
        refs: list[RelationRef],
    ) -> dict[str, Any]:
        section_titles: list[str] = []
        seen_sections: set[str] = set()
        for chunk in chunks:
            for value in [chunk.section, chunk.subsection, *(chunk.heading_path or [])]:
                text = clean_keyword_text(str(value or ""))
                if not text:
                    continue
                key = normalize_for_keyword_match(text)
                if key and key not in seen_sections:
                    seen_sections.add(key)
                    section_titles.append(text)

        entity_items: list[dict[str, str]] = []
        seen_entities: set[tuple[str, str]] = set()
        entity_counter = 1
        for graph in graphs:
            for entity in graph.entities:
                key = (normalize_for_keyword_match(entity.name), entity.entity_type)
                if key in seen_entities:
                    continue
                seen_entities.add(key)
                entity_items.append(
                    {"id": f"e{entity_counter:04d}", "name": entity.name, "entity_type": entity.entity_type}
                )
                entity_counter += 1

        relations: list[dict[str, Any]] = []
        for ref in refs:
            relation = graphs[ref.chunk_index].relations[ref.relation_index]
            chunk = chunks[ref.chunk_index]
            relations.append(
                {
                    "id": ref.relation_id,
                    "chunk_id": ref.chunk_id,
                    "section": chunk.section or "",
                    "subsection": chunk.subsection or "",
                    "subject": relation.subject,
                    "subject_type": relation.subject_type,
                    "relation_type": relation.relation_type,
                    "object": relation.object,
                    "object_type": relation.object_type,
                    "description": relation.description or "",
                    "evidence_text": relation.evidence_text or "",
                }
            )

        return {
            "article": {
                "title": title,
                "section_titles": section_titles[:40],
                "short_summary": self._short_summary(title, section_titles),
            },
            "entities": entity_items[:250],
            "relations": relations,
        }

    def _short_summary(self, title: str, section_titles: list[str]) -> str:
        sections = "; ".join(section_titles[:8])
        if sections:
            return f"Bài viết '{title}' gồm các phần: {sections}."
        return f"Bài viết y khoa: {title}."


ARTICLE_KEYWORD_PROFILER_SYSTEM_PROMPT = """
Bạn là Retrieval Keyword Profiler cho Vietnamese Medical GraphRAG.

Vai trò của bạn CHỈ là enrich metadata keyword phục vụ retrieval.
Bạn KHÔNG được tạo entity mới, KHÔNG tạo relation mới, KHÔNG tạo synonym,
KHÔNG đổi subject/object/relation_type/evidence.

Input gồm title, section_titles, short_summary, entities và validated relations của một bài.

Output JSON duy nhất:
{
  "relation_keyword_updates": [
    {"relation_id": "r0001", "keywords": ["..."]}
  ],
  "entity_keyword_updates": []
}

Mục tiêu keyword:
- Keyword dùng để tìm lại relation khi user hỏi bằng cách diễn đạt khác.
- Keyword phải là cụm y khoa/ngữ nghĩa truy hồi, không phải câu văn copy từ evidence.
- Ưu tiên keyword giúp nối query với subject, object, relation intent, cơ chế, vị trí, mức độ, nhóm thuốc, phương pháp.

Quy tắc:
- Mỗi keyword tối đa 5 từ.
- Mỗi relation nên có 3-8 keywords.
- Không sinh câu hoàn chỉnh hoặc mảnh câu điều kiện.
- Không copy cụm như: "nếu", "có thể", "không có", "người bệnh", "được sử dụng", "bao gồm".
- Không dùng schema enum như BENH_LY, CO_BIEU_HIEN, DIEU_TRI_BANG.
- Không thêm fact y khoa mới xa evidence.
- Không trả markdown, không giải thích.

Ví dụ tốt:

1. Relation:
subject="nấm lang ben", relation_type="DIEU_TRI_BANG", object="ketoconazole"
evidence="Kết hợp với bôi kem trị nấm như ketoconazole, Clotrimazole, Terbinafine."
keywords tốt:
["kem bôi trị nấm", "điều trị tại chỗ", "thuốc chống nấm", "azole", "điều trị lang ben"]

2. Relation:
subject="lác đồng tiền", relation_type="CO_BIEU_HIEN", object="ngứa dữ dội"
evidence="Lác đồng tiền có thể khiến người bệnh ngứa dữ dội, nhất là khi ra mồ hôi."
keywords tốt:
["ngứa dữ dội", "triệu chứng hắc lào", "ngứa khi ra mồ hôi", "biểu hiện nấm da"]

3. Relation:
subject="ngứa dữ dội", relation_type="DIEU_TRI_BANG", object="thuốc kháng Histamine"
evidence="Thuốc kháng Histamine giúp cải thiện tình trạng ngứa dữ dội."
keywords tốt:
["giảm ngứa", "kháng Histamine", "kiểm soát ngứa", "điều trị triệu chứng"]

4. Relation:
subject="nấm da đầu", relation_type="LIEN_QUAN_YEU_TO", object="Trichophyton"
evidence="Các loại nấm thuộc nhóm Dermatophytes, đặc biệt là chi Trichophyton, Microsporum và Epidermophyton, được xem là nguyên nhân chính gây ra bệnh nấm da đầu."
keywords tốt:
["nấm Dermatophytes", "nguyên nhân nấm da đầu", "vi nấm gây bệnh", "Trichophyton"]

5. Relation:
subject="lang ben", relation_type="CHAN_DOAN_BANG", object="đèn Wood"
evidence="Đèn Wood sử dụng ánh sáng tia cực tím khi chiếu lên da sẽ xuất hiện màu vàng xanh nếu vùng đó mắc bệnh lang ben."
keywords tốt:
["đèn Wood", "chẩn đoán lang ben", "ánh sáng tia cực tím", "huỳnh quang vàng xanh"]

6. Relation:
subject="hắc lào", relation_type="CAN_TRANH", object="corticoid"
evidence="Không tự ý dùng corticoid vì có thể làm bệnh nặng thêm."
keywords tốt:
["tránh corticoid", "lạm dụng corticoid", "bệnh nặng thêm", "điều trị sai cách"]

7. Relation:
subject="nấm", relation_type="LIEN_QUAN_TRUC_TIEP_DEN", object="thời tiết nóng ẩm"
evidence="Môi trường có độ ẩm cao, thời tiết nóng ẩm là điều kiện thuận lợi cho nấm phát triển."
keywords tốt:
["nấm phát triển", "môi trường ẩm", "thời tiết nóng ẩm", "điều kiện thuận lợi"]

Ví dụ xấu, không được sinh:
- "nếu griseofulvin không"
- "có thể được"
- "chúng bao gồm"
- "người bệnh cần"
- "các lựa chọn thay thế"
- "sử dụng theo chỉ định"
""".strip()