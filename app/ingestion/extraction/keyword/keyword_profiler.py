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

    This is intentionally small. The preferred source is the article-level LLM
    keyword profiler. When that profiler is disabled or returns nothing, keep
    only endpoint names and a short natural-language relation intent. Avoid
    mining arbitrary n-grams from evidence because those produced noisy keyword
    strings in graph audits.
    """

    intent = get_relation_keyword_intent(relation.relation_type, default="")
    candidates: list[str] = [
        relation.subject,
        relation.object,
        intent,
    ]

    # Anchor the intent with the main subject when it remains short. This keeps
    # fallback keywords retrieval-friendly without a hand-written medical dict.
    if intent and relation.subject:
        anchored = f"{intent} {relation.subject}"
        if len(anchored.split()) <= MAX_KEYWORD_WORDS:
            candidates.append(anchored)

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
- Keyword là nhãn truy hồi ngắn cho relation, không phải evidence.
- Keyword có thể generic vừa đủ nếu giúp retrieval, ví dụ: "da liễu", "nấm da", "bệnh móng", "thuốc kháng nấm".
- Keyword cũng nên có cụm specific theo subject/object, ví dụ: "điều trị lang ben", "soi tươi KOH", "itraconazole".
- Mỗi relation nên có 4-8 keywords.

Quy tắc format bắt buộc:
- Mỗi phần tử trong keywords là MỘT keyword riêng.
- Không nhét nhiều keyword vào một string bằng dấu phẩy.
- Mỗi keyword thường tối đa 5 từ; tên bệnh/thuốc/phương pháp dài có thể tối đa 6 từ.
- Không copy nguyên câu evidence, không sinh mảnh câu dài.
- Không dùng schema enum như BENH_LY, CO_BIEU_HIEN, DIEU_TRI_BANG.
- Không thêm fact y khoa mới xa evidence.
- Không trả markdown, không giải thích.

Few-shot tốt/xấu:

Ví dụ 1
Relation:
subject="nấm lang ben", relation_type="DIEU_TRI_BANG", object="itraconazole"
evidence="Chỉ định uống các thuốc kháng nấm như Gricin, ketoconazole, itraconazole, fluconazole..."
keywords tốt:
["da liễu", "nấm da", "lang ben", "điều trị lang ben", "thuốc kháng nấm", "itraconazole", "thuốc đường uống"]
keywords xấu:
["itraconazole sporanox onmel dạng", "chỉ định uống các thuốc kháng nấm như", "người bệnh sử dụng thuốc theo chỉ định"]

Ví dụ 2
Relation:
subject="nấm da đầu", relation_type="PHONG_NGUA_BANG", object="lau khô hoặc sấy khô đầu sau khi gội"
evidence="Việc lau khô hoặc sấy khô đầu sau khi gội là rất quan trọng để hạn chế sự ẩm ướt trên da đầu..."
keywords tốt:
["da liễu", "nấm da đầu", "phòng ngừa nấm da đầu", "vệ sinh da đầu", "giữ da đầu khô", "lau khô sau gội"]
keywords xấu:
["tạo điều kiện thuận lợi cho vi khuẩn và nấm phát triển", "việc lau khô hoặc sấy khô đầu sau khi gội là rất quan trọng"]

Ví dụ 3
Relation:
subject="nấm móng tay", relation_type="LIEN_QUAN_YEU_TO", object="vẩy nến"
evidence="Da xung quanh hoặc móng tay có vết tổn thương nhỏ hoặc đang mắc bệnh về da như vẩy nến."
keywords tốt:
["da liễu", "nấm móng", "bệnh móng", "vẩy nến", "yếu tố nguy cơ", "bệnh da liên quan"]
keywords xấu:
["da xung quanh hoặc móng tay có vết tổn thương nhỏ", "đang mắc bệnh về da như"]

Ví dụ 4
Relation:
subject="lang ben", relation_type="CHAN_DOAN_BANG", object="đèn Wood"
evidence="Đèn Wood sử dụng ánh sáng tia cực tím khi chiếu lên da sẽ xuất hiện màu vàng xanh nếu vùng đó mắc bệnh lang ben."
keywords tốt:
["da liễu", "lang ben", "chẩn đoán lang ben", "đèn Wood", "tia cực tím", "huỳnh quang vàng xanh"]
keywords xấu:
["đèn Wood sử dụng ánh sáng tia cực tím khi chiếu lên da", "nếu vùng đó mắc bệnh"]

Luôn ưu tiên keyword sạch, ngắn, truy hồi được. Keyword hơi generic nhưng sạch thì tốt hơn keyword dài bẩn.
""".strip()
