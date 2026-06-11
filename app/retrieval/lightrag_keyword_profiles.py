from __future__ import annotations

import json
import re
from typing import Iterable

from pydantic import BaseModel, Field


class EntityLightRAGProfile(BaseModel):
    """Index-time entity profile used by LightRAG-style entity retrieval."""

    profile_text: str = Field(default="")
    local_keys: list[str] = Field(default_factory=list)
    global_keys: list[str] = Field(default_factory=list)


class RelationLightRAGProfile(BaseModel):
    """Index-time relation/global keywords used by LightRAG-style relation retrieval."""

    keywords: list[str] = Field(default_factory=list)


class KeywordProfileGenerator:
    """Generate/index LightRAG-style keyword profiles for graph items.

    The goal is to stay close to LightRAG's keyword philosophy:
    - low-level/local keywords capture concrete entities, names, surface forms,
      attributes, and specific details;
    - high-level/global keywords capture broader concepts, topics, themes, and
      relationship intent.

    We keep the generation lightweight. The LLM may suggest natural-language
    keywords, but profile_text is built deterministically from existing graph
    evidence so that regeneration cannot introduce unsupported medical facts.
    """

    def __init__(
        self,
        llm_client: object | None = None,
        max_entity_local_keys: int = 6,
        max_entity_global_keys: int = 5,
        max_relation_keywords: int = 8,
    ):
        self.llm_client = llm_client
        self.max_entity_local_keys = max_entity_local_keys
        self.max_entity_global_keys = max_entity_global_keys
        self.max_relation_keywords = max_relation_keywords

    # =====================
    # Public API
    # =====================

    def generate_entity_profile(
        self,
        *,
        name: str,
        entity_type: str,
        description: str | None = None,
        surface_forms: list[str] | None = None,
        evidence_texts: list[str] | None = None,
    ) -> EntityLightRAGProfile:
        fallback = self.fallback_entity_profile(
            name=name,
            entity_type=entity_type,
            description=description,
            surface_forms=surface_forms,
            evidence_texts=evidence_texts,
        )
        if self.llm_client is None:
            return fallback

        user_prompt = json.dumps(
            {
                "name": name,
                "entity_type": entity_type,
                "description": description or "",
                "surface_forms": surface_forms or [],
                "evidence_texts": evidence_texts or [],
            },
            ensure_ascii=False,
        )
        try:
            raw = self.llm_client.generate(ENTITY_PROFILE_SYSTEM_PROMPT, user_prompt)
            parsed = self._extract_json(raw)
            proposed = EntityLightRAGProfile.model_validate(parsed)
        except Exception:
            return fallback

        support_text = "\n".join([name, description or "", *(surface_forms or []), *(evidence_texts or [])])
        local_keys = self._clean_local_entity_keys(
            [*fallback.local_keys, *proposed.local_keys],
            limit=self.max_entity_local_keys,
            support_text=support_text,
        )
        # LightRAG high-level keys should be broader topics/themes, not another
        # synonym list. We allow natural-language themes proposed by the LLM,
        # but filter out local surface forms and extremely generic terms.
        # Prefer the LLM's natural LightRAG-style high-level keywords.
        # Deterministic fallback hints are only used when the LLM returns nothing;
        # otherwise they make many nodes look identical (e.g. every disease gets
        # the same generic global key).
        global_keys = self._clean_global_entity_keys(
            proposed.global_keys,
            limit=self.max_entity_global_keys,
            local_keys=local_keys,
        )
        if not global_keys:
            global_keys = fallback.global_keys
        return EntityLightRAGProfile(
            profile_text=self._build_entity_profile_text(
                name=name,
                entity_type=entity_type,
                description=description,
                local_keys=local_keys,
                global_keys=global_keys,
                evidence_texts=evidence_texts,
            ),
            local_keys=local_keys,
            global_keys=global_keys,
        )

    def generate_relation_profile(
        self,
        *,
        relation_type: str,
        subject_name: str,
        subject_type: str,
        object_name: str,
        object_type: str,
        description: str | None = None,
        evidence_text: str | None = None,
    ) -> RelationLightRAGProfile:
        fallback = self.fallback_relation_profile(
            relation_type=relation_type,
            subject_name=subject_name,
            subject_type=subject_type,
            object_name=object_name,
            object_type=object_type,
            description=description,
            evidence_text=evidence_text,
        )
        if self.llm_client is None:
            return fallback

        user_prompt = json.dumps(
            {
                "relation_type": relation_type,
                "subject": {"name": subject_name, "entity_type": subject_type},
                "object": {"name": object_name, "entity_type": object_type},
                "description": description or "",
                "evidence_text": evidence_text or "",
            },
            ensure_ascii=False,
        )
        try:
            raw = self.llm_client.generate(RELATION_PROFILE_SYSTEM_PROMPT, user_prompt)
            parsed = self._extract_json(raw)
            proposed = RelationLightRAGProfile.model_validate(parsed)
        except Exception:
            return fallback

        # Prefer LLM-generated low/high keyword mix. Fallback keywords are only
        # used when the LLM returns an empty/invalid list; otherwise relation_type
        # templates dominate and make many relations nearly identical.
        keywords = self._clean_relation_keywords(
            proposed.keywords,
            limit=self.max_relation_keywords,
            relation_type=relation_type,
            subject_name=subject_name,
            object_name=object_name,
        )
        return RelationLightRAGProfile(keywords=keywords or fallback.keywords)

    # =====================
    # Deterministic fallbacks
    # =====================

    def fallback_entity_profile(
        self,
        *,
        name: str,
        entity_type: str,
        description: str | None = None,
        surface_forms: list[str] | None = None,
        evidence_texts: list[str] | None = None,
    ) -> EntityLightRAGProfile:
        support_text = "\n".join([name, description or "", *(surface_forms or []), *(evidence_texts or [])])
        local = self._clean_local_entity_keys(
            [name, *(surface_forms or [])],
            self.max_entity_local_keys,
            support_text=support_text,
        )
        global_keys = self._clean_global_entity_keys(
            [
                *self._entity_type_global_keys(entity_type),
                *self._semantic_hints_from_text(" ".join([description or "", *(evidence_texts or [])])),
            ],
            self.max_entity_global_keys,
            local_keys=local,
        )
        return EntityLightRAGProfile(
            profile_text=self._build_entity_profile_text(
                name=name,
                entity_type=entity_type,
                description=description,
                local_keys=local,
                global_keys=global_keys,
                evidence_texts=evidence_texts,
            ),
            local_keys=local,
            global_keys=global_keys,
        )

    def fallback_relation_profile(
        self,
        *,
        relation_type: str,
        subject_name: str,
        subject_type: str,
        object_name: str,
        object_type: str,
        description: str | None = None,
        evidence_text: str | None = None,
    ) -> RelationLightRAGProfile:
        # Fallback is intentionally simple and mixed-level: endpoints are
        # low-level retrieval signals; the relation type and semantic hints are
        # high-level/global signals. In normal regeneration, the LLM output is
        # preferred and this fallback is only used when the LLM fails.
        keywords = [subject_name, object_name]
        keywords.extend(RELATION_TYPE_KEYWORDS.get(relation_type, [self._humanize_relation_type(relation_type)]))
        keywords.extend(
            self._semantic_hints_from_relation(
                relation_type=relation_type,
                subject_name=subject_name,
                subject_type=subject_type,
                object_name=object_name,
                object_type=object_type,
                text=" ".join([description or "", evidence_text or ""]),
            )
        )
        return RelationLightRAGProfile(
            keywords=self._clean_relation_keywords(
                keywords,
                self.max_relation_keywords,
                relation_type=relation_type,
                subject_name=subject_name,
                object_name=object_name,
            )
        )

    # =====================
    # Helpers
    # =====================

    def _build_entity_profile_text(
        self,
        *,
        name: str,
        entity_type: str,
        description: str | None,
        local_keys: list[str],
        global_keys: list[str],
        evidence_texts: list[str] | None,
    ) -> str:
        parts = [f"Tên thực thể: {self._clean_text(name)}", f"Loại thực thể: {entity_type}"]
        if self._clean_text(description or ""):
            parts.append(f"Mô tả: {self._clean_text(description or '')}")
        if local_keys:
            parts.append("Local keys: " + "; ".join(local_keys))
        if global_keys:
            parts.append("Global keys: " + "; ".join(global_keys))
        evidence = [self._clean_text(x) for x in (evidence_texts or []) if self._clean_text(x)]
        # Do not append raw evidence to the entity profile. Evidence snippets can
        # be noisy because an entity may be mentioned in unrelated contexts; using
        # them directly in the profile can make entity vectors drift. Evidence is
        # still used as support text for validating local keys.
        return "\n".join(parts)

    def _entity_type_global_keys(self, entity_type: str) -> list[str]:
        return ENTITY_TYPE_GLOBAL_KEYS.get(entity_type, [entity_type.lower().replace("_", " ")])

    def _semantic_hints_from_relation(
        self,
        *,
        relation_type: str,
        subject_name: str,
        subject_type: str,
        object_name: str,
        object_type: str,
        text: str,
    ) -> list[str]:
        return self._semantic_hints_from_text(" ".join([subject_name, object_name, text]))

    def _semantic_hints_from_text(self, text: str) -> list[str]:
        normalized = self._normalize_ascii(text)
        hints: list[str] = []
        if any(x in normalized for x in ["kho khe", "kho tho", "ho ", "tuc nguc", "hen", "phoi", "phe quan"]):
            hints.append("triệu chứng hô hấp")
            hints.append("bệnh hô hấp")
        if any(x in normalized for x in ["ngua", "ban do", "da kho", "bong vay", "mun nuoc", "da dau", "khuu tay", "cham", "vay nen"]):
            hints.append("triệu chứng da")
            hints.append("bệnh da liễu")
        if any(x in normalized for x in ["bui", "phan hoa", "long thu", "mat bui", "di nguyen", "di ung", "hoa chat"]):
            hints.append("yếu tố dị ứng")
        if any(x in normalized for x in ["virus", "vi khuan", "nhiem trung", "nam"]):
            hints.append("nhiễm trùng")
        if any(x in normalized for x in ["tre so sinh", "tre em", "nguoi lon", "phu nu mang thai"]):
            hints.append("đối tượng nguy cơ")
        return hints

    def _humanize_relation_type(self, relation_type: str) -> str:
        return relation_type.lower().replace("_", " ")

    def _clean_keywords(
        self,
        values: Iterable[str],
        limit: int,
        fallback: list[str] | None = None,
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in values:
            value = self._clean_text(str(item))
            if not value:
                continue
            if len(value.split()) > 8:
                continue
            key = self._normalize_ascii(value)
            if not key or key in seen:
                continue
            if key in GENERIC_KEYWORDS:
                continue
            seen.add(key)
            result.append(value)
            if len(result) >= limit:
                break
        if not result and fallback:
            return self._clean_keywords(fallback, limit)
        return result

    def _clean_local_entity_keys(self, values: Iterable[str], limit: int, *, support_text: str) -> list[str]:
        support_norm = self._normalize_ascii(support_text)
        result: list[str] = []
        seen: set[str] = set()
        for item in values:
            value = self._clean_text(str(item))
            if not value or len(value.split()) > 8:
                continue
            key = self._normalize_ascii(value)
            if not key or key in seen or key in GENERIC_KEYWORDS:
                continue
            # Local keys must be concrete surface forms supported by existing
            # graph text. This blocks hallucinated keys like "xét nghiệm X".
            if key not in support_norm:
                continue
            seen.add(key)
            result.append(value)
            if len(result) >= limit:
                break
        return result

    def _clean_global_entity_keys(
        self,
        values: Iterable[str],
        limit: int,
        *,
        local_keys: Iterable[str] | None = None,
    ) -> list[str]:
        """Clean LightRAG high-level/entity-global keywords.

        Unlike the previous strict implementation, this is not limited to a
        closed vocabulary. LightRAG examples use natural themes such as
        "Environmental consequences" or "Global economic stability". For the
        medical graph, we keep broad themes such as "bệnh hô hấp",
        "triệu chứng da", "cơ địa dị ứng", etc., while filtering out terms that
        are merely low-level entity names/synonyms.
        """
        local_norms = {self._normalize_ascii(x) for x in (local_keys or []) if x}
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = self._clean_text(str(value))
            if not text or len(text.split()) > 8:
                continue
            key = self._normalize_ascii(text)
            if not key or key in seen or key in GENERIC_KEYWORDS:
                continue
            if key in local_norms:
                continue
            # Avoid using a local key wrapped in a trivial phrase as a global
            # theme, e.g. "viêm da cơ địa" or "bệnh viêm da cơ địa".
            if any(local and (key == local or key.endswith(" " + local)) for local in local_norms):
                continue
            seen.add(key)
            result.append(text)
            if len(result) >= limit:
                break
        return result

    def _clean_relation_keywords(
        self,
        values: Iterable[str],
        limit: int,
        *,
        relation_type: str,
        subject_name: str,
        object_name: str,
    ) -> list[str]:
        """Clean LightRAG relation keywords.

        LightRAG examples mix concrete low-level terms with broader high-level
        concepts. Therefore relation.keywords may include both endpoint-like
        specific terms and broad themes. We only remove empty/duplicate/very
        generic or sentence-like keywords; we do not force a closed vocabulary.
        """
        result: list[str] = []
        seen: set[str] = set()
        for item in values:
            value = self._clean_text(str(item))
            if not value or len(value.split()) > 8:
                continue
            key = self._normalize_ascii(value)
            if not key or key in seen or key in GENERIC_KEYWORDS:
                continue
            seen.add(key)
            result.append(value)
            if len(result) >= limit:
                break
        return result

    def _suggested_relation_keywords(self, relation_type: str) -> list[str]:
        return [
            *RELATION_TYPE_KEYWORDS.get(relation_type, []),
            *DOMAIN_THEME_KEYWORDS,
        ]

    def _clean_text(self, value: str) -> str:
        value = re.sub(r"\s+", " ", value or "").strip(" -–—:;,.()[]{}\n\t")
        return value.strip()

    def _normalize_ascii(self, text: str) -> str:
        import unicodedata

        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.replace("đ", "d")
        return re.sub(r"\s+", " ", text).strip()

    def _extract_json(self, raw: str) -> dict:
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))


ENTITY_TYPE_GLOBAL_KEYS = {
    "BENH_LY": ["bệnh và hội chứng"],
    "BIEU_HIEN_LAM_SANG": ["triệu chứng", "biểu hiện lâm sàng"],
    "YEU_TO_BENH_SINH": ["nguyên nhân", "cơ chế bệnh sinh", "yếu tố nguy cơ nội tại"],
    "YEU_TO_NGOAI_SINH": ["yếu tố môi trường", "yếu tố kích hoạt bên ngoài"],
    "PHUONG_PHAP_CHAN_DOAN": ["chẩn đoán", "xét nghiệm", "phương pháp chẩn đoán"],
    "CAN_THIEP_Y_TE": ["điều trị", "can thiệp y tế", "chăm sóc"],
    "THUOC_VA_HOAT_CHAT": ["thuốc", "hoạt chất", "điều trị bằng thuốc"],
    "VI_TRI_GIAI_PHAU": ["vị trí cơ thể", "vùng tổn thương", "giải phẫu"],
    "DOI_TUONG": ["nhóm bệnh nhân", "đối tượng nguy cơ"],
}

DOMAIN_THEME_KEYWORDS = [
    "triệu chứng hô hấp",
    "triệu chứng da",
    "bệnh hô hấp",
    "bệnh da liễu",
    "yếu tố dị ứng",
    "nhiễm trùng",
    "đối tượng nguy cơ",
]

RELATION_TYPE_KEYWORDS = {
    "LA_DANG_CUA": ["phân loại bệnh", "dạng bệnh", "thể bệnh"],
    "DONG_NGHIA_VOI": ["tên gọi khác", "đồng nghĩa", "thuật ngữ tương đương"],
    "CHAN_DOAN_PHAN_BIET_VOI": ["chẩn đoán phân biệt", "dễ nhầm với", "phân biệt bệnh"],
    "CO_BIEU_HIEN": ["triệu chứng", "biểu hiện lâm sàng", "gợi ý bệnh từ triệu chứng"],
    "LIEN_QUAN_YEU_TO": ["yếu tố liên quan", "yếu tố nguy cơ", "yếu tố kích hoạt"],
    "GAY_BIEN_CHUNG": ["biến chứng", "dẫn đến", "nguy cơ phát sinh bệnh"],
    "ANH_HUONG_DEN": ["vị trí ảnh hưởng", "vùng tổn thương", "đối tượng ảnh hưởng"],
    "CHAN_DOAN_BANG": ["chẩn đoán", "xét nghiệm", "phương pháp chẩn đoán"],
    "DIEU_TRI_BANG": ["điều trị", "kiểm soát bệnh", "can thiệp điều trị"],
    "PHONG_NGUA_BANG": ["phòng ngừa", "phòng tránh", "giảm nguy cơ"],
    "CAN_TRANH": ["cần tránh", "kiêng", "tránh yếu tố kích thích"],
    "CHONG_CHI_DINH": ["chống chỉ định", "không nên dùng", "cảnh báo điều trị"],
    "TUONG_TAC_VOI": ["tương tác", "tương tác thuốc", "tương tác hoạt chất"],
}

GENERIC_KEYWORDS = {
    "benh ly",
    "trieu chung benh",
    "van de",
    "tinh trang suc khoe",
    "trieu chung chung",
}


ENTITY_PROFILE_SYSTEM_PROMPT = """
---Role---
Bạn là expert tạo keyword profile cho LightRAG-style graph retrieval trong miền y khoa tiếng Việt.

---LightRAG Keyword Principle---
LightRAG dùng hai tầng keyword, giống các ví dụ kiểu biodiversity/environmental consequences:
- low-level/local keywords: entity cụ thể, tên gọi, thuộc tính/chi tiết cụ thể, thuật ngữ bề mặt.
- high-level/global keywords: chủ đề, khái niệm bao quát, theme rộng hơn của entity.

---Task---
Với ENTITY đã có trong graph, tạo:
1. profile_text: mô tả ngắn dựa trên input, không thêm fact mới.
2. local_keys: 2-5 low-level keywords nếu có đủ dữ liệu.
3. global_keys: 2-5 high-level keywords nếu có đủ dữ liệu.

---Rules---
- Chỉ dùng dữ liệu được cung cấp. Không thêm kiến thức y khoa mới.
- local_keys nên gồm name, surface_forms, viết tắt, tên tiếng Anh/Latin, synonym/tên tương đương nếu xuất hiện rõ trong input.
- Không đưa xét nghiệm/điều trị/biến chứng/triệu chứng/yếu tố nguy cơ vào local_keys, trừ khi chính chúng là tên của entity hiện tại.
- global_keys là theme rộng tự nhiên. Không cần chọn từ vocabulary cố định.
- global_keys không phải danh sách synonym và không lặp lại local_keys.
- Tránh keyword vô dụng/quá rộng như "bệnh lý", "vấn đề", "tình trạng sức khỏe".
- Trả JSON parseable, không markdown, không giải thích.

---Few-shot examples---
Example 1:
Input:
{
  "name": "Asthma",
  "entity_type": "BENH_LY",
  "description": "Tên tiếng Anh của bệnh hen suyễn.",
  "surface_forms": ["Asthma"],
  "evidence_texts": ["Asthma là tên tiếng Anh của bệnh hen suyễn."]
}
Output:
{
  "profile_text": "Asthma là tên tiếng Anh của bệnh hen suyễn.",
  "local_keys": ["Asthma", "hen suyễn"],
  "global_keys": ["bệnh hô hấp", "bệnh đường thở", "dị ứng hô hấp"]
}

Example 2:
Input:
{
  "name": "atopic dermatitis",
  "entity_type": "BENH_LY",
  "description": "Tên tiếng Anh của viêm da cơ địa.",
  "surface_forms": ["atopic dermatitis"],
  "evidence_texts": ["Viêm da cơ địa hay còn gọi là eczema hoặc atopic dermatitis."]
}
Output:
{
  "profile_text": "atopic dermatitis là tên tiếng Anh của viêm da cơ địa.",
  "local_keys": ["atopic dermatitis", "viêm da cơ địa", "eczema"],
  "global_keys": ["bệnh da liễu", "viêm da mạn tính", "cơ địa dị ứng"]
}

Example 3:
Input:
{
  "name": "thở khò khè",
  "entity_type": "BIEU_HIEN_LAM_SANG",
  "description": "Tiếng thở bất thường có thể gặp trong bệnh hô hấp.",
  "surface_forms": ["thở khò khè", "khò khè"],
  "evidence_texts": ["Người bệnh có thể thở khò khè, ho và khó thở."]
}
Output:
{
  "profile_text": "thở khò khè là biểu hiện lâm sàng được mô tả trong bệnh hô hấp.",
  "local_keys": ["thở khò khè", "khò khè"],
  "global_keys": ["triệu chứng hô hấp", "biểu hiện lâm sàng", "bệnh đường thở"]
}

---JSON Schema---
{
  "profile_text": "...",
  "local_keys": ["..."],
  "global_keys": ["..."]
}
""".strip()


RELATION_PROFILE_SYSTEM_PROMPT = """
---Role---
Bạn là expert tạo keywords cho LightRAG-style graph retrieval trong miền y khoa tiếng Việt.

---LightRAG Keyword Principle---
LightRAG tách keyword thành hai tầng, giống ví dụ biodiversity:
- low-level/local keywords: entity cụ thể, tên gọi, chi tiết cụ thể trong relation/evidence.
- high-level/global keywords: chủ đề, khái niệm bao quát, theme rộng hơn của relation.

---Task---
Với RELATION đã có trong graph, tạo một list "keywords" gồm cả hai tầng:
- 2-5 low-level keywords nếu evidence đủ thông tin.
- 2-5 high-level keywords nếu evidence đủ thông tin.
Tổng số keywords nên ngắn gọn, thường 4-8 item.

---Rules---
- Chỉ dùng input để hiểu quan hệ; không thêm fact y khoa mới.
- Không cần chọn từ vocabulary cố định. Hãy tạo cụm tự nhiên, đa dạng về ngữ nghĩa.
- Low-level keywords có thể là subject/object, triệu chứng cụ thể, yếu tố cụ thể, vị trí cụ thể, hoặc chi tiết cụ thể trong evidence.
- High-level keywords là theme rộng hơn, ví dụ: "triệu chứng hô hấp", "yếu tố môi trường", "bệnh da dị ứng", "chẩn đoán phân biệt bệnh da", "biến chứng nhiễm trùng".
- Không chỉ trả nhãn schema khô cứng như "CO_BIEU_HIEN" hoặc "LIEN_QUAN_YEU_TO".
- Không tạo keyword quá dài hoặc thành câu hoàn chỉnh.
- Trả JSON parseable, không markdown, không giải thích.

---Few-shot examples---
Example 1:
Input relation:
{
  "relation_type": "CO_BIEU_HIEN",
  "subject": {"name": "hen phế quản dị ứng", "entity_type": "BENH_LY"},
  "object": {"name": "thở khò khè", "entity_type": "BIEU_HIEN_LAM_SANG"},
  "evidence_text": "Người bệnh hen phế quản dị ứng có thể thở khò khè, ho và khó thở."
}
Output:
{
  "keywords": ["hen phế quản dị ứng", "thở khò khè", "ho", "khó thở", "triệu chứng hô hấp", "biểu hiện lâm sàng", "bệnh đường thở"]
}

Example 2:
Input relation:
{
  "relation_type": "LIEN_QUAN_YEU_TO",
  "subject": {"name": "viêm da cơ địa", "entity_type": "BENH_LY"},
  "object": {"name": "bụi", "entity_type": "YEU_TO_NGOAI_SINH"},
  "evidence_text": "Bệnh thường nặng thêm do các tác động của yếu tố môi trường như bụi, ô nhiễm, hóa chất."
}
Output:
{
  "keywords": ["viêm da cơ địa", "bụi", "ô nhiễm", "hóa chất", "yếu tố môi trường", "yếu tố kích hoạt", "bệnh da dị ứng"]
}

Example 3:
Input relation:
{
  "relation_type": "CHAN_DOAN_PHAN_BIET_VOI",
  "subject": {"name": "viêm da tiết bã", "entity_type": "BENH_LY"},
  "object": {"name": "bệnh vảy nến", "entity_type": "BENH_LY"},
  "evidence_text": "Viêm da tiết bã dễ nhầm lẫn với các bệnh như viêm da cơ địa hoặc vảy nến da đầu."
}
Output:
{
  "keywords": ["viêm da tiết bã", "bệnh vảy nến", "vảy nến da đầu", "viêm da cơ địa", "chẩn đoán phân biệt", "phân biệt bệnh da liễu", "bệnh da bong vảy"]
}

Example 4:
Input relation:
{
  "relation_type": "GAY_BIEN_CHUNG",
  "subject": {"name": "viêm da cơ địa", "entity_type": "BENH_LY"},
  "object": {"name": "nhiễm trùng da", "entity_type": "BENH_LY"},
  "evidence_text": "Da có thể bị tổn thương do gãi nhiều, làm tăng nguy cơ nhiễm vi khuẩn và vi rút."
}
Output:
{
  "keywords": ["viêm da cơ địa", "nhiễm trùng da", "gãi nhiều", "vi khuẩn", "vi rút", "biến chứng", "tổn thương da"]
}

---JSON Schema---
{
  "keywords": ["...", "..."]
}
""".strip()
