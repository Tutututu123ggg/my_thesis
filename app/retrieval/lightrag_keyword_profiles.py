from __future__ import annotations

import json
import re
from typing import Iterable

from pydantic import BaseModel, Field

from app.domain.medical_schema import ENTITY_TYPES, RELATION_TYPES


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
            schema_values=[entity_type],
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
            schema_values=[relation_type, subject_type, object_type],
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
        # Do not derive global keys from entity_type. Schema labels such as
        # BENH_LY/BIEU_HIEN_LAM_SANG are implementation metadata, not
        # LightRAG natural-language topic keys. Fallback global keys are mined
        # only from available description/evidence text.
        global_keys = self._clean_global_entity_keys(
            self._semantic_hints_from_text(" ".join([description or "", *(evidence_texts or [])])),
            self.max_entity_global_keys,
            local_keys=local,
            schema_values=[entity_type],
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
        # Fallback is intentionally simple and data-derived: endpoints are
        # low-level retrieval signals; text-mined n-grams are weak
        # global/detail signals. Do not add relation_type itself, because schema
        # enums such as CO_BIEU_HIEN/DIEU_TRI_BANG are not LightRAG natural
        # language topic keys.
        keywords = [subject_name, object_name]
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
                schema_values=[relation_type, subject_type, object_type],
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
        # Kept for backward compatibility. Entity schema labels are metadata,
        # not LightRAG natural-language global keys.
        return []

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
        """Do not mine arbitrary n-grams from evidence for fallback keywords.

        Graph audits showed that evidence-derived n-grams can become noisy
        strings such as long copied clauses or comma-glued medication lists.
        Keyword quality is primarily handled by the article-level keyword LLM
        profiler; deterministic fallback should stay conservative.
        """
        return []

    def _humanize_type(self, value: str) -> str:
        return (value or "").lower().replace("_", " ")

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
            if not key or key in seen:
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
        schema_values: Iterable[str] | None = None,
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
        schema_norms = self._schema_keyword_norms(schema_values)
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = self._clean_text(str(value))
            if not text or len(text.split()) > 8:
                continue
            key = self._normalize_ascii(text)
            if not key or key in seen:
                continue
            if key in local_norms or key in schema_norms:
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
        schema_values: Iterable[str] | None = None,
    ) -> list[str]:
        """Clean LightRAG relation keywords.

        LightRAG examples mix concrete low-level terms with broader high-level
        concepts. Therefore relation.keywords may include both endpoint-like
        specific terms and broad themes. We only remove empty/duplicate/very
        generic or sentence-like keywords; we do not force a closed vocabulary.
        """
        schema_norms = self._schema_keyword_norms(schema_values)
        result: list[str] = []
        seen: set[str] = set()
        for item in values:
            value = self._clean_text(str(item))
            if not value or len(value.split()) > 6:
                continue
            if "," in value or ";" in value:
                continue
            key = self._normalize_ascii(value)
            if not key or key in seen or key in schema_norms:
                continue
            # Keep relation keys as short retrieval labels, not copied clauses.
            if len(key) > 80:
                continue
            seen.add(key)
            result.append(value)
            if len(result) >= limit:
                break
        return result

    def _suggested_relation_keywords(self, relation_type: str) -> list[str]:
        # Kept for backward compatibility. Relation schema labels are metadata,
        # not LightRAG natural-language keyword keys.
        return []

    def _schema_keyword_norms(self, values: Iterable[str] | None = None) -> set[str]:
        """Normalized schema labels that must not become retrieval keywords.

        This is a schema-invariant guard, not a medical term blacklist. It only
        prevents implementation metadata such as BENH_LY or CO_BIEU_HIEN from
        leaking into LightRAG natural-language keyword vectors.
        """
        raw_values = {str(item or "") for item in (values or [])}
        raw_values.update(str(item or "") for item in ENTITY_TYPES)
        raw_values.update(str(item or "") for item in RELATION_TYPES)

        norms: set[str] = set()
        for value in raw_values:
            value = value.strip()
            if not value:
                continue
            norms.add(self._normalize_ascii(value))
            norms.add(self._normalize_ascii(value.lower().replace("_", " ")))
        return {item for item in norms if item}

    def _looks_like_keyword(self, value: str) -> bool:
        value = self._normalize_ascii(value)
        if not value or len(value) < 5:
            return False
        toks = value.split()
        if len(toks) > 8:
            return False
        if len(toks) == 1:
            return len(toks[0]) >= 6 or any(ch.isdigit() for ch in toks[0])
        return any(len(tok) >= 3 or any(ch.isdigit() for ch in tok) for tok in toks)

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


ENTITY_PROFILE_SYSTEM_PROMPT = """
---Role---
Bạn là expert tạo keyword profile cho LightRAG-style graph retrieval trong miền y khoa tiếng Việt.

---LightRAG Keyword Principle---
- local_keys: tên entity, surface form, viết tắt, synonym/tên tương đương nếu xuất hiện trong input.
- global_keys: chủ đề/theme tự nhiên rộng hơn entity, dùng cho retrieval ngữ nghĩa.
- global_keys không phải schema label, entity type, relation type enum.

---Task---
Với ENTITY đã có trong graph, tạo JSON:
{
  "profile_text": "...",
  "local_keys": ["..."],
  "global_keys": ["..."]
}

---Rules---
- Chỉ dùng dữ liệu được cung cấp, không thêm fact y khoa mới.
- local_keys không chứa xét nghiệm/điều trị/biến chứng/triệu chứng/yếu tố nguy cơ, trừ khi chính entity hiện tại là khái niệm đó.
- global_keys là cụm tự nhiên, không lặp lại local_keys, không phải synonym list.
- Không output schema/enum như BENH_LY, BIEU_HIEN_LAM_SANG, CO_BIEU_HIEN, DIEU_TRI_BANG.
- Tránh keyword quá rộng như "bệnh lý", "vấn đề", "tình trạng sức khỏe".
- Keywords ngắn gọn, thường 2-8 từ.
- Chỉ trả JSON parseable, không markdown, không giải thích.

---Examples---
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
  "profile_text": "thở khò khè là triệu chứng được mô tả trong bệnh hô hấp.",
  "local_keys": ["thở khò khè", "khò khè"],
  "global_keys": ["triệu chứng hô hấp", "triệu chứng bệnh đường thở", "khó thở và khò khè"]
}
""".strip()


RELATION_PROFILE_SYSTEM_PROMPT = """
---Role---
Bạn là expert tạo keywords cho LightRAG-style relation retrieval trong miền y khoa tiếng Việt.

---Principle---
Relation keywords là một list trộn hai tầng:
- local keys: subject, object, hoặc chi tiết cụ thể xuất hiện trong evidence_text.
- global keys: topic/theme tự nhiên của quan hệ, dùng cho relation/global-key retrieval.

---Task---
Với RELATION đã có trong graph, trả JSON:
{
  "keywords": ["..."]
}

---Rules---
- Mỗi relation nên có 4-8 keywords ngắn.
- Chỉ dùng input/evidence_text, không thêm fact y khoa mới.
- Không dùng schema label/entity type/relation enum làm keyword.
  Ví dụ cấm: BENH_LY, BIEU_HIEN_LAM_SANG, CO_BIEU_HIEN, DIEU_TRI_BANG, GAY_BIEN_CHUNG.
- Không chỉ dùng facet generic như "triệu chứng", "điều trị", "biến chứng", "chẩn đoán".
- Nếu có bệnh/entity chính, ưu tiên global key có anchor:
  "triệu chứng của viêm da cơ địa", "điều trị hen phế quản",
  "biến chứng của viêm phổi", "yếu tố làm nặng viêm da cơ địa".
- Không tạo keyword quá dài hoặc thành câu hoàn chỉnh.
- Chỉ trả JSON parseable, không markdown, không giải thích.

---Examples---
Input:
{
  "relation_type": "CO_BIEU_HIEN",
  "subject": {"name": "hen phế quản dị ứng", "entity_type": "BENH_LY"},
  "object": {"name": "thở khò khè", "entity_type": "BIEU_HIEN_LAM_SANG"},
  "evidence_text": "Người bệnh hen phế quản dị ứng có thể thở khò khè, ho và khó thở."
}
Output:
{
  "keywords": ["hen phế quản dị ứng", "thở khò khè", "ho", "khó thở", "triệu chứng của hen phế quản dị ứng", "triệu chứng hô hấp"]
}

Input:
{
  "relation_type": "LIEN_QUAN_YEU_TO",
  "subject": {"name": "viêm da cơ địa", "entity_type": "BENH_LY"},
  "object": {"name": "bụi", "entity_type": "YEU_TO_NGOAI_SINH"},
  "evidence_text": "Bệnh thường nặng thêm do các tác động của yếu tố môi trường như bụi, ô nhiễm, hóa chất."
}
Output:
{
  "keywords": ["viêm da cơ địa", "bụi", "ô nhiễm", "hóa chất", "yếu tố môi trường làm nặng viêm da cơ địa", "yếu tố kích hoạt bệnh da"]
}

Input:
{
  "relation_type": "GAY_BIEN_CHUNG",
  "subject": {"name": "viêm phổi", "entity_type": "BENH_LY"},
  "object": {"name": "suy hô hấp", "entity_type": "BENH_LY"},
  "evidence_text": "Viêm phổi nặng có thể dẫn đến suy hô hấp."
}
Output:
{
  "keywords": ["viêm phổi", "suy hô hấp", "viêm phổi nặng", "biến chứng của viêm phổi", "diễn tiến viêm phổi gây suy hô hấp", "biến chứng hô hấp nặng"]
}
""".strip()