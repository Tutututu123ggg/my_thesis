from __future__ import annotations

import os
import re
import unicodedata
from typing import Iterable

from pydantic import BaseModel, Field

from app.retrieval.retrieval_models import QueryKeywords


def _normalize_ascii(text: str) -> str:
    text = (text or "").strip().lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_SCHEMA_ENUM_KEYWORDS = {
    "BENH_LY",
    "BIEU_HIEN_LAM_SANG",
    "YEU_TO_BENH_SINH",
    "YEU_TO_NGOAI_SINH",
    "PHUONG_PHAP_CHAN_DOAN",
    "CAN_THIEP_Y_TE",
    "THUOC_VA_HOAT_CHAT",
    "BIEN_CHUNG",
    "GIAI_DOAN_BENH",
    "DOI_TUONG_NGUY_CO",
    "CO_TRIEU_CHUNG",
    "CO_BIEU_HIEN",
    "CO_NGUYEN_NHAN",
    "CO_YEU_TO_NGUY_CO",
    "GAY_BIEN_CHUNG",
    "CHAN_DOAN_BANG",
    "DIEU_TRI_BANG",
    "PHONG_NGUA_BANG",
    "DONG_NGHIA_VOI",
    "CHAN_DOAN_PHAN_BIET_VOI",
    "LIEN_QUAN_YEU_TO",
}
_SCHEMA_ENUM_KEYS = {
    _normalize_ascii(item) for item in _SCHEMA_ENUM_KEYWORDS
} | {
    _normalize_ascii(item.replace("_", " ")) for item in _SCHEMA_ENUM_KEYWORDS
}

_ADMIN_KEYWORD_FRAGMENTS = {
    "Vinmec",
    "MEDLATEC",
    "Tâm Anh",
    "Bệnh viện Đa khoa Tâm Anh",
    "hotline",
    "tổng đài",
    "đặt lịch",
    "đặt lịch khám",
    "đăng ký tư vấn",
    "gói khám",
    "app",
    "MyVinmec",
    "MEDON",
    "dịch vụ xét nghiệm",
    "dịch vụ y tế",
    "lấy mẫu xét nghiệm tại nhà",
}
_ADMIN_FRAGMENT_KEYS = {_normalize_ascii(item) for item in _ADMIN_KEYWORD_FRAGMENTS}

_GENERIC_HIGH_LEVEL_FACETS = {
    _normalize_ascii(item)
    for item in {
        "triệu chứng",
        "biểu hiện",
        "dấu hiệu",
        "nguyên nhân",
        "yếu tố nguy cơ",
        "yếu tố liên quan",
        "biến chứng",
        "chẩn đoán",
        "điều trị",
        "chăm sóc",
        "chăm sóc da",
        "giảm triệu chứng",
        "thuốc bôi",
        "dưỡng ẩm",
        "phòng ngừa",
        "cần tránh",
        "phân loại",
        "tên gọi khác",
        "thông tin y khoa",
    }
}


class QueryKeywordExtractionResult(BaseModel):
    """
    Output đúng tinh thần LightRAG:
    - low_level_keywords: entity/detail keys cho entity retrieval.
    - high_level_keywords: natural-language topic/theme keys cho relation/global-key retrieval.
    """

    low_level_keywords: list[str] = Field(default_factory=list)
    high_level_keywords: list[str] = Field(default_factory=list)


class QueryKeywordExtractor:
    """
    Extract low/high-level query keywords theo LightRAG.

    Paper mapping:
    - low-level keywords -> match candidate entities.
    - high-level keywords -> natural-language topics/themes used to match
      relation/global-key profiles.

    Có fallback không LLM để smoke test nhanh và tránh block retrieval khi thiếu API.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        use_llm: bool = True,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.use_llm = use_llm and bool(os.getenv("OPENAI_API_KEY"))
        self.client = None
        if self.use_llm:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception:
                self.use_llm = False
                self.client = None

    def extract(self, query: str) -> QueryKeywords:
        query = query.strip()
        if not query:
            return QueryKeywords()

        if not self.use_llm or self.client is None:
            return self._fallback_extract(query)

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": query},
                ],
                response_format=QueryKeywordExtractionResult,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                return self._fallback_extract(query)

            low = self._clean_keywords(parsed.low_level_keywords, limit=6, max_words=8)
            high = self._clean_keywords(parsed.high_level_keywords, limit=8, max_words=14)
            high = self._postprocess_high_keywords(query=query, low=low, high=high)
            if not low:
                return self._fallback_extract(query)

            return QueryKeywords(
                low_level_keywords=low,
                high_level_keywords=high,
            )
        except Exception:
            return self._fallback_extract(query)

    def _system_prompt(self) -> str:
        return """
Bạn là bộ tách low-level và high-level keywords cho LightRAG-style retrieval
trong hệ thống Medical RAG tiếng Việt.

Nguyên tắc LightRAG:
- low_level_keywords dùng để match entity/local information.
- high_level_keywords dùng để match relation/global-key profiles.
- high_level_keywords KHÔNG phải schema label, entity type, relation type hay classifier.

Nhiệm vụ:
Trả về JSON có đúng 2 trường:
{
  "low_level_keywords": [...],
  "high_level_keywords": [...]
}

Quy tắc cho low_level_keywords:
- Chọn các thực thể/chi tiết cụ thể xuất hiện hoặc được hỏi trực tiếp trong query.
- Ưu tiên: tên bệnh, triệu chứng cụ thể, biến chứng cụ thể, thuốc/can thiệp, xét nghiệm,
  tác nhân, yếu tố nguy cơ, vị trí giải phẫu, nhóm đối tượng cụ thể.
- Không đưa các từ chỉ intent/facet chung như: triệu chứng, nguyên nhân, điều trị,
  chẩn đoán, phòng ngừa, biến chứng, nếu chúng không phải entity/detail cụ thể.
- Nếu query so sánh nhiều bệnh/triệu chứng, giữ đủ các anchor quan trọng.

Quy tắc cho high_level_keywords:
- Là cụm chủ đề/theme tự nhiên bằng tiếng Việt, dùng để tìm relation/global-key.
- Không phải nhãn schema, không phải entity type, không phải relation enum.
- Nếu query có bệnh/entity anchor, hãy anchor hóa topic:
  tốt: "triệu chứng của viêm da cơ địa", "điều trị hen phế quản",
       "biến chứng của viêm phổi", "tên gọi khác của mày đay"
  kém: "triệu chứng", "điều trị", "biến chứng"
- Có thể giữ tối đa 1 keyword generic như "triệu chứng" hoặc "điều trị" làm phụ trợ,
  nhưng high_level_keywords không được chỉ gồm các facet generic.
- Với query quan hệ A gây B, tạo topic thể hiện quan hệ tự nhiên:
  "diễn tiến A gây B", "biến chứng của A", "liên quan giữa A và B".
- Với query so sánh/phân biệt, tạo topic như:
  "so sánh triệu chứng bệnh da liễu",
  "khác biệt biểu hiện A và B",
  "chẩn đoán phân biệt bệnh da".
- Với query tên gọi khác/đồng nghĩa, tạo topic như:
  "tên gọi khác của A", "đồng nghĩa thuật ngữ bệnh da liễu".

Cấm:
- Không xuất graph schema labels, entity types, relation type enums.
- Không xuất mã hoặc nhãn kiểu: BENH_LY, BIEU_HIEN_LAM_SANG, CO_TRIEU_CHUNG,
  DIEU_TRI_BANG, GAY_BIEN_CHUNG, CHAN_DOAN_BANG, PHONG_NGUA_BANG, DONG_NGHIA_VOI.
- Không xuất từ quảng bá/hành chính/dịch vụ như tên bệnh viện, hotline, đặt lịch,
  gói khám, app, dịch vụ xét nghiệm, tư vấn, bảng giá.
- Không thêm fact y khoa mới ngoài ý nghĩa của query.
- Keyword ngắn gọn, thường 2-8 từ; không viết thành câu trả lời.

Ví dụ:
Input: "Viêm da cơ địa có triệu chứng gì?"
Output:
{
  "low_level_keywords": ["viêm da cơ địa"],
  "high_level_keywords": [
    "triệu chứng của viêm da cơ địa",
    "biểu hiện ngứa và tổn thương da",
    "dấu hiệu bệnh da liễu"
  ]
}

Input: "Viêm phổi có gây suy hô hấp không?"
Output:
{
  "low_level_keywords": ["viêm phổi", "suy hô hấp"],
  "high_level_keywords": [
    "biến chứng của viêm phổi",
    "diễn tiến viêm phổi gây suy hô hấp",
    "biến chứng hô hấp nặng"
  ]
}

Input: "Hen phế quản điều trị thế nào?"
Output:
{
  "low_level_keywords": ["hen phế quản"],
  "high_level_keywords": [
    "điều trị hen phế quản",
    "kiểm soát triệu chứng hen",
    "dự phòng cơn hen cấp"
  ]
}

Input: "Mày đay còn gọi là gì?"
Output:
{
  "low_level_keywords": ["mày đay"],
  "high_level_keywords": [
    "tên gọi khác của mày đay",
    "đồng nghĩa thuật ngữ bệnh da liễu"
  ]
}

Input: "Vì sao cùng là ngứa và bong tróc, viêm da tiết bã nên được truy hồi khác với viêm da cơ địa?"
Output:
{
  "low_level_keywords": ["ngứa", "bong tróc", "viêm da tiết bã", "viêm da cơ địa"],
  "high_level_keywords": [
    "so sánh triệu chứng bệnh da liễu",
    "khác biệt biểu hiện viêm da tiết bã và viêm da cơ địa",
    "ngứa và bong tróc trong bệnh da"
  ]
}

Chỉ trả JSON parseable, không markdown, không giải thích.
""".strip()

    def _fallback_extract(self, query: str) -> QueryKeywords:
        guessed_low = self._guess_low_keywords(query, stopwords=set())
        low = self._clean_keywords(guessed_low, limit=6, max_words=8)
        if not low:
            low = self._clean_keywords([query], limit=1, max_words=14)

        guessed_high = self._fallback_high_keywords(query, low)
        high = self._clean_keywords(guessed_high, limit=8, max_words=14)
        high = self._postprocess_high_keywords(query=query, low=low, high=high)

        return QueryKeywords(
            low_level_keywords=low,
            high_level_keywords=high,
        )

    def _fallback_high_keywords(self, query: str, low: list[str]) -> list[str]:
        norm_query = _normalize_ascii(query)
        high: list[str] = []
        disease_anchors = self._select_disease_anchors(low)
        anchor = disease_anchors[0] if disease_anchors else (low[0] if low else "")
        symptoms = [item for item in low if item not in disease_anchors]

        def add(value: str) -> None:
            if value:
                high.append(value)

        is_comparison = self._contains_any(
            norm_query,
            ["khac voi", "so sanh", "phan biet", "cung la", "truy hoi khac"],
        )
        if is_comparison:
            if len(disease_anchors) >= 2:
                add("so sánh triệu chứng bệnh da liễu" if "da" in norm_query else "so sánh biểu hiện bệnh")
                add(f"khác biệt biểu hiện {disease_anchors[0]} và {disease_anchors[1]}")
            else:
                add("so sánh biểu hiện bệnh")
            if symptoms:
                symptom_phrase = " và ".join(symptoms[:2])
                add(f"{symptom_phrase} trong bệnh da" if "da" in norm_query else f"{symptom_phrase} trong bệnh")
            add("chẩn đoán phân biệt bệnh da liễu" if "da" in norm_query else "chẩn đoán phân biệt")
            return high

        is_alias = self._contains_any(norm_query, ["ten khac", "con goi", "tieng anh", "english", "aka", "dong nghia"])
        if is_alias:
            add(f"tên gọi khác của {anchor}" if anchor else "tên gọi khác")
            add("đồng nghĩa thuật ngữ bệnh da liễu" if "da" in norm_query or "may day" in norm_query else "đồng nghĩa thuật ngữ y khoa")

        is_complication = self._contains_any(norm_query, ["bien chung", "nguy hiem", "gay", "dan den", "dan toi"])
        if is_complication:
            add(f"biến chứng của {anchor}" if anchor else "biến chứng")
            if len(low) >= 2 and anchor:
                target = next((item for item in low if item != anchor), "")
                if target:
                    add(f"diễn tiến {anchor} gây {target}")
            add("biến chứng hô hấp nặng" if self._contains_any(norm_query, ["ho hap", "phoi", "pho quan"]) else "biến chứng bệnh")

        is_symptom = self._contains_any(norm_query, ["trieu chung", "bieu hien", "dau hieu"])
        if is_symptom:
            add(f"triệu chứng của {anchor}" if anchor else "triệu chứng")
            add("biểu hiện bệnh da liễu" if "da" in norm_query else "biểu hiện lâm sàng")
            add("triệu chứng")

        is_treatment = self._contains_any(
            norm_query,
            ["dieu tri", "chua", "thuoc", "boi", "uong", "lam sao", "lam gi", "xu tri", "giam", "do"],
        )
        if is_treatment:
            add(f"điều trị {anchor}" if anchor else "điều trị")
            if anchor and "hen" in _normalize_ascii(anchor):
                add("kiểm soát triệu chứng hen")
                add("dự phòng cơn hen cấp")
            else:
                add(f"kiểm soát triệu chứng {anchor}" if anchor else "kiểm soát triệu chứng")
            add("điều trị")

        is_cause = self._contains_any(norm_query, ["nguyen nhan", "vi sao", "tai sao", "yeu to"])
        if is_cause and not high:
            add(f"nguyên nhân của {anchor}" if anchor else "nguyên nhân")
            add(f"yếu tố liên quan đến {anchor}" if anchor else "yếu tố liên quan")

        is_prevention = self._contains_any(norm_query, ["phong ngua", "phong tranh", "du phong"])
        if is_prevention:
            add(f"phòng ngừa {anchor}" if anchor else "phòng ngừa")
            add("dự phòng tái phát" if anchor else "dự phòng bệnh")

        if not high:
            add(f"thông tin y khoa về {anchor}" if anchor else "thông tin y khoa")
        return high

    def _guess_low_keywords(self, query: str, stopwords: set[str]) -> list[str]:
        text = query.strip(" ?!.。")
        cleaned = text.lower()

        # Dùng connector như ranh giới giữa các entity/detail trong query.
        # Làm bước này trước khi xoá filler để các cặp như "khác với" không
        # bị nối thành một cụm bệnh sai.
        separator_patterns = [
            r"[,;:/]",
            r"\bkhác với\b",
            r"\bso với\b",
            r"\bliên quan đến\b",
            r"\bliên quan tới\b",
            r"\bvà\b",
            r"\bhay\b",
            r"\bhoặc\b",
            r"\bgây\b",
            r"\bdẫn đến\b",
            r"\bdẫn tới\b",
            r"\blàm tăng nguy cơ\b",
            r"\bkèm theo\b",
            r"\bkèm\b",
            r"\bdo\b",
            r"\bbởi\b",
            r"\bvới\b",
        ]
        for pattern in separator_patterns:
            cleaned = re.sub(pattern, " , ", cleaned, flags=re.IGNORECASE)

        # Cắt filler/intent phrase để giữ entity/detail surface forms.
        filler_patterns = [
            r"\b(vì sao|tại sao|cùng là|nên được truy hồi|được truy hồi|nên|có thể|có|gồm|là|bao gồm|như thế nào|thế nào|gì|nào|không)\b",
            r"\b(triệu chứng|biểu hiện|dấu hiệu|nguyên nhân|yếu tố nguy cơ|yếu tố|biến chứng|điều trị|chữa|thuốc|bôi|uống|giảm triệu chứng|chăm sóc da|dưỡng ẩm|thuốc bôi|chẩn đoán|phòng ngừa|cần tránh|phân loại|tên khác|còn gọi|tiếng anh|english|aka)\b",
            r"\b(liên quan|khác|đến|tới)\b",
        ]
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
        chunks = [chunk.strip(" -–—:;,.()[]{}\n\t") for chunk in cleaned.split(",")]

        result: list[str] = []
        seen: set[str] = set()
        normalized_stopwords = {_normalize_ascii(item) for item in stopwords}
        for chunk in chunks:
            chunk = re.sub(r"\s+", " ", chunk).strip()
            key = _normalize_ascii(chunk)
            if not chunk or len(chunk) < 2 or len(chunk) > 80:
                continue
            if key in normalized_stopwords or key in seen:
                continue
            seen.add(key)
            result.append(chunk)
        return result

    def _postprocess_high_keywords(self, *, query: str, low: list[str], high: list[str]) -> list[str]:
        if not high:
            high = self._fallback_high_keywords(query, low)

        anchor = self._select_primary_anchor(low)
        if anchor:
            anchor_key = _normalize_ascii(anchor)
            has_anchor = any(anchor_key and anchor_key in _normalize_ascii(item) for item in high)
            generic_items = [item for item in high if self._is_generic_high_keyword(item)]
            if generic_items and not has_anchor:
                anchored = [self._anchorize_generic_high_keyword(item, anchor) for item in generic_items]
                high = [item for item in anchored if item] + high

        return self._clean_keywords(high, limit=6, max_words=14)

    def _anchorize_generic_high_keyword(self, keyword: str, anchor: str) -> str:
        key = _normalize_ascii(keyword)
        mapping = {
            "trieu chung": f"triệu chứng của {anchor}",
            "bieu hien": f"biểu hiện của {anchor}",
            "dau hieu": f"dấu hiệu của {anchor}",
            "nguyen nhan": f"nguyên nhân của {anchor}",
            "yeu to nguy co": f"yếu tố nguy cơ của {anchor}",
            "yeu to lien quan": f"yếu tố liên quan đến {anchor}",
            "bien chung": f"biến chứng của {anchor}",
            "chan doan": f"chẩn đoán {anchor}",
            "dieu tri": f"điều trị {anchor}",
            "cham soc": f"chăm sóc trong {anchor}",
            "cham soc da": f"chăm sóc da trong {anchor}",
            "giam trieu chung": f"kiểm soát triệu chứng {anchor}",
            "thuoc boi": f"thuốc bôi điều trị {anchor}",
            "duong am": f"dưỡng ẩm trong chăm sóc {anchor}",
            "phong ngua": f"phòng ngừa {anchor}",
            "can tranh": f"yếu tố cần tránh trong {anchor}",
            "phan loai": f"phân loại {anchor}",
            "ten goi khac": f"tên gọi khác của {anchor}",
            "thong tin y khoa": f"thông tin y khoa về {anchor}",
        }
        return mapping.get(key, f"{keyword} của {anchor}")

    def _select_primary_anchor(self, low: list[str]) -> str:
        anchors = self._select_disease_anchors(low)
        if anchors:
            return anchors[0]
        return low[0] if low else ""

    def _select_disease_anchors(self, low: list[str]) -> list[str]:
        return [item for item in low if self._looks_like_disease_anchor(item)]

    def _looks_like_disease_anchor(self, value: str) -> bool:
        key = _normalize_ascii(value)
        disease_cues = [
            "benh",
            "viem",
            "hen",
            "phoi",
            "pho quan",
            "mui",
            "xoang",
            "hong",
            "amidan",
            "may day",
            "eczema",
            "copd",
            "suy ho hap",
        ]
        symptom_only_cues = {"ngua", "bong troc", "man do", "noi may", "kho tho", "ho"}
        if key in symptom_only_cues:
            return False
        return any(cue in key for cue in disease_cues)

    def _contains_any(self, text_norm: str, cues: Iterable[str]) -> bool:
        return any(_normalize_ascii(cue) in text_norm for cue in cues)

    def _clean_keywords(self, keywords: Iterable[str], limit: int, max_words: int = 12) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in keywords:
            value = re.sub(r"\s+", " ", str(item).strip(" -–—:;,.()[]{}\n\t"))
            if not value:
                continue
            if self._is_forbidden_keyword(value):
                continue
            if len(value.split()) > max_words:
                continue
            key = _normalize_ascii(value)
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
            if len(result) >= limit:
                break
        return result

    def _is_forbidden_keyword(self, value: str) -> bool:
        key = _normalize_ascii(value)
        if not key:
            return True
        if key in _SCHEMA_ENUM_KEYS:
            return True
        if any(fragment and fragment in key for fragment in _ADMIN_FRAGMENT_KEYS):
            return True
        return False

    def _is_generic_high_keyword(self, value: str) -> bool:
        return _normalize_ascii(value) in _GENERIC_HIGH_LEVEL_FACETS
