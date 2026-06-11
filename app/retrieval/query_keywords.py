from __future__ import annotations

import os
import re
from typing import Iterable

from pydantic import BaseModel, Field

from app.retrieval.retrieval_models import QueryKeywords


class QueryKeywordExtractionResult(BaseModel):
    """
    Output đúng tinh thần LightRAG:
    - low_level_keywords: entity/detail keys cho entity retrieval.
    - high_level_keywords: theme/relation intent keys cho relation retrieval.
    """

    low_level_keywords: list[str] = Field(default_factory=list)
    high_level_keywords: list[str] = Field(default_factory=list)


class QueryKeywordExtractor:
    """
    Extract low/high-level query keywords theo LightRAG.

    Paper mapping:
    - low-level keywords -> match candidate entities.
    - high-level keywords -> match relations/global keys.

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

            return QueryKeywords(
                low_level_keywords=self._clean_keywords(parsed.low_level_keywords, limit=6),
                high_level_keywords=self._clean_keywords(parsed.high_level_keywords, limit=6),
            )
        except Exception:
            return self._fallback_extract(query)

    def _system_prompt(self) -> str:
        return """
Bạn là bộ tách keyword truy hồi cho Medical GraphRAG/LightRAG tiếng Việt.

Nhiệm vụ: từ một câu hỏi y khoa, trả JSON gồm:
- low_level_keywords: các entity cụ thể, tên bệnh, triệu chứng, thuốc, yếu tố, vị trí, đối tượng.
- high_level_keywords: chủ đề/ý định quan hệ rộng hơn, ví dụ triệu chứng, nguyên nhân, yếu tố nguy cơ, biến chứng, chẩn đoán, điều trị, phòng ngừa, cần tránh, phân loại, tên gọi khác.

Quy tắc:
- Keyword ngắn, không phải câu dài.
- Không bịa khái niệm ngoài câu hỏi.
- Nếu query hỏi "có triệu chứng gì", high_level_keywords nên có "triệu chứng" hoặc "biểu hiện".
- Nếu query hỏi "nguyên nhân/yếu tố", high_level_keywords nên có "nguyên nhân", "yếu tố nguy cơ".
- Nếu query hỏi "điều trị", high_level_keywords nên có "điều trị".
- Nếu query hỏi tên khác/tiếng Anh/còn gọi là, high_level_keywords nên có "tên gọi khác".
""".strip()

    def _fallback_extract(self, query: str) -> QueryKeywords:
        lowered = query.lower()

        intent_map = [
            (("triệu chứng", "biểu hiện", "dấu hiệu"), "triệu chứng"),
            (("nguyên nhân", "vì sao", "tại sao", "yếu tố"), "nguyên nhân"),
            (("liên quan", "liên quan đến", "liên quan tới", "dị ứng"), "yếu tố liên quan"),
            (("biến chứng", "nguy hiểm"), "biến chứng"),
            (("chẩn đoán", "xét nghiệm", "phân biệt"), "chẩn đoán"),
            (("điều trị", "chữa", "thuốc", "bôi", "uống"), "điều trị"),
            (("phòng ngừa", "phòng tránh"), "phòng ngừa"),
            (("tránh", "kiêng"), "cần tránh"),
            (("loại", "dạng", "thể", "phân loại"), "phân loại"),
            (("tên khác", "còn gọi", "tiếng anh", "english", "aka"), "tên gọi khác"),
        ]

        high: list[str] = []
        for cues, keyword in intent_map:
            if any(cue in lowered for cue in cues):
                high.append(keyword)

        low = self._guess_low_keywords(query, stopwords=set(high))
        if not low:
            low = [query]
        if not high:
            high = ["thông tin y khoa"]

        return QueryKeywords(
            low_level_keywords=self._clean_keywords(low, limit=6),
            high_level_keywords=self._clean_keywords(high, limit=6),
        )

    def _guess_low_keywords(self, query: str, stopwords: set[str]) -> list[str]:
        text = query.strip(" ?!.。")

        # Cắt bớt intent phrase phổ biến để giữ entity phrase.
        patterns = [
            r"\b(có|gồm|là|bao gồm|như thế nào|gì|nào|không)\b",
            r"\b(liên quan đến|liên quan tới|liên quan|đến|tới)\b",
            r"\b(triệu chứng|biểu hiện|dấu hiệu|nguyên nhân|yếu tố|biến chứng|điều trị|chẩn đoán|phòng ngừa|cần tránh|phân loại)\b",
        ]
        cleaned = text.lower()
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Giữ cụm 2-6 token có vẻ là entity.
        chunks = re.split(r"[,;:/]|\bvà\b|\bhay\b|\bhoặc\b|\bliên quan đến\b|\bliên quan tới\b", cleaned)
        result = [c.strip() for c in chunks if 2 <= len(c.strip()) <= 80]

        return [x for x in result if x not in stopwords]

    def _clean_keywords(self, keywords: Iterable[str], limit: int) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in keywords:
            value = re.sub(r"\s+", " ", str(item).strip(" -–—:;,.()[]{}\n\t"))
            if not value:
                continue
            if len(value.split()) > 8:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
            if len(result) >= limit:
                break
        return result