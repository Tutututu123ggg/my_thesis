from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.retrieval.retrieval_models import QueryKeywords


@dataclass(frozen=True)
class QueryAnalysis:
    """Schema-neutral query analysis container.

    Earlier versions used hand-written cue arrays such as "khác với",
    "nặng hơn", "điều trị", etc. That made the retriever brittle and created
    hidden hard-coded features. This version deliberately does not infer a
    medical intent from phrase lists. It only carries data-driven retrieval
    controls that downstream scorers can combine with graph/corpus evidence.
    """

    intent: str = "schema_neutral"
    confidence: float = 0.5
    cues: list[str] = field(default_factory=list)
    relation_type_boosts: dict[str, float] = field(default_factory=dict)
    enable_same_article_expansion: bool = True
    enable_source_diversity: bool = True
    evidence_slot_keywords: list[str] = field(default_factory=list)

    def boosted_relation_types(self) -> list[str]:
        return list(self.relation_type_boosts.keys())


class QueryAnalyzer:
    """Hard-code-free analyzer.

    The analyzer no longer contains keyword/cue lists and no longer maps query
    phrases to relation weights. It exists for backwards compatibility with the
    retrievers, while keeping the critical ranking path free of hand-written
    features. Query-conditioned behavior should come from:

    * extracted low/high-level keywords;
    * linked graph entities/relations;
    * title/source/entity anchor consistency;
    * corpus/graph statistics such as node specificity and score calibration.
    """

    def analyze(self, query: str, keywords: QueryKeywords | None = None) -> QueryAnalysis:
        norm_query = self._norm(query)
        low_level = list(getattr(keywords, "low_level_keywords", []) or []) if keywords is not None else []
        high_level = list(getattr(keywords, "high_level_keywords", []) or []) if keywords is not None else []

        # Confidence here is only a diagnostic measure of how much query signal
        # was available. It is not an intent classifier and does not activate
        # relation-specific boosts.
        token_count = len(norm_query.split())
        keyword_count = len([item for item in [*low_level, *high_level] if str(item or "").strip()])
        confidence = 0.45
        confidence += min(token_count, 12) / 12.0 * 0.15
        confidence += min(keyword_count, 8) / 8.0 * 0.25
        confidence = max(0.0, min(0.85, confidence))

        return QueryAnalysis(
            intent="schema_neutral",
            confidence=confidence,
            cues=[],
            relation_type_boosts={},
            enable_same_article_expansion=True,
            enable_source_diversity=True,
            evidence_slot_keywords=[],
        )

    def _norm(self, text: str) -> str:
        text = (text or "").strip().lower().replace("đ", "d")
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()
