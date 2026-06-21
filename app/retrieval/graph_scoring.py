from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from app.retrieval.query_analyzer import QueryAnalysis
from app.retrieval.retrieval_models import QueryKeywords, RetrievedChunk, RetrievedEntity, RetrievedRelation


@dataclass(frozen=True)
class QueryAnchorProfile:
    """Query-side anchors shared by graph retrievers.

    The profile is intentionally derived from the query, keyword extractor and
    intent/operator analysis. It does not contain disease-specific or benchmark-
    specific keyword lists. Short symptom-like unigrams may still appear in
    ``query_terms``, but they are treated as weak lexical overlap, not anchors.
    """

    query_norm: str
    anchor_phrases: list[str] = field(default_factory=list)
    query_terms: list[str] = field(default_factory=list)
    slot_cues: list[str] = field(default_factory=list)
    operator_cues: list[str] = field(default_factory=list)
    term_specificity: dict[str, float] = field(default_factory=dict)
    local_document_frequency: dict[str, int] = field(default_factory=dict)
    local_document_count: int = 0
    local_specificity_applied: bool = False


@dataclass(frozen=True)
class GraphScoringConfig:
    """Query-conditioned scoring utilities for graph retrievers.

    This module is intentionally deterministic and data-driven. It does not
    special-case benchmark questions. The same signals are used by LightRAG,
    HippoRAG and mixed fusion:
    - lexical/semantic anchor consistency with low-level query concepts;
    - relation-type compatibility with the query profile;
    - IDF-like node specificity from graph mention statistics;
    - passage/chunk evidence overlap.
    """

    lexical_anchor_bonus: float = 1.15
    partial_anchor_bonus: float = 0.45
    disease_type_bonus: float = 0.35
    relation_type_scale: float = 0.35
    endpoint_anchor_scale: float = 0.90
    endpoint_specificity_scale: float = 0.25
    unanchored_relation_penalty: float = 0.55
    query_term_chunk_weight: float = 0.08
    slot_term_chunk_weight: float = 0.16
    anchor_term_chunk_weight: float = 0.26
    generic_term_chunk_weight: float = 0.04
    article_title_anchor_weight: float = 0.40
    slot_anchor_gate_floor: float = 0.05
    relation_chunk_gate_floor: float = 0.15
    query_phrase_limit: int = 96
    local_specificity_weight: float = 0.65
    fallback_specificity_weight: float = 0.35
    generic_match_min_specificity: float = 0.34
    generic_match_max_df_ratio: float = 0.60
    generic_match_min_token_len: int = 4
    generic_match_max_terms: int = 4

    # Mechanical guards for mined anchors. Trusted low-level keyword phrases may
    # be shorter, but raw n-grams from the question must contain enough
    # character-level information before they can behave as anchors.
    anchor_match_min_specificity: float = 0.26
    anchor_match_max_df_ratio: float = 0.78


class GraphQueryScorer:
    """Graph scoring utilities with no hand-written query features.

    This class intentionally avoids cue arrays, role-keyword dictionaries and
    medical-content stoplists. Candidate anchors are generated mechanically from
    query/keyword n-grams and then weakened/strengthened by structural signals
    such as title/entity matches, phrase length and graph node specificity.
    """

    def __init__(self, config: GraphScoringConfig | None = None):
        self.config = config or GraphScoringConfig()

    # ------------------------------------------------------------------
    # Query terms / normalization
    # ------------------------------------------------------------------

    def build_query_anchor_profile(
        self,
        query: str,
        keywords: QueryKeywords | None = None,
        analysis: QueryAnalysis | None = None,
    ) -> QueryAnchorProfile:
        """Build a reusable query profile without disease-specific keyword arrays.

        Anchors are mostly 2-5 token phrases plus long/specific unigrams from
        the query and low-level keywords. Slot cues are kept separate and are
        expected to be gated by anchor support at chunk-scoring time.
        """
        norm_query = self.norm(query)
        low_level = list(getattr(keywords, "low_level_keywords", []) or []) if keywords is not None else []
        high_level = list(getattr(keywords, "high_level_keywords", []) or []) if keywords is not None else []

        # Low-level keywords come from the query keyword extractor and are
        # treated as trusted query concepts. Raw n-grams mined from the question
        # are useful for context, but must pass a stricter mechanical filter so
        # filler fragments such as short grammatical phrases cannot become
        # anchors.
        anchor_candidates: list[tuple[str, bool]] = []
        anchor_candidates.extend((item, True) for item in low_level)
        anchor_candidates.extend((item, False) for item in self._phrase_candidates(query))
        anchor_candidates.extend((item, False) for item in self._ngram_phrase_candidates(norm_query))

        # High-level keywords from the keyword extractor are often abstract
        # facets (for example, generated section/role labels). They are not
        # content anchors unless the same phrase is actually present in the raw
        # query. This is still data-driven: the query text itself decides whether
        # a high-level phrase can act as an anchor.
        for item in high_level:
            item_norm = self.norm(item)
            if item_norm and self._contains_or_substring_norm(norm_query, item_norm):
                anchor_candidates.append((item, False))

        anchor_phrases = self._rank_anchor_candidates(
            anchor_candidates,
            limit=max(int(self.config.query_phrase_limit), 16),
        )

        # Keep the fields for backwards-compatible debug output, but do not
        # synthesize slot/operator cues from hard-coded lists.
        slot_cues: list[str] = []
        operator_cues: list[str] = []

        query_terms = self._dedupe_norm(
            token for token in norm_query.split() if self._is_query_token(token)
        )[:32]

        all_terms = self._dedupe_norm([*anchor_phrases, *query_terms, *slot_cues, *operator_cues])
        term_specificity = {term: self.term_specificity(term) for term in all_terms}
        return QueryAnchorProfile(
            query_norm=norm_query,
            anchor_phrases=anchor_phrases,
            query_terms=query_terms,
            slot_cues=slot_cues,
            operator_cues=operator_cues,
            term_specificity=term_specificity,
            local_document_frequency={},
            local_document_count=0,
            local_specificity_applied=False,
        )

    def anchor_terms(self, query: str, keywords: QueryKeywords | None = None) -> list[str]:
        return self.build_query_anchor_profile(query, keywords, None).anchor_phrases[:16]

    def slot_terms(self, keywords: QueryKeywords | None, analysis: QueryAnalysis | None) -> list[str]:
        return self.build_query_anchor_profile("", keywords, analysis).slot_cues[:24]

    def important_terms(self, query: str) -> list[str]:
        return self.build_query_anchor_profile(query, None, None).query_terms[:18]

    def norm(self, text: str) -> str:
        text = (text or "").strip().lower().replace("đ", "d")
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _phrase_candidates(self, query: str) -> list[str]:
        norm_query = self.norm(query)
        # Punctuation-only segmentation. Segments are still treated as mined
        # anchors, so they must pass the same mechanical content filter as
        # n-grams.
        parts = re.split(r"[,;:/?!.]+", norm_query)
        result = []
        for part in parts:
            part = re.sub(r"\s+", " ", part).strip()
            if self._is_mined_anchor_phrase(part):
                result.append(part)
        return result

    def _ngram_phrase_candidates(self, norm_query: str) -> list[str]:
        tokens = [tok for tok in norm_query.split() if tok]
        result: list[str] = []
        for n in range(5, 1, -1):
            for i in range(0, max(len(tokens) - n + 1, 0)):
                phrase = " ".join(tokens[i : i + n])
                if self._is_mined_anchor_phrase(phrase):
                    result.append(phrase)
        return result

    def _rank_anchor_candidates(self, values: Iterable[str | tuple[str, bool]], limit: int) -> list[str]:
        """Rank query phrase candidates without lexical feature lists.

        ``values`` may mark low-level keyword phrases as trusted query concepts.
        Trusted does not mean disease-specific hard-coding; it only means the
        upstream keyword extractor identified the phrase as a query concept.
        Raw mined n-grams remain stricter so generic grammatical fragments do
        not become anchors.
        """
        candidates: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        for index, item in enumerate(values):
            trusted = False
            value = item
            if isinstance(item, tuple):
                value, trusted = item
            key = self.norm(str(value or ""))
            if not key or key in seen:
                continue
            if trusted:
                if not self._is_trusted_anchor_phrase(key):
                    continue
            else:
                if not self._is_mined_anchor_phrase(key):
                    continue
            seen.add(key)
            tokens = key.split()
            token_count = len(tokens)
            char_count = len(key.replace(" ", ""))
            numeric_bonus = 0.08 if any(ch.isdigit() for ch in key) else 0.0
            length_bonus = min(char_count, 40) / 40.0 * 0.08
            phrase_bonus = min(token_count, 5) / 5.0 * 0.06
            compact_bonus = 0.06 if 2 <= token_count <= 3 else 0.0
            trusted_bonus = 0.16 if trusted else 0.0
            short_token_penalty = sum(1 for tok in tokens if len(tok) <= 2) * 0.08
            avg_token_len = sum(len(tok) for tok in tokens) / max(token_count, 1)
            token_quality_bonus = min(avg_token_len, 8.0) / 8.0 * 0.05
            score = (
                self.term_specificity(key)
                + trusted_bonus
                + length_bonus
                + phrase_bonus
                + compact_bonus
                + token_quality_bonus
                + numeric_bonus
                - short_token_penalty
            )
            candidates.append((score, -index, key))

        ranked = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)
        return [key for _, _, key in ranked[: max(1, limit)]]

    def select_search_phrases(self, profile: QueryAnchorProfile, limit: int = 8) -> list[str]:
        """Choose a compact, length-diverse subset of query phrases for seed search.

        ``anchor_phrases`` may contain many overlapping 4-5 token windows. Those
        are useful for body reranking, but vector/entity seed search often needs
        shorter exact phrases as well. This selector is purely mechanical: it
        balances phrase specificity, token quality and n-gram length diversity,
        without any language stoplist or medical keyword list.
        """
        max_items = max(int(limit or 0), 0)
        if max_items <= 0:
            return []

        candidates = self._dedupe_norm(profile.anchor_phrases)
        if not candidates:
            return []

        by_len: dict[int, list[tuple[float, str]]] = {}
        for phrase in candidates:
            tokens = self.norm(phrase).split()
            if not tokens:
                continue
            token_count = min(len(tokens), 5)
            by_len.setdefault(token_count, []).append((self._search_phrase_score(phrase, profile), phrase))

        for values in by_len.values():
            values.sort(key=lambda item: item[0], reverse=True)

        selected: list[str] = []
        selected_token_sets: list[set[str]] = []

        # First pass: preserve length diversity so compact 2-3 token anchors are
        # not crowded out by longer overlapping windows.
        for token_count in (2, 3, 4, 5, 1):
            for _, phrase in by_len.get(token_count, [])[:2]:
                if self._can_add_search_phrase(phrase, selected_token_sets):
                    selected.append(phrase)
                    selected_token_sets.append(set(self.norm(phrase).split()))
                    break
            if len(selected) >= max_items:
                return selected[:max_items]

        # Second pass: fill remaining slots by score with overlap control.
        all_candidates = sorted(
            ((score, phrase) for values in by_len.values() for score, phrase in values),
            key=lambda item: item[0],
            reverse=True,
        )
        for _, phrase in all_candidates:
            if phrase in selected:
                continue
            if self._can_add_search_phrase(phrase, selected_token_sets):
                selected.append(phrase)
                selected_token_sets.append(set(self.norm(phrase).split()))
            if len(selected) >= max_items:
                break

        if len(selected) < max_items:
            for _, phrase in all_candidates:
                if phrase not in selected:
                    selected.append(phrase)
                if len(selected) >= max_items:
                    break
        return selected[:max_items]

    def _search_phrase_score(self, phrase: str, profile: QueryAnchorProfile) -> float:
        norm_phrase = self.norm(phrase)
        tokens = norm_phrase.split()
        if not tokens:
            return 0.0
        specificity = float(profile.term_specificity.get(norm_phrase, self.term_specificity(norm_phrase)) or 0.0)
        avg_len = sum(len(tok) for tok in tokens) / max(len(tokens), 1)
        token_quality = min(avg_len, 8.0) / 8.0
        compactness = 1.0 / (1.0 + abs(len(tokens) - 3))
        short_penalty = sum(1 for tok in tokens if len(tok) <= 2) * 0.12
        found_bonus = 0.10 if profile.local_document_frequency.get(norm_phrase, 0) > 0 else 0.0
        return specificity + 0.16 * compactness + 0.10 * token_quality + found_bonus - short_penalty

    def _can_add_search_phrase(self, phrase: str, selected_token_sets: list[set[str]]) -> bool:
        tokens = set(self.norm(phrase).split())
        if not tokens:
            return False
        for old in selected_token_sets:
            if not old:
                continue
            overlap = len(tokens & old) / max(min(len(tokens), len(old)), 1)
            if overlap >= 0.85:
                return False
        return True

    def _dedupe_norm(self, values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = self.norm(value)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    def _is_query_token(self, token: str) -> bool:
        token = self.norm(token)
        # Purely mechanical filter: no stopword or medical keyword list.
        return bool(token and len(token) >= 3)

    def _is_anchor_phrase(self, phrase: str) -> bool:
        # Backward-compatible alias used by older callers.
        return self._is_mined_anchor_phrase(phrase) or self._is_trusted_anchor_phrase(phrase)

    def _is_trusted_anchor_phrase(self, phrase: str) -> bool:
        """Mechanical filter for low-level keyword concepts.

        Low-level keywords are allowed to be shorter than mined n-grams because
        they are produced by the query keyword extractor. They are still scored
        by local specificity later, so broad terms appearing in most candidate
        chunks become weak automatically.
        """
        phrase = self.norm(phrase)
        if not phrase or len(phrase) < 3:
            return False
        tokens = [tok for tok in phrase.split() if tok]
        if not tokens or len(tokens) > 5:
            return False
        if len(tokens) == 1:
            return len(tokens[0]) >= 3 or any(ch.isdigit() for ch in tokens[0])
        return all(len(tok) >= 2 or any(ch.isdigit() for ch in tok) for tok in tokens)

    def _is_mined_anchor_phrase(self, phrase: str) -> bool:
        """Stricter filter for raw query n-grams.

        This is deliberately feature-free: no stopword list, no medical keyword
        list. A mined phrase must carry enough character-level information:
        either a long token, two medium tokens, or a numeric marker. This drops
        filler-like fragments such as short grammatical windows while preserving
        compact context phrases such as two medium medical/context tokens.
        """
        phrase = self.norm(phrase)
        if not phrase or len(phrase) < 5:
            return False
        tokens = [tok for tok in phrase.split() if tok]
        if not tokens or len(tokens) > 5:
            return False
        if len(tokens) == 1:
            return len(tokens[0]) >= 6 or any(ch.isdigit() for ch in tokens[0])

        has_numeric = any(any(ch.isdigit() for ch in tok) for tok in tokens)
        long_tokens = sum(1 for tok in tokens if len(tok) >= 6)
        medium_tokens = sum(1 for tok in tokens if len(tok) >= 4)
        very_short_tokens = sum(1 for tok in tokens if len(tok) <= 2)

        if has_numeric:
            return True
        if very_short_tokens >= 2:
            return False
        if long_tokens >= 1:
            return True
        if medium_tokens >= 2:
            return True
        return False

    # ------------------------------------------------------------------
    # Specificity / anchors
    # ------------------------------------------------------------------

    def node_specificity(self, chunk_count: int, total_chunks: int) -> float:
        """Smooth IDF-like node specificity.

        The previous 1/chunk_count formula can over-punish exact disease anchors
        that naturally occur in many chunks of their own article. This keeps the
        HippoRAG specificity idea but uses a smoother IDF curve.
        """
        total = max(int(total_chunks or 0), 1)
        count = max(int(chunk_count or 0), 1)
        return 1.0 + math.log((total + 1.0) / (count + 1.0))

    def term_specificity(self, term: str) -> float:
        """Heuristic fallback until corpus-level DF/IDF stats are materialized.

        No medical content list is used. Multi-token phrases and long unigrams
        get higher specificity; short unigrams stay weak and cannot dominate
        anchor-gated scoring by themselves.
        """
        norm_term = self.norm(term)
        tokens = [tok for tok in norm_term.split() if tok]
        if not tokens:
            return 0.0
        if len(tokens) == 1:
            token = tokens[0]
            score = 0.16
            if len(token) >= 10:
                score += 0.38
            elif len(token) >= 7:
                score += 0.26
            elif len(token) >= 5:
                score += 0.10
            if any(ch.isdigit() for ch in token):
                score += 0.18
        elif len(tokens) == 2:
            # Two-token phrases are useful candidates, but they should not open
            # a full slot/relation gate unless they also match a strong
            # title/entity/source anchor or receive corpus-local support.
            score = 0.30
            score += min(sum(1 for tok in tokens if len(tok) >= 6) * 0.09, 0.18)
            if any(any(ch.isdigit() for ch in tok) for tok in tokens):
                score += 0.12
        else:
            score = 0.42 + 0.13 * min(len(tokens), 5)
            score += min(sum(1 for tok in tokens if len(tok) >= 6) * 0.08, 0.24)
        # No stopword/role-cue penalties here. Corpus-level DF/IDF can later
        # replace this fallback, but this function must not contain hand-written
        # lexical feature lists.
        return max(0.05, min(1.35, score))

    def enrich_profile_with_corpus(
        self,
        profile: QueryAnchorProfile,
        documents: Iterable[str],
    ) -> QueryAnchorProfile:
        """Reweight query terms by the actual candidate corpus.

        This is a lightweight, retrieval-time DF/IDF estimate. It avoids manual
        generic keyword lists: a term becomes weak if it appears in many candidate
        documents, and stronger if it appears in only a few. Terms not found in
        the current candidate pool are kept weak because they cannot explain any
        retrieved chunk.
        """
        docs = [self.norm(doc) for doc in documents if self.norm(doc)]
        if not docs:
            return profile

        terms = self._dedupe_norm([
            *profile.anchor_phrases,
            *profile.query_terms,
            *profile.slot_cues,
            *profile.operator_cues,
        ])
        if not terms:
            return profile

        total_docs = len(docs)
        df: dict[str, int] = {}
        specificity: dict[str, float] = {}
        log_den = math.log(total_docs + 1.5) if total_docs > 0 else 1.0
        for term in terms:
            norm_term = self.norm(term)
            count = sum(1 for doc in docs if self._contains_or_substring_norm(doc, norm_term))
            df[norm_term] = count
            fallback = profile.term_specificity.get(norm_term, self.term_specificity(norm_term))
            if count <= 0:
                value = min(fallback, 0.22)
            else:
                idf = math.log((total_docs + 1.0) / (count + 0.5))
                idf_norm = max(0.0, min(1.0, idf / max(log_den, 1e-9)))
                token_count = len(norm_term.split())
                phrase_factor = min(token_count, 5) / 5.0
                value = (
                    float(self.config.fallback_specificity_weight) * fallback
                    + float(self.config.local_specificity_weight) * idf_norm
                    + 0.08 * phrase_factor
                )
            specificity[norm_term] = max(0.03, min(1.35, value))

        return QueryAnchorProfile(
            query_norm=profile.query_norm,
            anchor_phrases=profile.anchor_phrases,
            query_terms=profile.query_terms,
            slot_cues=profile.slot_cues,
            operator_cues=profile.operator_cues,
            term_specificity=specificity,
            local_document_frequency=df,
            local_document_count=total_docs,
            local_specificity_applied=True,
        )

    def profile_debug(self, profile: QueryAnchorProfile, max_terms: int = 20) -> dict:
        terms = sorted(
            profile.term_specificity.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:max_terms]
        return {
            "query_norm": profile.query_norm,
            "anchor_phrases_top": profile.anchor_phrases[:max_terms],
            "query_terms_top": profile.query_terms[:max_terms],
            "term_specificity_top": [{"term": term, "specificity": value} for term, value in terms],
            "local_specificity_applied": profile.local_specificity_applied,
            "local_document_count": profile.local_document_count,
            "local_document_frequency_top": {
                term: profile.local_document_frequency.get(term, 0)
                for term, _ in terms[:max_terms]
            },
        }

    def text_anchor_score(self, text: str, anchor_terms: Iterable[str]) -> float:
        haystack = self.norm(text)
        score = 0.0
        for term in anchor_terms:
            term = self.norm(term)
            if not term:
                continue
            if self._contains_phrase(haystack, term):
                score += 1.0
            elif len(term) >= 6 and (term in haystack or haystack in term):
                score += 0.45
        return score

    def score_text_against_profile(self, text: str, profile: QueryAnchorProfile) -> dict[str, float | list[str]]:
        haystack = self.norm(text)
        raw_anchor_matches = [
            term for term in profile.anchor_phrases
            if term and self._contains_or_substring_norm(haystack, term)
        ]
        anchor_matches = self._filter_anchor_matches(raw_anchor_matches, profile)
        slot_matches = [
            term for term in profile.slot_cues
            if term and self._contains_or_substring_norm(haystack, term)
        ]
        query_matches = [
            term for term in profile.query_terms
            if term and self._contains_or_substring_norm(haystack, term)
        ]

        anchor_token_set: set[str] = set()
        for term in anchor_matches:
            anchor_token_set.update(self.norm(term).split())
        slot_token_set: set[str] = set()
        for term in slot_matches:
            slot_token_set.update(self.norm(term).split())
        generic_candidates = [
            term for term in query_matches
            if term not in anchor_token_set and term not in slot_token_set
        ]
        generic_matches = self._filter_generic_matches(generic_candidates, profile)

        anchor_values = [profile.term_specificity.get(term, self.term_specificity(term)) for term in anchor_matches[:8]]
        slot_values = [profile.term_specificity.get(term, self.term_specificity(term)) for term in slot_matches[:8]]
        query_values = [profile.term_specificity.get(term, self.term_specificity(term)) for term in query_matches[:10]]
        generic_values = [min(profile.term_specificity.get(term, self.term_specificity(term)), 0.28) for term in generic_matches]

        anchor_score = sum(anchor_values)
        slot_score = sum(slot_values)
        query_score = sum(query_values)
        generic_score = sum(generic_values)
        return {
            "anchor_score": float(anchor_score),
            "anchor_max_specificity": float(max(anchor_values) if anchor_values else 0.0),
            "slot_score": float(slot_score),
            "query_score": float(query_score),
            "generic_score": float(generic_score),
            "anchor_matches": anchor_matches[:8],
            "raw_anchor_matches": raw_anchor_matches[:10],
            "slot_matches": slot_matches[:8],
            "query_matches": query_matches[:10],
            "generic_matches": generic_matches[:10],
        }

    def _filter_anchor_matches(self, terms: Iterable[str], profile: QueryAnchorProfile) -> list[str]:
        """Filter mined anchors by local specificity instead of stopword lists.

        A phrase can match text but still be too broad in the current candidate
        pool. Broad anchors are kept in debug as ``raw_anchor_matches`` but do
        not open the anchor gate or add score.
        """
        kept: list[str] = []
        total_docs = max(int(profile.local_document_count or 0), 1)
        for raw in terms:
            term = self.norm(raw)
            if not term:
                continue
            specificity = float(profile.term_specificity.get(term, self.term_specificity(term)) or 0.0)
            df = int(profile.local_document_frequency.get(term, 0) or 0)
            df_ratio = df / total_docs if profile.local_specificity_applied else 0.0
            if specificity < float(self.config.anchor_match_min_specificity):
                continue
            if profile.local_specificity_applied and df_ratio > float(self.config.anchor_match_max_df_ratio):
                continue
            kept.append(term)
        return kept

    def _filter_generic_matches(self, terms: Iterable[str], profile: QueryAnchorProfile) -> list[str]:
        """Keep only mechanically specific unigram overlap as weak evidence.

        This replaces stopword/generic-word lists. A token is allowed to
        contribute generic lexical evidence only when it is long enough, not too
        frequent in the local candidate pool, and has enough local specificity.
        Short/common words may still appear in debug ``query_matches`` but they
        do not move the score.
        """
        kept: list[str] = []
        total_docs = max(int(profile.local_document_count or 0), 1)
        for raw in terms:
            term = self.norm(raw)
            if not term or " " in term:
                continue
            if len(term) < int(self.config.generic_match_min_token_len):
                continue
            specificity = float(profile.term_specificity.get(term, self.term_specificity(term)) or 0.0)
            df = int(profile.local_document_frequency.get(term, 0) or 0)
            df_ratio = df / total_docs if profile.local_specificity_applied else 0.0
            if specificity < float(self.config.generic_match_min_specificity):
                continue
            if profile.local_specificity_applied and df_ratio > float(self.config.generic_match_max_df_ratio):
                continue
            kept.append(term)
            if len(kept) >= int(self.config.generic_match_max_terms):
                break
        return kept

    def anchor_gate(
        self,
        *,
        anchor_score: float = 0.0,
        title_anchor_score: float = 0.0,
        entity_anchor_score: float = 0.0,
        anchor_max_specificity: float = 0.0,
    ) -> float:
        # Title/entity/source anchors are much stronger than body-only symptom
        # overlap. Body-only anchors may guide ranking, but they should not fully
        # unlock slot/relation boosts.
        hard_support = max(float(title_anchor_score or 0.0), float(entity_anchor_score or 0.0))
        if hard_support >= 0.85:
            return 1.0
        if hard_support >= 0.40:
            return 0.65

        body_total = float(anchor_score or 0.0)
        body_max = float(anchor_max_specificity or 0.0)
        if body_max >= 0.85 or body_total >= 1.60:
            return 0.65
        if body_max >= 0.30 or body_total > 0.0:
            return 0.30
        return 0.0

    def entity_anchor_multiplier(
        self,
        entity: RetrievedEntity,
        query: str,
        keywords: QueryKeywords | None,
    ) -> tuple[float, list[str]]:
        terms = self.anchor_terms(query, keywords)
        names = [entity.name, entity.normalized_name, *(entity.aliases or []), *(entity.surface_forms or [])]
        text = " ".join(str(x or "") for x in names)
        lexical = self.text_anchor_score(text, terms)
        reasons: list[str] = []
        multiplier = 1.0
        if lexical >= 1.0:
            multiplier += self.config.lexical_anchor_bonus
            reasons.append("lexical_anchor")
        elif lexical > 0.0:
            multiplier += self.config.partial_anchor_bonus
            reasons.append("partial_anchor")
        if entity.entity_type == "BENH_LY" and lexical > 0.0:
            multiplier += self.config.disease_type_bonus
            reasons.append("disease_anchor")
        return multiplier, reasons

    def relation_type_weight(self, relation_type: str, analysis: QueryAnalysis | None) -> float:
        if analysis is None or not analysis.relation_type_boosts:
            return 1.0
        boost = float(analysis.relation_type_boosts.get(relation_type, 0.0) or 0.0)
        if boost <= 0.0:
            # Softly demote instead of dropping. This prevents query-analyzer
            # mistakes from wiping all useful relations.
            return 0.70
        return 1.0 + self.config.relation_type_scale * boost

    def rerank_relations(
        self,
        relations: list[RetrievedRelation],
        *,
        seed_entity_scores: dict[str, float],
        chunk_counts: dict[str, int],
        total_chunks: int,
        analysis: QueryAnalysis | None,
        query: str,
        keywords: QueryKeywords | None,
    ) -> tuple[list[RetrievedRelation], dict[str, dict[str, float | list[str]]]]:
        if not relations:
            return relations, {}

        terms = self.anchor_terms(query, keywords)
        debug: dict[str, dict[str, float | list[str]]] = {}
        ranked: list[RetrievedRelation] = []
        for relation in relations:
            original = float(relation.score or 0.0)
            subject_anchor = float(seed_entity_scores.get(relation.subject_entity_id, 0.0) or 0.0)
            object_anchor = float(seed_entity_scores.get(relation.object_entity_id, 0.0) or 0.0)
            endpoint_anchor = max(subject_anchor, object_anchor)

            lexical_endpoint = self.text_anchor_score(
                " ".join([relation.subject_name, relation.object_name]),
                terms,
            )
            subject_spec = self.node_specificity(chunk_counts.get(relation.subject_entity_id, 1), total_chunks)
            object_spec = self.node_specificity(chunk_counts.get(relation.object_entity_id, 1), total_chunks)
            endpoint_spec = max(subject_spec, object_spec)
            type_weight = self.relation_type_weight(relation.relation_type, analysis)

            anchor_factor = 1.0 + self.config.endpoint_anchor_scale * endpoint_anchor + 0.25 * lexical_endpoint
            if endpoint_anchor <= 0.0 and lexical_endpoint <= 0.0:
                anchor_factor *= self.config.unanchored_relation_penalty

            specificity_factor = 1.0 + self.config.endpoint_specificity_scale * max(endpoint_spec - 1.0, 0.0)
            adjusted = original * type_weight * anchor_factor * specificity_factor

            relation.score = adjusted
            relation.metadata = {
                **dict(relation.metadata or {}),
                "anchor_original_score": original,
                "anchor_type_weight": type_weight,
                "anchor_endpoint_score": endpoint_anchor,
                "anchor_endpoint_lexical": lexical_endpoint,
                "anchor_endpoint_specificity": endpoint_spec,
                "anchor_adjusted_score": adjusted,
            }
            debug[relation.relation_id] = {
                "original": original,
                "type_weight": type_weight,
                "endpoint_anchor": endpoint_anchor,
                "endpoint_lexical": lexical_endpoint,
                "endpoint_specificity": endpoint_spec,
                "adjusted": adjusted,
            }
            ranked.append(relation)

        return sorted(ranked, key=lambda item: item.score, reverse=True), debug

    # ------------------------------------------------------------------
    # Passage/chunk scoring
    # ------------------------------------------------------------------

    def rerank_chunks(
        self,
        chunks: list[RetrievedChunk],
        *,
        query: str,
        keywords: QueryKeywords | None,
        analysis: QueryAnalysis | None,
        relation_chunk_boost: dict[str, float] | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        profile = self.build_query_anchor_profile(query, keywords, analysis)
        profile = self.enrich_profile_with_corpus(
            profile,
            [" ".join([chunk.title or "", chunk.section or "", chunk.subsection or "", chunk.text[:2000] or ""]) for chunk in chunks],
        )
        relation_chunk_boost = relation_chunk_boost or {}

        ranked: list[RetrievedChunk] = []
        for chunk in chunks:
            text = " ".join([chunk.title, chunk.section, chunk.subsection, chunk.text[:1600]])
            breakdown = self.score_text_against_profile(text, profile)
            title_breakdown = self.score_text_against_profile(chunk.title, profile)
            anchor_score = float(breakdown.get("anchor_score", 0.0) or 0.0)
            title_anchor = float(title_breakdown.get("anchor_score", 0.0) or 0.0)
            gate = self.anchor_gate(anchor_score=anchor_score, title_anchor_score=title_anchor)

            original = float(chunk.score or 0.0)
            slot_delta = float(breakdown.get("slot_score", 0.0) or 0.0) * self.config.slot_term_chunk_weight * max(gate, self.config.slot_anchor_gate_floor)
            relation_delta = float(relation_chunk_boost.get(chunk.chunk_id, 0.0) or 0.0) * max(gate, self.config.relation_chunk_gate_floor)
            boost = 0.0
            boost += float(breakdown.get("generic_score", 0.0) or 0.0) * self.config.generic_term_chunk_weight
            boost += anchor_score * self.config.anchor_term_chunk_weight
            boost += slot_delta
            boost += title_anchor * self.config.article_title_anchor_weight
            boost += relation_delta

            chunk.score = original + boost
            chunk.metadata = {
                **dict(chunk.metadata or {}),
                "query_anchor_original_score": original,
                "query_anchor_boost": boost,
                "query_anchor_gate": gate,
                "query_anchor_matches": breakdown.get("anchor_matches", [])[:8],
                "query_slot_matches": breakdown.get("slot_matches", [])[:8],
                "query_term_matches": breakdown.get("query_matches", [])[:8],
                "query_generic_matches": breakdown.get("generic_matches", [])[:8],
                "query_anchor_breakdown": breakdown,
                "query_anchor_profile_debug": self.profile_debug(profile, max_terms=12),
                "query_relation_delta": relation_delta,
            }
            ranked.append(chunk)
        return sorted(ranked, key=lambda item: item.score, reverse=True)

    def branch_confidence(self, chunks: list[RetrievedChunk], query: str, keywords: QueryKeywords | None) -> float:
        if not chunks:
            return 0.0
        top = chunks[: min(5, len(chunks))]
        article_counts: dict[str, int] = {}
        for chunk in top:
            key = chunk.source_url or chunk.article_id or chunk.title
            article_counts[key] = article_counts.get(key, 0) + 1
        concentration = max(article_counts.values()) / max(len(top), 1)
        anchor_hits = 0
        profile = self.build_query_anchor_profile(query, keywords, None)
        profile = self.enrich_profile_with_corpus(
            profile,
            [" ".join([chunk.title or "", chunk.text[:800] or ""]) for chunk in top],
        )
        for chunk in top:
            breakdown = self.score_text_against_profile(" ".join([chunk.title, chunk.text[:800]]), profile)
            if float(breakdown.get("anchor_score", 0.0) or 0.0) > 0.0:
                anchor_hits += 1
        anchor_coverage = anchor_hits / max(len(top), 1)
        # A branch that is merely concentrated on one article is not necessarily
        # trustworthy; it may simply be drifting consistently. Confidence is
        # therefore gated by observed anchor coverage, using only mined query
        # anchors and candidate text, not hand-written cue lists.
        base = 0.35 * concentration + 0.65 * anchor_coverage
        if anchor_coverage <= 0.0:
            base *= 0.35
        elif anchor_coverage < 0.25:
            base *= 0.60
        return max(0.0, min(1.0, base))

    def _contains_or_substring(self, text: str, term: str) -> bool:
        return self._contains_or_substring_norm(self.norm(text), term)

    def _contains_or_substring_norm(self, norm_text: str, term: str) -> bool:
        term = self.norm(term)
        if not term:
            return False
        return self._contains_phrase(norm_text, term) or (len(term) >= 4 and term in norm_text)

    def _contains_phrase(self, norm_text: str, norm_phrase: str) -> bool:
        if not norm_phrase:
            return False
        return re.search(rf"(^|\s){re.escape(norm_phrase)}($|\s)", norm_text) is not None
