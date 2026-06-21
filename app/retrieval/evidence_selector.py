from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Iterable

from app.retrieval.retrieval_models import RetrievedChunk


@dataclass
class EvidenceSelectionConfig:
    """Sentence-level evidence compaction for answer generation.

    The default scorer is local/statistical and feature-free: it does not use a
    disease list, symptom list, stopwords, operator cues or source exceptions.
    It estimates which sentences are most query-aligned from query phrases,
    token overlap and local sentence document frequency within each chunk.
    """

    enabled: bool = True
    max_sentences_per_chunk: int = 3
    sentence_window_size: int = 1
    max_sentence_chars: int = 420
    max_compact_chars_per_chunk: int = 1400
    min_sentence_chars: int = 20
    max_query_terms: int = 24
    max_query_phrases: int = 24
    include_title_header: bool = True
    scoring_mode: str = "local_statistical"


class SentenceEvidenceSelector:
    """Select compact sentence windows from retrieved chunks.

    This is a context relevance component, not a new retriever. It preserves the
    original chunk order and IDs, but replaces the text shown to the answer
    generator with the most relevant sentence windows. Retrieval metrics can
    still be computed on the original chunks.
    """

    def __init__(self, config: EvidenceSelectionConfig | None = None):
        self.config = config or EvidenceSelectionConfig()

    def select_for_chunks(self, query: str, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], dict]:
        if not self.config.enabled or not chunks:
            return chunks, {"enabled": False, "reason": "disabled_or_empty"}

        query_terms = self._query_terms(query)[: self.config.max_query_terms]
        query_phrases = self._query_phrases(query)[: self.config.max_query_phrases]
        selected_chunks: list[RetrievedChunk] = []
        debug_rows: list[dict] = []

        for chunk in chunks:
            selected, scores = self._select_sentences_for_chunk(
                query_terms=query_terms,
                query_phrases=query_phrases,
                chunk=chunk,
            )
            if not selected:
                selected_chunks.append(chunk)
                debug_rows.append({
                    "chunk_id": chunk.chunk_id,
                    "selected_count": 0,
                    "reason": "no_sentence_selected",
                })
                continue

            compact_text = self._build_compact_text(chunk, selected)
            metadata = {
                **dict(chunk.metadata or {}),
                "selected_sentences": selected,
                "sentence_scores": scores,
                "sentence_window_size": self.config.sentence_window_size,
                "sentence_selection_mode": self.config.scoring_mode,
                "original_chunk_id": chunk.chunk_id,
                "original_text_chars": len(chunk.text or chunk.contextualized_text or ""),
                "compact_text_chars": len(compact_text),
                "source_url": chunk.source_url,
            }
            selected_chunks.append(replace(chunk, text=compact_text, contextualized_text=compact_text, metadata=metadata))
            debug_rows.append({
                "chunk_id": chunk.chunk_id,
                "selected_count": len(selected),
                "top_score": scores[0]["score"] if scores else 0.0,
                "source_url": chunk.source_url,
            })

        return selected_chunks, {
            "enabled": True,
            "mode": self.config.scoring_mode,
            "max_sentences_per_chunk": self.config.max_sentences_per_chunk,
            "sentence_window_size": self.config.sentence_window_size,
            "query_terms_top": query_terms[:12],
            "query_phrases_top": query_phrases[:12],
            "chunks_processed": len(chunks),
            "chunks_with_selected_sentences": sum(1 for row in debug_rows if row.get("selected_count", 0) > 0),
            "rows_top": debug_rows[:20],
        }

    def _select_sentences_for_chunk(
        self,
        *,
        query_terms: list[str],
        query_phrases: list[str],
        chunk: RetrievedChunk,
    ) -> tuple[list[str], list[dict]]:
        raw_text = chunk.text or chunk.contextualized_text or ""
        sentences = self._split_sentences(raw_text)
        if not sentences:
            return [], []

        sentence_norms = [self._norm(sentence) for sentence in sentences]
        local_df = self._local_df(sentence_norms, [*query_terms, *query_phrases])
        total = max(len(sentences), 1)
        title_blob = self._norm(" ".join([chunk.title or "", chunk.section or "", chunk.subsection or ""]))

        scored: list[tuple[float, int, dict]] = []
        for idx, sentence in enumerate(sentences):
            norm_sentence = sentence_norms[idx]
            if len(sentence.strip()) < self.config.min_sentence_chars:
                continue
            token_hits = [term for term in query_terms if term and self._contains(norm_sentence, term)]
            phrase_hits = [phrase for phrase in query_phrases if phrase and self._contains(norm_sentence, phrase)]
            title_hits = [term for term in query_terms if term and self._contains(title_blob, term)]
            idf_score = 0.0
            for term in token_hits[:10]:
                df = max(local_df.get(term, 1), 1)
                idf_score += math.log((total + 1.0) / df)
            phrase_score = 0.0
            for phrase in phrase_hits[:6]:
                df = max(local_df.get(phrase, 1), 1)
                phrase_score += 1.25 * math.log((total + 1.0) / df)
            position_bonus = 0.12 / (idx + 1) if idx <= 4 else 0.0
            title_support = min(len(title_hits), 3) * 0.08
            length_penalty = max(0.0, (len(sentence) - self.config.max_sentence_chars) / self.config.max_sentence_chars) * 0.08
            score = idf_score + phrase_score + title_support + position_bonus - length_penalty
            scored.append((score, idx, {
                "index": idx,
                "score": float(score),
                "token_hits": token_hits[:8],
                "phrase_hits": phrase_hits[:6],
                "title_hits": title_hits[:6],
                "text": self._trim(sentence, self.config.max_sentence_chars),
            }))

        if not scored:
            return [], []

        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        top = scored[: max(1, self.config.max_sentences_per_chunk)]
        selected_indices = self._expand_windows([idx for _, idx, _ in top], len(sentences))
        selected_sentences = [self._trim(sentences[idx], self.config.max_sentence_chars) for idx in selected_indices]
        score_rows = [row for _, _, row in top]
        return selected_sentences, score_rows

    def _build_compact_text(self, chunk: RetrievedChunk, selected: list[str]) -> str:
        parts: list[str] = []
        if self.config.include_title_header:
            header = " | ".join(x for x in [chunk.title, chunk.section, chunk.subsection] if x)
            if header:
                parts.append(header)
            if chunk.source_url:
                parts.append(f"Nguồn: {chunk.source_url}")
        parts.extend(selected)
        text = "\n".join(part.strip() for part in parts if part and part.strip())
        return text[: max(256, int(self.config.max_compact_chars_per_chunk or 1400))]

    def _split_sentences(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", (text or "").strip())
        if not text:
            return []
        # Vietnamese medical pages often use semicolons and bullets; keep this
        # punctuation-only, not word-list based.
        parts = re.split(r"(?<=[.!?。！？])\s+|\s+[•●▪-]\s+|\n+", text)
        result: list[str] = []
        for part in parts:
            part = part.strip(" -•●▪\t\r\n")
            if not part:
                continue
            if len(part) > self.config.max_sentence_chars * 1.8:
                result.extend(self._split_long_sentence(part))
            else:
                result.append(part)
        return result

    def _split_long_sentence(self, sentence: str) -> list[str]:
        parts = re.split(r"(?<=[;:])\s+|,\s+(?=[A-ZÀ-Ỵ0-9])", sentence)
        cleaned = [part.strip() for part in parts if part.strip()]
        if len(cleaned) <= 1:
            return [sentence]
        return cleaned

    def _expand_windows(self, indices: list[int], sentence_count: int) -> list[int]:
        window = max(0, int(self.config.sentence_window_size or 0))
        selected: set[int] = set()
        for idx in indices:
            lo = max(0, idx - window)
            hi = min(sentence_count - 1, idx + window)
            selected.update(range(lo, hi + 1))
        return sorted(selected)

    def _local_df(self, sentence_norms: list[str], terms: Iterable[str]) -> dict[str, int]:
        df: dict[str, int] = {}
        for term in terms:
            norm_term = self._norm(term)
            if not norm_term:
                continue
            df[norm_term] = sum(1 for sentence in sentence_norms if self._contains(sentence, norm_term))
        return df

    def _query_terms(self, query: str) -> list[str]:
        values = []
        for token in self._norm(query).split():
            if len(token) >= 3 and token not in values:
                values.append(token)
        return values

    def _query_phrases(self, query: str) -> list[str]:
        tokens = self._norm(query).split()
        phrases: list[str] = []
        for n in range(5, 1, -1):
            for i in range(0, max(0, len(tokens) - n + 1)):
                phrase = " ".join(tokens[i : i + n])
                if self._is_phrase_like(phrase) and phrase not in phrases:
                    phrases.append(phrase)
        return phrases

    def _is_phrase_like(self, phrase: str) -> bool:
        tokens = phrase.split()
        if len(tokens) < 2 or len(tokens) > 5:
            return False
        long_tokens = sum(1 for tok in tokens if len(tok) >= 6)
        medium_tokens = sum(1 for tok in tokens if len(tok) >= 4)
        has_number = any(any(ch.isdigit() for ch in tok) for tok in tokens)
        return has_number or long_tokens >= 1 or medium_tokens >= 2

    def _contains(self, text_norm: str, term_norm: str) -> bool:
        if not text_norm or not term_norm:
            return False
        if " " in term_norm:
            return term_norm in text_norm
        return re.search(rf"(^|\s){re.escape(term_norm)}($|\s)", text_norm) is not None

    def _trim(self, text: str, limit: int) -> str:
        text = " ".join((text or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _norm(self, text: str) -> str:
        text = (text or "").strip().lower().replace("đ", "d")
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()
