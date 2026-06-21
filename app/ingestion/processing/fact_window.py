from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.ingestion.processing.ingestion_models import ArticleDocument, ChunkDocument
from app.ingestion.processing.chunker import estimate_token_count


@dataclass(frozen=True)
class FactWindowingStats:
    retrieval_chunk_count: int
    extraction_window_count: int
    split_parent_chunk_count: int
    max_tokens: int
    min_tokens: int
    overlap_sentences: int

    def model_dump(self) -> dict:
        return {
            "retrieval_chunk_count": self.retrieval_chunk_count,
            "extraction_window_count": self.extraction_window_count,
            "split_parent_chunk_count": self.split_parent_chunk_count,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "overlap_sentences": self.overlap_sentences,
        }


def make_fact_window_id(parent_chunk_id: str, window_index: int, text: str) -> str:
    raw = f"{parent_chunk_id}:fact_window:{window_index}:{text[:160]}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"{parent_chunk_id}_fw_{digest}"


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", str(text or "")) if block.strip()]
    result: list[str] = []
    for block in blocks:
        # Keep markdown headings as their own context blocks. For bullet-heavy
        # medical articles, each bullet often carries an independent fact, so
        # split very long bullet blocks by line before sentence splitting.
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) > 1 and any(line.lstrip().startswith(("- ", "* ", "+ ")) for line in lines):
            current: list[str] = []
            for line in lines:
                if line.lstrip().startswith(("- ", "* ", "+ ")) and current:
                    result.append("\n".join(current).strip())
                    current = [line]
                else:
                    current.append(line)
            if current:
                result.append("\n".join(current).strip())
        else:
            result.append(block)
    return result


def _split_sentences(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    # Vietnamese medical articles use ordinary full stops most of the time.
    # Keep delimiters attached to the sentence.
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences or [text]


def _block_to_units(block: str, max_tokens: int) -> list[str]:
    if estimate_token_count(block) <= max_tokens:
        return [block]

    # Long paragraph: split by sentences and repack sentences into small units.
    sentences = _split_sentences(block)
    if len(sentences) <= 1:
        return [block]

    units: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        if not current or estimate_token_count(candidate) <= max_tokens:
            current.append(sentence)
            continue
        units.append(" ".join(current).strip())
        current = [sentence]
    if current:
        units.append(" ".join(current).strip())
    return units


def _tail_sentences(text: str, n: int) -> str:
    if n <= 0:
        return ""
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    return " ".join(sentences[-n:]).strip()


def _pack_units(
    units: list[str],
    *,
    max_tokens: int,
    min_tokens: int,
    overlap_sentences: int,
) -> list[str]:
    windows: list[str] = []
    current: list[str] = []

    def current_text() -> str:
        return "\n\n".join(item for item in current if item.strip()).strip()

    def flush() -> None:
        nonlocal current
        text = current_text()
        if text:
            windows.append(text)
        current = []

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        candidate = "\n\n".join(current + [unit]).strip()
        if not current or estimate_token_count(candidate) <= max_tokens:
            current.append(unit)
            continue

        previous = current_text()
        flush()

        overlap = _tail_sentences(previous, overlap_sentences)
        if overlap and estimate_token_count(overlap + "\n\n" + unit) <= max_tokens:
            current = [overlap, unit]
        else:
            current = [unit]

    flush()

    # Merge tiny trailing windows into previous window when possible. This keeps
    # headings or one-line fragments from becoming isolated LLM calls.
    if min_tokens > 0 and len(windows) >= 2:
        merged: list[str] = []
        for window in windows:
            if merged and estimate_token_count(window) < min_tokens:
                candidate = (merged[-1] + "\n\n" + window).strip()
                if estimate_token_count(candidate) <= max_tokens:
                    merged[-1] = candidate
                    continue
            merged.append(window)
        windows = merged

    return windows


def _build_window_contextualized_text(article: ArticleDocument, parent: ChunkDocument, text: str) -> str:
    lines = [f"Bài viết: {article.title}"]
    heading_context = str(parent.metadata.get("heading_path_context") or "").strip()
    if heading_context:
        lines.append(f"Đường dẫn mục tại đầu chunk gốc: {heading_context}")
    if parent.section:
        lines.append(f"Mục gốc: {parent.section}")
    if parent.subsection:
        lines.append(f"Tiểu mục gốc: {parent.subsection}")
    lines.append("")
    lines.append(text.strip())
    return "\n".join(lines).strip()


def _make_window_chunk(
    article: ArticleDocument,
    parent: ChunkDocument,
    text: str,
    *,
    window_index: int,
    parent_window_count: int,
) -> ChunkDocument:
    chunk_id = make_fact_window_id(parent.chunk_id, window_index, text)
    contextualized_text = _build_window_contextualized_text(article, parent, text)
    metadata = {
        **dict(parent.metadata or {}),
        "chunk_role": "extraction_window",
        "extraction_window": True,
        "extraction_window_strategy": "fact_window_paragraph_sentence",
        "parent_chunk_id": parent.chunk_id,
        "parent_chunk_index": parent.chunk_index,
        "fact_window_index": window_index,
        "fact_window_count_for_parent": parent_window_count,
        "retrieval_chunk_id": parent.chunk_id,
        "retrieval_chunk_section": parent.section,
    }
    return ChunkDocument(
        chunk_id=chunk_id,
        article_id=parent.article_id,
        source_url=parent.source_url,
        title=parent.title,
        section=parent.section,
        subsection=parent.subsection,
        chunk_index=(parent.chunk_index * 1000) + window_index,
        heading_path=list(parent.heading_path or []),
        text=text.strip(),
        contextualized_text=contextualized_text,
        token_count=estimate_token_count(contextualized_text),
        metadata=metadata,
    )


def split_chunk_into_fact_windows(
    article: ArticleDocument,
    chunk: ChunkDocument,
    *,
    max_tokens: int = 450,
    min_tokens: int = 80,
    overlap_sentences: int = 1,
) -> list[ChunkDocument]:
    """Split one retrieval chunk into smaller LLM extraction windows.

    Retrieval chunks remain unchanged for Vector RAG. These windows are only for
    medical fact extraction, so the extractor sees less text per call while still
    getting article/heading context.
    """
    max_tokens = max(128, int(max_tokens or 450))
    min_tokens = max(0, int(min_tokens or 0))
    overlap_sentences = max(0, int(overlap_sentences or 0))

    if estimate_token_count(chunk.contextualized_text or chunk.text) <= max_tokens:
        return [chunk]

    blocks = _split_blocks(chunk.text)
    units: list[str] = []
    for block in blocks:
        units.extend(_block_to_units(block, max_tokens=max_tokens))

    window_texts = _pack_units(
        units,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_sentences=overlap_sentences,
    )

    if len(window_texts) <= 1:
        return [chunk]

    return [
        _make_window_chunk(
            article,
            chunk,
            text,
            window_index=i,
            parent_window_count=len(window_texts),
        )
        for i, text in enumerate(window_texts)
    ]


def build_fact_windows_for_chunks(
    article: ArticleDocument,
    chunks: list[ChunkDocument],
    *,
    max_tokens: int = 450,
    min_tokens: int = 80,
    overlap_sentences: int = 1,
) -> tuple[list[ChunkDocument], FactWindowingStats]:
    windows: list[ChunkDocument] = []
    split_parent_count = 0

    for chunk in chunks:
        chunk_windows = split_chunk_into_fact_windows(
            article,
            chunk,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            overlap_sentences=overlap_sentences,
        )
        if len(chunk_windows) > 1 or chunk_windows[0].chunk_id != chunk.chunk_id:
            split_parent_count += 1
        windows.extend(chunk_windows)

    stats = FactWindowingStats(
        retrieval_chunk_count=len(chunks),
        extraction_window_count=len(windows),
        split_parent_chunk_count=split_parent_count,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_sentences=overlap_sentences,
    )
    return windows, stats
