import hashlib
import re

from app.ingestion.processing.ingestion_models import (
    ArticleDocument,
    ChunkDocument,
    SectionDocument,
)


try:
    import tiktoken
except ImportError:
    tiktoken = None


_TOKENIZER = None


def estimate_token_count(text: str, model_name: str = "gpt-4o-mini") -> int:
    """
    Đếm token cho chunking.

    Ưu tiên dùng tiktoken để gần đúng với OpenAI models.
    Nếu chưa cài tiktoken thì fallback sang heuristic an toàn hơn.

    Lưu ý:
    - gpt-4o / gpt-4o-mini dùng encoding kiểu o200k_base.
    - Không dùng word_count * 1.3 nữa vì tiếng Việt dễ bị underestimate.
    """
    global _TOKENIZER

    if tiktoken is not None:
        if _TOKENIZER is None:
            try:
                _TOKENIZER = tiktoken.encoding_for_model(model_name)
            except KeyError:
                _TOKENIZER = tiktoken.get_encoding("o200k_base")

        return len(_TOKENIZER.encode(text))

    words = re.findall(r"\S+", text)
    return int(len(words) * 1.5)


def make_chunk_id(article_id: str, chunk_index: int, text: str) -> str:
    """
    Tạo chunk_id ổn định từ article_id + index + đoạn đầu text.
    """
    raw = f"{article_id}:{chunk_index}:{text[:120]}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"chunk_{h}"


def split_long_text_by_paragraphs(
    text: str,
    max_tokens: int,
    overlap_paragraphs: int = 1,
) -> list[str]:
    """
    Split section dài theo paragraph.

    Không semantic chunking ở đây vì:
    - bài y khoa đã có heading tự nhiên;
    - cần boundary ổn định cho evidence;
    - entity/relation extractor cần section context rõ.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        candidate = "\n\n".join(current + [paragraph])

        if estimate_token_count(candidate) <= max_tokens:
            current.append(paragraph)
            continue

        if current:
            chunks.append("\n\n".join(current))

        overlap = current[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
        current = overlap + [paragraph]

        # Nếu một paragraph đơn lẻ vẫn quá dài thì cắt mềm theo câu.
        if estimate_token_count("\n\n".join(current)) > max_tokens:
            long_text = "\n\n".join(current)
            sentence_chunks = split_long_paragraph_by_sentences(
                long_text,
                max_tokens=max_tokens,
            )

            if sentence_chunks:
                chunks.extend(sentence_chunks[:-1])
                current = [sentence_chunks[-1]]
            else:
                current = [paragraph]

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def split_long_paragraph_by_sentences(
    text: str,
    max_tokens: int,
) -> list[str]:
    """
    Fallback khi một paragraph quá dài.

    Cắt theo câu tiếng Việt/Anh tương đối đơn giản.
    Giữ Markdown trong câu, không plain-text hóa.
    """
    sentences = re.split(r"(?<=[.!?。])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence])

        if estimate_token_count(candidate) <= max_tokens:
            current.append(sentence)
            continue

        if current:
            chunks.append(" ".join(current))

        current = [sentence]

    if current:
        chunks.append(" ".join(current))

    return chunks


def _section_heading_lines(section: SectionDocument) -> list[str]:
    """
    Tạo heading Markdown cho một section khi pack nhiều section vào cùng chunk.

    Khi parse_sections, text của section không còn chứa heading gốc nữa.
    Nếu pack nhiều section mà không thêm lại heading, extractor sẽ mất context.
    """
    level = max(2, min(section.section_level, 6))
    heading = f"{'#' * level} {section.section_title}"

    return [heading, ""]


def _section_text_with_heading(section: SectionDocument) -> str:
    lines = _section_heading_lines(section)
    lines.append(section.text.strip())
    return "\n".join(lines).strip()


def build_contextualized_text(
    article: ArticleDocument,
    section: SectionDocument,
    text: str,
) -> str:
    """
    Tạo text có context cho một chunk chứa một section.

    Hàm này được giữ để tương thích với code cũ và dùng cho section quá dài
    phải split theo paragraph/sentence.
    """
    lines = [
        f"Bài viết: {article.title}",
    ]

    if section.parent_section_title:
        lines.append(f"Mục lớn: {section.parent_section_title}")

    lines.append(f"Mục: {section.section_title}")
    lines.append("")
    lines.append(text.strip())

    return "\n".join(lines).strip()


def build_packed_contextualized_text(
    article: ArticleDocument,
    sections: list[SectionDocument],
    text: str,
) -> str:
    """
    Tạo contextualized_text cho chunk chứa nhiều section liền kề.

    Giữ title bài viết + danh sách mục để extractor có context tổng quát,
    còn phần text bên dưới vẫn giữ heading Markdown từng section.
    """
    lines = [f"Bài viết: {article.title}"]

    section_titles = [section.section_title for section in sections]
    if section_titles:
        lines.append("Các mục trong chunk: " + " | ".join(section_titles))

    lines.append("")
    lines.append(text.strip())

    return "\n".join(lines).strip()


def make_packed_chunk(
    article: ArticleDocument,
    sections: list[SectionDocument],
    chunk_index: int,
) -> ChunkDocument:
    """
    Tạo một chunk từ nhiều section liền kề.

    Nguyên tắc:
    - Không cắt vỡ section nhỏ.
    - Thêm lại heading Markdown để không mất ngữ cảnh section.
    - Metadata lưu danh sách section để debug/audit.
    """
    text = "\n\n".join(
        _section_text_with_heading(section)
        for section in sections
        if section.text.strip()
    ).strip()

    contextualized_text = build_packed_contextualized_text(
        article=article,
        sections=sections,
        text=text,
    )

    section_titles = [section.section_title for section in sections]
    section_ids = [section.section_id for section in sections]
    section_orders = [section.order for section in sections]
    parent_titles = [
        section.parent_section_title
        for section in sections
        if section.parent_section_title
    ]

    return ChunkDocument(
        chunk_id=make_chunk_id(article.article_id, chunk_index, text),
        article_id=article.article_id,
        source_url=article.url,
        title=article.title,
        section=" | ".join(section_titles),
        subsection=None,
        chunk_index=chunk_index,
        text=text,
        contextualized_text=contextualized_text,
        token_count=estimate_token_count(contextualized_text),
        metadata={
            "section_ids": section_ids,
            "section_titles": section_titles,
            "section_orders": section_orders,
            "section_count": len(sections),
            "parent_section_titles": list(dict.fromkeys(parent_titles)),
            "chunking_strategy": "section_preserving_pack",
            "source": article.source,
            "category": article.category,
        },
    )


def make_single_section_chunk(
    article: ArticleDocument,
    section: SectionDocument,
    text: str,
    chunk_index: int,
    part_index: int | None = None,
) -> ChunkDocument:
    """
    Tạo chunk cho một section đơn lẻ.

    Dùng khi section quá dài nên phải split theo paragraph/sentence.
    """
    text = text.strip()
    contextualized_text = build_contextualized_text(
        article=article,
        section=section,
        text=text,
    )

    metadata = {
        "section_id": section.section_id,
        "section_level": section.section_level,
        "section_order": section.order,
        "section_ids": [section.section_id],
        "section_titles": [section.section_title],
        "section_orders": [section.order],
        "section_count": 1,
        "chunking_strategy": "single_long_section_split" if part_index is not None else "single_section",
        "source": article.source,
        "category": article.category,
    }

    if part_index is not None:
        metadata["section_part_index"] = part_index

    return ChunkDocument(
        chunk_id=make_chunk_id(article.article_id, chunk_index, text),
        article_id=article.article_id,
        source_url=article.url,
        title=article.title,
        section=section.parent_section_title or section.section_title,
        subsection=section.section_title if section.parent_section_title else None,
        chunk_index=chunk_index,
        text=text,
        contextualized_text=contextualized_text,
        token_count=estimate_token_count(contextualized_text),
        metadata=metadata,
    )


def _can_pack_sections(
    article: ArticleDocument,
    sections: list[SectionDocument],
    max_tokens: int,
) -> bool:
    if not sections:
        return False

    text = "\n\n".join(_section_text_with_heading(section) for section in sections).strip()
    contextualized_text = build_packed_contextualized_text(
        article=article,
        sections=sections,
        text=text,
    )
    return estimate_token_count(contextualized_text) <= max_tokens


def chunk_sections(
    article: ArticleDocument,
    sections: list[SectionDocument],
    max_tokens: int = 900,
) -> list[ChunkDocument]:
    """
    Tạo chunk theo kiểu section-preserving packed chunking.

    Khác logic cũ:
    - Không ép 1 section = 1 chunk.
    - Cố gắng pack nhiều section liền kề vào cùng chunk nếu tổng token còn <= max_tokens.
    - Không cắt vỡ section nhỏ.
    - Chỉ split theo paragraph/sentence nếu một section đơn lẻ đã vượt max_tokens.

    Mục tiêu:
    - Giảm số chunk/gọi LLM.
    - Giữ context giữa các section gần nhau.
    - Vẫn giữ boundary section rõ ràng bằng heading Markdown trong chunk text.
    """
    chunks: list[ChunkDocument] = []
    chunk_index = 0
    pending_sections: list[SectionDocument] = []

    def flush_pending() -> None:
        nonlocal chunk_index, pending_sections

        if not pending_sections:
            return

        chunks.append(
            make_packed_chunk(
                article=article,
                sections=pending_sections,
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1
        pending_sections = []

    for section in sections:
        if not section.text.strip():
            continue

        single_section_context = build_contextualized_text(
            article=article,
            section=section,
            text=section.text,
        )
        section_token_count = estimate_token_count(single_section_context)

        # Section đơn lẻ quá dài: flush pack hiện tại rồi split section đó.
        if section_token_count > max_tokens:
            flush_pending()

            parts = split_long_text_by_paragraphs(
                text=section.text,
                max_tokens=max_tokens,
                overlap_paragraphs=1,
            )

            for part_index, part in enumerate(parts):
                text = part.strip()
                if not text:
                    continue

                chunks.append(
                    make_single_section_chunk(
                        article=article,
                        section=section,
                        text=text,
                        chunk_index=chunk_index,
                        part_index=part_index,
                    )
                )
                chunk_index += 1

            continue

        candidate_sections = pending_sections + [section]

        if _can_pack_sections(
            article=article,
            sections=candidate_sections,
            max_tokens=max_tokens,
        ):
            pending_sections = candidate_sections
            continue

        flush_pending()
        pending_sections = [section]

    flush_pending()
    return chunks


def chunk_article(article: ArticleDocument, max_tokens: int = 900) -> list[ChunkDocument]:
    """
    Parse section rồi chunk article.
    """
    from app.ingestion.processing.section_parser import parse_sections

    sections = parse_sections(article)
    return chunk_sections(article, sections, max_tokens=max_tokens)
