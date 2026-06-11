import hashlib
import re

from app.ingestion.processing.ingestion_models import ArticleDocument, SectionDocument


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def strip_heading_format(text: str) -> str:
    text = text.strip()
    text = text.replace("**", "")
    text = text.replace("__", "")
    return text.strip()


def make_section_id(article_id: str, order: int, title: str) -> str:
    raw = f"{article_id}:{order}:{title}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"section_{h}"


def parse_sections(article: ArticleDocument) -> list[SectionDocument]:
    """
    Split Markdown theo heading.

    Quy tắc:
    - Bỏ H1 title vì Article đã có title.
    - H2 là section.
    - H3+ là subsection.
    - Nếu đoạn mở đầu trước H2 có nội dung, tạo section "Tóm tắt".
    """
    lines = article.body.splitlines()

    sections: list[SectionDocument] = []

    current_title = "Tóm tắt"
    current_level = 2
    current_parent: str | None = None
    current_lines: list[str] = []
    order = 0

    last_h2: str | None = None

    def flush() -> None:
        nonlocal order, current_lines, current_title, current_level, current_parent

        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            return

        section = SectionDocument(
            article_id=article.article_id,
            title=article.title,
            source_url=article.url,
            section_id=make_section_id(article.article_id, order, current_title),
            section_title=current_title,
            section_level=current_level,
            parent_section_title=current_parent,
            text=text,
            order=order,
        )
        sections.append(section)

        order += 1
        current_lines = []

    for line in lines:
        match = HEADING_RE.match(line.strip())

        if match:
            hashes, heading_text = match.groups()
            level = len(hashes)
            heading_text = strip_heading_format(heading_text)

            # Bỏ H1 trùng title
            if level == 1 and heading_text == article.title:
                continue

            flush()

            current_title = heading_text
            current_level = level

            if level == 2:
                last_h2 = heading_text
                current_parent = None
            elif level >= 3:
                current_parent = last_h2

            continue

        current_lines.append(line)

    flush()

    return sections