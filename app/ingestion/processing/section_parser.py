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
    """Split Markdown into section documents while preserving heading ancestry.

    Design:
    - H1 title is skipped because ``ArticleDocument.title`` already stores it.
    - Every non-empty text block under a heading becomes one ``SectionDocument``.
    - ``heading_path`` stores the active heading chain at the start of that
      section, e.g. ["2. Điều trị", "2.1. Thuốc bôi"].

    The heading path is not meant to replace raw text. It is extraction context
    so bullet/list chunks do not lose their parent subject.
    """
    lines = article.body.splitlines()

    sections: list[SectionDocument] = []
    heading_stack: list[tuple[int, str]] = []

    current_title = "Tóm tắt"
    current_level = 2
    current_parent: str | None = None
    current_heading_path: list[str] = []
    current_lines: list[str] = []
    order = 0

    def flush() -> None:
        nonlocal order, current_lines, current_title, current_level, current_parent, current_heading_path

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
            heading_path=list(current_heading_path),
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

            # Skip article H1. The article title is already passed separately.
            if level == 1 and heading_text == article.title:
                continue

            flush()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            parent = heading_stack[-1][1] if heading_stack else None
            heading_stack.append((level, heading_text))

            current_title = heading_text
            current_level = level
            current_parent = parent
            current_heading_path = [title for _, title in heading_stack]
            continue

        current_lines.append(line)

    flush()

    return sections
