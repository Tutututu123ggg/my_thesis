import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from app.ingestion.processing.ingestion_models import ArticleDocument
from app.ingestion.processing.markdown_cleaner import clean_markdown_body


def normalize_text(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_stable_id(prefix: str, value: str) -> str:
    h = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """
    Parse Markdown có YAML frontmatter:

    ---
    title: ...
    ---

    body...
    """
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    if not raw.startswith("---\n"):
        return {}, raw

    end = raw.find("\n---", 4)
    if end == -1:
        return {}, raw

    frontmatter_text = raw[4:end].strip()
    body = raw[end + len("\n---"):].strip()

    metadata = yaml.safe_load(frontmatter_text) or {}
    return metadata, body


def load_markdown_article(path: str | Path) -> ArticleDocument:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    metadata, body = parse_frontmatter(raw)

    url = str(metadata.get("url") or "")
    title = str(metadata.get("title") or path.stem)

    article_id = make_stable_id("article", url or title)

    body = clean_markdown_body(body, title=title)

    related_links = metadata.get("related_links") or []
    if not isinstance(related_links, list):
        related_links = []

    return ArticleDocument(
        article_id=article_id,
        source=str(metadata.get("source") or "unknown"),
        url=url,
        title=title,
        description=metadata.get("description"),
        author=metadata.get("author"),
        published_at=str(metadata.get("published_at") or "") or None,
        updated_at=str(metadata.get("updated_at") or "") or None,
        category=metadata.get("category"),
        crawled_at=str(metadata.get("crawled_at") or "") or None,
        related_links=related_links,
        body=body,
        file_path=path,
        metadata=metadata,
    )


def load_markdown_articles(input_path: str | Path) -> list[ArticleDocument]:
    input_path = Path(input_path)

    if input_path.is_file():
        return [load_markdown_article(input_path)]

    files = sorted(input_path.glob("*.md"))
    return [load_markdown_article(file) for file in files]