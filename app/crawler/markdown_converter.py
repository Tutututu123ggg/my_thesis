import re
import unicodedata
from datetime import datetime
from hashlib import md5
from pathlib import Path

from markdownify import markdownify as md

from app.crawler.article_item import ArticleItem


def slugify(text: str, max_len: int = 90) -> str:
    """
    Tạo slug tiếng Việt.
    Fix riêng chữ đ/Đ vì unicodedata không tự chuyển đ -> d.
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:max_len] or "untitled"


def normalize_space(text: str | None) -> str | None:
    if not text:
        return None

    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def html_to_markdown(html: str) -> str:
    markdown = md(
        html,
        heading_style="ATX",
        bullets="-",
        strip=["img", "script", "style", "iframe", "form"],
    )

    # Dọn dòng trắng thừa
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)

    # Dọn citation bị tách dòng kiểu:
    # (
    # 1
    # )
    markdown = re.sub(r"\(\s*\n\s*(\d+)\s*\n\s*\)", r"(\1)", markdown)

    return markdown.strip()


def escape_yaml(text: str | None) -> str:
    if not text:
        return ""

    return text.replace('"', "'").strip()


def build_frontmatter(item: ArticleItem) -> str:
    lines = [
        "---",
        f'source: "{escape_yaml(item.source)}"',
        f'url: "{escape_yaml(item.url)}"',
        f'title: "{escape_yaml(item.title)}"',
    ]

    if item.description:
        lines.append(f'description: "{escape_yaml(item.description)}"')

    if item.author:
        lines.append(f'author: "{escape_yaml(item.author)}"')

    if item.published_at:
        lines.append(f'published_at: "{escape_yaml(item.published_at)}"')

    if item.updated_at:
        lines.append(f'updated_at: "{escape_yaml(item.updated_at)}"')

    if item.category:
        lines.append(f'category: "{escape_yaml(item.category)}"')

    lines.append(f'crawled_at: "{datetime.now().isoformat(timespec="seconds")}"')

    lines.append("related_links:")
    for link in item.related_links:
        lines.append(f'  - "{escape_yaml(link)}"')

    lines.append("---")
    return "\n".join(lines)


def save_article_markdown(item: ArticleItem, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url_hash = md5(item.url.encode("utf-8")).hexdigest()[:8]
    filename = f"{item.source}__{slugify(item.title)}__{url_hash}.md"
    path = output_dir / filename

    content = build_frontmatter(item)
    content += "\n\n"
    content += f"# {item.title}\n\n"
    content += item.markdown.strip()
    content += "\n"

    path.write_text(content, encoding="utf-8")
    return path