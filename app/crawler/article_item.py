from dataclasses import dataclass, field


@dataclass
class ArticleItem:
    source: str
    url: str
    title: str
    markdown: str

    description: str | None = None
    author: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    category: str | None = None

    related_links: list[str] = field(default_factory=list)