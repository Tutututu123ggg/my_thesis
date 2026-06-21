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

    # Multi-source metadata. These fields are optional so old Tâm Anh code
    # stays compatible, but new strategies can persist source/domain signals.
    source_domain: str | None = None
    source_name: str | None = None
    medical_reviewed_by: str | None = None

    related_links: list[str] = field(default_factory=list)
