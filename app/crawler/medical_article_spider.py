from __future__ import annotations

import json

import scrapy

from app.crawler.article_item import ArticleItem
from app.crawler.markdown_converter import save_article_markdown
from app.crawler.strategies.medlatec_strategy import MedlatecStrategy
from app.crawler.strategies.tamanh_strategy import TamAnhStrategy
from app.crawler.strategies.vinmec_strategy import VinmecStrategy


STRATEGIES = {
    "tamanh": TamAnhStrategy,
    "vinmec": VinmecStrategy,
    "medlatec": MedlatecStrategy,
}


class MedicalArticleSpider(scrapy.Spider):
    """Shared spider for site-specific medical article strategies.

    The spider handles request/limit/save. Each strategy handles website layout.
    For approved seed URLs, pass ``start_url`` or ``start_urls_json`` and the
    spider will treat every provided URL as an exact article candidate instead
    of rediscovering links from the page.
    """

    name = "medical_article"

    custom_settings = {
        "DOWNLOAD_DELAY": 1.0,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "CONCURRENT_REQUESTS": 4,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "RETRY_TIMES": 2,
        "LOG_LEVEL": "INFO",
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.7,en;q=0.6",
        },
    }

    def __init__(
        self,
        strategy: str = "vinmec",
        output_dir: str = "data/raw/articles_multisource",
        max_articles: int = 20,
        start_url: str | None = None,
        start_urls_json: str | None = None,
        category: str = "benh",
        min_words: int = 250,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        strategy_key = strategy.strip().lower()
        if strategy_key not in STRATEGIES:
            supported = ", ".join(sorted(STRATEGIES))
            raise ValueError(f"Unsupported strategy='{strategy}'. Supported: {supported}")

        self.strategy = STRATEGIES[strategy_key]()
        self.allowed_domains = getattr(self.strategy, "allowed_domains", []) or [
            getattr(self.strategy, "allowed_domain", "")
        ]
        self.allowed_domains = [domain for domain in self.allowed_domains if domain]

        self.output_dir = output_dir
        self.max_articles = int(max_articles)
        self.category = category
        self.min_words = int(min_words)

        urls: list[str] = []
        if start_urls_json:
            loaded = json.loads(start_urls_json)
            if not isinstance(loaded, list):
                raise ValueError("start_urls_json must decode to a list[str]")
            urls = [str(item).strip() for item in loaded if str(item).strip()]
        elif start_url:
            urls = [start_url]

        self.direct_url_mode = bool(urls)
        self.start_urls = urls if urls else self.strategy.seed_urls

        self.discovered_urls: set[str] = set()
        self.saved_count = 0

    def parse(self, response):
        current_url = self.strategy.normalize_url(response.url)

        # Approved-seed mode: the input URL itself is the article candidate.
        # This matters for Tâm Anh root slugs and /tin-tuc/ pages, which are
        # valid approved links but do not all match the older /benh/ prefix.
        if self.direct_url_mode:
            self.parse_article(response)
            return

        if self.strategy.is_article_url(current_url):
            self.parse_article(response)
            return

        urls = self.strategy.extract_article_urls(response)
        self.logger.info("Discovered %s article URLs from %s", len(urls), response.url)

        for url in urls:
            if len(self.discovered_urls) >= self.max_articles:
                break
            if url in self.discovered_urls:
                continue

            self.discovered_urls.add(url)
            yield scrapy.Request(
                url=url,
                callback=self.parse_article,
                meta={"category": self.category},
            )

    def parse_article(self, response):
        if self.saved_count >= self.max_articles:
            return

        title = self.strategy.extract_title(response)
        if not title:
            self.logger.warning("Skip article without title: %s", response.url)
            return

        markdown = self.strategy.extract_markdown(response, title=title)
        word_count = len(markdown.split())

        if word_count < self.min_words:
            self.logger.warning(
                "Skip short article: words=%s min_words=%s url=%s",
                word_count,
                self.min_words,
                response.url,
            )
            return

        item = ArticleItem(
            source=self.strategy.source,
            source_domain=getattr(self.strategy, "source_domain", None),
            source_name=getattr(self.strategy, "source_name", None),
            url=self.strategy.normalize_url(response.url),
            title=title,
            description=self.strategy.extract_description(response),
            author=self.strategy.extract_author(response),
            medical_reviewed_by=(
                self.strategy.extract_medical_reviewed_by(response)
                if hasattr(self.strategy, "extract_medical_reviewed_by")
                else None
            ),
            published_at=self.strategy.extract_published_at(response),
            updated_at=self.strategy.extract_updated_at(response),
            category=response.meta.get("category", self.category),
            related_links=self.strategy.extract_related_links(response),
            markdown=markdown,
        )

        path = save_article_markdown(item, self.output_dir)
        self.saved_count += 1

        self.logger.info(
            "Saved article %s/%s: words=%s path=%s",
            self.saved_count,
            self.max_articles,
            word_count,
            path,
        )
