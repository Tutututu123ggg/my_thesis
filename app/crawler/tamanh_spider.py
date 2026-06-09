import scrapy

from app.crawler.article_item import ArticleItem
from app.crawler.markdown_converter import save_article_markdown
from app.crawler.strategies.tamanh_strategy import TamAnhStrategy


class TamAnhSpider(scrapy.Spider):
    name = "tamanh"

    allowed_domains = ["tamanhhospital.vn"]

    custom_settings = {
        "DOWNLOAD_DELAY": 1.0,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "CONCURRENT_REQUESTS": 4,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "RETRY_TIMES": 2,
        "LOG_LEVEL": "INFO",
        "USER_AGENT": (
            "medical-vietnamese-hybrid-rag/0.1 "
            "(academic thesis crawler; contact: ductu3003@gmail.com)"
        ),
    }

    def __init__(
        self,
        output_dir: str = "data/raw/articles",
        max_articles: int = 20,
        start_url: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.strategy = TamAnhStrategy()
        self.output_dir = output_dir
        self.max_articles = int(max_articles)

        self.start_urls = [start_url] if start_url else self.strategy.seed_urls

        self.discovered_urls: set[str] = set()
        self.saved_count = 0

    def parse(self, response):
        """
        Nếu start_url là article thì parse trực tiếp.
        Nếu start_url là seed/listing thì discover article URLs.
        """
        current_url = self.strategy.normalize_url(response.url)

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
                meta={"category": "benh"},
            )

    def parse_article(self, response):
        if self.saved_count >= self.max_articles:
            return

        title = self.strategy.extract_title(response)

        if not title:
            self.logger.warning("Skip article without title: %s", response.url)
            return

        markdown = self.strategy.extract_markdown(response, title=title)

        if len(markdown.split()) < 300:
            self.logger.warning(
                "Skip short article: words=%s url=%s",
                len(markdown.split()),
                response.url,
            )
            return

        item = ArticleItem(
            source=self.strategy.source,
            url=self.strategy.normalize_url(response.url),
            title=title,
            description=self.strategy.extract_description(response),
            author=self.strategy.extract_author(response),
            published_at=self.strategy.extract_published_at(response),
            updated_at=self.strategy.extract_updated_at(response),
            category=response.meta.get("category", "benh"),
            related_links=self.strategy.extract_related_links(response),
            markdown=markdown,
        )

        path = save_article_markdown(item, self.output_dir)
        self.saved_count += 1

        self.logger.info(
            "Saved article %s/%s: %s",
            self.saved_count,
            self.max_articles,
            path,
        )