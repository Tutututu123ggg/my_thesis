from __future__ import annotations

import re
from urllib.parse import urldefrag, urljoin, urlparse

import scrapy
from bs4 import BeautifulSoup, Tag

from app.crawler.markdown_converter import html_to_markdown, normalize_space


class BaseMedicalStrategy:
    """Thin helper base for site-specific medical crawling strategies.

    Important design rule:
    - This class does NOT decide the main article selector for every site.
    - Each concrete strategy must know its own website layout and implement or
      configure extract_raw_main_html / clean_article_html / clean_markdown.
    """

    source: str = ""
    source_name: str = ""
    source_domain: str = ""
    allowed_domains: list[str] = []
    seed_urls: list[str] = []

    article_prefixes: list[str] = []
    blacklist_paths: list[str] = [
        "/wp-content/",
        "/wp-json/",
        "/feed/",
        "/tag/",
        "/tags/",
        "/author/",
    ]
    bad_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".pdf",
        ".zip",
        ".doc",
        ".docx",
    )

    title_selectors: tuple[str, ...] = (
        "h1::text",
        "meta[property='og:title']::attr(content)",
        "title::text",
    )
    description_selectors: tuple[str, ...] = (
        "meta[name='description']::attr(content)",
        "meta[property='og:description']::attr(content)",
    )
    published_at_selectors: tuple[str, ...] = (
        "meta[property='article:published_time']::attr(content)",
        "time::attr(datetime)",
        "time::text",
    )
    updated_at_selectors: tuple[str, ...] = (
        "meta[property='article:modified_time']::attr(content)",
        "meta[name='last-modified']::attr(content)",
        "meta[name='lastmod']::attr(content)",
    )

    def normalize_url(self, url: str) -> str:
        clean, _ = urldefrag(url)
        parsed = urlparse(clean)
        return parsed._replace(query="", fragment="").geturl().rstrip("/")

    def is_same_domain(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        allowed = [item.lower().replace("www.", "") for item in self.allowed_domains]
        if self.source_domain:
            allowed.append(self.source_domain.lower().replace("www.", ""))
        return any(domain == item or domain.endswith("." + item) for item in allowed)

    def is_article_url(self, url: str) -> bool:
        url = self.normalize_url(url)
        parsed = urlparse(url)
        path = parsed.path.lower()

        if not self.is_same_domain(url):
            return False

        if path.endswith(self.bad_extensions):
            return False

        if any(bad in path for bad in self.blacklist_paths):
            return False

        return any(path.startswith(prefix) for prefix in self.article_prefixes)

    def extract_article_urls(self, response: scrapy.http.Response) -> list[str]:
        urls: list[str] = []
        for href in response.css("a::attr(href)").getall():
            url = urljoin(response.url, href)
            url = self.normalize_url(url)
            if self.is_article_url(url):
                urls.append(url)
        return self.unique_keep_order(urls)

    def first_text(self, response: scrapy.http.Response, selectors: tuple[str, ...] | list[str]) -> str | None:
        for selector in selectors:
            value = normalize_space(" ".join(response.css(selector).getall()))
            if value:
                return value
        return None

    def extract_title(self, response: scrapy.http.Response) -> str | None:
        title = self.first_text(response, self.title_selectors)
        if title and "|" in title:
            title = title.split("|", 1)[0].strip()
        return normalize_space(title)

    def extract_description(self, response: scrapy.http.Response) -> str | None:
        return self.first_text(response, self.description_selectors)

    def extract_published_at(self, response: scrapy.http.Response) -> str | None:
        value = self.first_text(response, self.published_at_selectors)
        if value:
            return value

        text = normalize_space(" ".join(response.css("body ::text").getall())) or ""
        match = re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", text)
        return match.group(0) if match else None

    def extract_updated_at(self, response: scrapy.http.Response) -> str | None:
        return self.first_text(response, self.updated_at_selectors)

    def extract_author(self, response: scrapy.http.Response) -> str | None:
        return self.first_text(
            response,
            (
                "meta[name='author']::attr(content)",
                ".author::text",
                ".post-author::text",
                "[class*='author']::text",
            ),
        )

    def extract_medical_reviewed_by(self, response: scrapy.http.Response) -> str | None:
        return None

    def extract_related_links(self, response: scrapy.http.Response, limit: int = 30) -> list[str]:
        links: list[str] = []
        for href in response.css("a::attr(href)").getall():
            url = self.normalize_url(urljoin(response.url, href))
            if self.is_article_url(url) and url != self.normalize_url(response.url):
                links.append(url)
        return self.unique_keep_order(links)[:limit]

    def extract_raw_main_html(self, response: scrapy.http.Response) -> str:
        raise NotImplementedError

    def clean_article_html(self, html: str) -> str:
        return html

    def clean_markdown(self, markdown: str, title: str | None = None) -> str:
        return markdown.strip()

    def extract_markdown(self, response: scrapy.http.Response, title: str | None = None) -> str:
        raw_html = self.extract_raw_main_html(response)
        if not raw_html:
            return ""
        clean_html = self.clean_article_html(raw_html)
        markdown = html_to_markdown(clean_html)
        markdown = self.clean_markdown(markdown, title=title)
        return markdown.strip()

    def remove_selectors_from_soup(self, soup: BeautifulSoup, selectors: list[str] | tuple[str, ...]) -> None:
        for selector in selectors:
            for tag in soup.select(selector):
                tag.decompose()

    def remove_empty_tags(self, soup: BeautifulSoup) -> None:
        for tag in list(soup.find_all(True)):
            if tag.name in {"br", "hr", "img"}:
                continue
            if tag.get_text(strip=True):
                continue
            if isinstance(tag, Tag):
                tag.decompose()

    def normalize_for_compare(self, text: str | None) -> str:
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r"^#{1,6}\s+", "", text)
        text = text.replace("**", "")
        text = text.replace("__", "")
        text = text.replace("*", "")
        text = text.replace("_", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()


    def strip_markdown_links(self, text: str) -> str:
        """Keep visible anchor text and remove Markdown URL wrappers.

        Medical entity extraction does not need inline source URLs in the body.
        Keeping only the visible text also prevents huge URL fragments from
        becoming noisy tokens. Frontmatter still preserves the source URL.
        """
        if not text:
            return text
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        return text

    def is_link_only_line(self, line: str) -> bool:
        return bool(re.fullmatch(r"\[[^\]]+\]\([^)]*\)", line.strip()))

    def unique_keep_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result
