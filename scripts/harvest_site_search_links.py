"""Harvest candidate medical article links from internal search pages of trusted sites.

v3.7: map Vinmec API fields post_slug/post_title/post_sapo.
v3.3: fix Scrapy 2.13+ start() hook; v3.2 planned-request debug retained.
This script only harvests search-result links. It does NOT crawl full article content.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import re
import time
import unicodedata
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, quote, quote_plus, urlencode, urlsplit, urlunsplit

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import Response

LOGGER = logging.getLogger("search_harvester")

DEFAULT_TOPICS = [
    {"cluster": "dermatology", "anchor_entity": "viêm da cơ địa", "aliases": ["chàm cơ địa", "eczema", "atopic dermatitis"]},
    {"cluster": "dermatology", "anchor_entity": "viêm da tiếp xúc", "aliases": ["contact dermatitis"]},
    {"cluster": "dermatology", "anchor_entity": "viêm da tiết bã", "aliases": ["seborrheic dermatitis", "viêm da dầu"]},
    {"cluster": "dermatology", "anchor_entity": "vảy nến", "aliases": ["psoriasis", "bệnh vảy nến"]},
    {"cluster": "dermatology", "anchor_entity": "nấm da", "aliases": ["hắc lào", "lang ben", "nấm ngoài da"]},
    {"cluster": "dermatology", "anchor_entity": "mụn trứng cá", "aliases": ["acne", "trứng cá"]},
    {"cluster": "respiratory", "anchor_entity": "viêm phổi bệnh viện", "aliases": ["hospital acquired pneumonia", "HAP", "viêm phổi liên quan thở máy"]},
    {"cluster": "respiratory", "anchor_entity": "viêm phổi cộng đồng", "aliases": ["community acquired pneumonia", "CAP"]},
    {"cluster": "respiratory", "anchor_entity": "viêm phế quản", "aliases": ["viêm phế quản cấp", "viêm phế quản mạn"]},
    {"cluster": "respiratory_ent", "anchor_entity": "viêm mũi dị ứng", "aliases": ["allergic rhinitis"]},
    {"cluster": "respiratory_ent", "anchor_entity": "viêm xoang", "aliases": ["sinusitis", "viêm mũi xoang"]},
]

BAD_PATH_KEYWORDS = (
    "chinh-sach", "quyen-rieng-tu", "bao-ve-du-lieu", "tuyen-dung", "thong-bao",
    "uu-dai", "khuyen-mai", "bang-gia", "dat-lich", "lien-he", "gioi-thieu",
    "dich-vu", "dich-vu-y-te", "chuyen-khoa", "chuyen-gia", "bac-si", "hoi-dap",
    "app", "v-app", "myvinmec", "customer", "newsletter", "login", "register", "resource",
)
TRACKING_QUERY_KEYS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}


def compact_text(text: str | None, limit: int = 300) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()[:limit]


def strip_html(text: str | None, limit: int = 700) -> str:
    if not text:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return compact_text(text, limit)


def remove_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def norm_for_match(text: str | None) -> str:
    text = remove_accents(text or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalize_url(url: str, *, drop_query_keys: set[str] | None = None, keep_query: bool = True) -> str:
    drop_query_keys = drop_query_keys or set()
    parsed = urlsplit(url.strip())
    query_items = []
    if keep_query:
        for key, value in parse_qsl(parsed.query, keep_blank_values=False):
            key_l = key.lower()
            if key_l in TRACKING_QUERY_KEYS or key_l in drop_query_keys:
                continue
            query_items.append((key, value))
    path = parsed.path
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), path, urlencode(query_items), ""))


def load_topics(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_TOPICS
    p = Path(path)
    if not p.exists():
        LOGGER.warning("topics file not found: %s; using DEFAULT_TOPICS", p)
        return DEFAULT_TOPICS
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        raw_topics = data
    elif isinstance(data, dict) and isinstance(data.get("topics"), list):
        raw_topics = data["topics"]
    elif isinstance(data, dict) and isinstance(data.get("clusters"), list):
        raw_topics = []
        for cluster in data["clusters"]:
            cluster_name = cluster.get("cluster") or cluster.get("name") or ""
            for anchor in cluster.get("anchors", []):
                if isinstance(anchor, str):
                    raw_topics.append({"cluster": cluster_name, "anchor_entity": anchor, "aliases": []})
                else:
                    item = dict(anchor)
                    item.setdefault("cluster", cluster_name)
                    raw_topics.append(item)
    else:
        raise ValueError(f"Unsupported topics-file shape: {p}")

    topics: list[dict[str, Any]] = []
    for item in raw_topics:
        if isinstance(item, str):
            topics.append({"cluster": "", "anchor_entity": item, "aliases": []})
            continue
        anchor = item.get("anchor_entity") or item.get("anchor") or item.get("name") or item.get("query")
        if not anchor:
            continue
        aliases = item.get("aliases") or item.get("keywords") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        topics.append({"cluster": item.get("cluster", ""), "anchor_entity": anchor, "aliases": aliases})
    return topics


@dataclass(frozen=True)
class Candidate:
    source: str
    cluster: str
    anchor_entity: str
    query: str
    url: str
    title: str
    snippet: str
    search_url: str
    page: int
    selector: str
    match_method: str


class BaseSearchStrategy:
    source = "base"
    max_pages_default = 1

    def build_search_url(self, query: str, page: int = 1) -> str:
        raise NotImplementedError

    def normalize_candidate_url(self, url: str) -> str:
        return normalize_url(url)

    def is_rejected_url(self, url: str) -> bool:
        u = url.lower()
        return any(bad in u for bad in BAD_PATH_KEYWORDS)

    def is_accepted_url(self, url: str) -> bool:
        return not self.is_rejected_url(url)

    def matches_topic(self, *, url: str, title: str, snippet: str, topic: dict[str, Any], query: str) -> bool:
        """Loose match used by the broad fallback so nav/menu links do not flood results."""
        hay = norm_for_match(" ".join([title, snippet, url]))
        needles = [query, str(topic.get("anchor_entity", ""))]
        needles += [a for a in topic.get("aliases", []) if isinstance(a, str)]
        for n in needles:
            nn = norm_for_match(n)
            if not nn:
                continue
            # exact phrase match catches Vietnamese title/url slugs well
            if nn in hay:
                return True
            # partial token match for longer anchors, e.g. "viem da co dia" in URL/title with small changes
            toks = [t for t in nn.split() if len(t) >= 3]
            if len(toks) >= 3 and sum(1 for t in toks if t in hay) >= max(2, len(toks) - 1):
                return True
        return False

    def _candidate_from_link(
        self,
        response: Response,
        *,
        href: str,
        title: str,
        snippet: str,
        topic: dict[str, Any],
        query: str,
        page: int,
        selector: str,
        match_method: str,
        require_topic_match: bool = False,
    ) -> Candidate | None:
        if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:") or href.startswith("tel:"):
            return None
        url = self.normalize_candidate_url(response.urljoin(href))
        if not self.is_accepted_url(url):
            return None
        title = compact_text(title, 180)
        snippet = compact_text(snippet, 500)
        if not title and not snippet:
            return None
        if require_topic_match and not self.matches_topic(url=url, title=title, snippet=snippet, topic=topic, query=query):
            return None
        return Candidate(
            source=self.source,
            cluster=str(topic.get("cluster", "")),
            anchor_entity=str(topic.get("anchor_entity", "")),
            query=query,
            url=url,
            title=title,
            snippet=snippet,
            search_url=response.url,
            page=page,
            selector=selector,
            match_method=match_method,
        )

    def fallback_extract_candidates(self, response: Response, *, topic: dict[str, Any], query: str, page: int) -> Iterable[Candidate]:
        selector = "a[href]::fallback_topic_match"
        for a in response.css("a[href]"):
            href = a.attrib.get("href", "")
            title = a.xpath("normalize-space(.)").get() or ""
            snippet = a.xpath("normalize-space(..)").get() or a.xpath("normalize-space(../..)").get() or ""
            cand = self._candidate_from_link(
                response,
                href=href,
                title=title,
                snippet=snippet,
                topic=topic,
                query=query,
                page=page,
                selector=selector,
                match_method="fallback",
                require_topic_match=True,
            )
            if cand:
                yield cand

    def extract_specific_candidates(self, response: Response, *, topic: dict[str, Any], query: str, page: int) -> Iterable[Candidate]:
        return []

    def is_json_response(self, response: Response) -> bool:
        content_type = response.headers.get(b"Content-Type", b"").decode("latin-1", errors="ignore").lower()
        text_head = response.text[:80].lstrip()
        return "json" in content_type or text_head.startswith("{") or text_head.startswith("[")

    def extract_candidates(self, response: Response, *, topic: dict[str, Any], query: str, page: int) -> Iterable[Candidate]:
        specific = list(self.extract_specific_candidates(response, topic=topic, query=query, page=page))
        if specific:
            yield from specific
            return

        # JSON/API responses cannot be traversed with CSS selectors.
        # If a JSON-specific extractor returns no candidates, skip fallback safely.
        if self.is_json_response(response):
            return

        yield from self.fallback_extract_candidates(response, topic=topic, query=query, page=page)


class TamanhSearchStrategy(BaseSearchStrategy):
    source = "tamanh"
    max_pages_default = 2

    def build_search_url(self, query: str, page: int = 1) -> str:
        if page <= 1:
            return "https://tamanhhospital.vn/?s=" + quote_plus(query)
        return f"https://tamanhhospital.vn/page/{page}/?s=" + quote_plus(query)

    def is_accepted_url(self, url: str) -> bool:
        if self.is_rejected_url(url) or "tamanhhospital.vn" not in url:
            return False
        return True  # Tâm Anh có nhiều bài y khoa ở root slug.

    def extract_specific_candidates(self, response: Response, *, topic: dict[str, Any], query: str, page: int) -> Iterable[Candidate]:
        # Primary selector from browser sample; fallback handles layout changes.
        selectors = [".search_news_item > a", "article a", ".search-results a"]
        for selector in selectors:
            for a in response.css(selector):
                cand = self._candidate_from_link(
                    response,
                    href=a.attrib.get("href", ""),
                    title=a.xpath("normalize-space(.)").get() or "",
                    snippet=a.xpath("normalize-space(..)").get() or "",
                    topic=topic,
                    query=query,
                    page=page,
                    selector=selector,
                    match_method="specific",
                    require_topic_match=True,
                )
                if cand:
                    yield cand


class VinmecSearchStrategy(BaseSearchStrategy):
    source = "vinmec"
    max_pages_default = 3

    def build_search_url(self, query: str, page: int = 1) -> str:
        # Browser search page loads results through this JSON endpoint:
        # /api/v3/search?term=...&type=post&page=1&limit=10&locale=vi
        params = {
            "term": query,
            "type": "post",
            "page": str(page),
            "limit": "10",
            "locale": "vi",
        }
        return "https://www.vinmec.com/api/v3/search?" + urlencode(params)

    def normalize_candidate_url(self, url: str) -> str:
        url = url.strip()
        if not url:
            return url
        if url.startswith("/"):
            url = "https://www.vinmec.com" + url
        elif url.startswith("vie/"):
            url = "https://www.vinmec.com/" + url
        elif not url.startswith("http"):
            # API type=post usually returns slugs for /vie/bai-viet/.
            url = "https://www.vinmec.com/vie/bai-viet/" + url.strip("/")
        return normalize_url(url)

    def is_accepted_url(self, url: str) -> bool:
        if self.is_rejected_url(url) or "vinmec.com" not in url:
            return False
        return "/vie/bai-viet/" in url or "/vie/benh/" in url

    def _find_first_string(self, obj: dict[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _find_url_in_item(self, obj: dict[str, Any]) -> str:
        direct_keys = (
            "url", "href", "link", "path", "slug", "post_slug", "postSlug", "alias", "canonical", "canonical_url",
            "full_path", "fullPath", "detail_url", "detailUrl", "web_url", "webUrl",
            "url_path", "urlPath", "slug_url", "slugUrl", "permalink", "permaLink",
            "share_url", "shareUrl", "redirect_url", "redirectUrl", "landing_url", "landingUrl",
        )
        for key in direct_keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Some APIs nest URL info under objects such as {"url": {"vi": "..."}}.
        for value in obj.values():
            if isinstance(value, dict):
                nested = self._find_url_in_item(value)
                if nested:
                    return nested
        return ""

    def _iter_json_dicts(self, obj: Any):
        if isinstance(obj, dict):
            yield obj
            for value in obj.values():
                yield from self._iter_json_dicts(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from self._iter_json_dicts(value)

    def _json_candidates(self, response: Response, *, topic: dict[str, Any], query: str, page: int) -> Iterable[Candidate]:
        try:
            data = json.loads(response.text)
        except Exception:
            return []

        out: list[Candidate] = []
        seen_item_urls: set[str] = set()
        title_keys = (
            "title", "title_vi", "titleVi", "name", "name_vi", "nameVi",
            "post_title", "postTitle", "heading", "display_title", "displayTitle",
        )
        snippet_keys = (
            "description", "description_vi", "descriptionVi", "desc", "excerpt", "summary",
            "short_description", "shortDescription", "short_content", "shortContent",
            "sapo", "post_sapo", "postSapo", "intro", "content", "body", "meta_description", "metaDescription",
        )
        for item in self._iter_json_dicts(data):
            url_raw = self._find_url_in_item(item)
            title = self._find_first_string(item, title_keys)
            snippet = self._find_first_string(item, snippet_keys)

            # Avoid interpreting wrapper objects as result items.
            if not url_raw or not (title or snippet):
                continue
            url = self.normalize_candidate_url(url_raw)
            if url in seen_item_urls:
                continue
            seen_item_urls.add(url)
            cand = self._candidate_from_link(
                response,
                href=url,
                title=strip_html(title, 180),
                snippet=strip_html(snippet, 500),
                topic=topic,
                query=query,
                page=page,
                selector="vinmec_api:/api/v3/search",
                match_method="api_json",
                require_topic_match=True,
            )
            if cand:
                out.append(cand)
        return out

    def extract_specific_candidates(self, response: Response, *, topic: dict[str, Any], query: str, page: int) -> Iterable[Candidate]:
        # Preferred path: JSON API. This is what the Vinmec browser UI uses.
        content_type = response.headers.get(b"Content-Type", b"").decode("latin-1", errors="ignore").lower()
        text_head = response.text[:50].lstrip()
        if "json" in content_type or text_head.startswith("{") or text_head.startswith("["):
            yield from self._json_candidates(response, topic=topic, query=query, page=page)
            return

        # HTML fallback kept for manually saved/browser-rendered pages.
        selectors = [
            "#hero-post a.f22.bold",
            "#search-result > li > a.f18.bold.mb2",
            "#search-result a[href]",
            "a[href*='/vie/bai-viet/']",
            "a[href*='/vie/benh/']",
        ]
        for selector in selectors:
            for a in response.css(selector):
                cand = self._candidate_from_link(
                    response,
                    href=a.attrib.get("href", ""),
                    title=a.xpath("normalize-space(.)").get() or "",
                    snippet=a.xpath("normalize-space(..)").get() or a.xpath("normalize-space(../..)").get() or "",
                    topic=topic,
                    query=query,
                    page=page,
                    selector=selector,
                    match_method="specific",
                    require_topic_match=True,
                )
                if cand:
                    yield cand


class MedlatecSearchStrategy(BaseSearchStrategy):
    source = "medlatec"
    max_pages_default = 2

    def build_search_url(self, query: str, page: int = 1) -> str:
        base = "https://medlatec.vn/tim-kiem-thong-tin"
        params = {"keyword": query, "type": "bai-viet"}
        if page > 1:
            params["pagenumber"] = str(page)
        return base + "?" + urlencode(params)

    def normalize_candidate_url(self, url: str) -> str:
        url = normalize_url(url)
        url = url.replace("https://medlatec.vn/amp/tin-tuc/", "https://medlatec.vn/tin-tuc/")
        url = url.replace("http://medlatec.vn/amp/tin-tuc/", "https://medlatec.vn/tin-tuc/")
        return url

    def is_accepted_url(self, url: str) -> bool:
        if self.is_rejected_url(url) or "medlatec.vn" not in url:
            return False
        return "/tin-tuc/" in url or "/tu-dien-benh-ly/" in url

    def extract_specific_candidates(self, response: Response, *, topic: dict[str, Any], query: str, page: int) -> Iterable[Candidate]:
        selectors = [
            ".search-wrapper .post-list > .post-item.post-item-list h3.post-item-title > a",
            ".search-wrapper h3.post-item-title > a",
            "a[href*='/tin-tuc/']",
            "a[href*='/tu-dien-benh-ly/']",
        ]
        for selector in selectors:
            for a in response.css(selector):
                cand = self._candidate_from_link(
                    response,
                    href=a.attrib.get("href", ""),
                    title=a.xpath("normalize-space(.)").get() or "",
                    snippet=a.xpath("normalize-space(ancestor::*[contains(@class, 'post-item')][1])").get() or a.xpath("normalize-space(..)").get() or "",
                    topic=topic,
                    query=query,
                    page=page,
                    selector=selector,
                    match_method="specific",
                    require_topic_match=True,
                )
                if cand:
                    yield cand


class MsdManualSearchStrategy(BaseSearchStrategy):
    source = "msd_manual_vi"
    max_pages_default = 1

    def build_search_url(self, query: str, page: int = 1) -> str:
        return "https://www.msdmanuals.com/vi/professional/searchresults?query=" + quote(query)

    def normalize_candidate_url(self, url: str) -> str:
        return normalize_url(url, drop_query_keys={"query"})

    def is_accepted_url(self, url: str) -> bool:
        if self.is_rejected_url(url) or "msdmanuals.com" not in url:
            return False
        if "/multimedia/image/" in url or "/multimedia/table/" in url:
            return False
        return "/vi/professional/" in url or "/vi/home/" in url

    def extract_specific_candidates(self, response: Response, *, topic: dict[str, Any], query: str, page: int) -> Iterable[Candidate]:
        selectors = [
            '#answersection div[class*="MedicalTopics_searchResultsMain"] > a',
            'a[href*="/vi/professional/"]',
            'a[href*="/vi/home/"]',
        ]
        for selector in selectors:
            for a in response.css(selector):
                cand = self._candidate_from_link(
                    response,
                    href=a.attrib.get("href", ""),
                    title=a.xpath("normalize-space(.)").get() or "",
                    snippet=a.xpath("normalize-space(..)").get() or "",
                    topic=topic,
                    query=query,
                    page=page,
                    selector=selector,
                    match_method="specific",
                    require_topic_match=True,
                )
                if cand:
                    yield cand


STRATEGIES: dict[str, type[BaseSearchStrategy]] = {
    "tamanh": TamanhSearchStrategy,
    "vinmec": VinmecSearchStrategy,
    "medlatec": MedlatecSearchStrategy,
    "msd_manual_vi": MsdManualSearchStrategy,
    "msd": MsdManualSearchStrategy,
}


class SearchHarvestSpider(scrapy.Spider):
    name = "search_harvester"
    custom_settings = {
        "LOG_LEVEL": "INFO",
        "CONCURRENT_REQUESTS": 4,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DOWNLOAD_DELAY": 0.3,
        "ROBOTSTXT_OBEY": False,
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        },
    }

    def __init__(
        self,
        *,
        topics: list[dict[str, Any]],
        sources: list[str],
        output: str,
        max_pages: int,
        max_queries_per_anchor: int,
        max_links_per_anchor_source: int,
        sleep: float,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.topics = topics
        self.sources = sources
        self.output = Path(output)
        self.max_pages = max_pages
        self.max_queries_per_anchor = max_queries_per_anchor
        self.max_links_per_anchor_source = max_links_per_anchor_source
        self.sleep = sleep
        self.strategies = {src: STRATEGIES[src]() for src in sources}
        self.seen_urls: set[str] = set()
        self.count_by_anchor_source: dict[tuple[str, str], int] = {}
        self.rows: list[dict[str, Any]] = []
        self.debug_rows: list[dict[str, Any]] = []
        self.planned_requests = 0

    def _make_start_requests(self):
        """Build search-page requests.

        Scrapy 2.13+ uses Spider.start() as the entry point. Older code that only
        overrides start_requests() may silently crawl 0 pages on newer Scrapy.
        Keep the request-building logic in one helper and expose both APIs.
        """
        for topic in self.topics:
            anchor = str(topic.get("anchor_entity", "")).strip()
            if not anchor:
                continue
            aliases = [a for a in topic.get("aliases", []) if isinstance(a, str) and a.strip()]
            queries = ([anchor] + aliases)[: self.max_queries_per_anchor]
            for source, strategy in self.strategies.items():
                max_pages = min(self.max_pages, strategy.max_pages_default)
                for query in queries:
                    for page in range(1, max_pages + 1):
                        url = strategy.build_search_url(query, page=page)
                        self.planned_requests += 1
                        self.debug_rows.append({
                            "event": "planned",
                            "source": source,
                            "anchor_entity": anchor,
                            "query": query,
                            "page": page,
                            "status": "",
                            "all_links": "",
                            "extracted_candidates": "",
                            "kept": "",
                            "page_title": "",
                            "search_url": url,
                        })
                        yield scrapy.Request(
                            url,
                            callback=self.parse_search_result,
                            cb_kwargs={"source": source, "topic": topic, "query": query, "page": page},
                            dont_filter=True,
                        )

    async def start(self):
        """Scrapy 2.13+ start hook."""
        for req in self._make_start_requests():
            yield req
            if self.sleep > 0:
                await asyncio.sleep(self.sleep)

    def start_requests(self):
        """Backward-compatible hook for older Scrapy versions."""
        yield from self._make_start_requests()

    def _is_json_response(self, response: Response) -> bool:
        content_type = response.headers.get(b"Content-Type", b"").decode("latin-1", errors="ignore").lower()
        text_head = response.text[:80].lstrip()
        return "json" in content_type or text_head.startswith("{") or text_head.startswith("[")

    def _safe_all_links_count(self, response: Response) -> int:
        if self._is_json_response(response):
            return 0
        try:
            return len(response.css("a[href]").getall())
        except Exception:
            return 0

    def _safe_page_title(self, response: Response, limit: int = 120) -> str:
        if self._is_json_response(response):
            return "JSON API response"
        try:
            return compact_text(response.css("title::text").get(), limit)
        except Exception:
            return ""

    def parse_search_result(self, response: Response, *, source: str, topic: dict[str, Any], query: str, page: int):
        strategy = self.strategies[source]
        anchor = str(topic.get("anchor_entity", ""))
        key = (anchor, source)
        all_links = self._safe_all_links_count(response)
        candidates = list(strategy.extract_candidates(response, topic=topic, query=query, page=page))
        kept = 0
        for candidate in candidates:
            if candidate.url in self.seen_urls:
                continue
            if self.count_by_anchor_source.get(key, 0) >= self.max_links_per_anchor_source:
                continue
            self.seen_urls.add(candidate.url)
            self.count_by_anchor_source[key] = self.count_by_anchor_source.get(key, 0) + 1
            kept += 1
            self.rows.append(candidate.__dict__)
        page_title = self._safe_page_title(response, 120)
        self.debug_rows.append({
            "event": "response",
            "source": source,
            "anchor_entity": anchor,
            "query": query,
            "page": page,
            "status": response.status,
            "all_links": all_links,
            "extracted_candidates": len(candidates),
            "kept": kept,
            "page_title": page_title,
            "search_url": response.url,
        })
        self.logger.info(
            "source=%s anchor=%s query=%s page=%s status=%s all_links=%s extracted=%s kept=%s title=%s",
            source, anchor, query, page, response.status, all_links, len(candidates), kept,
            self._safe_page_title(response, 60),
        )

    def closed(self, reason: str):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        csv_path = self.output.with_suffix(".csv")
        candidate_fields = list(Candidate.__dataclass_fields__.keys())
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=candidate_fields)
            writer.writeheader()
            if self.rows:
                writer.writerows(self.rows)
        coverage_path = self.output.with_name(self.output.stem + "_coverage.csv")
        sources = list(self.strategies.keys())
        coverage_rows: list[dict[str, Any]] = []
        anchors = sorted({str(t.get("anchor_entity", "")) for t in self.topics if t.get("anchor_entity")})
        for anchor in anchors:
            row: dict[str, Any] = {"anchor_entity": anchor}
            total = 0
            domains = 0
            for source in sources:
                count = self.count_by_anchor_source.get((anchor, source), 0)
                row[source] = count
                total += count
                domains += int(count > 0)
            row["total"] = total
            row["domain_count"] = domains
            coverage_rows.append(row)
        with coverage_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["anchor_entity"] + sources + ["total", "domain_count"])
            writer.writeheader()
            writer.writerows(coverage_rows)
        debug_path = self.output.with_name(self.output.stem + "_debug.csv")
        debug_fields = [
            "event", "source", "anchor_entity", "query", "page", "status",
            "all_links", "extracted_candidates", "kept", "page_title", "search_url",
        ]
        with debug_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=debug_fields)
            writer.writeheader()
            for row in self.debug_rows:
                writer.writerow({k: row.get(k, "") for k in debug_fields})
        self.logger.info("Saved %s candidates -> %s", len(self.rows), self.output)
        self.logger.info("Saved CSV -> %s", csv_path)
        self.logger.info("Saved coverage -> %s", coverage_path)
        self.logger.info("Saved debug -> %s", debug_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest candidate article links from internal site search pages.")
    parser.add_argument("--topics-file", default="data/crawl/topic_seed_dense_v1.json")
    parser.add_argument("--sources", nargs="+", default=["tamanh", "vinmec", "medlatec", "msd_manual_vi"])
    parser.add_argument("--output", default="data/crawl/search_candidate_links.jsonl")
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--max-queries-per-anchor", type=int, default=2)
    parser.add_argument("--max-links-per-anchor-source", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    unknown = [s for s in args.sources if s not in STRATEGIES]
    if unknown:
        raise SystemExit(f"Unknown sources: {unknown}. Valid: {sorted(STRATEGIES)}")
    topics = load_topics(args.topics_file)
    if not topics:
        print(f"[WARN] topics-file produced 0 topics: {args.topics_file}. Falling back to DEFAULT_TOPICS.")
        topics = DEFAULT_TOPICS
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    LOGGER.info("Loaded topics=%s sources=%s output=%s max_pages=%s max_queries_per_anchor=%s", len(topics), args.sources, args.output, args.max_pages, args.max_queries_per_anchor)
    print(f"[HARVEST CONFIG] topics={len(topics)} sources={args.sources} output={args.output}")
    process = CrawlerProcess()
    process.crawl(
        SearchHarvestSpider,
        topics=topics,
        sources=args.sources,
        output=args.output,
        max_pages=args.max_pages,
        max_queries_per_anchor=args.max_queries_per_anchor,
        max_links_per_anchor_source=args.max_links_per_anchor_source,
        sleep=args.sleep,
    )
    process.start()


if __name__ == "__main__":
    main()
