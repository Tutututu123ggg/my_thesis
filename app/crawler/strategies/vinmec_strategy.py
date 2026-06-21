from __future__ import annotations

import re
from urllib.parse import urljoin

import scrapy
from bs4 import BeautifulSoup, Tag

from app.crawler.markdown_converter import html_to_markdown, normalize_space
from app.crawler.strategies.base_strategy import BaseMedicalStrategy


class VinmecStrategy(BaseMedicalStrategy):
    """Vinmec-specific HTML -> Markdown strategy.

    This strategy deliberately keeps URL discovery simple because the project can
    feed curated medical URLs from a separate link-collection step. The important
    part here is extracting the article body and cutting Vinmec boilerplate such
    as breadcrumbs, appointment CTA, hotline, share block, and consultation form.
    """

    source = "vinmec"
    source_name = "Vinmec"
    source_domain = "vinmec.com"
    allowed_domains = ["vinmec.com"]

    seed_urls = [
        "https://www.vinmec.com/vie/bai-viet",
    ]

    article_prefixes = [
        "/vie/bai-viet/",
    ]

    blacklist_paths = BaseMedicalStrategy.blacklist_paths + [
        "/vie/chuyen-gia-y-te/",
        "/vie/dich-vu-y-te/",
        "/vie/he-thong-y-te/",
        "/vie/gioi-thieu/",
        "/vie/lien-he/",
        "/vie/dang-ky-kham/",
        "/myvinmec/",
    ]

    title_selectors = (
        "h1::text",
        "h1 *::text",
        "meta[property='og:title']::attr(content)",
        "title::text",
    )

    # Vinmec changes class names over time. Keep this list site-specific but
    # tolerant; do not move these selectors into BaseMedicalStrategy.
    main_content_selectors = (
        "article",
        "main article",
        "[class*='ArticleDetail']",
        "[class*='article-detail']",
        "[class*='ArticleContent']",
        "[class*='article-content']",
        "[class*='detail-content']",
        "[class*='content-detail']",
        "[class*='post-content']",
        "[class*='rich-text']",
        "main",
    )

    remove_selectors = (
        "script",
        "style",
        "noscript",
        "iframe",
        "form",
        "header",
        "footer",
        "nav",
        "aside",
        "svg",
        "picture",
        "figure",

        # Navigation / share / related / comments / ads.
        ".breadcrumb",
        ".breadcrumbs",
        "[class*='breadcrumb']",
        "[class*='Breadcrumb']",
        "[class*='share']",
        "[class*='Share']",
        "[class*='social']",
        "[class*='Social']",
        "[class*='related']",
        "[class*='Related']",
        "[class*='comment']",
        "[class*='Comment']",
        "[class*='advert']",
        "[class*='Advert']",
        "[class*='banner']",
        "[class*='Banner']",
        "[class*='popup']",
        "[class*='Popup']",
        "[class*='modal']",
        "[class*='Modal']",

        # Vinmec-specific CTA / doctor / booking / consultation blocks.
        "[class*='booking']",
        "[class*='Booking']",
        "[class*='appointment']",
        "[class*='Appointment']",
        "[class*='consult']",
        "[class*='Consult']",
        "[class*='hotline']",
        "[class*='Hotline']",
        "[class*='doctor']",
        "[class*='Doctor']",
        "[href*='/vie/dang-ky-kham']",
        "[href*='/vie/chuyen-gia-y-te']",
        "[href*='/myvinmec']",
    )

    useless_exact_lines = {
        "Mục lục",
        "☰",
        "Đặt lịch khám",
        "Xem chi tiết",
        "Cố vấn chuyên môn",
        "Chia sẻ",
        "ĐỂ LẠI THÔNG TIN TƯ VẤN",
        "Cảm ơn bạn đã gửi thông tin, chúng tôi sẽ liên hệ với bạn trong thời gian sớm nhất.",
        "Cảm ơn bạn đã gửi thông tin, chúng tôi sẽ liên hệ với bạn trong",
        "thời gian sớm nhất.",
        "Cảm ơn bạn đã gửi thông tin, chúng tôi sẽ liên hệ với bạn trong thời",
        "gian sớm nhất.",
        "[Trang chủ](/vie/)",
        "[Bài viết](/vie/bai-viet/)",
    }

    line_noise_prefixes = (
        "Trang chủ",
        "Bài viết",
        "Ngày ",
        "Cập nhật:",
        "Chia sẻ",
        "Để đặt lịch khám",
        "Tải và đặt lịch khám",
        "ĐỂ LẠI THÔNG TIN TƯ VẤN",
        "Cảm ơn bạn đã gửi thông tin",
    )

    line_noise_contains = (
        "javascript:modal.show",
        "HOTLINE",
        "MyVinmec",
        "đặt lịch trực tiếp",
        "đặt lịch khám tự động",
        "/vie/dang-ky-kham",
        "/myvinmec",
        "/vie/chuyen-gia-y-te/",
    )

    stop_markers = (
        "\nĐể đặt lịch khám",
        "\nCập nhật:",
        "\nChia sẻ",
        "\nĐỂ LẠI THÔNG TIN TƯ VẤN",
        "\nCảm ơn bạn đã gửi thông tin",
    )

    def extract_raw_main_html(self, response: scrapy.http.Response) -> str:
        """Return the best Vinmec article-body candidate.

        We score site-specific candidate selectors instead of using a universal
        article/main fallback. This still has fallback tolerance, but the logic
        lives in VinmecStrategy where it belongs.
        """
        candidates: list[tuple[int, str]] = []

        for selector in self.main_content_selectors:
            for html in response.css(selector).getall():
                score = self._score_html_candidate(html)
                if score > 0:
                    candidates.append((score, html))

        if not candidates:
            return ""

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _score_html_candidate(self, html: str) -> int:
        soup = BeautifulSoup(html, "lxml")
        text = normalize_space(soup.get_text(" ")) or ""
        if len(text) < 500:
            return 0

        score = len(text)

        positive_cues = (
            "triệu chứng",
            "nguyên nhân",
            "chẩn đoán",
            "điều trị",
            "phòng ngừa",
            "biến chứng",
            "bệnh",
            "xét nghiệm",
            "thuốc",
            "dấu hiệu",
            "yếu tố nguy cơ",
        )
        negative_cues = (
            "để lại thông tin tư vấn",
            "để đặt lịch khám",
            "hotline",
            "myvinmec",
            "chia sẻ",
        )

        lower = text.lower()
        score += 300 * sum(1 for cue in positive_cues if cue in lower)
        score -= 120 * sum(1 for cue in negative_cues if cue in lower)
        return score

    def clean_article_html(self, html: str) -> str:
        if not html:
            return ""

        soup = BeautifulSoup(html, "lxml")
        self.remove_selectors_from_soup(soup, self.remove_selectors)
        self.remove_vinmec_boilerplate_blocks(soup)
        self.remove_empty_tags(soup)
        return str(soup)

    def remove_vinmec_boilerplate_blocks(self, soup: BeautifulSoup) -> None:
        protected_tags = {"html", "body", "main", "article", "section", "div"}
        noisy_phrases = (
            "Để đặt lịch khám",
            "Tải và đặt lịch khám tự động",
            "ĐỂ LẠI THÔNG TIN TƯ VẤN",
            "Cảm ơn bạn đã gửi thông tin",
            "HOTLINE",
            "MyVinmec",
            "Chia sẻ",
        )

        for tag in list(soup.find_all(True)):
            text = normalize_space(tag.get_text(" ")) or ""
            if not text:
                continue

            if tag.name not in protected_tags and any(phrase.lower() in text.lower() for phrase in noisy_phrases):
                tag.decompose()

    def extract_author(self, response: scrapy.http.Response) -> str | None:
        author = self.first_text(
            response,
            (
                "meta[name='author']::attr(content)",
                "[class*='author']::text",
                "[class*='Author']::text",
            ),
        )
        return author

    def extract_medical_reviewed_by(self, response: scrapy.http.Response) -> str | None:
        text = normalize_space(" ".join(response.css("body ::text").getall())) or ""
        markers = (
            "Cố vấn chuyên môn",
            "Được tư vấn chuyên môn bởi",
            "Tham vấn chuyên môn",
            "Bài viết được tư vấn chuyên môn",
        )
        for marker in markers:
            idx = text.lower().find(marker.lower())
            if idx >= 0:
                snippet = text[idx: idx + 260]
                for stop in ("Mục lục", "Đặt lịch khám", "Xem chi tiết"):
                    if stop in snippet:
                        snippet = snippet.split(stop, 1)[0]
                return normalize_space(snippet)
        return None

    def extract_related_links(self, response: scrapy.http.Response, limit: int = 20) -> list[str]:
        # Do not fall back to every a[href]. On Vinmec that pulls header/footer/privacy
        # links and pollutes frontmatter. Only collect links from likely related blocks.
        selectors = (
            "[class*='related'] a::attr(href)",
            "[class*='Related'] a::attr(href)",
            "[class*='recommend'] a::attr(href)",
            "[class*='Recommend'] a::attr(href)",
            "[class*='same-topic'] a::attr(href)",
        )
        links: list[str] = []
        for selector in selectors:
            for href in response.css(selector).getall():
                url = self.normalize_url(urljoin(response.url, href))
                if self.is_article_url(url) and url != self.normalize_url(response.url):
                    links.append(url)
        return self.unique_keep_order(links)[:limit]

    def clean_markdown(self, markdown: str, title: str | None = None) -> str:
        markdown = self.cut_after_vinmec_footer(markdown)
        markdown = self.remove_duplicate_intro_and_noise(markdown, title=title)
        markdown = self.promote_common_heading_lines(markdown)
        markdown = self.final_markdown_cleanup(markdown, title=title)
        return markdown.strip()

    def cut_after_vinmec_footer(self, markdown: str) -> str:
        cut_pos = None
        for marker in self.stop_markers:
            pos = markdown.find(marker)
            if pos != -1:
                cut_pos = pos if cut_pos is None else min(cut_pos, pos)
        if cut_pos is not None:
            markdown = markdown[:cut_pos]
        return markdown.strip()

    def remove_duplicate_intro_and_noise(self, markdown: str, title: str | None = None) -> str:
        lines = markdown.splitlines()
        result: list[str] = []
        title_norm = self.normalize_for_compare(title) if title else ""
        skipped_title = False

        for idx, raw in enumerate(lines):
            line = raw.strip()

            if not line:
                if result and result[-1] != "":
                    result.append("")
                continue

            line_norm = self.normalize_for_compare(line)

            # Vinmec often emits title, breadcrumb, then title again near top.
            if title_norm and line_norm == title_norm and idx < 40:
                if not skipped_title:
                    skipped_title = True
                continue

            if line.startswith("# ") and title_norm and self.normalize_for_compare(line[2:]) == title_norm and idx < 40:
                if not skipped_title:
                    skipped_title = True
                continue

            if idx < 80 and self.is_noise_markdown_line(line):
                continue

            # Remove doctor-card role fragments before article body.
            if idx < 80 and line in {"Phó giáo sư,", "Tiến sĩ,", "Thạc sĩ,", "Bác sĩ,"}:
                continue

            result.append(raw)

        text = "\n".join(result)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def promote_common_heading_lines(self, markdown: str) -> str:
        heading_patterns = (
            r"^.+ là gì\??$",
            r"^Triệu chứng .+",
            r"^Dấu hiệu .+",
            r"^Nguyên nhân .+",
            r"^Chẩn đoán .+",
            r"^Điều trị .+",
            r"^Cách điều trị .+",
            r"^Phòng ngừa .+",
            r"^Biến chứng .+",
            r"^Yếu tố nguy cơ .+",
            r"^Khi nào cần gặp bác sĩ\??$",
        )
        numbered = re.compile(r"^\d+\.\s+.+")
        lines = markdown.splitlines()
        result: list[str] = []

        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                result.append(raw)
                continue

            if len(line.split()) <= 16 and any(re.match(pat, line, flags=re.IGNORECASE) for pat in heading_patterns):
                result.append(f"## {line}")
                continue

            if len(line.split()) <= 14 and numbered.match(line):
                result.append(f"### {line}")
                continue

            result.append(raw)

        return re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip()

    def final_markdown_cleanup(self, markdown: str, title: str | None = None) -> str:
        lines = markdown.splitlines()
        result: list[str] = []
        title_norm = self.normalize_for_compare(title) if title else ""
        seen_h1 = False

        for raw in lines:
            line = raw.strip()

            if not line:
                if result and result[-1] != "":
                    result.append("")
                continue

            if line.startswith("# ") and title_norm and self.normalize_for_compare(line[2:]) == title_norm:
                if seen_h1:
                    continue
                seen_h1 = True

            if self.is_noise_markdown_line(line):
                continue

            result.append(raw)

        text = "\n".join(result)
        text = self.strip_markdown_links(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def is_noise_markdown_line(self, line: str) -> bool:
        plain = line.strip()
        plain = re.sub(r"^#{1,6}\s+", "", plain)
        plain = plain.replace("**", "").replace("__", "").strip()
        plain = re.sub(r"^(=>|->|–>|—>|»|›|•|-|\*)\s*", "", plain).strip()

        if plain in self.useless_exact_lines:
            return True

        if any(plain.lower().startswith(prefix.lower()) for prefix in self.line_noise_prefixes):
            return True

        if any(fragment.lower() in plain.lower() for fragment in self.line_noise_contains):
            return True

        # Link-only navigation / doctor / appointment / topic chips.
        if re.fullmatch(r"\[(Trang chủ|Bài viết|Đặt lịch khám|Xem chi tiết|[^\]]+)\]\(/vie/(dang-ky-kham|chuyen-gia-y-te|chu-de|bai-viet|chuyen-trang-suc-khoe|mien-dich-di-ung|suc-khoe-tong-hop|tin-tuc|benh).*?\)", plain, flags=re.IGNORECASE):
            return True

        return False
