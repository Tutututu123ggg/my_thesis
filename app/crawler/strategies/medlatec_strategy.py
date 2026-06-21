from __future__ import annotations

import re
from urllib.parse import urljoin

import scrapy
from bs4 import BeautifulSoup, Tag

from app.crawler.markdown_converter import html_to_markdown, normalize_space
from app.crawler.strategies.base_strategy import BaseMedicalStrategy


class MedlatecStrategy(BaseMedicalStrategy):
    """Medlatec-specific HTML -> Markdown strategy."""

    source = "medlatec"
    source_name = "Medlatec"
    source_domain = "medlatec.vn"
    allowed_domains = ["medlatec.vn"]

    seed_urls = [
        "https://medlatec.vn/tin-tuc",
    ]

    article_prefixes = [
        "/tin-tuc/",
        "/tu-dien-benh-ly/",
    ]

    blacklist_paths = BaseMedicalStrategy.blacklist_paths + [
        "/dich-vu/",
        "/chuyen-khoa/",
        "/dat-lich/",
        "/gioi-thieu/",
        "/lien-he/",
        "/bang-gia/",
        "/uu-dai/",
    ]

    title_selectors = (
        "h1::text",
        "h1 *::text",
        "meta[property='og:title']::attr(content)",
        "title::text",
    )

    main_content_selectors = (
        "article",
        "main article",
        "[class*='detail-content']",
        "[class*='content-detail']",
        "[class*='post-detail']",
        "[class*='news-detail']",
        "[class*='article-detail']",
        "[class*='article-content']",
        "[class*='post-content']",
        "[class*='entry-content']",
        "[class*='content']",
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
        "[class*='booking']",
        "[class*='Booking']",
        "[class*='appointment']",
        "[class*='Appointment']",
        "[class*='consult']",
        "[class*='Consult']",
        "[class*='hotline']",
        "[class*='Hotline']",
        "[href*='/dat-lich']",
        "[href*='/chuyen-khoa']",
        "[href*='/dich-vu']",
    )

    useless_exact_lines = {
        "Mục lục",
        "Chia sẻ",
        "Đặt lịch khám",
        "Đăng ký tư vấn",
        "Bài viết liên quan",
        "Tin cùng chuyên mục",
        "Có thể bạn quan tâm",
        "Tin tức",
        "Từ khoá:",
        "Từ khóa:",
        "Bình luận ()",
        "Đăng ký khám và tư vấn",
        "Lựa chọn dịch vụ",
    }

    line_noise_prefixes = (
        "Xem thêm:",
        "Tìm hiểu thêm:",
        "Tham khảo thêm:",
        "Bài viết liên quan",
        "Tin liên quan",
        "Tin cùng chuyên mục",
        "Có thể bạn quan tâm",
        "Tin tức",
        "Từ khoá:",
        "Từ khóa:",
        "Bình luận ()",
        "Đăng ký khám và tư vấn",
        "Lựa chọn dịch vụ",
        "Đặt lịch khám",
        "Đăng ký tư vấn",
        "Hotline",
        "Tổng đài",
        "Cập nhật:",
        "Ngày đăng:",
        "Chia sẻ",
        "Từ khoá:",
        "Từ khóa:",
        "Bình luận",
        "Đăng ký khám",
        "Lựa chọn dịch vụ",
        "Quý khách hàng vui lòng",
        "Hệ thống Y tế MEDLATEC tự hào",
    )

    line_noise_contains = (
        "hotline",
        "đặt lịch",
        "đăng ký tư vấn",
        "tải app",
        "medon",
        "/dich-vu/",
        "/chuyen-khoa/",
        "/dat-lich",
        "bản quyền",
        "copyright",
        "lấy mẫu xét nghiệm tại nhà",
        "lựa chọn dịch vụ y tế",
        "hệ thống y tế medlatec tự hào",
        "địa chỉ y tế uy tín",
        "gửi thông tin tư vấn",
    )

    stop_markers = (
        "\nBài viết liên quan",
        "\nTin liên quan",
        "\nTin cùng chuyên mục",
        "\nCó thể bạn quan tâm",
        "\nTừ khoá:",
        "\nTừ khóa:",
        "\n## Bình luận",
        "\n## Đăng ký khám",
        "\n## Đăng ký tư vấn",
        "\n## Lựa chọn dịch vụ",
        "\nĐăng ký tư vấn",
        "\nĐặt lịch khám",
        "\nHotline",
        "\nTổng đài",
        "\nHệ thống Y tế MEDLATEC tự hào",
        "\nQuý khách hàng vui lòng lựa chọn dịch vụ",
        "\nCopyright",
    )

    def extract_raw_main_html(self, response: scrapy.http.Response) -> str:
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
        if len(text) < 450:
            return 0

        lower = text.lower()
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
            "đặt lịch khám",
            "đăng ký tư vấn",
            "hotline",
            "tin liên quan",
            "bài viết liên quan",
        )

        score = len(text)
        score += 250 * sum(1 for cue in positive_cues if cue in lower)
        score -= 150 * sum(1 for cue in negative_cues if cue in lower)
        return score

    def clean_article_html(self, html: str) -> str:
        if not html:
            return ""

        soup = BeautifulSoup(html, "lxml")
        self.remove_selectors_from_soup(soup, self.remove_selectors)
        self.remove_medlatec_boilerplate_blocks(soup)
        self.remove_empty_tags(soup)
        return str(soup)

    def remove_medlatec_boilerplate_blocks(self, soup: BeautifulSoup) -> None:
        protected_tags = {"html", "body", "main", "article", "section", "div"}
        noisy_phrases = (
            "Đặt lịch khám",
            "Đăng ký tư vấn",
            "Hotline",
            "Tổng đài",
            "Bài viết liên quan",
            "Tin liên quan",
            "Tin cùng chuyên mục",
            "Có thể bạn quan tâm",
        )

        for tag in list(soup.find_all(True)):
            if tag.name in protected_tags:
                continue
            text = normalize_space(tag.get_text(" ")) or ""
            if not text:
                continue
            if any(phrase.lower() in text.lower() for phrase in noisy_phrases):
                tag.decompose()

    def extract_related_links(self, response: scrapy.http.Response, limit: int = 30) -> list[str]:
        selectors = (
            "[class*='related'] a::attr(href)",
            "[class*='Related'] a::attr(href)",
            "[class*='post-list'] a::attr(href)",
        )
        links: list[str] = []
        for selector in selectors:
            for href in response.css(selector).getall():
                url = self.normalize_url(urljoin(response.url, href))
                if self.is_article_url(url) and url != self.normalize_url(response.url):
                    links.append(url)
        return self.unique_keep_order(links)[:limit]

    def clean_markdown(self, markdown: str, title: str | None = None) -> str:
        markdown = self.cut_after_footer(markdown)
        markdown = self.remove_duplicate_intro_and_noise(markdown, title=title)
        markdown = self.promote_common_heading_lines(markdown)
        markdown = self.final_markdown_cleanup(markdown, title=title)
        return markdown.strip()

    def cut_after_footer(self, markdown: str) -> str:
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

            # Drop generic Medlatec category heading and top recommended links.
            if idx < 60 and self.is_noise_markdown_line(line):
                continue

            line_norm = self.normalize_for_compare(line)
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

            # Drop standalone dates/source labels near top.
            if idx < 30 and re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{4}", line):
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

        # Top-of-article related links and tag chips in Medlatec are usually
        # rendered as Markdown links, often starting with a date.
        if re.fullmatch(r"\[\d{1,2}/\d{1,2}/\d{4}\s*\|.+?\]\(.+?\)", plain):
            return True
        if re.fullmatch(r"\[[^\]]+\]\(/tin-tuc/tag/[^)]*\)", plain, flags=re.IGNORECASE):
            return True
        if any(plain.lower().startswith(prefix.lower()) for prefix in self.line_noise_prefixes):
            return True
        if any(fragment.lower() in plain.lower() for fragment in self.line_noise_contains):
            return True
        if re.fullmatch(r"\[(Đặt lịch khám|Đăng ký tư vấn|Xem thêm|Tìm hiểu thêm|Tham khảo thêm|Xem chi tiết).*?\]\(.+?\)", plain, flags=re.IGNORECASE):
            return True
        return False
