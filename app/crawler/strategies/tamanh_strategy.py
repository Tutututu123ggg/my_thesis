import re
from urllib.parse import urljoin, urlparse

import scrapy
from bs4 import BeautifulSoup

from app.crawler.markdown_converter import html_to_markdown, normalize_space


class TamAnhStrategy:
    source = "tamanh"
    allowed_domain = "tamanhhospital.vn"

    seed_urls = [
        "https://tamanhhospital.vn/benh-hoc-a-z/",
    ]

    article_prefixes = [
        "/benh/",
    ]

    blacklist_paths = [
        "/wp-content/",
        "/wp-json/",
        "/feed/",
        "/tag/",
        "/author/",
        "/lien-he",
        "/dat-lich-kham",
        "/chuyen-gia/",
        "/chuyen-khoa/",
        "/dich-vu/",
    ]

    bad_extensions = (
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

    remove_selectors = [
        "script",
        "style",
        "noscript",
        "iframe",
        "form",
        "header",
        "footer",
        "nav",
        ".menu",
        ".sidebar",
        ".breadcrumb",
        ".comment",
        ".comments",
        ".social",
        ".share",
        ".ads",
        ".advertisement",
        ".toc",
        ".table-of-contents",
        ".ez-toc-container",
        ".related-post",
        ".related-posts",
    ]

    useless_exact_lines = {
        "Mục lục",
        "ĐẶT LỊCH HẸN",
        "XEM HỒ SƠ",
        "Tư vấn chuyên môn bài viết",
        "Bệnh viện Đa khoa Tâm Anh",
    }

    def normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        return parsed._replace(query="", fragment="").geturl().rstrip("/")

    def is_same_domain(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        return domain == self.allowed_domain

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

    def extract_title(self, response: scrapy.http.Response) -> str | None:
        selectors = [
            "h1::text",
            "meta[property='og:title']::attr(content)",
            "title::text",
        ]

        for selector in selectors:
            value = normalize_space(response.css(selector).get())
            if value:
                return value

        return None

    def extract_description(self, response: scrapy.http.Response) -> str | None:
        selectors = [
            "meta[name='description']::attr(content)",
            "meta[property='og:description']::attr(content)",
        ]

        for selector in selectors:
            value = normalize_space(response.css(selector).get())
            if value:
                return value

        return None

    def extract_published_at(self, response: scrapy.http.Response) -> str | None:
        selectors = [
            "meta[property='article:published_time']::attr(content)",
            "time::attr(datetime)",
        ]

        for selector in selectors:
            value = normalize_space(response.css(selector).get())
            if value:
                return value

        text = " ".join(response.css("body ::text").getall())
        text = normalize_space(text) or ""

        match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text)
        if match:
            return match.group(0)

        return None

    def extract_updated_at(self, response: scrapy.http.Response) -> str | None:
        selectors = [
            "meta[property='article:modified_time']::attr(content)",
            "meta[name='last-modified']::attr(content)",
        ]

        for selector in selectors:
            value = normalize_space(response.css(selector).get())
            if value:
                return value

        return None

    def extract_author(self, response: scrapy.http.Response) -> str | None:
        """
        Tâm Anh thường có block:
        Tư vấn chuyên môn bài viết
        TS.BS...
        Bệnh viện...
        """
        text = " ".join(response.css("body ::text").getall())
        text = normalize_space(text) or ""

        marker = "Tư vấn chuyên môn bài viết"
        if marker in text:
            after = text.split(marker, 1)[1].strip()

            stop_markers = [
                "ĐẶT LỊCH",
                "XEM HỒ SƠ",
                "Viêm da cơ địa",
                "Nội dung",
            ]

            for stop in stop_markers:
                if stop in after:
                    after = after.split(stop, 1)[0].strip()

            # Lấy vừa đủ, tránh nuốt nguyên bài
            author = after[:180].strip()
            return normalize_space(author)

        selectors = [
            "meta[name='author']::attr(content)",
            ".author::text",
            ".post-author::text",
        ]

        for selector in selectors:
            value = normalize_space(response.css(selector).get())
            if value:
                return value

        return None

    def extract_raw_main_html(self, response: scrapy.http.Response) -> str:
        """
        Lấy HTML vùng nội dung chính nhưng chưa clean.
        Phần này dùng cho cả related_links để không mất thẻ <a>.
        """
        candidate_selectors = [
            "article",
            "main",
            ".entry-content",
            ".post-content",
            ".single-content",
            ".content-detail",
            ".detail-content",
            ".the-content",
            ".news-detail",
        ]

        best_html = ""

        for selector in candidate_selectors:
            html = "\n".join(response.css(selector).getall())
            if len(html) > len(best_html):
                best_html = html

        if len(best_html) < 1000:
            best_html = response.css("body").get() or response.text

        return best_html

    def clean_article_html(self, html: str) -> str:
        """
        Clean DOM nhưng vẫn giữ h2/h3/a/p/ul/ol.
        Không xóa các container lớn như body/main/article để tránh mất toàn bộ bài.
        """
        soup = BeautifulSoup(html, "lxml")

        for selector in self.remove_selectors:
            for tag in soup.select(selector):
                tag.decompose()

        noisy_phrases = [
            "HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH",
            "Để đặt lịch khám",
            "Quý khách vui lòng",
            "Tổng đài tư vấn",
        ]

        protected_tags = {
            "html",
            "body",
            "main",
            "article",
            "section",
            "div",
        }

        # Chỉ xóa tag nhỏ chứa CTA, không xóa container lớn.
        for tag in list(soup.find_all(True)):
            if tag.name in protected_tags:
                continue

            text = normalize_space(tag.get_text(" "))
            if not text:
                continue

            if any(phrase in text for phrase in noisy_phrases):
                tag.decompose()

        return str(soup)
    def extract_markdown(
        self,
        response: scrapy.http.Response,
        title: str | None = None,
    ) -> str:
        raw_html = self.extract_raw_main_html(response)
        clean_html = self.clean_article_html(raw_html)
        markdown = html_to_markdown(clean_html)

        markdown = self.remove_duplicate_intro_lines(markdown, title=title)
        markdown = self.promote_plain_heading_lines(markdown)
        markdown = self.cut_after_medical_content(markdown)
        markdown = self.final_markdown_cleanup(markdown, title=title)

        return markdown.strip()

    def remove_duplicate_intro_lines(
        self,
        markdown: str,
        title: str | None = None,
    ) -> str:
        """
        Bỏ title/date/author bị lặp ở đầu body.
        Không động vào phần nội dung chính.
        """
        lines = markdown.splitlines()
        cleaned: list[str] = []

        title_norm = normalize_space(title) if title else None
        skipped_title = False

        for idx, line in enumerate(lines):
            stripped = normalize_space(line)

            if not stripped:
                # Tránh giữ quá nhiều dòng trắng ở đầu
                if cleaned:
                    cleaned.append("")
                continue

            # Bỏ title bị lặp ở đầu bài
            if (
                title_norm
                and not skipped_title
                and stripped == title_norm
                and idx < 20
            ):
                skipped_title = True
                continue

            # Bỏ ngày xuất bản bị lặp ở đầu bài dạng dd/mm/yyyy
            if idx < 30 and re.fullmatch(r"\d{2}/\d{2}/\d{4}", stripped):
                continue

            # Bỏ một số dòng rác chính xác
            if idx < 60 and stripped in self.useless_exact_lines:
                continue

            # Bỏ tên bệnh bị lặp ngay đầu bài nếu dòng quá ngắn
            if idx < 40 and title_norm:
                title_first_phrase = title_norm.split(" là gì")[0]
                if stripped == title_first_phrase:
                    continue

            cleaned.append(line)

        text = "\n".join(cleaned)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def promote_plain_heading_lines(self, markdown: str) -> str:
        """
        Một số trang Tâm Anh có h2 bị rơi thành text thường.
        Hàm này nâng các dòng giống heading y khoa thành ##.
        """
        known_heading_patterns = [
            r"^.+ là gì\?$",
            r"^Triệu chứng .+",
            r"^Dấu hiệu .+",
            r"^Nguyên nhân .+",
            r"^Các biến chứng .+",
            r"^Biến chứng .+",
            r"^Chẩn đoán .+",
            r"^Cách điều trị .+",
            r"^Điều trị .+",
            r"^Cách phòng ngừa .+",
            r"^Phòng ngừa .+",
            r"^Một số câu hỏi thường gặp$",
            r"^Câu hỏi thường gặp$",
            r"^Khi nào cần gặp bác sĩ\?$",
        ]

        numbered_subheading = re.compile(r"^\d+\.\s+.+")
        lines = markdown.splitlines()
        result: list[str] = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                result.append(line)
                continue

            if stripped.startswith("#"):
                result.append(line)
                continue

            is_known_heading = any(
                re.match(pattern, stripped, flags=re.IGNORECASE)
                for pattern in known_heading_patterns
            )

            if is_known_heading and len(stripped.split()) <= 14:
                result.append(f"## {stripped}")
                continue

            if numbered_subheading.match(stripped) and len(stripped.split()) <= 14:
                result.append(f"### {stripped}")
                continue

            result.append(line)

        text = "\n".join(result)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_related_links(
        self,
        response: scrapy.http.Response,
        limit: int = 30,
    ) -> list[str]:
        """
        Extract từ raw HTML để không mất thẻ <a>.
        """
        raw_html = self.extract_raw_main_html(response)
        soup = BeautifulSoup(raw_html, "lxml")

        links: list[str] = []

        for a in soup.find_all("a", href=True):
            url = urljoin(response.url, a["href"])
            url = self.normalize_url(url)

            if self.is_article_url(url) and url != self.normalize_url(response.url):
                links.append(url)

        return self.unique_keep_order(links)[:limit]

    def unique_keep_order(self, items: list[str]) -> list[str]:
        seen = set()
        result = []

        for item in items:
            if item in seen:
                continue

            seen.add(item)
            result.append(item)

        return result
    
    def cut_after_medical_content(self, markdown: str) -> str:
        """
        Cắt footer / địa chỉ bệnh viện / nguồn tham khảo / bài liên quan / CTA.
        Mục tiêu: extractor chỉ nhận nội dung y khoa chính.
        """
        stop_markers = [
            "\n- Bệnh viện Đa khoa Tâm Anh Hà Nội:",
            "\n- Bệnh viện Đa khoa Tâm Anh TP.HCM:",
            "\n- Bệnh viện Đa khoa Tâm Anh – Quận 8:",
            "\n- Phòng khám Đa khoa Tâm Anh Quận 7:",
            "\n- Phòng khám Đa khoa Tâm Anh Cầu Giấy:",
            "\n- Fanpage:",
            "\n- Website:",
            "\nCập nhật lần cuối:",
            "\nChia sẻ:",
            "\n**Nguồn tham khảo**",
            "\nNguồn tham khảo",
            "\nChủ đề:",
            "\n### BÀI VIẾT LIÊN QUAN",
            "\n### BÀI VIẾT CÙNG CHỦ ĐỀ",
            "\nĐĂNG KÝ NHẬN TIN",
            "\n## ĐỐI TÁC BẢO HIỂM",
            "\nCopyright ©",
        ]

        cut_pos = None

        for marker in stop_markers:
            pos = markdown.find(marker)
            if pos != -1:
                cut_pos = pos if cut_pos is None else min(cut_pos, pos)

        if cut_pos is not None:
            markdown = markdown[:cut_pos]

        return markdown.strip()


    def final_markdown_cleanup(self, markdown: str, title: str | None = None) -> str:
        """
        Dọn lần cuối:
        - bỏ title h1 bị lặp
        - bỏ CTA/author block đầu bài
        - bỏ dòng rác chính xác
        """
        lines = markdown.splitlines()
        result = []

        normalized_title = normalize_space(title) if title else None
        seen_h1 = False

        skip_exact = {
            "Tư vấn chuyên môn bài viết",
            "[ĐẶT LỊCH HẸN](https://tamanhhospital.vn/chuyen-gia/dang-thi-ngoc-bich/)",
            "[XEM HỒ SƠ](https://tamanhhospital.vn/chuyen-gia/dang-thi-ngoc-bich/)",
            "Bệnh viện Đa khoa Tâm Anh TP.HCM",
            "Phòng khám Đa khoa Tâm Anh Quận 7",
        }

        for line in lines:
            stripped = line.strip()

            # Bỏ H1 lặp: "# title"
            if stripped.startswith("# "):
                h1_text = stripped[2:].strip()
                h1_text = h1_text.replace("**", "")
                h1_text = normalize_space(h1_text)

                if normalized_title and h1_text == normalized_title:
                    if seen_h1:
                        continue
                    seen_h1 = True

            if stripped in skip_exact:
                continue

            if stripped.startswith("[Bệnh viện Đa khoa Tâm Anh]("):
                continue

            if stripped.startswith("Tư vấn chuyên môn bài viết["):
                continue

            result.append(line)

        text = "\n".join(result)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()