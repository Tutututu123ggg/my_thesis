import re
from urllib.parse import urljoin, urlparse

import scrapy
from bs4 import BeautifulSoup, Tag

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

    # Selector chính của nội dung bài viết Tâm Anh.
    main_content_selector = "div#ftwp-postcontent"

    # Xóa TRƯỚC markdownify để related/sidebar/footer không lọt vào Markdown.
    remove_selectors = [
        "script",
        "style",
        "noscript",
        "iframe",
        "form",

        # Layout / navigation chung
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

        # TOC / inserted content trong #ftwp-postcontent
        "#ftwp-container-outer",
        "#ftwp-container",
        ".ftwp-in-post",
        ".toc",
        ".table-of-contents",
        ".ez-toc-container",
        ".content_insert",
        ".alert.alert-info",

        # Related/noise blocks đã debug từ Tâm Anh
        ".div_over.w100",
        ".div_related_bycat",
        ".div_related_bytag",
        ".slide_show4",
        ".owl-carousel",

        # Related generic fallback
        ".related-post",
        ".related-posts",
        ".post-related",
        ".box-related",
        ".news-related",
    ]

    # Các dòng này nếu còn sót sau markdownify thì bỏ.
    useless_exact_lines = {
        "Mục lục",
        "ĐẶT LỊCH HẸN",
        "XEM HỒ SƠ",
        "Tư vấn chuyên môn bài viết",
        "Bệnh viện Đa khoa Tâm Anh",
    }

    line_noise_prefixes = (
        "Xem thêm:",
        "Tìm hiểu thêm:",
        "Tham khảo thêm:",
        "Xem chi tiết:",
        "Cập nhật lần cuối:",
        "Chia sẻ:",
        "Chủ đề:",
        "Hotline:",
        "Fanpage:",
        "Website:",
        "Địa chỉ:",
    )

    line_noise_contains = (
        "ĐẶT LỊCH HẸN",
        "XEM HỒ SƠ",
        "Tổng đài tư vấn",
        "Quý khách vui lòng",
        "Để đặt lịch khám",
        "HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH",
        "Bệnh viện Đa khoa Tâm Anh Hà Nội",
        "Bệnh viện Đa khoa Tâm Anh TP.HCM",
        "Bệnh viện Đa khoa Tâm Anh – Quận 8",
        "Phòng khám Đa khoa Tâm Anh Quận 7",
        "Phòng khám Đa khoa Tâm Anh Cầu Giấy",
        "BÀI VIẾT LIÊN QUAN",
        "BÀI VIẾT CÙNG CHỦ ĐỀ",
        "ĐĂNG KÝ NHẬN TIN",
        "ĐỐI TÁC BẢO HIỂM",
        "Copyright",
    )

    stop_markers = (
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
        "\n## BÀI VIẾT LIÊN QUAN",
        "\n### BÀI VIẾT CÙNG CHỦ ĐỀ",
        "\n## BÀI VIẾT CÙNG CHỦ ĐỀ",
        "\nĐĂNG KÝ NHẬN TIN",
        "\n## ĐỐI TÁC BẢO HIỂM",
        "\nCopyright ©",
    )

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
                "Nội dung",
                "Mục lục",
            ]

            for stop in stop_markers:
                if stop in after:
                    after = after.split(stop, 1)[0].strip()

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
        Lấy đúng vùng nội dung chính của Tâm Anh.

        Quan trọng:
        - Ưu tiên tuyệt đối div#ftwp-postcontent.
        - Không dùng article/main/.post_info.box_detail_post làm nguồn chính,
          vì các vùng đó dễ bao cả related/sidebar/footer.
        """
        html = response.css(self.main_content_selector).get()
        if html:
            return html

        # Fallback rất hẹp để debug nếu Tâm Anh đổi layout.
        # Không dùng fallback generic article/main nữa để tránh nuốt related.
        return ""

    def clean_article_html(self, html: str) -> str:
        """
        Clean DOM trước markdownify.

        Mục tiêu:
        - Giữ nội dung y khoa chính trong #ftwp-postcontent.
        - Xóa TOC, inserted content, related blocks, carousel, CTA, footer.
        """
        if not html:
            return ""

        soup = BeautifulSoup(html, "lxml")

        for selector in self.remove_selectors:
            for tag in soup.select(selector):
                tag.decompose()

        self.remove_cta_and_doctor_blocks(soup)
        self.remove_empty_tags(soup)

        return str(soup)

    def remove_cta_and_doctor_blocks(self, soup: BeautifulSoup) -> None:
        """
        Xóa các tag nhỏ chứa CTA/doctor/hospital/footer.
        Không xóa container lớn như div/section để tránh mất cả bài.
        """
        protected_tags = {
            "html",
            "body",
            "main",
            "article",
            "section",
            "div",
        }

        noisy_phrases = (
            "Tư vấn chuyên môn bài viết",
            "ĐẶT LỊCH HẸN",
            "XEM HỒ SƠ",
            "HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH",
            "Để đặt lịch khám",
            "Quý khách vui lòng",
            "Tổng đài tư vấn",
            "Hotline",
            "Fanpage",
            "Website",
            "Bệnh viện Đa khoa Tâm Anh Hà Nội",
            "Bệnh viện Đa khoa Tâm Anh TP.HCM",
            "Bệnh viện Đa khoa Tâm Anh – Quận 8",
            "Phòng khám Đa khoa Tâm Anh Quận 7",
            "Phòng khám Đa khoa Tâm Anh Cầu Giấy",
        )

        for tag in list(soup.find_all(True)):
            if tag.name in protected_tags:
                continue

            text = normalize_space(tag.get_text(" "))
            if not text:
                continue

            if any(phrase.lower() in text.lower() for phrase in noisy_phrases):
                tag.decompose()

    def remove_empty_tags(self, soup: BeautifulSoup) -> None:
        """
        Xóa tag rỗng sau khi decompose noise.
        """
        for tag in list(soup.find_all(True)):
            if tag.name in {"br", "hr", "img"}:
                continue

            if tag.get_text(strip=True):
                continue

            if isinstance(tag, Tag):
                tag.decompose()

    def extract_markdown(
        self,
        response: scrapy.http.Response,
        title: str | None = None,
    ) -> str:
        raw_html = self.extract_raw_main_html(response)

        if not raw_html:
            return ""

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

        title_norm = self.normalize_for_compare(title) if title else None
        skipped_title = False

        for idx, line in enumerate(lines):
            stripped = normalize_space(line)

            if not stripped:
                if cleaned:
                    cleaned.append("")
                continue

            stripped_norm = self.normalize_for_compare(stripped)

            # Bỏ title bị lặp ở đầu bài.
            if (
                title_norm
                and not skipped_title
                and stripped_norm == title_norm
                and idx < 20
            ):
                skipped_title = True
                continue

            # Bỏ H1 title nếu markdownify sinh ra H1 trùng title.
            if (
                title_norm
                and not skipped_title
                and stripped.startswith("# ")
                and self.normalize_for_compare(stripped[2:]) == title_norm
                and idx < 20
            ):
                skipped_title = True
                continue

            # Bỏ ngày xuất bản bị lặp ở đầu bài dạng dd/mm/yyyy.
            if idx < 30 and re.fullmatch(r"\d{2}/\d{2}/\d{4}", stripped):
                continue

            if idx < 60 and stripped in self.useless_exact_lines:
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
        Lưu related links như metadata nếu cần, nhưng KHÔNG đưa vào body Markdown.

        Vì extract_markdown chỉ lấy #ftwp-postcontent đã clean related blocks,
        related article titles không còn lọt vào chunk text.
        """
        related_selectors = [
            ".div_related_bycat",
            ".div_related_bytag",
            ".related-post",
            ".related-posts",
            ".post-related",
            ".box-related",
            ".news-related",
        ]

        links: list[str] = []

        for selector in related_selectors:
            for href in response.css(f"{selector} a::attr(href)").getall():
                url = urljoin(response.url, href)
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
        Đây là lớp bảo vệ sau markdownify.
        """
        cut_pos = None

        for marker in self.stop_markers:
            pos = markdown.find(marker)
            if pos != -1:
                cut_pos = pos if cut_pos is None else min(cut_pos, pos)

        if cut_pos is not None:
            markdown = markdown[:cut_pos]

        return markdown.strip()

    def final_markdown_cleanup(self, markdown: str, title: str | None = None) -> str:
        """
        Dọn lần cuối:
        - bỏ duplicate H1/title.
        - bỏ dòng Xem thêm/Tham khảo thêm/Tìm hiểu thêm/Xem chi tiết.
        - bỏ CTA/author/hospital/footer còn sót.
        """
        lines = markdown.splitlines()
        result: list[str] = []

        normalized_title = self.normalize_for_compare(title) if title else None
        seen_title_heading = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if result and result[-1] != "":
                    result.append("")
                continue

            # Bỏ H1 trùng title nếu bị lặp.
            if stripped.startswith("# "):
                h1_text = stripped[2:].strip()
                h1_norm = self.normalize_for_compare(h1_text)

                if normalized_title and h1_norm == normalized_title:
                    if seen_title_heading:
                        continue
                    seen_title_heading = True

            if self.is_noise_markdown_line(stripped):
                continue

            result.append(line)

        text = "\n".join(result)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def is_noise_markdown_line(self, line: str) -> bool:
        plain = line.strip()

        plain_no_markdown = plain
        plain_no_markdown = re.sub(r"^#{1,6}\s+", "", plain_no_markdown)
        plain_no_markdown = plain_no_markdown.replace("**", "")
        plain_no_markdown = plain_no_markdown.replace("__", "")
        plain_no_markdown = plain_no_markdown.strip()

        # Bỏ prefix ký hiệu điều hướng kiểu:
        # => Xem thêm:
        # -> Xem thêm:
        # » Xem thêm:
        # • Xem thêm:
        plain_no_markdown = re.sub(
            r"^(=>|->|–>|—>|»|›|•|-|\*)\s*",
            "",
            plain_no_markdown,
        ).strip()

        if plain_no_markdown in self.useless_exact_lines:
            return True

        if any(
            plain_no_markdown.lower().startswith(prefix.lower())
            for prefix in self.line_noise_prefixes
        ):
            return True

        if any(
            phrase.lower() in plain_no_markdown.lower()
            for phrase in self.line_noise_contains
        ):
            return True

        # Link-only CTA kiểu [ĐẶT LỊCH HẸN](...)
        if re.fullmatch(
            r"\[(ĐẶT LỊCH HẸN|XEM HỒ SƠ)\]\(.+?\)",
            plain_no_markdown,
            flags=re.IGNORECASE,
        ):
            return True

        # Link-only điều hướng kiểu [Xem thêm: ...](...)
        if re.fullmatch(
            r"\[(Xem thêm|Tìm hiểu thêm|Tham khảo thêm|Xem chi tiết):?.*?\]\(.+?\)",
            plain_no_markdown,
            flags=re.IGNORECASE,
        ):
            return True

        return False

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