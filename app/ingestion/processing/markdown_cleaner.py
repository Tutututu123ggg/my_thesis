import re


NOISE_LINE_PATTERNS = [
    r"^ĐẶT LỊCH HẸN$",
    r"^XEM HỒ SƠ$",
    r"^Tư vấn chuyên môn bài viết$",
    r"^Cập nhật lần cuối:",
    r"^Chia sẻ:",
    r"^BÀI VIẾT LIÊN QUAN$",
    r"^BÀI VIẾT CÙNG CHỦ ĐỀ$",
    r"^ĐĂNG KÝ NHẬN TIN$",
    r"^ĐỐI TÁC BẢO HIỂM$",
    r"^Copyright",
]


def normalize_newlines(text: str) -> str:
    """
    Chuẩn hóa xuống dòng nhưng không làm mất Markdown formatting.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_heading_for_compare(text: str) -> str:
    """
    Chỉ dùng để so sánh heading/title.

    Không dùng hàm này để biến đổi body/chunk text,
    vì **bold**, __bold__, _italic_ trong Markdown là tín hiệu hữu ích
    cho entity extractor.
    """
    text = text.strip()

    # Bỏ marker heading nếu có
    text = re.sub(r"^#{1,6}\s+", "", text).strip()

    # Bỏ markdown emphasis chỉ để so sánh
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("*", "")
    text = text.replace("_", "")

    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_duplicate_h1(body: str, title: str) -> str:
    """
    Crawler có thể tạo:
    # Title
    # Title

    Ta chỉ giữ H1 đầu tiên.

    Lưu ý:
    - Không xóa Markdown formatting trong body.
    - Chỉ normalize heading để so sánh.
    """
    lines = body.splitlines()
    result: list[str] = []

    normalized_title = normalize_heading_for_compare(title)
    seen_h1 = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("# "):
            h1_text = normalize_heading_for_compare(stripped)

            if h1_text == normalized_title:
                if seen_h1:
                    continue

                seen_h1 = True

        result.append(line)

    return "\n".join(result)


def remove_noise_lines(body: str) -> str:
    """
    Xóa các dòng rác còn sót từ crawler.
    Không xóa Markdown formatting của nội dung y khoa.
    """
    result: list[str] = []

    for line in body.splitlines():
        stripped = line.strip()

        if any(
            re.search(pattern, stripped, flags=re.IGNORECASE)
            for pattern in NOISE_LINE_PATTERNS
        ):
            continue

        result.append(line)

    return "\n".join(result)


def clean_markdown_body(body: str, title: str) -> str:
    """
    Clean body Markdown cho ingestion.

    Nguyên tắc:
    - Giữ nguyên Markdown gốc càng nhiều càng tốt.
    - Không plain-text hóa body.
    - Không xóa **bold**, _italic_, link.
    - Chỉ xóa duplicate H1, noise line, newline thừa.
    """
    body = normalize_newlines(body)
    body = remove_duplicate_h1(body, title)
    body = remove_noise_lines(body)
    body = normalize_newlines(body)
    return body