from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CATEGORY_LINES = {
    "sức khoẻ tổng quát",
    "sức khỏe tổng quát",
    "sức khoẻ tổng hợp",
    "sức khỏe tổng hợp",
    "chuyên trang sức khoẻ",
    "chuyên trang sức khỏe",
    "miễn dịch dị ứng",
}


CUT_MARKERS = [
    "\nTừ khoá:",
    "\nTừ khóa:",
    "\n## Bình luận",
    "\n## Đăng ký khám",
    "\n## Lựa chọn dịch vụ",
    "\n## Video đề xuất",
    "\nVideo đề xuất",
    "\nBài viết liên quan",
    "\nCó thể bạn quan tâm",
    "\nKhách hàng có thể đăng ký",
    "\nQuý khách hàng vui lòng",
    "\nHệ thống Y tế MEDLATEC tự hào",
    "\nMEDLATEC đã có",
    "\nNgười dân có nhu cầu",
    "\nĐể đặt lịch khám",
    "\nĐặt lịch khám",
    "\nTải app",
]


DROP_LINE_PATTERNS = [
    r"^\s*\*\*\s*>+\s*Xem.*$",
    r"^\s*\*?>+\s*Xem.*$",
    r"^\s*Xem thêm\s*:?.*$",
    r"^\s*Xem chi tiết.*$",
    r"^\s*\*\*\s*Xem thêm\s*:?.*$",
    r"^\s*Video đề xuất.*$",
    r"^\s*Bài viết liên quan.*$",
    r"^\s*Có thể bạn quan tâm.*$",
    r"^\s*\[\s*(Chuyên trang sức khoẻ|Chuyên trang sức khỏe|Sức khoẻ tổng hợp|Sức khỏe tổng hợp|Sức khoẻ tổng quát|Sức khỏe tổng quát|Miễn dịch dị ứng)\s*\]\([^)]+\)\s*$",
]


PROMO_PATTERNS = [
    r"kiểm tra sức khỏe định kỳ",
    r"khám sức khỏe tổng quát",
    r"gói khám sức khỏe",
    r"đăng ký khám trực tuyến",
    r"tổng đài.*\d{3,}",
    r"hotline.*\d{3,}",
    r"đường dây nóng.*\d{3,}",
]


def split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text

    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, flags=re.S)
    if not m:
        return "", text

    fm = m.group(1).strip()
    body = text[m.end():].strip()
    return fm, body


def remove_related_links_from_frontmatter(fm: str) -> str:
    if not fm:
        return fm

    lines = fm.splitlines()
    out: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip() == "related_links:":
            i += 1
            while i < len(lines) and re.match(r"^\s+-\s+", lines[i]):
                i += 1
            out.append("related_links:")
            continue

        out.append(line)
        i += 1

    return "\n".join(out).strip()


def strip_markdown_links(text: str) -> str:
    # [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Autolink raw urls
    text = re.sub(r"https?://\S+", "", text)

    return text


def cut_at_noise_markers(body: str) -> str:
    cut_pos = len(body)

    for marker in CUT_MARKERS:
        idx = body.find(marker)
        if idx != -1:
            cut_pos = min(cut_pos, idx)

    return body[:cut_pos].strip()


def is_category_line(line: str) -> bool:
    cleaned = line.strip().strip("*_# ").lower()
    return cleaned in CATEGORY_LINES


def is_top_related_bullet(line: str) -> bool:
    # MEDLATEC hay có block link liên quan ngay sau intro:
    # - [25/09/2024 | Một số bệnh ...](/tin-tuc/...)
    return bool(re.match(r"^\s*-\s*\[\d{2}/\d{2}/\d{4}\s*\|.+\]\(.+\)\s*$", line))


def is_drop_line(line: str) -> bool:
    s = line.strip()

    if not s:
        return False

    if is_category_line(s):
        return True

    if is_top_related_bullet(s):
        return True

    for pat in DROP_LINE_PATTERNS:
        if re.search(pat, s, flags=re.I):
            return True

    # Drop line-level promo, không drop cả paragraph dài nếu chỉ có chữ thường.
    low = s.lower()
    for pat in PROMO_PATTERNS:
        if re.search(pat, low, flags=re.I):
            return True

    return False


def normalize_heading(line: str) -> str:
    # ## **1. Tiêu đề** -> ## 1. Tiêu đề
    line = re.sub(r"^(#{1,6})\s+\*\*(.*?)\*\*\s*$", r"\1 \2", line)
    return line.rstrip()


def clean_body(body: str) -> str:
    body = cut_at_noise_markers(body)
    body = strip_markdown_links(body)

    lines = body.splitlines()
    cleaned: list[str] = []

    for line in lines:
        line = normalize_heading(line)

        if is_drop_line(line):
            continue

        # Bỏ H1 rác MEDLATEC nếu còn sót
        if line.strip().lower() in {"# tin tức", "# sức khỏe tổng quát", "# sức khoẻ tổng quát"}:
            continue

        cleaned.append(line.rstrip())

    body = "\n".join(cleaned)

    # Dọn spacing
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)

    return body.strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def get_source_from_name(path: Path) -> str:
    return path.name.split("__", 1)[0] if "__" in path.name else ""


def get_title_from_frontmatter(fm: str) -> str:
    m = re.search(r'^title:\s*"(.*)"\s*$', fm, flags=re.M)
    return m.group(1) if m else ""


def process_file(path: Path, output_dir: Path, min_words: int, drop_related_links: bool) -> dict:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    fm, body = split_frontmatter(raw)

    words_before = count_words(body)

    if drop_related_links:
        fm = remove_related_links_from_frontmatter(fm)

    cleaned_body = clean_body(body)
    words_after = count_words(cleaned_body)

    status = "kept"
    reason = ""

    if words_after < min_words:
        status = "dropped"
        reason = f"too_short_after_clean:{words_after}"

    if status == "kept":
        output_dir.mkdir(parents=True, exist_ok=True)

        if fm:
            final_text = f"---\n{fm}\n---\n\n{cleaned_body}\n"
        else:
            final_text = cleaned_body + "\n"

        (output_dir / path.name).write_text(final_text, encoding="utf-8")

    return {
        "file": path.name,
        "source": get_source_from_name(path),
        "title": get_title_from_frontmatter(fm),
        "words_before": words_before,
        "words_after": words_after,
        "status": status,
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/articles_approved_v1")
    parser.add_argument("--output-dir", default="data/raw/articles_final_v1")
    parser.add_argument("--report", default="data/raw/articles_final_v1_report.csv")
    parser.add_argument("--min-words", type=int, default=500)
    parser.add_argument("--keep-related-links", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)

    files = sorted(input_dir.glob("*.md"))

    if output_dir.exists():
        for old in output_dir.glob("*.md"):
            old.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in files:
        rows.append(
            process_file(
                path=path,
                output_dir=output_dir,
                min_words=args.min_words,
                drop_related_links=not args.keep_related_links,
            )
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "source",
                "title",
                "words_before",
                "words_after",
                "status",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    kept = sum(1 for r in rows if r["status"] == "kept")
    dropped = sum(1 for r in rows if r["status"] == "dropped")

    print(f"[POST CLEAN] input={input_dir}")
    print(f"[POST CLEAN] output={output_dir}")
    print(f"[POST CLEAN] files={len(files)} kept={kept} dropped={dropped}")
    print(f"[POST CLEAN] report={report_path}")


if __name__ == "__main__":
    main()