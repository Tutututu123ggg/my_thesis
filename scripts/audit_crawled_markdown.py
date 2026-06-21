from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


BAD_PATTERNS = [
    "đặt lịch khám",
    "tải app",
    "hotline",
    "bài viết liên quan",
    "có thể bạn quan tâm",
    "theo dõi chúng tôi",
    "đăng ký nhận tin",
    "cookie",
    "copyright",
    "để lại thông tin tư vấn",
    "myvinmec",
    "javascript:modal.show",
]

REQUIRED_FRONTMATTER = [
    "source:",
    "source_domain:",
    "source_name:",
    "url:",
    "title:",
    "crawled_at:",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def body_text(text: str) -> str:
    return re.sub(r"^---.*?---", "", text, count=1, flags=re.S).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", body_text(text), flags=re.UNICODE))


def count_headings(text: str) -> int:
    return len(re.findall(r"^#{1,6}\s+", text, flags=re.M))


def repeated_h1_titles(text: str) -> int:
    h1 = re.findall(r"^#\s+(.+)$", text, flags=re.M)
    return len(h1) - len(set(item.strip().lower() for item in h1))


def frontmatter_missing(text: str) -> list[str]:
    head = text[:1600]
    return [key for key in REQUIRED_FRONTMATTER if key not in head]


def bad_hits(text: str) -> list[str]:
    low = text.lower()
    return [p for p in BAD_PATTERNS if p in low]


def first_frontmatter_value(text: str, key: str) -> str:
    for line in text[:2000].splitlines():
        if line.startswith(key + ":"):
            return line.split(":", 1)[1].strip().strip('"')
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="data/raw/articles_multisource_test")
    parser.add_argument("--preview", type=int, default=500)
    parser.add_argument("--csv", default=None, help="Optional CSV audit report path.")
    parser.add_argument("--min-words", type=int, default=250)
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("*.md"))
    print(f"[AUDIT] root={root} files={len(files)}")
    print()

    rows: list[dict[str, str | int]] = []

    for path in files:
        text = read_text(path)
        words = count_words(text)
        headings = count_headings(text)
        missing = frontmatter_missing(text)
        bad = bad_hits(text)
        dup_h1 = repeated_h1_titles(text)
        source = first_frontmatter_value(text, "source")
        title = first_frontmatter_value(text, "title")
        url = first_frontmatter_value(text, "url")

        status = "ok"
        reasons: list[str] = []
        if words < args.min_words:
            reasons.append("short")
        if missing:
            reasons.append("missing_frontmatter")
        if bad:
            reasons.append("bad_hits")
        if dup_h1:
            reasons.append("duplicate_h1")
        if reasons:
            status = "review"

        rows.append(
            {
                "file": path.name,
                "source": source,
                "title": title,
                "url": url,
                "words": words,
                "headings": headings,
                "dup_h1": dup_h1,
                "missing": ";".join(missing),
                "bad_hits": ";".join(bad),
                "status": status,
                "reasons": ";".join(reasons),
            }
        )

        print("=" * 100)
        print(f"FILE       : {path.name}")
        print(f"WORDS      : {words}")
        print(f"HEADINGS   : {headings}")
        print(f"DUP_H1     : {dup_h1}")
        print(f"MISSING    : {missing if missing else 'OK'}")
        print(f"BAD HITS   : {bad if bad else 'OK'}")
        print(f'title: "{title}"')
        print()
        print(body_text(text)[: args.preview].replace("\n\n\n", "\n\n"))
        print()

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file",
                    "source",
                    "title",
                    "url",
                    "words",
                    "headings",
                    "dup_h1",
                    "missing",
                    "bad_hits",
                    "status",
                    "reasons",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[AUDIT] Saved CSV -> {out}")


if __name__ == "__main__":
    main()
