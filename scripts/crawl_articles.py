from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# Allow running from repository root without a Scrapy project/scrapy.cfg.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.crawler.medical_article_spider import MedicalArticleSpider  # noqa: E402


SUPPORTED_STRATEGIES = {"tamanh", "vinmec", "medlatec"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl medical articles with site-specific strategies.")
    parser.add_argument("--strategy", default="vinmec", help="Strategy name for single URL mode: tamanh, vinmec, medlatec")
    parser.add_argument("--start-url", default=None, help="Exact article URL. Useful for curated link input.")
    parser.add_argument("--input-csv", default=None, help="CSV file containing approved seed URLs for batch crawl.")
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--sources", nargs="*", default=None, help="Optional source filter in batch mode.")
    parser.add_argument("--limit", type=int, default=None, help="Optional total URL limit in batch mode.")
    parser.add_argument("--limit-per-source", type=int, default=None, help="Optional per-source URL limit in batch mode.")
    parser.add_argument("--output-dir", default="data/raw/articles_multisource_test")
    parser.add_argument("--max-articles", type=int, default=5, help="Discovery mode limit; single URL mode should use 1.")
    parser.add_argument("--min-words", type=int, default=250)
    parser.add_argument("--category", default="benh")
    return parser.parse_args()


def load_batch_urls(args: argparse.Namespace) -> dict[str, list[str]]:
    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    wanted_sources = {item.strip().lower() for item in args.sources} if args.sources else None
    grouped: dict[str, list[str]] = defaultdict(list)
    total = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if args.url_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing URL column '{args.url_column}'. Columns: {reader.fieldnames}")
        if args.source_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing source column '{args.source_column}'. Columns: {reader.fieldnames}")

        seen: set[str] = set()
        for row in reader:
            url = (row.get(args.url_column) or "").strip()
            source = (row.get(args.source_column) or "").strip().lower()
            if not url or not source:
                continue
            if wanted_sources and source not in wanted_sources:
                continue
            if source not in SUPPORTED_STRATEGIES:
                print(f"[WARN] Skip unsupported source={source} url={url}")
                continue
            if url in seen:
                continue
            if args.limit_per_source is not None and len(grouped[source]) >= args.limit_per_source:
                continue
            if args.limit is not None and total >= args.limit:
                break
            seen.add(url)
            grouped[source].append(url)
            total += 1

    return dict(grouped)


def main() -> None:
    args = parse_args()
    settings = get_project_settings()
    process = CrawlerProcess(settings=settings)

    if args.input_csv:
        grouped = load_batch_urls(args)
        if not grouped:
            raise SystemExit("No URLs to crawl after filtering.")

        total = sum(len(urls) for urls in grouped.values())
        print(f"[BATCH CRAWL] total_urls={total} output_dir={args.output_dir}")
        for source, urls in grouped.items():
            print(f"[BATCH CRAWL] source={source} urls={len(urls)}")
            process.crawl(
                MedicalArticleSpider,
                strategy=source,
                start_urls_json=json.dumps(urls, ensure_ascii=False),
                output_dir=args.output_dir,
                max_articles=len(urls),
                min_words=args.min_words,
                category=args.category,
            )
    else:
        process.crawl(
            MedicalArticleSpider,
            strategy=args.strategy,
            start_url=args.start_url,
            output_dir=args.output_dir,
            max_articles=args.max_articles,
            min_words=args.min_words,
            category=args.category,
        )

    process.start()


if __name__ == "__main__":
    main()
