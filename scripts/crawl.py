import argparse
from pathlib import Path

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from app.crawler.tamanh_spider import TamAnhSpider


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl Vietnamese medical articles.")

    parser.add_argument(
        "--source",
        type=str,
        default="tamanh",
        choices=["tamanh"],
        help="Nguồn crawl.",
    )

    parser.add_argument(
        "--start-url",
        type=str,
        default=None,
        help="URL bắt đầu. Có thể là listing page hoặc article page.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/articles",
        help="Thư mục lưu markdown.",
    )

    parser.add_argument(
        "--max-articles",
        type=int,
        default=20,
        help="Số bài tối đa.",
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    settings = get_project_settings()
    process = CrawlerProcess(settings)

    if args.source == "tamanh":
        process.crawl(
            TamAnhSpider,
            output_dir=args.output_dir,
            max_articles=args.max_articles,
            start_url=args.start_url,
        )

    process.start()


if __name__ == "__main__":
    main()