import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.ingestion.extraction import ExtractionService, MedicalExtractor
from app.ingestion.processing import chunk_article, load_markdown_articles
from app.ingestion.processing.ingestion_models import ArticleDocument, ChunkDocument


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_URLS = [
    "https://tamanhhospital.vn/benh/viem-da-co-dia/",
    "https://tamanhhospital.vn/benh/benh-cham/",
    "https://tamanhhospital.vn/benh/benh-vay-nen/",
]

SYNONYM_CUES = [
    "còn gọi là",
    "còn được gọi là",
    "hay gọi là",
    "còn có tên là",
    "tên khác là",
    "tiếng anh là",
    "viết tắt là",
    "aka",
    "also known as",
]

SUBTYPE_CUES = [
    "là dạng của",
    "là thể của",
    "là một dạng",
    "là một thể",
    "là loại",
    "phân loại",
    "bao gồm các thể",
    "các dạng",
]

GENERIC_ENTITY_NAMES = {
    "doi_tuong",
    "benh_ly",
    "bieu_hien_lam_sang",
    "dấu hiệu",
    "dau hieu",
    "triệu chứng",
    "trieu chung",
    "nguyên nhân",
    "nguyen nhan",
    "biến chứng",
    "bien chung",
    "điều trị",
    "dieu tri",
    "phòng ngừa",
    "phong ngua",
    "chẩn đoán",
    "chan doan",
}


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_light(text: str) -> str:
    text = (text or "").strip().lower()
    return " ".join(text.split())


def has_any(text: str, cues: list[str]) -> bool:
    t = normalize_light(text)
    return any(cue in t for cue in cues)


def run_crawler(urls: list[str], output_dir: Path, clean_output: bool) -> None:
    if clean_output and output_dir.exists():
        print(f"[CLEAN] Remove old crawl dir: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    spider_path = PROJECT_ROOT / "app" / "crawler" / "tamanh_spider.py"

    for idx, url in enumerate(urls, start=1):
        print(f"\n[CRAWL] {idx}/{len(urls)} {url}")

        cmd = [
            sys.executable,
            "-m",
            "scrapy",
            "runspider",
            str(spider_path),
            "-a",
            f"start_url={url}",
            "-a",
            "max_articles=1",
            "-a",
            f"output_dir={output_dir}",
        ]

        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
        )

        if completed.returncode != 0:
            raise RuntimeError(f"Crawler failed for url: {url}")


def flatten_chunks(articles: list[ArticleDocument], max_tokens: int) -> list[ChunkDocument]:
    chunks: list[ChunkDocument] = []

    for article in articles:
        article_chunks = chunk_article(article, max_tokens=max_tokens)
        chunks.extend(article_chunks)
        print(
            f"[CHUNK] {article.title} | chunks={len(article_chunks)}"
        )

    return chunks


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    entity_type_counter = Counter()
    relation_type_counter = Counter()
    keyword_len_counter = Counter()

    unique_entities: set[tuple[str, str]] = set()
    unique_relations: set[tuple[str, str, str, str, str]] = set()

    generic_entities: list[dict[str, Any]] = []
    empty_keyword_relations: list[dict[str, Any]] = []
    synonym_relations: list[dict[str, Any]] = []
    suspicious_synonyms: list[dict[str, Any]] = []

    raw_entities = 0
    raw_relations = 0

    for record in records:
        chunk_id = record["chunk_id"]
        title = record["title"]
        graph = record["graph"]

        for entity in graph["entities"]:
            raw_entities += 1
            name = entity["name"]
            entity_type = entity["entity_type"]
            entity_type_counter[entity_type] += 1
            unique_entities.add((normalize_light(name), entity_type))

            if normalize_light(name) in GENERIC_ENTITY_NAMES:
                generic_entities.append(
                    {
                        "chunk_id": chunk_id,
                        "title": title,
                        "name": name,
                        "entity_type": entity_type,
                        "evidence_text": entity.get("evidence_text", ""),
                    }
                )

        for relation in graph["relations"]:
            raw_relations += 1
            relation_type = relation["relation_type"]
            relation_type_counter[relation_type] += 1
            unique_relations.add(
                (
                    normalize_light(relation["subject"]),
                    relation["subject_type"],
                    relation_type,
                    normalize_light(relation["object"]),
                    relation["object_type"],
                )
            )

            keywords = relation.get("keywords") or []
            keyword_len_counter[len(keywords)] += 1

            relation_view = {
                "chunk_id": chunk_id,
                "title": title,
                "subject": relation["subject"],
                "subject_type": relation["subject_type"],
                "relation_type": relation_type,
                "object": relation["object"],
                "object_type": relation["object_type"],
                "keywords": keywords,
                "evidence_text": relation.get("evidence_text", ""),
                "confidence": relation.get("confidence", None),
            }

            if not keywords:
                empty_keyword_relations.append(relation_view)

            if relation_type == "DONG_NGHIA_VOI":
                synonym_relations.append(relation_view)

                evidence = relation.get("evidence_text", "")
                has_positive = has_any(evidence, SYNONYM_CUES)
                has_negative = has_any(evidence, SUBTYPE_CUES)
                same_type = relation["subject_type"] == relation["object_type"]

                if not has_positive or has_negative or not same_type:
                    suspicious_synonyms.append(
                        {
                            **relation_view,
                            "has_positive_synonym_cue": has_positive,
                            "has_negative_subtype_cue": has_negative,
                            "same_entity_type": same_type,
                        }
                    )

    return {
        "totals": {
            "chunks": len(records),
            "raw_entities": raw_entities,
            "unique_entities": len(unique_entities),
            "raw_relations": raw_relations,
            "unique_relations": len(unique_relations),
            "relations_missing_keywords": len(empty_keyword_relations),
            "synonym_relations": len(synonym_relations),
            "suspicious_synonyms": len(suspicious_synonyms),
            "generic_entities": len(generic_entities),
        },
        "entity_type_counter": dict(entity_type_counter),
        "relation_type_counter": dict(relation_type_counter),
        "keyword_len_counter": dict(keyword_len_counter),
        "synonym_relations": synonym_relations,
        "suspicious_synonyms": suspicious_synonyms,
        "empty_keyword_relations_sample": empty_keyword_relations[:30],
        "generic_entities_sample": generic_entities[:30],
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n========== EXTRACTION QUALITY SUMMARY ==========")
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))

    print("\n[RELATION TYPES]")
    for k, v in sorted(summary["relation_type_counter"].items(), key=lambda x: (-x[1], x[0])):
        print(f"- {k}: {v}")

    print("\n[ENTITY TYPES]")
    for k, v in sorted(summary["entity_type_counter"].items(), key=lambda x: (-x[1], x[0])):
        print(f"- {k}: {v}")

    print("\n[KEYWORD LENGTHS]")
    for k, v in sorted(summary["keyword_len_counter"].items(), key=lambda x: int(x[0])):
        print(f"- len={k}: {v}")

    if summary["synonym_relations"]:
        print("\n[DONG_NGHIA_VOI RELATIONS]")
        for item in summary["synonym_relations"][:20]:
            print(
                f"- {item['subject']} --DONG_NGHIA_VOI--> {item['object']} | "
                f"keywords={item['keywords']} | evidence={item['evidence_text']}"
            )

    if summary["suspicious_synonyms"]:
        print("\n[WARN] Suspicious synonym relations:")
        for item in summary["suspicious_synonyms"][:20]:
            print(
                f"- {item['subject']} -> {item['object']} | "
                f"positive={item['has_positive_synonym_cue']} "
                f"negative={item['has_negative_subtype_cue']} "
                f"same_type={item['same_entity_type']} | "
                f"evidence={item['evidence_text']}"
            )

    if summary["empty_keyword_relations_sample"]:
        print("\n[WARN] Relations missing keywords sample:")
        for item in summary["empty_keyword_relations_sample"][:10]:
            print(
                f"- {item['subject']} --{item['relation_type']}--> {item['object']} | "
                f"evidence={item['evidence_text']}"
            )

    if summary["generic_entities_sample"]:
        print("\n[WARN] Generic entities sample:")
        for item in summary["generic_entities_sample"][:10]:
            print(
                f"- {item['name']} ({item['entity_type']}) | evidence={item['evidence_text']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Temporary smoke test: crawl 3 Tam Anh articles from scratch, "
            "run strict extractor+validator, and audit output quality."
        )
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=DEFAULT_URLS,
        help="Article URLs to crawl. Default: 3 dermatology articles.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tmp/extract_3_articles_quality",
        help="Output directory for crawled markdown and extraction audit.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=550,
        help="Max tokens per chunk.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OpenAI model. Defaults to OPENAI_MODEL or MedicalExtractor default.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before crawling.",
    )
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Skip crawling and reuse existing markdown in output_dir/markdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Put it in .env or export it before running."
        )

    output_dir = PROJECT_ROOT / args.output_dir
    markdown_dir = output_dir / "markdown"
    extraction_jsonl = output_dir / "strict_extraction_records.jsonl"
    summary_json = output_dir / "strict_extraction_summary.json"

    if args.clean and output_dir.exists():
        print(f"[CLEAN] Remove old output dir: {output_dir}")
        shutil.rmtree(output_dir)

    if not args.skip_crawl:
        run_crawler(
            urls=args.urls,
            output_dir=markdown_dir,
            clean_output=True,
        )
    else:
        print(f"[SKIP CRAWL] Reuse markdown dir: {markdown_dir}")

    articles = load_markdown_articles(markdown_dir)
    print(f"\n[LOAD] articles={len(articles)} from {markdown_dir}")

    if not articles:
        raise RuntimeError(f"No markdown articles found in {markdown_dir}")

    chunks = flatten_chunks(articles, max_tokens=args.max_tokens)
    print(f"[CHUNK] total_chunks={len(chunks)}")

    if extraction_jsonl.exists():
        extraction_jsonl.unlink()

    extractor = MedicalExtractor(
        model=args.model,
        prompt_mode="strict_text",
    )
    service = ExtractionService(extractor=extractor)

    records: list[dict[str, Any]] = []

    print(f"\n[EXTRACT] chunks={len(chunks)}")
    for idx, chunk in enumerate(chunks, start=1):
        print(
            f"[EXTRACT] {idx}/{len(chunks)} | "
            f"{chunk.title} | section={chunk.section} | subsection={chunk.subsection}"
        )

        graph = service.extract_and_validate_chunk(chunk)

        record = {
            "chunk_id": chunk.chunk_id,
            "article_id": chunk.article_id,
            "title": chunk.title,
            "source_url": chunk.source_url,
            "section": chunk.section,
            "subsection": chunk.subsection,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "graph": graph.model_dump(),
        }

        records.append(record)
        append_jsonl(extraction_jsonl, record)

    summary = audit_records(records)
    save_json(summary_json, summary)

    print_summary(summary)

    print("\n[DONE]")
    print(f"- extraction_jsonl = {extraction_jsonl}")
    print(f"- summary_json      = {summary_json}")


if __name__ == "__main__":
    main()
