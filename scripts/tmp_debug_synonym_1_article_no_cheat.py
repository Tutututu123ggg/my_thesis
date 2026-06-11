import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.ingestion.extraction import MedicalExtractor
from app.ingestion.extraction.extraction_validator import ExtractionValidator
from app.ingestion.processing import chunk_article, load_markdown_articles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://tamanhhospital.vn/benh/benh-cham/"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "tmp" / "debug_extract_1_article"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_crawler(url: str, output_dir: Path, clean: bool) -> None:
    markdown_dir = output_dir / "markdown"

    if clean and markdown_dir.exists():
        shutil.rmtree(markdown_dir)

    markdown_dir.mkdir(parents=True, exist_ok=True)

    spider_path = PROJECT_ROOT / "app" / "crawler" / "tamanh_spider.py"

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
        f"output_dir={markdown_dir}",
    ]

    print(f"[CRAWL] {url}", flush=True)

    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(f"Crawler failed for url: {url}")


def entity_to_dict(entity: Any) -> dict[str, Any]:
    return entity.model_dump() if hasattr(entity, "model_dump") else dict(entity)


def relation_to_dict(relation: Any) -> dict[str, Any]:
    return relation.model_dump() if hasattr(relation, "model_dump") else dict(relation)


def entity_key(entity: Any) -> str:
    return f"{entity.name} ({entity.entity_type})"


def relation_key(relation: Any) -> str:
    return (
        f"{relation.subject} ({relation.subject_type}) "
        f"-[{relation.relation_type}]-> "
        f"{relation.object} ({relation.object_type})"
    )


def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    return getattr(obj, attr, default)


def print_entities(title: str, entities: list[Any]) -> None:
    print(f"\n{title} ({len(entities)})")

    if not entities:
        print("  <none>")
        return

    for idx, entity in enumerate(entities, start=1):
        print(f"  {idx:02d}. {entity_key(entity)}")

        normalized_name = safe_get(entity, "normalized_name")
        aliases = safe_get(entity, "aliases", [])
        surface_forms = safe_get(entity, "surface_forms", [])
        description = safe_get(entity, "description", "")

        if normalized_name:
            print(f"      normalized_name: {normalized_name}")

        print(f"      aliases: {aliases}")
        print(f"      surface_forms: {surface_forms}")

        if description:
            print(f"      description: {description}")


def print_relations(title: str, relations: list[Any]) -> None:
    print(f"\n{title} ({len(relations)})")

    if not relations:
        print("  <none>")
        return

    for idx, relation in enumerate(relations, start=1):
        print(f"  {idx:02d}. {relation_key(relation)}")

        evidence_text = safe_get(relation, "evidence_text", "")
        keywords = safe_get(relation, "keywords", [])
        confidence = safe_get(relation, "confidence", None)

        if evidence_text:
            print(f"      evidence: {evidence_text}")

        print(f"      keywords: {keywords}")

        if confidence is not None:
            print(f"      confidence: {confidence}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Debug medical extraction on one article. "
            "Logs every extracted entity and relation before/after validation. "
            "Does not write Neo4j/Qdrant."
        )
    )

    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--model", default=None)

    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    output_dir = Path(args.output_dir)
    markdown_dir = output_dir / "markdown"

    print("========== DEBUG EXTRACT 1 ARTICLE ==========", flush=True)
    print(f"url={args.url}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"max_tokens={args.max_tokens}", flush=True)
    print(f"max_chunks={args.max_chunks}", flush=True)

    if not args.skip_crawl:
        run_crawler(
            url=args.url,
            output_dir=output_dir,
            clean=args.clean,
        )

    articles = load_markdown_articles(markdown_dir)

    if not articles:
        raise RuntimeError(f"No markdown articles found at: {markdown_dir}")

    article = articles[0]

    chunks = chunk_article(
        article,
        max_tokens=args.max_tokens,
    )

    if args.max_chunks and args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]

    print("\n========== LOADED ==========")
    print(f"article={article.title}")
    print(f"url={article.url}")
    print(f"chunks_total={len(chunks)}")
    print(f"chunks_extracted={len(chunks)}")

    extractor = MedicalExtractor(
        model=args.model,
        prompt_mode="strict_text",
    )

    validator = ExtractionValidator()

    records: list[dict[str, Any]] = []

    raw_entity_type_counter: Counter[str] = Counter()
    validated_entity_type_counter: Counter[str] = Counter()

    raw_relation_type_counter: Counter[str] = Counter()
    validated_relation_type_counter: Counter[str] = Counter()

    raw_entity_total = 0
    validated_entity_total = 0
    raw_relation_total = 0
    validated_relation_total = 0

    print("\n========== EXTRACT ==========")

    for idx, chunk in enumerate(chunks, start=1):
        print(
            f"\n[EXTRACT] {idx}/{len(chunks)} | "
            f"chunk_id={chunk.chunk_id} | "
            f"tokens={chunk.token_count}",
            flush=True,
        )

        print(f"          title={chunk.title}")
        print(f"          section={chunk.section}")
        print(f"          subsection={chunk.subsection}")

        raw_graph = extractor.extract_chunk(chunk)
        validated_graph = validator.validate(raw_graph)

        raw_entity_total += len(raw_graph.entities)
        validated_entity_total += len(validated_graph.entities)
        raw_relation_total += len(raw_graph.relations)
        validated_relation_total += len(validated_graph.relations)

        for entity in raw_graph.entities:
            raw_entity_type_counter[entity.entity_type] += 1

        for entity in validated_graph.entities:
            validated_entity_type_counter[entity.entity_type] += 1

        for relation in raw_graph.relations:
            raw_relation_type_counter[relation.relation_type] += 1

        for relation in validated_graph.relations:
            validated_relation_type_counter[relation.relation_type] += 1

        print(
            f"          raw_entities={len(raw_graph.entities)} "
            f"raw_relations={len(raw_graph.relations)}"
        )

        print(
            f"          val_entities={len(validated_graph.entities)} "
            f"val_relations={len(validated_graph.relations)}"
        )

        print_entities("RAW ENTITIES", raw_graph.entities)
        print_relations("RAW RELATIONS", raw_graph.relations)

        print_entities("VALIDATED ENTITIES", validated_graph.entities)
        print_relations("VALIDATED RELATIONS", validated_graph.relations)

        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "token_count": chunk.token_count,
                "text": chunk.contextualized_text,
                "raw": {
                    "entity_count": len(raw_graph.entities),
                    "relation_count": len(raw_graph.relations),
                    "entities": [entity_to_dict(e) for e in raw_graph.entities],
                    "relations": [relation_to_dict(r) for r in raw_graph.relations],
                },
                "validated": {
                    "entity_count": len(validated_graph.entities),
                    "relation_count": len(validated_graph.relations),
                    "entities": [entity_to_dict(e) for e in validated_graph.entities],
                    "relations": [relation_to_dict(r) for r in validated_graph.relations],
                },
            }
        )

    summary = {
        "article": article.title,
        "url": article.url,
        "chunks_total": len(chunks),
        "chunks_extracted": len(chunks),
        "raw_entity_total": raw_entity_total,
        "validated_entity_total": validated_entity_total,
        "raw_relation_total": raw_relation_total,
        "validated_relation_total": validated_relation_total,
        "raw_entity_type_counter": dict(raw_entity_type_counter),
        "validated_entity_type_counter": dict(validated_entity_type_counter),
        "raw_relation_type_counter": dict(raw_relation_type_counter),
        "validated_relation_type_counter": dict(validated_relation_type_counter),
    }

    save_json(output_dir / "debug_extract_records.json", records)
    save_json(output_dir / "debug_extract_summary.json", summary)

    print("\n========== SUMMARY ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n[OUTPUT] {output_dir / 'debug_extract_summary.json'}")
    print(f"[OUTPUT] {output_dir / 'debug_extract_records.json'}")


if __name__ == "__main__":
    main()