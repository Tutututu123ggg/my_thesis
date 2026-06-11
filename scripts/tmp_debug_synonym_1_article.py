import argparse
import json
import os
import re
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
from app.ingestion.processing.ingestion_models import ChunkDocument


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://tamanhhospital.vn/benh/benh-cham/"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "tmp" / "debug_synonym_1_article"

SYNONYM_CUE_RE = re.compile(
    r"(còn\s+(được\s+)?gọi\s+là|hay\s+gọi\s+là|còn\s+có\s+tên\s+là|"
    r"tên\s+khác\s+là|tiếng\s+Anh\s+là|viết\s+tắt\s+là|aka|also\s+known\s+as|"
    r"\b[A-ZÀ-Ỵa-zà-ỵ0-9][^\n]{1,80}\([^\n()]{2,80}\))",
    flags=re.IGNORECASE,
)

WATCH_TERMS = [
    "bệnh chàm",
    "eczema",
    "viêm da dị ứng",
    "viêm da cơ địa",
    "chàm thể tạng",
    "atopic dermatitis",
]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Crawler failed for url: {url}")


def has_synonym_cue(text: str) -> bool:
    return bool(SYNONYM_CUE_RE.search(text))


def extract_cue_snippets(text: str, window: int = 180) -> list[str]:
    snippets: list[str] = []
    for match in SYNONYM_CUE_RE.finditer(text):
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        snippets.append(snippet)
    return snippets[:5]


def contains_watch_term(text: str) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in WATCH_TERMS)


def relation_key(relation: Any) -> str:
    return (
        f"{relation.subject} ({relation.subject_type}) "
        f"-[{relation.relation_type}]-> "
        f"{relation.object} ({relation.object_type})"
    )


def entity_to_dict(entity: Any) -> dict[str, Any]:
    return entity.model_dump() if hasattr(entity, "model_dump") else dict(entity)


def relation_to_dict(relation: Any) -> dict[str, Any]:
    return relation.model_dump() if hasattr(relation, "model_dump") else dict(relation)


def print_relations(title: str, relations: list[Any]) -> None:
    print(f"\n{title} ({len(relations)})")
    for idx, relation in enumerate(relations, start=1):
        print(f"  {idx:02d}. {relation_key(relation)}")
        print(f"      evidence: {relation.evidence_text}")
        print(f"      keywords: {relation.keywords}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug DONG_NGHIA_VOI extraction on one article without writing Neo4j/Qdrant.",
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--cue-only", action="store_true", help="Only extract chunks containing synonym cues/watch terms.")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    output_dir = Path(args.output_dir)
    markdown_dir = output_dir / "markdown"

    print("========== DEBUG SYNONYM 1 ARTICLE ==========", flush=True)
    print(f"url={args.url}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"max_tokens={args.max_tokens}", flush=True)

    if not args.skip_crawl:
        run_crawler(url=args.url, output_dir=output_dir, clean=args.clean)

    articles = load_markdown_articles(markdown_dir)
    if not articles:
        raise RuntimeError(f"No markdown articles found at: {markdown_dir}")

    article = articles[0]
    chunks = chunk_article(article, max_tokens=args.max_tokens)
    if args.max_chunks and args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]

    cue_chunks: list[ChunkDocument] = []
    for chunk in chunks:
        if has_synonym_cue(chunk.contextualized_text) or contains_watch_term(chunk.contextualized_text):
            cue_chunks.append(chunk)

    target_chunks = cue_chunks if args.cue_only else chunks

    print("\n========== LOADED ==========")
    print(f"article={article.title}")
    print(f"chunks_total={len(chunks)} cue_or_watch_chunks={len(cue_chunks)} extracting={len(target_chunks)}")

    print("\n========== CUE/WATCH CHUNKS ==========")
    for idx, chunk in enumerate(cue_chunks, start=1):
        print(f"\n[CUE] {idx}. chunk_id={chunk.chunk_id}")
        print(f"      section={chunk.section} | {chunk.subsection}")
        snippets = extract_cue_snippets(chunk.contextualized_text)
        if snippets:
            for sidx, snippet in enumerate(snippets, start=1):
                print(f"      cue_snippet_{sidx}: {snippet}")
        else:
            print("      no regex cue, but watch term appears")

    extractor = MedicalExtractor(model=args.model, prompt_mode="strict_text")
    validator = ExtractionValidator()

    records: list[dict[str, Any]] = []
    raw_relation_type_counter: Counter[str] = Counter()
    validated_relation_type_counter: Counter[str] = Counter()
    raw_synonyms: list[dict[str, Any]] = []
    validated_synonyms: list[dict[str, Any]] = []

    print("\n========== EXTRACT ==========")
    for idx, chunk in enumerate(target_chunks, start=1):
        print(f"\n[EXTRACT] {idx}/{len(target_chunks)} | {chunk.section} | {chunk.subsection}", flush=True)

        raw_graph = extractor.extract_chunk(chunk)
        validated_graph = validator.validate(raw_graph)

        for relation in raw_graph.relations:
            raw_relation_type_counter[relation.relation_type] += 1
            if relation.relation_type == "DONG_NGHIA_VOI":
                raw_synonyms.append(relation_to_dict(relation) | {"chunk_id": chunk.chunk_id})

        for relation in validated_graph.relations:
            validated_relation_type_counter[relation.relation_type] += 1
            if relation.relation_type == "DONG_NGHIA_VOI":
                validated_synonyms.append(relation_to_dict(relation) | {"chunk_id": chunk.chunk_id})

        raw_synonym_relations = [r for r in raw_graph.relations if r.relation_type == "DONG_NGHIA_VOI"]
        validated_synonym_relations = [r for r in validated_graph.relations if r.relation_type == "DONG_NGHIA_VOI"]

        print(f"          raw_entities={len(raw_graph.entities)} raw_relations={len(raw_graph.relations)} raw_synonyms={len(raw_synonym_relations)}")
        print(f"          val_entities={len(validated_graph.entities)} val_relations={len(validated_graph.relations)} val_synonyms={len(validated_synonym_relations)}")

        if has_synonym_cue(chunk.contextualized_text):
            print("          [TEXT HAS SYNONYM CUE]")

        if raw_synonym_relations or validated_synonym_relations:
            print_relations("RAW DONG_NGHIA_VOI", raw_synonym_relations)
            print_relations("VALIDATED DONG_NGHIA_VOI", validated_synonym_relations)
        else:
            interesting_relations = [
                r for r in raw_graph.relations
                if contains_watch_term(f"{r.subject} {r.object} {r.evidence_text}")
            ]
            if interesting_relations:
                print_relations("RAW WATCH-TERM RELATIONS", interesting_relations[:8])

        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "token_count": chunk.token_count,
                "has_synonym_cue": has_synonym_cue(chunk.contextualized_text),
                "cue_snippets": extract_cue_snippets(chunk.contextualized_text),
                "raw": {
                    "entities": [entity_to_dict(e) for e in raw_graph.entities],
                    "relations": [relation_to_dict(r) for r in raw_graph.relations],
                },
                "validated": {
                    "entities": [entity_to_dict(e) for e in validated_graph.entities],
                    "relations": [relation_to_dict(r) for r in validated_graph.relations],
                },
            }
        )

    summary = {
        "article": article.title,
        "url": article.url,
        "chunks_total": len(chunks),
        "cue_or_watch_chunks": len(cue_chunks),
        "chunks_extracted": len(target_chunks),
        "raw_relation_type_counter": dict(raw_relation_type_counter),
        "validated_relation_type_counter": dict(validated_relation_type_counter),
        "raw_synonym_count": len(raw_synonyms),
        "validated_synonym_count": len(validated_synonyms),
        "raw_synonyms": raw_synonyms,
        "validated_synonyms": validated_synonyms,
    }

    save_json(output_dir / "debug_synonym_records.json", records)
    save_json(output_dir / "debug_synonym_summary.json", summary)

    print("\n========== SUMMARY ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if len(raw_synonyms) == 0:
        print("\n[DIAGNOSIS] LLM did not create DONG_NGHIA_VOI in raw output. Fix extraction_prompt / raw schema examples.")
    elif len(validated_synonyms) == 0:
        print("\n[DIAGNOSIS] LLM created DONG_NGHIA_VOI, but validator dropped it. Fix extraction_validator rules.")
    else:
        print("\n[PASS] DONG_NGHIA_VOI exists after validation.")

    print(f"\n[OUTPUT] {output_dir / 'debug_synonym_summary.json'}")
    print(f"[OUTPUT] {output_dir / 'debug_synonym_records.json'}")


if __name__ == "__main__":
    main()
