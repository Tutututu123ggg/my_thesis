import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.vector_database import QdrantClientWrapper, VectorRepository
from app.ingestion.builders import VectorGraphBuilder
from app.ingestion.extraction import ExtractionService, MedicalExtractor
from app.ingestion.processing import chunk_article, load_markdown_articles
from app.retrieval import VectorRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_URLS = [
    "https://tamanhhospital.vn/benh/viem-da-co-dia/",
    "https://tamanhhospital.vn/benh/benh-cham/",
    "https://tamanhhospital.vn/benh/benh-vay-nen/",
]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_crawler(urls: list[str], output_dir: Path, clean: bool) -> None:
    if clean and output_dir.exists():
        print(f"[CLEAN] {output_dir}")
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
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Crawler failed for url: {url}")


def print_result_group(title: str, results) -> None:
    print(f"\n========== {title} ==========")
    for idx, item in enumerate(results, start=1):
        payload = item.payload
        label = payload.get("title") or payload.get("name") or payload.get("relation_type")
        print(f"{idx:02d}. score={item.score:.4f} | {label}")
        if payload.get("section"):
            print(f"    section={payload.get('section')}")
        if payload.get("subject_name") and payload.get("object_name"):
            print(f"    {payload.get('subject_name')} -[{payload.get('relation_type')}]-> {payload.get('object_name')}")
        preview = payload.get("text") or payload.get("evidence_text") or payload.get("description") or ""
        if preview:
            print(f"    {str(preview)[:220]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Clean markdown output before crawling.")
    parser.add_argument("--skip-crawl", action="store_true", help="Use existing markdown directory.")
    parser.add_argument("--skip-extract", action="store_true", help="Only index chunk vectors, skip entity/relation vectors.")
    parser.add_argument("--recreate-qdrant", action="store_true", help="Recreate Qdrant collections before indexing.")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunks for a cheap smoke run. 0 means all.")
    parser.add_argument("--query", default="chàm thể tạng có triệu chứng gì")
    parser.add_argument("--model", default=None, help="Optional OpenAI extraction model.")
    parser.add_argument("--urls", nargs="*", default=None)
    parser.add_argument(
        "--output-dir",
        default="data/tmp/qdrant_3_articles",
        help="Output directory for markdown and audit json.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    output_dir = PROJECT_ROOT / args.output_dir
    markdown_dir = output_dir / "markdown"
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = args.urls or DEFAULT_URLS

    if not args.skip_crawl:
        run_crawler(urls=urls, output_dir=markdown_dir, clean=args.clean)

    articles = load_markdown_articles(markdown_dir)
    if not articles:
        raise RuntimeError(f"No markdown articles found at: {markdown_dir}")

    chunks = []
    for article in articles:
        article_chunks = chunk_article(article, max_tokens=args.max_tokens)
        chunks.extend(article_chunks)

    if args.max_chunks and args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]

    print("\n========== LOADED ==========")
    print(f"articles={len(articles)} chunks={len(chunks)} max_tokens={args.max_tokens}")

    embedder = EmbeddingService()
    print(f"[EMBEDDING] model={embedder.model_name} dim={embedder.vector_dim}")

    qdrant_client = QdrantClientWrapper(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    vector_repo = VectorRepository(
        client=qdrant_client,
        vector_dim=embedder.vector_dim,
        distance=os.getenv("VECTOR_DISTANCE", "cosine"),
    )
    if not vector_repo.health_check():
        raise RuntimeError("Qdrant health check failed. Check Qdrant is running.")

    builder = VectorGraphBuilder(
        vector_repo=vector_repo,
        embedder=embedder,
        resolver=None,
        batch_size=16,
    )
    builder.setup_collections(recreate=args.recreate_qdrant)

    print("\n[QDRANT] upsert chunk vectors")
    chunk_vector_count = builder.upsert_chunks(chunks)

    extraction_records = []
    entity_vector_count = 0
    relation_vector_count = 0

    if not args.skip_extract:
        extractor = MedicalExtractor(model=args.model, prompt_mode="strict_text")
        extraction_service = ExtractionService(extractor=extractor)

        print("\n[EXTRACT + QDRANT] entity/relation vectors")
        for idx, chunk in enumerate(chunks, start=1):
            print(f"[EXTRACT] {idx}/{len(chunks)} | {chunk.title} | {chunk.section}")
            graph = extraction_service.extract_and_validate_chunk(chunk)
            stats = builder.upsert_chunk_extraction(chunk, graph)

            entity_vector_count += stats["entity_vector_count"]
            relation_vector_count += stats["relation_vector_count"]

            extraction_records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "article_id": chunk.article_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "entities": [entity.model_dump() for entity in graph.entities],
                    "relations": [relation.model_dump() for relation in graph.relations],
                    "vector_stats": stats,
                }
            )

        save_json(output_dir / "extraction_records.json", extraction_records)

    summary = {
        "articles": len(articles),
        "chunks": len(chunks),
        "chunk_vectors": chunk_vector_count,
        "entity_vectors": entity_vector_count,
        "relation_vectors": relation_vector_count,
        "embedding_model": embedder.model_name,
        "embedding_dim": embedder.vector_dim,
        "skip_extract": args.skip_extract,
    }
    save_json(output_dir / "qdrant_ingest_summary.json", summary)

    retriever = VectorRetriever(vector_repo=vector_repo, embedder=embedder)
    results = retriever.hybrid_search(args.query, chunk_limit=5, entity_limit=5, relation_limit=5)

    print("\n========== QDRANT INGEST SUMMARY ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print_result_group("VECTOR CHUNKS", results.chunks)
    print_result_group("VECTOR ENTITIES", results.entities)
    print_result_group("VECTOR RELATIONS", results.relations)

    qdrant_client.close()


if __name__ == "__main__":
    main()
