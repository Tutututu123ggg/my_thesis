"""
Overnight hybrid ingestion for Tam Anh articles.

This script is designed for the current repository snapshot:
- crawls Tam Anh disease articles
- chunks with section-preserving chunker
- writes Article/Chunk to Neo4j
- writes chunk/entity/relation vectors to Qdrant
- extracts entities/relations with bounded concurrency
- writes extraction output to Neo4j + Qdrant
- checkpoints after every article so the run can resume after crashes

Run 1-article smoke test:
    python -m scripts.overnight_hybrid_ingest_tamanh_v45 --fresh-start --limit 1 --target-articles 60 --max-tokens 900 --extract-concurrency 5

Run overnight:
    python -m scripts.overnight_hybrid_ingest_tamanh_v45 --fresh-start --target-articles 60 --max-tokens 900 --extract-concurrency 8

Resume after crash:
    python -m scripts.overnight_hybrid_ingest_tamanh_v45 --resume --target-articles 60 --max-tokens 900 --extract-concurrency 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv
from scrapy.http import HtmlResponse

from app.crawler.article_item import ArticleItem
from app.crawler.markdown_converter import save_article_markdown
from app.crawler.strategies.tamanh_strategy import TamAnhStrategy
from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.graph_database import GraphRepository, Neo4jClient
from app.infrastructure.vector_database import (
    CHUNK_COLLECTION,
    ENTITY_COLLECTION,
    RELATION_COLLECTION,
    QdrantClientWrapper,
    VectorRepository,
)
from app.ingestion.builders import GraphBuilder, MedicalGraphBuilder, VectorGraphBuilder
from app.ingestion.extraction import ExtractionService, ExtractionValidator, MedicalExtractor
from app.ingestion.processing import chunk_article, load_markdown_article
from app.ingestion.processing.ingestion_models import ArticleDocument, ChunkDocument
from app.ingestion.resolution import EntityResolver
from app.retrieval.vector_retriever import VectorRetriever


DEFAULT_TARGET_ARTICLES = 60
DEFAULT_OUTPUT_DIR = Path("data/tmp/overnight_hybrid_ingest_tamanh_v45")

DISCOVERY_URLS = [
    "https://tamanhhospital.vn/benh-hoc-a-z/",
    "https://tamanhhospital.vn/benh/",
    *[f"https://tamanhhospital.vn/benh/page/{i}/" for i in range(2, 25)],
]

# Curated URLs are only seeds. Missing/changed URLs are skipped safely.
# The later discovery step adds more URLs from Tâm Anh index/listing pages.
CURATED_RELATED_URLS = [
    # Dermatology / allergy / immune cluster
    "https://tamanhhospital.vn/benh/viem-da-co-dia/",
    "https://tamanhhospital.vn/benh/benh-cham/",
    "https://tamanhhospital.vn/benh/cham-dong-tien/",
    "https://tamanhhospital.vn/benh/benh-vay-nen/",
    "https://tamanhhospital.vn/benh/viem-da-tiep-xuc/",
    "https://tamanhhospital.vn/benh/mun-trung-ca/",
    "https://tamanhhospital.vn/benh/nam-da/",
    "https://tamanhhospital.vn/benh/zona-than-kinh/",
    "https://tamanhhospital.vn/benh/di-ung-thoi-tiet/",
    "https://tamanhhospital.vn/benh/di-ung-thuc-pham/",
    "https://tamanhhospital.vn/benh/viem-mui-di-ung/",
    "https://tamanhhospital.vn/benh/hen-suyen/",
    # Respiratory / ENT cluster
    "https://tamanhhospital.vn/benh/viem-xoang/",
    "https://tamanhhospital.vn/benh/viem-hong/",
    "https://tamanhhospital.vn/benh/viem-phe-quan/",
    "https://tamanhhospital.vn/benh/viem-phoi/",
    "https://tamanhhospital.vn/benh/copd/",
    "https://tamanhhospital.vn/benh/lao-phoi/",
    # Metabolic / cardiovascular cluster
    "https://tamanhhospital.vn/benh/dai-thao-duong/",
    "https://tamanhhospital.vn/benh/tang-huyet-ap/",
    "https://tamanhhospital.vn/benh/roi-loan-mo-mau/",
    "https://tamanhhospital.vn/benh/beo-phi/",
    "https://tamanhhospital.vn/benh/benh-mach-vanh/",
    "https://tamanhhospital.vn/benh/suy-tim/",
    # Digestive / liver cluster
    "https://tamanhhospital.vn/benh/trao-nguoc-da-day-thuc-quan/",
    "https://tamanhhospital.vn/benh/viem-da-day/",
    "https://tamanhhospital.vn/benh/viem-gan-b/",
    "https://tamanhhospital.vn/benh/gan-nhiem-mo/",
]

PRIORITY_KEYWORDS = [
    # strong priority: dermatology/allergy/immune
    "viem-da", "da-lieu", "cham", "eczema", "vay-nen", "mun", "nam-da",
    "di-ung", "hen", "suyen", "viem-mui", "zona", "noi-me-day", "ngua",
    # related systems
    "viem-xoang", "viem-hong", "viem-phe-quang", "viem-phe-quan", "viem-phoi",
    "copd", "lao-phoi",
    # metabolic/cardiovascular/digestive related comorbidities
    "dai-thao-duong", "tieu-duong", "tang-huyet-ap", "roi-loan-mo-mau",
    "beo-phi", "mach-vanh", "suy-tim", "gan-nhiem-mo", "viem-gan", "trao-nguoc",
]

SYNONYM_RELATION = "DONG_NGHIA_VOI"


@dataclass
class ArticleRunResult:
    url: str
    status: str
    article_id: str | None = None
    title: str | None = None
    markdown_path: str | None = None
    chunks: int = 0
    raw_entities: int = 0
    raw_relations: int = 0
    synonym_relations: int = 0
    missing_keyword_relations: int = 0
    graph_entity_writes: int = 0
    graph_relation_writes: int = 0
    mention_writes: int = 0
    chunk_vectors: int = 0
    entity_vectors: int = 0
    relation_vectors: int = 0
    failed_chunks: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stable_short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(query="", fragment="").geturl().rstrip("/")


def is_probably_article_url(url: str, strategy: TamAnhStrategy) -> bool:
    try:
        return strategy.is_article_url(normalize_url(url))
    except Exception:
        return False


def score_url(url: str) -> tuple[int, str]:
    slug = urlparse(url).path.lower()
    score = 0
    for idx, keyword in enumerate(PRIORITY_KEYWORDS):
        if keyword in slug:
            score += max(1, 80 - idx)
    return (-score, slug)


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        url = normalize_url(item)
        if url in seen:
            continue
        seen.add(url)
        result.append(url)
    return result


def fetch_html(url: str, timeout: int = 30) -> HtmlResponse:
    headers = {
        "User-Agent": (
            "medical-vietnamese-hybrid-rag/0.1 "
            "(academic thesis crawler; contact: ductu3003@gmail.com)"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    body = response.content
    return HtmlResponse(url=response.url, body=body, encoding=response.encoding or "utf-8")


def discover_article_urls(
    target_articles: int,
    max_listing_pages: int,
    output_dir: Path,
    refresh: bool = False,
) -> list[str]:
    selected_path = output_dir / "selected_urls.json"
    if selected_path.exists() and not refresh:
        data = json.loads(selected_path.read_text(encoding="utf-8"))
        urls = data.get("selected_urls") or data if isinstance(data, list) else []
        return unique_keep_order([str(u) for u in urls])[:target_articles]

    strategy = TamAnhStrategy()
    discovered: list[str] = []

    print("========== DISCOVER URLS ==========", flush=True)

    for url in CURATED_RELATED_URLS:
        if is_probably_article_url(url, strategy):
            discovered.append(normalize_url(url))

    for idx, listing_url in enumerate(DISCOVERY_URLS[:max_listing_pages], start=1):
        try:
            print(f"[DISCOVER] {idx}/{min(max_listing_pages, len(DISCOVERY_URLS))} {listing_url}", flush=True)
            response = fetch_html(listing_url, timeout=30)
            urls = strategy.extract_article_urls(response)
            discovered.extend(urls)
            time.sleep(0.25)
        except Exception as exc:
            print(f"[WARN] discover failed: {listing_url} | {exc}", flush=True)

    discovered = unique_keep_order(discovered)
    selected = sorted(discovered, key=score_url)[:target_articles]

    write_json(
        selected_path,
        {
            "created_at": now_iso(),
            "target_articles": target_articles,
            "discovered_count": len(discovered),
            "selected_count": len(selected),
            "selected_urls": selected,
        },
    )

    print(f"[DISCOVER] discovered={len(discovered)} selected={len(selected)}", flush=True)
    return selected


def crawl_article_to_markdown(
    url: str,
    markdown_dir: Path,
    min_words: int = 300,
) -> Path:
    strategy = TamAnhStrategy()
    response = fetch_html(url, timeout=45)
    current_url = strategy.normalize_url(response.url)

    if not strategy.is_article_url(current_url):
        raise RuntimeError(f"Not a supported Tam Anh article URL: {current_url}")

    title = strategy.extract_title(response)
    if not title:
        raise RuntimeError(f"Missing article title: {current_url}")

    markdown = strategy.extract_markdown(response, title=title)
    word_count = len(re.findall(r"\S+", markdown))
    if word_count < min_words:
        raise RuntimeError(f"Article too short after cleanup: words={word_count} url={current_url}")

    item = ArticleItem(
        source=strategy.source,
        url=current_url,
        title=title,
        description=strategy.extract_description(response),
        author=strategy.extract_author(response),
        published_at=strategy.extract_published_at(response),
        updated_at=strategy.extract_updated_at(response),
        category="benh",
        related_links=strategy.extract_related_links(response),
        markdown=markdown,
    )
    return save_article_markdown(item, markdown_dir)


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "completed_urls": [],
            "failed_urls": {},
            "article_results": [],
            "totals": {},
        }
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json(state_path, state)


def load_urls_file(path: Path) -> list[str]:
    """Load selected URLs from .json or .txt.

    JSON accepted formats:
    - {"selected_urls": [...]}
    - {"urls": [...]}
    - ["url1", "url2"]

    TXT format: one URL per line.
    """
    if not path.exists():
        raise FileNotFoundError(f"URLs file not found: {path}")

    if path.suffix.lower() == ".txt":
        urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return unique_keep_order([url for url in urls if url and not url.startswith("#")])

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return unique_keep_order([str(x) for x in data])
    if isinstance(data, dict):
        if "selected_urls" in data:
            return unique_keep_order([str(x) for x in data["selected_urls"]])
        if "urls" in data:
            return unique_keep_order([str(x) for x in data["urls"]])

    raise ValueError(f"Unsupported URLs file format: {path}")


def build_graph_connections() -> tuple[Neo4jClient, GraphRepository]:
    client = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "admin123456"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    repo = GraphRepository(client)
    return client, repo


def build_vector_connections(embedder: EmbeddingService) -> tuple[QdrantClientWrapper, VectorRepository]:
    client = QdrantClientWrapper(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    repo = VectorRepository(
        client=client,
        vector_dim=embedder.vector_dim,
        distance=os.getenv("VECTOR_DISTANCE", "cosine"),
    )
    return client, repo


def create_extraction_service(model: str | None = None) -> ExtractionService:
    return ExtractionService(
        extractor=MedicalExtractor(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        validator=ExtractionValidator(),
    )


def extract_one_chunk_with_retry(
    chunk: ChunkDocument,
    model: str | None,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_error: str | None = None

    for attempt in range(1, retries + 2):
        try:
            service = create_extraction_service(model=model)
            graph = service.extract_and_validate_chunk(chunk)
            return {
                "ok": True,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "article_id": chunk.article_id,
                "title": chunk.title,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "source_url": chunk.source_url,
                "entities": [e.model_dump() for e in graph.entities],
                "relations": [r.model_dump() for r in graph.relations],
                "attempt": attempt,
                "error": None,
            }
        except Exception as exc:
            last_error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            if attempt <= retries:
                sleep_s = retry_sleep * (2 ** (attempt - 1))
                time.sleep(sleep_s)

    return {
        "ok": False,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "article_id": chunk.article_id,
        "title": chunk.title,
        "section": chunk.section,
        "subsection": chunk.subsection,
        "source_url": chunk.source_url,
        "entities": [],
        "relations": [],
        "attempt": retries + 1,
        "error": last_error or "unknown extraction error",
    }


def extract_chunks_parallel(
    chunks: list[ChunkDocument],
    model: str | None,
    concurrency: int,
    retries: int,
    retry_sleep: float,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    results_by_chunk_id: dict[str, dict[str, Any]] = {}

    max_workers = max(1, int(concurrency))
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_chunk = {
            pool.submit(
                extract_one_chunk_with_retry,
                chunk,
                model,
                retries,
                retry_sleep,
            ): chunk
            for chunk in chunks
        }

        done_count = 0
        for fut in futures.as_completed(future_to_chunk):
            chunk = future_to_chunk[fut]
            done_count += 1

            try:
                record = fut.result()
            except Exception as exc:
                record = {
                    "ok": False,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "article_id": chunk.article_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "source_url": chunk.source_url,
                    "entities": [],
                    "relations": [],
                    "attempt": retries + 1,
                    "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                }

            status = "OK" if record.get("ok") else "FAIL"
            print(
                f"          [EXTRACT {status}] {done_count}/{len(chunks)} "
                f"chunk={chunk.chunk_index} entities={len(record.get('entities') or [])} "
                f"relations={len(record.get('relations') or [])}",
                flush=True,
            )
            results_by_chunk_id[chunk.chunk_id] = record

    return [
        results_by_chunk_id[chunk.chunk_id]
        for chunk in chunks
        if chunk.chunk_id in results_by_chunk_id
    ]


def graph_from_record(record: dict[str, Any]):
    from app.ingestion.extraction import ExtractedEntity, ExtractedGraph, ExtractedRelation

    return ExtractedGraph(
        entities=[ExtractedEntity(**e) for e in record.get("entities", [])],
        relations=[ExtractedRelation(**r) for r in record.get("relations", [])],
    )


def count_qdrant_points(vector_client: QdrantClientWrapper, collection_name: str) -> int:
    try:
        if not vector_client.client.collection_exists(collection_name):
            return 0
        return int(vector_client.client.count(collection_name=collection_name, exact=True).count)
    except Exception:
        return -1


def run_vector_smoke(embedder: EmbeddingService, vector_repo: VectorRepository) -> dict[str, Any]:
    retriever = VectorRetriever(vector_repo=vector_repo, embedding_service=embedder)
    query = "viêm da cơ địa có triệu chứng gì"

    chunks = retriever.search_chunks(query, top_k=5)
    entities = retriever.search_entities(query, top_k=5)
    relations = retriever.search_relations(query, top_k=5)

    print("\n========== RETRIEVAL SMOKE ==========", flush=True)
    print(f"query={query}", flush=True)
    print(f"[SMOKE] chunks={len(chunks)} entities={len(entities)} relations={len(relations)}", flush=True)

    for idx, chunk in enumerate(chunks[:3], start=1):
        print(f"  CHUNK {idx}. score={chunk.score:.4f} | {chunk.title} | {chunk.section}", flush=True)

    for idx, entity in enumerate(entities[:3], start=1):
        print(f"  ENTITY {idx}. score={entity.score:.4f} | {entity.name} ({entity.entity_type})", flush=True)

    for idx, rel in enumerate(relations[:3], start=1):
        print(
            f"  REL {idx}. score={rel.score:.4f} | "
            f"{rel.subject_name} -[{rel.relation_type}]-> {rel.object_name}",
            flush=True,
        )

    return {
        "query": query,
        "chunk_hits": len(chunks),
        "entity_hits": len(entities),
        "relation_hits": len(relations),
        "top_chunks": [
            {
                "score": c.score,
                "chunk_id": c.chunk_id,
                "title": c.title,
                "section": c.section,
                "source_url": c.source_url,
            }
            for c in chunks
        ],
        "top_entities": [
            {
                "score": e.score,
                "entity_id": e.entity_id,
                "name": e.name,
                "entity_type": e.entity_type,
            }
            for e in entities
        ],
        "top_relations": [
            {
                "score": r.score,
                "relation_id": r.relation_id,
                "relation_type": r.relation_type,
                "subject": r.subject_name,
                "object": r.object_name,
            }
            for r in relations
        ],
    }


def ingest_one_article(
    url: str,
    args: argparse.Namespace,
    graph_builder: GraphBuilder,
    medical_graph_builder: MedicalGraphBuilder,
    vector_builder: VectorGraphBuilder,
    markdown_dir: Path,
    records_jsonl: Path,
    article_records_dir: Path,
) -> ArticleRunResult:
    started = time.time()
    result = ArticleRunResult(url=url, status="started")

    try:
        print(f"\n========== ARTICLE ==========\n{url}", flush=True)

        markdown_path = crawl_article_to_markdown(url, markdown_dir=markdown_dir)
        article = load_markdown_article(markdown_path)
        chunks = chunk_article(article, max_tokens=args.max_tokens)

        if args.max_chunks_per_article > 0:
            chunks = chunks[: args.max_chunks_per_article]

        print(
            f"[LOAD] title={article.title} chunks={len(chunks)} markdown={markdown_path}",
            flush=True,
        )

        result.article_id = article.article_id
        result.title = article.title
        result.markdown_path = str(markdown_path)
        result.chunks = len(chunks)

        if not chunks:
            raise RuntimeError("No chunks generated.")

        print("[GRAPH] upsert article/chunks", flush=True)
        graph_builder.ingest_article_with_chunks(article, chunks)

        print("[QDRANT] upsert chunk vectors", flush=True)
        result.chunk_vectors = vector_builder.upsert_chunks(chunks)

        print(
            f"[EXTRACT] chunks={len(chunks)} concurrency={args.extract_concurrency}",
            flush=True,
        )
        extraction_records = extract_chunks_parallel(
            chunks=chunks,
            model=args.model,
            concurrency=args.extract_concurrency,
            retries=args.extract_retries,
            retry_sleep=args.retry_sleep,
        )

        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

        print("[WRITE] graph + entity/relation vectors", flush=True)
        for record in extraction_records:
            append_jsonl(records_jsonl, record)

            if not record.get("ok"):
                result.failed_chunks += 1
                continue

            graph = graph_from_record(record)
            chunk = chunk_by_id[record["chunk_id"]]

            result.raw_entities += len(graph.entities)
            result.raw_relations += len(graph.relations)
            result.synonym_relations += sum(
                1 for relation in graph.relations if relation.relation_type == SYNONYM_RELATION
            )
            result.missing_keyword_relations += sum(
                1 for relation in graph.relations if not relation.keywords
            )

            graph_stats = medical_graph_builder.write_chunk_extraction(chunk, graph)
            vector_stats = vector_builder.upsert_chunk_extraction(chunk, graph)

            result.graph_entity_writes += int(graph_stats.get("entity_write_count", 0))
            result.graph_relation_writes += int(graph_stats.get("relation_write_count", 0))
            result.mention_writes += int(graph_stats.get("mention_write_count", 0))
            result.entity_vectors += int(vector_stats.get("entity_vector_count", 0))
            result.relation_vectors += int(vector_stats.get("relation_vector_count", 0))

        result.status = "completed"
        result.elapsed_seconds = time.time() - started

        article_record = {
            **asdict(result),
            "completed_at": now_iso(),
            "extraction_record_count": len(extraction_records),
        }
        article_records_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            article_records_dir / f"{stable_short_hash(url)}.json",
            article_record,
        )

        print(
            f"[DONE] {article.title} | chunks={result.chunks} "
            f"entities={result.raw_entities} relations={result.raw_relations} "
            f"synonyms={result.synonym_relations} failed_chunks={result.failed_chunks} "
            f"elapsed={result.elapsed_seconds:.1f}s",
            flush=True,
        )
        return result

    except Exception as exc:
        result.status = "failed"
        result.elapsed_seconds = time.time() - started
        result.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()

        print(f"[ERROR] article failed: {url}\n{result.error}", flush=True)
        return result


def summarize_article_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") == "failed"]

    return {
        "articles_completed": len(completed),
        "articles_failed": len(failed),
        "chunks": sum(int(r.get("chunks") or 0) for r in completed),
        "raw_entities": sum(int(r.get("raw_entities") or 0) for r in completed),
        "raw_relations": sum(int(r.get("raw_relations") or 0) for r in completed),
        "synonym_relations": sum(int(r.get("synonym_relations") or 0) for r in completed),
        "missing_keyword_relations": sum(int(r.get("missing_keyword_relations") or 0) for r in completed),
        "chunk_vectors": sum(int(r.get("chunk_vectors") or 0) for r in completed),
        "entity_vectors": sum(int(r.get("entity_vectors") or 0) for r in completed),
        "relation_vectors": sum(int(r.get("relation_vectors") or 0) for r in completed),
        "failed_chunks": sum(int(r.get("failed_chunks") or 0) for r in completed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overnight hybrid graph+vector ingestion for Tam Anh articles.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fresh-start", action="store_true", help="Delete output_dir, reset Neo4j, recreate Qdrant, then run from scratch.")
    mode.add_argument("--resume", action="store_true", help="Resume from output_dir/state.json without deleting graph/vector data.")

    parser.add_argument("--target-articles", type=int, default=DEFAULT_TARGET_ARTICLES)
    parser.add_argument("--limit", type=int, default=0, help="Hard limit for this run only. Use --limit 1 for smoke test.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-listing-pages", type=int, default=len(DISCOVERY_URLS))
    parser.add_argument("--refresh-discovery", action="store_true")
    parser.add_argument("--dry-run-discover", action="store_true")
    parser.add_argument("--urls-file", type=Path, default=None, help="Use a prevalidated URL file instead of auto-discovery.")

    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--max-chunks-per-article", type=int, default=0)
    parser.add_argument("--extract-concurrency", type=int, default=5)
    parser.add_argument("--extract-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--model", type=str, default=None)

    parser.add_argument("--skip-vector-smoke", action="store_true")
    parser.add_argument("--sleep-between-articles", type=float, default=0.5)

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    output_dir: Path = args.output_dir
    markdown_dir = output_dir / "markdown"
    state_path = output_dir / "state.json"
    records_jsonl = output_dir / "extraction_records.jsonl"
    article_records_dir = output_dir / "article_records"
    last_error_path = output_dir / "last_error.json"

    if not args.fresh_start and not args.resume:
        print("[INFO] Neither --fresh-start nor --resume was provided. Using --resume-safe behavior.", flush=True)

    if args.fresh_start:
        if output_dir.exists():
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    article_records_dir.mkdir(parents=True, exist_ok=True)

    if args.urls_file:
        selected_urls = load_urls_file(args.urls_file)[: args.target_articles]
        write_json(
            output_dir / "selected_urls.json",
            {
                "created_at": now_iso(),
                "source": str(args.urls_file),
                "target_articles": args.target_articles,
                "selected_count": len(selected_urls),
                "selected_urls": selected_urls,
            },
        )
    else:
        selected_urls = discover_article_urls(
            target_articles=args.target_articles,
            max_listing_pages=args.max_listing_pages,
            output_dir=output_dir,
            refresh=args.refresh_discovery or args.fresh_start,
        )

    if args.dry_run_discover:
        print(json.dumps(selected_urls, ensure_ascii=False, indent=2))
        return

    if args.limit > 0:
        selected_urls = selected_urls[: args.limit]

    print("\n========== OVERNIGHT HYBRID INGEST ==========", flush=True)
    print(f"output_dir={output_dir.resolve()}", flush=True)
    print(f"target_articles={args.target_articles} run_limit={args.limit or 'none'} selected={len(selected_urls)}", flush=True)
    print(f"max_tokens={args.max_tokens} extract_concurrency={args.extract_concurrency}", flush=True)
    print(f"fresh_start={args.fresh_start} resume={args.resume}", flush=True)

    graph_client = None
    vector_client = None

    try:
        print("\n========== CONNECT ==========", flush=True)
        graph_client, graph_repo = build_graph_connections()
        if not graph_repo.health_check():
            raise RuntimeError("Neo4j health check failed. Is Neo4j running?")

        graph_repo.setup_schema()
        if args.fresh_start:
            print("[NEO4J] delete_all_data()", flush=True)
            graph_repo.delete_all_data()

        print("[PASS] Neo4j", flush=True)

        embedder = EmbeddingService()
        print(f"[EMBEDDING] model={embedder.model_name} dim={embedder.vector_dim}", flush=True)

        vector_client, vector_repo = build_vector_connections(embedder)
        if not vector_repo.health_check():
            raise RuntimeError("Qdrant health check failed. Is Qdrant running?")

        vector_builder = VectorGraphBuilder(
            vector_repo=vector_repo,
            embedder=embedder,
            resolver=EntityResolver(graph_repo),
            batch_size=32,
        )
        print(f"[QDRANT] setup_collections recreate={args.fresh_start}", flush=True)
        vector_builder.setup_collections(recreate=args.fresh_start)

        print("[PASS] Qdrant", flush=True)

        graph_builder = GraphBuilder(graph_repo)
        resolver = EntityResolver(graph_repo)
        medical_graph_builder = MedicalGraphBuilder(graph_repo, resolver)

        state = load_state(state_path)
        completed_urls = set(state.get("completed_urls") or [])
        article_results: list[dict[str, Any]] = list(state.get("article_results") or [])

        urls_to_run = [url for url in selected_urls if normalize_url(url) not in completed_urls]
        print(f"\n[STATE] completed={len(completed_urls)} pending={len(urls_to_run)}", flush=True)

        if not urls_to_run:
            print("[STATE] nothing to run.", flush=True)

        for idx, url in enumerate(urls_to_run, start=1):
            normalized = normalize_url(url)
            print(f"\n[RUN] {idx}/{len(urls_to_run)}", flush=True)

            result = ingest_one_article(
                url=normalized,
                args=args,
                graph_builder=graph_builder,
                medical_graph_builder=medical_graph_builder,
                vector_builder=vector_builder,
                markdown_dir=markdown_dir,
                records_jsonl=records_jsonl,
                article_records_dir=article_records_dir,
            )

            result_dict = asdict(result)
            article_results.append(result_dict)

            if result.status == "completed":
                completed_urls.add(normalized)
                state.setdefault("completed_urls", [])
                if normalized not in state["completed_urls"]:
                    state["completed_urls"].append(normalized)
            else:
                state.setdefault("failed_urls", {})
                state["failed_urls"][normalized] = {
                    "error": result.error,
                    "failed_at": now_iso(),
                }
                write_json(last_error_path, result_dict)

            state["article_results"] = article_results
            state["totals"] = summarize_article_results(article_results)
            save_state(state_path, state)

            if args.sleep_between_articles > 0:
                time.sleep(args.sleep_between_articles)

        graph_stats = graph_repo.get_graph_stats()
        qdrant_stats = {
            "chunk_vectors_total": count_qdrant_points(vector_client, CHUNK_COLLECTION),
            "entity_vectors_total": count_qdrant_points(vector_client, ENTITY_COLLECTION),
            "relation_vectors_total": count_qdrant_points(vector_client, RELATION_COLLECTION),
        }

        vector_smoke = None
        if not args.skip_vector_smoke:
            vector_smoke = run_vector_smoke(embedder, vector_repo)

        totals = summarize_article_results(article_results)
        summary = {
            "finished_at": now_iso(),
            "config": {
                "target_articles": args.target_articles,
                "run_limit": args.limit,
                "max_tokens": args.max_tokens,
                "extract_concurrency": args.extract_concurrency,
                "model": args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "embedding_model": embedder.model_name,
                "embedding_dim": embedder.vector_dim,
            },
            "totals": totals,
            "neo4j_stats_total": graph_stats,
            "qdrant_stats_total": qdrant_stats,
            "vector_smoke": vector_smoke,
            "completed_urls": sorted(completed_urls),
            "failed_urls": state.get("failed_urls", {}),
        }
        write_json(output_dir / "overnight_ingest_summary.json", summary)

        print("\n========== FINAL SUMMARY ==========", flush=True)
        print(json.dumps(summary["totals"], ensure_ascii=False, indent=2), flush=True)
        print("[NEO4J]", json.dumps(graph_stats, ensure_ascii=False), flush=True)
        print("[QDRANT]", json.dumps(qdrant_stats, ensure_ascii=False), flush=True)

        critical_fail = (
            totals["articles_completed"] <= 0
            or graph_stats.get("chunk_count", 0) <= 0
            or qdrant_stats.get("chunk_vectors_total", 0) <= 0
            or qdrant_stats.get("entity_vectors_total", 0) <= 0
            or qdrant_stats.get("relation_vectors_total", 0) <= 0
        )

        if critical_fail:
            print("\n[RESULT] FAIL: critical ingestion counters are zero. Check logs/state.", flush=True)
            sys.exit(2)

        if totals.get("missing_keyword_relations", 0) > 0:
            print("\n[WARN] Some relations still miss keywords.", flush=True)

        print("\n[RESULT] PASS: overnight hybrid graph+vector ingestion completed/resumed safely.", flush=True)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] State has been checkpointed after completed articles. Re-run with --resume.", flush=True)
        raise
    except Exception as exc:
        error = {
            "failed_at": now_iso(),
            "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        write_json(last_error_path, error)
        print("\n[FATAL] Overnight ingestion failed. See last_error.json.", flush=True)
        print(error["error"], flush=True)
        raise
    finally:
        if graph_client is not None:
            graph_client.close()
        if vector_client is not None:
            vector_client.close()


if __name__ == "__main__":
    main()
