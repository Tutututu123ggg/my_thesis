import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.graph_database import GraphRepository, Neo4jClient
from app.infrastructure.vector_database import QdrantClientWrapper, VectorRepository
from app.infrastructure.vector_database.vector_collections import (
    CHUNK_COLLECTION,
    ENTITY_COLLECTION,
    RELATION_COLLECTION,
)
from app.ingestion.builders.graph_builder import GraphBuilder
from app.ingestion.builders.medical_graph_builder import MedicalGraphBuilder
from app.ingestion.builders.vector_graph_builder import VectorGraphBuilder
from app.ingestion.extraction import ExtractionService, MedicalExtractor
from app.ingestion.processing import chunk_article, load_markdown_articles
from app.ingestion.resolution import EntityResolver
from app.retrieval import VectorRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_URLS = [
    "https://tamanhhospital.vn/benh/viem-da-co-dia/",
    "https://tamanhhospital.vn/benh/benh-cham/",
    "https://tamanhhospital.vn/benh/benh-vay-nen/",
]

DEFAULT_ACCEPTANCE_QUERY = "chàm thể tạng có triệu chứng gì"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def require_env(keys: list[str]) -> None:
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing env variables: "
            + ", ".join(missing)
            + f"\nChecked .env at: {PROJECT_ROOT / '.env'}"
        )


def run_crawler(urls: list[str], output_dir: Path, clean: bool) -> None:
    if clean and output_dir.exists():
        print(f"[CLEAN] markdown_dir={output_dir}", flush=True)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    spider_path = PROJECT_ROOT / "app" / "crawler" / "tamanh_spider.py"

    for idx, url in enumerate(urls, start=1):
        print(f"\n[CRAWL] {idx}/{len(urls)} {url}", flush=True)
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


def make_graph_repo(reset_graph: bool) -> tuple[Neo4jClient, GraphRepository]:
    require_env(["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"])

    client = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    repo = GraphRepository(client)

    if not repo.health_check():
        client.close()
        raise RuntimeError("Neo4j health check failed. Check Neo4j is running and .env is correct.")

    repo.setup_schema()

    if reset_graph:
        print("[NEO4J] delete_all_data()", flush=True)
        repo.delete_all_data()

    return client, repo


def make_vector_repo(embedder: EmbeddingService, recreate_qdrant: bool) -> tuple[QdrantClientWrapper, VectorRepository]:
    client = QdrantClientWrapper(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    repo = VectorRepository(
        client=client,
        vector_dim=embedder.vector_dim,
        distance=os.getenv("VECTOR_DISTANCE", "cosine"),
    )

    if not repo.health_check():
        client.close()
        raise RuntimeError("Qdrant health check failed. Check Qdrant is running.")

    repo.setup_collections(recreate=recreate_qdrant)
    return client, repo


def qdrant_count(client: QdrantClientWrapper, collection_name: str) -> int:
    try:
        return int(client.client.count(collection_name=collection_name, exact=True).count)
    except Exception as exc:
        print(f"[WARN] Cannot count Qdrant collection {collection_name}: {exc}", flush=True)
        return -1


def print_vector_results(title: str, results, limit: int = 5) -> None:
    print(f"\n========== {title} ==========")
    for idx, item in enumerate(results[:limit], start=1):
        payload = item.payload
        label = payload.get("title") or payload.get("name") or payload.get("relation_type") or item.point_id
        print(f"{idx:02d}. score={item.score:.4f} | {label}")
        if payload.get("section"):
            print(f"    section={payload.get('section')}")
        if payload.get("subject_name") and payload.get("object_name"):
            print(
                f"    {payload.get('subject_name')} -[{payload.get('relation_type')}]-> {payload.get('object_name')}"
            )
        preview = payload.get("text") or payload.get("evidence_text") or payload.get("description") or ""
        if preview:
            print(f"    {str(preview)[:260]}")


def acceptance_status(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    tag = "PASS" if ok else "WARN"
    line = f"[{tag}] {name}"
    if detail:
        line += f" | {detail}"
    print(line, flush=True)
    return {"name": name, "ok": ok, "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end hybrid ingestion acceptance script: crawl/load markdown, chunk, "
            "write Article/Chunk/Entity/Relation to Neo4j, write Chunk/Entity/Relation vectors to Qdrant, "
            "then print acceptance checks."
        )
    )
    parser.add_argument("--clean", action="store_true", help="Clean markdown output before crawling.")
    parser.add_argument("--skip-crawl", action="store_true", help="Use existing markdown directory.")
    parser.add_argument("--reset-graph", action="store_true", help="DANGER: delete all Neo4j data before ingest.")
    parser.add_argument("--recreate-qdrant", action="store_true", help="Recreate Qdrant collections before ingest.")
    parser.add_argument("--skip-extract", action="store_true", help="Only ingest Article/Chunk + chunk vectors.")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunks for a cheap run. 0 means all.")
    parser.add_argument("--model", default=None, help="Optional OpenAI extraction model override.")
    parser.add_argument("--query", default=DEFAULT_ACCEPTANCE_QUERY)
    parser.add_argument("--urls", nargs="*", default=None)
    parser.add_argument(
        "--output-dir",
        default="data/tmp/hybrid_ingest_acceptance",
        help="Output directory for markdown/audit records.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    output_dir = PROJECT_ROOT / args.output_dir
    markdown_dir = output_dir / "markdown"
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = args.urls or DEFAULT_URLS

    print("========== HYBRID INGEST ACCEPTANCE ==========")
    print(f"output_dir={output_dir}")
    print(f"max_tokens={args.max_tokens} max_chunks={args.max_chunks or 'all'}")
    print(f"reset_graph={args.reset_graph} recreate_qdrant={args.recreate_qdrant}")
    print(f"skip_crawl={args.skip_crawl} skip_extract={args.skip_extract}")

    if not args.skip_crawl:
        run_crawler(urls=urls, output_dir=markdown_dir, clean=args.clean)

    articles = load_markdown_articles(markdown_dir)
    if not articles:
        raise RuntimeError(f"No markdown articles found at: {markdown_dir}")

    chunks = []
    chunks_by_article = {}
    for article in articles:
        article_chunks = chunk_article(article, max_tokens=args.max_tokens)
        chunks_by_article[article.article_id] = article_chunks
        chunks.extend(article_chunks)

    if args.max_chunks and args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]

    print("\n========== LOADED ==========")
    print(f"articles={len(articles)} chunks={len(chunks)}")
    for article in articles:
        print(f"- {article.title} | chunks={len(chunks_by_article.get(article.article_id, []))}")

    # Connections
    print("\n========== CONNECT ==========")
    graph_client, graph_repo = make_graph_repo(reset_graph=args.reset_graph)
    print("[PASS] Neo4j health check")

    embedder = EmbeddingService()
    print(f"[EMBEDDING] model={embedder.model_name} dim={embedder.vector_dim}")
    qdrant_client, vector_repo = make_vector_repo(embedder=embedder, recreate_qdrant=args.recreate_qdrant)
    print("[PASS] Qdrant health check")

    resolver = EntityResolver(graph_repo)
    article_chunk_graph_builder = GraphBuilder(graph_repo)
    medical_graph_builder = MedicalGraphBuilder(graph_repo=graph_repo, resolver=resolver)
    vector_builder = VectorGraphBuilder(
        vector_repo=vector_repo,
        embedder=embedder,
        resolver=resolver,
        batch_size=16,
    )

    # Article + Chunk to Neo4j
    print("\n========== INGEST ARTICLE/CHUNK ==========")
    for article in articles:
        article_chunk_graph_builder.upsert_article(article)
    article_chunk_graph_builder.upsert_chunks(chunks)
    print(f"[NEO4J] upsert articles={len(articles)} chunks={len(chunks)}")

    # Chunk vectors to Qdrant
    chunk_vector_count = vector_builder.upsert_chunks(chunks)
    print(f"[QDRANT] upsert chunk_vectors={chunk_vector_count}")

    extraction_records: list[dict[str, Any]] = []
    raw_entity_count = 0
    raw_relation_count = 0
    graph_entity_write_count = 0
    graph_relation_write_count = 0
    mention_write_count = 0
    entity_vector_count = 0
    relation_vector_count = 0

    entity_type_counter = Counter()
    relation_type_counter = Counter()
    keyword_len_counter = Counter()
    unique_entity_keys = set()
    unique_relation_keys = set()
    missing_keyword_relations = []
    synonym_relations = []

    if not args.skip_extract:
        require_env(["OPENAI_API_KEY"])
        extractor = MedicalExtractor(model=args.model, prompt_mode="strict_text")
        extraction_service = ExtractionService(extractor=extractor)

        print("\n========== EXTRACT + GRAPH + VECTOR ==========")
        for idx, chunk in enumerate(chunks, start=1):
            print(f"\n[EXTRACT] {idx}/{len(chunks)} | {chunk.title} | {chunk.section}", flush=True)
            graph = extraction_service.extract_and_validate_chunk(chunk)

            raw_entity_count += len(graph.entities)
            raw_relation_count += len(graph.relations)

            for entity in graph.entities:
                entity_type_counter[entity.entity_type] += 1
                unique_entity_keys.add((entity.entity_type, entity.name.strip().lower()))

            for relation in graph.relations:
                relation_type_counter[relation.relation_type] += 1
                keyword_len_counter[len(relation.keywords)] += 1
                unique_relation_keys.add(
                    (
                        relation.subject_type,
                        relation.subject.strip().lower(),
                        relation.relation_type,
                        relation.object_type,
                        relation.object.strip().lower(),
                    )
                )
                if not relation.keywords:
                    missing_keyword_relations.append(relation.model_dump())
                if relation.relation_type == "DONG_NGHIA_VOI":
                    synonym_relations.append(relation.model_dump())

            graph_stats = medical_graph_builder.write_chunk_extraction(chunk, graph)
            vector_stats = vector_builder.upsert_chunk_extraction(chunk, graph)

            graph_entity_write_count += graph_stats.get("entity_write_count", 0)
            graph_relation_write_count += graph_stats.get("relation_write_count", 0)
            mention_write_count += graph_stats.get("mention_write_count", 0)
            entity_vector_count += vector_stats.get("entity_vector_count", 0)
            relation_vector_count += vector_stats.get("relation_vector_count", 0)

            print(
                "          "
                f"entities={len(graph.entities)} relations={len(graph.relations)} | "
                f"graph_entities={graph_stats.get('entity_write_count', 0)} "
                f"graph_relations={graph_stats.get('relation_write_count', 0)} | "
                f"entity_vectors={vector_stats.get('entity_vector_count', 0)} "
                f"relation_vectors={vector_stats.get('relation_vector_count', 0)}",
                flush=True,
            )

            extraction_records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "article_id": chunk.article_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "entities": [entity.model_dump() for entity in graph.entities],
                    "relations": [relation.model_dump() for relation in graph.relations],
                    "graph_stats": graph_stats,
                    "vector_stats": vector_stats,
                }
            )

        save_json(output_dir / "extraction_records.json", extraction_records)
        save_jsonl(output_dir / "extraction_records.jsonl", extraction_records)

    neo4j_stats = graph_repo.get_graph_stats()
    qdrant_stats = {
        "chunk_vectors_total": qdrant_count(qdrant_client, CHUNK_COLLECTION),
        "entity_vectors_total": qdrant_count(qdrant_client, ENTITY_COLLECTION),
        "relation_vectors_total": qdrant_count(qdrant_client, RELATION_COLLECTION),
    }

    extraction_summary = {
        "raw_entities": raw_entity_count,
        "unique_entities_by_name_type": len(unique_entity_keys),
        "raw_relations": raw_relation_count,
        "unique_relations_by_names": len(unique_relation_keys),
        "entity_type_counter": dict(entity_type_counter),
        "relation_type_counter": dict(relation_type_counter),
        "keyword_len_counter": {str(k): v for k, v in keyword_len_counter.items()},
        "relations_missing_keywords": len(missing_keyword_relations),
        "synonym_relations": len(synonym_relations),
        "synonym_relations_sample": synonym_relations[:20],
        "missing_keyword_relations_sample": missing_keyword_relations[:20],
    }

    write_summary = {
        "graph_entity_write_count": graph_entity_write_count,
        "graph_relation_write_count": graph_relation_write_count,
        "mention_write_count": mention_write_count,
        "chunk_vector_count_this_run": chunk_vector_count,
        "entity_vector_count_this_run": entity_vector_count,
        "relation_vector_count_this_run": relation_vector_count,
    }

    summary = {
        "config": {
            "urls": urls,
            "max_tokens": args.max_tokens,
            "max_chunks": args.max_chunks,
            "skip_extract": args.skip_extract,
            "reset_graph": args.reset_graph,
            "recreate_qdrant": args.recreate_qdrant,
            "embedding_model": embedder.model_name,
            "embedding_dim": embedder.vector_dim,
        },
        "loaded": {
            "articles": len(articles),
            "chunks": len(chunks),
        },
        "extraction": extraction_summary,
        "write_summary": write_summary,
        "neo4j_stats_total": neo4j_stats,
        "qdrant_stats_total": qdrant_stats,
    }
    save_json(output_dir / "hybrid_ingest_acceptance_summary.json", summary)

    print("\n========== ACCEPTANCE SUMMARY ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    checks = []
    checks.append(acceptance_status("articles loaded", len(articles) > 0, f"articles={len(articles)}"))
    checks.append(acceptance_status("chunks loaded", len(chunks) > 0, f"chunks={len(chunks)}"))
    checks.append(
        acceptance_status(
            "chunk vectors upserted this run",
            chunk_vector_count == len(chunks),
            f"chunk_vectors={chunk_vector_count}, chunks={len(chunks)}",
        )
    )

    if not args.skip_extract:
        checks.append(
            acceptance_status(
                "entities extracted",
                raw_entity_count > 0,
                f"raw_entities={raw_entity_count}",
            )
        )
        checks.append(
            acceptance_status(
                "relations extracted",
                raw_relation_count > 0,
                f"raw_relations={raw_relation_count}",
            )
        )
        checks.append(
            acceptance_status(
                "entity vectors upserted",
                entity_vector_count > 0,
                f"entity_vectors_this_run={entity_vector_count}",
            )
        )
        checks.append(
            acceptance_status(
                "relation vectors upserted",
                relation_vector_count > 0,
                f"relation_vectors_this_run={relation_vector_count}",
            )
        )
        checks.append(
            acceptance_status(
                "relation keywords present",
                len(missing_keyword_relations) == 0,
                f"missing_keywords={len(missing_keyword_relations)}",
            )
        )
        checks.append(
            acceptance_status(
                "synonym edge smoke signal",
                len(synonym_relations) > 0,
                f"DONG_NGHIA_VOI={len(synonym_relations)}; WARN is acceptable if source text has no explicit synonym cue",
            )
        )

    # Vector retrieval smoke
    print("\n========== VECTOR RETRIEVAL SMOKE ==========")
    try:
        retriever = VectorRetriever(vector_repo=vector_repo, embedder=embedder)
        results = retriever.hybrid_search(
            args.query,
            chunk_limit=5,
            entity_limit=5,
            relation_limit=5,
        )
        print(f"query={args.query}")
        print_vector_results("VECTOR CHUNKS", results.chunks)
        print_vector_results("VECTOR ENTITIES", results.entities)
        print_vector_results("VECTOR RELATIONS", results.relations)

        top_entity_ids = [
            item.payload.get("entity_id")
            for item in results.entities
            if item.payload.get("entity_id")
        ]
        top_relation_ids = [
            item.payload.get("relation_id")
            for item in results.relations
            if item.payload.get("relation_id")
        ]

        if top_entity_ids:
            bundle = graph_repo.get_entity_context_bundle(
                top_entity_ids[:3],
                max_relations=20,
                max_chunks=10,
                expand_synonyms=True,
            )
            print("\n========== GRAPH CONTEXT FROM TOP ENTITIES ==========")
            print(
                f"seed_entities={len(top_entity_ids[:3])} "
                f"bundle_entities={len(bundle.entities)} "
                f"bundle_relations={len(bundle.relations)} "
                f"bundle_chunks={len(bundle.chunks)}"
            )

        if top_relation_ids:
            bundle = graph_repo.get_relation_context_bundle(top_relation_ids[:3], max_chunks=10)
            print("\n========== GRAPH CONTEXT FROM TOP RELATIONS ==========")
            print(
                f"seed_relations={len(top_relation_ids[:3])} "
                f"bundle_entities={len(bundle.entities)} "
                f"bundle_relations={len(bundle.relations)} "
                f"bundle_chunks={len(bundle.chunks)}"
            )

        checks.append(acceptance_status("vector retrieval smoke", True, "hybrid_search completed"))
    except Exception as exc:
        checks.append(acceptance_status("vector retrieval smoke", False, str(exc)))

    save_json(output_dir / "hybrid_ingest_acceptance_checks.json", checks)

    print("\n========== FINAL CHECKS ==========")
    hard_failures = [c for c in checks if not c["ok"] and c["name"] != "synonym edge smoke signal"]
    for check in checks:
        print(f"{'PASS' if check['ok'] else 'WARN'} | {check['name']} | {check['detail']}")

    if hard_failures:
        print("\n[RESULT] WARN: some required checks did not pass. Inspect logs above.")
    else:
        print("\n[RESULT] PASS: hybrid graph + vector ingestion base is usable for retrieval coding.")

    qdrant_client.close()
    graph_client.close()


if __name__ == "__main__":
    main()
