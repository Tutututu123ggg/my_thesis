from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.graph_database import GraphRepository, Neo4jClient
from app.infrastructure.vector_database import (
    CHUNK_COLLECTION,
    ENTITY_COLLECTION,
    NAIVE_CHUNK_COLLECTION,
    RELATION_COLLECTION,
    QdrantClientWrapper,
    VectorRepository,
)
from app.ingestion.builders import GraphBuilder, MedicalGraphBuilder, VectorGraphBuilder
from app.ingestion.extraction import ExtractionService, ExtractionValidator, MedicalExtractor
from app.ingestion.resolution import EntityResolver

from scripts.ingest_articles_final_v1 import process_file


def first_existing(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_processed_article_ids(article_stats_jsonl: Path) -> set[str]:
    if not article_stats_jsonl.exists():
        return set()

    done: set[str] = set()

    with article_stats_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            article_id = str(obj.get("article_id", "")).strip()
            if article_id:
                done.add(article_id)

    return done


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_csv_stream(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def choose_topic_auto(input_csv: Path) -> str:
    counter: Counter[str] = Counter()

    for row in read_csv_stream(input_csv):
        topic = first_existing(row, ["topic_group", "topic_hint", "cluster"])
        decision = first_existing(row, ["content_decision", "decision"])

        if decision and decision.upper() not in {"APPROVE", "REVIEW"}:
            continue

        if topic:
            counter[topic] += 1

    if not counter:
        raise RuntimeError("Không tìm được topic_group/topic_hint trong CSV để auto chọn topic.")

    topic, count = counter.most_common(1)[0]
    print(f"[AUTO TOPIC] selected topic={topic} count={count}", flush=True)
    return topic


def select_topic_rows(
    input_csv: Path,
    *,
    topic: str,
    limit: int,
    source_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    for row in read_csv_stream(input_csv):
        row_topic = first_existing(row, ["topic_group", "topic_hint", "cluster"])
        source = first_existing(row, ["source", "domain"])
        md_path = first_existing(row, ["md_path", "file_path", "path", "markdown_path"])

        if topic != "all" and row_topic != topic:
            continue

        if source_filter and source not in source_filter:
            continue

        if not md_path:
            continue

        path = resolve_path(md_path)
        if not path.exists():
            print(f"[WARN] missing md_path: {path}", flush=True)
            continue

        row["_resolved_md_path"] = str(path)
        selected.append(row)

        if limit > 0 and len(selected) >= limit:
            break

    return selected


def save_selected_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames = [k for k in rows[0].keys() if not k.startswith("_")]
    if "resolved_md_path" not in fieldnames:
        fieldnames.append("resolved_md_path")

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            out = {k: v for k, v in row.items() if not k.startswith("_")}
            out["resolved_md_path"] = row.get("_resolved_md_path", "")
            writer.writerow(out)


def build_graph_repo() -> tuple[Neo4jClient, GraphRepository]:
    client = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "admin123456"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    return client, GraphRepository(client)


def build_vector_repo(embedder: EmbeddingService) -> tuple[QdrantClientWrapper, VectorRepository]:
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


def run_cypher(graph_repo: GraphRepository, query: str) -> list[dict[str, Any]]:
    with graph_repo.client.session() as session:
        result = session.run(query)
        return [dict(record) for record in result]


def collect_graph_distribution(graph_repo: GraphRepository) -> dict[str, Any]:
    return {
        "node_labels": run_cypher(
            graph_repo,
            """
            MATCH (n)
            RETURN labels(n) AS labels, count(n) AS count
            ORDER BY count DESC
            """,
        ),
        "entity_types": run_cypher(
            graph_repo,
            """
            MATCH (e:Entity)
            RETURN coalesce(e.entity_type, '') AS entity_type, count(e) AS count
            ORDER BY count DESC, entity_type
            """,
        ),
        "relationship_types": run_cypher(
            graph_repo,
            """
            MATCH ()-[r]->()
            RETURN type(r) AS edge_label,
                   coalesce(r.relation_type, '') AS relation_type,
                   count(r) AS count
            ORDER BY count DESC, edge_label, relation_type
            """,
        ),
        "top_entities_by_mentions": run_cypher(
            graph_repo,
            """
            MATCH (e:Entity)
            RETURN e.name AS name,
                   e.entity_type AS entity_type,
                   coalesce(e.mention_count, 0) AS mention_count,
                   coalesce(e.source_count, 0) AS source_count
            ORDER BY mention_count DESC, source_count DESC, name
            LIMIT 30
            """,
        ),
        "top_articles_by_chunks": run_cypher(
            graph_repo,
            """
            MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk)
            RETURN a.title AS title, count(c) AS chunks
            ORDER BY chunks DESC, title
            LIMIT 20
            """,
        ),
        "orphan_entities": run_cypher(
            graph_repo,
            """
            MATCH (e:Entity)
            WHERE NOT (e)-[:MEDICAL_RELATION]-(:Entity)
            RETURN e.entity_type AS entity_type, count(e) AS count
            ORDER BY count DESC
            """,
        ),
    }


def collect_qdrant_counts(vector_repo: VectorRepository) -> dict[str, int]:
    return {
        CHUNK_COLLECTION: vector_repo.count_points(CHUNK_COLLECTION),
        NAIVE_CHUNK_COLLECTION: vector_repo.count_points(NAIVE_CHUNK_COLLECTION),
        ENTITY_COLLECTION: vector_repo.count_points(ENTITY_COLLECTION),
        RELATION_COLLECTION: vector_repo.count_points(RELATION_COLLECTION),
    }


def print_distribution_short(distribution: dict[str, Any], qdrant_counts: dict[str, int]) -> None:
    print("\n" + "=" * 100)
    print("[GRAPH DISTRIBUTION]")

    print("\nNode labels:")
    for row in distribution["node_labels"]:
        print(f"  {row['labels']}: {row['count']}")

    print("\nEntity types:")
    for row in distribution["entity_types"]:
        print(f"  {row['entity_type']}: {row['count']}")

    print("\nRelation types:")
    for row in distribution["relationship_types"]:
        label = row["edge_label"]
        rel_type = row["relation_type"]
        count = row["count"]
        if rel_type:
            print(f"  {label}.{rel_type}: {count}")
        else:
            print(f"  {label}: {count}")

    print("\nQdrant counts:")
    for name, count in qdrant_counts.items():
        print(f"  {name}: {count}")

    print("=" * 100 + "\n")


def cleanup_memory() -> None:
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def reset_output_files(output_dir: Path) -> None:
    if output_dir.exists():
        for name in [
            "chunk_extractions.jsonl",
            "article_stats.jsonl",
            "summary.json",
            "graph_distribution.json",
            "qdrant_counts.json",
            "selected_articles.csv",
        ]:
            path = output_dir / name
            if path.exists():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-memory smoke ingest 10-20 final selected articles from one topic into Neo4j + Qdrant."
    )
    parser.add_argument(
        "--input-csv",
        default="data/final_selection/articles_final_selected_v1.csv",
        help="CSV final selected articles, must contain md_path/file_path and topic_group.",
    )
    parser.add_argument("--topic", default="viem_da_cham", help="topic_group to ingest. Use 'auto' or 'all'.")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--sources", nargs="*", default=None, help="Optional source filter: tamanh vinmec medlatec")
    parser.add_argument("--output-dir", default="data/processed/ingest_smoke_topic_v1")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fresh-start", action="store_true", help="XÓA Neo4j data và recreate Qdrant collections trước khi ingest.")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-vectors", action="store_true")
    parser.add_argument("--skip-graph-document", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    input_csv = resolve_path(args.input_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Không thấy input CSV: {input_csv}")

    topic = args.topic
    if topic == "auto":
        topic = choose_topic_auto(input_csv)

    source_filter = set(args.sources) if args.sources else None

    rows = select_topic_rows(
        input_csv,
        topic=topic,
        limit=args.limit,
        source_filter=source_filter,
    )

    if not rows:
        raise RuntimeError(
            f"Không chọn được bài nào. input_csv={input_csv}, topic={topic}, limit={args.limit}"
        )

    if args.fresh_start:
        reset_output_files(output_dir)

    selected_csv = output_dir / "selected_articles.csv"
    save_selected_csv(selected_csv, rows)

    extraction_jsonl = output_dir / "chunk_extractions.jsonl"
    article_stats_jsonl = output_dir / "article_stats.jsonl"

    processed_article_ids = set() if args.fresh_start else load_processed_article_ids(article_stats_jsonl)

    print("=" * 100)
    print("[SMOKE INGEST TOPIC]")
    print(f"input_csv  : {input_csv}")
    print(f"topic      : {topic}")
    print(f"limit      : {args.limit}")
    print(f"selected   : {len(rows)}")
    print(f"output_dir : {output_dir}")
    print(f"fresh_start: {args.fresh_start}")
    print(f"model      : {args.model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    print(f"selected_csv: {selected_csv}")
    print("=" * 100)

    graph_client = None
    qdrant_client = None

    try:
        graph_client, graph_repo = build_graph_repo()
        if not graph_repo.health_check():
            raise RuntimeError("Neo4j health check failed")

        graph_repo.setup_schema()

        if args.fresh_start:
            print("[FRESH START] Deleting all Neo4j data...", flush=True)
            graph_repo.delete_all_data()
            graph_repo.setup_schema()

        embedder = EmbeddingService()
        qdrant_client, vector_repo = build_vector_repo(embedder)

        if not vector_repo.health_check():
            raise RuntimeError("Qdrant health check failed")

        vector_builder = VectorGraphBuilder(
            vector_repo=vector_repo,
            embedder=embedder,
            resolver=EntityResolver(graph_repo),
            batch_size=16,
        )
        vector_builder.setup_collections(recreate=args.fresh_start)

        resolver = EntityResolver(graph_repo)
        graph_builder = GraphBuilder(graph_repo)
        medical_graph_builder = MedicalGraphBuilder(graph_repo, resolver)

        extraction_service = ExtractionService(
            extractor=MedicalExtractor(model=args.model),
            validator=ExtractionValidator(),
        )

        worker_args = argparse.Namespace(
            max_tokens=args.max_tokens,
            skip_graph_document=args.skip_graph_document,
            skip_vectors=args.skip_vectors,
            skip_extract=args.skip_extract,
        )

        totals: dict[str, int] = {
            "articles": 0,
            "chunks": 0,
            "medical_chunks": 0,
            "skipped_chunks": 0,
            "entities": 0,
            "relations": 0,
            "entity_writes": 0,
            "relation_writes": 0,
            "chunk_vectors": 0,
            "naive_chunk_vectors": 0,
            "entity_vectors": 0,
            "relation_vectors": 0,
        }

        for index, row in enumerate(rows, start=1):
            path = Path(row["_resolved_md_path"])

            # process_file tự load 1 markdown -> chunk -> ghi graph/vector.
            # Sau mỗi article, function return và mình gc để tránh giữ content lâu.
            print(f"\n[{index}/{len(rows)}] INGEST {path.name}", flush=True)

            stats = process_file(
                path,
                args=worker_args,
                graph_builder=graph_builder,
                medical_graph_builder=medical_graph_builder,
                vector_builder=vector_builder,
                extraction_service=extraction_service,
                extraction_jsonl=extraction_jsonl,
            )

            article_id = str(stats.get("article_id", "")).strip()

            if article_id in processed_article_ids:
                print(f"[SKIP STATS DUP] article_id already in stats: {article_id}", flush=True)
            else:
                enriched_stats = {
                    **stats,
                    "selection_topic": topic,
                    "selection_row_id": first_existing(row, ["id"]),
                    "selection_source": first_existing(row, ["source", "domain"]),
                    "selection_topic_group": first_existing(row, ["topic_group", "topic_hint", "cluster"]),
                    "selection_quality_score": first_existing(row, ["quality_score"]),
                    "selection_content_decision": first_existing(row, ["content_decision"]),
                }
                write_jsonl(article_stats_jsonl, enriched_stats)
                processed_article_ids.add(article_id)

            totals["articles"] += 1
            for key in totals:
                if key == "articles":
                    continue
                totals[key] += int(stats.get(key, 0) or 0)

            print(
                f"[DONE] title={stats['title']} | "
                f"chunks={stats['chunks']} medical={stats['medical_chunks']} "
                f"skip={stats['skipped_chunks']} entities={stats['entities']} "
                f"relations={stats['relations']} entity_writes={stats['entity_writes']} "
                f"relation_writes={stats['relation_writes']}",
                flush=True,
            )

            del stats
            cleanup_memory()

        distribution = collect_graph_distribution(graph_repo)
        qdrant_counts = collect_qdrant_counts(vector_repo)

        write_json(output_dir / "graph_distribution.json", distribution)
        write_json(output_dir / "qdrant_counts.json", qdrant_counts)

        summary = {
            "input_csv": str(input_csv),
            "output_dir": str(output_dir),
            "topic": topic,
            "selected_count": len(rows),
            "max_tokens": args.max_tokens,
            "fresh_start": args.fresh_start,
            "model": args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "embedding_model": embedder.model_name,
            "embedding_dim": embedder.vector_dim,
            "totals": totals,
            "qdrant_counts": qdrant_counts,
        }

        write_json(output_dir / "summary.json", summary)

        print_distribution_short(distribution, qdrant_counts)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    finally:
        if graph_client is not None:
            graph_client.close()
        if qdrant_client is not None:
            qdrant_client.close()

        cleanup_memory()


if __name__ == "__main__":
    main()