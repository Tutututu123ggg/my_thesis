from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.graph_database import GraphRepository, Neo4jClient
from app.ingestion.builders import GraphBuilder, MedicalGraphBuilder
from app.ingestion.extraction import ExtractionValidator, MedicalExtractor
from app.ingestion.processing import chunk_article, load_markdown_article
from app.ingestion.resolution import EntityResolver


def norm(text: str | None) -> str:
    text = str(text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def short(text: str | None, n: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_graph_repo() -> tuple[Neo4jClient, GraphRepository]:
    client = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "admin123456"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    return client, GraphRepository(client)


def resolve_md_path(value: str) -> Path:
    raw = str(value or "").strip().strip('"')
    if not raw:
        raise ValueError("md_path is empty")

    # CSV may contain Windows-style separators even when script runs on Linux/WSL.
    candidates = [Path(raw), PROJECT_ROOT / raw]
    raw_slash = raw.replace("\\", "/")
    candidates += [Path(raw_slash), PROJECT_ROOT / raw_slash]

    for p in candidates:
        if p.exists():
            return p.resolve()

    # Return a project-root path so the error is useful.
    return (PROJECT_ROOT / raw_slash).resolve()


def load_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def choose_row(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, str]:
    candidates = rows

    if args.topic:
        topic = args.topic.strip().lower()
        candidates = [r for r in candidates if str(r.get("topic_group", "")).strip().lower() == topic]

    if args.source:
        source = args.source.strip().lower()
        candidates = [r for r in candidates if str(r.get("source", "")).strip().lower() == source]

    if args.title_contains:
        q = norm(args.title_contains)
        candidates = [r for r in candidates if q in norm(r.get("title", ""))]

    if args.exclude_title_contains:
        q = norm(args.exclude_title_contains)
        candidates = [r for r in candidates if q not in norm(r.get("title", ""))]

    if args.min_words:
        candidates = [r for r in candidates if int(float(r.get("word_count") or 0)) >= args.min_words]

    if not candidates:
        raise RuntimeError("No article matched the filters")

    if args.random:
        random.seed(args.seed)
        return random.choice(candidates)

    idx = max(args.index, 0)
    if idx >= len(candidates):
        raise RuntimeError(f"index={idx} out of range, candidates={len(candidates)}")
    return candidates[idx]


def relation_key(relation: Any) -> str:
    return f"{relation.subject} [{relation.subject_type}] --{relation.relation_type}--> {relation.object} [{relation.object_type}]"


def audit_relation(relation: Any, chunk_text: str, article_title: str) -> dict[str, Any]:
    evidence = relation.evidence_text or ""
    combined = f"{article_title}\n{chunk_text}"
    n_e = norm(evidence)
    n_c = norm(combined)
    n_s = norm(relation.subject)
    n_o = norm(relation.object)

    evidence_in_chunk = bool(n_e and n_e in n_c)
    subject_in_evidence = bool(n_s and n_s in n_e)
    object_in_evidence = bool(n_o and n_o in n_e)
    subject_in_chunk = bool(n_s and n_s in n_c)
    object_in_chunk = bool(n_o and n_o in n_c)

    if subject_in_evidence and object_in_evidence:
        support = "DIRECT_IN_EVIDENCE"
    elif evidence_in_chunk and (subject_in_evidence or object_in_evidence):
        support = "PARTIAL_IN_EVIDENCE"
    elif subject_in_chunk and object_in_chunk:
        support = "HEADING_OR_CONTEXT_INFERRED"
    else:
        support = "WEAK_ENDPOINT_MATCH"

    return {
        "relation": relation_key(relation),
        "relation_type": relation.relation_type,
        "subject": relation.subject,
        "subject_type": relation.subject_type,
        "object": relation.object,
        "object_type": relation.object_type,
        "confidence": relation.confidence,
        "evidence_text": evidence,
        "evidence_in_chunk": evidence_in_chunk,
        "subject_in_evidence": subject_in_evidence,
        "object_in_evidence": object_in_evidence,
        "subject_in_chunk": subject_in_chunk,
        "object_in_chunk": object_in_chunk,
        "support": support,
    }


def graph_shape_metrics(relations: list[Any]) -> dict[str, Any]:
    if not relations:
        return {
            "relation_count": 0,
            "entity_endpoint_count": 0,
            "top_hub": None,
            "top_hub_incident_relations": 0,
            "top_hub_ratio": 0.0,
            "non_hub_relation_count": 0,
            "non_hub_relation_ratio": 0.0,
        }

    incident = Counter()
    for r in relations:
        incident[(r.subject, r.subject_type)] += 1
        incident[(r.object, r.object_type)] += 1

    top_entity, top_count = incident.most_common(1)[0]
    relation_count = len(relations)
    non_hub = [
        r for r in relations
        if (r.subject, r.subject_type) != top_entity and (r.object, r.object_type) != top_entity
    ]
    return {
        "relation_count": relation_count,
        "entity_endpoint_count": len(incident),
        "top_hub": {"name": top_entity[0], "entity_type": top_entity[1]},
        "top_hub_incident_relations": top_count,
        "top_hub_ratio": round(top_count / relation_count, 4),
        "non_hub_relation_count": len(non_hub),
        "non_hub_relation_ratio": round(len(non_hub) / relation_count, 4),
    }


def render_markdown_report(article: Any, chunk_records: list[dict[str, Any]], output_dir: Path) -> str:
    all_relations = []
    all_entities = []
    for rec in chunk_records:
        all_relations.extend(rec["validated_relations"])
        all_entities.extend(rec["validated_entities"])

    shape = graph_shape_metrics(all_relations)
    support_counts = Counter()
    type_counts = Counter()
    for rec in chunk_records:
        for item in rec["relation_audit"]:
            support_counts[item["support"]] += 1
            type_counts[item["relation_type"]] += 1

    lines: list[str] = []
    lines.append(f"# Graph audit — {article.title}\n")
    lines.append(f"- article_id: `{article.article_id}`")
    lines.append(f"- source_url: {article.url}")
    lines.append(f"- chunks: {len(chunk_records)}")
    lines.append(f"- entities(validated endpoints): {len(all_entities)}")
    lines.append(f"- relations(validated, written to Neo4j): {len(all_relations)}")
    if shape["top_hub"]:
        lines.append(f"- top_hub: {shape['top_hub']['name']} [{shape['top_hub']['entity_type']}] — {shape['top_hub_ratio']}")
    lines.append(f"- non_hub_relation_ratio: {shape['non_hub_relation_ratio']}")
    lines.append("")

    lines.append("## Relation type distribution\n")
    for k, v in type_counts.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Evidence support distribution\n")
    for k, v in support_counts.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Per-chunk audit\n")
    for rec in chunk_records:
        chunk = rec["chunk"]
        lines.append(f"### Chunk {chunk.chunk_index} — `{chunk.chunk_id}`\n")
        lines.append(f"- section: {chunk.section or ''}")
        lines.append(f"- heading_context: {chunk.metadata.get('heading_path_context') or ''}")
        lines.append(f"- graph_write: {rec['graph_write']}")
        lines.append("")
        lines.append("#### Text gốc")
        lines.append("```markdown")
        lines.append(chunk.text.strip())
        lines.append("```")
        lines.append("")

        lines.append("#### Relations sau validate + ghi Neo4j")
        if not rec["validated_relations"]:
            lines.append("- Không có relation.")
        else:
            for item in rec["relation_audit"]:
                rel = item["relation"]
                flags = []
                flags.append(item["support"])
                if not item["subject_in_evidence"]:
                    flags.append("subject_not_in_evidence")
                if not item["object_in_evidence"]:
                    flags.append("object_not_in_evidence")
                lines.append(f"- {rel}")
                lines.append(f"  - support: `{', '.join(flags)}`")
                lines.append(f"  - evidence: {short(item['evidence_text'], 260)}")
        lines.append("")

    # Browser queries are written into the report and a separate .cypher file.
    cypher = f"""
// Visualize this article's graph
MATCH (a:Article {{article_id: '{article.article_id}'}})-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
OPTIONAL MATCH (e)-[r:MEDICAL_RELATION]-(e2:Entity)
RETURN a, c, e, r, e2
LIMIT 300;

// Relation-only view from chunks of this article
MATCH (a:Article {{article_id: '{article.article_id}'}})-[:HAS_CHUNK]->(c:Chunk)
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WHERE c.chunk_id IN coalesce(r.evidence_chunk_ids, [])
RETURN s, r, o
LIMIT 300;
""".strip()
    (output_dir / "neo4j_browser_queries.cypher").write_text(cypher, encoding="utf-8")
    lines.append("## Neo4j Browser queries\n")
    lines.append("```cypher")
    lines.append(cypher)
    lines.append("```")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick one different article, extract graph, write Neo4j, and generate source-text graph audit."
    )
    parser.add_argument("--input-csv", default="data/final_selection/articles_final_selected_v1.csv")
    parser.add_argument("--output-dir", default="data/debug/one_article_graph_audit")
    parser.add_argument("--topic", default="")
    parser.add_argument("--source", default="")
    parser.add_argument("--title-contains", default="")
    parser.add_argument("--exclude-title-contains", default="viêm da cơ địa nên bổ sung vitamin")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=800)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    input_csv = PROJECT_ROOT / args.input_csv
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    row = choose_row(rows, args)
    md_path = resolve_md_path(row.get("md_path", ""))
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    print(f"[SELECTED] {row.get('title')} | topic={row.get('topic_group')} | source={row.get('source')}")
    print(f"[MD] {md_path}")

    graph_client = None
    try:
        graph_client, graph_repo = build_graph_repo()
        if not graph_repo.health_check():
            raise RuntimeError("Neo4j health check failed")
        graph_repo.setup_schema()
        if args.fresh_start:
            graph_repo.delete_all_data()
            graph_repo.setup_schema()

        resolver = EntityResolver(graph_repo)
        graph_builder = GraphBuilder(graph_repo)
        medical_graph_builder = MedicalGraphBuilder(graph_repo, resolver)
        extractor = MedicalExtractor(model=args.model)
        validator = ExtractionValidator()

        article = load_markdown_article(md_path)
        chunks = chunk_article(article, max_tokens=args.max_tokens)
        if args.max_chunks and args.max_chunks > 0:
            chunks = chunks[: args.max_chunks]

        graph_builder.ingest_article_with_chunks(article, chunks)

        chunk_records: list[dict[str, Any]] = []
        raw_jsonl = output_dir / "raw_and_validated_chunk_extractions.jsonl"
        if raw_jsonl.exists():
            raw_jsonl.unlink()

        for i, chunk in enumerate(chunks, start=1):
            print(f"[EXTRACT] chunk {i}/{len(chunks)} {chunk.chunk_id}", flush=True)
            raw_graph = extractor.extract_chunk(chunk)
            graph = validator.validate(raw_graph)
            graph_write = medical_graph_builder.write_chunk_extraction(chunk, graph)

            relation_audit = [audit_relation(r, chunk.text, article.title) for r in graph.relations]
            rec = {
                "chunk_id": chunk.chunk_id,
                "article_id": chunk.article_id,
                "title": chunk.title,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "heading_context": chunk.metadata.get("heading_path_context"),
                "is_medical_knowledge_chunk": graph.is_medical_knowledge_chunk,
                "should_ingest_graph": graph.should_ingest_graph,
                "skip_reason": graph.skip_reason,
                "raw_entities": [e.model_dump() for e in raw_graph.entities],
                "raw_relations": [r.model_dump() for r in raw_graph.relations],
                "validated_entities": [e.model_dump() for e in graph.entities],
                "validated_relations": [r.model_dump() for r in graph.relations],
                "relation_audit": relation_audit,
                "graph_write": graph_write,
            }
            append_jsonl(raw_jsonl, rec)
            chunk_records.append({
                "chunk": chunk,
                "raw_graph": raw_graph,
                "validated_graph": graph,
                "raw_entities": raw_graph.entities,
                "raw_relations": raw_graph.relations,
                "validated_entities": graph.entities,
                "validated_relations": graph.relations,
                "relation_audit": relation_audit,
                "graph_write": graph_write,
            })

        all_relations = [r for rec in chunk_records for r in rec["validated_relations"]]
        summary = {
            "selected_row": row,
            "md_path": str(md_path),
            "output_dir": str(output_dir),
            "model": args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "article_id": article.article_id,
            "title": article.title,
            "url": article.url,
            "chunks": len(chunks),
            "entities": sum(len(rec["validated_entities"]) for rec in chunk_records),
            "relations": len(all_relations),
            "graph_shape": graph_shape_metrics(all_relations),
        }
        write_json(output_dir / "summary.json", summary)
        report = render_markdown_report(article, chunk_records, output_dir)
        (output_dir / "graph_audit.md").write_text(report, encoding="utf-8")

        print("[DONE]")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"[REPORT] {output_dir / 'graph_audit.md'}")
        print(f"[CYPHER] {output_dir / 'neo4j_browser_queries.cypher'}")

    finally:
        if graph_client is not None:
            graph_client.close()


if __name__ == "__main__":
    main()
