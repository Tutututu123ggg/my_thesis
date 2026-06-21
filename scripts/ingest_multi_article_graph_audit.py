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
from app.ingestion.extraction import ArticleExtractionPipeline, ArticleKeywordProfiler, ExtractionValidator, MedicalExtractor
from app.ingestion.processing import chunk_article, load_markdown_article
from app.ingestion.resolution import EntityResolver


def norm(text: str | None) -> str:
    text = str(text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def short(text: str | None, n: int = 260) -> str:
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
    candidates = [Path(raw), PROJECT_ROOT / raw]
    raw_slash = raw.replace("\\", "/")
    candidates += [Path(raw_slash), PROJECT_ROOT / raw_slash]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return (PROJECT_ROOT / raw_slash).resolve()


def load_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


ASPECT_PATTERNS: list[tuple[str, list[str]]] = [
    ("thuốc/điều trị", ["thuoc", "dieu tri", "boi", "uong", "kem", "kháng nấm", "khang nam"]),
    ("triệu chứng/dấu hiệu", ["trieu chung", "dau hieu", "bieu hien", "ngua", "mun", "ban", "kho da"]),
    ("nguyên nhân/yếu tố", ["nguyen nhan", "do dau", "yeu to", "nguy co", "vi nam", "di ung"]),
    ("chẩn đoán/phân biệt", ["chan doan", "phan biet", "xet nghiem", "nhan biet", "dễ nhầm", "de nham"]),
    ("phòng ngừa/chăm sóc", ["phong ngua", "cham soc", "kieng", "nen an", "khong nen", "can tranh", "ve sinh"]),
    ("tổng quan", ["la gi", "tong quan", "benh gi", "co nguy hiem", "bao lau", "lay khong"]),
]


def infer_aspect(row: dict[str, str]) -> str:
    text = norm(" | ".join([
        row.get("title", ""),
        row.get("headings", ""),
        row.get("excerpt", ""),
        row.get("matched_keywords", ""),
    ]))
    for aspect, pats in ASPECT_PATTERNS:
        if any(norm(p) in text for p in pats):
            return aspect
    return "khác"


def filter_candidates(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    candidates = rows
    if args.topic:
        topic = args.topic.strip().lower()
        candidates = [r for r in candidates if str(r.get("topic_group", "")).strip().lower() == topic]
    if args.source:
        source = args.source.strip().lower()
        candidates = [r for r in candidates if str(r.get("source", "")).strip().lower() == source]
    if args.disease:
        q = norm(args.disease)
        candidates = [
            r for r in candidates
            if q in norm(" | ".join([
                r.get("title", ""), r.get("candidate_anchors", ""), r.get("matched_keywords", ""), r.get("excerpt", "")
            ]))
        ]
    if args.title_contains:
        q = norm(args.title_contains)
        candidates = [r for r in candidates if q in norm(r.get("title", ""))]
    if args.exclude_title_contains:
        q = norm(args.exclude_title_contains)
        candidates = [r for r in candidates if q not in norm(r.get("title", ""))]
    if args.min_words:
        candidates = [r for r in candidates if int(float(r.get("word_count") or 0)) >= args.min_words]
    return candidates


def choose_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    candidates = filter_candidates(rows, args)
    if not candidates:
        raise RuntimeError("No article matched the filters")

    random.seed(args.seed)
    if args.random:
        random.shuffle(candidates)

    limit = max(1, args.limit)
    if not args.aspect_diverse:
        return candidates[:limit]

    by_aspect: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in candidates:
        by_aspect[infer_aspect(r)].append(r)

    selected: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    # Prefer one article per recognizable aspect.
    for aspect, _ in ASPECT_PATTERNS:
        bucket = by_aspect.get(aspect, [])
        if args.random:
            random.shuffle(bucket)
        for r in bucket:
            url = r.get("canonical_url") or r.get("url") or r.get("md_path")
            if url not in seen_urls:
                selected.append(r)
                seen_urls.add(url)
                break
        if len(selected) >= limit:
            return selected[:limit]

    # Fill remaining slots.
    for r in candidates:
        url = r.get("canonical_url") or r.get("url") or r.get("md_path")
        if url not in seen_urls:
            selected.append(r)
            seen_urls.add(url)
        if len(selected) >= limit:
            break

    return selected[:limit]


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


def triangle_candidates(relations: list[Any]) -> list[dict[str, Any]]:
    """Find simple two-hop motifs disease -> symptom/factor -> treatment plus direct disease -> treatment."""
    rels = list(relations)
    direct_pairs = {(r.subject, r.object) for r in rels}
    candidates: list[dict[str, Any]] = []
    for a in rels:
        if a.relation_type not in {"CO_BIEU_HIEN", "LIEN_QUAN_YEU_TO", "GAY_BIEN_CHUNG"}:
            continue
        for b in rels:
            if b.subject != a.object:
                continue
            if b.relation_type not in {"DIEU_TRI_BANG", "PHONG_NGUA_BANG", "CAN_TRANH", "ANH_HUONG_DEN"}:
                continue
            has_direct = (a.subject, b.object) in direct_pairs
            candidates.append({
                "disease": a.subject,
                "middle": a.object,
                "target": b.object,
                "path": f"{a.subject} --{a.relation_type}--> {a.object} --{b.relation_type}--> {b.object}",
                "has_direct_edge": has_direct,
            })
    return candidates


def render_markdown_report(records: list[dict[str, Any]], output_dir: Path) -> str:
    all_relations = [r for rec in records for cr in rec["chunk_records"] for r in cr["validated_relations"]]
    all_entities = [e for rec in records for cr in rec["chunk_records"] for e in cr["validated_entities"]]
    shape = graph_shape_metrics(all_relations)
    triangles = triangle_candidates(all_relations)

    support_counts = Counter()
    type_counts = Counter()
    for rec in records:
        for cr in rec["chunk_records"]:
            for item in cr["relation_audit"]:
                support_counts[item["support"]] += 1
                type_counts[item["relation_type"]] += 1

    lines: list[str] = []
    lines.append("# Multi-article graph audit\n")
    lines.append(f"- articles: {len(records)}")
    lines.append(f"- chunks: {sum(len(r['chunk_records']) for r in records)}")
    lines.append(f"- entities(validated endpoints, duplicated by chunk): {len(all_entities)}")
    lines.append(f"- relations(validated, written to Neo4j): {len(all_relations)}")
    if shape["top_hub"]:
        lines.append(f"- top_hub: {shape['top_hub']['name']} [{shape['top_hub']['entity_type']}] — {shape['top_hub_ratio']}")
    lines.append(f"- non_hub_relation_ratio: {shape['non_hub_relation_ratio']}")
    lines.append(f"- triangle_or_2hop_candidates: {len(triangles)}")
    lines.append("")

    lines.append("## Selected articles\n")
    for idx, rec in enumerate(records, start=1):
        row = rec["row"]
        article = rec["article"]
        lines.append(f"{idx}. **{article.title}**")
        lines.append(f"   - topic: {row.get('topic_group', '')}; source: {row.get('source', '')}; aspect_guess: {infer_aspect(row)}")
        lines.append(f"   - url: {article.url}")
    lines.append("")

    lines.append("## Overall relation type distribution\n")
    for k, v in type_counts.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Overall evidence support distribution\n")
    for k, v in support_counts.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Triangle / 2-hop candidates\n")
    if not triangles:
        lines.append("- Chưa phát hiện motif 2-hop kiểu bệnh -> trung gian -> đích trong graph đã extract.")
    else:
        for tri in triangles[:80]:
            mark = " + direct edge" if tri["has_direct_edge"] else ""
            lines.append(f"- {tri['path']}{mark}")
    lines.append("")

    lines.append("## Per-article / per-chunk audit\n")
    for rec in records:
        article = rec["article"]
        row = rec["row"]
        chunk_records = rec["chunk_records"]
        rels = [r for cr in chunk_records for r in cr["validated_relations"]]
        lines.append(f"## Article — {article.title}\n")
        lines.append(f"- article_id: `{article.article_id}`")
        lines.append(f"- url: {article.url}")
        lines.append(f"- aspect_guess: {infer_aspect(row)}")
        lines.append(f"- chunks: {len(chunk_records)}")
        lines.append(f"- relations: {len(rels)}")
        art_shape = graph_shape_metrics(rels)
        if art_shape["top_hub"]:
            lines.append(f"- top_hub: {art_shape['top_hub']['name']} [{art_shape['top_hub']['entity_type']}] — {art_shape['top_hub_ratio']}")
        lines.append(f"- non_hub_relation_ratio: {art_shape['non_hub_relation_ratio']}")
        lines.append("")
        for cr in chunk_records:
            chunk = cr["chunk"]
            lines.append(f"### Chunk {chunk.chunk_index} — `{chunk.chunk_id}`\n")
            lines.append(f"- section: {chunk.section or ''}")
            lines.append(f"- heading_context: {chunk.metadata.get('heading_path_context') or ''}")
            lines.append(f"- graph_write: {cr['graph_write']}")
            lines.append("")
            lines.append("#### Text gốc")
            lines.append("```markdown")
            lines.append(chunk.text.strip())
            lines.append("```")
            lines.append("")
            lines.append("#### Relations sau validate + ghi Neo4j")
            if not cr["validated_relations"]:
                lines.append("- Không có relation.")
            else:
                for item in cr["relation_audit"]:
                    flags = [item["support"]]
                    if not item["subject_in_evidence"]:
                        flags.append("subject_not_in_evidence")
                    if not item["object_in_evidence"]:
                        flags.append("object_not_in_evidence")
                    lines.append(f"- {item['relation']}")
                    lines.append(f"  - support: `{', '.join(flags)}`")
                    lines.append(f"  - evidence: {short(item['evidence_text'], 260)}")
            lines.append("")

    article_ids = [rec["article"].article_id for rec in records]
    ids_literal = json.dumps(article_ids, ensure_ascii=False)
    cypher = f"""
// Visualize graph from these articles
MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk)
WHERE a.article_id IN {ids_literal}
OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
OPTIONAL MATCH (e)-[r:MEDICAL_RELATION]-(e2:Entity)
RETURN a, c, e, r, e2
LIMIT 800;

// Relation-only view from chunks of these articles
MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk)
WHERE a.article_id IN {ids_literal}
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WHERE c.chunk_id IN coalesce(r.evidence_chunk_ids, [])
RETURN s, r, o
LIMIT 800;

// Non-hub-ish relations: useful for checking whether the graph is only a flower
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WHERE any(cid IN coalesce(r.evidence_chunk_ids, []) WHERE cid STARTS WITH 'chunk_')
RETURN s.name AS subject, r.relation_type AS relation_type, o.name AS object, r.evidence_text AS evidence
LIMIT 200;

// 2-hop paths from disease-like nodes through symptoms/factors to treatments/factors
MATCH p = (d:Entity)-[:MEDICAL_RELATION]->(mid:Entity)-[:MEDICAL_RELATION]->(target:Entity)
WHERE d.entity_type = 'BENH_LY'
RETURN p
LIMIT 200;
""".strip()
    (output_dir / "neo4j_browser_queries.cypher").write_text(cypher, encoding="utf-8")
    lines.append("## Neo4j Browser queries\n")
    lines.append("```cypher")
    lines.append(cypher)
    lines.append("```")
    return "\n".join(lines).strip() + "\n"


def process_article(row: dict[str, str], args: argparse.Namespace, services: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    md_path = resolve_md_path(row.get("md_path", ""))
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    article = load_markdown_article(md_path)
    retrieval_chunks = chunk_article(article, max_tokens=args.max_tokens)
    if args.max_chunks and args.max_chunks > 0:
        retrieval_chunks = retrieval_chunks[: args.max_chunks]

    services["graph_builder"].ingest_article_with_chunks(article, retrieval_chunks)

    raw_jsonl = output_dir / "raw_and_validated_chunk_extractions.jsonl"

    def progress(i: int, total: int, window) -> None:
        print(f"[EXTRACT] {article.title} | window {i}/{total} {window.chunk_id}", flush=True)

    result = services["extraction_pipeline"].extract_article(
        article=article,
        retrieval_chunks=retrieval_chunks,
        max_windows=args.max_extraction_windows,
        progress_callback=progress,
    )

    extra_windows = result.extra_extraction_windows
    if extra_windows:
        services["graph_builder"].upsert_chunks(extra_windows)

    chunk_records: list[dict[str, Any]] = []
    for chunk, raw_graph, graph in zip(
        result.extraction_windows,
        result.raw_graphs,
        result.validated_graphs,
        strict=True,
    ):
        graph_write = services["medical_graph_builder"].write_chunk_extraction(chunk, graph)
        relation_audit = [audit_relation(r, chunk.text, article.title) for r in graph.relations]
        rec_json = {
            "chunk_id": chunk.chunk_id,
            "article_id": chunk.article_id,
            "title": chunk.title,
            "section": chunk.section,
            "subsection": chunk.subsection,
            "heading_context": chunk.metadata.get("heading_path_context"),
            "is_extraction_window": bool(chunk.metadata.get("extraction_window")),
            "parent_chunk_id": chunk.metadata.get("parent_chunk_id"),
            "windowing_debug": result.windowing_debug,
            "is_medical_knowledge_chunk": graph.is_medical_knowledge_chunk,
            "should_ingest_graph": graph.should_ingest_graph,
            "skip_reason": graph.skip_reason,
            "raw_entities": [e.model_dump() for e in raw_graph.entities],
            "raw_relations": [r.model_dump() for r in raw_graph.relations],
            "validated_entities": [e.model_dump() for e in graph.entities],
            "validated_relations": [r.model_dump() for r in graph.relations],
            "keyword_profile_debug": result.keyword_profile_debug,
            "relation_audit": relation_audit,
            "graph_write": graph_write,
        }
        append_jsonl(raw_jsonl, rec_json)
        chunk_records.append({
            "chunk": chunk,
            "raw_graph": raw_graph,
            "validated_graph": graph,
            "raw_entities": raw_graph.entities,
            "raw_relations": raw_graph.relations,
            "validated_entities": graph.entities,
            "validated_relations": graph.relations,
            "keyword_profile_debug": result.keyword_profile_debug,
            "windowing_debug": result.windowing_debug,
            "relation_audit": relation_audit,
            "graph_write": graph_write,
        })
    return {
        "row": row,
        "md_path": str(md_path),
        "article": article,
        "retrieval_chunk_count": len(retrieval_chunks),
        "extraction_window_count": len(result.extraction_windows),
        "windowing_debug": result.windowing_debug,
        "keyword_profile_debug": result.keyword_profile_debug,
        "chunk_records": chunk_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick 3-5 articles about the same disease, extract graph, write Neo4j, and audit graph shape / triangle motifs."
    )
    parser.add_argument("--input-csv", default="data/final_selection/articles_final_selected_v1.csv")
    parser.add_argument("--output-dir", default="data/debug/multi_article_graph_audit")
    parser.add_argument("--topic", default="")
    parser.add_argument("--disease", default="")
    parser.add_argument("--source", default="")
    parser.add_argument("--title-contains", default="")
    parser.add_argument("--exclude-title-contains", default="")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--aspect-diverse", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=900, help="Retrieval/vector chunk size.")
    parser.add_argument("--extraction-window-tokens", type=int, default=450, help="Fact-window size for LLM extraction only.")
    parser.add_argument("--extraction-window-min-tokens", type=int, default=80)
    parser.add_argument("--extraction-window-overlap-sentences", type=int, default=1)
    parser.add_argument("--disable-extraction-windows", action="store_true")
    parser.add_argument("--max-extraction-windows", type=int, default=0, help="Debug cap per article. 0 = no cap.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--keyword-model", default=None, help="Optional article-level keyword profiler model. Empty = deterministic only.")
    parser.add_argument("--skip-keyword-profiler", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    input_csv = PROJECT_ROOT / args.input_csv
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_dir / "raw_and_validated_chunk_extractions.jsonl"
    if raw_jsonl.exists():
        raw_jsonl.unlink()

    rows = load_rows(input_csv)
    selected = choose_rows(rows, args)
    print(f"[SELECTED] {len(selected)} articles")
    for i, row in enumerate(selected, start=1):
        print(f"  {i}. {row.get('title')} | topic={row.get('topic_group')} | aspect={infer_aspect(row)}")

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
        keyword_profiler = None if args.skip_keyword_profiler else ArticleKeywordProfiler(model=args.keyword_model)
        services = {
            "graph_builder": GraphBuilder(graph_repo),
            "medical_graph_builder": MedicalGraphBuilder(graph_repo, resolver),
            "extraction_pipeline": ArticleExtractionPipeline(
                extractor=MedicalExtractor(model=args.model),
                validator=ExtractionValidator(),
                keyword_profiler=keyword_profiler,
                use_fact_windows=not args.disable_extraction_windows,
                fact_window_max_tokens=args.extraction_window_tokens,
                fact_window_min_tokens=args.extraction_window_min_tokens,
                fact_window_overlap_sentences=args.extraction_window_overlap_sentences,
            ),
        }

        records = [process_article(row, args, services, output_dir) for row in selected]
        all_relations = [r for rec in records for cr in rec["chunk_records"] for r in cr["validated_relations"]]
        all_entities = [e for rec in records for cr in rec["chunk_records"] for e in cr["validated_entities"]]
        triangles = triangle_candidates(all_relations)
        summary = {
            "selected_rows": selected,
            "output_dir": str(output_dir),
            "model": args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "keyword_model": (args.keyword_model or "deterministic_only") if not args.skip_keyword_profiler else "disabled",
            "extraction_window_enabled": not args.disable_extraction_windows,
            "extraction_window_tokens": args.extraction_window_tokens,
            "articles": [
                {
                    "article_id": rec["article"].article_id,
                    "title": rec["article"].title,
                    "url": rec["article"].url,
                    "topic_group": rec["row"].get("topic_group", ""),
                    "aspect_guess": infer_aspect(rec["row"]),
                    "retrieval_chunks": rec.get("retrieval_chunk_count", 0),
                    "extraction_windows": rec.get("extraction_window_count", len(rec["chunk_records"])),
                    "chunks": len(rec["chunk_records"]),
                    "windowing_debug": rec.get("windowing_debug", {}),
                    "relations": sum(len(cr["validated_relations"]) for cr in rec["chunk_records"]),
                    "graph_shape": graph_shape_metrics([r for cr in rec["chunk_records"] for r in cr["validated_relations"]]),
                }
                for rec in records
            ],
            "total_retrieval_chunks": sum(int(rec.get("retrieval_chunk_count", 0) or 0) for rec in records),
            "total_extraction_windows": sum(int(rec.get("extraction_window_count", len(rec["chunk_records"])) or 0) for rec in records),
            "total_chunks": sum(len(rec["chunk_records"]) for rec in records),
            "total_entities": len(all_entities),
            "total_relations": len(all_relations),
            "graph_shape": graph_shape_metrics(all_relations),
            "triangle_or_2hop_candidates": triangles[:200],
            "triangle_or_2hop_count": len(triangles),
        }
        write_json(output_dir / "summary.json", summary)
        report = render_markdown_report(records, output_dir)
        (output_dir / "multi_graph_audit.md").write_text(report, encoding="utf-8")

        print("[DONE]")
        print(json.dumps({
            "articles": len(records),
            "total_retrieval_chunks": summary["total_retrieval_chunks"],
            "total_extraction_windows": summary["total_extraction_windows"],
            "total_relations": summary["total_relations"],
            "graph_shape": summary["graph_shape"],
            "triangle_or_2hop_count": summary["triangle_or_2hop_count"],
            "output_dir": str(output_dir),
        }, ensure_ascii=False, indent=2), flush=True)
        print(f"[REPORT] {output_dir / 'multi_graph_audit.md'}")
        print(f"[CYPHER] {output_dir / 'neo4j_browser_queries.cypher'}")

    finally:
        if graph_client is not None:
            graph_client.close()


if __name__ == "__main__":
    main()