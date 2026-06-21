from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.extraction.extraction_models import ExtractedGraph
from app.ingestion.extraction.extraction_prompt import build_system_prompt
from app.ingestion.extraction.extraction_validator import ExtractionValidator
from app.ingestion.extraction.medical_extractor import (
    MedicalExtractor,
    build_extraction_user_prompt,
)
from app.ingestion.processing import chunk_article, load_markdown_article


def norm(text: str) -> str:
    import unicodedata

    text = str(text or "").strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


def find_md_path_from_csv(
    csv_path: Path,
    *,
    article_id: str | None,
    title_contains: str | None,
    row_index: int | None,
) -> Path:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if row_index is not None:
        if row_index < 1 or row_index > len(rows):
            raise ValueError(f"row_index={row_index} ngoài range 1..{len(rows)}")
        row = rows[row_index - 1]
        md_path = first_existing(row, ["md_path", "file_path", "path", "markdown_path", "resolved_md_path"])
        if not md_path:
            raise ValueError("Row được chọn không có md_path/file_path/path.")
        return resolve_path(md_path)

    for row in rows:
        rid = first_existing(row, ["id", "article_id", "document_id"])
        title = first_existing(row, ["title", "article_title", "name"])

        if article_id and rid == article_id:
            md_path = first_existing(row, ["md_path", "file_path", "path", "markdown_path", "resolved_md_path"])
            if not md_path:
                raise ValueError(f"Row id={article_id} không có md_path/file_path/path.")
            return resolve_path(md_path)

        if title_contains and norm(title_contains) in norm(title):
            md_path = first_existing(row, ["md_path", "file_path", "path", "markdown_path", "resolved_md_path"])
            if not md_path:
                raise ValueError(f"Row title match nhưng không có md_path/file_path/path: {title}")
            return resolve_path(md_path)

    raise ValueError("Không tìm được article trong CSV theo điều kiện đã nhập.")


def entity_text(entity: Any) -> str:
    return f"{getattr(entity, 'name', '')} [{getattr(entity, 'entity_type', '')}]"


def relation_text(relation: Any) -> str:
    return (
        f"{getattr(relation, 'subject', '')}"
        f" [{getattr(relation, 'subject_type', '')}] "
        f"--{getattr(relation, 'relation_type', '')}--> "
        f"{getattr(relation, 'object', '')}"
        f" [{getattr(relation, 'object_type', '')}]"
    )


def entity_sig(entity: Any) -> tuple[str, str]:
    return norm(getattr(entity, "name", "")), str(getattr(entity, "entity_type", "")).strip()


def relation_sig(relation: Any) -> tuple[str, str, str, str, str]:
    return (
        norm(getattr(relation, "subject", "")),
        str(getattr(relation, "subject_type", "")).strip(),
        str(getattr(relation, "relation_type", "")).strip(),
        norm(getattr(relation, "object", "")),
        str(getattr(relation, "object_type", "")).strip(),
    )


def graph_to_dict(graph: ExtractedGraph) -> dict[str, Any]:
    return {
        "is_medical_knowledge_chunk": graph.is_medical_knowledge_chunk,
        "should_ingest_graph": graph.should_ingest_graph,
        "skip_reason": graph.skip_reason,
        "entities": [safe_dump(e) for e in graph.entities],
        # These are already expanded binary relations after relation_group splitting
        # and direction normalization in LLMExtractedGraph.to_internal_graph().
        "relations": [safe_dump(r) for r in graph.relations],
    }


def entity_degrees(relations: list[Any]) -> Counter[tuple[str, str]]:
    deg: Counter[tuple[str, str]] = Counter()
    for relation in relations:
        deg[(norm(getattr(relation, "subject", "")), str(getattr(relation, "subject_type", "")).strip())] += 1
        deg[(norm(getattr(relation, "object", "")), str(getattr(relation, "object_type", "")).strip())] += 1
    return deg


def contains_any(text: str, words: list[str]) -> bool:
    normalized = norm(text)
    return any(norm(word) in normalized for word in words if word)


def diagnose_anchor_watch(
    *,
    raw_graph: ExtractedGraph,
    validated_graph: ExtractedGraph,
    anchor: str,
    watch_terms: list[str],
) -> dict[str, Any]:
    anchor_n = norm(anchor)
    out: dict[str, Any] = {"anchor": anchor, "watch_terms": watch_terms, "terms": []}

    for term in watch_terms:
        term_n = norm(term)

        def ent_match(entity: Any) -> bool:
            return term_n in norm(getattr(entity, "name", ""))

        def rel_match_anchor_term(relation: Any) -> bool:
            s = norm(getattr(relation, "subject", ""))
            o = norm(getattr(relation, "object", ""))
            has_anchor = anchor_n in s or anchor_n in o
            has_term = term_n in s or term_n in o
            return has_anchor and has_term

        raw_entities = [entity_text(e) for e in raw_graph.entities if ent_match(e)]
        val_entities = [entity_text(e) for e in validated_graph.entities if ent_match(e)]
        raw_relations = [relation_text(r) for r in raw_graph.relations if rel_match_anchor_term(r)]
        val_relations = [relation_text(r) for r in validated_graph.relations if rel_match_anchor_term(r)]

        if raw_relations and not val_relations:
            diagnosis = "RAW_CO_VALIDATED_DROP"
        elif not raw_relations:
            diagnosis = "RAW_KHONG_SINH_RELATION"
        else:
            diagnosis = "VALIDATED_CO_RELATION"

        out["terms"].append(
            {
                "term": term,
                "raw_expanded_entities": raw_entities,
                "validated_entities": val_entities,
                "raw_expanded_anchor_term_relations": raw_relations,
                "validated_anchor_term_relations": val_relations,
                "diagnosis": diagnosis,
            }
        )

    return out


def analyze_chunk(
    *,
    chunk: Any,
    raw_graph: ExtractedGraph,
    validated_graph: ExtractedGraph,
    anchor: str,
    watch_terms: list[str],
) -> dict[str, Any]:
    val_entity_sigs = {entity_sig(e) for e in validated_graph.entities}
    val_relation_sigs = {relation_sig(r) for r in validated_graph.relations}

    removed_entities = [entity_text(e) for e in raw_graph.entities if entity_sig(e) not in val_entity_sigs]
    removed_relations = [
        {
            "triple": relation_text(r),
            "evidence_text": getattr(r, "evidence_text", ""),
            "confidence": getattr(r, "confidence", None),
        }
        for r in raw_graph.relations
        if relation_sig(r) not in val_relation_sigs
    ]

    raw_deg = entity_degrees(raw_graph.relations)
    val_deg = entity_degrees(validated_graph.relations)

    return {
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "title": chunk.title,
        "section": chunk.section,
        "subsection": chunk.subsection,
        "heading_path": getattr(chunk, "heading_path", []),
        "heading_context": chunk.metadata.get("heading_path_context"),
        "token_count": chunk.token_count,
        "raw_expanded_counts": {
            "entities": len(raw_graph.entities),
            "relations_after_group_split": len(raw_graph.relations),
        },
        "validated_counts": {
            "entities": len(validated_graph.entities),
            "relations": len(validated_graph.relations),
        },
        "removed_by_validator": {
            "entities": removed_entities,
            "relations": removed_relations,
        },
        "zero_degree_entities": {
            "raw_expanded": [entity_text(e) for e in raw_graph.entities if entity_sig(e) not in raw_deg],
            "validated": [entity_text(e) for e in validated_graph.entities if entity_sig(e) not in val_deg],
        },
        "anchor_watch_diagnosis": diagnose_anchor_watch(
            raw_graph=raw_graph,
            validated_graph=validated_graph,
            anchor=anchor,
            watch_terms=watch_terms,
        )
        if anchor and watch_terms
        else None,
    }


def render_chunk_markdown(
    *,
    chunk: Any,
    user_prompt: str,
    raw_graph: ExtractedGraph,
    validated_graph: ExtractedGraph,
    analysis: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append(f"# Chunk {chunk.chunk_index} — {chunk.chunk_id}")
    lines.append("")
    lines.append(f"- title: {chunk.title}")
    lines.append(f"- section: {chunk.section}")
    lines.append(f"- subsection: {chunk.subsection}")
    lines.append(f"- heading_context: {chunk.metadata.get('heading_path_context')}")
    lines.append(f"- token_count: {chunk.token_count}")
    lines.append("")
    lines.append("## Prompt sent to LLM")
    lines.append("```markdown")
    lines.append(user_prompt)
    lines.append("```")
    lines.append("")
    lines.append("## Raw chunk text")
    lines.append("```markdown")
    lines.append(chunk.text)
    lines.append("```")
    lines.append("")
    lines.append("## RAW entities")
    for entity in raw_graph.entities:
        lines.append(f"- {entity_text(entity)} | evidence: {getattr(entity, 'evidence_text', '')}")

    lines.append("")
    lines.append("## RAW expanded relations after relation_group split")
    lines.append("<!-- relation_groups are not printed here; only binary triples after split + direction normalization are shown. -->")
    for relation in raw_graph.relations:
        lines.append(f"- {relation_text(relation)}")
        lines.append(f"  - evidence: {getattr(relation, 'evidence_text', '')}")
        lines.append(f"  - confidence: {getattr(relation, 'confidence', '')}")

    lines.append("")
    lines.append("## VALIDATED entities")
    for entity in validated_graph.entities:
        lines.append(f"- {entity_text(entity)} | evidence: {getattr(entity, 'evidence_text', '')}")

    lines.append("")
    lines.append("## VALIDATED relations")
    for relation in validated_graph.relations:
        lines.append(f"- {relation_text(relation)}")
        lines.append(f"  - evidence: {getattr(relation, 'evidence_text', '')}")
        lines.append(f"  - confidence: {getattr(relation, 'confidence', '')}")

    lines.append("")
    lines.append("## Removed by validator")
    lines.append("### Entities")
    for item in analysis["removed_by_validator"]["entities"]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("### Relations")
    for item in analysis["removed_by_validator"]["relations"]:
        lines.append(f"- {item['triple']}")
        lines.append(f"  - evidence: {item['evidence_text']}")
        lines.append(f"  - confidence: {item['confidence']}")

    lines.append("")
    lines.append("## Zero-degree entities")
    lines.append("### RAW expanded")
    for item in analysis["zero_degree_entities"]["raw_expanded"]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("### VALIDATED")
    for item in analysis["zero_degree_entities"]["validated"]:
        lines.append(f"- {item}")

    if analysis.get("anchor_watch_diagnosis"):
        lines.append("")
        lines.append("## Anchor-watch diagnosis")
        for item in analysis["anchor_watch_diagnosis"]["terms"]:
            lines.append(f"### {item['term']}")
            lines.append(f"- diagnosis: {item['diagnosis']}")
            lines.append(f"- raw_expanded_entities: {item['raw_expanded_entities']}")
            lines.append(f"- validated_entities: {item['validated_entities']}")
            lines.append(f"- raw_expanded_anchor_term_relations: {item['raw_expanded_anchor_term_relations']}")
            lines.append(f"- validated_anchor_term_relations: {item['validated_anchor_term_relations']}")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md-path", default=None)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--id", default=None)
    parser.add_argument("--title-contains", default=None)
    parser.add_argument("--row-index", type=int, default=None)

    parser.add_argument("--output-dir", default="data/debug/extract_one_article")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-mode", default="strict_text")
    parser.add_argument("--max-tokens", type=int, default=900)

    parser.add_argument("--anchor", default="viêm da cơ địa")
    parser.add_argument(
        "--watch",
        nargs="*",
        default=["sữa thực vật", "omega-3", "vitamin D", "vitamin E", "vitamin C", "kẽm", "rau xanh", "trái cây tươi"],
    )
    parser.add_argument("--chunk-keyword", default=None)
    parser.add_argument("--limit-chunks", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"Missing OPENAI_API_KEY. Đã thử đọc {PROJECT_ROOT / '.env'}")

    if args.md_path:
        md_path = resolve_path(args.md_path)
    else:
        if not args.input_csv:
            raise ValueError("Cần --md-path hoặc --input-csv + (--id/--title-contains/--row-index)")
        md_path = find_md_path_from_csv(
            resolve_path(args.input_csv),
            article_id=args.id,
            title_contains=args.title_contains,
            row_index=args.row_index,
        )

    if not md_path.exists():
        raise FileNotFoundError(f"Không thấy markdown: {md_path}")

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    extractor = MedicalExtractor(model=model, temperature=args.temperature, prompt_mode=args.prompt_mode)
    validator = ExtractionValidator()

    article = load_markdown_article(md_path)
    chunks = chunk_article(article, max_tokens=args.max_tokens)

    if args.chunk_keyword:
        chunks = [
            chunk
            for chunk in chunks
            if contains_any(chunk.text, [args.chunk_keyword])
            or contains_any(chunk.contextualized_text, [args.chunk_keyword])
        ]

    if args.limit_chunks > 0:
        chunks = chunks[: args.limit_chunks]

    if not chunks:
        raise RuntimeError("Không có chunk nào để extract sau khi filter.")

    system_prompt = build_system_prompt(args.prompt_mode)
    write_text(out_dir / "system_prompt.txt", system_prompt)

    article_summary = {
        "md_path": str(md_path),
        "article_id": article.article_id,
        "title": article.title,
        "url": article.url,
        "source": article.source,
        "model": model,
        "max_tokens": args.max_tokens,
        "chunk_count_selected": len(chunks),
        "anchor": args.anchor,
        "watch": args.watch,
        "note": "RAW relations in reports are already expanded binary triples after relation_group split + direction normalization.",
    }
    write_json(out_dir / "article_summary.json", article_summary)

    all_records: list[dict[str, Any]] = []
    all_analyses: list[dict[str, Any]] = []
    totals = Counter()

    for index, chunk in enumerate(chunks, start=1):
        print(f"[{index}/{len(chunks)}] extract chunk_index={chunk.chunk_index} id={chunk.chunk_id}", flush=True)

        user_prompt = build_extraction_user_prompt(chunk)
        raw_graph = extractor.extract_chunk(chunk)
        validated_graph = validator.validate(raw_graph)

        analysis = analyze_chunk(
            chunk=chunk,
            raw_graph=raw_graph,
            validated_graph=validated_graph,
            anchor=args.anchor,
            watch_terms=args.watch,
        )

        chunk_dir = out_dir / f"chunk_{chunk.chunk_index:03d}_{chunk.chunk_id}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        write_text(chunk_dir / "user_prompt.txt", user_prompt)
        write_text(chunk_dir / "raw_chunk_text.md", chunk.text)
        write_text(chunk_dir / "contextualized_text.md", chunk.contextualized_text)
        write_json(chunk_dir / "raw_expanded_graph.json", graph_to_dict(raw_graph))
        write_json(chunk_dir / "validated_graph.json", graph_to_dict(validated_graph))
        write_json(chunk_dir / "analysis.json", analysis)
        write_text(
            chunk_dir / "report.md",
            render_chunk_markdown(
                chunk=chunk,
                user_prompt=user_prompt,
                raw_graph=raw_graph,
                validated_graph=validated_graph,
                analysis=analysis,
            ),
        )

        all_records.append(
            {
                "chunk": {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "article_id": chunk.article_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "heading_path": getattr(chunk, "heading_path", []),
                    "heading_context": chunk.metadata.get("heading_path_context"),
                    "token_count": chunk.token_count,
                },
                "raw_expanded_graph": graph_to_dict(raw_graph),
                "validated_graph": graph_to_dict(validated_graph),
                "analysis": analysis,
            }
        )
        all_analyses.append(analysis)

        totals["raw_expanded_entities"] += len(raw_graph.entities)
        totals["raw_expanded_relations"] += len(raw_graph.relations)
        totals["validated_entities"] += len(validated_graph.entities)
        totals["validated_relations"] += len(validated_graph.relations)

    write_json(out_dir / "all_records.json", all_records)
    write_json(out_dir / "all_analyses.json", all_analyses)

    diagnosis_counter: Counter[str] = Counter()
    for analysis in all_analyses:
        diag = analysis.get("anchor_watch_diagnosis")
        if not diag:
            continue
        for item in diag["terms"]:
            diagnosis_counter[item["diagnosis"]] += 1

    final_summary = {
        **article_summary,
        "totals": dict(totals),
        "anchor_watch_diagnosis_counts": dict(diagnosis_counter),
        "interpretation": {
            "RAW_KHONG_SINH_RELATION": "LLM không tạo relation giữa anchor và term sau khi expand relation_groups.",
            "RAW_CO_VALIDATED_DROP": "Relation có sau expand nhưng validator/schema/evidence đã drop.",
            "VALIDATED_CO_RELATION": "Relation qua validator rồi; nếu Neo4j không có thì nghi graph writer/ingest hoặc graph cũ.",
        },
    }
    write_json(out_dir / "final_summary.json", final_summary)

    print("\nDONE")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print(f"\nOutput dir: {out_dir}")


if __name__ == "__main__":
    main()
