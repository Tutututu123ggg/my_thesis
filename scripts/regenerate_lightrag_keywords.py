from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from app.evaluation.retriever_factory import make_embedding_service, make_graph_repo, make_vector_repo
from app.infrastructure.vector_database import (
    ENTITY_COLLECTION,
    RELATION_COLLECTION,
    EntityVectorPayload,
    RelationVectorPayload,
    VectorPoint,
    make_qdrant_point_id,
)
from app.infrastructure.graph_database import EntityNode, MedicalRelationView
from app.reasoning import ChatLLMClient, ChatLLMConfig
from app.retrieval.lightrag_keyword_profiles import KeywordProfileGenerator


DEFAULT_OUTPUT = Path("data/maintenance/lightrag_keyword_regen.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate LightRAG-style entity profiles and relation/global keywords "
            "for existing Neo4j graph nodes/edges, then optionally re-embed Qdrant vectors."
        )
    )
    parser.add_argument("--target", choices=["entities", "relations", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Max items per target. 0 means all.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="Generate and write report only; do not update Neo4j/Qdrant.")
    parser.add_argument("--skip-qdrant", action="store_true", help="Update Neo4j only; do not re-embed Qdrant vectors.")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic fallback profiles only.")
    parser.add_argument("--llm-provider", choices=["openai", "ollama"], default=os.getenv("KEYWORD_REGEN_PROVIDER", "openai"))
    parser.add_argument("--llm-model", default=os.getenv("KEYWORD_REGEN_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_llm_client(args: argparse.Namespace) -> ChatLLMClient | None:
    if args.no_llm:
        return None
    return ChatLLMClient(
        ChatLLMConfig(
            provider=args.llm_provider,
            model=args.llm_model,
            temperature=args.llm_temperature,
            timeout_seconds=int(os.getenv("KEYWORD_REGEN_TIMEOUT_SECONDS", "180")),
            base_url=os.getenv("OPENAI_BASE_URL") if args.llm_provider == "openai" else os.getenv("OLLAMA_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY") if args.llm_provider == "openai" else None,
        )
    )


def entity_vector_text(entity: EntityNode, profile_text: str, local_keys: list[str], global_keys: list[str]) -> str:
    parts = [
        f"Tên thực thể: {entity.name}",
        f"Loại thực thể: {entity.entity_type}",
    ]
    if profile_text:
        parts.append(f"Profile: {profile_text}")
    if entity.description:
        parts.append(f"Mô tả: {entity.description}")
    if local_keys:
        parts.append("Local keys: " + "; ".join(local_keys))
    if global_keys:
        parts.append("Global keys: " + "; ".join(global_keys))
    if entity.surface_forms:
        parts.append("Surface forms: " + "; ".join(entity.surface_forms[:8]))
    return "\n".join(parts)


def relation_vector_text(relation: MedicalRelationView, keywords: list[str]) -> str:
    parts = [
        f"Quan hệ: {relation.relation_type}",
        f"Subject: {relation.subject_name}",
        f"Object: {relation.object_name}",
    ]
    if keywords:
        parts.append("Global keywords: " + "; ".join(keywords))
    if relation.description:
        parts.append(f"Mô tả: {relation.description}")
    if relation.evidence_text:
        parts.append(f"Bằng chứng: {relation.evidence_text}")
    return "\n".join(parts)


def iter_entity_pages(graph_repo, page_size: int, offset: int, limit: int):
    produced = 0
    current_offset = offset
    while True:
        remaining = None if limit <= 0 else max(limit - produced, 0)
        if remaining == 0:
            break
        page_limit = page_size if remaining is None else min(page_size, remaining)
        rows = graph_repo.list_entities_for_lightrag_keyword_regen(limit=page_limit, offset=current_offset)
        if not rows:
            break
        yield rows
        produced += len(rows)
        current_offset += len(rows)


def iter_relation_pages(graph_repo, page_size: int, offset: int, limit: int):
    produced = 0
    current_offset = offset
    while True:
        remaining = None if limit <= 0 else max(limit - produced, 0)
        if remaining == 0:
            break
        page_limit = page_size if remaining is None else min(page_size, remaining)
        rows = graph_repo.list_relations_for_lightrag_keyword_regen(limit=page_limit, offset=current_offset)
        if not rows:
            break
        yield rows
        produced += len(rows)
        current_offset += len(rows)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def regenerate_entities(args, graph_repo, vector_repo, embedder, profile_generator) -> int:
    total = 0
    for page in iter_entity_pages(graph_repo, args.page_size, args.offset, args.limit):
        report_rows: list[dict] = []
        qdrant_points: list[VectorPoint] = []
        vector_texts: list[str] = []
        vector_payloads: list[EntityVectorPayload] = []

        for entity, evidence_texts in page:
            profile = profile_generator.generate_entity_profile(
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                surface_forms=entity.surface_forms,
                evidence_texts=evidence_texts,
            )
            vtext = entity_vector_text(entity, profile.profile_text, profile.local_keys, profile.global_keys)
            payload = EntityVectorPayload(
                entity_id=entity.entity_id,
                name=entity.name,
                normalized_name=entity.normalized_name,
                entity_type=entity.entity_type,
                aliases=entity.aliases,
                surface_forms=entity.surface_forms,
                description=entity.description,
                profile_text=profile.profile_text,
                local_keys=profile.local_keys,
                global_keys=profile.global_keys,
            )
            report_rows.append(
                {
                    "target": "entity",
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "old_profile_text": entity.profile_text,
                    "old_local_keys": entity.local_keys,
                    "old_global_keys": entity.global_keys,
                    "new_profile_text": profile.profile_text,
                    "new_local_keys": profile.local_keys,
                    "new_global_keys": profile.global_keys,
                }
            )
            if not args.dry_run:
                graph_repo.update_entity_lightrag_profile(
                    entity_id=entity.entity_id,
                    profile_text=profile.profile_text,
                    local_keys=profile.local_keys,
                    global_keys=profile.global_keys,
                )
            if not args.dry_run and not args.skip_qdrant:
                vector_texts.append(vtext)
                vector_payloads.append(payload)

        if vector_texts:
            vectors = embedder.embed_texts(vector_texts, batch_size=args.batch_size)
            for payload, vector in zip(vector_payloads, vectors, strict=True):
                qdrant_points.append(
                    VectorPoint(
                        point_id=make_qdrant_point_id(f"entity::{payload.entity_id}"),
                        vector=vector,
                        payload=payload.model_dump(),
                    )
                )
            vector_repo.upsert_points(ENTITY_COLLECTION, qdrant_points)

        append_jsonl(args.output, report_rows)
        total += len(page)
        print(f"[entities] processed={total}")
    return total


def regenerate_relations(args, graph_repo, vector_repo, embedder, profile_generator) -> int:
    total = 0
    for page in iter_relation_pages(graph_repo, args.page_size, args.offset, args.limit):
        report_rows: list[dict] = []
        qdrant_points: list[VectorPoint] = []
        vector_texts: list[str] = []
        vector_payloads: list[RelationVectorPayload] = []

        for relation in page:
            profile = profile_generator.generate_relation_profile(
                relation_type=relation.relation_type,
                subject_name=relation.subject_name,
                subject_type=relation.subject_type,
                object_name=relation.object_name,
                object_type=relation.object_type,
                description=relation.description,
                evidence_text=relation.evidence_text,
            )
            vtext = relation_vector_text(relation, profile.keywords)
            payload = RelationVectorPayload(
                relation_id=relation.relation_id,
                relation_type=relation.relation_type,
                subject_entity_id=relation.subject_entity_id,
                subject_name=relation.subject_name,
                subject_type=relation.subject_type,
                object_entity_id=relation.object_entity_id,
                object_name=relation.object_name,
                object_type=relation.object_type,
                description=relation.description,
                keywords=profile.keywords,
                evidence_text=relation.evidence_text,
                evidence_chunk_ids=relation.evidence_chunk_ids,
                chunk_id=(relation.evidence_chunk_ids[0] if relation.evidence_chunk_ids else None),
                section=relation.section,
                source_url=relation.source_url,
            )
            report_rows.append(
                {
                    "target": "relation",
                    "relation_id": relation.relation_id,
                    "relation_type": relation.relation_type,
                    "subject": relation.subject_name,
                    "object": relation.object_name,
                    "old_keywords": relation.keywords,
                    "new_keywords": profile.keywords,
                    "evidence_text": relation.evidence_text,
                }
            )
            if not args.dry_run:
                graph_repo.update_relation_lightrag_keywords(
                    relation_id=relation.relation_id,
                    keywords=profile.keywords,
                )
            if not args.dry_run and not args.skip_qdrant:
                vector_texts.append(vtext)
                vector_payloads.append(payload)

        if vector_texts:
            vectors = embedder.embed_texts(vector_texts, batch_size=args.batch_size)
            for payload, vector in zip(vector_payloads, vectors, strict=True):
                qdrant_points.append(
                    VectorPoint(
                        point_id=make_qdrant_point_id(f"relation::{payload.relation_id}"),
                        vector=vector,
                        payload=payload.model_dump(),
                    )
                )
            vector_repo.upsert_points(RELATION_COLLECTION, qdrant_points)

        append_jsonl(args.output, report_rows)
        total += len(page)
        print(f"[relations] processed={total}")
    return total


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.output.exists():
        args.output.unlink()

    llm_client = build_llm_client(args)
    profile_generator = KeywordProfileGenerator(llm_client=llm_client)

    neo4j_client, graph_repo = make_graph_repo()
    embedder = make_embedding_service()
    qdrant_client, vector_repo = make_vector_repo(embedder)

    try:
        print(
            f"[LightRAG keyword regeneration] target={args.target} dry_run={args.dry_run} "
            f"llm={'off' if args.no_llm else args.llm_provider + ':' + args.llm_model}"
        )
        entity_count = 0
        relation_count = 0
        if args.target in {"entities", "all"}:
            entity_count = regenerate_entities(args, graph_repo, vector_repo, embedder, profile_generator)
        if args.target in {"relations", "all"}:
            relation_count = regenerate_relations(args, graph_repo, vector_repo, embedder, profile_generator)
        print(f"[DONE] entities={entity_count} relations={relation_count} report={args.output}")
    finally:
        qdrant_client.close()
        neo4j_client.close()


if __name__ == "__main__":
    main()
