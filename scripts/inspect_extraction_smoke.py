from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROMO_CUES = (
    "bệnh viện",
    "đội ngũ bác sĩ",
    "khách hàng",
    "hotline",
    "tổng đài",
    "đặt lịch",
    "dịch vụ",
    "thăm khám",
    "tư vấn",
    "khoa da liễu",
)
SCHEMA_ENUM_NAMES = {
    "BENH_LY",
    "BIEU_HIEN_LAM_SANG",
    "YEU_TO_BENH_SINH",
    "YEU_TO_NGOAI_SINH",
    "PHUONG_PHAP_CHAN_DOAN",
    "CAN_THIEP_Y_TE",
    "THUOC_VA_HOAT_CHAT",
    "VI_TRI_GIAI_PHAU",
    "DOI_TUONG",
}


def norm(value: str) -> str:
    text = str(value or "").strip().lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def relation_key(relation: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        norm(relation.get("subject", "")),
        str(relation.get("subject_type", "")),
        str(relation.get("relation_type", "")),
        norm(relation.get("object", "")),
        str(relation.get("object_type", "")),
    )


def entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return norm(entity.get("name", "")), str(entity.get("entity_type", ""))


def has_promo_evidence(relation: dict[str, Any]) -> bool:
    evidence = str(relation.get("evidence_text") or "").lower()
    return any(cue in evidence for cue in PROMO_CUES)


def is_schema_enum_entity(entity: dict[str, Any]) -> bool:
    return str(entity.get("name", "")).strip() in SCHEMA_ENUM_NAMES


def is_suspicious_relation(relation: dict[str, Any]) -> str | None:
    if has_promo_evidence(relation):
        return "promo_evidence"
    if str(relation.get("subject", "")).strip() in SCHEMA_ENUM_NAMES:
        return "schema_enum_subject"
    if str(relation.get("object", "")).strip() in SCHEMA_ENUM_NAMES:
        return "schema_enum_object"
    support = str(relation.get("support", "")) or " ".join(relation.get("support_flags", []) or [])
    if "WEAK_ENDPOINT_MATCH" in support:
        return "weak_endpoint"
    if "object_not_in_evidence" in support:
        return "object_not_in_evidence"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect smoke extraction JSONL for final ingest checks.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--focus-title-contains", default="")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    records = load_records(args.jsonl)
    focus = args.focus_title_contains.strip().lower()
    if focus:
        records = [r for r in records if focus in str(r.get("title", "")).lower()]

    all_entities = [e for r in records for e in r.get("validated_entities", [])]
    all_relations = [rel for r in records for rel in r.get("validated_relations", [])]
    endpoint_keys = set()
    for rel in all_relations:
        endpoint_keys.add((norm(rel.get("subject", "")), str(rel.get("subject_type", ""))))
        endpoint_keys.add((norm(rel.get("object", "")), str(rel.get("object_type", ""))))

    entity_counter = Counter(entity_key(e) for e in all_entities)
    orphan_entities = [e for e in all_entities if entity_key(e) not in endpoint_keys]
    schema_entities = [e for e in all_entities if is_schema_enum_entity(e)]
    suspect_relations: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for record in records:
        for rel in record.get("validated_relations", []):
            reason = is_suspicious_relation(rel)
            if reason:
                suspect_relations.append((reason, rel, record))

    by_article = defaultdict(lambda: {"chunks": 0, "entities": 0, "relations": 0, "suspects": 0})
    for record in records:
        title = record.get("title", "")
        by_article[title]["chunks"] += 1
        by_article[title]["entities"] += len(record.get("validated_entities", []))
        by_article[title]["relations"] += len(record.get("validated_relations", []))
        for rel in record.get("validated_relations", []):
            if is_suspicious_relation(rel):
                by_article[title]["suspects"] += 1

    print("[SMOKE INSPECT]")
    print(f"records/chunks     : {len(records)}")
    print(f"validated_entities : {len(all_entities)}")
    print(f"validated_relations: {len(all_relations)}")
    print(f"endpoint_keys      : {len(endpoint_keys)}")
    print(f"orphan_entities    : {len(orphan_entities)}")
    print(f"schema_enum_entities: {len(schema_entities)}")
    print(f"suspect_relations  : {len(suspect_relations)}")
    print()

    print("[BY ARTICLE]")
    for title, stats in by_article.items():
        print(f"- {title}")
        print(
            f"  chunks={stats['chunks']} entities={stats['entities']} "
            f"relations={stats['relations']} suspects={stats['suspects']}"
        )
    print()

    if schema_entities:
        print("[SCHEMA ENUM ENTITY NAMES]")
        for e in schema_entities[: args.limit]:
            print(f"- {e.get('name')} [{e.get('entity_type')}] evidence={e.get('evidence_text','')[:160]}")
        print()

    if orphan_entities:
        print("[ORPHAN ENTITY CANDIDATES IN VALIDATED JSONL]")
        for e in orphan_entities[: args.limit]:
            anchor = e.get("is_anchor_candidate", False)
            reason = e.get("anchor_reason", "")
            print(f"- {e.get('name')} [{e.get('entity_type')}] anchor={anchor} reason={reason}")
        print()

    if suspect_relations:
        print("[SUSPECT RELATIONS]")
        for reason, rel, record in suspect_relations[: args.limit]:
            print(f"- reason={reason} | title={record.get('title')}")
            print(
                f"  {rel.get('subject')} [{rel.get('subject_type')}] --{rel.get('relation_type')}--> "
                f"{rel.get('object')} [{rel.get('object_type')}]"
            )
            print(f"  evidence={str(rel.get('evidence_text',''))[:240]}")


if __name__ == "__main__":
    main()
