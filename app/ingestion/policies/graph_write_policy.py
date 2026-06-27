from __future__ import annotations

import re
import unicodedata

from app.ingestion.extraction.common.extraction_models import ExtractedEntity, ExtractedGraph


def _normalize_key_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def entity_key(name: str, entity_type: str) -> tuple[str, str]:
    """Stable local key used by Neo4j and Qdrant write builders.

    This is intentionally exact-normalized-name + entity_type only. It does not
    use aliases, surface_forms, fuzzy matching, embeddings, or LLM canonicalizers.
    """
    return (_normalize_key_text(name), str(entity_type or "").strip())


def is_keepable_anchor_entity(entity: ExtractedEntity) -> bool:
    """Return whether an orphan extracted entity should still be persisted.

    Keep this policy conservative and aligned with the previous
    MedicalGraphBuilder behavior: relation endpoints are handled separately;
    orphan candidates must explicitly be anchor candidates and belong to a
    high-value medical type.
    """
    if not getattr(entity, "is_anchor_candidate", False):
        return False

    if entity.entity_type in {
        "BENH_LY",
        "THUOC_VA_HOAT_CHAT",
        "PHUONG_PHAP_CHAN_DOAN",
        "YEU_TO_BENH_SINH",
    }:
        return True

    if entity.entity_type == "BIEU_HIEN_LAM_SANG" and len(str(entity.name or "").split()) >= 2:
        return True

    return False


def entities_selected_for_graph_write(
    extracted_graph: ExtractedGraph,
) -> tuple[list[ExtractedEntity], dict[str, int]]:
    """Persist relation endpoints + high-value anchor candidates only.

    LLM may extract broad entity candidates to protect relation recall. The graph
    should receive a pruned graph, not every candidate. Relation endpoints are
    always kept; orphan candidates are kept only when explicitly marked as
    high-value anchors. Vector entity writes must use this same policy so Qdrant
    does not contain orphan entity vectors that do not exist in Neo4j.
    """
    endpoint_keys: set[tuple[str, str]] = set()
    for relation in extracted_graph.relations:
        endpoint_keys.add(entity_key(relation.subject, relation.subject_type))
        endpoint_keys.add(entity_key(relation.object, relation.object_type))

    selected: list[ExtractedEntity] = []
    seen: set[tuple[str, str]] = set()
    endpoint_count = 0
    anchor_count = 0
    pruned_count = 0

    for entity in extracted_graph.entities:
        key = entity_key(entity.name, entity.entity_type)
        if not key[0] or key in seen:
            continue

        is_endpoint = key in endpoint_keys
        is_anchor = is_keepable_anchor_entity(entity)

        if is_endpoint or is_anchor:
            selected.append(entity)
            seen.add(key)
            if is_endpoint:
                endpoint_count += 1
            elif is_anchor:
                anchor_count += 1
        else:
            pruned_count += 1

    return selected, {
        "candidate_entity_count": len(extracted_graph.entities),
        "selected_entity_count": len(selected),
        "endpoint_entity_count": endpoint_count,
        "kept_anchor_entity_count": anchor_count,
        "pruned_orphan_candidate_count": pruned_count,
    }


def graph_with_selected_entities(extracted_graph: ExtractedGraph) -> ExtractedGraph:
    """Return a copy whose entities match the graph-write policy."""
    selected, _stats = entities_selected_for_graph_write(extracted_graph)

    if hasattr(extracted_graph, "model_copy"):
        return extracted_graph.model_copy(update={"entities": selected})

    copied = extracted_graph.copy(deep=True)
    copied.entities = selected
    return copied
