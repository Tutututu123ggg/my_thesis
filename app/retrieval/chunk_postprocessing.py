from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from app.retrieval.retrieval_models import RetrievedChunk


def fact_window_parent_id(chunk: RetrievedChunk) -> str:
    """Return the retrieval parent chunk id for a fact-window chunk, if any."""
    meta = dict(chunk.metadata or {})
    parent_id = str(meta.get("parent_chunk_id") or meta.get("retrieval_chunk_id") or "").strip()
    if not parent_id:
        return ""

    is_fact_window = (
        bool(meta.get("extraction_window"))
        or meta.get("chunk_role") == "extraction_window"
        or "_fw_" in str(chunk.chunk_id or "")
    )
    if not is_fact_window:
        return ""
    if parent_id == chunk.chunk_id:
        return ""
    return parent_id


def collapse_fact_window_chunks_to_parents(
    chunks: Iterable[RetrievedChunk],
    *,
    graph_repo: Any | None = None,
) -> list[RetrievedChunk]:
    """Collapse child fact-window hits to their parent retrieval chunk.

    Extraction fact windows are useful for graph scoring, but they should not
    occupy separate final-context/UI slots. This function keeps the highest
    scoring evidence, maps child hits back to parent chunks, and records child
    provenance in metadata for debugging.
    """
    items = [chunk for chunk in chunks if chunk is not None]
    if not items:
        return []

    parent_ids = {fact_window_parent_id(chunk) for chunk in items}
    parent_ids.discard("")

    by_chunk_id: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in items if chunk.chunk_id}
    missing_parent_ids = [parent_id for parent_id in parent_ids if parent_id not in by_chunk_id]
    fetched_parents: dict[str, RetrievedChunk] = {}

    if missing_parent_ids and graph_repo is not None and hasattr(graph_repo, "get_chunks_by_ids"):
        try:
            for node in graph_repo.get_chunks_by_ids(missing_parent_ids):
                fetched_parents[node.chunk_id] = _chunk_from_parent_node(node)
        except Exception:
            # Retrieval must not fail just because the presentation collapse
            # cannot fetch a parent. In that case we keep the child hit.
            fetched_parents = {}

    by_parent_key: dict[str, RetrievedChunk] = {}
    child_ids_by_parent: dict[str, list[str]] = defaultdict(list)
    child_sources_by_parent: dict[str, list[str]] = defaultdict(list)
    child_scores_by_parent: dict[str, list[float]] = defaultdict(list)

    for chunk in items:
        parent_id = fact_window_parent_id(chunk)
        key = parent_id or chunk.chunk_id

        representative = chunk
        if parent_id:
            representative = by_chunk_id.get(parent_id) or fetched_parents.get(parent_id) or chunk
            child_ids_by_parent[key].append(chunk.chunk_id)
            child_sources_by_parent[key].append(str(chunk.source))
            child_scores_by_parent[key].append(float(chunk.score or 0.0))

        old = by_parent_key.get(key)
        if old is None:
            # Copy so that later metadata/score edits do not mutate the same
            # object shared by branch internals.
            by_parent_key[key] = _copy_chunk(representative)
        else:
            if float(representative.score or 0.0) > float(old.score or 0.0):
                # Keep parent text/identity if old is already the parent, but
                # allow stronger score/source from the child evidence.
                old.score = float(representative.score or 0.0)
                old.source = representative.source
            old.metadata = {**dict(old.metadata or {}), **dict(representative.metadata or {})}

        current = by_parent_key[key]
        if parent_id:
            current.chunk_id = parent_id if representative.chunk_id == parent_id else current.chunk_id
            # If parent fetch succeeded, current already contains parent text.
            # Otherwise keep the child text but mark the unresolved collapse.
            current.metadata = {
                **dict(current.metadata or {}),
                "parent_chunk_collapse": True,
                "retrieval_parent_chunk_id": parent_id,
            }
            if representative.chunk_id != parent_id:
                current.metadata["parent_chunk_collapse_parent_missing"] = True

    result: list[RetrievedChunk] = []
    for key, chunk in by_parent_key.items():
        child_ids = _dedupe(child_ids_by_parent.get(key, []))
        if child_ids:
            max_child_score = max(child_scores_by_parent.get(key, [float(chunk.score or 0.0)]))
            chunk.score = max(float(chunk.score or 0.0), max_child_score)
            chunk.metadata = {
                **dict(chunk.metadata or {}),
                "parent_chunk_collapse": True,
                "collapsed_child_chunk_ids": child_ids,
                "collapsed_child_sources": _dedupe(child_sources_by_parent.get(key, [])),
                "collapsed_child_count": len(child_ids),
                "collapsed_child_max_score": max_child_score,
            }
        result.append(chunk)

    return sorted(result, key=lambda item: float(item.score or 0.0), reverse=True)


def _chunk_from_parent_node(node: Any) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=node.chunk_id,
        score=float(getattr(node, "score", 0.0) or 0.0),
        source="graph_expansion",
        article_id=node.article_id,
        title=node.title,
        section=node.section or "",
        subsection=node.subsection or "",
        source_url=node.source_url,
        text=node.text,
        contextualized_text=node.contextualized_text or node.text,
        metadata=dict(node.metadata or {}),
    )


def _copy_chunk(chunk: RetrievedChunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        score=float(chunk.score or 0.0),
        source=chunk.source,
        article_id=chunk.article_id,
        title=chunk.title,
        section=chunk.section,
        subsection=chunk.subsection,
        source_url=chunk.source_url,
        text=chunk.text,
        contextualized_text=chunk.contextualized_text,
        metadata=dict(chunk.metadata or {}),
    )


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = str(item or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
