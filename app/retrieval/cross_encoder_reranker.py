from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any, Iterable

from app.retrieval.retrieval_models import RetrievalResult, RetrievedChunk


@dataclass
class CrossEncoderRerankingConfig:
    """Configuration for safe late-stage cross-encoder reranking.

    This component is deliberately feature-free: it does not use disease lists,
    symptom lists, stopwords, query operator cues, source URL exceptions, or qid
    patches. It only scores the semantic pair ``(query, chunk_text)`` and then
    calibrates that score with the original retrieval score.

    The default weights are intentionally conservative. In this project the
    vector / graph retrievers are the recall backbone; the cross-encoder is a
    final calibration signal and should not be able to wipe out the strongest
    pre-rerank evidence by itself.
    """

    model_name: str = ""
    candidate_top_k: int = 50
    final_top_k: int = 12
    batch_size: int = 16
    max_text_chars: int = 2400
    cross_encoder_weight: float = 0.35
    retrieval_weight: float = 0.65
    enable_source_diversity: bool = True
    max_chunks_per_source: int = 3
    fail_open: bool = True
    device: str = ""

    # Backbone preservation. This prevents a small/noisy reranker from demoting
    # the most trusted vector/mixed candidates in cases where pre-rerank already
    # has a strong exact source match. It is still generic and source-agnostic.
    preserve_pre_rerank_top_k: int = 2
    preserve_pre_rerank_at_top: bool = True

    # Safety gate for ``preserve_pre_rerank_at_top``. A protected pre-rerank
    # chunk is guaranteed to stay in the final window, but it is forced into
    # the first positions only when the CE does not strongly disagree. This
    # avoids q095-like failures where mixed candidate pre-rank protects a
    # broad-but-wrong article above a CE-supported specific article. Set 0 to
    # disable the gate and always force protected chunks to the top.
    preserve_at_top_max_cross_encoder_rank: int = 5


class CrossEncoderReranker:
    """Late reranker for a candidate pool of RetrievedChunk objects.

    The model is loaded lazily so normal vector/graph experiments do not pay the
    cost unless ``vector_ce`` or ``mixed_ce`` is actually used.
    """

    def __init__(self, config: CrossEncoderRerankingConfig | None = None):
        self.config = config or CrossEncoderRerankingConfig()
        self.model_name = (
            self.config.model_name
            or os.getenv("CROSS_ENCODER_MODEL")
            or "BAAI/bge-reranker-v2-m3"
        )
        self._model: Any | None = None
        self._model_error: str = ""

    def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        method_name: str,
        final_top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        candidates = self._dedupe_chunks(chunks)[: max(1, int(self.config.candidate_top_k))]
        target_top_k = max(1, int(final_top_k or self.config.final_top_k))
        if not candidates:
            return [], {
                "enabled": True,
                "available": False,
                "reason": "no_candidates",
                "method_name": method_name,
                "model_name": self.model_name,
            }

        pre_ranked = self._attach_pre_rerank_metadata(candidates)

        model = self._get_model()
        if model is None:
            if not self.config.fail_open:
                raise RuntimeError(self._model_error or "Cross-encoder model is unavailable.")
            return self._fallback(
                pre_ranked,
                target_top_k=target_top_k,
                method_name=method_name,
                reason=self._model_error or "model_unavailable",
            )

        pairs = [(query, self._chunk_text(chunk)) for chunk in pre_ranked]
        try:
            raw_scores = self._predict(model, pairs)
        except Exception as exc:  # pragma: no cover - depends on local model/runtime.
            self._model_error = f"{type(exc).__name__}: {exc}"
            if not self.config.fail_open:
                raise
            return self._fallback(
                pre_ranked,
                target_top_k=target_top_k,
                method_name=method_name,
                reason=self._model_error,
            )

        if len(raw_scores) != len(pre_ranked):
            reason = f"cross_encoder_returned_{len(raw_scores)}_scores_for_{len(pre_ranked)}_chunks"
            if not self.config.fail_open:
                raise RuntimeError(reason)
            return self._fallback(
                pre_ranked,
                target_top_k=target_top_k,
                method_name=method_name,
                reason=reason,
            )

        reranked = self._combine_scores(pre_ranked, raw_scores, method_name=method_name)
        selected, diversity_debug = self._apply_source_diversity(reranked, target_top_k=target_top_k)
        selected, preserve_debug = self._apply_pre_rerank_preservation(
            reranked=reranked,
            selected=selected,
            target_top_k=target_top_k,
        )

        for rank, chunk in enumerate(selected, start=1):
            chunk.metadata = {
                **dict(chunk.metadata or {}),
                "final_rerank_rank": rank,
            }

        return selected, {
            "enabled": True,
            "available": True,
            "method_name": method_name,
            "model_name": self.model_name,
            "candidate_count": len(pre_ranked),
            "returned_count": len(selected),
            "candidate_top_k": self.config.candidate_top_k,
            "final_top_k": target_top_k,
            "batch_size": self.config.batch_size,
            "max_text_chars": self.config.max_text_chars,
            "cross_encoder_weight": self.config.cross_encoder_weight,
            "retrieval_weight": self.config.retrieval_weight,
            "score_calibration": "pool_minmax(cross_encoder_score) + pool_minmax(pre_rerank_score)",
            "source_diversity": diversity_debug,
            "pre_rerank_preservation": preserve_debug,
            "preserve_at_top_max_cross_encoder_rank": self.config.preserve_at_top_max_cross_encoder_rank,
            "top_candidates": [
                {
                    "rank": i,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "source_url": chunk.source_url,
                    "cross_encoder_score": chunk.metadata.get("cross_encoder_score"),
                    "pre_rerank_rank": chunk.metadata.get("pre_rerank_rank"),
                    "pre_rerank_score": chunk.metadata.get("pre_rerank_score"),
                    "final_combined_score": chunk.metadata.get("final_combined_score"),
                    "pre_rerank_source": chunk.metadata.get("pre_rerank_source"),
                    "rerank_preserved_by_backbone": chunk.metadata.get("rerank_preserved_by_backbone"),
                }
                for i, chunk in enumerate(selected[:10], start=1)
            ],
        }

    def _get_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if self._model_error:
            return None

        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover - environment dependent.
            self._model_error = f"sentence_transformers.CrossEncoder import failed: {type(exc).__name__}: {exc}"
            return None

        try:
            kwargs: dict[str, Any] = {}
            device = self.config.device or os.getenv("CROSS_ENCODER_DEVICE") or ""
            if device:
                kwargs["device"] = device
            try:
                self._model = CrossEncoder(self.model_name, **kwargs)
            except TypeError:
                # Older sentence-transformers versions may accept fewer kwargs.
                self._model = CrossEncoder(self.model_name)
            return self._model
        except Exception as exc:  # pragma: no cover - model download/runtime dependent.
            self._model_error = f"CrossEncoder load failed for '{self.model_name}': {type(exc).__name__}: {exc}"
            return None

    def _predict(self, model: Any, pairs: list[tuple[str, str]]) -> list[float]:
        try:
            values = model.predict(
                pairs,
                batch_size=max(1, int(self.config.batch_size)),
                show_progress_bar=False,
            )
        except TypeError:
            values = model.predict(pairs, batch_size=max(1, int(self.config.batch_size)))

        if hasattr(values, "tolist"):
            values = values.tolist()
        result: list[float] = []
        for value in values:
            if isinstance(value, (list, tuple)):
                # Some classifiers return a vector. Use the last logit as the
                # relevance logit, which is the common convention for pairwise
                # relevance heads.
                value = value[-1] if value else 0.0
            result.append(float(value))
        return result

    def _combine_scores(
        self,
        chunks: list[RetrievedChunk],
        cross_scores: list[float],
        *,
        method_name: str,
    ) -> list[RetrievedChunk]:
        retrieval_scores = [float(chunk.score or 0.0) for chunk in chunks]
        retrieval_norm = self._minmax_values(retrieval_scores)
        cross_norm = self._minmax_values(cross_scores)

        ce_rank_lookup = self._rank_lookup(cross_scores)
        combined_items: list[RetrievedChunk] = []

        ce_weight = max(0.0, float(self.config.cross_encoder_weight))
        retrieval_weight = max(0.0, float(self.config.retrieval_weight))
        weight_sum = ce_weight + retrieval_weight
        if weight_sum <= 0.0:
            ce_weight, retrieval_weight, weight_sum = 0.35, 0.65, 1.0
        ce_weight /= weight_sum
        retrieval_weight /= weight_sum

        for idx, chunk in enumerate(chunks):
            final_score = ce_weight * cross_norm[idx] + retrieval_weight * retrieval_norm[idx]
            metadata = {
                **dict(chunk.metadata or {}),
                "cross_encoder_score": float(cross_scores[idx]),
                "cross_encoder_score_norm": float(cross_norm[idx]),
                "cross_encoder_rank": int(ce_rank_lookup[idx]),
                "retrieval_score_norm": float(retrieval_norm[idx]),
                "final_combined_score": float(final_score),
                "rerank_model_name": self.model_name,
                "rerank_method_name": method_name,
                "cross_encoder_available": True,
                "cross_encoder_weight": ce_weight,
                "retrieval_weight": retrieval_weight,
                "rerank_preserved_by_backbone": False,
            }
            combined_items.append(replace(chunk, score=float(final_score), metadata=metadata))

        return sorted(
            combined_items,
            key=lambda item: (
                float(item.metadata.get("final_combined_score", item.score) or 0.0),
                -int(item.metadata.get("pre_rerank_rank", 10**9) or 10**9),
            ),
            reverse=True,
        )

    def _apply_pre_rerank_preservation(
        self,
        *,
        reranked: list[RetrievedChunk],
        selected: list[RetrievedChunk],
        target_top_k: int,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        preserve_k = max(0, int(self.config.preserve_pre_rerank_top_k or 0))
        if preserve_k <= 0:
            return selected[:target_top_k], {"enabled": False}

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        protected = [
            chunk for chunk in sorted(
                reranked,
                key=lambda item: int(item.metadata.get("pre_rerank_rank", 10**9) or 10**9),
            )
            if int(chunk.metadata.get("pre_rerank_rank", 10**9) or 10**9) <= preserve_k
        ]
        protected = protected[:preserve_k]
        protected_ids = {chunk.chunk_id for chunk in protected}
        selected_ids_before = {chunk.chunk_id for chunk in selected}
        inserted_ids = [chunk.chunk_id for chunk in protected if chunk.chunk_id not in selected_ids_before]

        max_ce_rank = max(0, int(self.config.preserve_at_top_max_cross_encoder_rank or 0))

        def qualifies_for_top(chunk: RetrievedChunk) -> bool:
            if not self.config.preserve_pre_rerank_at_top:
                return False
            if max_ce_rank <= 0:
                return True
            try:
                ce_rank = int(chunk.metadata.get("cross_encoder_rank", 10**9) or 10**9)
            except (TypeError, ValueError):
                ce_rank = 10**9
            return ce_rank <= max_ce_rank

        top_protected_ids = {chunk.chunk_id for chunk in protected if qualifies_for_top(chunk)}

        def mark(chunk: RetrievedChunk, preserved: bool) -> RetrievedChunk:
            meta = dict(chunk.metadata or {})
            if preserved:
                forced_top = chunk.chunk_id in top_protected_ids
                meta.update({
                    "rerank_preserved_by_backbone": True,
                    "rerank_preserve_reason": "pre_rerank_top_k",
                    "rerank_preserve_top_k": preserve_k,
                    "rerank_preserve_at_top": bool(forced_top),
                    "rerank_preserve_at_top_qualified": bool(forced_top),
                    "rerank_preserve_at_top_max_ce_rank": max_ce_rank,
                })
                if self.config.preserve_pre_rerank_at_top and not forced_top:
                    meta["rerank_preserve_not_forced_reason"] = "cross_encoder_rank_too_low"
            return replace(chunk, metadata=meta)

        protected_marked = [mark(reranked_by_id.get(chunk.chunk_id, chunk), True) for chunk in protected]
        selected_marked = [mark(chunk, chunk.chunk_id in protected_ids) for chunk in selected]

        by_id = {chunk.chunk_id: chunk for chunk in selected_marked}
        merged = list(selected_marked)
        for chunk in protected_marked:
            if chunk.chunk_id not in by_id:
                merged.append(chunk)

        forced_top = [chunk for chunk in protected_marked if chunk.chunk_id in top_protected_ids]
        forced_top_ids = {chunk.chunk_id for chunk in forced_top}
        rest = [chunk for chunk in merged if chunk.chunk_id not in forced_top_ids]
        rest = sorted(
            rest,
            key=lambda item: (
                float(item.metadata.get("final_combined_score", item.score) or 0.0),
                -int(item.metadata.get("pre_rerank_rank", 10**9) or 10**9),
            ),
            reverse=True,
        )
        final = [*forced_top, *rest]

        # If inserting protected chunks overfills the window, remove the lowest
        # non-protected tail chunks first. Protected chunks are generic top-N
        # backbone candidates, not source/qid-specific exceptions.
        if len(final) > target_top_k:
            kept: list[RetrievedChunk] = []
            non_protected_tail: list[RetrievedChunk] = []
            for chunk in final:
                if chunk.chunk_id in protected_ids:
                    kept.append(chunk)
                else:
                    non_protected_tail.append(chunk)
            final = [*kept, *non_protected_tail][:target_top_k]

        return final[:target_top_k], {
            "enabled": True,
            "preserve_pre_rerank_top_k": preserve_k,
            "preserve_pre_rerank_at_top": bool(self.config.preserve_pre_rerank_at_top),
            "preserve_at_top_max_cross_encoder_rank": max_ce_rank,
            "protected_chunk_ids": [chunk.chunk_id for chunk in protected],
            "top_forced_chunk_ids": [chunk.chunk_id for chunk in forced_top],
            "window_only_protected_chunk_ids": [
                chunk.chunk_id for chunk in protected if chunk.chunk_id not in top_protected_ids
            ],
            "inserted_chunk_ids": inserted_ids,
            "selected_count_before": len(selected),
            "selected_count_after": len(final[:target_top_k]),
        }

    def _attach_pre_rerank_metadata(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        result: list[RetrievedChunk] = []
        for rank, chunk in enumerate(chunks, start=1):
            result.append(
                replace(
                    chunk,
                    metadata={
                        **dict(chunk.metadata or {}),
                        "pre_rerank_score": float(chunk.score or 0.0),
                        "pre_rerank_rank": rank,
                        "pre_rerank_source": str(chunk.source),
                    },
                )
            )
        return result

    def _fallback(
        self,
        chunks: list[RetrievedChunk],
        *,
        target_top_k: int,
        method_name: str,
        reason: str,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        selected = sorted(chunks, key=lambda item: item.score, reverse=True)[:target_top_k]
        preserve_k = max(0, int(self.config.preserve_pre_rerank_top_k or 0))
        fallback_chunks: list[RetrievedChunk] = []
        for rank, chunk in enumerate(selected, start=1):
            pre_rank = int(chunk.metadata.get("pre_rerank_rank", rank) or rank)
            metadata = {
                **dict(chunk.metadata or {}),
                "cross_encoder_score": None,
                "cross_encoder_rank": None,
                "final_combined_score": float(chunk.score or 0.0),
                "rerank_model_name": self.model_name,
                "rerank_method_name": method_name,
                "cross_encoder_available": False,
                "cross_encoder_error": reason,
                "final_rerank_rank": rank,
                "rerank_preserved_by_backbone": pre_rank <= preserve_k,
                "rerank_preserve_reason": "fallback_pre_rerank_order" if pre_rank <= preserve_k else "",
            }
            fallback_chunks.append(replace(chunk, metadata=metadata))

        return fallback_chunks, {
            "enabled": True,
            "available": False,
            "method_name": method_name,
            "model_name": self.model_name,
            "candidate_count": len(chunks),
            "returned_count": len(fallback_chunks),
            "candidate_top_k": self.config.candidate_top_k,
            "final_top_k": target_top_k,
            "fallback": "pre_rerank_order",
            "reason": reason,
            "pre_rerank_preservation": {
                "enabled": preserve_k > 0,
                "preserve_pre_rerank_top_k": preserve_k,
            },
        }

    def _dedupe_chunks(self, chunks: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
        by_id: dict[str, RetrievedChunk] = {}
        first_order: dict[str, int] = {}
        for index, chunk in enumerate(chunks):
            if not chunk.chunk_id:
                continue
            first_order.setdefault(chunk.chunk_id, index)
            old = by_id.get(chunk.chunk_id)
            if old is None or float(chunk.score or 0.0) > float(old.score or 0.0):
                by_id[chunk.chunk_id] = chunk
        return sorted(
            by_id.values(),
            key=lambda item: (float(item.score or 0.0), -first_order.get(item.chunk_id, 10**9)),
            reverse=True,
        )

    def _chunk_text(self, chunk: RetrievedChunk) -> str:
        parts = [
            chunk.title or "",
            chunk.section or "",
            chunk.subsection or "",
            chunk.contextualized_text or chunk.text or "",
        ]
        text = "\n".join(part.strip() for part in parts if part and part.strip())
        text = " ".join(text.split())
        limit = max(256, int(self.config.max_text_chars or 2400))
        return text[:limit]

    def _minmax_values(self, values: list[float]) -> list[float]:
        if not values:
            return []
        clean = [float(v) if math.isfinite(float(v)) else 0.0 for v in values]
        lo = min(clean)
        hi = max(clean)
        if hi <= lo:
            # Ties still need a stable weak signal so the original order remains
            # visible in the combined score.
            n = len(clean)
            if n <= 1:
                return [1.0]
            return [(n - idx) / n for idx in range(n)]
        return [(value - lo) / (hi - lo) for value in clean]

    def _rank_lookup(self, values: list[float]) -> dict[int, int]:
        order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
        ranks: dict[int, int] = {}
        for rank, idx in enumerate(order, start=1):
            ranks[idx] = rank
        return ranks

    def _apply_source_diversity(
        self,
        chunks: list[RetrievedChunk],
        *,
        target_top_k: int,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        if not self.config.enable_source_diversity:
            return chunks[:target_top_k], {"enabled": False}

        max_per_source = max(1, int(self.config.max_chunks_per_source or 1))
        selected: list[RetrievedChunk] = []
        deferred: list[RetrievedChunk] = []
        source_counts: dict[str, int] = {}

        for chunk in chunks:
            key = self._source_key(chunk)
            count = source_counts.get(key, 0)
            if count < max_per_source:
                selected.append(chunk)
                source_counts[key] = count + 1
            else:
                deferred.append(chunk)
            if len(selected) >= target_top_k:
                break

        if len(selected) < target_top_k:
            selected_ids = {chunk.chunk_id for chunk in selected}
            for chunk in deferred:
                if chunk.chunk_id in selected_ids:
                    continue
                selected.append(chunk)
                selected_ids.add(chunk.chunk_id)
                key = self._source_key(chunk)
                source_counts[key] = source_counts.get(key, 0) + 1
                if len(selected) >= target_top_k:
                    break

        return selected[:target_top_k], {
            "enabled": True,
            "max_chunks_per_source": max_per_source,
            "source_counts": source_counts,
            "deferred_count": len(deferred),
            "backfill_if_needed": True,
        }

    def _source_key(self, chunk: RetrievedChunk) -> str:
        source_url = (chunk.source_url or "").strip().lower().rstrip("/")
        if source_url:
            return source_url
        if chunk.article_id:
            return str(chunk.article_id)
        if chunk.title:
            return chunk.title.strip().lower()
        return chunk.chunk_id


class RerankedRetriever:
    """Adapter that turns any retriever into a cross-encoder reranked method."""

    def __init__(
        self,
        *,
        base_retriever: Any,
        reranker: CrossEncoderReranker,
        method_name: str,
        config: CrossEncoderRerankingConfig | None = None,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.method_name = method_name
        self.config = config or reranker.config

    def retrieve(self, query: str) -> RetrievalResult:
        base_result = self._retrieve_base(query)
        reranked_chunks, rerank_debug = self.reranker.rerank(
            query=query,
            chunks=base_result.chunks,
            method_name=self.method_name,
            final_top_k=self.config.final_top_k,
        )

        debug = {
            "base_method": base_result.method,
            "base_candidate_count": len(base_result.chunks),
            "rerank": rerank_debug,
            "base_debug": base_result.debug,
        }
        # Keep vector contract fields visible for evaluation/debug tooling.
        for key in ("collection", "vector_text_mode"):
            if key in base_result.debug:
                debug[key] = base_result.debug[key]
        if "branch_debug" in base_result.debug:
            debug["branch_debug"] = base_result.debug["branch_debug"]

        return RetrievalResult(
            query=query,
            method=self.method_name,
            chunks=reranked_chunks,
            entities=base_result.entities,
            relations=base_result.relations,
            keywords=base_result.keywords,
            debug=debug,
        )

    def _retrieve_base(self, query: str) -> RetrievalResult:
        try:
            return self.base_retriever.retrieve(query, self.config.candidate_top_k)
        except TypeError:
            return self.base_retriever.retrieve(query)
