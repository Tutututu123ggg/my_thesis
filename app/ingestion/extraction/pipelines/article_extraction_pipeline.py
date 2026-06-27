from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.ingestion.extraction.common.extraction_models import ExtractedGraph
from app.ingestion.extraction.relation.extraction_validator import ExtractionValidator
from app.ingestion.extraction.keyword.keyword_profiler import ArticleKeywordProfiler
from app.ingestion.extraction.entity.medical_extractor import MedicalExtractor
from app.ingestion.processing import ArticleDocument, ChunkDocument
from app.ingestion.processing.fact_window import build_fact_windows_for_chunks


ProgressCallback = Callable[[int, int, ChunkDocument], None]


@dataclass
class ArticleExtractionResult:
    retrieval_chunks: list[ChunkDocument]
    extraction_windows: list[ChunkDocument]
    raw_graphs: list[ExtractedGraph]
    validated_graphs: list[ExtractedGraph]
    keyword_profile_debug: dict[str, Any] = field(default_factory=dict)
    windowing_debug: dict[str, Any] = field(default_factory=dict)

    @property
    def extra_extraction_windows(self) -> list[ChunkDocument]:
        retrieval_ids = {chunk.chunk_id for chunk in self.retrieval_chunks}
        return [window for window in self.extraction_windows if window.chunk_id not in retrieval_ids]


class ArticleExtractionPipeline:
    """Multi-stage article extraction orchestrator.

    Stage 1: build smaller fact windows for extraction only.
    Stage 2: run the medical fact extractor and validator per window.
    Stage 3: run article-level keyword profiling over validated relations.

    This keeps Vector RAG chunks stable while improving graph extraction recall:
    LLM fact extraction sees shorter windows, but keyword profiling still sees
    the article-level validated graph context.
    """

    def __init__(
        self,
        *,
        extractor: MedicalExtractor | None = None,
        validator: ExtractionValidator | None = None,
        keyword_profiler: ArticleKeywordProfiler | None = None,
        use_fact_windows: bool = True,
        fact_window_max_tokens: int = 450,
        fact_window_min_tokens: int = 80,
        fact_window_overlap_sentences: int = 1,
    ):
        self.extractor = extractor or MedicalExtractor()
        self.validator = validator or ExtractionValidator()
        self.keyword_profiler = keyword_profiler
        self.use_fact_windows = bool(use_fact_windows)
        self.fact_window_max_tokens = int(fact_window_max_tokens or 450)
        self.fact_window_min_tokens = int(fact_window_min_tokens or 80)
        self.fact_window_overlap_sentences = int(fact_window_overlap_sentences or 1)

    def build_extraction_windows(
        self,
        *,
        article: ArticleDocument,
        retrieval_chunks: list[ChunkDocument],
    ) -> tuple[list[ChunkDocument], dict[str, Any]]:
        if not self.use_fact_windows:
            return retrieval_chunks, {
                "enabled": False,
                "retrieval_chunk_count": len(retrieval_chunks),
                "extraction_window_count": len(retrieval_chunks),
                "split_parent_chunk_count": 0,
            }

        windows, stats = build_fact_windows_for_chunks(
            article,
            retrieval_chunks,
            max_tokens=self.fact_window_max_tokens,
            min_tokens=self.fact_window_min_tokens,
            overlap_sentences=self.fact_window_overlap_sentences,
        )
        debug = {"enabled": True, **stats.model_dump()}
        return windows, debug

    def extract_article(
        self,
        *,
        article: ArticleDocument,
        retrieval_chunks: list[ChunkDocument],
        max_windows: int = 0,
        progress_callback: ProgressCallback | None = None,
    ) -> ArticleExtractionResult:
        extraction_windows, windowing_debug = self.build_extraction_windows(
            article=article,
            retrieval_chunks=retrieval_chunks,
        )
        if max_windows and max_windows > 0:
            extraction_windows = extraction_windows[:max_windows]
            windowing_debug = {**windowing_debug, "max_windows_applied": int(max_windows)}

        raw_graphs: list[ExtractedGraph] = []
        validated_graphs: list[ExtractedGraph] = []

        total = len(extraction_windows)
        for i, window in enumerate(extraction_windows, start=1):
            if progress_callback is not None:
                progress_callback(i, total, window)
            raw_graph = self.extractor.extract_chunk(window)
            graph = self.validator.validate(raw_graph)
            raw_graphs.append(raw_graph)
            validated_graphs.append(graph)

        keyword_profile_debug: dict[str, Any] = {}
        if self.keyword_profiler is not None:
            keyword_profile_debug = self.keyword_profiler.enrich_graphs(
                title=article.title,
                chunks=extraction_windows,
                graphs=validated_graphs,
            )

        return ArticleExtractionResult(
            retrieval_chunks=retrieval_chunks,
            extraction_windows=extraction_windows,
            raw_graphs=raw_graphs,
            validated_graphs=validated_graphs,
            keyword_profile_debug=keyword_profile_debug,
            windowing_debug=windowing_debug,
        )