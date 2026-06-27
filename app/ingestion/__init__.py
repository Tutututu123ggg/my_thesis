"""Public extraction API shim.

This package-level __init__ keeps legacy imports such as
`from app.ingestion.extraction import ExtractedGraph` working while the
implementation lives in explicit subpackages.
"""

from app.ingestion.extraction.common.extraction_models import (
    ExtractedEntity,
    ExtractedGraph,
    ExtractedRelation,
)
from app.ingestion.extraction.entity.medical_extractor import MedicalExtractor
from app.ingestion.extraction.keyword.keyword_profiler import ArticleKeywordProfiler

try:
    from app.ingestion.extraction.pipeline.article_extraction_pipeline import (
        ArticleExtractionPipeline,
        ArticleExtractionResult,
    )
    from app.ingestion.extraction.pipeline.extraction_service import ExtractionService
except ModuleNotFoundError:  # pragma: no cover - compatibility with plural package variant
    ArticleExtractionPipeline = None  # type: ignore[assignment]
    ArticleExtractionResult = None  # type: ignore[assignment]
    ExtractionService = None  # type: ignore[assignment]

from app.ingestion.extraction.relation.extraction_validator import ExtractionValidator

__all__ = [
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractedGraph",
    "MedicalExtractor",
    "ExtractionValidator",
    "ArticleKeywordProfiler",
    "ArticleExtractionPipeline",
    "ArticleExtractionResult",
    "ExtractionService",
]
