from app.ingestion.extraction.pipelines.article_extraction_pipeline import (
    ArticleExtractionPipeline,
    ArticleExtractionResult,
)

try:
    from app.ingestion.extraction.pipelines.extraction_service import ExtractionService
except ModuleNotFoundError:  # Some repo snapshots may not keep this module here.
    ExtractionService = None  # type: ignore[assignment]

__all__ = [
    "ArticleExtractionPipeline",
    "ArticleExtractionResult",
    "ExtractionService",
]
