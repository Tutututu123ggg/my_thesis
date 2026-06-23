# Backward-compatible public API. Implementation is split into common/entity/relation/keyword/pipeline.
from app.ingestion.extraction.extraction_models import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractedGraph,
)
from app.ingestion.extraction.medical_extractor import MedicalExtractor
from app.ingestion.extraction.extraction_validator import ExtractionValidator
from app.ingestion.extraction.extraction_service import ExtractionService
from app.ingestion.extraction.keyword_profiler import ArticleKeywordProfiler
from app.ingestion.extraction.article_extraction_pipeline import ArticleExtractionPipeline, ArticleExtractionResult

__all__ = [
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractedGraph",
    "MedicalExtractor",
    "ExtractionValidator",
    "ExtractionService",
    "ArticleKeywordProfiler",
    "ArticleExtractionPipeline",
    "ArticleExtractionResult",
]