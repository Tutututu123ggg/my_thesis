import json
from pathlib import Path

from app.ingestion.extraction.extraction_models import ExtractedGraph
from app.ingestion.extraction.extraction_validator import ExtractionValidator
from app.ingestion.extraction.medical_extractor import MedicalExtractor
from app.ingestion.processing import ChunkDocument


class ExtractionService:
    """
    Orchestrate LLM extraction + validation.

    Chưa ghi Neo4j/Qdrant ở bước này.
    """

    def __init__(
        self,
        extractor: MedicalExtractor | None = None,
        validator: ExtractionValidator | None = None,
    ):
        self.extractor = extractor or MedicalExtractor()
        self.validator = validator or ExtractionValidator()

    def extract_and_validate_chunk(self, chunk: ChunkDocument) -> ExtractedGraph:
        raw_graph = self.extractor.extract_chunk(chunk)
        return self.validator.validate(raw_graph)

    def extract_chunks(
        self,
        chunks: list[ChunkDocument],
        output_jsonl_path: str | Path | None = None,
    ) -> list[dict]:
        results: list[dict] = []

        output_file = Path(output_jsonl_path) if output_jsonl_path else None
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)

        for chunk in chunks:
            graph = self.extract_and_validate_chunk(chunk)

            record = {
                "chunk_id": chunk.chunk_id,
                "article_id": chunk.article_id,
                "title": chunk.title,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "source_url": chunk.source_url,
                "entities": [e.model_dump() for e in graph.entities],
                "relations": [r.model_dump() for r in graph.relations],
            }

            results.append(record)

            if output_file:
                with output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return results