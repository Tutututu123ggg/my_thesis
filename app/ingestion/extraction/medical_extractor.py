import os

from openai import OpenAI

from app.ingestion.extraction.extraction_models import (
    ExtractedGraph,
    LLMExtractedGraph,
)
from app.ingestion.extraction.extraction_prompt import (
    build_system_prompt,
    build_user_prompt,
)
from app.ingestion.processing import ChunkDocument


class MedicalExtractor:
    """
    Gọi LLM để extract entity/relation từ một chunk.

    Chỉ extract.
    Không validate.
    Không merge.
    Không ghi database.

    LLM chỉ nhìn thấy LLMExtractedGraph.
    Internal pipeline dùng ExtractedGraph.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        prompt_mode: str = "strict_text",
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.prompt_mode = prompt_mode
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def extract_chunk(self, chunk: ChunkDocument) -> ExtractedGraph:
        system_prompt = build_system_prompt(self.prompt_mode)

        user_prompt = build_user_prompt(
            article_title=chunk.title,
            section=chunk.section,
            subsection=chunk.subsection,
            chunk_text=chunk.contextualized_text,
        )

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format=LLMExtractedGraph,
        )

        parsed = completion.choices[0].message.parsed

        if parsed is None:
            return ExtractedGraph()

        return parsed.to_internal_graph()