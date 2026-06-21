import os
import re

from openai import OpenAI

from app.ingestion.extraction.extraction_models import (
    ExtractedGraph,
    LLMExtractedGraph,
)
from app.ingestion.extraction.extraction_prompt import build_system_prompt
from app.ingestion.processing import ChunkDocument


def _strip_heading_marker(text: str) -> str:
    return re.sub(r"^\s*#{1,6}\s*", "", str(text or "").strip()).strip()


def _split_heading_context(heading_context: str) -> list[str]:
    return [
        _strip_heading_marker(part)
        for part in str(heading_context or "").split("|")
        if _strip_heading_marker(part)
    ]


def build_contextualized_markdown_for_extraction(chunk: ChunkDocument) -> str:
    """
    Tạo input markdown cho LLM theo đúng cấu trúc tài liệu:

    # Article title
    ## Heading context
    ### Raw chunk headings/content

    Vector retrieval vẫn có thể dùng raw chunk text ở pipeline khác;
    hàm này chỉ phục vụ extraction graph.
    """

    parts: list[str] = []

    title = _strip_heading_marker(chunk.title)
    if title:
        parts.append(f"# {title}")

    metadata = getattr(chunk, "metadata", {}) or {}
    heading_context = metadata.get("heading_path_context") or ""

    for index, heading in enumerate(_split_heading_context(heading_context), start=2):
        level = min(index, 6)
        parts.append(f"{'#' * level} {heading}")

    text = str(chunk.text or "").strip()
    if text:
        parts.append(text)

    return "\n\n".join(parts).strip()


def build_extraction_user_prompt(chunk: ChunkDocument) -> str:
    extraction_markdown = build_contextualized_markdown_for_extraction(chunk)

    return f"""
Nội dung markdown cần trích xuất:
{extraction_markdown}
""".strip()


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
        user_prompt = build_extraction_user_prompt(chunk)

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
