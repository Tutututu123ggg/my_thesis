from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


LLMProvider = Literal["openai", "ollama"]


@dataclass
class ChatLLMConfig:
    """Configuration for a small OpenAI-compatible/Ollama chat client."""

    provider: LLMProvider = "ollama"
    model: str = "qwen2.5:3b"
    temperature: float = 0.0
    timeout_seconds: int = 180
    base_url: str | None = None
    api_key: str | None = None


@dataclass
class MedicalAnswerConfig:
    """Controls answer generation from retrieved context."""

    max_entities: int = 30
    max_relations: int = 40
    max_chunks: int = 8
    context_mode: Literal["chunks_only", "full_graph"] = "chunks_only"
    system_prompt: str = (
        "Bạn là trợ lý RAG y khoa tiếng Việt. Chỉ dùng context được cung cấp. "
        "Không chẩn đoán chắc chắn; chỉ nêu khả năng liên quan và khuyến nghị đi khám khi cần."
    )


@dataclass
class GeneratedAnswer:
    question: str
    answer: str
    method: str
    context: str
    context_chunks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
