from __future__ import annotations

import os
from typing import Iterable, Protocol

from app.infrastructure.embedding import EmbeddingService


class TextEmbeddingClient(Protocol):
    """Small interface used by RAGAS-style answer relevance.

    This is intentionally separate from the retrieval embedding service so the
    evaluator can use an independent embedding model (e.g. OpenAI
    text-embedding-3-large) while retrieval still uses BGE-M3.
    """

    model_name: str

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        ...


class LocalTextEmbeddingClient:
    """Local sentence-transformer embedding client used as an offline fallback."""

    def __init__(self, model_name: str | None = None):
        self.service = EmbeddingService(model_name=model_name)
        self.model_name = self.service.model_name

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return self.service.embed_texts(texts)


class OpenAITextEmbeddingClient:
    """OpenAI embedding client for evaluation-only answer relevance.

    Default model is text-embedding-3-large. The original RAGAS paper used
    text-embedding-ada-002 for this generated-question similarity step; this
    implementation keeps the same metric structure but uses a newer official
    OpenAI embedding model by default.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        values = [text or "" for text in texts]
        if not values:
            return []

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(model=self.model_name, input=values)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]


def build_answer_relevance_embedding_client(
    provider: str = "openai",
    model_name: str = "text-embedding-3-large",
) -> TextEmbeddingClient:
    provider = (provider or "openai").strip().lower()
    model_name = (model_name or "text-embedding-3-large").strip()
    if provider == "openai":
        return OpenAITextEmbeddingClient(model_name=model_name)
    if provider == "local":
        return LocalTextEmbeddingClient(model_name=model_name or None)
    raise ValueError(f"Unsupported answer relevance embedding provider: {provider}")
