import os
from functools import cached_property
from typing import Iterable

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Thin wrapper around sentence-transformers for all vector indexing/search.

    Default model is BAAI/bge-m3 because it works well for Vietnamese and is
    already used in the project playground. The service normalizes embeddings
    by default so Qdrant cosine search behaves predictably.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.device = device or os.getenv("EMBEDDING_DEVICE") or None
        self.normalize_embeddings = normalize_embeddings

    @cached_property
    def model(self) -> SentenceTransformer:
        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        return SentenceTransformer(self.model_name, **kwargs)

    @cached_property
    def vector_dim(self) -> int:
        dim = int(self.model.get_sentence_embedding_dimension())
        if dim <= 0:
            probe = self.embed_text("dimension probe")
            dim = len(probe)
        return dim

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        values = [text or "" for text in texts]
        if not values:
            return []

        vectors = self.model.encode(
            values,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

        return [vector.astype(float).tolist() for vector in vectors]
