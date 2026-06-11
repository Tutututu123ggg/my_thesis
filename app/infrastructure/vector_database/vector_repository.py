from typing import Any

from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.infrastructure.vector_database.qdrant_client import QdrantClientWrapper
from app.infrastructure.vector_database.vector_collections import (
    ALL_COLLECTIONS,
    CHUNK_COLLECTION,
    NAIVE_CHUNK_COLLECTION,
    ENTITY_COLLECTION,
    RELATION_COLLECTION,
)
from app.infrastructure.vector_database.vector_models import (
    VectorPoint,
    VectorSearchResult,
)


class VectorRepository:
    """
    Cổng duy nhất để các module khác giao tiếp với Qdrant.

    Collections:
    - medical_chunks: vector RAG thường
    - medical_entities: LightRAG entity retrieval
    - medical_relations: LightRAG relation retrieval
    """

    def __init__(
        self,
        client: QdrantClientWrapper,
        vector_dim: int,
        distance: str = "cosine",
    ):
        self.client = client
        self.vector_dim = int(vector_dim)
        self.distance = self._parse_distance(distance)

    # =====================
    # Setup
    # =====================

    def setup_collections(self, recreate: bool = False) -> None:
        for collection_name in ALL_COLLECTIONS:
            self.ensure_collection(collection_name, recreate=recreate)

    def ensure_collection(self, collection_name: str, recreate: bool = False) -> None:
        exists = self.client.client.collection_exists(collection_name)

        if exists and not recreate:
            return

        if exists and recreate:
            self.client.client.delete_collection(collection_name)

        self.client.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.vector_dim,
                distance=self.distance,
            ),
        )

    def delete_all_collections(self) -> None:
        for collection_name in ALL_COLLECTIONS:
            if self.client.client.collection_exists(collection_name):
                self.client.client.delete_collection(collection_name)

    def health_check(self) -> bool:
        return self.client.health_check()

    def collection_exists(self, collection_name: str) -> bool:
        return bool(self.client.client.collection_exists(collection_name))

    def count_points(self, collection_name: str) -> int:
        if not self.collection_exists(collection_name):
            return 0
        result = self.client.client.count(
            collection_name=collection_name,
            exact=True,
        )
        return int(result.count or 0)

    # =====================
    # Generic write/search
    # =====================

    def upsert_points(
        self,
        collection_name: str,
        points: list[VectorPoint],
    ) -> None:
        if not points:
            return

        self.client.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=p.point_id,
                    vector=p.vector,
                    payload=p.payload,
                )
                for p in points
            ],
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        query_filter = self._build_filter(filters)

        # qdrant-client bản mới dùng query_points.
        result = self.client.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            VectorSearchResult(
                point_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in result.points
        ]

    def delete_points(
        self,
        collection_name: str,
        point_ids: list[str],
    ) -> None:
        if not point_ids:
            return

        self.client.client.delete(
            collection_name=collection_name,
            points_selector=point_ids,
        )

    # =====================
    # Typed wrappers
    # =====================

    def upsert_chunk_vectors(self, points: list[VectorPoint]) -> None:
        self.upsert_points(CHUNK_COLLECTION, points)

    def upsert_naive_chunk_vectors(self, points: list[VectorPoint]) -> None:
        self.upsert_points(NAIVE_CHUNK_COLLECTION, points)

    def upsert_entity_vectors(self, points: list[VectorPoint]) -> None:
        self.upsert_points(ENTITY_COLLECTION, points)

    def upsert_relation_vectors(self, points: list[VectorPoint]) -> None:
        self.upsert_points(RELATION_COLLECTION, points)

    def search_chunks(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        return self.search(CHUNK_COLLECTION, query_vector, limit, filters)

    def search_naive_chunks(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        return self.search(NAIVE_CHUNK_COLLECTION, query_vector, limit, filters)

    def search_entities(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        return self.search(ENTITY_COLLECTION, query_vector, limit, filters)

    def search_relations(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        return self.search(RELATION_COLLECTION, query_vector, limit, filters)

    # =====================
    # Helpers
    # =====================

    def _parse_distance(self, distance: str) -> Distance:
        value = distance.lower().strip()

        if value == "cosine":
            return Distance.COSINE

        if value == "dot":
            return Distance.DOT

        if value in {"euclid", "euclidean"}:
            return Distance.EUCLID

        raise ValueError(f"Unsupported Qdrant distance: {distance}")

    def _build_filter(self, filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            if value is None:
                continue

            if isinstance(value, list):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(any=value),
                    )
                )
            else:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )

        if not conditions:
            return None

        return Filter(must=conditions)