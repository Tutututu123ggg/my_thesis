import os
import random

from dotenv import load_dotenv

from app.infrastructure.vector_database import (
    QdrantClientWrapper,
    VectorRepository,
    VectorPoint,
)
from app.infrastructure.vector_database.vector_models import make_qdrant_point_id


def fake_vector(dim: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def main() -> None:
    load_dotenv()

    dim = int(os.getenv("VECTOR_DIM", "768"))

    client = QdrantClientWrapper(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )

    repo = VectorRepository(
        client=client,
        vector_dim=dim,
        distance=os.getenv("VECTOR_DISTANCE", "cosine"),
    )

    repo.setup_collections(recreate=False)

    chunk_point = VectorPoint(
        point_id=make_qdrant_point_id("chunk_test_001"),
        vector=fake_vector(dim, seed=1),
        payload={
            "type": "chunk",
            "chunk_id": "chunk_test_001",
            "article_id": "article_test_001",
            "title": "Viêm da cơ địa",
            "section": "Triệu chứng",
            "source_url": "https://tamanhhospital.vn/benh/viem-da-co-dia",
            "text": "Viêm da cơ địa có biểu hiện ngứa, ban đỏ và da khô.",
        },
    )

    entity_point = VectorPoint(
        point_id=make_qdrant_point_id("entity_test_001"),
        vector=fake_vector(dim, seed=2),
        payload={
            "type": "entity",
            "entity_id": "entity_test_001",
            "name": "viêm da cơ địa",
            "entity_type": "BENH_LY",
        },
    )

    relation_point = VectorPoint(
        point_id=make_qdrant_point_id("relation_test_001"),
        vector=fake_vector(dim, seed=3),
        payload={
            "type": "relation",
            "relation_id": "relation_test_001",
            "relation_type": "CO_BIEU_HIEN",
            "subject_name": "viêm da cơ địa",
            "object_name": "ngứa",
            "evidence_text": "Dấu hiệu điển hình là ngứa.",
        },
    )

    repo.upsert_chunk_vectors([chunk_point])
    repo.upsert_entity_vectors([entity_point])
    repo.upsert_relation_vectors([relation_point])

    print("[Search chunks]")
    for item in repo.search_chunks(fake_vector(dim, seed=1), limit=3):
        print(item.score, item.payload)

    print("[Search entities]")
    for item in repo.search_entities(fake_vector(dim, seed=2), limit=3):
        print(item.score, item.payload)

    print("[Search relations]")
    for item in repo.search_relations(fake_vector(dim, seed=3), limit=3):
        print(item.score, item.payload)

    client.close()


if __name__ == "__main__":
    main()