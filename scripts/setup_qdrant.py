import os

from dotenv import load_dotenv

from app.infrastructure.vector_database import (
    QdrantClientWrapper,
    VectorRepository,
)


def main() -> None:
    load_dotenv()

    client = QdrantClientWrapper(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )

    repo = VectorRepository(
        client=client,
        vector_dim=int(os.getenv("VECTOR_DIM", "768")),
        distance=os.getenv("VECTOR_DISTANCE", "cosine"),
    )

    if not repo.health_check():
        raise RuntimeError("Qdrant health check failed. Check Qdrant server.")

    repo.setup_collections(recreate=False)

    print("[OK] Qdrant collections are ready.")
    print(f"- {os.getenv('QDRANT_CHUNK_COLLECTION', 'medical_chunks')}")
    print(f"- {os.getenv('QDRANT_ENTITY_COLLECTION', 'medical_entities')}")
    print(f"- {os.getenv('QDRANT_RELATION_COLLECTION', 'medical_relations')}")

    client.close()


if __name__ == "__main__":
    main()