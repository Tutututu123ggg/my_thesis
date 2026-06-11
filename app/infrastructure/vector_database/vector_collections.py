import os


# Context-enriched chunk vectors kept for compatibility/strong baseline only.
CHUNK_COLLECTION = os.getenv("QDRANT_CHUNK_COLLECTION", "medical_chunks")

# Real naive Vector RAG baseline: embeddings are built from raw chunk.text only.
NAIVE_CHUNK_COLLECTION = os.getenv("QDRANT_NAIVE_CHUNK_COLLECTION", "medical_chunks_naive")

ENTITY_COLLECTION = os.getenv("QDRANT_ENTITY_COLLECTION", "medical_entities")
RELATION_COLLECTION = os.getenv("QDRANT_RELATION_COLLECTION", "medical_relations")


ALL_COLLECTIONS = [
    CHUNK_COLLECTION,
    NAIVE_CHUNK_COLLECTION,
    ENTITY_COLLECTION,
    RELATION_COLLECTION,
]
