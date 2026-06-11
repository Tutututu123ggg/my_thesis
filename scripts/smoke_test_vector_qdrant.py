import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_test_collections(use_project_collections: bool) -> None:
    if use_project_collections:
        return

    os.environ["QDRANT_CHUNK_COLLECTION"] = "medical_chunks_smoke"
    os.environ["QDRANT_ENTITY_COLLECTION"] = "medical_entities_smoke"
    os.environ["QDRANT_RELATION_COLLECTION"] = "medical_relations_smoke"


def print_results(title: str, results) -> None:
    print(f"\n========== {title} ==========")
    for idx, item in enumerate(results, start=1):
        payload = item.payload
        label = (
            payload.get("title")
            or payload.get("name")
            or payload.get("relation_type")
            or payload.get("chunk_id")
            or payload.get("entity_id")
            or payload.get("relation_id")
        )
        print(f"{idx:02d}. score={item.score:.4f} | {label}")
        if payload.get("section"):
            print(f"    section={payload.get('section')}")
        if payload.get("subject_name") and payload.get("object_name"):
            print(f"    {payload.get('subject_name')} -[{payload.get('relation_type')}]-> {payload.get('object_name')}")
        text = payload.get("text") or payload.get("evidence_text") or payload.get("description")
        if text:
            print(f"    {str(text)[:180]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Recreate smoke collections before writing.")
    parser.add_argument("--cleanup-after", action="store_true", help="Delete smoke collections after test.")
    parser.add_argument(
        "--use-project-collections",
        action="store_true",
        help="Use QDRANT_* collections from .env. Default uses *_smoke collections.",
    )
    parser.add_argument("--query", default="chàm thể tạng có triệu chứng gì")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    configure_test_collections(args.use_project_collections)

    # Import sau khi set env collection names.
    from app.infrastructure.embedding import EmbeddingService
    from app.infrastructure.vector_database import QdrantClientWrapper, VectorRepository
    from app.ingestion.builders import VectorGraphBuilder
    from app.ingestion.extraction import ExtractedEntity, ExtractedGraph, ExtractedRelation
    from app.ingestion.processing import ChunkDocument
    from app.retrieval import VectorRetriever

    embedder = EmbeddingService()
    print(f"[EMBEDDING] model={embedder.model_name} dim={embedder.vector_dim}")

    qdrant_client = QdrantClientWrapper(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    vector_repo = VectorRepository(
        client=qdrant_client,
        vector_dim=embedder.vector_dim,
        distance=os.getenv("VECTOR_DISTANCE", "cosine"),
    )

    if not vector_repo.health_check():
        raise RuntimeError("Qdrant health check failed. Check Qdrant is running.")

    builder = VectorGraphBuilder(
        vector_repo=vector_repo,
        embedder=embedder,
        resolver=None,
        batch_size=16,
    )
    builder.setup_collections(recreate=args.recreate)

    chunks = [
        ChunkDocument(
            chunk_id="smoke_chunk_vdcd_001",
            article_id="smoke_article_vdcd",
            source_url="https://example.test/viem-da-co-dia",
            title="Viêm da cơ địa",
            section="Triệu chứng",
            subsection=None,
            chunk_index=0,
            text="Viêm da cơ địa thường gây ngứa, da khô, ban đỏ và có thể tái phát nhiều lần.",
            contextualized_text=(
                "Bài viết: Viêm da cơ địa\n"
                "Mục: Triệu chứng\n\n"
                "Viêm da cơ địa thường gây ngứa, da khô, ban đỏ và có thể tái phát nhiều lần."
            ),
            token_count=40,
            metadata={"smoke_test": True},
        ),
        ChunkDocument(
            chunk_id="smoke_chunk_vaynen_001",
            article_id="smoke_article_vaynen",
            source_url="https://example.test/benh-vay-nen",
            title="Bệnh vảy nến",
            section="Triệu chứng",
            subsection=None,
            chunk_index=1,
            text="Bệnh vảy nến có thể gây mảng đỏ, bong vảy trắng bạc và ngứa.",
            contextualized_text=(
                "Bài viết: Bệnh vảy nến\n"
                "Mục: Triệu chứng\n\n"
                "Bệnh vảy nến có thể gây mảng đỏ, bong vảy trắng bạc và ngứa."
            ),
            token_count=35,
            metadata={"smoke_test": True},
        ),
    ]

    graph = ExtractedGraph(
        entities=[
            ExtractedEntity(
                name="viêm da cơ địa",
                entity_type="BENH_LY",
                aliases=[],
                surface_forms=["viêm da cơ địa"],
                description="Bệnh da viêm mạn tính có thể gây ngứa và da khô.",
                evidence_text="Viêm da cơ địa thường gây ngứa, da khô, ban đỏ.",
            ),
            ExtractedEntity(
                name="chàm thể tạng",
                entity_type="BENH_LY",
                aliases=[],
                surface_forms=["chàm thể tạng"],
                description="Tên gọi khác được văn bản nêu rõ của viêm da cơ địa.",
                evidence_text="Viêm da cơ địa còn gọi là chàm thể tạng.",
            ),
            ExtractedEntity(
                name="ngứa",
                entity_type="BIEU_HIEN_LAM_SANG",
                aliases=[],
                surface_forms=["ngứa"],
                description="Triệu chứng thường gặp.",
                evidence_text="Viêm da cơ địa thường gây ngứa.",
            ),
        ],
        relations=[
            ExtractedRelation(
                subject="viêm da cơ địa",
                subject_type="BENH_LY",
                relation_type="CO_BIEU_HIEN",
                object="ngứa",
                object_type="BIEU_HIEN_LAM_SANG",
                description="Viêm da cơ địa có biểu hiện ngứa.",
                keywords=["viêm da cơ địa", "ngứa", "triệu chứng"],
                evidence_text="Viêm da cơ địa thường gây ngứa.",
                confidence=0.95,
            ),
            ExtractedRelation(
                subject="viêm da cơ địa",
                subject_type="BENH_LY",
                relation_type="DONG_NGHIA_VOI",
                object="chàm thể tạng",
                object_type="BENH_LY",
                description="Hai tên gọi được văn bản khẳng định là đồng nghĩa.",
                keywords=["viêm da cơ địa", "chàm thể tạng", "tên gọi khác"],
                evidence_text="Viêm da cơ địa còn gọi là chàm thể tạng.",
                confidence=0.98,
            ),
        ],
    )

    print("[UPSERT] chunks")
    chunk_count = builder.upsert_chunks(chunks)

    print("[UPSERT] entities/relations")
    extraction_stats = builder.upsert_chunk_extraction(chunks[0], graph)

    retriever = VectorRetriever(vector_repo=vector_repo, embedder=embedder)
    results = retriever.hybrid_search(
        query=args.query,
        chunk_limit=5,
        entity_limit=5,
        relation_limit=5,
    )

    print(f"\n[SUMMARY] chunk_vectors={chunk_count} {extraction_stats}")
    print_results("CHUNKS", results.chunks)
    print_results("ENTITIES", results.entities)
    print_results("RELATIONS", results.relations)

    assert results.chunks, "No chunk vector results returned."
    assert results.entities, "No entity vector results returned."
    assert results.relations, "No relation vector results returned."

    print("\n[PASS] Qdrant vector smoke test completed.")

    if args.cleanup_after:
        if args.use_project_collections:
            raise RuntimeError("Refuse to cleanup project collections. Run without --use-project-collections.")
        vector_repo.delete_all_collections()
        print("[CLEANUP] deleted smoke collections")

    qdrant_client.close()


if __name__ == "__main__":
    main()
