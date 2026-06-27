from app.infrastructure.graph_database import (
    ArticleNode,
    ChunkNode,
    GraphRepository,
)
from app.ingestion.processing import ArticleDocument, ChunkDocument


class GraphBuilder:
    """
    Ghi Article + Chunk vào Neo4j.

    Entity/relation sẽ được ghi ở bước extractor sau.
    """

    def __init__(self, graph_repo: GraphRepository):
        self.graph_repo = graph_repo

    def upsert_article(self, article: ArticleDocument) -> None:
        node = ArticleNode(
            article_id=article.article_id,
            source=article.source,
            url=article.url,
            title=article.title,
            description=article.description,
            author=article.author,
            published_at=article.published_at,
            updated_at=article.updated_at,
            category=article.category,
            crawled_at=article.crawled_at,
            metadata={
                **article.metadata,
                "related_links": article.related_links,
                "file_path": str(article.file_path) if article.file_path else None,
            },
        )

        self.graph_repo.upsert_article(node)

    def upsert_chunks(self, chunks: list[ChunkDocument]) -> None:
        for chunk in chunks:
            node = ChunkNode(
                chunk_id=chunk.chunk_id,
                article_id=chunk.article_id,
                source_url=chunk.source_url,
                title=chunk.title,
                section=chunk.section,
                subsection=chunk.subsection,
                text=chunk.text,
                contextualized_text=chunk.contextualized_text,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
            )

            self.graph_repo.upsert_chunk(node)

    def link_fact_windows_to_parents(self, chunks: list[ChunkDocument]) -> int:
        count = 0
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            parent_chunk_id = str(metadata.get("parent_chunk_id") or "").strip()
            if not parent_chunk_id:
                continue
            if parent_chunk_id == chunk.chunk_id:
                continue

            self.graph_repo.link_fact_window_to_parent(
                child_chunk_id=chunk.chunk_id,
                parent_chunk_id=parent_chunk_id,
            )
            count += 1

        return count

    def ingest_article_with_chunks(
        self,
        article: ArticleDocument,
        chunks: list[ChunkDocument],
    ) -> None:
        self.upsert_article(article)
        self.upsert_chunks(chunks)
        self.link_fact_windows_to_parents(chunks)
