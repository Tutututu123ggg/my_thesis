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

    def ingest_article_with_chunks(
        self,
        article: ArticleDocument,
        chunks: list[ChunkDocument],
    ) -> None:
        self.upsert_article(article)
        self.upsert_chunks(chunks)