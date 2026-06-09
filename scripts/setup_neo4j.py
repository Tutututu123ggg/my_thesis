import os

from dotenv import load_dotenv

from app.infrastructure.graph_database import Neo4jClient, GraphRepository


def main() -> None:
    load_dotenv()

    client = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )

    repo = GraphRepository(client)

    if not repo.health_check():
        raise RuntimeError("Neo4j health check failed. Check .env and Neo4j server.")

    repo.setup_schema()
    stats = repo.get_graph_stats()

    print("[OK] Neo4j schema is ready.")
    print(f"Articles : {stats['article_count']}")
    print(f"Chunks   : {stats['chunk_count']}")
    print(f"Entities : {stats['entity_count']}")
    print(f"Relations: {stats['relation_count']}")

    client.close()


if __name__ == "__main__":
    main()