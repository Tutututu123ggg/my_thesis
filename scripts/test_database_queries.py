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

    print("Stats:", repo.get_graph_stats())

    print("\n[LightRAG] search entities")
    entities = repo.search_entities_for_lightrag("viêm da cơ địa", limit=5)
    for e in entities:
        print(e.score, e.entity_type, e.name)

    print("\n[LightRAG] search relations")
    relations = repo.search_relations_for_lightrag("ngứa", limit=5)
    for r in relations:
        print(r.score, r.relation_type, r.subject_name, "->", r.object_name)

    print("\n[LightRAG] context bundle")
    if entities:
        bundle = repo.get_entity_context_bundle([entities[0].entity_id])
        print("entities:", len(bundle.entities))
        print("relations:", len(bundle.relations))
        print("chunks:", len(bundle.chunks))

    print("\n[HippoRAG] seed entities")
    seeds = repo.find_seed_entities("viêm da cơ địa", limit=5)
    for s in seeds:
        print(s.score, s.entity_type, s.name)

    print("\n[HippoRAG] adjacency around seeds")
    if seeds:
        edges = repo.get_entity_adjacency_around_seeds(
            seed_entity_ids=[s.entity_id for s in seeds],
            max_hops=2,
            limit=20,
        )
        for edge in edges:
            print(edge.source_entity_id, edge.relation_type, edge.target_entity_id)

    print("\n[HippoRAG] chunk entity links")
    if seeds:
        links = repo.get_chunk_entity_links(
            entity_ids=[s.entity_id for s in seeds],
            limit=20,
        )
        for link in links:
            print(link.chunk_id, link.entity_name, link.confidence)

    client.close()


if __name__ == "__main__":
    main()