CREATE_CONSTRAINTS = [
    """
    CREATE CONSTRAINT article_id_unique IF NOT EXISTS
    FOR (a:Article)
    REQUIRE a.article_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
    FOR (c:Chunk)
    REQUIRE c.chunk_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
    FOR (e:Entity)
    REQUIRE e.entity_id IS UNIQUE
    """,
]

CREATE_INDEXES = [
    """
    CREATE INDEX article_url_index IF NOT EXISTS
    FOR (a:Article)
    ON (a.url)
    """,
    """
    CREATE INDEX chunk_article_id_index IF NOT EXISTS
    FOR (c:Chunk)
    ON (c.article_id)
    """,
    """
    CREATE INDEX chunk_section_index IF NOT EXISTS
    FOR (c:Chunk)
    ON (c.section)
    """,
    """
    CREATE INDEX entity_normalized_name_index IF NOT EXISTS
    FOR (e:Entity)
    ON (e.normalized_name)
    """,
    """
    CREATE INDEX entity_type_index IF NOT EXISTS
    FOR (e:Entity)
    ON (e.entity_type)
    """,
    """
    CREATE INDEX medical_relation_type_index IF NOT EXISTS
    FOR ()-[r:MEDICAL_RELATION]-()
    ON (r.relation_type)
    """,
]

UPSERT_ARTICLE = """
MERGE (a:Article {article_id: $article_id})
SET
    a.source = $source,
    a.url = $url,
    a.title = $title,
    a.description = $description,
    a.author = $author,
    a.published_at = $published_at,
    a.updated_at = $updated_at,
    a.category = $category,
    a.crawled_at = $crawled_at,
    a.metadata_json = $metadata_json,
    a.updated_in_neo4j_at = datetime()
"""

UPSERT_CHUNK = """
MERGE (c:Chunk {chunk_id: $chunk_id})
SET
    c.article_id = $article_id,
    c.source_url = $source_url,
    c.title = $title,
    c.section = $section,
    c.subsection = $subsection,
    c.text = $text,
    c.contextualized_text = $contextualized_text,
    c.chunk_index = $chunk_index,
    c.token_count = $token_count,
    c.metadata_json = $metadata_json,
    c.updated_in_neo4j_at = datetime()

WITH c
MATCH (a:Article {article_id: $article_id})
MERGE (a)-[r:HAS_CHUNK]->(c)
SET
    r.chunk_index = $chunk_index,
    r.section = $section,
    r.subsection = $subsection
"""

UPSERT_ENTITY = """
MERGE (e:Entity {entity_id: $entity_id})
SET
    e.name = $name,
    e.normalized_name = $normalized_name,
    e.entity_type = $entity_type,
    e.aliases = $aliases,
    e.description = $description,
    e.profile_text = $profile_text,
    e.local_keys = $local_keys,
    e.global_keys = $global_keys,
    e.mention_count = coalesce(e.mention_count, 0) + $mention_count,
    e.source_count = CASE
        WHEN $source_count > coalesce(e.source_count, 0)
        THEN $source_count
        ELSE coalesce(e.source_count, 0)
    END,
    e.metadata_json = $metadata_json,
    e.updated_in_neo4j_at = datetime()
"""

LINK_CHUNK_MENTIONS_ENTITY = """
MATCH (c:Chunk {chunk_id: $chunk_id})
MATCH (e:Entity {entity_id: $entity_id})
MERGE (c)-[r:MENTIONS]->(e)
SET
    r.confidence = $confidence,
    r.evidence_text = $evidence_text,
    r.section = $section,
    r.updated_in_neo4j_at = datetime()
"""

UPSERT_MEDICAL_RELATION = """
MATCH (s:Entity {entity_id: $subject_entity_id})
MATCH (o:Entity {entity_id: $object_entity_id})
MERGE (s)-[r:MEDICAL_RELATION {relation_id: $relation_id}]->(o)
SET
    r.relation_type = $relation_type,
    r.evidence_text = $evidence_text,
    r.evidence_chunk_ids = $evidence_chunk_ids,
    r.confidence = $confidence,
    r.section = $section,
    r.source_url = $source_url,
    r.metadata_json = $metadata_json,
    r.updated_in_neo4j_at = datetime()
"""

UPSERT_SYNONYM = """
MATCH (a:Entity {entity_id: $entity_id_1})
MATCH (b:Entity {entity_id: $entity_id_2})
MERGE (a)-[r:SYNONYM_OF]-(b)
SET
    r.score = $score,
    r.method = $method,
    r.updated_in_neo4j_at = datetime()
"""

GET_ENTITY_BY_ID = """
MATCH (e:Entity {entity_id: $entity_id})
RETURN
    e.entity_id AS entity_id,
    e.name AS name,
    e.normalized_name AS normalized_name,
    e.entity_type AS entity_type,
    e.aliases AS aliases,
    e.description AS description,
    e.profile_text AS profile_text,
    e.local_keys AS local_keys,
    e.global_keys AS global_keys,
    e.mention_count AS mention_count,
    e.source_count AS source_count,
    e.metadata_json AS metadata_json
LIMIT 1
"""

FIND_ENTITIES_BY_NORMALIZED_NAME = """
MATCH (e:Entity)
WHERE e.normalized_name CONTAINS $text
   OR $text CONTAINS e.normalized_name
   OR any(alias IN e.aliases WHERE alias CONTAINS $text OR $text CONTAINS alias)
RETURN
    e.entity_id AS entity_id,
    e.name AS name,
    e.normalized_name AS normalized_name,
    e.entity_type AS entity_type,
    e.aliases AS aliases,
    e.description AS description,
    e.profile_text AS profile_text,
    e.local_keys AS local_keys,
    e.global_keys AS global_keys,
    e.mention_count AS mention_count,
    e.source_count AS source_count,
    e.metadata_json AS metadata_json
LIMIT $limit
"""

GET_CHUNKS_BY_ENTITY_IDS = """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE e.entity_id IN $entity_ids
RETURN DISTINCT
    c.chunk_id AS chunk_id,
    c.article_id AS article_id,
    c.source_url AS source_url,
    c.title AS title,
    c.section AS section,
    c.subsection AS subsection,
    c.text AS text,
    c.contextualized_text AS contextualized_text,
    c.chunk_index AS chunk_index,
    c.token_count AS token_count,
    c.metadata_json AS metadata_json,
    count(e) AS matched_entity_count
ORDER BY matched_entity_count DESC
LIMIT $limit
"""

GET_NEIGHBOR_ENTITIES = """
MATCH (seed:Entity)-[r:MEDICAL_RELATION]-(n:Entity)
WHERE seed.entity_id IN $entity_ids
RETURN DISTINCT
    n.entity_id AS entity_id,
    n.name AS name,
    n.normalized_name AS normalized_name,
    n.entity_type AS entity_type,
    n.aliases AS aliases,
    n.description AS description,
    n.profile_text AS profile_text,
    n.local_keys AS local_keys,
    n.global_keys AS global_keys,
    n.mention_count AS mention_count,
    n.source_count AS source_count,
    n.metadata_json AS metadata_json,
    count(r) AS relation_count
ORDER BY relation_count DESC
LIMIT $limit
"""

GET_GRAPH_STATS = """
MATCH (a:Article)
WITH count(a) AS article_count
MATCH (c:Chunk)
WITH article_count, count(c) AS chunk_count
MATCH (e:Entity)
WITH article_count, chunk_count, count(e) AS entity_count
MATCH ()-[r:MEDICAL_RELATION]->()
RETURN
    article_count,
    chunk_count,
    entity_count,
    count(r) AS relation_count
"""

DELETE_ALL_DATA = """
MATCH (n)
DETACH DELETE n
"""

# =========================
# Entity lookup / resolution
# =========================

GET_ENTITIES_BY_IDS = """
MATCH (e:Entity)
WHERE e.entity_id IN $entity_ids
RETURN
    e.entity_id AS entity_id,
    e.name AS name,
    e.normalized_name AS normalized_name,
    e.entity_type AS entity_type,
    e.aliases AS aliases,
    e.description AS description,
    e.profile_text AS profile_text,
    e.local_keys AS local_keys,
    e.global_keys AS global_keys,
    e.mention_count AS mention_count,
    e.source_count AS source_count,
    e.metadata_json AS metadata_json,
    0.0 AS score
"""

GET_ENTITIES_BY_NORMALIZED_NAMES = """
MATCH (e:Entity)
WHERE e.normalized_name IN $normalized_names
RETURN
    e.entity_id AS entity_id,
    e.name AS name,
    e.normalized_name AS normalized_name,
    e.entity_type AS entity_type,
    e.aliases AS aliases,
    e.description AS description,
    e.profile_text AS profile_text,
    e.local_keys AS local_keys,
    e.global_keys AS global_keys,
    e.mention_count AS mention_count,
    e.source_count AS source_count,
    e.metadata_json AS metadata_json,
    0.0 AS score
"""

GET_CHUNKS_BY_IDS = """
MATCH (c:Chunk)
WHERE c.chunk_id IN $chunk_ids
RETURN
    c.chunk_id AS chunk_id,
    c.article_id AS article_id,
    c.source_url AS source_url,
    c.title AS title,
    c.section AS section,
    c.subsection AS subsection,
    c.text AS text,
    c.contextualized_text AS contextualized_text,
    c.chunk_index AS chunk_index,
    c.token_count AS token_count,
    c.metadata_json AS metadata_json,
    0.0 AS score
ORDER BY c.chunk_index ASC
"""

# =========================
# LightRAG-style retrieval
# =========================

SEARCH_ENTITIES_FOR_LIGHTRAG = """
MATCH (e:Entity)
WITH e,
     toLower($query_text) AS q
WITH e, q,
     (
        CASE WHEN toLower(e.name) CONTAINS q THEN 4 ELSE 0 END +
        CASE WHEN toLower(e.normalized_name) CONTAINS q THEN 4 ELSE 0 END +
        CASE WHEN any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS q) THEN 3 ELSE 0 END +
        CASE WHEN toLower(coalesce(e.description, '')) CONTAINS q THEN 2 ELSE 0 END +
        CASE WHEN toLower(coalesce(e.profile_text, '')) CONTAINS q THEN 2 ELSE 0 END +
        CASE WHEN any(k IN coalesce(e.local_keys, []) WHERE toLower(k) CONTAINS q) THEN 2 ELSE 0 END +
        CASE WHEN any(k IN coalesce(e.global_keys, []) WHERE toLower(k) CONTAINS q) THEN 2 ELSE 0 END
     ) AS score
WHERE score > 0
  AND ($entity_types IS NULL OR e.entity_type IN $entity_types)
RETURN
    e.entity_id AS entity_id,
    e.name AS name,
    e.normalized_name AS normalized_name,
    e.entity_type AS entity_type,
    e.aliases AS aliases,
    e.description AS description,
    e.profile_text AS profile_text,
    e.local_keys AS local_keys,
    e.global_keys AS global_keys,
    e.mention_count AS mention_count,
    e.source_count AS source_count,
    e.metadata_json AS metadata_json,
    score AS score
ORDER BY score DESC, coalesce(e.mention_count, 0) DESC
LIMIT $limit
"""

SEARCH_RELATIONS_FOR_LIGHTRAG = """
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WITH s, r, o, toLower($query_text) AS q
WITH s, r, o, q,
     (
        CASE WHEN toLower(coalesce(r.relation_type, '')) CONTAINS q THEN 2 ELSE 0 END +
        CASE WHEN toLower(coalesce(r.evidence_text, '')) CONTAINS q THEN 4 ELSE 0 END +
        CASE WHEN toLower(s.name) CONTAINS q THEN 3 ELSE 0 END +
        CASE WHEN toLower(o.name) CONTAINS q THEN 3 ELSE 0 END +
        CASE WHEN any(a IN coalesce(s.aliases, []) WHERE toLower(a) CONTAINS q) THEN 2 ELSE 0 END +
        CASE WHEN any(a IN coalesce(o.aliases, []) WHERE toLower(a) CONTAINS q) THEN 2 ELSE 0 END +
        CASE WHEN toLower(coalesce(s.profile_text, '')) CONTAINS q THEN 1 ELSE 0 END +
        CASE WHEN toLower(coalesce(o.profile_text, '')) CONTAINS q THEN 1 ELSE 0 END
     ) AS score
WHERE score > 0
  AND ($relation_types IS NULL OR r.relation_type IN $relation_types)
RETURN
    r.relation_id AS relation_id,
    r.relation_type AS relation_type,

    s.entity_id AS subject_entity_id,
    s.name AS subject_name,
    s.entity_type AS subject_type,

    o.entity_id AS object_entity_id,
    o.name AS object_name,
    o.entity_type AS object_type,

    r.evidence_text AS evidence_text,
    r.evidence_chunk_ids AS evidence_chunk_ids,
    r.confidence AS confidence,
    r.section AS section,
    r.source_url AS source_url,
    r.metadata_json AS metadata_json,
    score AS score
ORDER BY score DESC, coalesce(r.confidence, 1.0) DESC
LIMIT $limit
"""

GET_RELATIONS_BY_ENTITY_IDS = """
MATCH (s:Entity)-[r:MEDICAL_RELATION]-(o:Entity)
WHERE (s.entity_id IN $entity_ids OR o.entity_id IN $entity_ids)
  AND ($relation_types IS NULL OR r.relation_type IN $relation_types)
RETURN DISTINCT
    r.relation_id AS relation_id,
    r.relation_type AS relation_type,

    startNode(r).entity_id AS subject_entity_id,
    startNode(r).name AS subject_name,
    startNode(r).entity_type AS subject_type,

    endNode(r).entity_id AS object_entity_id,
    endNode(r).name AS object_name,
    endNode(r).entity_type AS object_type,

    r.evidence_text AS evidence_text,
    r.evidence_chunk_ids AS evidence_chunk_ids,
    r.confidence AS confidence,
    r.section AS section,
    r.source_url AS source_url,
    r.metadata_json AS metadata_json,
    coalesce(r.confidence, 1.0) AS score
ORDER BY score DESC
LIMIT $limit
"""

GET_CHUNKS_BY_RELATION_IDS = """
MATCH ()-[r:MEDICAL_RELATION]->()
WHERE r.relation_id IN $relation_ids
WITH collect(DISTINCT r.evidence_chunk_ids) AS nested_ids
WITH reduce(all_ids = [], ids IN nested_ids | all_ids + ids) AS chunk_ids
MATCH (c:Chunk)
WHERE c.chunk_id IN chunk_ids
RETURN
    c.chunk_id AS chunk_id,
    c.article_id AS article_id,
    c.source_url AS source_url,
    c.title AS title,
    c.section AS section,
    c.subsection AS subsection,
    c.text AS text,
    c.contextualized_text AS contextualized_text,
    c.chunk_index AS chunk_index,
    c.token_count AS token_count,
    c.metadata_json AS metadata_json,
    0.0 AS score
LIMIT $limit
"""

# =========================
# HippoRAG-style retrieval
# =========================

FIND_SEED_ENTITIES = """
MATCH (e:Entity)
WITH e, toLower($query_text) AS q
WITH e, q,
     (
        CASE WHEN toLower(e.name) CONTAINS q THEN 5 ELSE 0 END +
        CASE WHEN toLower(e.normalized_name) CONTAINS q THEN 5 ELSE 0 END +
        CASE WHEN any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS q) THEN 4 ELSE 0 END
     ) AS score
WHERE score > 0
RETURN
    e.entity_id AS entity_id,
    e.name AS name,
    e.normalized_name AS normalized_name,
    e.entity_type AS entity_type,
    e.aliases AS aliases,
    e.description AS description,
    e.profile_text AS profile_text,
    e.local_keys AS local_keys,
    e.global_keys AS global_keys,
    e.mention_count AS mention_count,
    e.source_count AS source_count,
    e.metadata_json AS metadata_json,
    score AS score
ORDER BY score DESC, coalesce(e.mention_count, 0) DESC
LIMIT $limit
"""

GET_ENTITY_ADJACENCY = """
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WHERE ($relation_types IS NULL OR r.relation_type IN $relation_types)
RETURN
    s.entity_id AS source_entity_id,
    o.entity_id AS target_entity_id,
    r.relation_id AS relation_id,
    r.relation_type AS relation_type,
    coalesce(r.confidence, 1.0) AS confidence,
    coalesce(r.confidence, 1.0) AS weight
LIMIT $limit
"""

GET_ENTITY_ADJACENCY_AROUND_SEEDS_1HOP = """
MATCH (seed:Entity)
WHERE seed.entity_id IN $seed_entity_ids
MATCH (seed)-[:MEDICAL_RELATION]-(n:Entity)
WITH collect(DISTINCT seed.entity_id) + collect(DISTINCT n.entity_id) AS raw_ids
UNWIND raw_ids AS id
WITH collect(DISTINCT id) AS ids
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WHERE s.entity_id IN ids
  AND o.entity_id IN ids
  AND ($relation_types IS NULL OR r.relation_type IN $relation_types)
RETURN
    s.entity_id AS source_entity_id,
    o.entity_id AS target_entity_id,
    r.relation_id AS relation_id,
    r.relation_type AS relation_type,
    coalesce(r.confidence, 1.0) AS confidence,
    coalesce(r.confidence, 1.0) AS weight
LIMIT $limit
"""

GET_ENTITY_ADJACENCY_AROUND_SEEDS_2HOP = """
MATCH (seed:Entity)
WHERE seed.entity_id IN $seed_entity_ids
MATCH (seed)-[:MEDICAL_RELATION*1..2]-(n:Entity)
WITH collect(DISTINCT seed.entity_id) + collect(DISTINCT n.entity_id) AS raw_ids
UNWIND raw_ids AS id
WITH collect(DISTINCT id) AS ids
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WHERE s.entity_id IN ids
  AND o.entity_id IN ids
  AND ($relation_types IS NULL OR r.relation_type IN $relation_types)
RETURN
    s.entity_id AS source_entity_id,
    o.entity_id AS target_entity_id,
    r.relation_id AS relation_id,
    r.relation_type AS relation_type,
    coalesce(r.confidence, 1.0) AS confidence,
    coalesce(r.confidence, 1.0) AS weight
LIMIT $limit
"""

GET_ENTITY_ADJACENCY_AROUND_SEEDS_3HOP = """
MATCH (seed:Entity)
WHERE seed.entity_id IN $seed_entity_ids
MATCH (seed)-[:MEDICAL_RELATION*1..3]-(n:Entity)
WITH collect(DISTINCT seed.entity_id) + collect(DISTINCT n.entity_id) AS raw_ids
UNWIND raw_ids AS id
WITH collect(DISTINCT id) AS ids
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WHERE s.entity_id IN ids
  AND o.entity_id IN ids
  AND ($relation_types IS NULL OR r.relation_type IN $relation_types)
RETURN
    s.entity_id AS source_entity_id,
    o.entity_id AS target_entity_id,
    r.relation_id AS relation_id,
    r.relation_type AS relation_type,
    coalesce(r.confidence, 1.0) AS confidence,
    coalesce(r.confidence, 1.0) AS weight
LIMIT $limit
"""

GET_CHUNK_ENTITY_LINKS = """
MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity)
WHERE e.entity_id IN $entity_ids
RETURN
    c.chunk_id AS chunk_id,
    e.entity_id AS entity_id,
    e.name AS entity_name,
    e.entity_type AS entity_type,
    coalesce(m.confidence, 1.0) AS confidence,
    m.section AS section,
    m.evidence_text AS evidence_text
LIMIT $limit
"""
GET_RELATIONS_BY_IDS = """
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WHERE r.relation_id IN $relation_ids
RETURN
    r.relation_id AS relation_id,
    r.relation_type AS relation_type,

    s.entity_id AS subject_entity_id,
    s.name AS subject_name,
    s.entity_type AS subject_type,

    o.entity_id AS object_entity_id,
    o.name AS object_name,
    o.entity_type AS object_type,

    r.evidence_text AS evidence_text,
    r.evidence_chunk_ids AS evidence_chunk_ids,
    r.confidence AS confidence,
    r.section AS section,
    r.source_url AS source_url,
    r.metadata_json AS metadata_json,
    coalesce(r.confidence, 1.0) AS score
ORDER BY score DESC
"""