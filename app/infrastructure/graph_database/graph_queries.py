# =========================
# Schema
# =========================

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

# =========================
# Reusable Cypher fragments
# =========================

ENTITY_RETURN_FIELDS = """
    {alias}.entity_id AS entity_id,
    {alias}.name AS name,
    {alias}.normalized_name AS normalized_name,
    {alias}.entity_type AS entity_type,
    {alias}.aliases AS aliases,
    {alias}.surface_forms AS surface_forms,
    {alias}.description AS description,
    {alias}.profile_text AS profile_text,
    {alias}.local_keys AS local_keys,
    {alias}.global_keys AS global_keys,
    {alias}.mention_count AS mention_count,
    {alias}.source_count AS source_count,
    {alias}.metadata_json AS metadata_json
"""

CHUNK_RETURN_FIELDS = """
    {alias}.chunk_id AS chunk_id,
    {alias}.article_id AS article_id,
    {alias}.source_url AS source_url,
    {alias}.title AS title,
    {alias}.section AS section,
    {alias}.subsection AS subsection,
    {alias}.text AS text,
    {alias}.contextualized_text AS contextualized_text,
    {alias}.chunk_index AS chunk_index,
    {alias}.token_count AS token_count,
    {alias}.metadata_json AS metadata_json
"""

RELATION_RETURN_FIELDS = """
    r.relation_id AS relation_id,
    r.relation_type AS relation_type,

    {subject_id_expr} AS subject_entity_id,
    {subject_name_expr} AS subject_name,
    {subject_type_expr} AS subject_type,

    {object_id_expr} AS object_entity_id,
    {object_name_expr} AS object_name,
    {object_type_expr} AS object_type,

    r.description AS description,
    r.keywords AS keywords,
    r.evidence_text AS evidence_text,
    r.evidence_chunk_ids AS evidence_chunk_ids,
    r.confidence AS confidence,
    r.section AS section,
    r.source_url AS source_url,
    r.metadata_json AS metadata_json
"""

ADJACENCY_RETURN_FIELDS = """
    s.entity_id AS source_entity_id,
    o.entity_id AS target_entity_id,
    r.relation_id AS relation_id,
    r.relation_type AS relation_type,
    coalesce(r.confidence, 1.0) AS confidence,
    coalesce(r.confidence, 1.0) AS weight
"""


def entity_return(alias: str = "e", score: str | None = None) -> str:
    fields = ENTITY_RETURN_FIELDS.format(alias=alias)

    if score is not None:
        fields += f",\n    {score} AS score"

    return fields


def chunk_return(alias: str = "c", score: str | None = None) -> str:
    fields = CHUNK_RETURN_FIELDS.format(alias=alias)

    if score is not None:
        fields += f",\n    {score} AS score"

    return fields


def relation_return(
    *,
    score: str | None = None,
    use_start_end: bool = False,
) -> str:
    if use_start_end:
        fields = RELATION_RETURN_FIELDS.format(
            subject_id_expr="startNode(r).entity_id",
            subject_name_expr="startNode(r).name",
            subject_type_expr="startNode(r).entity_type",
            object_id_expr="endNode(r).entity_id",
            object_name_expr="endNode(r).name",
            object_type_expr="endNode(r).entity_type",
        )
    else:
        fields = RELATION_RETURN_FIELDS.format(
            subject_id_expr="s.entity_id",
            subject_name_expr="s.name",
            subject_type_expr="s.entity_type",
            object_id_expr="o.entity_id",
            object_name_expr="o.name",
            object_type_expr="o.entity_type",
        )

    if score is not None:
        fields += f",\n    {score} AS score"

    return fields


def adjacency_around_seeds_query(max_hops: int) -> str:
    return f"""
MATCH (seed:Entity)
WHERE seed.entity_id IN $seed_entity_ids
MATCH (seed)-[:MEDICAL_RELATION*1..{max_hops}]-(n:Entity)
WITH collect(DISTINCT seed.entity_id) + collect(DISTINCT n.entity_id) AS raw_ids
UNWIND raw_ids AS id
WITH collect(DISTINCT id) AS ids
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WHERE s.entity_id IN ids
  AND o.entity_id IN ids
  AND ($relation_types IS NULL OR r.relation_type IN $relation_types)
RETURN
{ADJACENCY_RETURN_FIELDS}
LIMIT $limit
"""


# =========================
# Article / Chunk / Entity upsert
# =========================

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
    e.name = CASE
        WHEN e.name IS NULL OR e.name = ''
        THEN $name
        ELSE e.name
    END,
    e.normalized_name = CASE
        WHEN e.normalized_name IS NULL OR e.normalized_name = ''
        THEN $normalized_name
        ELSE e.normalized_name
    END,
    e.entity_type = $entity_type,
    e.description = CASE
        WHEN coalesce(e.description, '') = ''
        THEN $description
        ELSE e.description
    END,
    e.profile_text = CASE
        WHEN coalesce(e.profile_text, '') = ''
        THEN $profile_text
        ELSE e.profile_text
    END,
    e.local_keys = CASE
        WHEN coalesce(e.local_keys, []) = []
        THEN $local_keys
        ELSE e.local_keys
    END,
    e.global_keys = CASE
        WHEN coalesce(e.global_keys, []) = []
        THEN $global_keys
        ELSE e.global_keys
    END,
    e.mention_count = coalesce(e.mention_count, 0) + $mention_count,
    e.source_count = CASE
        WHEN $source_count > coalesce(e.source_count, 0)
        THEN $source_count
        ELSE coalesce(e.source_count, 0)
    END,
    e.metadata_json = CASE
        WHEN coalesce(e.metadata_json, '{}') = '{}'
        THEN $metadata_json
        ELSE e.metadata_json
    END,
    e.updated_in_neo4j_at = datetime()

WITH e
UNWIND (coalesce(e.aliases, []) + coalesce($aliases, []) + [null]) AS alias
WITH e, collect(DISTINCT alias) AS all_aliases
SET e.aliases = [x IN all_aliases WHERE x IS NOT NULL AND x <> '']

WITH e
UNWIND (coalesce(e.surface_forms, []) + coalesce($surface_forms, []) + [null]) AS surface_form
WITH e, collect(DISTINCT surface_form) AS all_surface_forms
SET e.surface_forms = [x IN all_surface_forms WHERE x IS NOT NULL AND x <> '']
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
    r.description = CASE
        WHEN coalesce(r.description, '') = ''
        THEN $description
        ELSE r.description
    END,
    r.evidence_text = CASE
        WHEN coalesce(r.evidence_text, '') = ''
        THEN $evidence_text
        ELSE r.evidence_text
    END,
    r.confidence = CASE
        WHEN $confidence > coalesce(r.confidence, 0.0)
        THEN $confidence
        ELSE coalesce(r.confidence, 1.0)
    END,
    r.section = CASE
        WHEN coalesce(r.section, '') = ''
        THEN $section
        ELSE r.section
    END,
    r.source_url = CASE
        WHEN coalesce(r.source_url, '') = ''
        THEN $source_url
        ELSE r.source_url
    END,
    r.metadata_json = CASE
        WHEN coalesce(r.metadata_json, '{}') = '{}'
        THEN $metadata_json
        ELSE r.metadata_json
    END,
    r.updated_in_neo4j_at = datetime()

WITH r
UNWIND (coalesce(r.keywords, []) + coalesce($keywords, []) + [null]) AS keyword
WITH r, collect(DISTINCT keyword) AS all_keywords
SET r.keywords = [x IN all_keywords WHERE x IS NOT NULL AND x <> '']

WITH r
UNWIND (coalesce(r.evidence_chunk_ids, []) + coalesce($evidence_chunk_ids, []) + [null]) AS chunk_id
WITH r, collect(DISTINCT chunk_id) AS all_chunk_ids
SET r.evidence_chunk_ids = [x IN all_chunk_ids WHERE x IS NOT NULL AND x <> '']
"""

# =========================
# Basic lookup
# =========================

GET_ENTITY_BY_ID = (
    """
MATCH (e:Entity {entity_id: $entity_id})
RETURN
"""
    + entity_return("e")
    + """
LIMIT 1
"""
)

FIND_ENTITIES_BY_NORMALIZED_NAME = (
    """
MATCH (e:Entity)
WHERE e.normalized_name = $text
RETURN
"""
    + entity_return("e")
    + """
LIMIT $limit
"""
)

GET_CHUNKS_BY_ENTITY_IDS = (
    """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE e.entity_id IN $entity_ids
RETURN DISTINCT
"""
    + chunk_return("c")
    + """,
    count(e) AS matched_entity_count,
    toFloat(count(e)) AS score
ORDER BY matched_entity_count DESC
LIMIT $limit
"""
)

GET_NEIGHBOR_ENTITIES = (
    """
MATCH (seed:Entity)-[r:MEDICAL_RELATION]-(n:Entity)
WHERE seed.entity_id IN $entity_ids
RETURN DISTINCT
"""
    + entity_return("n")
    + """,
    count(r) AS relation_count,
    toFloat(count(r)) AS score
ORDER BY relation_count DESC
LIMIT $limit
"""
)

GET_SYNONYM_NEIGHBORS = (
    """
MATCH (seed:Entity)-[r:MEDICAL_RELATION]-(syn:Entity)
WHERE seed.entity_id IN $entity_ids
  AND r.relation_type = 'DONG_NGHIA_VOI'
  AND syn.entity_type = seed.entity_type
RETURN DISTINCT
"""
    + entity_return("syn", "coalesce(r.confidence, 1.0)")
    + """
ORDER BY score DESC, coalesce(syn.mention_count, 0) DESC
LIMIT $limit
"""
)

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

GET_ENTITIES_BY_IDS = (
    """
MATCH (e:Entity)
WHERE e.entity_id IN $entity_ids
RETURN
"""
    + entity_return("e", "0.0")
)

GET_ENTITIES_BY_NORMALIZED_NAMES = (
    """
MATCH (e:Entity)
WHERE e.normalized_name IN $normalized_names
RETURN
"""
    + entity_return("e", "0.0")
)

GET_CHUNKS_BY_IDS = (
    """
MATCH (c:Chunk)
WHERE c.chunk_id IN $chunk_ids
RETURN
"""
    + chunk_return("c", "0.0")
    + """
ORDER BY c.chunk_index ASC
"""
)

GET_ALL_CHUNKS = (
    """
MATCH (c:Chunk)
RETURN
"""
    + chunk_return("c", "0.0")
    + """
ORDER BY c.article_id ASC, c.chunk_index ASC
LIMIT $limit
"""
)

GET_ENTITY_CHUNK_COUNTS = """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE e.entity_id IN $entity_ids
RETURN e.entity_id AS entity_id, count(DISTINCT c) AS chunk_count
"""


# =========================
# LightRAG-style retrieval
# =========================

SEARCH_ENTITIES_FOR_LIGHTRAG = (
    """
MATCH (e:Entity)
WITH e,
     toLower($query_text) AS q
WITH e, q,
     (
        CASE WHEN toLower(e.name) CONTAINS q THEN 4 ELSE 0 END +
        CASE WHEN toLower(e.normalized_name) CONTAINS q THEN 4 ELSE 0 END +
        CASE WHEN any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS q) THEN 3 ELSE 0 END +
        CASE WHEN any(sf IN coalesce(e.surface_forms, []) WHERE toLower(sf) CONTAINS q) THEN 3 ELSE 0 END +
        CASE WHEN toLower(coalesce(e.description, '')) CONTAINS q THEN 2 ELSE 0 END +
        CASE WHEN toLower(coalesce(e.profile_text, '')) CONTAINS q THEN 2 ELSE 0 END
     ) AS score
WHERE score > 0
  AND ($entity_types IS NULL OR e.entity_type IN $entity_types)
RETURN
"""
    + entity_return("e", "score")
    + """
ORDER BY score DESC, coalesce(e.mention_count, 0) DESC
LIMIT $limit
"""
)

SEARCH_RELATIONS_FOR_LIGHTRAG = (
    """
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WITH s, r, o, toLower($query_text) AS q
WITH s, r, o, q,
     (
        CASE WHEN toLower(coalesce(r.relation_type, '')) CONTAINS q THEN 2 ELSE 0 END +
        CASE WHEN toLower(coalesce(r.description, '')) CONTAINS q THEN 4 ELSE 0 END +
        CASE WHEN toLower(coalesce(r.evidence_text, '')) CONTAINS q THEN 3 ELSE 0 END +
        CASE WHEN any(k IN coalesce(r.keywords, []) WHERE toLower(k) CONTAINS q) THEN 4 ELSE 0 END +
        CASE WHEN toLower(s.name) CONTAINS q THEN 3 ELSE 0 END +
        CASE WHEN toLower(o.name) CONTAINS q THEN 3 ELSE 0 END +
        CASE WHEN any(a IN coalesce(s.aliases, []) WHERE toLower(a) CONTAINS q) THEN 2 ELSE 0 END +
        CASE WHEN any(a IN coalesce(o.aliases, []) WHERE toLower(a) CONTAINS q) THEN 2 ELSE 0 END +
        CASE WHEN any(sf IN coalesce(s.surface_forms, []) WHERE toLower(sf) CONTAINS q) THEN 2 ELSE 0 END +
        CASE WHEN any(sf IN coalesce(o.surface_forms, []) WHERE toLower(sf) CONTAINS q) THEN 2 ELSE 0 END
     ) AS score
WHERE score > 0
  AND ($relation_types IS NULL OR r.relation_type IN $relation_types)
RETURN
"""
    + relation_return(score="score")
    + """
ORDER BY score DESC, coalesce(r.confidence, 1.0) DESC
LIMIT $limit
"""
)

GET_RELATIONS_BY_ENTITY_IDS = (
    """
MATCH (s:Entity)-[r:MEDICAL_RELATION]-(o:Entity)
WHERE (s.entity_id IN $entity_ids OR o.entity_id IN $entity_ids)
  AND ($relation_types IS NULL OR r.relation_type IN $relation_types)
RETURN DISTINCT
"""
    + relation_return(
        score="coalesce(r.confidence, 1.0)",
        use_start_end=True,
    )
    + """
ORDER BY score DESC
LIMIT $limit
"""
)

GET_CHUNKS_BY_RELATION_IDS = (
    """
MATCH ()-[r:MEDICAL_RELATION]->()
WHERE r.relation_id IN $relation_ids
UNWIND coalesce(r.evidence_chunk_ids, []) AS chunk_id
WITH chunk_id, count(DISTINCT r) AS matched_relation_count
MATCH (c:Chunk {chunk_id: chunk_id})
RETURN
"""
    + chunk_return("c", "toFloat(matched_relation_count)")
    + """
ORDER BY score DESC, c.chunk_index ASC
LIMIT $limit
"""
)

GET_RELATIONS_BY_IDS = (
    """
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WHERE r.relation_id IN $relation_ids
RETURN
"""
    + relation_return(score="coalesce(r.confidence, 1.0)")
    + """
ORDER BY score DESC
"""
)


# =========================
# HippoRAG-style retrieval
# =========================

FIND_SEED_ENTITIES = (
    """
MATCH (e:Entity)
WITH e, toLower($query_text) AS q
WITH e, q,
     (
        CASE WHEN toLower(e.name) CONTAINS q THEN 5 ELSE 0 END +
        CASE WHEN toLower(e.normalized_name) CONTAINS q THEN 5 ELSE 0 END +
        CASE WHEN any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS q) THEN 4 ELSE 0 END +
        CASE WHEN any(sf IN coalesce(e.surface_forms, []) WHERE toLower(sf) CONTAINS q) THEN 4 ELSE 0 END +
        CASE WHEN toLower(coalesce(e.description, '')) CONTAINS q THEN 2 ELSE 0 END +
        CASE WHEN toLower(coalesce(e.profile_text, '')) CONTAINS q THEN 2 ELSE 0 END
     ) AS score
WHERE score > 0
RETURN
"""
    + entity_return("e", "score")
    + """
ORDER BY score DESC, coalesce(e.mention_count, 0) DESC
LIMIT $limit
"""
)

GET_ENTITY_ADJACENCY = (
    """
MATCH (s:Entity)-[r:MEDICAL_RELATION]->(o:Entity)
WHERE ($relation_types IS NULL OR r.relation_type IN $relation_types)
RETURN
"""
    + ADJACENCY_RETURN_FIELDS
    + """
LIMIT $limit
"""
)

GET_ENTITY_ADJACENCY_AROUND_SEEDS_1HOP = adjacency_around_seeds_query(1)

GET_ENTITY_ADJACENCY_AROUND_SEEDS_2HOP = adjacency_around_seeds_query(2)

GET_ENTITY_ADJACENCY_AROUND_SEEDS_3HOP = adjacency_around_seeds_query(3)

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