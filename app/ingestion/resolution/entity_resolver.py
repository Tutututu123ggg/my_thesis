from dataclasses import dataclass

from app.infrastructure.graph_database import EntityNode, GraphRepository
from app.ingestion.extraction import ExtractedEntity
from app.ingestion.resolution.normalizer import (
    clean_surface_text,
    make_entity_id,
    normalize_text,
)


@dataclass
class ResolvedEntity:
    entity_id: str
    name: str
    normalized_name: str
    entity_type: str
    aliases: list[str]
    surface_forms: list[str]
    description: str
    merge_method: str


class EntityResolver:
    """
    Resolve extracted entity về canonical entity theo exact-name policy.

    Không dùng lexicon thủ công.
    Không fuzzy match.
    Không embedding merge.
    Không LLM canonicalizer.
    Không dùng aliases/surface_forms để hard merge.

    Hard merge chỉ dựa trên:
    - same entity_type
    - same normalize(entity.name)

    Các tên khác nhau nhưng text chỉ đích danh là đồng nghĩa sẽ được biểu diễn
    bằng relation DONG_NGHIA_VOI trong graph, không rewrite entity_id.
    """

    def __init__(self, graph_repo: GraphRepository):
        self.graph_repo = graph_repo

    def resolve(self, entity: ExtractedEntity) -> ResolvedEntity:
        canonical_name = clean_surface_text(entity.name)

        existing = self._find_existing_entity(
            name=canonical_name,
            entity_type=entity.entity_type,
        )

        if existing:
            aliases = self._dedupe_texts(existing.aliases + entity.aliases)
            surface_forms = self._dedupe_texts(
                existing.surface_forms + entity.surface_forms + [canonical_name]
            )
            description = entity.description.strip() or (existing.description or "")

            return ResolvedEntity(
                entity_id=existing.entity_id,
                name=existing.name,
                normalized_name=existing.normalized_name,
                entity_type=existing.entity_type,
                aliases=aliases,
                surface_forms=surface_forms,
                description=description,
                merge_method="exact_name",
            )

        aliases = self._dedupe_texts(entity.aliases)
        surface_forms = self._dedupe_texts(entity.surface_forms + [canonical_name])

        return ResolvedEntity(
            entity_id=make_entity_id(canonical_name, entity.entity_type),
            name=canonical_name,
            normalized_name=normalize_text(canonical_name),
            entity_type=entity.entity_type,
            aliases=aliases,
            surface_forms=surface_forms,
            description=entity.description.strip(),
            merge_method="new",
        )

    def to_entity_node(
        self,
        resolved: ResolvedEntity,
        mention_count: int = 1,
        source_count: int = 1,
    ) -> EntityNode:
        """
        Convert ResolvedEntity sang EntityNode để GraphRepository ghi Neo4j.

        profile_text hiện lấy từ description.
        local_keys/global_keys để rỗng vì LightRAG keywords sinh ở query-time.
        """
        return EntityNode(
            entity_id=resolved.entity_id,
            name=resolved.name,
            normalized_name=resolved.normalized_name,
            entity_type=resolved.entity_type,
            aliases=resolved.aliases,
            surface_forms=resolved.surface_forms,
            description=resolved.description,
            profile_text=resolved.description,
            local_keys=[],
            global_keys=[],
            mention_count=mention_count,
            source_count=source_count,
            metadata={
                "merge_method": resolved.merge_method,
            },
        )

    def _find_existing_entity(
        self,
        name: str,
        entity_type: str,
    ) -> EntityNode | None:
        normalized = normalize_text(name)

        candidates = self.graph_repo.find_entities_by_normalized_name(
            text=normalized,
            limit=10,
        )

        for candidate in candidates:
            if candidate.entity_type != entity_type:
                continue

            if candidate.normalized_name == normalized:
                return candidate

        return None

    def _dedupe_texts(self, texts: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for text in texts:
            cleaned = clean_surface_text(text)
            if not cleaned:
                continue

            key = normalize_text(cleaned)
            if key in seen:
                continue

            result.append(cleaned)
            seen.add(key)

        return result
