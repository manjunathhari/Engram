"""Names: entities and their aliases. A name is never guessed, and two entities may share a name."""
import json

from .models import Entity, EntityAlias, RecordError, normalize_name
from .schema import SCHEMA_VERSION


class EntityStore:
    def add_entity(self, entity: Entity) -> Entity:
        """Two different people may have the same name, so a name already in use does not stop a new entity."""
        self._write(lambda: self.db.execute("INSERT INTO entities VALUES(?,?,?,?,?,?,?,?)", (entity.entity_id, entity.canonical_name, normalize_name(entity.canonical_name), entity.entity_type, entity.created_at, entity.updated_at, json.dumps(dict(entity.metadata)), SCHEMA_VERSION)))
        return entity

    def find(self, name) -> list[str]:
        """Every entity known by this exact name or a stated alias, oldest first. Never merges them."""
        key = normalize_name(name)
        if not key: return []
        rows = self.db.execute("""SELECT entity_id FROM entities WHERE normalized_name=?
                                  UNION SELECT entity_id FROM entity_aliases WHERE normalized_alias=?""", (key, key)).fetchall()
        ids = {r[0] for r in rows}
        return [r[0] for r in self.db.execute("SELECT entity_id FROM entities ORDER BY created_at, entity_id").fetchall() if r[0] in ids]

    def resolve(self, name) -> str | None:
        """The one entity with this name or alias. None if there is none, or more than one: then it is for a person to say which."""
        found = self.find(name)
        return found[0] if len(found) == 1 else None

    def entity(self, entity_id) -> Entity | None:
        row = self.db.execute("SELECT entity_id, canonical_name, entity_type, created_at, updated_at, metadata FROM entities WHERE entity_id=?", (entity_id,)).fetchone()
        return Entity(*row[:5], json.loads(row[5])) if row else None

    def aliases(self, entity_id) -> list[EntityAlias]:
        rows = self.db.execute("SELECT alias_id, entity_id, alias, normalized_alias, source_kind, source_id, created_at FROM entity_aliases WHERE entity_id=? ORDER BY created_at, alias_id", (entity_id,)).fetchall()
        return [EntityAlias(*r) for r in rows]

    def add_alias(self, alias: EntityAlias) -> EntityAlias:
        """Another name for an entity. It may also belong to someone else; lookups then report both."""
        if not self.entity(alias.entity_id): raise RecordError("no such entity")
        if alias.source_kind == "sentence" and not self._sentence_exists(alias.source_id): raise RecordError(f"source sentence does not exist: {alias.source_id}")
        mine = next((a for a in self.aliases(alias.entity_id) if a.normalized_alias == alias.normalized_alias), None)
        if mine or normalize_name(self.entity(alias.entity_id).canonical_name) == alias.normalized_alias: return mine or alias  # already known by this name
        self._write(lambda: self.db.execute("INSERT INTO entity_aliases VALUES(?,?,?,?,?,?,?)", (alias.alias_id, alias.entity_id, alias.alias, alias.normalized_alias, alias.source_kind, alias.source_id, alias.created_at)))
        return alias
