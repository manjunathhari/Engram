"""Engram knowledge representation: typed records, always traceable to the original sentences.

Stored beside the sentences. It does not change how sentences are stored or searched, and nothing here writes records by
itself: no extraction, reasoning or automatic name matching.

valid_from / valid_to  when the information became true and stopped being true, if known
recorded_at            when Engram learned it
status                 the record's life: active, superseded, disputed, retracted, needs_review, invalidated
validation_status      whether it has been checked: pending, accepted, rejected
source_type            explicit (stated) or inferred; always chosen, so an inference cannot pass as a statement
evidence               the original sentences behind it

If a supporting sentence is deleted, a fact still supported by another is marked needs_review. A fact with no support left
is invalidated and its content erased, so deleted information does not survive here.
"""
import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

SCHEMA_VERSION = 1
FACT_TYPES = ("attribute", "relationship", "event", "preference", "plan", "observation")
STATUSES = ("active", "superseded", "disputed", "retracted", "needs_review", "invalidated")
SETTABLE_STATUSES = ("active", "superseded", "disputed", "retracted")  # needs_review and invalidated come from deletion
VALIDATION = ("pending", "accepted", "rejected")
SOURCE_TYPES = ("explicit", "inferred")
OBJECT_TYPES = ("text", "number", "date", "entity")
EVIDENCE_ROLES = ("supports", "contradicts")
ALIAS_SOURCES = ("sentence", "manual")
DATE = re.compile(r"\d{4}(-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?)?")

SCHEMA = """
    CREATE TABLE IF NOT EXISTS knowledge_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS entities(entity_id TEXT PRIMARY KEY, canonical_name TEXT NOT NULL, normalized_name TEXT NOT NULL UNIQUE, entity_type TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}', schema_version INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS entity_aliases(alias_id TEXT PRIMARY KEY, entity_id TEXT NOT NULL, alias TEXT NOT NULL, normalized_alias TEXT NOT NULL UNIQUE, source_kind TEXT NOT NULL, source_id TEXT, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS knowledge_records(fact_id TEXT PRIMARY KEY, subject_id TEXT, predicate TEXT, object_type TEXT, object_value TEXT, object_entity_id TEXT, fact_type TEXT NOT NULL, status TEXT NOT NULL, validation_status TEXT NOT NULL, source_type TEXT NOT NULL, valid_from TEXT, valid_to TEXT, recorded_at TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}', note TEXT NOT NULL DEFAULT '', schema_version INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS record_evidence(evidence_id TEXT PRIMARY KEY, fact_id TEXT NOT NULL, source_kind TEXT NOT NULL, source_id TEXT NOT NULL, evidence_role TEXT NOT NULL, extraction_method TEXT, extraction_version TEXT, created_at TEXT NOT NULL, UNIQUE(fact_id, source_kind, source_id));
    CREATE TABLE IF NOT EXISTS review_queue(review_id TEXT PRIMARY KEY, candidate TEXT NOT NULL, reasons TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS review_sources(review_id TEXT NOT NULL, sid INTEGER NOT NULL, PRIMARY KEY(review_id, sid));
    CREATE INDEX IF NOT EXISTS review_sources_sid ON review_sources(sid);
    CREATE INDEX IF NOT EXISTS record_evidence_source ON record_evidence(source_kind, source_id);
    CREATE INDEX IF NOT EXISTS record_evidence_fact ON record_evidence(fact_id);
    CREATE INDEX IF NOT EXISTS knowledge_records_subject ON knowledge_records(subject_id);
    CREATE INDEX IF NOT EXISTS entity_aliases_source ON entity_aliases(source_kind, source_id);
"""


class RecordError(ValueError):
    """A record was refused. Nothing was saved."""


def prepare(db) -> None:
    """Checks the version first, so a database from other code is refused before anything in it changes."""
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_meta'").fetchone():
        row = db.execute("SELECT value FROM knowledge_meta WHERE key='schema_version'").fetchone()
        if row and row[0] != str(SCHEMA_VERSION):
            raise RecordError(f"this database's knowledge records are version {row[0]}, but this code uses version {SCHEMA_VERSION}. Nothing was changed.")
    db.executescript(SCHEMA)
    db.execute("INSERT OR IGNORE INTO knowledge_meta VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
    db.commit()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tidy(text) -> str:
    return " ".join(str(text or "").split())


def normalize_name(name) -> str:
    """Same apart from case and spacing. Similar names are never treated as the same."""
    return tidy(name).casefold()


def _timestamp(value, label):
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise RecordError(f"{label} must be an ISO date and time") from None
    if moment.tzinfo is None: raise RecordError(f"{label} must include a time zone")
    return value


def _period(value, label):
    if value is not None and not (isinstance(value, str) and DATE.fullmatch(value)):
        raise RecordError(f"{label} must be YYYY, YYYY-MM or YYYY-MM-DD")


def _choice(value, allowed, label):
    if value not in allowed: raise RecordError(f"{label} must be one of {allowed}")


def _meta(value) -> Mapping[str, Any]:
    """A read-only copy, so a frozen record is really frozen."""
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class Entity:
    entity_id: str
    canonical_name: str
    entity_type: str
    created_at: str
    updated_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self):
        if not tidy(self.canonical_name): raise RecordError("an entity needs a name")
        if not tidy(self.entity_type): raise RecordError("an entity needs a type")
        object.__setattr__(self, "metadata", _meta(self.metadata))

    @classmethod
    def create(cls, canonical_name, entity_type, timestamp=None, metadata=None):
        moment = _timestamp(timestamp or utc_now(), "timestamp")
        return cls(new_id("ent"), tidy(canonical_name), tidy(entity_type).lower(), moment, moment, metadata)


@dataclass(frozen=True)
class EntityAlias:
    alias_id: str
    entity_id: str
    alias: str
    normalized_alias: str
    source_kind: str
    source_id: str | None
    created_at: str

    def __post_init__(self):
        if not self.normalized_alias: raise RecordError("an alias cannot be blank")
        _choice(self.source_kind, ALIAS_SOURCES, "alias source")
        if self.source_kind == "sentence" and not self.source_id: raise RecordError("an alias needs a source sentence, or must be marked manual")
        if self.source_kind == "manual" and self.source_id: raise RecordError("a manual alias has no source sentence")

    @classmethod
    def create(cls, entity_id, alias, source_kind, source_id=None, timestamp=None):
        clean = tidy(alias)
        return cls(new_id("als"), entity_id, clean, clean.casefold(), source_kind, None if source_id is None else str(source_id), _timestamp(timestamp or utc_now(), "timestamp"))


@dataclass(frozen=True)
class KnowledgeFact:
    fact_id: str
    subject_id: str | None
    predicate: str | None
    object_type: str | None
    object_value: str | None
    object_entity_id: str | None
    fact_type: str
    status: str
    validation_status: str
    source_type: str
    valid_from: str | None
    valid_to: str | None
    recorded_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)
    note: str = ""

    def __post_init__(self):
        _choice(self.fact_type, FACT_TYPES, "fact type")
        _choice(self.status, STATUSES, "status")
        _choice(self.validation_status, VALIDATION, "validation status")
        _choice(self.source_type, SOURCE_TYPES, "source type")
        _timestamp(self.recorded_at, "recorded_at")
        object.__setattr__(self, "metadata", _meta(self.metadata))
        if self.status == "invalidated": return  # its content has been erased
        if not self.subject_id: raise RecordError("a fact needs a subject")
        if not tidy(self.predicate): raise RecordError("a fact needs a predicate")
        _choice(self.object_type, OBJECT_TYPES, "object type")
        has_value, has_entity = bool(tidy(self.object_value)), bool(self.object_entity_id)
        if has_value == has_entity: raise RecordError("a fact needs exactly one of a value or an object entity")
        if has_entity != (self.object_type == "entity"): raise RecordError("object type does not match the object")
        _period(self.valid_from, "valid_from"), _period(self.valid_to, "valid_to")
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from: raise RecordError("valid_to is before valid_from")

    @classmethod
    def create(cls, subject_id, predicate, object_type, fact_type, source_type, object_value=None, object_entity_id=None, status="active", validation_status="pending", valid_from=None, valid_to=None, recorded_at=None, metadata=None, note=""):
        """fact_type and source_type have no default: an inference must never pass silently as a statement."""
        return cls(new_id("fact"), subject_id, tidy(predicate), object_type, tidy(object_value) if object_value is not None else None, object_entity_id, fact_type, status, validation_status, source_type, valid_from, valid_to, recorded_at or utc_now(), metadata, note)


@dataclass(frozen=True)
class FactEvidence:
    evidence_id: str
    fact_id: str
    source_kind: str
    source_id: str
    evidence_role: str
    extraction_method: str | None
    extraction_version: str | None
    created_at: str

    def __post_init__(self):
        if self.source_kind != "sentence": raise RecordError("evidence must be a sentence")
        if not self.source_id: raise RecordError("evidence needs a source")
        _choice(self.evidence_role, EVIDENCE_ROLES, "evidence role")

    @classmethod
    def create(cls, fact_id, source_id, evidence_role="supports", extraction_method=None, extraction_version=None, timestamp=None):
        return cls(new_id("ev"), fact_id, "sentence", str(source_id), evidence_role, extraction_method, extraction_version, _timestamp(timestamp or utc_now(), "timestamp"))


@dataclass(frozen=True)
class FactWithEvidence:
    fact: KnowledgeFact
    evidence: tuple[FactEvidence, ...]

    def __post_init__(self):
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if any(e.fact_id != self.fact.fact_id for e in self.evidence): raise RecordError("evidence belongs to a different fact")
        if self.fact.status != "invalidated" and not any(e.evidence_role == "supports" for e in self.evidence):
            raise RecordError("a fact needs at least one supporting sentence")


@dataclass(frozen=True)
class SourceSentence:
    sid: int
    chat: int
    speaker: str
    date: str
    text: str
    caption: str
    evidence_role: str


class KnowledgeBase:
    """Works on ChatMemory's own database. Changes only its own four tables."""

    def __init__(self, db):
        self.db = db
        prepare(db)

    def _sentence_exists(self, source_id) -> bool:
        return str(source_id).isdigit() and self.db.execute("SELECT 1 FROM sentences WHERE sid=?", (int(source_id),)).fetchone() is not None

    def _write(self, action):
        try:
            result = action()
            self.db.commit()
            return result
        except Exception:
            self.db.rollback()
            raise

    # ---------- names ----------
    def add_entity(self, entity: Entity) -> Entity:
        key = normalize_name(entity.canonical_name)
        if self.resolve(entity.canonical_name): raise RecordError("that name already belongs to an entity")
        self._write(lambda: self.db.execute("INSERT INTO entities VALUES(?,?,?,?,?,?,?,?)", (entity.entity_id, entity.canonical_name, key, entity.entity_type, entity.created_at, entity.updated_at, json.dumps(dict(entity.metadata)), SCHEMA_VERSION)))
        return entity

    def resolve(self, name) -> str | None:
        """An entity's id by its exact name or a stated alias. Never guesses."""
        key = normalize_name(name)
        if not key: return None
        row = self.db.execute("SELECT entity_id FROM entities WHERE normalized_name=?", (key,)).fetchone() or self.db.execute("SELECT entity_id FROM entity_aliases WHERE normalized_alias=?", (key,)).fetchone()
        return row[0] if row else None

    def entity(self, entity_id) -> Entity | None:
        row = self.db.execute("SELECT entity_id, canonical_name, entity_type, created_at, updated_at, metadata FROM entities WHERE entity_id=?", (entity_id,)).fetchone()
        return Entity(*row[:5], json.loads(row[5])) if row else None

    def add_alias(self, alias: EntityAlias) -> EntityAlias:
        if not self.entity(alias.entity_id): raise RecordError("no such entity")
        if alias.source_kind == "sentence" and not self._sentence_exists(alias.source_id): raise RecordError(f"source sentence does not exist: {alias.source_id}")
        holder = self.resolve(alias.alias)
        if holder and holder != alias.entity_id: raise RecordError("that alias belongs to another entity; they were not merged")
        if holder: return alias  # already known by this name
        self._write(lambda: self.db.execute("INSERT INTO entity_aliases VALUES(?,?,?,?,?,?,?)", (alias.alias_id, alias.entity_id, alias.alias, alias.normalized_alias, alias.source_kind, alias.source_id, alias.created_at)))
        return alias

    # ---------- waiting for a person ----------
    def add_review(self, candidate: str, reasons: list, sids: list) -> str:
        """Holds an uncertain candidate. It is not knowledge: nothing reads it as a fact."""
        review_id = new_id("rev")

        def save():
            self.db.execute("INSERT INTO review_queue VALUES(?,?,?,?)", (review_id, candidate, json.dumps(reasons), utc_now()))
            self.db.executemany("INSERT OR IGNORE INTO review_sources VALUES(?,?)", [(review_id, s) for s in sids])
        self._write(save)
        return review_id

    def review(self, review_id):
        row = self.db.execute("SELECT review_id, candidate, reasons, created_at FROM review_queue WHERE review_id=?", (review_id,)).fetchone()
        return dict(review_id=row[0], candidate=row[1], reasons=json.loads(row[2]), created_at=row[3]) if row else None

    def reviews(self):
        return [self.review(r[0]) for r in self.db.execute("SELECT review_id FROM review_queue ORDER BY created_at, review_id").fetchall()]

    def drop_review(self, review_id):
        self._write(lambda: (self.db.execute("DELETE FROM review_sources WHERE review_id=?", (review_id,)), self.db.execute("DELETE FROM review_queue WHERE review_id=?", (review_id,))))

    # ---------- facts ----------
    def add_evidence(self, evidence: FactEvidence):
        """One more sentence behind a fact already stored."""
        if not self._sentence_exists(evidence.source_id): raise RecordError(f"source sentence does not exist: {evidence.source_id}")
        self._write(lambda: self.db.execute("INSERT OR IGNORE INTO record_evidence VALUES(?,?,?,?,?,?,?,?)", (evidence.evidence_id, evidence.fact_id, evidence.source_kind, evidence.source_id, evidence.evidence_role, evidence.extraction_method, evidence.extraction_version, evidence.created_at)))

    def add(self, record: FactWithEvidence) -> FactWithEvidence:
        """Saves a fact and its evidence together. Every check comes first, so a refused record changes nothing."""
        f = record.fact
        if f.status not in SETTABLE_STATUSES: raise RecordError(f"a new fact's status must be one of {SETTABLE_STATUSES}")
        if not self.entity(f.subject_id): raise RecordError("no such subject entity")
        if f.object_entity_id and not self.entity(f.object_entity_id): raise RecordError("no such object entity")
        seen = set()
        for e in record.evidence:
            if not self._sentence_exists(e.source_id): raise RecordError(f"source sentence does not exist: {e.source_id}")
            if e.source_id in seen: raise RecordError(f"source sentence listed twice: {e.source_id}")
            seen.add(e.source_id)

        def save():
            self.db.execute("INSERT INTO knowledge_records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (f.fact_id, f.subject_id, f.predicate, f.object_type, f.object_value, f.object_entity_id, f.fact_type, f.status, f.validation_status, f.source_type, f.valid_from, f.valid_to, f.recorded_at, json.dumps(dict(f.metadata)), f.note, SCHEMA_VERSION))
            self.db.executemany("INSERT INTO record_evidence VALUES(?,?,?,?,?,?,?,?)", [(e.evidence_id, e.fact_id, e.source_kind, e.source_id, e.evidence_role, e.extraction_method, e.extraction_version, e.created_at) for e in record.evidence])
        self._write(save)
        return self.get(f.fact_id)

    def get(self, fact_id) -> FactWithEvidence | None:
        row = self.db.execute("SELECT fact_id, subject_id, predicate, object_type, object_value, object_entity_id, fact_type, status, validation_status, source_type, valid_from, valid_to, recorded_at, metadata, note FROM knowledge_records WHERE fact_id=?", (fact_id,)).fetchone()
        if not row: return None
        fact = KnowledgeFact(*row[:13], json.loads(row[13]), row[14])
        rows = self.db.execute("SELECT evidence_id, fact_id, source_kind, source_id, evidence_role, extraction_method, extraction_version, created_at FROM record_evidence WHERE fact_id=? ORDER BY CAST(source_id AS INTEGER)", (fact_id,)).fetchall()
        return FactWithEvidence(fact, tuple(FactEvidence(*r) for r in rows))

    def about(self, name, statuses=("active",), source_types=("explicit",)) -> list[FactWithEvidence]:
        """Facts about an entity. By default active and stated only: inferences must be asked for. None means all."""
        entity_id = self.resolve(name)
        if not entity_id: return []
        rows = self.db.execute("SELECT fact_id FROM knowledge_records WHERE subject_id=? ORDER BY recorded_at, fact_id", (entity_id,)).fetchall()
        found = [self.get(r[0]) for r in rows]
        return [x for x in found if (statuses is None or x.fact.status in statuses) and (source_types is None or x.fact.source_type in source_types)]

    def trace(self, fact_id) -> list[SourceSentence]:
        """The exact original sentences behind a fact, read from the permanent record."""
        rows = self.db.execute("""SELECT s.sid, s.chat, s.speaker, s.date, s.text, s.caption, e.evidence_role FROM record_evidence e
                                  JOIN sentences s ON s.sid = CAST(e.source_id AS INTEGER) WHERE e.fact_id=? AND e.source_kind='sentence' ORDER BY s.sid""", (fact_id,)).fetchall()
        return [SourceSentence(*r) for r in rows]

    def set_status(self, fact_id, status, note="", valid_to=None) -> FactWithEvidence:
        """Marks a fact superseded, disputed, retracted or active again. Its content and evidence stay."""
        if status not in SETTABLE_STATUSES: raise RecordError(f"status must be one of {SETTABLE_STATUSES}")
        current = self.get(fact_id)
        if current is None: raise RecordError("no such fact")
        if current.fact.status == "invalidated": raise RecordError("an invalidated fact has no content and cannot return")
        updated = replace(current.fact, status=status, note=note or current.fact.note, valid_to=valid_to if valid_to else current.fact.valid_to)  # replace runs the checks again
        self._write(lambda: self.db.execute("UPDATE knowledge_records SET status=?, note=?, valid_to=? WHERE fact_id=?", (updated.status, updated.note, updated.valid_to, fact_id)))
        return self.get(fact_id)

    def set_validation(self, fact_id, validation_status, note="") -> FactWithEvidence:
        """Records whether a fact has been checked. This never changes whether it was stated or inferred."""
        _choice(validation_status, VALIDATION, "validation status")
        current = self.get(fact_id)
        if current is None: raise RecordError("no such fact")
        if current.fact.status == "invalidated": raise RecordError("an invalidated fact has no content and cannot return")
        self._write(lambda: self.db.execute("UPDATE knowledge_records SET validation_status=?, note=? WHERE fact_id=?", (validation_status, note or current.fact.note, fact_id)))
        return self.get(fact_id)


def forget_sentence(db, sid: int) -> dict:
    """Called inside ChatMemory.forget_sentence, before its commit, so a sentence and what rests on it go together."""
    source = str(sid)
    for row in db.execute("SELECT review_id FROM review_sources WHERE sid=?", (sid,)).fetchall():  # a waiting candidate quotes its sentence
        db.execute("DELETE FROM review_sources WHERE review_id=?", row)
        db.execute("DELETE FROM review_queue WHERE review_id=?", row)
    touched = {r[0] for r in db.execute("SELECT entity_id FROM entity_aliases WHERE source_kind='sentence' AND source_id=?", (source,))}
    db.execute("DELETE FROM entity_aliases WHERE source_kind='sentence' AND source_id=?", (source,))
    facts = [r[0] for r in db.execute("SELECT fact_id FROM record_evidence WHERE source_kind='sentence' AND source_id=?", (source,)).fetchall()]
    db.execute("DELETE FROM record_evidence WHERE source_kind='sentence' AND source_id=?", (source,))
    review = invalid = 0
    for fact in facts:
        row = db.execute("SELECT subject_id, object_entity_id FROM knowledge_records WHERE fact_id=?", (fact,)).fetchone()
        supported = db.execute("SELECT 1 FROM record_evidence WHERE fact_id=? AND evidence_role='supports'", (fact,)).fetchone()
        if supported:
            db.execute("UPDATE knowledge_records SET status='needs_review', note='A linked sentence was deleted' WHERE fact_id=?", (fact,))
            review += 1
        else:  # nothing supports it now: the content goes too, leaving only that it once existed
            touched.update(x for x in row if x)
            db.execute("DELETE FROM record_evidence WHERE fact_id=?", (fact,))
            db.execute("UPDATE knowledge_records SET status='invalidated', subject_id=NULL, predicate=NULL, object_type=NULL, object_value=NULL, object_entity_id=NULL, valid_from=NULL, valid_to=NULL, metadata='{}', note='All supporting sentences were deleted' WHERE fact_id=?", (fact,))
            invalid += 1
    for entity in touched:
        used = db.execute("SELECT 1 FROM knowledge_records WHERE subject_id=? OR object_entity_id=? LIMIT 1", (entity, entity)).fetchone() or db.execute("SELECT 1 FROM entity_aliases WHERE entity_id=? LIMIT 1", (entity,)).fetchone()
        if not used: db.execute("DELETE FROM entities WHERE entity_id=?", (entity,))
    return dict(needs_review=review, invalidated=invalid)
