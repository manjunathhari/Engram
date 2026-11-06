"""The typed records Engram's knowledge is made of. Frozen and checked, so a bad record is refused before it is stored."""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

FACT_TYPES = ("attribute", "relationship", "event", "preference", "plan", "observation")
STATUSES = ("active", "superseded", "disputed", "retracted", "needs_review", "invalidated")
SETTABLE_STATUSES = ("active", "superseded", "disputed", "retracted")  # needs_review and invalidated come from deletion
VALIDATION = ("pending", "accepted", "rejected")
SOURCE_TYPES = ("explicit", "inferred")
OBJECT_TYPES = ("text", "number", "date", "boolean", "json", "entity")
EVIDENCE_ROLES = ("supports", "contradicts", "qualifies")  # qualifies limits a fact without supporting or contradicting it
ALIAS_SOURCES = ("sentence", "manual")
DATE = re.compile(r"\d{4}(-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?)?")

class RecordError(ValueError):
    """A record was refused. Nothing was saved."""


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


def encode_value(object_type, value):
    """A Python value as the text stored for it, so numbers, true or false and structured values come back as they went in."""
    if value is None: return None
    if object_type == "number":
        if isinstance(value, bool): raise RecordError("a number cannot be true or false")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise RecordError("a number value must be a number") from None
        if number != number or number in (float("inf"), float("-inf")): raise RecordError("a number value must be finite")
        return str(int(number)) if number.is_integer() and "." not in str(value) and "e" not in str(value).lower() else repr(number)
    if object_type == "boolean":
        if isinstance(value, bool): return "true" if value else "false"
        if str(value).strip().lower() in ("true", "false"): return str(value).strip().lower()
        raise RecordError("a boolean value must be true or false")
    if object_type == "json":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                raise RecordError("a json value must be valid JSON") from None
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return tidy(value)


def decode_value(object_type, text):
    if text is None: return None
    if object_type == "number":
        number = float(text)
        return int(number) if re.fullmatch(r"-?\d+", text) else number
    if object_type == "boolean": return text == "true"
    if object_type == "json": return json.loads(text)
    return text


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

    @property
    def value(self):
        """The object as a Python value: a number, true or false, a structure, or text."""
        return decode_value(self.object_type, self.object_value)

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
        if has_value and self.object_type != "entity" and encode_value(self.object_type, self.object_value) != self.object_value: raise RecordError(f"object value is not a stored {self.object_type}")
        if self.object_type == "date": _period(self.object_value, "a date value")
        _period(self.valid_from, "valid_from"), _period(self.valid_to, "valid_to")
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from: raise RecordError("valid_to is before valid_from")

    @classmethod
    def create(cls, subject_id, predicate, object_type, fact_type, source_type, object_value=None, object_entity_id=None, status="active", validation_status="pending", valid_from=None, valid_to=None, recorded_at=None, metadata=None, note=""):
        """fact_type and source_type have no default: an inference must never pass silently as a statement."""
        return cls(new_id("fact"), subject_id, tidy(predicate), object_type, encode_value(object_type, object_value) if object_type in OBJECT_TYPES and object_type != "entity" else (None if object_value is None else tidy(object_value)), object_entity_id, fact_type, status, validation_status, source_type, valid_from, valid_to, recorded_at or utc_now(), metadata, note)


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


