"""A standard way to represent knowledge, always traceable to the original sentences.

This does not change how sentences are stored or searched, and nothing here writes records by itself. A record says who
or what it is about, what is held, when it was true, when Engram learned it, its status, and which sentences support it.

valid_from / valid_to  when the information became true and stopped being true, if known
recorded_at            when Engram learned it
source_ids             the original sentences that support it
status                 active, superseded, disputed, retracted, needs_review or invalidated

If a supporting sentence is deleted, a record with other sources is marked needs_review. A record with no sources left is
invalidated and its content is erased, so deleted information does not survive here.
"""
import re
import time
from dataclasses import dataclass, field

FACT_TYPES = ("attribute", "relationship", "event", "preference", "plan", "observation")
STATUSES = ("active", "superseded", "disputed", "retracted", "needs_review", "invalidated")
# A person or caller may set these. needs_review and invalidated come from deleting sentences.
SETTABLE = ("active", "superseded", "disputed", "retracted")
DATE = re.compile(r"\d{4}(-\d{2}(-\d{2})?)?")

SCHEMA = """
    CREATE TABLE IF NOT EXISTS entities(entity INTEGER PRIMARY KEY, name TEXT NOT NULL, key TEXT NOT NULL UNIQUE, kind TEXT NOT NULL DEFAULT '');
    CREATE TABLE IF NOT EXISTS knowledge_records(record INTEGER PRIMARY KEY, subject INTEGER, fact_type TEXT NOT NULL, predicate TEXT, value TEXT, object INTEGER, valid_from TEXT, valid_to TEXT, recorded_at REAL NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL DEFAULT '');
    CREATE TABLE IF NOT EXISTS record_evidence(record INTEGER NOT NULL, sid INTEGER NOT NULL, PRIMARY KEY(record, sid));
    CREATE INDEX IF NOT EXISTS record_evidence_sid ON record_evidence(sid);
    CREATE INDEX IF NOT EXISTS knowledge_records_subject ON knowledge_records(subject);
"""


class RecordError(ValueError):
    """A record was refused. Nothing was saved."""


@dataclass
class KnowledgeRecord:
    subject: str
    fact_type: str
    predicate: str
    value: str
    source_ids: list[int] = field(default_factory=list)
    object_entity: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    recorded_at: float | None = None
    status: str = "active"
    note: str = ""
    record_id: int | None = None


@dataclass
class Evidence:
    sid: int
    chat: int
    speaker: str
    date: str
    text: str
    caption: str


def key_of(name: str) -> str:
    """Names match only when the same apart from case and spacing. Similar names are never merged here."""
    return " ".join((name or "").lower().split())


def _date(value, label):
    if value is None: return None
    if not isinstance(value, str) or not DATE.fullmatch(value):
        raise RecordError(f"{label} must be YYYY, YYYY-MM or YYYY-MM-DD")
    return value


class KnowledgeBase:
    """Works on ChatMemory's own database, beside the sentences. Changes only its three tables."""

    def __init__(self, db):
        self.db = db
        db.executescript(SCHEMA)

    def _entity(self, name, kind=""):
        key = key_of(name)
        row = self.db.execute("SELECT entity FROM entities WHERE key=?", (key,)).fetchone()
        return row[0] if row else self.db.execute("INSERT INTO entities(name, key, kind) VALUES(?,?,?)", (" ".join(name.split()), key, kind)).lastrowid

    def add(self, r: KnowledgeRecord) -> KnowledgeRecord:
        """Saves one record. Every check comes first, so a refused record changes nothing."""
        if not key_of(r.subject): raise RecordError("a record needs a subject")
        if r.fact_type not in FACT_TYPES: raise RecordError(f"fact type must be one of {FACT_TYPES}")
        if not (r.predicate or "").strip(): raise RecordError("a record needs a predicate")
        if not (r.value or "").strip() and not r.object_entity: raise RecordError("a record needs a value or an object")
        if r.status not in SETTABLE: raise RecordError(f"a new record's status must be one of {SETTABLE}")
        sources = list(dict.fromkeys(r.source_ids or []))
        if not sources: raise RecordError("a record needs at least one source sentence")
        for sid in sources:
            if not isinstance(sid, int) or not self.db.execute("SELECT 1 FROM sentences WHERE sid=?", (sid,)).fetchone():
                raise RecordError(f"source sentence does not exist: {sid}")
        start, end = _date(r.valid_from, "valid_from"), _date(r.valid_to, "valid_to")
        if start and end and end < start: raise RecordError("valid_to is before valid_from")
        recorded = r.recorded_at if r.recorded_at is not None else time.time()
        try:
            subject = self._entity(r.subject)
            target = self._entity(r.object_entity) if r.object_entity else None
            rid = self.db.execute("INSERT INTO knowledge_records(subject, fact_type, predicate, value, object, valid_from, valid_to, recorded_at, status, note) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                  (subject, r.fact_type, r.predicate.strip(), (r.value or "").strip(), target, start, end, recorded, r.status, r.note)).lastrowid
            self.db.executemany("INSERT INTO record_evidence(record, sid) VALUES(?,?)", [(rid, s) for s in sources])
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return self.get(rid)

    def get(self, record_id: int) -> KnowledgeRecord | None:
        row = self.db.execute("""SELECT r.record, s.name, r.fact_type, r.predicate, r.value, o.name, r.valid_from, r.valid_to, r.recorded_at, r.status, r.note
                                 FROM knowledge_records r LEFT JOIN entities s ON s.entity=r.subject LEFT JOIN entities o ON o.entity=r.object WHERE r.record=?""", (record_id,)).fetchone()
        if not row: return None
        sources = [x[0] for x in self.db.execute("SELECT sid FROM record_evidence WHERE record=? ORDER BY sid", (record_id,))]
        return KnowledgeRecord(subject=row[1] or "", fact_type=row[2], predicate=row[3] or "", value=row[4] or "", source_ids=sources, object_entity=row[5], valid_from=row[6], valid_to=row[7], recorded_at=row[8], status=row[9], note=row[10], record_id=row[0])

    def about(self, subject: str, statuses=("active",)) -> list[KnowledgeRecord]:
        """Records about one entity, by exact name. Pass statuses=None for all of them."""
        rows = self.db.execute("""SELECT r.record, r.status FROM knowledge_records r JOIN entities e ON e.entity=r.subject WHERE e.key=? ORDER BY r.record""", (key_of(subject),)).fetchall()
        return [self.get(rid) for rid, status in rows if statuses is None or status in statuses]

    def trace(self, record_id: int) -> list[Evidence]:
        """The exact original sentences behind a record, read from the permanent record."""
        rows = self.db.execute("""SELECT s.sid, s.chat, s.speaker, s.date, s.text, s.caption FROM record_evidence e JOIN sentences s ON s.sid=e.sid WHERE e.record=? ORDER BY s.sid""", (record_id,)).fetchall()
        return [Evidence(*r) for r in rows]

    def set_status(self, record_id: int, status: str, note: str = "", valid_to: str | None = None) -> KnowledgeRecord:
        """Marks a record superseded, disputed, retracted or active again. The old content and evidence stay, unless invalidated."""
        if status not in SETTABLE: raise RecordError(f"status must be one of {SETTABLE}")
        current = self.get(record_id)
        if current is None: raise RecordError("no such record")
        if current.status == "invalidated": raise RecordError("an invalidated record has no content and cannot return")
        end = _date(valid_to, "valid_to") if valid_to else current.valid_to
        if current.valid_from and end and end < current.valid_from: raise RecordError("valid_to is before valid_from")
        self.db.execute("UPDATE knowledge_records SET status=?, note=?, valid_to=? WHERE record=?", (status, note or current.note, end, record_id))
        self.db.commit()
        return self.get(record_id)


def forget_sentence(db, sid: int) -> dict:
    """Called inside ChatMemory.forget_sentence, before its commit, so a sentence and what rests on it go together."""
    records = [r[0] for r in db.execute("SELECT record FROM record_evidence WHERE sid=?", (sid,)).fetchall()]
    db.execute("DELETE FROM record_evidence WHERE sid=?", (sid,))
    review = invalid = 0
    for record in records:
        left = db.execute("SELECT count(*) FROM record_evidence WHERE record=?", (record,)).fetchone()[0]
        if left:
            db.execute("UPDATE knowledge_records SET status='needs_review', note='A supporting sentence was deleted' WHERE record=?", (record,))
            review += 1
        else:  # nothing supports it now: the content goes too, leaving only that it once existed
            db.execute("UPDATE knowledge_records SET status='invalidated', subject=NULL, object=NULL, predicate=NULL, value=NULL, valid_from=NULL, valid_to=NULL, note='All supporting sentences were deleted' WHERE record=?", (record,))
            invalid += 1
    db.execute("""DELETE FROM entities WHERE entity NOT IN (SELECT subject FROM knowledge_records WHERE subject IS NOT NULL)
                  AND entity NOT IN (SELECT object FROM knowledge_records WHERE object IS NOT NULL)""")
    return dict(needs_review=review, invalidated=invalid)
