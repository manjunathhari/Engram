"""Facts: saving them with their evidence, reading them, changing their status, and the review queue."""
import json
from dataclasses import replace

from .models import SETTABLE_STATUSES, VALIDATION, FactEvidence, FactWithEvidence, KnowledgeFact, RecordError, _choice, new_id, utc_now
from .schema import SCHEMA_VERSION


class FactStore:
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
        return self.about_entity(entity_id, statuses, source_types) if entity_id else []

    def about_entity(self, entity_id, statuses=("active",), source_types=("explicit",)) -> list[FactWithEvidence]:
        """The same, for an entity already known by its id, so two people with one name stay apart."""
        rows = self.db.execute("SELECT fact_id FROM knowledge_records WHERE subject_id=? ORDER BY recorded_at, fact_id", (entity_id,)).fetchall()
        found = [self.get(r[0]) for r in rows]
        return [x for x in found if (statuses is None or x.fact.status in statuses) and (source_types is None or x.fact.source_type in source_types)]

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
