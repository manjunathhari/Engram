"""The sentences behind each fact: adding them, reading them, and what happens when one is deleted."""
from .models import FactEvidence, FactWithEvidence, RecordError, SourceSentence


class EvidenceStore:
    def add_evidence(self, evidence: FactEvidence):
        """One more sentence behind a fact already stored."""
        if not self.db.execute("SELECT 1 FROM knowledge_records WHERE fact_id=?", (evidence.fact_id,)).fetchone(): raise RecordError("no such fact")
        if not self._sentence_exists(evidence.source_id): raise RecordError(f"source sentence does not exist: {evidence.source_id}")
        self._write(lambda: self.db.execute("INSERT OR IGNORE INTO record_evidence VALUES(?,?,?,?,?,?,?,?)", (evidence.evidence_id, evidence.fact_id, evidence.source_kind, evidence.source_id, evidence.evidence_role, evidence.extraction_method, evidence.extraction_version, evidence.created_at)))

    def trace(self, fact_id) -> list[SourceSentence]:
        """The exact original sentences behind a fact, read from the permanent record."""
        rows = self.db.execute("""SELECT s.sid, s.chat, s.speaker, s.date, s.text, s.caption, e.evidence_role FROM record_evidence e
                                  JOIN sentences s ON s.sid = CAST(e.source_id AS INTEGER) WHERE e.fact_id=? AND e.source_kind='sentence' ORDER BY s.sid""", (fact_id,)).fetchall()
        return [SourceSentence(*r) for r in rows]

    def facts_for_source(self, sid) -> list[FactWithEvidence]:
        """Every fact a sentence is evidence for, in any role."""
        rows = self.db.execute("""SELECT DISTINCT r.fact_id, r.recorded_at FROM record_evidence e JOIN knowledge_records r ON r.fact_id = e.fact_id
                                  WHERE e.source_kind='sentence' AND e.source_id=? ORDER BY r.recorded_at, r.fact_id""", (str(sid),)).fetchall()
        return [self.get(r[0]) for r in rows]


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
