"""The knowledge store on Engram's own database connection, so a sentence and what rests on it are always changed together."""
from .entities import EntityStore
from .evidence import EvidenceStore
from .facts import FactStore
from .migrations import migrate, stored_version
from .schema import SCHEMA, SCHEMA_VERSION


def prepare(db) -> list:
    """Checks the version before anything changes: a newer or unreadable version is refused, an older one is brought up to date."""
    version = stored_version(db)
    done = migrate(db, version) if version is not None else []
    db.executescript(SCHEMA)
    db.execute("INSERT OR IGNORE INTO knowledge_meta VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
    db.commit()
    db.execute("PRAGMA foreign_keys=ON")
    return done


class KnowledgeBase(EntityStore, FactStore, EvidenceStore):
    """Works on ChatMemory's own database. Changes only its own tables."""

    def __init__(self, db):
        self.db = db
        self.migrated = prepare(db)

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

    def integrity_check(self) -> list[str]:
        """Every problem found, in plain words. An empty list means the store is sound."""
        db, problems = self.db, []
        ok = db.execute("PRAGMA integrity_check").fetchone()[0]
        if ok != "ok": problems.append(f"the database file is damaged: {ok}")
        problems += [f"{r[0]} row {r[1]} links to a missing {r[2]} row" for r in db.execute("PRAGMA foreign_key_check").fetchall()]
        sids = "SELECT 1 FROM sentences s WHERE s.sid = CAST({} AS INTEGER)"
        problems += [f"fact {r[0]} has no supporting sentence" for r in db.execute("SELECT fact_id FROM knowledge_records r WHERE status <> 'invalidated' AND NOT EXISTS (SELECT 1 FROM record_evidence e WHERE e.fact_id = r.fact_id AND e.evidence_role = 'supports')")]
        problems += [f"evidence {r[0]} points to sentence {r[1]}, which does not exist" for r in db.execute(f"SELECT evidence_id, source_id FROM record_evidence e WHERE NOT EXISTS ({sids.format('e.source_id')})")]
        problems += [f"invalidated fact {r[0]} still has evidence" for r in db.execute("SELECT DISTINCT r.fact_id FROM knowledge_records r JOIN record_evidence e ON e.fact_id = r.fact_id WHERE r.status = 'invalidated'")]
        problems += [f"alias {r[0]} points to sentence {r[1]}, which does not exist" for r in db.execute(f"SELECT alias_id, source_id FROM entity_aliases a WHERE source_kind = 'sentence' AND NOT EXISTS ({sids.format('a.source_id')})")]
        problems += [f"review item {r[0]} quotes sentence {r[1]}, which does not exist" for r in db.execute("SELECT review_id, sid FROM review_sources v WHERE NOT EXISTS (SELECT 1 FROM sentences s WHERE s.sid = v.sid)")]
        problems += [f"review item {r[0]} has no row in the queue" for r in db.execute("SELECT DISTINCT review_id FROM review_sources v WHERE NOT EXISTS (SELECT 1 FROM review_queue q WHERE q.review_id = v.review_id)")]
        return problems
