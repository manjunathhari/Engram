"""Brings an older knowledge database up to the current version. Each step runs as one transaction:
if anything in the old data breaks a rule of the new version, nothing is changed and the reason is given."""
from .models import RecordError
from .schema import SCHEMA, SCHEMA_VERSION, TABLES


def _v1_to_v2(db):
    """Rebuilds the four tables with the checks of version 2. Every row is copied as it was."""
    for t in TABLES: db.execute(f"ALTER TABLE {t} RENAME TO {t}_v1")
    for t in ("entity_aliases_source", "record_evidence_source", "record_evidence_fact", "knowledge_records_subject"): db.execute(f"DROP INDEX IF EXISTS {t}")
    for statement in [s for s in SCHEMA.split(";") if s.strip()]: db.execute(statement)
    for t in TABLES:
        db.execute(f"INSERT INTO {t} SELECT * FROM {t}_v1")
        db.execute(f"DROP TABLE {t}_v1")


STEPS = {1: _v1_to_v2}


def stored_version(db):
    """The version on disk: from knowledge_meta, or 1 for the first release, which had the tables but no version table."""
    has = lambda name: db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    if has("knowledge_meta"):
        row = db.execute("SELECT value FROM knowledge_meta WHERE key='schema_version'").fetchone()
        if row: return row[0]
    return "1" if has("entities") else None


def migrate(db, version):
    """From version to SCHEMA_VERSION, one step at a time. Refuses what it cannot do, before changing anything."""
    if not str(version).isdigit(): raise RecordError(f"this database's knowledge records are version {version}, which this code cannot read. Nothing was changed.")
    version = int(version)
    if version > SCHEMA_VERSION: raise RecordError(f"this database's knowledge records are version {version}, newer than this code (version {SCHEMA_VERSION}). Nothing was changed.")
    if any(v not in STEPS for v in range(version, SCHEMA_VERSION)): raise RecordError(f"there is no way to bring version {version} up to version {SCHEMA_VERSION}. Nothing was changed.")
    if version == SCHEMA_VERSION: return []
    db.commit()
    db.execute("PRAGMA foreign_keys=OFF")  # copying happens with links unchecked, then every link is checked before saving
    done = []
    try:
        db.execute("BEGIN IMMEDIATE")
        for v in range(version, SCHEMA_VERSION):
            STEPS[v](db)
            done.append(f"{v} to {v + 1}")
        broken = db.execute("PRAGMA foreign_key_check").fetchall()
        if broken: raise RecordError(f"the old data has {len(broken)} links to records that do not exist, for example in {broken[0][0]}. Nothing was changed.")
        db.execute("CREATE TABLE IF NOT EXISTS knowledge_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT OR REPLACE INTO knowledge_meta VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
        db.execute("COMMIT")
    except Exception as e:
        db.execute("ROLLBACK")
        db.execute("PRAGMA foreign_keys=ON")
        if isinstance(e, RecordError): raise
        raise RecordError(f"the old data breaks a rule of version {SCHEMA_VERSION} ({e}). Nothing was changed.") from None
    db.execute("PRAGMA foreign_keys=ON")
    return done
