"""Export, import and backup of the knowledge store.

An export holds personal content, so it is written for the owner only, never over an existing file, and it cannot be recalled once
made: deleting a sentence later does not reach copies already exported. Import therefore never brings back what was deleted.
Every record rests on its sentences, so an export carries a copy of each sentence it cites, and on import a record is kept only where
the same sentence, word for word, is in the memory it goes into. The review queue is not exported: it is work waiting for a person,
not knowledge.
"""
import datetime
import hashlib
import json
import os
import sqlite3

from .models import RecordError
from .schema import SCHEMA_VERSION

FORMAT = "engram-knowledge-export"
FORMAT_VERSION = 1
TABLES = ("entities", "entity_aliases", "knowledge_records", "record_evidence")


def _rows(db, table):
    cur = db.execute(f"SELECT * FROM {table} ORDER BY 1")
    names = [c[0] for c in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def checksum(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _write_private(path, text):
    """Created new and readable by the owner only. An existing file is never replaced."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def export_knowledge(kb, path):
    """Writes every entity, alias, fact and evidence link, with a copy of each sentence they cite and a checksum."""
    problems = kb.integrity_check()
    if problems: raise RecordError("the store has problems, so it was not exported: " + "; ".join(problems[:3]))
    db = kb.db
    data = {t: _rows(db, t) for t in TABLES}
    cited = sorted({int(e["source_id"]) for e in data["record_evidence"]} | {int(a["source_id"]) for a in data["entity_aliases"] if a["source_kind"] == "sentence"})
    data["sentences"] = [dict(sid=r[0], speaker=r[1], date=r[2], text=r[3]) for s in cited for r in [db.execute("SELECT sid, speaker, date, text FROM sentences WHERE sid=?", (s,)).fetchone()]]
    payload = dict(format=FORMAT, format_version=FORMAT_VERSION, schema_version=SCHEMA_VERSION, exported_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   counts={k: len(v) for k, v in data.items()}, checksum=checksum(data), data=data)
    _write_private(path, json.dumps(payload, ensure_ascii=False, indent=1))
    return dict(path=str(path), counts=payload["counts"], checksum=payload["checksum"])


def _read(path):
    try:
        payload = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RecordError(f"the export could not be read ({e}). Nothing was changed.") from None
    if not isinstance(payload, dict) or payload.get("format") != FORMAT: raise RecordError("this is not an Engram knowledge export. Nothing was changed.")
    if payload.get("format_version") != FORMAT_VERSION or payload.get("schema_version") != SCHEMA_VERSION:
        raise RecordError(f"the export is format {payload.get('format_version')}, version {payload.get('schema_version')}; this code reads format {FORMAT_VERSION}, version {SCHEMA_VERSION}. Nothing was changed.")
    data = payload.get("data")
    if not isinstance(data, dict) or checksum(data) != payload.get("checksum"): raise RecordError("the export's checksum does not match: it was changed or damaged. Nothing was changed.")
    return data


def import_knowledge(kb, path):
    """Adds an export's records to this memory, all together or not at all.

    A record whose supporting sentences are missing here, or differ, is left out and reported: that is how a deleted sentence stays deleted.
    The same row already present is skipped. A row with the same id but different content stops the whole import."""
    data = _read(path)
    db = kb.db
    here = {s["sid"]: s for s in data["sentences"]}
    same = lambda sid: (lambda r: r is not None and sid in here and (r[0], r[1], r[2]) == (here[sid]["speaker"], here[sid]["date"], here[sid]["text"]))(db.execute("SELECT speaker, date, text FROM sentences WHERE sid=?", (sid,)).fetchone())
    evidence = [e for e in data["record_evidence"] if same(int(e["source_id"]))]
    supported = {e["fact_id"] for e in evidence if e["evidence_role"] == "supports"}
    facts = [f for f in data["knowledge_records"] if f["fact_id"] in supported]
    kept = {f["fact_id"] for f in facts}
    evidence = [e for e in evidence if e["fact_id"] in kept]
    aliases = [a for a in data["entity_aliases"] if a["source_kind"] == "manual" or same(int(a["source_id"]))]
    used = {f["subject_id"] for f in facts} | {f["object_entity_id"] for f in facts if f["object_entity_id"]} | {a["entity_id"] for a in aliases}
    entities = [e for e in data["entities"] if e["entity_id"] in used]
    aliases = [a for a in aliases if a["entity_id"] in {e["entity_id"] for e in entities}]
    plan = dict(entities=entities, entity_aliases=aliases, knowledge_records=facts, record_evidence=evidence)
    added = {t: 0 for t in TABLES}
    db.commit()
    try:
        db.execute("BEGIN IMMEDIATE")
        for table in TABLES:
            for row in plan[table]:
                key = next(iter(row))
                old = db.execute(f"SELECT * FROM {table} WHERE {key}=?", (row[key],)).fetchone()
                if old is not None:
                    if list(old) == list(row.values()): continue
                    raise RecordError(f"{table} already holds {row[key]} with different content")
                db.execute(f"INSERT INTO {table}({','.join(row)}) VALUES({','.join('?' * len(row))})", list(row.values()))
                added[table] += 1
        if db.execute("PRAGMA foreign_key_check").fetchall(): raise RecordError("the export links to records that are not in it")
        db.execute("COMMIT")
    except Exception as e:
        db.execute("ROLLBACK")
        if isinstance(e, RecordError): raise RecordError(f"{e}. Nothing was imported.") from None
        raise RecordError(f"the export breaks a rule of this store ({e}). Nothing was imported.") from None
    left_out = dict(facts=len(data["knowledge_records"]) - len(facts), evidence=len(data["record_evidence"]) - len(evidence), aliases=len(data["entity_aliases"]) - len(aliases), entities=len(data["entities"]) - len(entities))
    return dict(counts=added, left_out=left_out, note="records resting on sentences missing or changed here were left out" if any(left_out.values()) else "")


def backup(db, path):
    """A complete copy of the whole memory, sentences included, made safely while it is open, then checked. Never replaces a file."""
    if os.path.exists(path): raise RecordError(f"{path} already exists; a backup never replaces a file")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.close(fd)
    target = sqlite3.connect(path)
    try:
        db.commit()
        db.backup(target)
        ok = target.execute("PRAGMA integrity_check").fetchone()[0]
        tables = [r[0] for r in target.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%index%'").fetchall()]
    finally:
        target.close()
    return dict(path=str(path), integrity_check=ok, tables=len(tables))


def integrity_report(kb):
    """The integrity check with row counts, for a person to read."""
    problems = kb.integrity_check()
    return dict(ok=not problems, problems=problems, row_counts={t: kb.db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES})
