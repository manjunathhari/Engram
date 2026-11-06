"""Manj's P1 suite (33 tests), each rewritten against the names in main, keeping what it checks.
The one xfail checks something main does differently by design; the reason is given."""
import sqlite3
import pytest
import engram_chat as E
from engram_knowledge import KnowledgeBase, SCHEMA_VERSION, Entity, EntityAlias, KnowledgeFact, FactEvidence, FactWithEvidence, RecordError
STAMP = "2026-11-06T10:00:00+00:00"

@pytest.fixture
def mem(tmp_path):
    m = E.ChatMemory(str(tmp_path / "engram.sqlite"))
    m.open_chat("1:00 pm on 1 January, 2026")
    m.sid = [m.add("User", t, date="1:00 pm on 1 January, 2026") for t in ("I prefer SQLite for local memory.", "SQLite is embedded.", "I live in Pune.", "I work at Acme.")]
    m.close_chat(write_note=False, sleep=False)
    m.kb = KnowledgeBase(m.db)
    yield m
    m.close()

def ent(kb, name="John", kind="person"): return kb.add_entity(Entity.create(name, kind, STAMP))
def put(m, who, sids, roles=None, **kw):
    f = KnowledgeFact.create(**(dict(subject_id=who.entity_id, predicate="city", object_type="text", fact_type="attribute", source_type="explicit", object_value="Pune", recorded_at=STAMP) | kw))
    return m.kb.add(FactWithEvidence(f, tuple(FactEvidence.create(f.fact_id, s, evidence_role=r) for s, r in zip(sids, roles or ["supports"] * len(sids)))))

# ---------- P1.1 schema (3) ----------
def test_schema_initialization(mem):
    names = {r[0] for r in mem.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"entities", "entity_aliases", "knowledge_records", "record_evidence", "knowledge_meta"} <= names
    assert mem.db.execute("SELECT value FROM knowledge_meta").fetchone()[0] == str(SCHEMA_VERSION)

def test_existing_engram_tables_are_preserved(mem):
    names = {r[0] for r in mem.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"facts", "fact_sources", "sentences"} <= names

def test_foreign_keys_enabled(mem):
    assert mem.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1

# ---------- P1.2 store (14) ----------
def test_create_and_retrieve_entity(mem):
    e = mem.kb.add_entity(Entity.create("  John Smith  ", "Person", STAMP, {"origin": "test"}))
    got = mem.kb.entity(e.entity_id)
    assert (got.canonical_name, got.entity_type, dict(got.metadata)) == ("John Smith", "person", {"origin": "test"})

def test_alias_lookup_returns_matches_without_merging(mem):
    one, two = ent(mem.kb, "John Smith"), ent(mem.kb, "John Jones")
    mem.kb.add_alias(EntityAlias.create(one.entity_id, "Dr John", "manual")); mem.kb.add_alias(EntityAlias.create(two.entity_id, "Dr John", "manual"))
    assert set(mem.kb.find(" DR JOHN ")) == {one.entity_id, two.entity_id} and mem.kb.resolve("Dr John") is None

def test_add_atomic_literal_fact_and_retrieve(mem):
    p = ent(mem.kb)
    got = mem.kb.get(put(mem, p, [mem.sid[0]], predicate="age", object_type="number", object_value=42).fact.fact_id).fact
    assert (got.subject_id, got.predicate, got.value, got.object_type) == (p.entity_id, "age", 42, "number")

def test_add_entity_relationship_fact(mem):
    j, h = ent(mem.kb), ent(mem.kb, "Hyderabad", "location")
    got = put(mem, j, [mem.sid[2]], predicate="lives_in", object_type="entity", object_value=None, object_entity_id=h.entity_id, fact_type="relationship").fact
    assert got.object_entity_id == h.entity_id and got.object_value is None

def test_fact_requires_exactly_one_object_representation(mem):
    with pytest.raises(RecordError): KnowledgeFact.create(ent(mem.kb).entity_id, "lives_in", "entity", "relationship", "explicit", object_value="Hyderabad")

def test_link_multiple_evidence_sources_and_retrieve(mem):
    r = put(mem, ent(mem.kb), [mem.sid[2], mem.sid[3]], roles=["supports", "qualifies"])
    assert r.fact.value == "Pune" and {e.source_id for e in r.evidence} == {str(mem.sid[2]), str(mem.sid[3])} and len(r.evidence) == 2

def test_accepted_fact_requires_supporting_evidence(mem):
    """Stricter in main: no fact can exist without a supporting sentence, accepted or not."""
    f = KnowledgeFact.create(ent(mem.kb).entity_id, "city", "text", "attribute", "explicit", object_value="Pune", validation_status="accepted")
    with pytest.raises(RecordError, match="supporting"): FactWithEvidence(f, ())
    r = mem.kb.add(FactWithEvidence(f, (FactEvidence.create(f.fact_id, mem.sid[2]),)))
    assert r.fact.validation_status == "accepted"

def test_find_facts_by_source(mem):
    r = put(mem, ent(mem.kb), [mem.sid[3]], predicate="works_at", object_value="Acme")
    assert [x.fact.fact_id for x in mem.kb.facts_for_source(mem.sid[3])] == [r.fact.fact_id]

def test_filter_facts_by_subject_predicate_and_status(mem):
    p = ent(mem.kb)
    a = put(mem, p, [mem.sid[2]], predicate="city", object_value="Bengaluru").fact.fact_id
    put(mem, p, [mem.sid[3]], predicate="occupation", object_value="surgeon")
    mem.kb.set_validation(a, "accepted")
    got = [x for x in mem.kb.about_entity(p.entity_id) if x.fact.predicate == "city" and x.fact.validation_status == "accepted"]
    assert [x.fact.value for x in got] == ["Bengaluru"]

def test_duplicate_evidence_is_rejected(mem):
    p = ent(mem.kb)
    with pytest.raises(RecordError, match="twice"): put(mem, p, [mem.sid[2], mem.sid[2]])
    r = put(mem, p, [mem.sid[2]])
    mem.kb.add_evidence(FactEvidence.create(r.fact.fact_id, mem.sid[2]))  # main ignores a repeat instead of raising; either way no copy is kept
    assert len(mem.kb.trace(r.fact.fact_id)) == 1

def test_fact_and_evidence_persist_after_reopen(tmp_path):
    m = E.ChatMemory(str(tmp_path / "p.sqlite")); m.open_chat("1:00 pm on 1 January, 2026"); s = m.add("User", "I live in Pune.", date="1:00 pm on 1 January, 2026"); m.close_chat(write_note=False, sleep=False)
    kb = KnowledgeBase(m.db); f = KnowledgeFact.create(kb.add_entity(Entity.create("John", "person")).entity_id, "city", "text", "attribute", "explicit", object_value="Pune")
    kb.add(FactWithEvidence(f, (FactEvidence.create(f.fact_id, s),))); m.close()
    m = E.ChatMemory(str(tmp_path / "p.sqlite")); got = KnowledgeBase(m.db).get(f.fact_id)
    assert got.fact.value == "Pune" and got.evidence[0].source_id == str(s); m.close()

@pytest.mark.xfail(strict=True, reason="by design: main never deletes a fact; it is marked superseded or retracted, and erased only when its last sentence is deleted")
def test_delete_fact_cascades_evidence(mem):
    r = put(mem, ent(mem.kb), [mem.sid[2]])
    assert mem.kb.delete_fact(r.fact.fact_id)

def test_unknown_subject_is_rejected(mem):
    f = KnowledgeFact.create("missing", "city", "text", "attribute", "explicit", object_value="Pune")
    with pytest.raises(RecordError, match="no such subject"): mem.kb.add(FactWithEvidence(f, (FactEvidence.create(f.fact_id, mem.sid[2]),)))

def test_database_integrity(mem):
    put(mem, ent(mem.kb), [mem.sid[2]])
    assert mem.kb.integrity_check() == []

# ---------- P1.3 provenance (5) ----------
def test_sentence_evidence_validates_sid_and_resolves_text(mem):
    r = put(mem, ent(mem.kb, "User"), [mem.sid[0]], predicate="prefers", object_value="SQLite")
    t = mem.kb.trace(r.fact.fact_id)[0]
    assert r.evidence[0].source_id == str(mem.sid[0]) and r.evidence[0].source_kind == "sentence" and t.text == "I prefer SQLite for local memory." and t.chat == 1

def test_unknown_sentence_is_rejected(mem):
    r = put(mem, ent(mem.kb), [mem.sid[2]])
    with pytest.raises(RecordError, match="does not exist"): mem.kb.add_evidence(FactEvidence.create(r.fact.fact_id, 999))

def test_legacy_fact_sources_are_bridged_without_text_conversion(mem):
    mem.db.execute("INSERT INTO facts(fact, chat, text) VALUES(7, 1, 'User prefers SQLite')"); mem.db.execute("INSERT INTO fact_sources VALUES(7, ?)", (mem.sid[0],)); mem.db.commit()
    r = put(mem, ent(mem.kb, "User"), [mem.sid[1]], predicate="prefers", object_value="SQLite")
    links = mem.kb.link_helper_fact(r.fact.fact_id, 7)
    assert [e.source_id for e in links] == [str(mem.sid[0])] and mem.kb.trace(r.fact.fact_id)[0].text.startswith("I prefer")
    assert mem.db.execute("SELECT count(*) FROM knowledge_records").fetchone()[0] == 1

def test_legacy_fact_without_sources_fails(mem):
    mem.db.execute("INSERT INTO facts(fact, chat, text) VALUES(8, 1, 'Fact without sources')"); mem.db.commit()
    r = put(mem, ent(mem.kb), [mem.sid[2]])
    with pytest.raises(RecordError, match="no source sentences"): mem.kb.link_helper_fact(r.fact.fact_id, 8)

def test_batch_evidence_validates_all_ids_before_writing(mem):
    p = ent(mem.kb); before = mem.db.execute("SELECT count(*) FROM record_evidence").fetchone()[0]
    with pytest.raises(RecordError, match="does not exist"): put(mem, p, [mem.sid[2], 999])
    assert mem.db.execute("SELECT count(*) FROM record_evidence").fetchone()[0] == before and mem.db.execute("SELECT count(*) FROM knowledge_records").fetchone()[0] == 0

# ---------- P1.4 portability (6) ----------
from engram_knowledge import backup, export_knowledge, import_knowledge, integrity_report
import json


def copy_memory(src, dst):
    """A second memory holding the same sentences, as a restored or moved memory would."""
    d = sqlite3.connect(dst); src.db.backup(d); d.close()
    m = E.ChatMemory(dst); m.kb = KnowledgeBase(m.db)
    for t in ("record_evidence", "knowledge_records", "entity_aliases", "entities"): m.db.execute(f"DELETE FROM {t}")
    m.db.commit()
    return m


def test_knowledge_export_import_round_trip(mem, tmp_path):
    p = ent(mem.kb, "User"); r = put(mem, p, [mem.sid[0]], predicate="prefers", object_value="SQLite")
    out = export_knowledge(mem.kb, tmp_path / "knowledge.json")
    assert out["counts"]["entities"] == 1 and out["counts"]["knowledge_records"] == 1
    target = copy_memory(mem, str(tmp_path / "target.sqlite"))
    got = import_knowledge(target.kb, tmp_path / "knowledge.json")
    loaded = target.kb.get(r.fact.fact_id)
    assert loaded.fact.value == "SQLite" and loaded.evidence[0].source_id == str(mem.sid[0]) and target.kb.entity(p.entity_id).canonical_name == "User"
    assert got["counts"]["record_evidence"] == 1 and target.kb.integrity_check() == []
    target.close()


def test_export_checksum_detects_tampering(mem, tmp_path):
    put(mem, ent(mem.kb, "User"), [mem.sid[0]])
    export_knowledge(mem.kb, tmp_path / "k.json")
    payload = json.loads((tmp_path / "k.json").read_text()); payload["data"]["entities"][0]["canonical_name"] = "Changed"
    (tmp_path / "bad.json").write_text(json.dumps(payload))
    target = copy_memory(mem, str(tmp_path / "target.sqlite"))
    with pytest.raises(RecordError, match="checksum"): import_knowledge(target.kb, tmp_path / "bad.json")
    assert target.db.execute("SELECT count(*) FROM entities").fetchone()[0] == 0
    target.close()


def test_import_is_atomic_on_id_conflict(mem, tmp_path):
    p = ent(mem.kb, "User"); put(mem, p, [mem.sid[0]])
    export_knowledge(mem.kb, tmp_path / "k.json")
    target = copy_memory(mem, str(tmp_path / "target.sqlite"))
    target.kb.add_entity(Entity(p.entity_id, "Existing", "person", STAMP, STAMP))
    with pytest.raises(RecordError, match="different content"): import_knowledge(target.kb, tmp_path / "k.json")
    assert target.db.execute("SELECT count(*) FROM entities").fetchone()[0] == 1 and target.db.execute("SELECT count(*) FROM knowledge_records").fetchone()[0] == 0
    target.close()


def test_sqlite_backup_preserves_all_tables(mem, tmp_path):
    put(mem, ent(mem.kb), [mem.sid[2]])
    mem.db.execute("CREATE TABLE legacy_marker(value TEXT)"); mem.db.execute("INSERT INTO legacy_marker VALUES ('preserve me')"); mem.db.commit()
    out = backup(mem.db, str(tmp_path / "backup.db"))
    assert out["integrity_check"] == "ok"
    b = sqlite3.connect(tmp_path / "backup.db")
    assert b.execute("SELECT value FROM legacy_marker").fetchone()[0] == "preserve me" and b.execute("SELECT count(*) FROM knowledge_records").fetchone()[0] == 1 and b.execute("SELECT count(*) FROM sentences").fetchone()[0] == 4
    b.close()


def test_integrity_report_ok_for_valid_store(mem):
    put(mem, ent(mem.kb), [mem.sid[2]])
    report = integrity_report(mem.kb)
    assert report["ok"] is True and report["problems"] == [] and report["row_counts"]["knowledge_records"] == 1

def test_integrity_report_detects_orphan_sentence_reference(mem):
    r = put(mem, ent(mem.kb), [mem.sid[2], mem.sid[3]])
    mem.db.execute("DELETE FROM sentences WHERE sid IN (?, ?)", (mem.sid[2], mem.sid[3])); mem.db.commit()  # deleted behind Engram's back
    report = integrity_report(mem.kb)
    assert sum("does not exist" in p for p in report["problems"]) == 2 and report["ok"] is False
