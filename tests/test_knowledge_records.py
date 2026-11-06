"""P1: a knowledge record survives restart and can always be traced to its original evidence."""
import dataclasses
import json
import os
import signal
import sqlite3
import subprocess
import sys

import pytest

import engram_chat as E
import engram_records as R
from conftest import ROOT, fill, load, record
from engram_records import Entity, EntityAlias, FactEvidence, FactWithEvidence, KnowledgeBase, KnowledgeFact, RecordError

DATA = load("basic_conversation.json")
T = "2026-11-06T12:00:00+00:00"


def setup(path):
    m = E.ChatMemory(path)
    ids = fill(m, DATA)
    kb = KnowledgeBase(m.db)
    return m, kb, ids, kb.add_entity(Entity.create("Asha", "person")), kb.add_entity(Entity.create("Pixel", "pet"))


def beagle(asha, pixel, sources, **kw):
    fields = dict(subject_id=asha.entity_id, predicate="owns", object_type="entity", object_entity_id=pixel.entity_id, fact_type="relationship", source_type="explicit", valid_from="2023-03-02")
    fact = KnowledgeFact.create(**(fields | kw))
    return FactWithEvidence(fact, tuple(FactEvidence.create(fact.fact_id, s, extraction_method="test", extraction_version="1") for s in sources))


def counts(m):
    return tuple(m.db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("entities", "entity_aliases", "knowledge_records", "record_evidence"))


def test_the_acceptance_sequence(path):
    """Insert one fact, link it to two sentences, restart, retrieve, verify both, delete a source, check it is marked."""
    m, kb, ids, asha, pixel = setup(path)
    saved = kb.add(beagle(asha, pixel, [ids["B1:1"], ids["B3:1"]]))
    fid = saved.fact.fact_id
    assert saved.fact.status == "active" and saved.fact.validation_status == "pending" and len(saved.evidence) == 2
    m.close()

    m = E.ChatMemory(path)
    kb = KnowledgeBase(m.db)
    got = kb.get(fid)
    assert (got.fact.predicate, got.fact.object_entity_id, got.fact.valid_from, got.fact.source_type) == ("owns", pixel.entity_id, "2023-03-02", "explicit")
    assert [e.source_id for e in got.evidence] == [str(ids["B1:1"]), str(ids["B3:1"])] and got.evidence[0].extraction_method == "test"
    texts = [s.text for s in kb.trace(fid)]
    assert texts == ["I adopted a beagle called Pixel from the shelter yesterday.", "Pixel turned one and we celebrated with a walk by the lake."]
    assert kb.trace(fid)[0].date == "2:15 pm on 3 March, 2023" and kb.trace(fid)[0].evidence_role == "supports"

    m.forget_sentence(ids["B1:1"])
    after = kb.get(fid)
    assert after.fact.status == "needs_review" and [e.source_id for e in after.evidence] == [str(ids["B3:1"])] and after.fact.note
    assert [s.text for s in kb.trace(fid)] == ["Pixel turned one and we celebrated with a walk by the lake."]
    m.close()

    m = E.ChatMemory(path)
    assert KnowledgeBase(m.db).get(fid).fact.status == "needs_review"
    m.close()


def test_deleting_the_last_source_erases_the_content(path):
    m, kb, ids, asha, pixel = setup(path)
    fid = kb.add(beagle(asha, pixel, [ids["B1:1"], ids["B3:1"]], object_type="entity", metadata={"secret": "chews slippers"})).fact.fact_id
    m.forget_sentence(ids["B1:1"]), m.forget_sentence(ids["B3:1"])
    gone = kb.get(fid)
    assert gone.fact.status == "invalidated" and gone.evidence == () and kb.trace(fid) == []
    assert (gone.fact.subject_id, gone.fact.predicate, gone.fact.object_value, gone.fact.object_entity_id, gone.fact.valid_from) == (None, None, None, None, None)
    dump = json.dumps([m.db.execute(f"SELECT * FROM {t}").fetchall() for t in ("entities", "entity_aliases", "knowledge_records", "record_evidence")])
    assert not any(w in dump for w in ("Pixel", "Asha", "owns", "slippers", "2023-03-02"))
    with pytest.raises(RecordError): kb.set_status(fid, "active")
    with pytest.raises(RecordError): kb.set_validation(fid, "accepted")
    m.close()


def test_a_contradicting_sentence_alone_does_not_keep_a_fact_alive(path):
    m, kb, ids, asha, pixel = setup(path)
    built = beagle(asha, pixel, [ids["B1:1"]])
    both = FactWithEvidence(built.fact, built.evidence + (FactEvidence.create(built.fact.fact_id, ids["B2:1"], evidence_role="contradicts"),))
    kb.add(both)
    assert {s.evidence_role for s in kb.trace(built.fact.fact_id)} == {"supports", "contradicts"}
    m.forget_sentence(ids["B1:1"])
    assert kb.get(built.fact.fact_id).fact.status == "invalidated" and kb.trace(built.fact.fact_id) == []
    m.close()


def test_deleting_an_unrelated_sentence_changes_nothing(path):
    m, kb, ids, asha, pixel = setup(path)
    fid = kb.add(beagle(asha, pixel, [ids["B1:1"], ids["B3:1"]])).fact.fact_id
    m.forget_sentence(ids["B2:4"])
    assert kb.get(fid).fact.status == "active" and len(kb.trace(fid)) == 2
    m.close()


def test_review_problem_a_fact_cannot_exist_without_support():
    fact = KnowledgeFact.create("ent", "owns", "text", "attribute", "explicit", object_value="dog")
    with pytest.raises(RecordError, match="supporting sentence"): FactWithEvidence(fact, ())
    with pytest.raises(RecordError, match="supporting sentence"): FactWithEvidence(fact, (FactEvidence.create(fact.fact_id, 1, evidence_role="contradicts"),))


def test_review_problem_an_inference_cannot_pass_as_stated():
    with pytest.raises(TypeError): KnowledgeFact.create("ent", "owns", "text", object_value="dog")
    with pytest.raises(TypeError): KnowledgeFact.create("ent", "owns", "text", "attribute", object_value="dog")


def test_review_problem_evidence_must_belong_to_its_fact():
    a = KnowledgeFact.create("ent", "owns", "text", "attribute", "explicit", object_value="dog")
    b = KnowledgeFact.create("ent", "owns", "text", "attribute", "explicit", object_value="cat")
    with pytest.raises(RecordError, match="different fact"): FactWithEvidence(a, (FactEvidence.create(b.fact_id, 1),))


def test_review_problem_records_can_be_hashed_and_are_really_frozen():
    fact = KnowledgeFact.create("ent", "owns", "text", "attribute", "explicit", object_value="dog", metadata={"a": 1})
    entity = Entity.create("Asha", "person", metadata={"a": 1})
    assert hash(fact) and hash(entity) and {fact} and {entity}
    with pytest.raises(dataclasses.FrozenInstanceError): fact.predicate = "x"
    with pytest.raises(TypeError): fact.metadata["a"] = 2
    with pytest.raises(TypeError): entity.metadata["a"] = 2
    source = {"a": 1}
    copy = KnowledgeFact.create("ent", "owns", "text", "attribute", "explicit", object_value="dog", metadata=source)
    source["a"] = 2
    assert copy.metadata["a"] == 1


GOOD = dict(subject_id="ent", predicate="owns", object_type="text", fact_type="attribute", source_type="explicit", object_value="dog")


@pytest.mark.parametrize("change,why", [
    (dict(fact_type="vibe"), "fact type"),
    (dict(status="banana"), "status"),
    (dict(validation_status="banana"), "validation"),
    (dict(source_type="guess"), "source type"),
    (dict(predicate="   "), "predicate"),
    (dict(subject_id=""), "subject"),
    (dict(object_type="blob"), "object type"),
    (dict(valid_from="last spring"), "valid_from"),
    (dict(valid_to="2023-13"), "valid_to"),
    (dict(valid_from="2024", valid_to="2020"), "before"),
    (dict(object_value=None), "exactly one"),
    (dict(object_value="  "), "exactly one"),
    (dict(object_entity_id="ent_x"), "exactly one"),
    (dict(object_type="entity"), "object type does not match"),
    (dict(recorded_at="yesterday"), "recorded_at"),
    (dict(recorded_at="2026-11-06T12:00:00"), "time zone"),
])
def test_review_problem_bad_values_refused(change, why):
    with pytest.raises(RecordError, match=why): KnowledgeFact.create(**(GOOD | change))


def test_names_handled_the_same_way():
    assert Entity.create("  Asha   Rao ", "PERSON", T).canonical_name == "Asha Rao"
    assert Entity.create("Asha", "PERSON", T).entity_type == "person"
    alias = EntityAlias.create("ent", " Asha   Rao ", "manual", timestamp=T)
    assert (alias.alias, alias.normalized_alias) == ("Asha Rao", "asha rao")
    for bad in (lambda: Entity.create("  ", "person"), lambda: Entity.create("Asha", " "), lambda: EntityAlias.create("ent", "  ", "manual")):
        with pytest.raises(RecordError): bad()


def test_alias_needs_a_source_or_a_manual_label():
    with pytest.raises(RecordError, match="source sentence"): EntityAlias.create("ent", "Mel", "sentence")
    with pytest.raises(RecordError, match="alias source"): EntityAlias.create("ent", "Mel", None)
    with pytest.raises(RecordError, match="manual"): EntityAlias.create("ent", "Mel", "manual", source_id=5)
    assert EntityAlias.create("ent", "Mel", "manual").source_id is None
    assert EntityAlias.create("ent", "Mel", "sentence", source_id=5).source_id == "5"


def test_alias_reaches_the_same_entity_and_similar_names_stay_apart(path):
    m, kb, ids, asha, pixel = setup(path)
    assert kb.resolve("Ash") is None and kb.resolve("  ASHA ") == asha.entity_id
    kb.add_alias(EntityAlias.create(asha.entity_id, "Ash", "sentence", source_id=ids["B1:1"]))
    assert kb.resolve("ash") == asha.entity_id and kb.resolve("Asha R") is None
    fid = kb.add(beagle(asha, pixel, [ids["B1:1"]])).fact.fact_id
    assert [x.fact.fact_id for x in kb.about("ASH")] == [fid]
    m.close()


def test_alias_cannot_take_a_name_from_another_entity(path):
    m, kb, ids, asha, pixel = setup(path)
    before = counts(m)
    with pytest.raises(RecordError, match="another entity"): kb.add_alias(EntityAlias.create(pixel.entity_id, "Asha", "manual"))
    with pytest.raises(RecordError, match="already belongs"): kb.add_entity(Entity.create(" asha ", "person"))
    with pytest.raises(RecordError, match="does not exist"): kb.add_alias(EntityAlias.create(asha.entity_id, "Ash", "sentence", source_id=99999))
    with pytest.raises(RecordError, match="no such entity"): kb.add_alias(EntityAlias.create("ent_none", "Ash", "manual"))
    assert counts(m) == before
    kb.add_alias(EntityAlias.create(asha.entity_id, "Ash", "manual"))
    kb.add_alias(EntityAlias.create(asha.entity_id, "ASH", "manual"))
    assert counts(m)[1] == 1
    m.close()


def test_aliases_survive_restart_and_go_with_their_sentence(path):
    m, kb, ids, asha, pixel = setup(path)
    kb.add_alias(EntityAlias.create(asha.entity_id, "Ash", "sentence", source_id=ids["B1:1"]))
    kb.add_alias(EntityAlias.create(asha.entity_id, "Aa", "manual"))
    m.close()
    m = E.ChatMemory(path)
    kb = KnowledgeBase(m.db)
    assert kb.resolve("Ash") == asha.entity_id
    m.forget_sentence(ids["B1:1"])
    assert kb.resolve("Ash") is None and kb.resolve("Aa") == asha.entity_id  # a manual alias has no sentence to delete
    m.close()


@pytest.mark.parametrize("sources,why", [([99999], "does not exist"), (["abc"], "does not exist")])
def test_refused_facts_save_nothing(path, sources, why):
    m, kb, ids, asha, pixel = setup(path)
    before = counts(m)
    with pytest.raises(RecordError, match=why): kb.add(beagle(asha, pixel, sources))
    assert counts(m) == before
    m.close()


def test_other_refusals_save_nothing(path):
    m, kb, ids, asha, pixel = setup(path)
    before = counts(m)
    with pytest.raises(RecordError, match="twice"): kb.add(beagle(asha, pixel, [ids["B1:1"], ids["B1:1"]]))
    with pytest.raises(RecordError, match="no such subject"): kb.add(beagle(Entity.create("Ghost", "person"), pixel, [ids["B1:1"]]))
    with pytest.raises(RecordError, match="no such object"): kb.add(beagle(asha, Entity.create("Ghost", "pet"), [ids["B1:1"]]))
    with pytest.raises(RecordError, match="new fact"): kb.add(beagle(asha, pixel, [ids["B1:1"]], status="needs_review"))
    assert counts(m) == before
    m.close()


@pytest.mark.parametrize("kind", R.FACT_TYPES)
def test_all_fact_types_stored(path, kind):
    m, kb, ids, asha, pixel = setup(path)
    assert kb.add(beagle(asha, pixel, [ids["B1:1"]], fact_type=kind)).fact.fact_type == kind
    m.close()


def test_time_fields_are_separate(path):
    m, kb, ids, asha, pixel = setup(path)
    saved = kb.add(beagle(asha, pixel, [ids["B1:1"]], valid_from="2023-03", valid_to="2024", recorded_at=T)).fact
    assert (saved.valid_from, saved.valid_to, saved.recorded_at) == ("2023-03", "2024", T)
    m.close()


def test_status_changes_keep_content_and_evidence(path):
    m, kb, ids, asha, pixel = setup(path)
    old = kb.add(beagle(asha, pixel, [ids["B1:1"], ids["B3:1"]])).fact.fact_id
    new = kb.add(beagle(asha, pixel, [ids["B3:1"]], valid_from="2024-01-01")).fact.fact_id
    changed = kb.set_status(old, "superseded", note="replaced", valid_to="2024-01-01")
    assert changed.fact.status == "superseded" and changed.fact.valid_to == "2024-01-01" and changed.fact.object_entity_id == pixel.entity_id and len(kb.trace(old)) == 2
    assert [x.fact.fact_id for x in kb.about("asha")] == [new] and len(kb.about("Asha", statuses=None)) == 2
    assert kb.set_status(old, "disputed").fact.status == "disputed" and kb.set_status(old, "retracted").fact.status == "retracted"
    with pytest.raises(RecordError): kb.set_status(old, "needs_review")
    with pytest.raises(RecordError): kb.set_status(old, "banana")
    with pytest.raises(RecordError): kb.set_status("fact_none", "active")
    with pytest.raises(RecordError): kb.set_status(new, "superseded", valid_to="2020")
    assert kb.get(new).fact.status == "active"
    m.close()


def test_validation_never_turns_an_inference_into_a_statement(path):
    m, kb, ids, asha, pixel = setup(path)
    fid = kb.add(beagle(asha, pixel, [ids["B1:1"]], source_type="inferred")).fact.fact_id
    assert kb.about("Asha") == [] and len(kb.about("Asha", source_types=None)) == 1
    accepted = kb.set_validation(fid, "accepted")
    assert accepted.fact.validation_status == "accepted" and accepted.fact.source_type == "inferred"
    assert kb.about("Asha") == []
    with pytest.raises(RecordError): kb.set_validation(fid, "banana")
    m.close()


def test_records_do_not_change_the_memory_or_retrieval(path):
    plain = E.ChatMemory(path + "p")
    fill(plain, DATA)
    m, kb, ids, asha, pixel = setup(path)
    before = record(m)
    q = DATA["questions"][0]["q"]
    found = m.find(q)[0], m.context_for(q)[0]
    kb.add(beagle(asha, pixel, [ids["B1:1"]]))
    assert record(m) == before and record(m) == record(plain)
    assert (m.find(q)[0], m.context_for(q)[0]) == found
    plain.close(), m.close()


def test_no_conversation_text_is_copied(path):
    m, kb, ids, asha, pixel = setup(path)
    kb.add(beagle(asha, pixel, [ids["B1:1"]]))
    dump = json.dumps([m.db.execute(f"SELECT * FROM {t}").fetchall() for t in ("entities", "entity_aliases", "knowledge_records", "record_evidence")])
    assert not any(r[6] in dump for r in record(m))
    m.close()


def test_schema_version_is_stored(path):
    m, kb, ids, asha, pixel = setup(path)
    kb.add(beagle(asha, pixel, [ids["B1:1"]]))
    assert m.db.execute("SELECT schema_version FROM knowledge_records").fetchone()[0] == R.SCHEMA_VERSION
    assert m.db.execute("SELECT schema_version FROM entities").fetchone()[0] == R.SCHEMA_VERSION
    m.close()


def test_older_database_gets_the_new_tables(path):
    old = sqlite3.connect(path)
    old.executescript("""CREATE TABLE chats(chat INTEGER PRIMARY KEY, opened_on TEXT, last_active REAL, closed INTEGER DEFAULT 0, note TEXT);
CREATE TABLE sentences(sid INTEGER PRIMARY KEY, chat INTEGER, pos INTEGER, ref TEXT, speaker TEXT, date TEXT, text TEXT, caption TEXT, tokens INTEGER);
CREATE VIRTUAL TABLE words_index USING fts5(words, tokenize="unicode61 remove_diacritics 0");
INSERT INTO chats VALUES(1,'1 May 2023',0,1,NULL);
INSERT INTO sentences VALUES(1,1,0,'','A','1 May 2023','My dog is called Rex.','',6);""")
    old.commit(); old.close()
    m = E.ChatMemory(path)
    kb = KnowledgeBase(m.db)
    a = kb.add_entity(Entity.create("A", "person"))
    f = KnowledgeFact.create(a.entity_id, "owns", "text", "relationship", "explicit", object_value="dog")
    kb.add(FactWithEvidence(f, (FactEvidence.create(f.fact_id, 1),)))
    assert kb.trace(f.fact_id)[0].text == "My dog is called Rex."
    m.close()


@pytest.mark.parametrize("other", ["2", "0", "banana"])
def test_a_database_of_another_version_is_refused_unchanged(path, other):
    m, kb, ids, asha, pixel = setup(path)
    fid = kb.add(beagle(asha, pixel, [ids["B1:1"]])).fact.fact_id
    m.db.execute("UPDATE knowledge_meta SET value=? WHERE key='schema_version'", (other,))
    m.db.commit()
    before = record(m)
    m.close()
    with pytest.raises(RecordError, match="Nothing was changed"): E.ChatMemory(path)
    raw = sqlite3.connect(path)
    with pytest.raises(RecordError, match="Nothing was changed"): KnowledgeBase(raw)
    assert raw.execute("SELECT value FROM knowledge_meta").fetchall() == [(other,)]
    assert raw.execute("SELECT count(*) FROM knowledge_records WHERE fact_id=?", (fid,)).fetchone()[0] == 1
    raw.execute("UPDATE knowledge_meta SET value=?", (str(R.SCHEMA_VERSION),))
    raw.commit(); raw.close()
    m = E.ChatMemory(path)
    assert record(m) == before
    m.close()


def test_the_version_is_written_once_and_reopening_keeps_it(path):
    m, kb, ids, asha, pixel = setup(path)
    m.close()
    m = E.ChatMemory(path)
    KnowledgeBase(m.db)
    assert m.db.execute("SELECT key, value FROM knowledge_meta").fetchall() == [("schema_version", str(R.SCHEMA_VERSION))]
    m.close()


def test_existing_rows_are_unchanged_when_records_are_added(path):
    """Every row of every existing table, not only that the tables are still there."""
    m = E.ChatMemory(path)
    fill(m, DATA)
    m.db.execute("DROP TABLE knowledge_meta")
    for t in ("entities", "entity_aliases", "knowledge_records", "record_evidence"): m.db.execute(f"DROP TABLE {t}")
    m.db.commit()
    tables = [r[0] for r in m.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%index%' AND name NOT LIKE 'sqlite_%'")]
    assert {"sentences", "chats", "facts", "fact_sources"} <= set(tables)
    dump = lambda db: {t: db.execute(f"SELECT * FROM {t}").fetchall() for t in tables}
    m.db.execute("INSERT INTO facts(fact, chat, text) VALUES(900, 1, 'An older helper fact')")
    m.db.execute("INSERT INTO fact_sources VALUES(900, 1)")
    m.db.commit()
    before = dump(m.db)
    m.close()
    m = E.ChatMemory(path)
    kb = KnowledgeBase(m.db)
    a = kb.add_entity(Entity.create("Asha", "person"))
    f = KnowledgeFact.create(a.entity_id, "owns", "text", "relationship", "explicit", object_value="dog")
    kb.add(FactWithEvidence(f, (FactEvidence.create(f.fact_id, 1),)))
    assert dump(m.db) == before and before["sentences"]
    m.close()


def test_a_saved_record_survives_a_hard_crash(path):
    """Kills the process right after saving, with no clean close, as a power cut would."""
    code = f"""
import sys, time
sys.path.insert(0, {ROOT!r})
import engram_chat as E
from engram_records import *
m = E.ChatMemory({path!r})
m.open_chat("1 May 2023")
a = m.add("Asha", "I adopted a dog called Rex.", date="1 May 2023")
b = m.add("Asha", "Rex is my dog.", date="1 May 2023")
kb = KnowledgeBase(m.db)
e = kb.add_entity(Entity.create("Asha", "person"))
f = KnowledgeFact.create(e.entity_id, "owns", "text", "relationship", "explicit", object_value="dog")
kb.add(FactWithEvidence(f, (FactEvidence.create(f.fact_id, a), FactEvidence.create(f.fact_id, b))))
print(f.fact_id, flush=True)
time.sleep(60)
"""
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    fact_id = p.stdout.readline().strip()
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    m = E.ChatMemory(path)
    kb = KnowledgeBase(m.db)
    assert kb.get(fact_id).fact.object_value == "dog" and len(kb.trace(fact_id)) == 2
    m.close()
