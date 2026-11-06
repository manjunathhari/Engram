"""P1: a knowledge record survives restart and can always be traced to its original evidence."""
import json
import sqlite3

import pytest

import engram_chat as E
import engram_records as R
from conftest import fill, load, record
from engram_records import KnowledgeBase, KnowledgeRecord, RecordError

DATA = load("basic_conversation.json")


def setup(path):
    m = E.ChatMemory(path)
    ids = fill(m, DATA)
    return m, KnowledgeBase(m.db), ids


def beagle(ids, **kw):
    fields = dict(subject="Asha", fact_type="relationship", predicate="owns", value="beagle", object_entity="Pixel", source_ids=[ids["B1:1"], ids["B3:1"]], valid_from="2023-03-02")
    return KnowledgeRecord(**(fields | kw))


def test_the_acceptance_sequence(path):
    """Insert one fact, link it to two sentences, restart, retrieve, verify both, delete a source, check it is marked."""
    m, kb, ids = setup(path)
    saved = kb.add(beagle(ids))
    assert saved.record_id and saved.source_ids == sorted([ids["B1:1"], ids["B3:1"]]) and saved.recorded_at
    m.close()

    m = E.ChatMemory(path)
    kb = KnowledgeBase(m.db)
    got = kb.get(saved.record_id)
    assert (got.subject, got.predicate, got.value, got.object_entity, got.valid_from, got.status) == ("Asha", "owns", "beagle", "Pixel", "2023-03-02", "active")
    texts = [e.text for e in kb.trace(saved.record_id)]
    assert texts == ["I adopted a beagle called Pixel from the shelter yesterday.", "Pixel turned one and we celebrated with a walk by the lake."]
    assert kb.trace(saved.record_id)[0].date == "2:15 pm on 3 March, 2023"

    m.forget_sentence(ids["B1:1"])
    after = kb.get(saved.record_id)
    assert after.status == "needs_review" and after.source_ids == [ids["B3:1"]] and after.note
    assert [e.text for e in kb.trace(saved.record_id)] == ["Pixel turned one and we celebrated with a walk by the lake."]
    m.close()

    m = E.ChatMemory(path)
    kb = KnowledgeBase(m.db)
    assert kb.get(saved.record_id).status == "needs_review"
    m.close()


def test_deleting_the_last_source_erases_the_content(path):
    m, kb, ids = setup(path)
    saved = kb.add(beagle(ids))
    m.forget_sentence(ids["B1:1"]), m.forget_sentence(ids["B3:1"])
    gone = kb.get(saved.record_id)
    assert gone.status == "invalidated" and gone.source_ids == [] and kb.trace(saved.record_id) == []
    assert (gone.subject, gone.predicate, gone.value, gone.object_entity, gone.valid_from) == ("", "", "", None, None)
    dump = json.dumps(m.db.execute("SELECT * FROM knowledge_records").fetchall()) + json.dumps(m.db.execute("SELECT * FROM entities").fetchall())
    assert "Pixel" not in dump and "beagle" not in dump and "Asha" not in dump
    with pytest.raises(RecordError):
        kb.set_status(saved.record_id, "active")
    m.close()


def test_deleting_an_unrelated_sentence_changes_nothing(path):
    m, kb, ids = setup(path)
    saved = kb.add(beagle(ids))
    m.forget_sentence(ids["B2:4"])
    assert kb.get(saved.record_id).status == "active" and len(kb.trace(saved.record_id)) == 2
    m.close()


@pytest.mark.parametrize("change,why", [
    (dict(source_ids=[]), "source"),
    (dict(source_ids=[99999]), "does not exist"),
    (dict(subject=" "), "subject"),
    (dict(fact_type="vibe"), "fact type"),
    (dict(predicate=""), "predicate"),
    (dict(value="", object_entity=None), "value or an object"),
    (dict(status="verified"), "status"),
    (dict(status="invalidated"), "status"),
    (dict(valid_from="last spring"), "valid_from"),
    (dict(valid_from="2023-05-01", valid_to="2023-01-01"), "before"),
])
def test_refused_records_save_nothing(path, change, why):
    m, kb, ids = setup(path)
    before = m.db.execute("SELECT (SELECT count(*) FROM knowledge_records), (SELECT count(*) FROM entities), (SELECT count(*) FROM record_evidence)").fetchone()
    with pytest.raises(RecordError, match=why):
        kb.add(beagle(ids, **change))
    assert m.db.execute("SELECT (SELECT count(*) FROM knowledge_records), (SELECT count(*) FROM entities), (SELECT count(*) FROM record_evidence)").fetchone() == before
    m.close()


@pytest.mark.parametrize("kind", R.FACT_TYPES)
def test_all_fact_types_stored(path, kind):
    m, kb, ids = setup(path)
    assert kb.get(kb.add(beagle(ids, fact_type=kind)).record_id).fact_type == kind
    m.close()


def test_time_fields_are_separate(path):
    m, kb, ids = setup(path)
    saved = kb.add(beagle(ids, valid_from="2023-03", valid_to="2024", recorded_at=1700000000.0))
    assert (saved.valid_from, saved.valid_to, saved.recorded_at) == ("2023-03", "2024", 1700000000.0)
    m.close()


def test_status_changes_keep_content_and_evidence(path):
    m, kb, ids = setup(path)
    old = kb.add(beagle(ids))
    new = kb.add(beagle(ids, value="labrador", valid_from="2024-01-01"))
    changed = kb.set_status(old.record_id, "superseded", note=f"replaced by record {new.record_id}", valid_to="2024-01-01")
    assert changed.status == "superseded" and changed.valid_to == "2024-01-01" and changed.value == "beagle" and len(kb.trace(old.record_id)) == 2
    assert [r.record_id for r in kb.about("asha")] == [new.record_id]
    assert len(kb.about("Asha", statuses=None)) == 2
    assert kb.set_status(old.record_id, "disputed").status == "disputed"
    assert kb.set_status(old.record_id, "retracted").status == "retracted"
    with pytest.raises(RecordError): kb.set_status(old.record_id, "needs_review")
    with pytest.raises(RecordError): kb.set_status(999, "active")
    m.close()


def test_similar_names_are_not_merged(path):
    m, kb, ids = setup(path)
    kb.add(beagle(ids))
    kb.add(beagle(ids, subject="  ASHA "))
    kb.add(beagle(ids, subject="Ash"))
    assert len(kb.about("asha")) == 2 and len(kb.about("ash")) == 1
    m.close()


def test_records_do_not_change_the_memory_or_retrieval(path):
    data = DATA
    plain = E.ChatMemory(path + "p")
    fill(plain, data)
    m, kb, ids = setup(path)
    before = record(m)
    q = data["questions"][0]["q"]
    found = m.find(q)[0], m.context_for(q)[0]
    kb.add(beagle(ids))
    assert record(m) == before and record(m) == record(plain)
    assert (m.find(q)[0], m.context_for(q)[0]) == found
    plain.close(), m.close()


def test_no_conversation_text_is_copied(path):
    m, kb, ids = setup(path)
    kb.add(beagle(ids))
    dump = json.dumps([m.db.execute(f"SELECT * FROM {t}").fetchall() for t in ("entities", "knowledge_records", "record_evidence")])
    assert not any(r[6] in dump for r in record(m))
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
    saved = kb.add(KnowledgeRecord(subject="A", fact_type="relationship", predicate="owns", value="dog", source_ids=[1]))
    assert kb.trace(saved.record_id)[0].text == "My dog is called Rex."
    m.close()


def test_a_saved_record_survives_a_hard_crash(path):
    """Kills the process right after saving, with no clean close, as a power cut would."""
    import os, signal, subprocess, sys
    from conftest import ROOT
    code = f"""
import sys, time
sys.path.insert(0, {ROOT!r})
import engram_chat as E
from engram_records import KnowledgeBase, KnowledgeRecord
m = E.ChatMemory({path!r})
m.open_chat("1 May 2023")
a = m.add("Asha", "I adopted a dog called Rex.", date="1 May 2023")
b = m.add("Asha", "Rex is my dog.", date="1 May 2023")
r = KnowledgeBase(m.db).add(KnowledgeRecord(subject="Asha", fact_type="relationship", predicate="owns", value="dog", source_ids=[a, b]))
print(r.record_id, flush=True)
time.sleep(60)
"""
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    record_id = int(p.stdout.readline())
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    m = E.ChatMemory(path)
    kb = KnowledgeBase(m.db)
    assert kb.get(record_id).value == "dog" and len(kb.trace(record_id)) == 2
    m.close()
