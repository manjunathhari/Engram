"""P1.4 beyond Manj's suite: deletion stays deleted, privacy of the files, and what an import leaves out."""
import json
import os
import sqlite3
import stat

import pytest

import engram_chat as E
from conftest import fill, load
from engram_knowledge import backup, export_knowledge, import_knowledge
from engram_records import Entity, EntityAlias, FactEvidence, FactWithEvidence, KnowledgeBase, KnowledgeFact, RecordError

DATA = load("basic_conversation.json")


def setup(path):
    m = E.ChatMemory(path); ids = fill(m, DATA); kb = KnowledgeBase(m.db)
    a = kb.add_entity(Entity.create("Asha", "person"))
    def put(sids, value="beagle", roles=None):
        f = KnowledgeFact.create(a.entity_id, "owns", "text", "relationship", "explicit", object_value=value)
        return kb.add(FactWithEvidence(f, tuple(FactEvidence.create(f.fact_id, s, evidence_role=r) for s, r in zip(sids, roles or ["supports"] * len(sids))))).fact.fact_id
    return m, kb, ids, a, put


def restore(path, tmp):
    """The same memory before records, as a backup taken earlier would be."""
    m = E.ChatMemory(str(tmp)); kb = KnowledgeBase(m.db)
    return m, kb


def test_a_sentence_deleted_after_export_stays_deleted_on_import(tmp_path):
    m, kb, ids, a, put = setup(str(tmp_path / "m.db"))
    one = put([ids["B1:1"]]); both = put([ids["B1:1"], ids["B3:1"]], value="cat")
    kb.add_alias(EntityAlias.create(a.entity_id, "Ash", "sentence", source_id=ids["B1:1"]))
    export_knowledge(kb, tmp_path / "k.json")
    m.forget_sentence(ids["B1:1"])
    for t in ("record_evidence", "knowledge_records", "entity_aliases", "entities"): m.db.execute(f"DELETE FROM {t}")
    m.db.commit()
    got = import_knowledge(kb, tmp_path / "k.json")
    assert kb.get(one) is None and [s.sid for s in kb.trace(both)] == [ids["B3:1"]] and kb.find("Ash") == []
    assert got["left_out"]["facts"] == 1 and got["left_out"]["evidence"] == 2 and got["left_out"]["aliases"] == 1 and got["note"]
    text = json.dumps([m.db.execute(f"SELECT * FROM {t}").fetchall() for t in ("entities", "entity_aliases", "knowledge_records", "record_evidence")])
    assert "I adopted a beagle" not in text and kb.integrity_check() == []
    m.close()


def test_a_changed_sentence_does_not_carry_a_record(tmp_path):
    m, kb, ids, a, put = setup(str(tmp_path / "m.db"))
    f = put([ids["B1:1"]]); export_knowledge(kb, tmp_path / "k.json")
    for t in ("record_evidence", "knowledge_records", "entity_aliases", "entities"): m.db.execute(f"DELETE FROM {t}")
    m.db.execute("UPDATE sentences SET text='Something else entirely.' WHERE sid=?", (ids["B1:1"],)); m.db.commit()
    assert import_knowledge(kb, tmp_path / "k.json")["left_out"]["facts"] == 1 and kb.get(f) is None
    m.close()


def test_into_a_different_memory_nothing_lands_on_the_wrong_sentence(tmp_path):
    m, kb, ids, a, put = setup(str(tmp_path / "m.db"))
    put([ids["B1:1"]]); export_knowledge(kb, tmp_path / "k.json")
    other = E.ChatMemory(str(tmp_path / "other.db")); other.open_chat("1:00 pm on 1 May, 2023")
    for n in range(40): other.add("Ben", f"Unrelated sentence {n}.", date="1:00 pm on 1 May, 2023")
    other.close_chat(write_note=False, sleep=False)
    okb = KnowledgeBase(other.db)
    got = import_knowledge(okb, tmp_path / "k.json")
    assert got["left_out"]["facts"] == 1 and other.db.execute("SELECT count(*) FROM knowledge_records").fetchone()[0] == 0
    m.close(); other.close()


def test_importing_twice_adds_nothing_new(tmp_path):
    m, kb, ids, a, put = setup(str(tmp_path / "m.db"))
    put([ids["B1:1"]], roles=None); export_knowledge(kb, tmp_path / "k.json")
    got = import_knowledge(kb, tmp_path / "k.json")
    assert all(v == 0 for v in got["counts"].values()) and m.db.execute("SELECT count(*) FROM knowledge_records").fetchone()[0] == 1
    m.close()


def test_files_are_private_and_never_overwritten(tmp_path):
    m, kb, ids, a, put = setup(str(tmp_path / "m.db"))
    put([ids["B1:1"]])
    export_knowledge(kb, tmp_path / "k.json"); backup(m.db, str(tmp_path / "b.db"))
    for p in ("k.json", "b.db"): assert stat.S_IMODE(os.stat(tmp_path / p).st_mode) == 0o600
    with pytest.raises(FileExistsError): export_knowledge(kb, tmp_path / "k.json")
    with pytest.raises(RecordError, match="never replaces"): backup(m.db, str(tmp_path / "b.db"))
    m.close()


def test_a_damaged_store_is_not_exported(tmp_path):
    m, kb, ids, a, put = setup(str(tmp_path / "m.db"))
    put([ids["B1:1"]]); m.db.execute("DELETE FROM sentences WHERE sid=?", (ids["B1:1"],)); m.db.commit()
    with pytest.raises(RecordError, match="not exported"): export_knowledge(kb, tmp_path / "k.json")
    assert not (tmp_path / "k.json").exists()
    m.close()


@pytest.mark.parametrize("change,why", [
    (lambda p: p.update(format="other"), "not an Engram"), (lambda p: p.update(schema_version=1), "this code reads"),
    (lambda p: p["data"]["knowledge_records"][0].update(object_value="cat"), "checksum"), (lambda p: p.pop("checksum"), "checksum")])
def test_bad_exports_change_nothing(tmp_path, change, why):
    m, kb, ids, a, put = setup(str(tmp_path / "m.db"))
    put([ids["B1:1"]]); export_knowledge(kb, tmp_path / "k.json")
    p = json.loads((tmp_path / "k.json").read_text()); change(p); (tmp_path / "bad.json").write_text(json.dumps(p))
    before = m.db.execute("SELECT * FROM knowledge_records").fetchall()
    with pytest.raises(RecordError, match=why): import_knowledge(kb, tmp_path / "bad.json")
    with pytest.raises(RecordError, match="could not be read"): import_knowledge(kb, tmp_path / "missing.json")
    assert m.db.execute("SELECT * FROM knowledge_records").fetchall() == before
    m.close()


def test_an_export_breaking_a_store_rule_imports_nothing(tmp_path):
    m, kb, ids, a, put = setup(str(tmp_path / "m.db"))
    put([ids["B1:1"]]); export_knowledge(kb, tmp_path / "k.json")
    for t in ("record_evidence", "knowledge_records", "entity_aliases", "entities"): m.db.execute(f"DELETE FROM {t}")
    m.db.commit()
    p = json.loads((tmp_path / "k.json").read_text()); p["data"]["knowledge_records"][0]["status"] = "banana"
    import engram_knowledge.serialization as Z; p["checksum"] = Z.checksum(p["data"]); (tmp_path / "bad.json").write_text(json.dumps(p))
    with pytest.raises(RecordError, match="Nothing was imported"): import_knowledge(kb, tmp_path / "bad.json")
    assert m.db.execute("SELECT count(*) FROM entities").fetchone()[0] == 0
    m.close()


def test_backup_restores_to_a_working_memory(tmp_path):
    m, kb, ids, a, put = setup(str(tmp_path / "m.db"))
    f = put([ids["B1:1"]]); backup(m.db, str(tmp_path / "b.db")); m.close()
    r = E.ChatMemory(str(tmp_path / "b.db")); rkb = KnowledgeBase(r.db)
    assert rkb.get(f).fact.value == "beagle" and rkb.trace(f)[0].sid == ids["B1:1"] and rkb.integrity_check() == []
    q = DATA["questions"][0]["q"]; assert r.find(q)[0]
    r.close()
