"""The store's additions: checks inside the database, values that keep their type, qualifying evidence,
facts by source sentence, the integrity check, and moving an older database to the current version."""
import json
import sqlite3

import pytest

import engram_chat as E
import engram_knowledge.schema as S
from conftest import fill, load, record
from engram_knowledge.migrations import migrate
from engram_records import Entity, EntityAlias, FactEvidence, FactWithEvidence, KnowledgeBase, KnowledgeFact, RecordError

DATA = load("basic_conversation.json")


def setup(path):
    m = E.ChatMemory(path)
    ids = fill(m, DATA)
    kb = KnowledgeBase(m.db)
    return m, kb, ids, kb.add_entity(Entity.create("Asha", "person"))


def fact(kb, who, sids, roles=None, **kw):
    fields = dict(subject_id=who.entity_id, predicate="owns", object_type="text", fact_type="relationship", source_type="explicit", object_value="beagle")
    f = KnowledgeFact.create(**(fields | kw))
    roles = roles or ["supports"] * len(sids)
    return kb.add(FactWithEvidence(f, tuple(FactEvidence.create(f.fact_id, s, evidence_role=r) for s, r in zip(sids, roles))))


# ---------- checks inside the database ----------
def raw_insert(m, table, values):
    with pytest.raises(sqlite3.IntegrityError):
        m.db.execute(f"INSERT INTO {table} VALUES({','.join('?' * len(values))})", values)
    m.db.rollback()


GOOD_FACT = ["fact_x", None, "owns", "text", "dog", None, "event", "active", "pending", "explicit", None, None, "2026-11-06T00:00:00+00:00", "{}", "", 2]


@pytest.mark.parametrize("position,bad", [(7, "banana"), (8, "maybe"), (9, "guess"), (6, "vibe"), (3, "blob"), (3, None), (2, "  "), (2, None), (4, None), (4, " "), (1, None), (13, "not json")])
def test_the_database_refuses_bad_facts_even_without_the_code(path, position, bad):
    m, kb, ids, asha = setup(path)
    row = list(GOOD_FACT); row[1] = asha.entity_id; row[position] = bad
    raw_insert(m, "knowledge_records", row)
    m.close()


def test_the_database_refuses_broken_links_and_leftover_content(path):
    m, kb, ids, asha = setup(path)
    row = list(GOOD_FACT); row[1] = "ent_missing"
    raw_insert(m, "knowledge_records", row)  # a subject that does not exist
    row = list(GOOD_FACT); row[1] = asha.entity_id; row[3] = "boolean"; row[4] = "perhaps"
    raw_insert(m, "knowledge_records", row)
    row = list(GOOD_FACT); row[1] = asha.entity_id; row[7] = "invalidated"
    raw_insert(m, "knowledge_records", row)  # invalidated but still holding content
    raw_insert(m, "record_evidence", ["ev_x", "fact_missing", "sentence", "1", "supports", None, None, "2026"])
    f = fact(kb, asha, [ids["B1:1"]]).fact.fact_id
    raw_insert(m, "record_evidence", ["ev_y", f, "sentence", str(ids["B3:1"]), "banana", None, None, "2026"])
    raw_insert(m, "entity_aliases", ["als_x", asha.entity_id, "Ash", "ash", "sentence", None, "2026"])
    raw_insert(m, "entity_aliases", ["als_y", "ent_missing", "Ash", "ash", "manual", None, "2026"])
    m.close()


def test_a_good_raw_row_is_accepted(path):
    m, kb, ids, asha = setup(path)
    row = list(GOOD_FACT); row[1] = asha.entity_id
    m.db.execute(f"INSERT INTO knowledge_records VALUES({','.join('?' * len(row))})", row)
    m.db.rollback()
    m.close()


# ---------- values that keep their type ----------
@pytest.mark.parametrize("kind,given,stored,back", [
    ("number", 42, "42", 42), ("number", "3.5", "3.5", 3.5), ("number", 2.0, "2.0", 2.0), ("boolean", True, "true", True), ("boolean", "FALSE", "false", False),
    ("json", {"b": 1, "a": [1, 2]}, '{"a": [1, 2], "b": 1}', {"a": [1, 2], "b": 1}), ("date", "2023-05", "2023-05", "2023-05"), ("text", "  red   bike ", "red bike", "red bike")])
def test_values_keep_their_type_through_storage(path, kind, given, stored, back):
    m, kb, ids, asha = setup(path)
    saved = fact(kb, asha, [ids["B1:1"]], object_type=kind, object_value=given)
    m.close()
    m = E.ChatMemory(path)
    got = KnowledgeBase(m.db).get(saved.fact.fact_id).fact
    assert got.object_value == stored and got.value == back and type(got.value) is type(back)
    m.close()


@pytest.mark.parametrize("kind,bad", [("number", "lots"), ("number", True), ("number", float("nan")), ("boolean", "perhaps"), ("json", "{not json"), ("date", "last spring")])
def test_wrong_values_for_a_type_are_refused(kind, bad):
    with pytest.raises(RecordError): KnowledgeFact.create("ent", "owns", kind, "attribute", "explicit", object_value=bad)


def test_a_stored_value_must_already_be_in_stored_form():
    with pytest.raises(RecordError, match="not a stored number"): KnowledgeFact("f", "ent", "age", "number", " 42 ", None, "attribute", "active", "pending", "explicit", None, None, "2026-11-06T00:00:00+00:00")


# ---------- qualifying evidence ----------
def test_a_qualifying_sentence_is_kept_but_never_supports(path):
    m, kb, ids, asha = setup(path)
    f = KnowledgeFact.create(asha.entity_id, "owns", "text", "relationship", "explicit", object_value="beagle")
    with pytest.raises(RecordError, match="supporting"): FactWithEvidence(f, (FactEvidence.create(f.fact_id, ids["B1:1"], evidence_role="qualifies"),))
    saved = fact(kb, asha, [ids["B1:1"], ids["B2:1"]], roles=["supports", "qualifies"])
    assert [s.evidence_role for s in kb.trace(saved.fact.fact_id)] == ["supports", "qualifies"]
    m.forget_sentence(ids["B1:1"])
    assert kb.get(saved.fact.fact_id).fact.status == "invalidated" and kb.trace(saved.fact.fact_id) == []
    m.close()


# ---------- facts by source sentence ----------
def test_facts_for_a_source_sentence(path):
    m, kb, ids, asha = setup(path)
    a = fact(kb, asha, [ids["B1:1"]]).fact.fact_id
    b = fact(kb, asha, [ids["B3:1"], ids["B1:1"]], roles=["supports", "contradicts"], object_value="cat").fact.fact_id
    fact(kb, asha, [ids["B3:1"]], object_value="fish")
    assert [x.fact.fact_id for x in kb.facts_for_source(ids["B1:1"])] == [a, b] and kb.facts_for_source(99999) == []
    m.close()


# ---------- integrity check ----------
def test_a_sound_store_has_no_problems(path):
    m, kb, ids, asha = setup(path)
    a = fact(kb, asha, [ids["B1:1"], ids["B3:1"]]).fact.fact_id
    kb.add_alias(EntityAlias.create(asha.entity_id, "Ash", "sentence", source_id=ids["B1:1"]))
    kb.add_review("{}", ["x"], [ids["B2:1"]])
    m.forget_sentence(ids["B1:1"])
    m.forget_sentence(ids["B2:1"])
    assert kb.integrity_check() == []
    m.close()


def test_the_integrity_check_finds_planted_problems(path):
    m, kb, ids, asha = setup(path)
    f = fact(kb, asha, [ids["B1:1"]]).fact.fact_id
    kb.add_alias(EntityAlias.create(asha.entity_id, "Ash", "sentence", source_id=ids["B3:1"]))
    kb.add_review("{}", ["x"], [ids["B2:1"]])
    m.db.execute("DELETE FROM sentences WHERE sid IN (?, ?, ?)", (ids["B1:1"], ids["B3:1"], ids["B2:1"]))  # deleted behind Engram's back
    m.db.execute("INSERT INTO review_sources VALUES('rev_lost', ?)", (ids["B2:4"],))
    m.db.commit()
    problems = kb.integrity_check()
    assert any("evidence" in p and "does not exist" in p for p in problems) and any(p.startswith("alias") for p in problems) and any("quotes sentence" in p for p in problems) and any("no row in the queue" in p for p in problems)
    m.db.execute("DELETE FROM record_evidence WHERE fact_id=?", (f,)); m.db.commit()
    assert any("no supporting sentence" in p for p in kb.integrity_check())
    m.close()


# ---------- moving an older database to the current version ----------
def version_one(path, extra=""):
    """A database as the first release left it: version 1 tables, a sentence, and records."""
    m = E.ChatMemory(path)
    ids = fill(m, DATA)
    for t in ("knowledge_meta", "review_sources", "review_queue", "record_evidence", "knowledge_records", "entity_aliases", "entities"): m.db.execute(f"DROP TABLE {t}")
    m.db.executescript(S.V1_SCHEMA)
    m.db.execute("INSERT INTO knowledge_meta VALUES('schema_version', '1')")
    m.db.execute("INSERT INTO entities VALUES('ent_a', 'Asha', 'asha', 'person', '2026-11-06T00:00:00+00:00', '2026-11-06T00:00:00+00:00', '{\"k\": 1}', 1)")
    m.db.execute("INSERT INTO entity_aliases VALUES('als_a', 'ent_a', 'Ash', 'ash', 'sentence', ?, '2026-11-06T00:00:00+00:00')", (str(ids["B1:1"]),))
    m.db.execute("INSERT INTO knowledge_records VALUES('fact_a', 'ent_a', 'owns', 'text', 'beagle', NULL, 'relationship', 'active', 'accepted', 'explicit', '2023-03', NULL, '2026-11-06T00:00:00+00:00', '{}', 'kept', 1)")
    m.db.execute("INSERT INTO record_evidence VALUES('ev_a', 'fact_a', 'sentence', ?, 'supports', 'test', '1', '2026-11-06T00:00:00+00:00')", (str(ids["B1:1"]),))
    m.db.execute("INSERT INTO review_queue VALUES('rev_a', '{}', '[]', '2026-11-06T00:00:00+00:00')")
    if extra: m.db.execute(extra)
    m.db.commit()
    before = {t: m.db.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in ("entities", "entity_aliases", "knowledge_records", "record_evidence", "review_queue")}
    sentences = record(m)
    m.db.close()
    return ids, before, sentences


def test_a_version_one_database_is_brought_up_to_date_with_nothing_lost(path):
    ids, before, sentences = version_one(path)
    m = E.ChatMemory(path)
    kb = KnowledgeBase(m.db)
    after = {t: m.db.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in before}
    assert after == before and record(m) == sentences
    assert m.db.execute("SELECT value FROM knowledge_meta").fetchone()[0] == str(S.SCHEMA_VERSION)
    assert kb.get("fact_a").fact.value == "beagle" and [s.text for s in kb.trace("fact_a")] == ["I adopted a beagle called Pixel from the shelter yesterday."]
    assert kb.integrity_check() == []
    kb.add_alias(EntityAlias.create(kb.add_entity(Entity.create("Bo", "pet")).entity_id, "Ash", "manual"))  # version 2 lets a second entity be called Ash
    assert len(kb.find("Ash")) == 2
    raw_insert(m, "knowledge_records", ["fact_b", "ent_a", "owns", "text", "dog", None, "event", "banana", "pending", "explicit", None, None, "x", "{}", "", 2])  # and has the checks
    m.forget_sentence(ids["B1:1"])
    assert kb.get("fact_a").fact.status == "invalidated" and kb.find("Ash") != [] and kb.integrity_check() == []
    m.close()
    m = E.ChatMemory(path)
    assert KnowledgeBase(m.db).migrated == []  # only once
    m.close()


def test_the_first_release_without_a_version_table_is_recognised(path):
    ids, before, sentences = version_one(path)
    raw = sqlite3.connect(path); raw.execute("DROP TABLE knowledge_meta"); raw.commit(); raw.close()
    m = E.ChatMemory(path)
    after = {t: m.db.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in before}
    assert after == before and m.db.execute("SELECT value FROM knowledge_meta").fetchone()[0] == str(S.SCHEMA_VERSION)
    raw_insert(m, "knowledge_records", ["fact_b", "ent_a", "owns", "text", "dog", None, "event", "banana", "pending", "explicit", None, None, "x", "{}", "", 2])  # the new checks are really there
    m.close()


@pytest.mark.parametrize("extra,why", [
    ("INSERT INTO record_evidence VALUES('ev_b', 'fact_missing', 'sentence', '1', 'supports', NULL, NULL, 'x')", "links"),
    ("INSERT INTO knowledge_records VALUES('fact_b', 'ent_a', 'owns', 'text', 'dog', NULL, 'event', 'banana', 'pending', 'explicit', NULL, NULL, 'x', '{}', '', 1)", "breaks a rule"),
])
def test_old_data_that_breaks_the_new_rules_changes_nothing(path, extra, why):
    ids, before, sentences = version_one(path, extra)
    with pytest.raises(RecordError, match=why): E.ChatMemory(path)
    raw = sqlite3.connect(path)
    assert {t: raw.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in before} == before and raw.execute("SELECT value FROM knowledge_meta").fetchone()[0] == "1"
    assert not raw.execute("SELECT name FROM sqlite_master WHERE name LIKE '%_v1'").fetchall()
    raw.close()


def test_migrate_refuses_what_it_cannot_do(path):
    db = sqlite3.connect(path)
    for v in ("0", "3", "banana"):
        with pytest.raises(RecordError, match="Nothing was changed"): migrate(db, v)
    assert migrate(db, str(S.SCHEMA_VERSION)) == []
    db.close()
