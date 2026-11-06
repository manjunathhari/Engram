"""Scenarios written separately from the engine, by an author who saw the rules but not the code."""
import calendar
import json
import os

import pytest

import engram_chat as E
from engram_knowledge import StateEngine
from engram_knowledge.normalizer import PREDICATE_TYPES
from engram_knowledge.transition import slot_of
from engram_records import Entity, FactEvidence, FactWithEvidence, KnowledgeBase, KnowledgeFact

HERE = os.path.dirname(__file__)
FILE = os.path.join(HERE, "fixtures", "state_scenarios.json")
SCENARIOS = json.load(open(FILE)) if os.path.exists(FILE) else []


def long_date(d):
    y, mo, day = d.split("-")
    return f"1:00 pm on {int(day)} {calendar.month_name[int(mo)]}, {y}"


def build(s, path):
    m = E.ChatMemory(path)
    kb = KnowledgeBase(m.db)
    engine = StateEngine(kb)
    sids = {}
    for x in s["sentences"]:
        d = long_date(x["date"])
        m.open_chat(d)
        sids[x["id"]] = m.add(x["speaker"], x["text"], date=d)
        m.close_chat(write_note=False, sleep=False)
    ids = []
    for o in s["observations"]:
        who = kb.resolve(o["subject"]) or kb.add_entity(Entity.create(o["subject"], "person")).entity_id
        f = KnowledgeFact.create(who, o["predicate"], "text", "plan" if o.get("plan") else PREDICATE_TYPES[o["predicate"]], "explicit", object_value=o["object"], valid_from=o.get("valid_from"), valid_to=o.get("valid_to"), validation_status="accepted")
        r = FactWithEvidence(f, tuple(FactEvidence.create(f.fact_id, sids[x]) for x in o["sources"]))
        if o.get("correction_of") is not None: engine.apply_correction(ids[o["correction_of"]], r)
        else: engine.apply_observation(r)
        ids.append(f.fact_id)
    return m, kb, engine


def check(s, path):
    m, kb, engine = build(s, path)
    problems = []
    who = lambda name: kb.resolve(name)
    for e in s["expect"].get("current", []):
        st = engine.get_current_state(who(e["subject"])).get(e["slot"]) if who(e["subject"]) else None
        got = (sorted(p.value for p in st.values), st.conflict) if st else ([], False)
        if got != (sorted(e["values"]), e["conflict"]): problems.append(("current", e, got))
    for e in s["expect"].get("at", []):
        st = engine.get_state_at(who(e["subject"]), e["date"]).get(e["slot"]) if who(e["subject"]) else None
        got = (sorted(p.value for p in st.values), st.conflict) if st else ([], False)
        if got != (sorted(e["values"]), e["conflict"]): problems.append(("at", e, got))
    for e in s["expect"].get("history", []):
        got = [dict(value=p.value, start=p.start, end=p.end) for p in engine.get_history(who(e["subject"]), e["slot"])["periods"]] if who(e["subject"]) else []
        key = lambda x: (x["value"], x["start"] or "", x["end"] or "")
        if sorted(got, key=key) != sorted(e["periods"], key=key): problems.append(("history", e, got))
    # No silent loss: every observation still has its evidence, and can be found in the store.
    for row in m.db.execute("SELECT fact_id FROM knowledge_records").fetchall():
        if not kb.trace(row[0]): problems.append(("lost evidence", row[0]))
    m.close()
    return problems


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_scenario(scenario, tmp_path):
    assert check(scenario, str(tmp_path / "s.db")) == []
