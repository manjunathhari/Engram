"""Whole-chat filing into the knowledge store (engram_knowledge/filing.py), with stand-in writer and checker models."""
import datetime as dt
import json

import pytest

import engram_chat as E
from engram_knowledge.database import KnowledgeBase
from engram_knowledge.filing import DROPPED, REVIEW, STORED, FactFiler, resolve_time
from engram_knowledge.state_engine import StateEngine

DAY = "1:00 pm on 20 May, 2023"  # a Saturday


def writer_of(*facts):
    """A stand-in writer that always proposes these facts."""
    def w(prompt, schema):
        return json.dumps(dict(facts=[dict(dict(time="", ended=False), **f) for f in facts]))
    return w


def fact(who, kind, topic, thing, text, lines, **kw):
    return dict(who=who, kind=kind, topic=topic, thing=thing, fact=text, lines=lines, **kw)


def memory(tmp_path, writer, lines, date=DAY, checker=None, name="m", **kw):
    m = E.ChatMemory(str(tmp_path / f"{name}.db"), durable=False, filer=lambda mem: FactFiler(mem, writer, checker=checker, writer_name="stand-in writer", checker_name="stand-in checker" if checker else ""), use_filed=True)
    m.open_chat(date)
    for speaker, text in lines: m.add(speaker, text, date=date)
    out = m.close_chat(write_note=False)
    return m, out["filing"]


def test_stores_a_fact_with_kind_topic_dates_and_evidence(tmp_path):
    w = writer_of(fact("Nate", "event", "pet", "dog", "Nate adopted a dog named Max on 13 May 2023.", ["L2"], time="last Saturday"))
    m, f = memory(tmp_path, w, [("Nate", "Hi there!"), ("Nate", "I adopted a dog named Max last Saturday.")])
    assert f["stored"] == 1 and f["results"][0]["verdict"] == STORED
    kb = KnowledgeBase(m.db)
    rec = kb.get(f["results"][0]["fact_id"])
    assert (rec.fact.predicate, rec.fact.fact_type, rec.fact.object_value, rec.fact.valid_from) == ("topic:pet", "event", "dog", "2023-05-13")
    assert rec.fact.note.startswith("Nate adopted") and rec.fact.metadata["retention"] == "episodic" and rec.fact.metadata["said_on"] == "2023-05-20"
    assert [s.text for s in kb.trace(rec.fact.fact_id)] == ["I adopted a dog named Max last Saturday."]
    assert "[Filed fact | event | said 20 May 2023 | happened 13 May 2023] Nate adopted" in m.context_for("What pet did Nate adopt?")[0]


@pytest.mark.parametrize("proposal, line, reason", [
    (fact("Nate", "attribute", "greeting", "hi", "Nate said hi.", ["L1"]), "Hi there!", "every cited line"),
    (fact("Nate", "attribute", "pet", "cat", "Nate owns a cat.", ["L1"]), "I adopted a dog.", "thing not in its cited lines"),
    (fact("Nate", "preference", "food", "lentil soup", "Nate likes lentil soup.", ["L1"]), "I like the sound of the lentil soup.", "a reaction"),
    (fact("Nate", "preference", "nudge", "nudge", "Nate likes a nudge.", ["L1"]), "It's like a nudge to stay true to myself.", "comparison"),
    (fact("Nate", "attribute", "mood", "bored", "Nate is bored.", ["L1"]), "I am so bored today with work.", "passing feeling"),
    (fact("Nate", "attribute", "pet", "cat", "Nate owns a cat.", ["L9"]), "I adopted a cat.", "no cited line"),
    (fact("Nate", "observation", "pet", "cat", "Nate owns a cat.", ["L1"]), "I adopted a cat.", "kind not allowed"),
    (fact("Assistant", "attribute", "trail", "GR-90 trail", "The assistant suggested the GR-90 trail.", ["L1"]), "You could try the GR-90 trail.", "about the assistant"),
    (fact("Joanna", "attribute", "pet", "cat", "Joanna owns a cat.", ["L1"]), "I adopted a cat.", "not in its source"),
])
def test_drops(tmp_path, proposal, line, reason):
    speaker = "Assistant" if "GR-90" in line else "Nate"
    m, f = memory(tmp_path, writer_of(proposal), [(speaker, line)])
    assert f["dropped"] == 1 and f["stored"] == 0
    assert reason in " ".join(f["results"][0]["reasons"])


@pytest.mark.parametrize("proposal, line, reason", [
    (fact("Nate", "attribute", "pet", "cat", "Nate wants a cat.", ["L1"]), "I want to get a cat soon.", "wish"),
    (fact("Nate", "attribute", "pet", "dog", "Nate owns a dog.", ["L1"]), "I never had a dog growing up.", "not so"),
    (fact("Nate", "attribute", "job", "nurse", "Nate is a nurse.", ["L1"]), "My sister is a nurse and she loves it.", "someone else"),
])
def test_held_for_review(tmp_path, proposal, line, reason):
    m, f = memory(tmp_path, writer_of(proposal), [("Nate", line)])
    assert f["review"] == 1 and f["stored"] == 0, f
    item = KnowledgeBase(m.db).reviews()[0]
    assert json.loads(item["candidate"])["kind"] == "filed_candidate"


def test_review_waits_in_p3(tmp_path):
    m, f = memory(tmp_path, writer_of(fact("Nate", "attribute", "pet", "cat", "Nate wants a cat.", ["L1"])), [("Nate", "I want to get a cat soon.")])
    kb = KnowledgeBase(m.db)
    state = StateEngine(kb).get_current_state(kb.resolve("Nate"))
    assert state["topic:pet"].waiting and not state["topic:pet"].values


def test_same_thing_in_a_later_chat_adds_evidence_not_a_record(tmp_path):
    w = writer_of(fact("Nate", "attribute", "pet", "turtles", "Nate owns turtles.", ["L1"]))
    m = E.ChatMemory(str(tmp_path / "x.db"), durable=False, filer=lambda mem: FactFiler(mem, w), use_filed=True)
    for text in ("I took my turtles to the beach.", "My turtles love the sun."):
        m.open_chat(DAY); m.add("Nate", text, date=DAY)
        last = m.close_chat(write_note=False, sleep=False)["filing"]
    assert last["linked"] == 1 and last["stored"] == 0
    kb = KnowledgeBase(m.db)
    recs = m.filer.records()
    assert len(recs) == 1 and len(kb.trace(recs[0]["fact_id"])) == 2


def test_filing_again_adds_nothing_twice(tmp_path):
    w = writer_of(fact("Nate", "attribute", "pet", "turtles", "Nate owns turtles.", ["L1"]), fact("Nate", "attribute", "pet", "cat", "Nate wants a cat.", ["L2"]))
    m, f = memory(tmp_path, w, [("Nate", "I have two turtles."), ("Nate", "I want to get a cat soon.")])
    again = m.filer.file_chat(1)
    kb = KnowledgeBase(m.db)
    assert again["linked"] == 1 and len(m.filer.records()) == 1 and len(kb.reviews()) == 1


def test_a_collection_is_gathered_for_counting(tmp_path):
    w = writer_of(fact("Nate", "attribute", "instrument", "guitar", "Nate owns a guitar.", ["L1"]),
                  fact("Nate", "attribute", "instrument", "piano", "Nate owns a piano.", ["L2"]),
                  fact("Nate", "attribute", "instrument", "drum set", "Nate owns a drum set.", ["L3"]),
                  fact("Nate", "preference", "food", "pizza", "Nate likes pizza.", ["L4"]))
    m, f = memory(tmp_path, w, [("Nate", "I bought a guitar last year."), ("Nate", "I also have a piano at home."), ("Nate", "My drum set is in the garage."), ("Nate", "I love pizza.")])
    assert f["stored"] == 4
    found = [r["thing"] for r in m.filer.find("How many guitars and other instruments does Nate have?")]
    assert set(found[:3]) == {"guitar", "piano", "drum set"}
    kb = KnowledgeBase(m.db)
    held = StateEngine(kb).get_current_state(kb.resolve("Nate"))["topic:instrument"].values
    assert {p.value for p in held} == {"guitar", "piano", "drum set"}


def test_ended_closes_the_topic_slot(tmp_path):
    w1 = writer_of(fact("Nate", "attribute", "car", "Honda", "Nate owns a Honda.", ["L1"], time="in 2020"))
    m = E.ChatMemory(str(tmp_path / "c.db"), durable=False, filer=lambda mem: FactFiler(mem, w1), use_filed=True)
    m.open_chat(DAY); m.add("Nate", "I bought a Honda in 2020.", date=DAY); m.close_chat(write_note=False, sleep=False)
    m.filer.writer = writer_of(fact("Nate", "event", "car", "Honda", "Nate sold his Honda on 19 May 2023.", ["L1"], time="yesterday", ended=True))
    m.open_chat(DAY); m.add("Nate", "I sold the Honda yesterday.", date=DAY); f = m.close_chat(write_note=False, sleep=False)["filing"]
    assert f["stored"] == 1, f
    kb = KnowledgeBase(m.db)
    state = StateEngine(kb).get_current_state(kb.resolve("Nate"))
    assert state["topic:car"].values == []
    hist = StateEngine(kb).get_history(kb.resolve("Nate"), "topic:car")
    assert hist["periods"][0].end == "2023-05-19" and hist["periods"][0].end_kind == "closed"


def test_a_new_home_replaces_the_old_one(tmp_path):
    m = E.ChatMemory(str(tmp_path / "h.db"), durable=False, filer=lambda mem: FactFiler(mem, writer_of(fact("Nate", "attribute", "home", "Boston", "Nate lives in Boston.", ["L1"], time="in 2019"))), use_filed=True)
    m.open_chat(DAY); m.add("Nate", "I moved to Boston in 2019.", date=DAY); m.close_chat(write_note=False, sleep=False)
    m.filer.writer = writer_of(fact("Nate", "attribute", "home", "Denver", "Nate lives in Denver.", ["L1"], time="in 2022"))
    m.open_chat(DAY); m.add("Nate", "I moved to Denver in 2022.", date=DAY); m.close_chat(write_note=False, sleep=False)
    kb = KnowledgeBase(m.db)
    state = StateEngine(kb).get_current_state(kb.resolve("Nate"))["topic:home"]
    assert [p.value for p in state.values] == ["Denver"] and not state.conflict


def test_deleting_the_sentence_removes_the_filed_fact(tmp_path):
    m, f = memory(tmp_path, writer_of(fact("Nate", "attribute", "pet", "turtles", "Nate owns turtles.", ["L1"])), [("Nate", "I have two turtles.")])
    sid = m.db.execute("SELECT sid FROM sentences").fetchone()[0]
    m.forget_sentence(sid)
    assert m.filer.records() == [] and "turtles" not in m.context_for("turtles")[0]
    assert KnowledgeBase(m.db).integrity_check() == []


def test_switch_off_changes_nothing(tmp_path):
    lines = [("Nate", "I have two turtles."), ("Nate", "I bought a guitar last year.")]
    off = E.ChatMemory(str(tmp_path / "off.db"), durable=False)
    off.open_chat(DAY)
    for s, t in lines: off.add(s, t, date=DAY)
    res = off.close_chat(write_note=False)
    assert "filing" not in res and off.filer is None
    on, _ = memory(tmp_path, writer_of(fact("Nate", "attribute", "pet", "turtles", "Nate owns turtles.", ["L1"])), lines, name="on")
    on.use_filed = False
    assert on.context_for("turtles")[0] == off.context_for("turtles")[0]


def test_checker(tmp_path):
    def checker_saying(states, standing):
        return lambda prompt, schema: json.dumps(dict(states=states, standing=standing))
    p = fact("Nate", "attribute", "pet", "turtles", "Nate owns turtles.", ["L1"])
    line = [("Nate", "I have two turtles.")]
    assert memory(tmp_path, writer_of(p), line, checker=checker_saying("yes", "lasting"), name="a")[1]["stored"] == 1
    assert memory(tmp_path, writer_of(p), line, checker=checker_saying("no", "lasting"), name="b")[1]["dropped"] == 1
    assert memory(tmp_path, writer_of(p), line, checker=checker_saying("yes", "passing"), name="c")[1]["dropped"] == 1
    assert memory(tmp_path, writer_of(p), line, checker=checker_saying("partly", "lasting"), name="d")[1]["review"] == 1
    assert memory(tmp_path, writer_of(p), line, checker=checker_saying("yes", "event"), name="e")[1]["review"] == 1
    assert memory(tmp_path, writer_of(p), line, checker=lambda prompt, schema: "not json", name="f")[1]["review"] == 1


def test_the_checker_must_not_be_the_writer(tmp_path):
    w = writer_of()
    m = E.ChatMemory(str(tmp_path / "w.db"), durable=False)
    with pytest.raises(ValueError):
        FactFiler(m, w, checker=w)
    with pytest.raises(ValueError):
        FactFiler(m, w, checker=writer_of(), writer_name="gpt-4o", checker_name="gpt-4o")


def test_writer_failure_and_bad_replies_store_nothing(tmp_path):
    def broken(prompt, schema): raise RuntimeError("no connection")
    m, f = memory(tmp_path, broken, [("Nate", "I have two turtles.")], name="a")
    assert f["status"] == "failed" and m.filer.records() == []
    m, f = memory(tmp_path, lambda p, s: "I cannot do that", [("Nate", "I have two turtles.")], name="b")
    assert f["status"] == "unreadable" and m.filer.records() == []
    m, f = memory(tmp_path, lambda p, s: json.dumps(dict(facts=[dict(who="Nate")])), [("Nate", "I have two turtles.")], name="c")
    assert f["unreadable"] == 1 and m.filer.records() == []


def test_approve_a_held_proposal(tmp_path):
    m, f = memory(tmp_path, writer_of(fact("Nate", "attribute", "pet", "cat", "Nate wants a cat.", ["L1"])), [("Nate", "I want to get a cat soon.")])
    review_id = f["results"][0]["review_id"]
    stored = m.filer.approve(review_id)
    assert KnowledgeBase(m.db).get(stored["fact_id"]).fact.predicate == "topic:pet" and KnowledgeBase(m.db).reviews() == []


def test_whole_chat_prompt_shows_ids_dates_and_skip_list(tmp_path):
    seen = {}
    def w(prompt, schema):
        seen["p"], seen["s"] = prompt, schema
        return json.dumps(dict(facts=[]))
    memory(tmp_path, w, [("Nate", "Hi!"), ("Nate", "I adopted Max yesterday.")])
    assert "L1 Nate: Hi!" in seen["p"] and "L2 Nate: I adopted Max yesterday [19 May 2023]." in seen["p"]
    assert "Skip feelings and moods" in seen["p"] and seen["s"]["properties"]["facts"]["items"]["properties"]["kind"]["enum"] == ["attribute", "relationship", "preference", "event", "plan"]


@pytest.mark.parametrize("phrase, plan, expected", [
    ("yesterday", False, "2023-05-19"), ("last Saturday", False, "2023-05-13"), ("last Tues", False, "2023-05-16"), ("on Friday", False, "2023-05-19"),
    ("Friday", True, "2023-05-26"), ("this month", False, "2023-05"), ("last month", False, "2023-04"), ("next month", True, "2023-06"),
    ("in 2019", False, "2019"), ("12 March 2021", False, "2021-03-12"), ("two weeks ago", False, None), ("last week", False, None),
    ("recently", False, None), ("this weekend", True, None), ("", False, None),
])
def test_resolve_time(phrase, plan, expected):
    assert resolve_time(phrase, dt.date(2023, 5, 20), plan) == expected


def test_a_question_naming_one_item_gets_the_rest_of_its_collection(tmp_path):
    w = writer_of(fact("Nate", "preference", "food", "pizza", "Nate likes pizza.", ["L1"]),
                  fact("Nate", "attribute", "instrument", "guitar", "Nate owns a guitar.", ["L2"]),
                  fact("Nate", "attribute", "instrument", "piano", "Nate owns a piano.", ["L3"]),
                  fact("Nate", "attribute", "instrument", "drum set", "Nate owns a drum set.", ["L4"]))
    m, f = memory(tmp_path, w, [("Nate", "I have pizza every Friday."), ("Nate", "I bought a guitar last year."), ("Nate", "I also have a piano at home."), ("Nate", "My drum set is in the garage.")])
    assert [r["thing"] for r in m.filer.find("What does Nate play besides the guitar?")][:3] == ["guitar", "piano", "drum set"]


def test_topic_spellings_join_one_collection(tmp_path):
    w = writer_of(fact("Nate", "attribute", "Pets", "turtles", "Nate owns turtles.", ["L1"]), fact("Nate", "attribute", "pet", "dog", "Nate owns a dog.", ["L2"]))
    m, f = memory(tmp_path, w, [("Nate", "I have two turtles."), ("Nate", "I have a dog called Max.")])
    kb = KnowledgeBase(m.db)
    assert {p.value for p in StateEngine(kb).get_current_state(kb.resolve("Nate"))["topic:pet"].values} == {"turtles", "dog"}
