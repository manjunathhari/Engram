"""P2.5 cautions shown live, and P6 recording what is shown in shadow (engram_knowledge/recall_feedback.py)."""
import json

import pytest

import engram_chat as E
from engram_knowledge.database import KnowledgeBase
from engram_knowledge.filing import FactFiler
from engram_knowledge.recall_feedback import category, settle, shadow_scores
from engram_knowledge.models import RecordError
from engram_knowledge.state_engine import StateEngine

DAY = "1:00 pm on 20 May, 2023"


def writer_of(*facts):
    return lambda prompt, schema: json.dumps(dict(facts=[dict(dict(time="", ended=False), **f) for f in facts]))


def fact(who, kind, topic, thing, text, lines):
    return dict(who=who, kind=kind, topic=topic, thing=thing, fact=text, lines=lines)


@pytest.fixture
def mem(tmp_path):
    w = writer_of(fact("Nate", "attribute", "pet", "turtles", "Nate owns two turtles.", ["L1"]),
                  fact("Nate", "attribute", "home", "Tampa", "Nate lives in Tampa.", ["L2"]),
                  fact("Nate", "preference", "food", "pizza", "Nate loves pizza.", ["L3"]))
    m = E.ChatMemory(str(tmp_path / "m.db"), durable=False, filer=lambda x: FactFiler(x, w), use_filed=True, use_feedback=True, record_use=True)
    m.open_chat(DAY)
    for t in ("I have two turtles.", "I live in Tampa near the beach.", "I love pizza on Fridays."): m.add("Nate", t, date=DAY)
    m.close_chat(write_note=False)
    m.fid = {r["thing"]: r["fact_id"] for r in m.filer.records()}
    m.sid = {t: s for s, t in m.db.execute("SELECT sid, text FROM sentences")}
    yield m
    m.close()


Q = "Where does Nate live and what pets does he have?"


def test_outdated_shows_a_caution_queues_a_review_and_p3_waits(mem):
    out = mem.give_feedback("fact:" + mem.fid["Tampa"], "outdated", Q)
    assert "caution: marked outdated" in mem.context_for(Q)[0]
    kb = KnowledgeBase(mem.db)
    assert out["review"] and json.loads(kb.review(out["review"])["candidate"])["kind"] == "feedback_review"
    assert StateEngine(kb).get_current_state(kb.resolve("Nate"))["topic:home"].waiting == [out["review"]]
    mem.give_feedback("fact:" + mem.fid["Tampa"], "confirm", Q)
    assert "caution" not in mem.context_for(Q)[0]


def test_a_caution_on_a_sentence(mem):
    mem.give_feedback(f"sentence:{mem.sid['I live in Tampa near the beach.']}", "wrong", Q)
    assert "I live in Tampa near the beach. [caution: marked wrong]" in mem.context_for(Q)[0]


def test_irrelevant_hides_for_that_question_only(mem):
    mem.give_feedback(f"sentence:{mem.sid['I love pizza on Fridays.']}", "irrelevant", Q)
    mem.give_feedback("fact:" + mem.fid["pizza"], "irrelevant", Q)
    assert "pizza" not in mem.context_for(Q)[0]
    assert "pizza" in mem.context_for("What food does Nate love?")[0]


def test_helpful_moves_a_filed_fact_up_for_that_question(mem):
    q = "Tell me about Nate."
    first = mem.filer.find(q)[0]["thing"]
    other = next(t for t in ("turtles", "Tampa", "pizza") if t != first)
    for _ in range(3): mem.give_feedback("fact:" + mem.fid[other], "helpful", q)
    assert mem.filer.find(q, view=__import__("engram_knowledge.recall_feedback", fromlist=["FeedbackView"]).FeedbackView(mem.db))[0]["thing"] == other


def test_p6_records_what_was_shown_without_text(mem):
    ctx, info = mem.context_for(Q)
    assert info["retrieval_event"] and info["shown_refs"]
    row = mem.db.execute("SELECT event_type, context_key, candidate_refs, metadata FROM p6_experience_events WHERE event_id=?", (info["retrieval_event"],)).fetchone()
    assert row[0] == "retrieval" and row[1] == category(Q) and set(json.loads(row[2])) == set(info["shown_refs"])
    dump = json.dumps(mem.db.execute("SELECT * FROM p6_experience_events").fetchall())
    assert "Tampa" not in dump and "turtles" not in dump and "pets" not in dump.lower()
    fb = mem.give_feedback(info["shown_refs"][0], "helpful", Q, retrieval_event=info["retrieval_event"])
    assert mem.db.execute("SELECT source_event_ref FROM p6_experience_events WHERE event_id=?", (fb["event"],)).fetchone()[0] == info["retrieval_event"]


def test_p6_learns_in_shadow_and_changes_nothing_shown(mem):
    q1, q2 = "Where does Nate live now?", "Where does Nate live these days?"
    assert category(q1) == category(q2)
    before = mem.context_for(q2)[0]
    for n in range(5): mem.give_feedback("fact:" + mem.fid["pizza"], "irrelevant" if n % 2 else "wrong", q1)
    assert shadow_scores(mem.db, q2, ["fact:" + mem.fid["pizza"]])["fact:" + mem.fid["pizza"]] < 1.0  # P6 has learned
    after = mem.context_for(q2)[0]
    assert after.replace(" | caution: marked wrong", "") == before  # only P2.5's caution differs; P6's score is not used


def test_settle_retract_or_keep(mem):
    kb = KnowledgeBase(mem.db)
    r1 = mem.give_feedback("fact:" + mem.fid["Tampa"], "outdated", Q)["review"]
    settle(mem.db, r1, keep=False, reviewer="Manj", reason="moved away")
    assert "Tampa" not in [r["thing"] for r in mem.filer.records()] and kb.get(mem.fid["Tampa"]).fact.status == "retracted"
    assert "topic:home" not in StateEngine(kb).get_current_state(kb.resolve("Nate"))
    r2 = mem.give_feedback("fact:" + mem.fid["turtles"], "wrong", Q)["review"]
    settle(mem.db, r2, keep=True, reviewer="Manj", reason="checked, still true")
    assert "caution" not in mem.context_for(Q)[0] and kb.reviews() == []
    r3 = mem.give_feedback(f"sentence:{mem.sid['I have two turtles.']}", "wrong", Q)["review"]
    with pytest.raises(RecordError, match="never retracted"): settle(mem.db, r3, keep=False, reviewer="Manj", reason="x")
    with pytest.raises(RecordError, match="reviewer"): settle(mem.db, r3, keep=True, reviewer="", reason="x")


def test_a_retried_click_counts_once(mem):
    a = mem.give_feedback("fact:" + mem.fid["Tampa"], "outdated", Q, idempotency_key="click-1")
    b = mem.give_feedback("fact:" + mem.fid["Tampa"], "outdated", Q, idempotency_key="click-1")
    assert a["event"] == b["event"] and b["review"] is None and len(KnowledgeBase(mem.db).reviews()) == 1


def test_deleting_the_sentence_removes_its_feedback_and_review(mem):
    s = mem.sid["I have two turtles."]
    mem.give_feedback("fact:" + mem.fid["turtles"], "wrong", Q)
    mem.give_feedback(f"sentence:{s}", "outdated", Q)
    mem.forget_sentence(s)
    left = json.dumps(mem.db.execute("SELECT target_ref, candidate_refs FROM p6_experience_events").fetchall())
    assert f"sentence:{s}" not in left and mem.fid["turtles"] not in left
    assert KnowledgeBase(mem.db).reviews() == [] and KnowledgeBase(mem.db).integrity_check() == []


def test_switches_off_change_nothing(tmp_path):
    w = writer_of(fact("Nate", "attribute", "pet", "turtles", "Nate owns two turtles.", ["L1"]))
    paths = {}
    for name, on in (("off", False), ("on", True)):
        m = E.ChatMemory(str(tmp_path / f"{name}.db"), durable=False, filer=lambda x: FactFiler(x, w), use_filed=True, use_feedback=on, record_use=on)
        m.open_chat(DAY); m.add("Nate", "I have two turtles.", date=DAY); m.add("Nate", "I live in Tampa near the beach.", date=DAY); m.close_chat(write_note=False)
        paths[name] = m
    off = paths["off"]
    sid = off.db.execute("SELECT sid FROM sentences WHERE text LIKE '%Tampa%'").fetchone()[0]
    clean = off.context_for(Q)[0]
    assert "Tampa near the beach" in clean
    off.give_feedback(f"sentence:{sid}", "wrong", Q)
    assert off.context_for(Q)[0] == clean and off.context_for(Q)[1].get("retrieval_event") is None
    assert off.db.execute("SELECT count(*) FROM p6_experience_events WHERE event_type='retrieval'").fetchone()[0] == 0
    assert paths["on"].context_for(Q)[0] == clean  # no feedback yet: identical


@pytest.mark.parametrize("target, signal, why", [("fact:none", "wrong", "no such fact"), ("sentence:999", "wrong", "no such sentence"), ("sentence:1", "banana", "feedback must be")])
def test_bad_feedback_is_refused(mem, target, signal, why):
    with pytest.raises(RecordError, match=why): mem.give_feedback(target, signal, Q)
