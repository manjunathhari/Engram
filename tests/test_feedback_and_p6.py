"""Feedback stored once, in P6.1's log; P6 learns from it but changes nothing in search; deletion reaches it."""
import json

import pytest

import engram_chat as E
from engram_knowledge.feedback import store_feedback
from engram_p6 import AdaptiveScorer, ContextEvaluator, ExperienceRecorder
from engram_records import Entity, FactEvidence, FactWithEvidence, KnowledgeBase, KnowledgeFact, RecordError

DAY = "1:00 pm on 5 March, 2023"


@pytest.fixture
def mem(tmp_path):
    m = E.ChatMemory(str(tmp_path / "m.db"))
    m.open_chat(DAY)
    m.s1 = m.add("Asha", "I adopted a beagle called Pixel.", date=DAY)
    m.s2 = m.add("Asha", "I live in Pune.", date=DAY)
    m.close_chat(write_note=False)
    kb = KnowledgeBase(m.db)
    who = kb.add_entity(Entity.create("Asha", "person")).entity_id
    f = KnowledgeFact.create(who, "owns", "text", "relationship", "explicit", object_value="beagle")
    m.fid = kb.add(FactWithEvidence(f, (FactEvidence.create(f.fact_id, m.s1),))).fact.fact_id
    m.kb = kb
    yield m
    m.close()


def test_feedback_goes_into_the_one_event_log_without_text(mem):
    e = store_feedback(mem.db, f"fact:{mem.fid}", "helpful", "What pet does Asha have?")
    again = store_feedback(mem.db, f"fact:{mem.fid}", "helpful", "what PET does asha have")
    assert again.replayed and ExperienceRecorder(mem.db).count("feedback") == 1
    row = mem.db.execute("SELECT * FROM p6_experience_events").fetchone()
    assert "beagle" not in json.dumps(row) and "pet" not in json.dumps(row).lower()


@pytest.mark.parametrize("target,signal,why", [("fact:fact_none", "helpful", "no such fact"), ("sentence:99999", "helpful", "no such sentence"), ("memory-1", "helpful", "targets are"), ("sentence:1", "banana", "feedback must be")])
def test_bad_feedback_is_refused(mem, target, signal, why):
    with pytest.raises(RecordError, match=why): store_feedback(mem.db, target, signal, "q")


def test_p6_learns_a_bounded_score_and_search_is_unchanged(mem):
    q = "What pet does Asha have?"
    before = mem.find(q)[0]
    for n in range(4): store_feedback(mem.db, f"fact:{mem.fid}", "helpful", q, idempotency_key=f"k{n}", context_key="pets")
    assert ContextEvaluator(mem.db).evaluate(f"fact:{mem.fid}", "pets").positive_count == 4
    score = AdaptiveScorer(mem.db).score(f"fact:{mem.fid}", "pets")
    assert 1.0 < score.multiplier <= 1.2
    assert mem.find(q)[0] == before  # shadow mode: nothing in search reads the score


def test_deleting_the_sentence_removes_its_feedback_and_scores(mem):
    store_feedback(mem.db, f"fact:{mem.fid}", "wrong", "q1", idempotency_key="a", context_key="c")
    store_feedback(mem.db, f"sentence:{mem.s1}", "helpful", "q2", idempotency_key="b", context_key="c")
    store_feedback(mem.db, f"sentence:{mem.s2}", "helpful", "q3", idempotency_key="c", context_key="c")
    ExperienceRecorder(mem.db).record_retrieval(idempotency_key="r", context_key="c", query_kind="pets", candidate_refs=[f"fact:{mem.fid}", f"sentence:{mem.s2}"], selected_refs=[f"fact:{mem.fid}"])
    AdaptiveScorer(mem.db).build_policy("c", note="t")
    mem.forget_sentence(mem.s1)
    left = mem.db.execute("SELECT target_ref, candidate_refs, selected_refs FROM p6_experience_events ORDER BY occurred_at").fetchall()
    assert all(mem.fid not in json.dumps(r) and f"sentence:{mem.s1}" not in json.dumps(r) for r in left)
    assert (f"sentence:{mem.s2}",) in {(r[0],) for r in left} and any(json.loads(r[1]) == [f"sentence:{mem.s2}"] for r in left)
    assert not mem.db.execute("SELECT 1 FROM p6_scoring_policies WHERE target_ref IN (?, ?)", (f"fact:{mem.fid}", f"sentence:{mem.s1}")).fetchone()
    assert mem.kb.integrity_check() == []
