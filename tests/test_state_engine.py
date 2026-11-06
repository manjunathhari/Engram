"""P3 and P4 rules, on made-up people. The wider scenario set is in test_state_scenarios.py."""
import pytest

import engram_chat as E
from engram_knowledge import Conflict, KnowledgePipeline, KnowledgeRetriever, QueryRouter, StateEngine, Candidate
from engram_knowledge.temporal import covers, overlaps, step_before
from engram_records import Entity, EntityAlias, FactEvidence, FactWithEvidence, KnowledgeBase, KnowledgeFact


class World:
    def __init__(self, path):
        self.m = E.ChatMemory(path)
        self.kb = KnowledgeBase(self.m.db)
        self.engine = StateEngine(self.kb)
        self.ret = KnowledgeRetriever(self.m)
        self.n = 0

    def who(self, name):
        return self.kb.resolve(name) or self.kb.add_entity(Entity.create(name, "person")).entity_id

    def say(self, speaker, text, date="1:00 pm on 5 March, 2024"):
        self.m.open_chat(date)
        sid = self.m.add(speaker, text, date=date)
        self.m.close_chat(write_note=False)
        return sid

    def obs(self, subject, predicate, value, valid_from=None, speaker=None, date="1:00 pm on 5 March, 2024", plan=False, valid_to=None, status_kw=None, correct=None, source_type="explicit"):
        sid = self.say(speaker or subject, f"{subject} {predicate} {value}.", date)
        f = KnowledgeFact.create(self.who(subject), predicate, "text", "plan" if plan else "event", source_type, object_value=value, valid_from=valid_from, valid_to=valid_to, validation_status="accepted", **(status_kw or {}))
        record = FactWithEvidence(f, (FactEvidence.create(f.fact_id, sid),))
        if correct: self.engine.apply_correction(correct, record)
        else: self.engine.apply_observation(record)
        return f.fact_id, sid

    def current(self, subject, slot="works_at"):
        s = self.engine.get_current_state(self.who(subject)).get(slot)
        return ([p.value for p in s.values], s.conflict) if s else ([], False)

    def at(self, subject, date, slot="works_at"):
        s = self.engine.get_state_at(self.who(subject), date).get(slot)
        return sorted(p.value for p in s.values) if s else []

    def history(self, subject, slot="works_at"):
        return [(p.value, p.start, p.end) for p in self.engine.get_history(self.who(subject), slot)["periods"]]


@pytest.fixture
def w(tmp_path):
    world = World(str(tmp_path / "w.db"))
    yield world
    world.m.close()


def test_the_worked_example(w):
    w.obs("Dr A", "works_at", "Hospital X", "2024-01")
    w.obs("Dr A", "started_job", "Hospital Y", "2025-07")
    w.obs("Dr A", "started_job", "Hospital Z", "2026-10")
    assert w.current("Dr A") == (["Hospital Z"], False)
    assert w.history("Dr A") == [("Hospital X", "2024-01", "2025-06"), ("Hospital Y", "2025-07", "2026-09"), ("Hospital Z", "2026-10", None)]
    assert w.at("Dr A", "2024") == ["Hospital X"] and w.at("Dr A", "2025") == ["Hospital X", "Hospital Y"] and w.at("Dr A", "2023") == []
    assert w.at("Dr A", "2026-09") == ["Hospital Y"]


def test_rule1_a_new_fact_erases_nothing(w):
    a, _ = w.obs("Asha", "works_at", "X", "2024")
    w.obs("Asha", "works_at", "Y", "2025")
    assert w.kb.get(a).fact.status == "active" and w.kb.get(a).fact.valid_to is None and len(w.kb.trace(a)) == 1


def test_rule2_another_source_does_not_override(w):
    w.obs("Ben", "works_at", "X", "2024", speaker="Asha")
    w.obs("Ben", "works_at", "Y", "2026", speaker="Ravi")
    values, conflict = w.current("Ben")
    assert sorted(values) == ["X", "Y"] and conflict
    assert w.engine.get_current_state(w.who("Ben"))["works_at"].conflicts[0].kind == "value"


def test_rule2_one_speaker_updating_ends_their_own_account(w):
    w.obs("Ben", "works_at", "X", "2024", speaker="Asha")
    w.obs("Ben", "works_at", "Y", "2025", speaker="Ravi")
    w.obs("Ben", "works_at", "Y", "2026", speaker="Asha")
    state = w.engine.get_current_state(w.who("Ben"))["works_at"]
    assert [p.value for p in state.values] == ["Y"] and [c.kind for c in state.conflicts] == ["date"]  # they agree on Y, not on when
    assert w.history("Ben") == [("X", "2024", "2025"), ("Y", "2025", None)]


def test_conflict_is_reported_before_saving_and_saves_nothing(w):
    w.obs("Ben", "works_at", "X", "2024", speaker="Asha")
    sid = w.say("Ravi", "Ben works at Y.")
    f = KnowledgeFact.create(w.who("Ben"), "works_at", "text", "event", "explicit", object_value="Y", valid_from="2026", validation_status="accepted")
    r = FactWithEvidence(f, (FactEvidence.create(f.fact_id, sid),))
    assert isinstance(w.engine.detect_conflict(r)[0], Conflict) and w.kb.get(f.fact_id) is None
    assert w.engine.apply_observation(r)["conflicts"] and w.kb.get(f.fact_id)


def test_rule3_a_correction_supersedes_and_keeps_the_old_evidence(w):
    a, _ = w.obs("Asha", "works_at", "X", "2024")
    w.obs("Asha", "works_at", "Y", "2024", correct=a)
    assert w.current("Asha") == (["Y"], False) and w.history("Asha") == [("Y", "2024", None)]
    assert w.kb.get(a).fact.status == "superseded" and len(w.kb.trace(a)) == 1
    assert [x.fact.fact_id for x in w.engine.get_history(w.who("Asha"), "works_at")["not_counted"]] == [a]


def test_retroactive_correction_of_a_middle_value(w):
    w.obs("Asha", "works_at", "X", "2020")
    b, _ = w.obs("Asha", "works_at", "Y", "2022")
    w.obs("Asha", "works_at", "Z", "2024")
    w.obs("Asha", "works_at", "Q", "2022", correct=b)
    assert w.history("Asha") == [("X", "2020", "2021"), ("Q", "2022", "2023"), ("Z", "2024", None)]


def test_rule4_a_plan_never_becomes_state(w):
    w.obs("Asha", "works_at", "X", "2024")
    w.obs("Asha", "started_job", "Y", "2026", plan=True)
    assert w.current("Asha") == (["X"], False) and w.at("Asha", "2026") == ["X"] and [p.fact.object_value for p in w.engine.get_plans(w.who("Asha"))] == ["Y"]
    w.obs("Asha", "started_job", "Y", "2026-05")
    assert w.current("Asha") == (["Y"], False) and len(w.engine.get_plans(w.who("Asha"))) == 1


def test_rule5_past_questions_get_the_past(w):
    w.obs("Asha", "lives_in", "Pune", "2020")
    w.obs("Asha", "moved_to", "Goa", "2023")
    assert w.at("Asha", "2021", "lives_in") == ["Pune"] and w.at("Asha", "2024", "lives_in") == ["Goa"] and w.current("Asha", "lives_in") == (["Goa"], False)


def test_rule6_unknown_dates_stay_unknown(w):
    w.obs("Asha", "works_at", "X", date="1:00 pm on 5 March, 2023")
    p = w.engine.get_current_state(w.who("Asha"))["works_at"].values[0]
    assert p.start is None and p.first_said == "2023-03-05" and p.end is None
    assert w.at("Asha", "2023") == ["X"] and w.at("Asha", "2022") == []
    w.obs("Asha", "works_at", "Y", "2024-07", date="1:00 pm on 6 July, 2024")
    assert w.history("Asha") == [("X", None, "2024-06"), ("Y", "2024-07", None)]


def test_out_of_order_arrival_follows_dates(w):
    w.obs("Asha", "works_at", "Y", "2025")
    w.obs("Asha", "works_at", "X", "2020")
    assert w.current("Asha") == (["Y"], False) and w.history("Asha") == [("X", "2020", "2024"), ("Y", "2025", None)]


def test_same_value_repeated_is_one_period(w):
    w.obs("Asha", "works_at", "X", "2022")
    w.obs("Asha", "works_at", "x", "2023")
    assert w.history("Asha") == [("X", "2022", None)] and len(w.engine.get_current_state(w.who("Asha"))["works_at"].values[0].fact_ids) == 2


def test_sources_with_different_dates_keep_both(w):
    a, _ = w.obs("Ben", "moved_to", "Rome", "2023-05", speaker="Asha")
    b, _ = w.obs("Ben", "moved_to", "Rome", "2023-08", speaker="Ravi")
    s = w.engine.get_current_state(w.who("Ben"))["lives_in"]
    assert s.conflict and s.conflicts[0].kind == "date" and s.values[0].start == "2023-05" and set(s.conflicts[0].fact_ids) == {a, b}


def test_values_at_overlapping_dates_cannot_be_ordered(w):
    w.obs("Asha", "works_at", "X", "2025")
    w.obs("Asha", "works_at", "Y", "2025-07")
    values, conflict = w.current("Asha")
    assert sorted(values) == ["X", "Y"] and conflict


def test_closing_events(w):
    w.obs("Asha", "works_at", "X", "2022")
    w.obs("Asha", "left_job", "Q", "2023")
    assert w.current("Asha") == (["X"], True) or w.current("Asha")[0] == ["X"]
    w.obs("Asha", "left_job", "X", "2024-02")
    assert w.current("Asha")[0] == [] and w.history("Asha") == [("X", "2022", "2024-02")]


def test_several_values_can_hold_at_once(w):
    w.obs("Asha", "adopted", "Rex", "2020")
    w.obs("Asha", "bought", "Car", "2021")
    assert sorted(w.current("Asha", "owns")[0]) == ["Car", "Rex"] and not w.current("Asha", "owns")[1]
    w.obs("Asha", "sold", "Car", "2022")
    assert w.current("Asha", "owns")[0] == ["Rex"]


def test_inferred_and_rejected_do_not_count(w):
    w.obs("Asha", "works_at", "X", "2024", source_type="inferred")
    assert w.current("Asha")[0] == []
    assert len(w.engine.get_history(w.who("Asha"), "works_at")["not_counted"]) == 1


def test_a_person_can_settle_a_conflict(w):
    a, _ = w.obs("Ben", "works_at", "X", "2024", speaker="Asha")
    b, _ = w.obs("Ben", "works_at", "Y", "2026", speaker="Ravi")
    w.engine.resolve_conflict(b, a)
    assert w.current("Ben") == (["Y"], False) and len(w.kb.trace(a)) == 1


def test_deleting_a_source_changes_the_state(w):
    a, s1 = w.obs("Asha", "works_at", "X", "2020")
    b, s2 = w.obs("Asha", "works_at", "Y", "2024")
    w.m.forget_sentence(s2)
    assert w.current("Asha") == (["X"], False) and w.history("Asha") == [("X", "2020", None)]
    assert w.kb.get(b).fact.status == "invalidated"


def test_state_is_derived_and_reading_writes_nothing(w):
    w.obs("Asha", "works_at", "X", "2020")
    w.obs("Asha", "works_at", "Y", "2024")
    before = w.m.db.total_changes
    w.engine.get_current_state(w.who("Asha")); w.engine.get_state_at(w.who("Asha"), "2021"); w.engine.get_history(w.who("Asha"), "works_at"); w.ret.retrieve("Where does Asha work now?")
    assert w.m.db.total_changes == before
    assert not [r for r in w.m.db.execute("SELECT name FROM sqlite_master WHERE name LIKE '%state%'")]


def test_restart_gives_the_same_state(w, tmp_path):
    w.obs("Asha", "works_at", "X", "2020")
    w.obs("Asha", "works_at", "Y", "2024")
    before = w.history("Asha")
    w.m.close()
    again = World(str(tmp_path / "w.db"))
    assert again.history("Asha") == before and again.current("Asha") == (["Y"], False)
    w.m = again.m


@pytest.mark.parametrize("value,want", [("2025-07", "2025-06"), ("2025-01", "2024-12"), ("2025-07-01", "2025-06-30"), ("2024-03-01", "2024-02-29"), ("2025", "2024")])
def test_step_before(value, want):
    assert step_before(value) == want


def test_date_ranges():
    assert overlaps("2025", "2025-07") and overlaps("2025-07", "2025-07-09") and not overlaps("2025-06", "2025-07")
    assert covers("2024-01", "2025-06", "2025") and not covers("2024-01", "2025-06", "2025-07") and covers("2024", None, "2030") and not covers(None, None, "2024")


# ---------- P4 ----------
@pytest.mark.parametrize("question,kind", [
    ("What is the current employer of Asha?", "current_state"), ("Where does Asha work now?", "current_state"), ("Where does Asha work?", "current_state"),
    ("Where did Asha work in 2024?", "historical_state"), ("Where did Asha live in March 2023?", "historical_state"),
    ("When did Asha change jobs?", "change"), ("What evidence supports the current employer of Asha?", "evidence"),
    ("What was the original employer of Asha?", "original"), ("Are there conflicting observations about Asha's employer?", "conflict"),
    ("What did we talk about at dinner?", "general_memory"), ("Tell me about Asha's job", "combined"),
])
def test_router(question, kind):
    assert QueryRouter().classify(question) == kind


def test_extract_date():
    r = QueryRouter()
    assert r.extract_date("in 2024?") == "2024" and r.extract_date("in March 2023") == "2023-03" and r.extract_date("on 5 March, 2023") == "2023-03-05" and r.extract_date("lately") is None


@pytest.fixture
def career(w):
    a, s1 = w.obs("Asha", "works_at", "Acme", "2022-01")
    b, s2 = w.obs("Asha", "started_job", "Globex", "2024-07")
    w.ids = dict(a=a, b=b, s1=s1, s2=s2)
    return w


def test_every_returned_fact_has_the_agreed_form_and_its_sentence(career):
    r = career.ret.retrieve("What is the current employer of Asha?")
    f = r["facts"][0]
    assert r["query_type"] == "current_state" and f["fact"] == "Works at Globex" and f["status"] == "active" and f["valid_from"] == "2024-07" and f["evidence"] == [career.ids["s2"]] and f["confidence"] is None
    assert f["sentences"][0]["text"] == "Asha started_job Globex." and r["memory_sids"] == [] and r["sentences_used"] == 1


def test_questions_get_the_right_records(career):
    q = career.ret.retrieve
    assert [f["value"] for f in q("Where did Asha work in 2023?")["facts"]] == ["Acme"]
    assert sorted(f["value"] for f in q("Where did Asha work in 2024?")["facts"]) == ["Acme", "Globex"]
    assert [f["value"] for f in q("What was the original employer of Asha?")["facts"]] == ["Acme"]
    assert q("When did Asha change jobs?")["facts"][0]["valid_from"] == "2024-07"
    assert [f["value"] for f in q("What was Asha's job history over time?")["facts"]] != []
    assert q("Are there conflicting observations about Asha's employer?")["conflicts"] == []


def test_a_conflict_comes_back_with_both_sides_and_both_sentences(career):
    _, s3 = career.obs("Asha", "works_at", "Hooli", "2025", speaker="Ben")
    r = career.ret.retrieve("Are there conflicting observations about Asha's employer?")
    assert sorted(f["value"] for f in r["facts"]) == ["Globex", "Hooli"] and {s for f in r["facts"] for s in f["evidence"]} == {career.ids["s2"], s3} and all(f["conflict"] for f in r["facts"])
    now = career.ret.retrieve("Where does Asha work now?")
    assert len(now["facts"]) == 2 and now["conflicts"]


def test_no_record_never_invents(career):
    r = career.ret.retrieve("Where did Asha work in 2019?")
    assert r["facts"] == [] and r["note"] == "no record for that time"
    r = career.ret.retrieve("Where does Zed work now?")
    assert r["facts"] == [] and r["entity_id"] is None and "no single known entity" in r["note"]


def test_two_people_named_gives_no_single_answer(career):
    career.who("Ben")
    assert career.ret.retrieve("Does Asha work with Ben now?")["entity_id"] is None


def test_alias_and_given_entity(career):
    career.kb.add_alias(EntityAlias.create(career.who("Asha"), "Ash", "manual"))
    assert career.ret.retrieve("Where does Ash work now?")["facts"][0]["value"] == "Globex"
    assert career.ret.retrieve("Where does she work now?", entity_id=career.who("Asha"))["facts"][0]["value"] == "Globex"


def test_a_newer_statement_in_review_warns_that_the_answer_may_be_stale(career):
    sid = career.say("Asha", "Maybe I went to Wonka company.")
    pipe = KnowledgePipeline(career.m.db)
    pipe.process(Candidate("Asha", "works_at", "Wonka", (sid,)))
    r = career.ret.retrieve("Where does Asha work now?")
    assert r["facts"][0]["value"] == "Globex" and r["waiting_review"] and "review" in r["note"]


def test_general_questions_use_ordinary_memory_unchanged(career):
    q = "What did Asha say about Acme?"
    assert career.ret.retrieve(q)["memory_sids"] == career.m.find(q)[0]
    combined = career.ret.retrieve("Tell me about Asha's job")
    assert combined["query_type"] == "combined" and combined["facts"][0]["value"] == "Globex" and combined["memory_sids"] == career.m.find("Tell me about Asha's job")[0]


def test_text_for_a_model_holds_the_original_sentences(career):
    text = career.ret.evidence.text(career.ret.retrieve("Where does Asha work now?"))
    assert "Works at Globex (2024-07 to now)" in text and "Asha started_job Globex." in text
