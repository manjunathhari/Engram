"""P2: which entity a statement refers to, and whether a candidate may be stored. Made-up conversations."""
import inspect
import json

import pytest

import engram_chat as E
import engram_knowledge as K
from engram_knowledge import ACCEPTED, REJECTED, UNCERTAIN, Candidate, FactValidator, KnowledgePipeline, parse_reply
from engram_knowledge.fact_validator import SourceLine
from engram_knowledge import normalizer as N
from engram_records import Entity, EntityAlias, KnowledgeBase, RecordError

DAY = "2:15 pm on 3 March, 2023"
LINES = {
    "adopt": ("Asha", "I adopted a beagle called Pixel yesterday."),
    "adopt2": ("Asha", "Pixel the beagle is one now and I adopted well."),
    "not": ("Asha", "I never adopted a parrot."),
    "ask": ("Asha", "Should I adopt a hamster?"),
    "maybe": ("Asha", "Maybe I adopted a rabbit by accident."),
    "two": ("Asha", "Should I go? I adopted a kitten."),
    "she": ("Asha", "My sister adopted a turtle."),
    "you": ("Asha", "You adopted a pony."),
    "other": ("Ben", "Asha adopted a goat."),
    "plain": ("Asha", "The weather was nice and we saw a lizard."),
    "bot": ("Assistant", "Asha adopted a ferret, I recall."),
    "ben": ("Ben", "I adopted a llama."),
}


@pytest.fixture
def m(tmp_path):
    memory = E.ChatMemory(str(tmp_path / "m.db"))
    memory.open_chat(DAY)
    memory.ids = {k: memory.add(s, t, date=DAY) for k, (s, t) in LINES.items()}
    yield memory
    memory.close()


def cand(m, subject, predicate, obj, *keys, time=""):
    return Candidate(subject, predicate, obj, tuple(m.ids[k] for k in keys), time)


def verdict(m, *a, **k):
    pipe = KnowledgePipeline(m.db)
    c = cand(m, *a, **k)
    return pipe.validator.validate(c, pipe.lines(c))


def test_a_plain_statement_is_accepted(m):
    d = verdict(m, "Asha", "adopted", "beagle", "adopt")
    assert d.verdict == ACCEPTED and d.reasons == () and d.subject.status == "new" and d.subject.name == "Asha"


def test_first_person_means_the_speaker(m):
    assert verdict(m, "I", "adopted", "beagle", "adopt").subject.name == "Asha"
    assert verdict(m, "I", "adopted", "beagle", "adopt", "ben").subject.status == "ambiguous"


@pytest.mark.parametrize("mention", ["he", "she", "They", "my friend", "you"])
def test_pronouns_are_never_guessed(m, mention):
    d = verdict(m, mention, "adopted", "beagle", "adopt")
    assert d.verdict == UNCERTAIN and d.subject.status == "unresolved"


def test_a_name_not_in_its_source_is_not_accepted(m):
    assert verdict(m, "Carol", "adopted", "beagle", "adopt").verdict == REJECTED
    KnowledgeBase(m.db).add_entity(Entity.create("Carol", "person"))
    d = verdict(m, "Carol", "adopted", "beagle", "adopt")
    assert d.verdict == REJECTED and any("speaker of the source" in r for r in d.reasons)


def test_alias_reaches_the_entity_and_a_similar_name_goes_to_review(m):
    kb = KnowledgeBase(m.db)
    a = kb.add_entity(Entity.create("Asha Rao", "person"))
    kb.add_alias(EntityAlias.create(a.entity_id, "Asha", "manual"))
    assert verdict(m, "Asha", "adopted", "beagle", "adopt").subject.entity_id == a.entity_id
    kb2 = KnowledgeBase(m.db)
    b = kb2.add_entity(Entity.create("Ben Thomas", "person"))
    d = verdict(m, "Ben", "adopted", "llama", "ben")
    assert d.verdict == UNCERTAIN and d.subject.status == "ambiguous" and d.subject.entity_id is None and kb.resolve("Ben") is None


@pytest.mark.parametrize("args,why", [
    (("Asha", "adopted", "parrot", "not"), "negate"),
    (("Asha", "adopted", "hamster", "ask"), "question"),
    (("Asha", "adopted", "rabbit", "maybe"), "hypothetical"),
    (("Asha", "adopted", "turtle", "she"), "someone else"),
    (("Asha", "adopted", "pony", "you"), "someone else"),
    (("Asha", "adopted", "goat", "other"), "reported"),
    (("Asha", "adopted", "lizard", "plain"), "nothing in the source says adopted"),
    (("Asha", "adopted", "ferret", "bot"), "assistant"),
])
def test_doubtful_statements_go_to_review(m, args, why):
    d = verdict(m, *args)
    assert d.verdict == UNCERTAIN and any(why in r for r in d.reasons), d.reasons


@pytest.mark.parametrize("args,why", [
    (("Asha", "adopted", "dragon", "adopt"), "object not found"),
    (("Asha", "adopted", "the", "adopt"), "no meaning"),
    (("Asha", "banana", "beagle", "adopt"), "predicate not allowed"),
    (("Asha", "adopted", "llama", "ben"), "not in its source"),
])
def test_unsupported_statements_are_rejected(m, args, why):
    d = verdict(m, *args)
    assert d.verdict == REJECTED and any(why in r for r in d.reasons), d.reasons


def test_no_source_or_a_missing_source_is_rejected(m):
    pipe = KnowledgePipeline(m.db)
    assert pipe.process(Candidate("Asha", "adopted", "beagle"))[0].verdict == REJECTED
    assert pipe.process(Candidate("Asha", "adopted", "beagle", (m.ids["adopt"], 99999)))[0].verdict == REJECTED
    assert pipe.process(Candidate("Asha", "adopted", "beagle", (None,)))[0].verdict == REJECTED


def test_only_the_sentence_holding_the_object_is_judged(m):
    assert verdict(m, "Asha", "adopted", "kitten", "two").verdict == ACCEPTED


def test_time_must_come_from_the_source(m):
    assert verdict(m, "Asha", "adopted", "beagle", "adopt", time="2 March 2023").time == "2023-03-02"
    assert verdict(m, "Asha", "adopted", "beagle", "adopt", time="2 March 2023").verdict == ACCEPTED
    assert any("time is not in" in r for r in verdict(m, "Asha", "adopted", "beagle", "adopt", time="9 May 2021").reasons)
    assert any("plain date" in r for r in verdict(m, "Asha", "adopted", "beagle", "adopt", time="sometime").reasons)


def test_the_model_has_no_say():
    assert "verdict" not in Candidate.__dataclass_fields__ and "confidence" not in Candidate.__dataclass_fields__
    assert list(inspect.signature(FactValidator.validate).parameters) == ["self", "candidate", "source_sentences"]
    assert not any(w in inspect.getsource(K.fact_validator) for w in ("llm", "helper(", "urllib"))


def test_an_accepted_fact_is_stored_and_traces_to_its_sentence(m):
    pipe = KnowledgePipeline(m.db)
    d, fid = pipe.process(cand(m, "I", "adopted", "beagle", "adopt", time="2 March 2023"))
    got = pipe.kb.get(fid)
    assert d.verdict == ACCEPTED and got.fact.validation_status == "accepted" and got.fact.source_type == "explicit" and got.fact.fact_type == "event" and got.fact.valid_from == "2023-03-02" and got.fact.metadata["checked_by"].startswith("rules")
    assert [s.text for s in pipe.kb.trace(fid)] == [LINES["adopt"][1]] and [x.fact.fact_id for x in pipe.kb.about("Asha")] == [fid]


def test_uncertain_waits_outside_the_knowledge_and_survives_restart(m, tmp_path):
    pipe = KnowledgePipeline(m.db)
    d, rid = pipe.process(cand(m, "Asha", "adopted", "parrot", "not"))
    assert d.verdict == UNCERTAIN and pipe.kb.about("Asha", statuses=None, source_types=None) == [] and m.db.execute("SELECT count(*) FROM knowledge_records").fetchone()[0] == 0
    m.db.commit()
    other = E.ChatMemory(str(tmp_path / "m.db"))
    item = KnowledgeBase(other.db).review(rid)
    assert json.loads(item["candidate"])["object"] == "parrot" and item["reasons"]
    other.close()


def test_rejected_saves_nothing(m):
    pipe = KnowledgePipeline(m.db)
    d, none = pipe.process(cand(m, "Asha", "adopted", "dragon", "adopt"))
    assert none is None and pipe.kb.reviews() == [] and m.db.execute("SELECT count(*) FROM knowledge_records").fetchone()[0] == 0 and m.db.execute("SELECT count(*) FROM entities").fetchone()[0] == 0


def test_the_same_fact_twice_adds_evidence_not_a_copy(m):
    pipe = KnowledgePipeline(m.db)
    _, first = pipe.process(cand(m, "Asha", "adopted", "beagle", "adopt"))
    _, second = pipe.process(cand(m, "asha", "adopted", "Beagle", "adopt2"))
    assert first == second and len(pipe.kb.trace(first)) == 2 and m.db.execute("SELECT count(*) FROM knowledge_records").fetchone()[0] == 1


def test_a_person_can_settle_a_review(m):
    pipe = KnowledgePipeline(m.db)
    known = pipe.kb.add_entity(Entity.create("Asha Rao", "person"))
    _, rid = pipe.process(cand(m, "she", "adopted", "parrot", "not"))
    with pytest.raises(RecordError, match="no entity"): pipe.approve(rid, "Nobody")
    stored = pipe.approve(rid, "Asha Rao")
    assert stored.fact.subject_id == known.entity_id and stored.fact.metadata["checked_by"] == "person" and pipe.kb.review(rid) is None
    _, rid2 = pipe.process(cand(m, "she", "adopted", "hamster", "ask"))
    pipe.decline(rid2)
    assert pipe.kb.review(rid2) is None
    with pytest.raises(RecordError): pipe.approve(rid2, "Asha Rao")


def test_a_review_cannot_override_missing_support(m):
    pipe = KnowledgePipeline(m.db)
    rid = pipe.kb.add_review(json.dumps(dict(subject="Asha", predicate="adopted", object="dragon", source_sids=[m.ids["adopt"]], time="", method="", version="", raw="")), ["x"], [m.ids["adopt"]])
    with pytest.raises(RecordError, match="object not found"): pipe.approve(rid, "Asha")


def test_deleting_a_source_removes_what_quotes_it(m):
    pipe = KnowledgePipeline(m.db)
    _, rid = pipe.process(cand(m, "Asha", "adopted", "parrot", "not"))
    _, fid = pipe.process(cand(m, "Asha", "adopted", "beagle", "adopt"))
    m.forget_sentence(m.ids["not"]); m.forget_sentence(m.ids["adopt"])
    assert pipe.kb.review(rid) is None and m.db.execute("SELECT count(*) FROM review_sources").fetchone()[0] == 0
    assert pipe.kb.get(fid).fact.status == "invalidated" and "parrot" not in json.dumps(m.db.execute("SELECT * FROM review_queue").fetchall())


def test_reading_a_helper_reply(m):
    ids = {"D1:1": m.ids["adopt"]}
    found, unread = parse_reply("Here you go:\nD1:1 | I | Adopted | beagle | unknown\nD1:1 | Asha | owns\nD1:9 | Asha | adopted | beagle\n- D1:1 | Asha | gave-away | shoes | 2 March 2023", ids, method="m", version="2")
    assert unread == ["Here you go:", "D1:1 | Asha | owns"]
    assert (found[0].subject, found[0].predicate, found[0].time, found[0].source_sids, found[0].method) == ("I", "adopted", "", (m.ids["adopt"],), "m")
    assert found[1].source_sids == (None,) and found[2].predicate == "gave_away" and found[2].time == "2 March 2023"
    assert KnowledgePipeline(m.db).process(found[1])[0].verdict == REJECTED


@pytest.mark.parametrize("text,want", [("7 May 2023", "2023-05-07"), ("May 7, 2023", "2023-05-07"), ("May 2023", "2023-05"), ("2023", "2023"), ("2023-05", "2023-05"), ("last week", None), ("", None)])
def test_dates(text, want):
    assert N.time_value(text) == want


def test_words_match_their_forms():
    assert N.stem("adopted") == N.stem("adopt") and N.stem("studies") == N.stem("study") and N.stem("houses") == N.stem("house") and N.tokens("Two dogs") == ["2", "dogs"]


def test_a_name_shared_by_two_known_entities_goes_to_review(m):
    kb = KnowledgeBase(m.db)
    kb.add_entity(Entity.create("Ben", "person")); kb.add_entity(Entity.create("Ben", "person"))
    d = verdict(m, "Ben", "adopted", "llama", "ben")
    assert d.verdict == UNCERTAIN and d.subject.status == "ambiguous" and any("names 2 known entities" in r for r in d.reasons)


def test_a_speaker_called_user_is_a_name_not_an_endless_loop(tmp_path):
    m = E.ChatMemory(str(tmp_path / "u.db")); m.open_chat(DAY)
    sid = m.add("user", "I adopted a beagle called Pixel.", date=DAY); m.close_chat(write_note=False)
    pipe = KnowledgePipeline(m.db)
    for who in ("user", "I"):
        d = pipe.validator.validate(Candidate(who, "adopted", "beagle", (sid,)), pipe.lines(Candidate(who, "adopted", "beagle", (sid,))))
        assert d.verdict == ACCEPTED and d.subject.name == "user"
    m.close()
