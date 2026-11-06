"""The fixes to P2.1 to P2.3 found on real conversations, and the join to P1. Each case here was misread before the fixes."""
import json

import pytest

import engram_chat as E
from engram_knowledge import KnowledgeBase
from engram_knowledge.candidate import build_candidate
from engram_knowledge.fact_validator import KnowledgePipeline
from engram_knowledge.interpreter import interpret, interpret_message, split_message
from engram_knowledge.learning import evaluate
from engram_knowledge.learning_queue import p1_hint, prepare_stored, queue_for_review
from engram_knowledge.state_engine import StateEngine
from engram_records import RecordError

DAY = "1:00 pm on 5 March, 2023"


@pytest.mark.parametrize("text,act,kind,stance,attribution", [
    ("I went to Paris in May.", "assertion", "experience", "asserted", "speaker"),         # May the month, not "might"
    ("It may rain later.", "assertion", "hypothesis", "uncertain", "speaker"),             # may the verb still counts
    ("I said I would come.", "assertion", "claim", "asserted", "speaker"),                 # "I" is not someone else
    ("She said the clinic moved.", "assertion", "claim", "asserted", "third_party_reported"),
    ("I write poetry on weekends.", "assertion", "claim", "asserted", "speaker"),           # not a request
    ("I review papers for a journal.", "assertion", "claim", "asserted", "speaker"),
    ("Write me a short poem.", "request", "instruction_request", "asserted", "speaker"),
    ("I always walk to work.", "assertion", "claim", "asserted", "speaker"),               # a habit, not an instruction
    ("I never eat meat.", "assertion", "claim", "asserted", "speaker"),
    ("Always use metric units.", "assertion", "instruction", "asserted", "speaker"),
    ("I'm not tired but hungry.", "assertion", "claim", "asserted", "speaker"),            # not a correction
    ("No, I meant Rust.", "correction", "correction", "asserted", "speaker"),
    ("I've been painting and it's going well.", "assertion", "claim", "asserted", "speaker"),  # contractions are not quotes
    ("She told me 'never again' twice.", "assertion", "claim", "quoted", "quoted_or_external"),
    ("I did it, thanks to my coach.", "assertion", "experience", "asserted", "speaker"),   # not thanks
])
def test_sentences_once_misread(text, act, kind, stance, attribution):
    x = interpret("user", text)
    assert (x.speech_act, x.content_kind, x.epistemic_stance, x.attribution) == (act, kind, stance, attribution)


def test_named_people_are_users_of_their_own_memory():
    assert interpret("Caroline", "I adopted a dog.").speaker_role == "other"
    x = interpret("Caroline", "I adopted a dog.", people=("caroline",))
    assert x.speaker_role == "user" and evaluate(x).decision == "candidate"


def test_a_greeting_no_longer_hides_the_fact_after_it():
    text = "Thanks, Caroline! I'm adopting a dog. Isn't that great?"
    parts = interpret_message("Melanie", text, people=("Melanie",))
    assert [text[a:b] for a, b, _ in parts] == ["Thanks, Caroline!", "I'm adopting a dog.", "Isn't that great?"]
    assert [evaluate(x).decision for _, _, x in parts] == ["ignore", "candidate", "ignore"]


@pytest.mark.parametrize("text", ["Hi. I moved.\nNew line here", "  spaced  out.  ", "No full stop", "", "Wow!!! Really?! Yes."])
def test_splitting_keeps_the_exact_text(text):
    for a, b, body in split_message(text): assert text[a:b] == body and body == body.strip()
    assert "".join(text[a:b] for a, b, _ in split_message(text)).replace(" ", "") == "".join(text.split()).replace(" ", "")


def test_assistant_content_is_kept_as_said_by_the_assistant_never_as_user_memory():
    d = evaluate(interpret("assistant", "Miss Bee Providore serves a great nasi goreng."))
    assert d.decision == "ignore" and d.keep_as == "assistant_statement"
    assert evaluate(interpret("assistant", "Hello! How can I help?")).keep_as == "none"
    assert evaluate(interpret("assistant", "Would you like more ideas?")).keep_as == "none"
    source = dict(sid=4, speaker="assistant", text="Miss Bee Providore serves a great nasi goreng.")
    parsed = interpret("assistant", source["text"])
    assert build_candidate(source, parsed, evaluate(parsed)).status == "ignored"  # off unless asked for
    kept = build_candidate(source, parsed, evaluate(parsed), keep_assistant=True)
    assert kept.envelope.lifecycle == "assistant_statement" and p1_hint(kept.envelope)["about"] == "assistant"


@pytest.fixture
def mem(tmp_path):
    m = E.ChatMemory(str(tmp_path / "m.db"))
    m.open_chat(DAY)
    m.ids = dict(a=m.add("Asha", "Hey Ravi! I adopted a beagle called Pixel. Maybe I'll get a cat too.", date=DAY),
                 b=m.add("assistant", "Beagles need two walks a day.", date=DAY))
    m.close_chat(write_note=False)
    yield m
    m.close()


def test_a_stored_message_gives_one_envelope_per_sentence_with_its_place(mem):
    out = prepare_stored(mem.db, mem.ids["a"], people=("Asha",))
    stored = mem.db.execute("SELECT text FROM sentences WHERE sid=?", (mem.ids["a"],)).fetchone()[0]
    assert [r.status for r in out] == ["ignored", "created", "created"]
    learn, review = out[1].envelope, out[2].envelope
    assert learn.lifecycle == "proposed" and review.lifecycle == "needs_review"
    a, b = learn.source["span"]
    assert stored[a:b] == learn.original_text == "I adopted a beagle called Pixel." and learn.source["sid"] == mem.ids["a"]
    assert learn.candidate_id != review.candidate_id and p1_hint(learn) == dict(fact_type="observation", source_type="explicit", about="speaker", goes_to="extraction")
    assert p1_hint(review)["goes_to"] == "review_queue"
    assistant = prepare_stored(mem.db, mem.ids["b"])
    assert assistant[0].envelope.lifecycle == "assistant_statement"
    with pytest.raises(ValueError): prepare_stored(mem.db, 99999)


def test_review_envelopes_share_the_one_review_queue_and_go_when_the_sentence_goes(mem):
    kb = KnowledgeBase(mem.db)
    review = prepare_stored(mem.db, mem.ids["a"], people=("Asha",))[2].envelope
    rid = queue_for_review(kb, review)
    item = json.loads(kb.review(rid)["candidate"])
    assert item["kind"] == "sentence_envelope" and item["envelope"]["original_text"] == "Maybe I'll get a cat too."
    with pytest.raises(ValueError): queue_for_review(kb, prepare_stored(mem.db, mem.ids["a"], people=("Asha",))[1].envelope)
    assert StateEngine(kb).get_current_state("ent_none") == {}  # the state engine reads past it
    with pytest.raises(RecordError, match="needs who, what and which"): KnowledgePipeline(mem.db).approve(rid, "Asha")
    mem.forget_sentence(mem.ids["a"])
    assert kb.review(rid) is None and kb.integrity_check() == []


def test_nothing_here_writes_a_fact(mem):
    kb = KnowledgeBase(mem.db)
    for r in prepare_stored(mem.db, mem.ids["a"], people=("Asha",)):
        if r.envelope and r.envelope.lifecycle == "needs_review": queue_for_review(kb, r.envelope)
    assert mem.db.execute("SELECT count(*) FROM knowledge_records").fetchone()[0] == 0 and mem.db.execute("SELECT count(*) FROM entities").fetchone()[0] == 0
