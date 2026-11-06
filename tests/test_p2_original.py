"""Manj's P2.1 to P2.3 tests (41), kept as written apart from imports. They must still pass after the fixes."""
import json

import pytest

from engram_knowledge.candidate import build_candidate, prepare_sentence
from engram_knowledge.interpreter import interpret
from engram_knowledge.learning import evaluate


# ---------- test_interpreter.py ----------
def test_user_preference_is_not_assistant_suggestion():
    x = interpret("user", "I prefer concise reports.")
    assert x.speaker_role == "user"
    assert x.content_kind == "preference"
    assert x.attribution == "speaker"


def test_assistant_proposal_stays_attributed_to_assistant():
    x = interpret("assistant", "I suggest using SQLite for this component.")
    assert x.speaker_role == "assistant"
    assert x.speech_act == "proposal"
    assert x.attribution == "speaker"


def test_explicit_decision():
    x = interpret("user", "We decided to use SQLite.")
    assert x.content_kind == "decision"


def test_correction_has_priority():
    x = interpret("user", "No, I meant Rust, not Python.")
    assert x.speech_act == "correction"
    assert x.polarity == "negative"


def test_question():
    x = interpret("user", "Why does this happen?")
    assert x.speech_act == "question"
    assert x.content_kind == "question"


def test_request_not_misclassified_as_question():
    x = interpret("user", "Can you explain the design?")
    assert x.speech_act == "request"


def test_hypothesis_is_uncertain():
    x = interpret("user", "This approach might be faster.")
    assert x.content_kind == "hypothesis"
    assert x.epistemic_stance == "uncertain"
    assert x.needs_review


def test_commitment_and_plan():
    x = interpret("user", "I will finish the prototype next week.")
    assert x.speech_act == "commitment"
    assert x.content_kind == "plan"
    assert x.temporal_markers == ("next week",)


def test_goal():
    x = interpret("user", "My goal is to build a portable memory engine.")
    assert x.content_kind == "goal"


def test_instruction_is_not_a_fact():
    x = interpret("user", "From now on, keep the original sentences unchanged.")
    assert x.content_kind == "instruction"


def test_assistant_fact_is_not_user_fact():
    x = interpret("assistant", "The database uses SQLite.")
    assert x.speaker_role == "assistant"
    assert x.attribution == "speaker"


def test_third_party_reference_flagged():
    x = interpret("user", "Dr Rao said the operation was completed yesterday.")
    assert x.speaker_role == "user"  # actual speaker remains the user
    assert x.attribution == "third_party_reported"  # reported claim source is distinct
    assert "yesterday" in x.temporal_markers


def test_context_dependent_reference_flagged():
    x = interpret("user", "That is what I meant.", context="previous sentence")
    assert "context_dependent_reference" in x.signals
    assert x.needs_review


def test_greeting_and_thanks_are_social():
    assert interpret("user", "Hello.").content_kind == "social"
    assert interpret("user", "Thank you.").speech_act == "thanks"


def test_empty_sentence_is_reviewable():
    x = interpret("user", "")
    assert x.speech_act == "unclear"
    assert x.needs_review


def test_original_text_is_not_mutated():
    text = "Yesterday, I decided to use Rust."
    interpret("user", text, date="2026-11-06")
    assert text == "Yesterday, I decided to use Rust."


def test_output_is_json_friendly():
    d = interpret("user", "I will start tomorrow.").to_dict()
    assert isinstance(d["temporal_markers"], list)
    assert isinstance(d["signals"], list)


# ---------- test_learning_evaluator.py ----------
def ev(speaker, text, **kw):
    return evaluate(interpret(speaker, text, **kw))


def test_user_preference_becomes_durable_candidate():
    x = ev("user", "I prefer concise reports.")
    assert (x.decision, x.retention_hint, x.proposed_validation) == ("candidate", "durable", "pending")
    assert x.proposed_source_type == "explicit"


def test_user_decision_candidate():
    x = ev("user", "We decided to use SQLite.")
    assert x.decision == "candidate" and x.content_kind == "decision"


def test_user_goal_candidate():
    assert ev("user", "My goal is to build a portable memory engine.").decision == "candidate"


def test_instruction_candidate_not_fact():
    x = ev("user", "From now on, keep the original sentences unchanged.")
    assert x.decision == "candidate" and x.content_kind == "instruction"


def test_project_plan_candidate():
    x = ev("user", "I will finish the prototype next week.")
    assert x.decision == "candidate" and x.retention_hint == "project"
    assert "time_bounded_plan" in x.reason_codes


def test_experience_is_episodic_candidate():
    x = ev("user", "I underwent surgery in 2024.")
    assert x.decision == "candidate" and x.retention_hint == "episodic"


def test_assistant_suggestion_never_becomes_user_memory():
    x = ev("assistant", "I suggest using SQLite for this component.")
    assert x.decision == "ignore" and "non_user_speaker" in x.reason_codes


def test_assistant_statement_not_user_fact():
    assert ev("assistant", "The database uses SQLite.").decision == "ignore"


def test_questions_are_not_learned():
    assert ev("user", "Why does this happen?").decision == "ignore"


def test_requests_are_not_learned():
    assert ev("user", "Can you explain the design?").decision == "ignore"


def test_social_content_is_ignored():
    assert ev("user", "Thank you.").decision == "ignore"


def test_hypothesis_is_deferred():
    x = ev("user", "This approach might be faster.")
    assert x.decision == "review" and x.proposed_validation == "uncertain"


def test_third_party_report_is_not_promoted_to_user_fact():
    x = ev("user", "Dr Rao said the operation was completed yesterday.")
    assert x.decision == "review" and "externally_attributed_statement" in x.reason_codes


def test_correction_requires_target_resolution():
    x = ev("user", "No, I meant Rust, not Python.")
    assert x.decision == "review" and x.requires_context


def test_context_dependent_reference_is_deferred():
    x = ev("user", "That is what I meant.", context="previous sentence")
    assert x.decision == "review" and x.requires_context


def test_negative_explicit_preference_can_be_candidate():
    x = ev("user", "I do not like lengthy reports.")
    assert x.decision == "candidate"
    assert "explicit_negative_statement" in x.reason_codes


def test_unknown_speaker_requires_review():
    x = ev("moderator", "We decided to use SQLite.")
    assert x.decision == "review"


def test_policy_score_is_not_probability_and_is_bounded():
    x = ev("user", "I prefer concise reports.")
    assert 0 <= x.score <= 1


def test_accepts_json_dict_from_p2_1():
    p21 = interpret("user", "I prefer concise reports.").to_dict()
    assert evaluate(p21).decision == "candidate"


def test_evaluator_does_not_mutate_interpretation_or_source():
    sentence = "I prefer concise reports."
    x = interpret("user", sentence)
    before = x.to_dict()
    evaluate(x)
    assert x.to_dict() == before
    assert sentence == "I prefer concise reports."


def test_first_person_plural_decision_is_not_misread_as_third_party_report():
    p = interpret("user", "We decided to use SQLite.")
    assert p.attribution == "speaker"
    assert evaluate(p).decision == "candidate"


# ---------- test_candidate_builder.py ----------
def source(text, speaker="user", sid=101, **extra):
    return {"sid": sid, "chat": 7, "speaker": speaker, "date": "2026-11-06",
            "ref": "", "caption": "", "text": text, **extra}


def test_prepare_sentence_builds_candidate_and_preserves_original():
    raw = "  I prefer concise reports.  "
    result = prepare_sentence(source(raw))
    assert result.status == "created"
    env = result.envelope
    assert env.lifecycle == "proposed"
    assert env.source["sid"] == 101
    assert env.original_text == raw
    assert env.source["text_sha256"]
    assert env.learning["proposed_validation"] == "pending"


def test_candidate_id_is_deterministic():
    a = prepare_sentence(source("I prefer concise reports.", sid=101)).envelope
    b = prepare_sentence(source("I prefer concise reports.", sid=101)).envelope
    assert a.candidate_id == b.candidate_id


def test_different_sentence_ids_produce_different_ids():
    a = prepare_sentence(source("I prefer concise reports.", sid=101)).envelope
    b = prepare_sentence(source("I prefer concise reports.", sid=102)).envelope
    assert a.candidate_id != b.candidate_id


def test_assistant_statement_is_excluded():
    result = prepare_sentence(source("I prefer concise reports.", speaker="assistant"))
    assert result.status == "ignored"
    assert result.envelope is None


def test_question_is_excluded():
    result = prepare_sentence(source("Can you explain SQLite?"))
    assert result.status == "ignored"
    assert result.envelope is None


def test_correction_is_packaged_for_review_not_auto_applied():
    result = prepare_sentence(source("No, I meant Rust, not Python."))
    assert result.status == "created"
    assert result.envelope.lifecycle == "needs_review"
    assert result.envelope.learning["decision"] == "review"


def test_uncertain_claim_is_packaged_for_review():
    result = prepare_sentence(source("I think this approach might work."))
    assert result.status == "created"
    assert result.envelope.lifecycle == "needs_review"


def test_speaker_mismatch_rejected():
    parsed = interpret("user", "I prefer concise reports.")
    decision = evaluate(parsed)
    with pytest.raises(ValueError, match="speaker"):
        build_candidate(source("I prefer concise reports.", speaker="assistant"),
                        parsed, decision)


def test_content_kind_mismatch_rejected():
    parsed = interpret("user", "I prefer concise reports.")
    decision = evaluate(parsed).to_dict()
    decision["content_kind"] = "goal"
    with pytest.raises(ValueError, match="content_kind"):
        build_candidate(source("I prefer concise reports."), parsed, decision)


@pytest.mark.parametrize("bad", [
    {}, {"sid": 1, "speaker": "user"}, {"sid": None, "speaker": "user", "text": "x"},
    {"sid": 1, "speaker": "", "text": "x"},
])
def test_invalid_source_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        prepare_sentence(bad)


def test_sid_zero_is_valid():
    result = prepare_sentence(source("I prefer concise reports.", sid=0))
    assert result.status == "created"
    assert result.envelope.source["sid"] == 0


def test_json_round_trip():
    env = prepare_sentence(source("I prefer concise reports.")).envelope
    decoded = json.loads(env.to_json())
    assert decoded["candidate_id"] == env.candidate_id
    assert decoded["original_text"] == env.original_text
    assert decoded["schema_version"] == 1


def test_to_dict_returns_detached_nested_data():
    env = prepare_sentence(source("I prefer concise reports.")).envelope
    first = env.to_dict()
    first["source"]["sid"] = 999
    assert env.source["sid"] == 101


def test_text_hash_changes_when_original_text_changes():
    a = prepare_sentence(source("I prefer concise reports.", sid=101)).envelope
    b = prepare_sentence(source("I prefer short reports.", sid=101)).envelope
    assert a.source["text_sha256"] != b.source["text_sha256"]


def test_date_ref_caption_are_preserved():
    env = prepare_sentence(source("I prefer concise reports.", date="2026-11-05",
                                  ref="turn-ref-2", caption="diagram")).envelope
    assert env.source["date"] == "2026-11-05"
    assert env.source["ref"] == "turn-ref-2"
    assert env.source["caption"] == "diagram"


def test_candidate_does_not_claim_p1_persistence():
    env = prepare_sentence(source("I prefer concise reports.")).envelope
    assert env.lifecycle == "proposed"
    assert env.learning["proposed_validation"] == "pending"
    assert "persisted" not in env.to_dict()


def test_unknown_decision_rejected():
    parsed = interpret("user", "I prefer concise reports.")
    decision = evaluate(parsed).to_dict()
    decision["decision"] = "accepted"
    with pytest.raises(ValueError, match="candidate, review, or ignore"):
        build_candidate(source("I prefer concise reports."), parsed, decision)
