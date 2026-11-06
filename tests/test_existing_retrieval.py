"""Retrieval stays as it is. Fast checks on made-up chats, and slow checks against recorded benchmark results."""
import hashlib
import json
import os
import re

import numpy as np
import pytest

from conftest import FIXTURES, ROOT, fill, load

DATA = load("basic_conversation.json")
BASELINE = os.path.join(FIXTURES, "retrieval_baseline.json")


def refs(memory, sids):
    return [memory.db.execute("SELECT ref FROM sentences WHERE sid=?", (s,)).fetchone()[0] for s in sids]


@pytest.mark.parametrize("item", DATA["questions"], ids=lambda x: x["q"][:30])
def test_finds_the_sentence_for_a_past_question(memory, item):
    fill(memory, DATA)
    assert item["expect"] in refs(memory, memory.find(item["q"])[0])


def test_search_gives_the_same_answer_twice(memory):
    fill(memory, DATA)
    q = DATA["questions"][0]["q"]
    assert memory.find(q)[0] == memory.find(q)[0]


def test_no_sentence_returned_twice_and_budget_respected(memory):
    fill(memory, DATA)
    for item in DATA["questions"]:
        sids = memory.find(item["q"])[0]
        assert len(sids) == len(set(sids))
        assert sum(memory._tokens(s) + 8 for s in sids) <= memory.s.retrieved_tokens + 100


def test_empty_memory_finds_nothing(memory):
    assert memory.find("anything at all")[0] == []


def test_open_chat_is_not_returned_as_past_memory(memory):
    ids = fill(memory, DATA, close=False)
    assert ids["B3:1"] not in memory.find("Pixel walk by the lake")[0]
    assert "walk by the lake" in memory.context_for("lake")[0]  # still on the desk


def test_retrieval_code_unchanged():
    """A change to these files must be deliberate: rerun the baseline and update this record in the same commit."""
    recorded = json.load(open(BASELINE))["code"]
    for name, digest in recorded.items():
        with open(os.path.join(ROOT, name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, f"{name} changed"


NORM = lambda s: re.sub(r"\W+", " ", s.lower()).strip()


def test_saved_final_test_scores_reproduce():
    """Recomputes the published scores from the saved answers and blind labels."""
    labels = json.load(open(os.path.join(ROOT, "results", "final_labels.json")))
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "final_answers.jsonl"))]
    expected = {"with_facts": (62.4, 79.9), "no_facts": (58.6, 75.4), "current": (53.9, 67.5)}
    for system, (strict, generous) in expected.items():
        chosen = [labels["|".join((r["q"], r["gold"], NORM(r["pred"])))] for r in rows if r["system"] == system]
        assert len(chosen) == 776
        assert round(np.mean([x["strict"] for x in chosen]) * 100, 1) == strict
        assert round(np.mean([x["generous"] for x in chosen]) * 100, 1) == generous


@pytest.mark.slow
def test_benchmark_evidence_matches_the_baseline():
    """Replays conversations 0-4 through real chat memory. Needs the benchmark data, wordllama and spacy."""
    if not os.path.exists(os.path.join(ROOT, "locomo", "data", "locomo10.json")): pytest.skip("benchmark data not present")
    pytest.importorskip("wordllama"), pytest.importorskip("spacy")
    from record_baseline import measure
    base = json.load(open(BASELINE))
    now = measure()
    assert now["questions"] == base["questions"]
    for system, tolerance in (("current", 0), ("plain", 0.05), ("cabinet", 0.05)):
        assert abs(now[system]["evidence"] - base[system]["evidence"]) <= tolerance, system
    assert now["current"]["per_question"] == base["current"]["per_question"]
    assert np.mean(np.abs(np.array(now["plain"]["per_question"]) - np.array(base["plain"]["per_question"]))) < 0.002
