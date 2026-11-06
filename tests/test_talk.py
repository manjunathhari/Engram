import datetime, json, os, stat

import pytest

import engram_talk as T
from engram_memory import open_memory


class FakeModel:
    """Stands in for Gemma: records prompts, replies with a fixed line, writes a note, and writes no facts."""
    def __init__(self, fail=False):
        self.prompts, self.fail = [], fail

    def __call__(self, prompt, max_tokens=300, temperature=0.0):
        if self.fail: raise RuntimeError("server down")
        self.prompts.append(prompt)
        if "Write a short handover note" in prompt: return "Goal: none. Important facts: Pixel is a beagle.", {}, 0.01
        return "Noted.", {}, 0.01


def make(tmp_path, design="light", fail=False):
    model = FakeModel(fail)
    m = open_memory(str(tmp_path / "memory.db"), design=design, llm=model)
    return T.Talk(m, model, "Asha", str(tmp_path / "log.jsonl")), model, m


def test_memory_folder_inside_the_repository_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        T.private_folder(os.path.join(T.HERE, "results", "mine"))
    p = T.private_folder(str(tmp_path / "priv"))
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o700


def test_date_is_in_the_form_engram_reads():
    assert T.said_on(datetime.datetime(2026, 11, 7, 16, 5)) == "4:05 pm on 7 November, 2026"
    assert T.said_on(datetime.datetime(2026, 11, 7, 0, 30)) == "12:30 am on 7 November, 2026"


@pytest.mark.parametrize("design", ["light", "tested"])
def test_a_closed_chat_is_found_in_the_next_one(tmp_path, design):
    talk, model, m = make(tmp_path, design)
    day1 = datetime.datetime(2026, 11, 1, 10, 0)
    talk.say("I adopted a beagle called Pixel from the shelter yesterday.", now=day1)
    talk.say("She loves long walks in Cubbon Park.", now=day1)
    assert "closed" in talk.close()
    talk.say("What is my dog called?", now=datetime.datetime(2026, 11, 7, 9, 0))
    last = model.prompts[-1]
    assert "beagle called Pixel" in last and "Goal: none" in last  # the past sentence itself and the handover note reach the model
    assert "31 October" in last                               # "yesterday" is resolved from the day it was said
    rows = m.db.execute("SELECT speaker, text FROM sentences ORDER BY sid").fetchall()
    assert [r[0] for r in rows] == ["Asha", "Assistant", "Asha", "Assistant", "Asha", "Assistant"]
    m.close()


def test_failed_model_keeps_the_message_and_logs_the_failure(tmp_path):
    talk, _, m = make(tmp_path, fail=True)
    out = talk.say("Remember that my clinic opens at nine.")
    assert "did not answer" in out
    assert m.db.execute("SELECT count(*) FROM sentences").fetchone()[0] == 1
    log = [json.loads(l) for l in open(tmp_path / "log.jsonl")]
    assert log[-1]["kind"] == "turn" and "error" in log[-1]
    m.close()


def test_log_holds_no_text_and_is_private(tmp_path):
    talk, _, m = make(tmp_path)
    talk.say("My sister lives in Mysuru.")
    assert talk.rate(False) == "Marked."
    raw = open(tmp_path / "log.jsonl").read()
    assert "Mysuru" not in raw and "sister" not in raw
    assert [json.loads(l)["kind"] for l in raw.splitlines()] == ["turn", "rating"]
    assert stat.S_IMODE(os.stat(tmp_path / "log.jsonl").st_mode) == 0o600
    m.close()


def test_recall_calls_no_model(tmp_path):
    talk, model, m = make(tmp_path)
    talk.say("The workshop is in Pune next month.")
    talk.close()
    n = len(model.prompts)
    assert "Pune" in talk.recall("workshop Pune")
    assert len(model.prompts) == n
    m.close()


def test_failed_fact_writing_still_closes_the_chat(tmp_path):
    model = FakeModel()
    def broken_helper(prompt, max_tokens=700): raise RuntimeError("helper down")
    m = open_memory(str(tmp_path / "memory.db"), design="light", llm=model)
    m.helper, m.use_facts = broken_helper, True
    talk = T.Talk(m, model, "Asha", str(tmp_path / "log.jsonl"))
    talk.say("I moved to Bengaluru in 2019.")
    out = talk.close()
    assert "closed" in out and "failed" in out and m.chat is None
    assert m.db.execute("SELECT count(*) FROM sentences").fetchone()[0] == 2
    m.close()
