"""Facts are optional, must have sources, and must never harm the record."""
import json
import os
import subprocess
import sys

import pytest

import engram_chat as E
import engram_facts as F
from conftest import FIXTURES, ROOT, fill, load, record, replying

DATA = load("basic_conversation.json")


def with_helper(path, reply=None):
    return E.ChatMemory(path, helper=replying(reply or DATA["helper_reply"]), helper_name="stand-in")


def test_existing_fact_checks_pass():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "kit_tools", "test_facts.py")], capture_output=True, text=True, env=dict(os.environ, ENGRAM=ROOT))
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]


def test_facts_are_off_by_default(memory):
    fill(memory, DATA)
    assert memory.db.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert "Noted fact" not in memory.context_for(DATA["questions"][0]["q"])[0]


def test_stored_facts_are_not_shown_unless_switched_on(path):
    m = with_helper(path)
    fill(m, DATA)
    assert m.db.execute("SELECT count(*) FROM facts").fetchone()[0] > 0
    assert "Noted fact" not in m.context_for("What is the name of Asha's beagle?")[0]
    m.close()


def test_supported_facts_accepted_and_unsupported_rejected(path):
    m = with_helper(path)
    m.open_chat(DATA["chats"][0]["date"])
    for s, t, ref, c in DATA["chats"][0]["lines"]:
        m.add(s, t, date=DATA["chats"][0]["date"], ref=ref, caption=c)
    out = m.close_chat(write_note=False)["facts"]
    assert (out["accepted"], out["rejected"]) == (DATA["accepted"], DATA["rejected"])
    texts = " ".join(r[0] for r in m.db.execute("SELECT text FROM facts"))
    assert "beagle" in texts and "spaceship" not in texts
    m.close()


def test_every_accepted_fact_links_to_a_sentence_in_its_own_chat(path):
    m = with_helper(path)
    fill(m, DATA)
    facts = m.db.execute("SELECT fact, chat FROM facts").fetchall()
    assert facts
    for fact, chat in facts:
        sources = m.db.execute("SELECT s.chat FROM fact_sources x JOIN sentences s ON s.sid=x.sid WHERE x.fact=?", (fact,)).fetchall()
        assert sources and all(r[0] == chat for r in sources)
    assert m.db.execute("SELECT count(*) FROM fact_sources WHERE sid NOT IN (SELECT sid FROM sentences)").fetchone()[0] == 0
    m.close()


def test_a_wrong_link_is_moved_to_the_supporting_sentence():
    sentences = {"D1:1": "Asha: I adopted a beagle called Pixel.", "D1:2": "Ravi: Thanks, talk soon."}
    _, facts, _ = F.parse("D1:2 | Asha adopted a beagle called Pixel.", sentences, E.words)
    assert facts[0]["links"] == ["D1:1"] and facts[0]["moved"]


def test_a_source_from_another_chat_is_not_accepted():
    sentences = {"D2:1": "Asha: I adopted a beagle called Pixel."}
    _, facts, _ = F.parse("D9:9 | Asha adopted a beagle called Pixel.", sentences, E.words)
    assert facts[0]["links"] == ["D2:1"]


def test_helper_does_not_change_the_record(path):
    plain, helped = E.ChatMemory(path + "a"), with_helper(path + "b")
    fill(plain, DATA), fill(helped, DATA)
    assert record(plain) == record(helped)
    plain.close(), helped.close()


def test_deleting_a_sentence_removes_dependent_facts(path):
    m = with_helper(path)
    ids = fill(m, DATA)
    assert m.find_facts("beagle")
    out = m.forget_sentence(ids["B1:1"])
    assert out["sentences"] == 1 and out["facts"] >= 1
    assert not m.db.execute("SELECT 1 FROM facts WHERE text LIKE '%beagle%'").fetchone()
    assert not m.db.execute("SELECT 1 FROM fact_sources WHERE sid=?", (ids["B1:1"],)).fetchone()
    assert not m.db.execute("SELECT 1 FROM sentences WHERE sid=?", (ids["B1:1"],)).fetchone()
    assert ids["B1:1"] not in m.find("beagle Pixel shelter")[0]
    assert "called Pixel" not in m.context_for("beagle")[0]
    m.close()
    again = E.ChatMemory(path)
    assert not again.db.execute("SELECT 1 FROM sentences WHERE sid=?", (ids["B1:1"],)).fetchone()
    again.close()


def test_helper_failure_leaves_original_memory_intact(path):
    def broken(prompt, max_tokens=0):
        raise RuntimeError("model down")
    m = E.ChatMemory(path, helper=broken, helper_name="stand-in")
    fill(m, DATA)
    plain = E.ChatMemory(path + "plain")
    fill(plain, DATA)
    assert record(m) == record(plain)
    assert m.db.execute("SELECT count(*) FROM chats WHERE closed=1").fetchone()[0] == 3
    assert m.db.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert {r[0] for r in m.db.execute("SELECT status FROM episodes")} == {"failed"}
    q = DATA["questions"][0]["q"]
    assert m.find(q)[0] == plain.find(q)[0]
    m.helper = replying(DATA["helper_reply"])
    assert m.write_facts(1)["status"] == "done"
    m.close(), plain.close()


def test_rerunning_the_helper_does_not_duplicate(path):
    m = with_helper(path)
    fill(m, DATA)
    before = m.db.execute("SELECT count(*) FROM facts").fetchone()[0]
    m.write_facts(1), m.write_facts(1)
    assert m.db.execute("SELECT count(*) FROM facts").fetchone()[0] == before
    assert m.db.execute("SELECT count(*) FROM facts_index").fetchone()[0] == before
    m.close()


def test_facts_add_to_the_exact_sentences_and_do_not_replace_them(path):
    m = with_helper(path)
    fill(m, DATA)
    m.use_facts = True
    ctx = m.context_for("What is the name of Asha's beagle?")[0]
    assert "[Noted fact]" in ctx and "beagle called Pixel from the shelter" in ctx and "Asha:" in ctx
    m.close()


@pytest.mark.parametrize("name,facts", [("facts_dev_gemma3-4b.json", 1499), ("facts_final.json", 1712)])
@pytest.mark.slow
def test_real_gemma_replies_still_parse_to_the_same_facts(name, facts):
    locomo = os.path.join(ROOT, "locomo", "data", "locomo10.json")
    if not os.path.exists(locomo): pytest.skip("benchmark data not present")
    conv = json.load(open(locomo))
    saved = json.load(open(os.path.join(ROOT, "results", name)))
    total = 0
    for key, v in saved.items():
        c, s = key.split(":")
        chat = conv[int(c)]["conversation"]
        date = chat[f"session_{s}_date_time"]
        sentences = {t["dia_id"]: f"{t['speaker']}: {E.with_dates(t['text'], date)}" + (f" [image: {t['blip_caption']}]" if t.get("blip_caption") else "") for t in chat[f"session_{s}"]}
        total += len(F.parse(v["raw"], sentences, E.words)[1])
    assert total == facts
