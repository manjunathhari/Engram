"""Restart must not lose or duplicate anything."""
import json
import os
import signal
import subprocess
import sys
import time

import engram_chat as E
from conftest import ROOT, fill, load, record, replying


def counts(m):
    return {t: m.db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("chats", "sentences", "words_index", "facts", "fact_sources", "facts_index", "episodes")}


def test_restart_recovers_all_memories(path):
    data = load("basic_conversation.json")
    m = E.ChatMemory(path)
    fill(m, data)
    before, found = record(m), {q["q"]: m.find(q["q"])[0] for q in data["questions"]}
    m.close()
    m = E.ChatMemory(path)
    assert record(m) == before
    assert {q["q"]: m.find(q["q"])[0] for q in data["questions"]} == found
    m.close()


def test_repeated_restarts_do_not_duplicate(path):
    data = load("basic_conversation.json")
    m = E.ChatMemory(path, helper=replying(data["helper_reply"]), helper_name="stand-in")
    fill(m, data)
    before = counts(m)
    m.close()
    for _ in range(4):
        m = E.ChatMemory(path)
        assert counts(m) == before
        m.close()


def test_open_chat_continues_after_restart(path):
    data = load("basic_conversation.json")
    m = E.ChatMemory(path)
    fill(m, data, close=False)
    chat, desk = m.chat, [s for s, _ in m.desk]
    m.close()
    m = E.ChatMemory(path)
    assert m.chat == chat and [s for s, _ in m.desk] == desk
    assert "the recital in August" in m.context_for("recital")[0]
    sid = m.add("Asha", "One more line after the restart.", date=data["chats"][-1]["date"])
    assert m.db.execute("SELECT chat, pos FROM sentences WHERE sid=?", (sid,)).fetchone() == (chat, 2)
    m.close()


def test_closed_chat_stays_closed_and_searchable(path):
    data = load("basic_conversation.json")
    m = E.ChatMemory(path)
    fill(m, data)
    closed = m.db.execute("SELECT chat, closed FROM chats ORDER BY chat").fetchall()
    m.close()
    m = E.ChatMemory(path)
    assert m.db.execute("SELECT chat, closed FROM chats ORDER BY chat").fetchall() == closed and m.chat is None
    assert m.find("What is the name of Asha's beagle?")[0]
    m.close()


def test_hard_crash_loses_nothing_that_was_saved(path):
    """Kills a running process without warning, as a power cut would, then reopens the file."""
    code = f"""
import sys
sys.path.insert(0, {ROOT!r})
import engram_chat as E
m = E.ChatMemory({path!r})
m.open_chat("1 May 2023")
n = 0
while True:
    m.add("Asha", "Crash test line %d." % n, date="1 May 2023")
    print(n, flush=True)
    n += 1
"""
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    last = -1
    while last < 20:
        last = int(p.stdout.readline())
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    m = E.ChatMemory(path)
    texts = [r[0] for r in m.db.execute("SELECT text FROM sentences ORDER BY sid")]
    assert texts[:last + 1] == [f"Crash test line {n}." for n in range(last + 1)]  # all it reported saving
    assert len(texts) == len(set(texts))
    assert m.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    m.close()


def test_older_database_upgrades_without_changing_sentences(path):
    import sqlite3
    old = sqlite3.connect(path)
    old.executescript("""CREATE TABLE chats(chat INTEGER PRIMARY KEY, opened_on TEXT, last_active REAL, closed INTEGER DEFAULT 0, note TEXT);
CREATE TABLE sentences(sid INTEGER PRIMARY KEY, chat INTEGER, pos INTEGER, ref TEXT, speaker TEXT, date TEXT, text TEXT, caption TEXT, tokens INTEGER);
CREATE VIRTUAL TABLE words_index USING fts5(words, tokenize="unicode61 remove_diacritics 0");
INSERT INTO chats VALUES(1,'1 May 2023',0,1,NULL);
INSERT INTO sentences VALUES(1,1,0,'','A','1 May 2023','My dog is called Rex.','',6);
INSERT INTO words_index(rowid,words) VALUES(1,'dog called rex');""")
    old.commit(); old.close()
    m = E.ChatMemory(path)
    assert m.db.execute("SELECT text FROM sentences").fetchall() == [("My dog is called Rex.",)]
    assert m.find("dog")[0]
    m.close()
