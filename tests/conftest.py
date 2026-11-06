"""Shared helpers for the regression suite. Benchmark and made-up data only."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
sys.path.insert(0, ROOT)

import engram_chat as E  # noqa: E402


def load(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def fill(memory, data, close=True, note=False, sleep=True):
    """Puts a fixture's chats into memory. Returns {ref: sentence id}."""
    ids = {}
    chats = data["chats"]
    for n, chat in enumerate(chats):
        memory.open_chat(chat["date"])
        for speaker, text, ref, caption in chat["lines"]:
            ids[ref] = memory.add(speaker, text, date=chat["date"], ref=ref, caption=caption)
        if close or n < len(chats) - 1:
            memory.close_chat(write_note=note, sleep=sleep)
    return ids


def record(memory):
    """The permanent record exactly as stored."""
    return memory.db.execute("SELECT sid, chat, pos, ref, speaker, date, text, caption, tokens FROM sentences ORDER BY sid").fetchall()


def replying(text):
    def helper(prompt, max_tokens=0):
        return text, {}, 0.01
    return helper


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "memory.db")


@pytest.fixture
def memory(path):
    m = E.ChatMemory(path)
    yield m
    m.db.close()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: replays the benchmark; needs locomo, wordllama and spacy")
