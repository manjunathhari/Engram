"""Conversation ingestion and the live desk."""
import engram_chat as E
from conftest import fill, load, record


def lines(data):
    return [(a, b, c, d) for chat in data["chats"] for a, b, c, d in chat["lines"]]


def test_all_original_sentences_preserved_exactly(memory):
    data = load("basic_conversation.json")
    fill(memory, data)
    stored = [(r[4], r[6], r[3], r[7]) for r in record(memory)]
    assert stored == lines(data)


def test_order_chat_and_date_kept(memory):
    data = load("basic_conversation.json")
    fill(memory, data)
    rows = record(memory)
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)
    expected = [(n + 1, i, chat["date"]) for n, chat in enumerate(data["chats"]) for i in range(len(chat["lines"]))]
    assert [(r[1], r[2], r[5]) for r in rows] == expected


def test_saved_before_any_close(memory):
    data = load("basic_conversation.json")
    memory.open_chat(data["chats"][0]["date"])
    memory.add("Asha", "Saved right away.", date=data["chats"][0]["date"])
    assert memory.db.execute("SELECT count(*) FROM sentences").fetchone()[0] == 1


def test_changing_and_contradictory_versions_all_kept(memory):
    for name in ("changing_facts.json", "contradictory_facts.json"):
        data = load(name)
        fill(memory, data)
        stored = [r[6] for r in record(memory)]
        for sentence in data["keep"]:
            assert sentence in stored


def test_date_resolving_does_not_change_the_record(memory):
    data = load("basic_conversation.json")
    fill(memory, data)
    original = "I adopted a beagle called Pixel from the shelter yesterday."
    assert original in [r[6] for r in record(memory)]
    shown = E.with_dates(original, data["chats"][0]["date"])
    assert "2 March 2023" in shown and original.split(" yesterday")[0] in shown


def test_a_long_chat_keeps_everything_while_the_desk_stays_within_limit(path):
    m = E.ChatMemory(path, settings=E.Settings(desk_tokens=60, push_tokens=20))
    m.open_chat("1 May 2023")
    for n in range(60):
        m.add("Asha", f"Line number {n} about topic{n}.", date="1 May 2023")
    assert m.db.execute("SELECT count(*) FROM sentences").fetchone()[0] == 60
    assert m.desk_used <= 60
    assert "topic3" in m.context_for("What about topic3?")[0]  # off the desk, still found by word search
    m.db.close()


def test_new_questions_do_not_touch_the_record(memory):
    data = load("basic_conversation.json")
    fill(memory, data)
    before = record(memory)
    for q in data["questions"]:
        memory.context_for(q["q"])
    assert record(memory) == before
