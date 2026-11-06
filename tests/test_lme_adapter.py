"""The adapter that feeds LongMemEval to Engram. Made-up sessions in the published format."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kit_tools"))
import lme_test as L


def no_spaces(text):
    return re.sub(r"\s+", "", text)


@pytest.mark.parametrize("text", [
    "I adopted a dog. He is called Rex! Is that fine? Yes.",
    "Here is a list:\n\n1. **Walk** daily.\n2. Eat well.\n\nDone",
    "word " * 400,
    "x" * 1500,
    "No full stop at the end",
])
def test_splitting_keeps_every_word(text):
    pieces = L.split_sentences(text)
    assert no_spaces("".join(pieces)) == no_spaces(text)
    assert all(0 < len(p) <= L.LONGEST_SENTENCE for p in pieces)


def test_splitting_makes_sentences():
    assert L.split_sentences("I adopted a dog. He is Rex!\nGood.") == ["I adopted a dog.", "He is Rex!", "Good."]
    assert L.split_sentences("") == [] and L.split_sentences(None) == []


@pytest.mark.parametrize("given,want", [
    ("2023/05/20 (Sat) 02:21", "2:21 am on 20 May, 2023"),
    ("2023/05/30 (Tue) 23:40", "11:40 pm on 30 May, 2023"),
    ("2023/01/01 (Sun) 00:05", "12:05 am on 1 January, 2023"),
    ("2023/06/15 (Thu) 12:00", "12:00 pm on 15 June, 2023"),
])
def test_dates_become_what_engram_reads(given, want):
    import engram
    assert L.session_date(given) == want
    assert engram.parse_session_date(want) is not None


def test_development_and_test_questions_never_overlap_and_are_stable():
    data = [dict(question_id=f"q{n}") for n in range(100)]
    dev, test = L.split_questions(data)
    assert len(dev) == L.DEV_QUESTIONS and len(test) == 70
    assert not {q["question_id"] for q in dev} & {q["question_id"] for q in test}
    assert L.split_questions(list(reversed(data)))[0] == dev


SESSION = [dict(role="user", content="I bought a Fitbit. It helps."), dict(role="assistant", content="Well done.\nKeep walking.", has_answer=False), dict(role="user", content="Thanks.", has_answer=True)]


def test_session_lines_keep_turn_and_answer_marks():
    lines = L.session_lines(SESSION, 7)
    assert [x[0] for x in lines] == ["D7:1", "D7:2", "D7:3", "D7:4", "D7:5"]
    assert [x[1] for x in lines] == ["User", "User", "Assistant", "Assistant", "User"]
    assert [x[3] for x in lines] == [0, 0, 1, 1, 2] and [x[4] for x in lines] == [False, False, False, False, True]


def test_a_session_has_one_number_in_every_question():
    a = dict(haystack_session_ids=["s1", "s2"], haystack_sessions=[SESSION, SESSION[:1]])
    b = dict(haystack_session_ids=["s2", "s3"], haystack_sessions=[SESSION[:1], SESSION])
    numbers = L.session_numbers([a, b])
    assert len(numbers) == 3 and L.session_key("s2", SESSION[:1]) in numbers


def test_line_ids_match_what_the_facts_parser_reads():
    import engram_facts as F
    assert F.LINE_ID.findall(L.session_lines(SESSION, 7)[0][0]) == ["D7:1"]


def prompt_with(lines):
    import engram_facts as F
    return F.PROMPT.format(chat="[1:00 pm on 1 May, 2023]\n" + "\n".join(lines))


def test_short_prompt_is_unchanged():
    p = prompt_with(["D1:1 User: Hello there."])
    assert L.fit_prompt(p) == (p, 0)


def test_long_prompt_fits_and_keeps_the_instructions_and_the_user_first():
    lines = []
    for n in range(300):
        lines += [f"D1:{2 * n + 1} Assistant: " + "advice " * 60, f"D1:{2 * n + 2} User: my fact number {n}"]
    fitted, dropped = L.fit_prompt(prompt_with(lines))
    assert len(fitted) // 4 <= L.FACT_PROMPT_TOKENS and dropped > 0
    assert fitted.startswith("Below is one conversation") and "at most 12" in fitted and fitted.rstrip().endswith("Facts:") and "[1:00 pm on 1 May, 2023]" in fitted
    assert all(f"my fact number {n}\n" in fitted or f"my fact number {n}\n\nFacts" in fitted for n in range(300))
    order = [int(x) for x in re.findall(r"D1:(\d+) ", fitted)]
    assert order == sorted(order)
    assert L.fit_prompt(fitted) == (fitted, 0)
