"""P2.6: the writer only proposes; the validator checks every field, including qualifiers and time, against the sentence."""
import pytest

import engram_chat as E
from engram_knowledge.extract import ExtractedFact, RuleWriter, resolve_time
from engram_knowledge.fact_validator import ACCEPTED, REJECTED, KnowledgePipeline

DAY = "4:22 pm on 13 October, 2023"


@pytest.mark.parametrize("phrase,want", [("yesterday", "2023-10-12"), ("last month", "2023-09"), ("last year", "2022"), ("two years ago", "2021"), ("3 days ago", "2023-10-10"),
                                         ("on 20 October", "2022-10-20"), ("in March", "2023-03"), ("in May 2021", "2021-05"), ("next year", "2024"),
                                         ("last week", None), ("recently", None), ("last spring", None)])
def test_time_is_worked_out_only_where_the_phrase_allows(phrase, want):
    assert resolve_time(phrase, DAY) == want


def test_no_date_when_it_was_said_is_unknown():
    assert resolve_time("yesterday", "") is None and resolve_time("in May 2021", "") == "2021-05"


@pytest.fixture
def mem(tmp_path):
    m = E.ChatMemory(str(tmp_path / "x.db")); m.open_chat(DAY)
    m.sid = m.add("Asha", "I adopted a beagle called Pixel from the shelter yesterday.", date=DAY)
    m.close_chat(write_note=False)
    yield m
    m.close()


def test_a_written_fact_passes_field_by_field_and_is_stored_with_its_details(mem):
    fact = RuleWriter().extract("Asha", "I adopted a beagle called Pixel from the shelter yesterday.", DAY, sid=mem.sid)[0]
    assert (fact.predicate, fact.object, dict(fact.qualifiers), fact.time_value) == ("adopted", "beagle", {"name": "Pixel", "from": "shelter"}, "2023-10-12")
    pipe = KnowledgePipeline(mem.db)
    d, fid = pipe.process_extracted(fact)
    stored = pipe.kb.get(fid).fact
    assert d.verdict == ACCEPTED and stored.valid_from == "2023-10-12" and stored.metadata["qualifiers"] == {"name": "Pixel", "from": "shelter"} and stored.metadata["writer"] == "rules"


@pytest.mark.parametrize("change,why", [(dict(qualifiers=(("name", "Rex"),)), "qualifier name not in"), (dict(qualifiers=(("colour", "brown"),)), "unknown qualifier"),
                                        (dict(time_text="last week", time_value=None), "time phrase not in"), (dict(time_value="2020-01-01"), "does not follow"),
                                        (dict(time_text=None, time_value="2023-10-12"), "no time phrase"), (dict(object="dragon"), "object not found")])
def test_every_field_is_checked_and_a_wrong_one_is_refused(mem, change, why):
    good = dict(subject="Asha", predicate="adopted", object="beagle", qualifiers=(("name", "Pixel"),), time_text="yesterday", time_value="2023-10-12", source_sid=mem.sid, writer="test", writer_version="1")
    d, fid = KnowledgePipeline(mem.db).process_extracted(ExtractedFact(**(good | change)))
    assert d.verdict == REJECTED and fid is None and any(why in r for r in d.reasons), d.reasons


@pytest.mark.parametrize("text", ["I never adopted a cat.", "I hate it.", "Do you like dogs?", "She adopted a dog."])
def test_the_rule_writer_gives_no_fact_rather_than_a_guess(text):
    assert RuleWriter().extract("Asha", text, DAY) == []


def test_yesterday_with_a_part_of_day_is_the_day_before():
    from engram_knowledge.extract import resolve_time
    assert resolve_time("yesterday morning", "3:56 pm on 6 June, 2023") == "2023-06-05"
    assert resolve_time("yesterday", "") is None


def test_model_reply_code_fences_and_header_are_not_facts():
    from engram_knowledge.extract import parse_writer_reply
    facts, unread = parse_writer_reply("Facts:\n```\nAsha | owns | dog | | \n```", "Asha")
    assert [f.object for f in facts] == ["dog"] and unread == []
