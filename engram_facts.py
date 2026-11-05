"""Facts and episode context for ChatMemory, written by a helper model when a chat closes.

The helper only adds. The exact sentences stay the record, and if a fact and a sentence disagree the sentence wins.
Ordinary code checks every fact against the sentences of its chat: a fact is linked to a sentence that supports it,
and a fact that no sentence supports is rejected. Everything here is optional and off unless a helper is given.
"""
import re

from engram import CAP, STOP

SCHEMA = """
    CREATE TABLE IF NOT EXISTS episodes(chat INTEGER PRIMARY KEY, context TEXT, status TEXT, model TEXT, seconds REAL, accepted INTEGER, moved INTEGER, rejected INTEGER, raw TEXT);
    CREATE TABLE IF NOT EXISTS facts(fact INTEGER PRIMARY KEY, chat INTEGER, text TEXT, person TEXT, place TEXT, time TEXT, number TEXT, checked TEXT, model TEXT);
    CREATE TABLE IF NOT EXISTS fact_sources(fact INTEGER, sid INTEGER, PRIMARY KEY(fact, sid));
    CREATE INDEX IF NOT EXISTS fact_sources_sid ON fact_sources(sid);
    CREATE INDEX IF NOT EXISTS facts_chat ON facts(chat);
    CREATE VIRTUAL TABLE IF NOT EXISTS facts_index USING fts5(words, tokenize="unicode61 remove_diacritics 0");
"""

PROMPT = """Below is one conversation. Each line starts with its id. Write the facts worth remembering later.
Write one line per fact, at most 12, in exactly this form:
ID | fact
The ID must be copied from the line the fact came from. If a fact uses several lines, separate their ids with commas.
Choose the facts from the whole conversation, not only its start. Each fact must make sense alone: use full names instead of he, she or they, and give real dates from the square brackets. Check who actually did each thing. Skip greetings, thanks, compliments and small talk. Use only what is written. Write nothing else.

Conversation:
{chat}

Facts:
"""

LINE_ID = re.compile(r"\b(?:D\d+:\d+|S\d+)\b")
TIME = re.compile(r"\[[^\]]*\b\d{4}\b[^\]]*\]|\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December)(?: \d{4})?\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b")
NUMBER = re.compile(r"(?<![\w:\-])\d[\d,\.]*%?(?![\w:\-])")
MIN_SUPPORT = .2


def _words(text, words):
    return set(words(text))


def _support(fact_words, sentence_words):
    return len(fact_words & sentence_words) / max(1, len(fact_words))


def parse(reply, sentences, words):
    """sentences: {line id: 'speaker: text'} for this chat only. Returns context, accepted facts and the rejected count."""
    context, out, seen, rejected = "", [], set(), 0
    tokens = {k: _words(v, words) for k, v in sentences.items()}
    names = {w for v in sentences.values() for w in words(v.split(":")[0])}
    prefix = re.match(r"(D\d+:|S)", next(iter(sentences), "") or "")
    prefix = prefix.group(1) if prefix else ""
    for line in reply.splitlines():
        line = line.strip().strip("*-• ")
        m = re.match(r"context\s*:\s*(.+)", line, re.I)
        if m:
            context = m.group(1).strip()[:400]
            continue
        cited = [x for x in LINE_ID.findall(line) if x in sentences]
        text = LINE_ID.sub(" ", line)
        stamp = re.match(r"^[\s,]*\d{1,2}:\d{2}(\s*[ap]m)?( on [^|]*)?\|", text, flags=re.I)  # some replies start with the time, not an id
        text = text[stamp.end():] if stamp else text
        if "|" in text:
            left, right = text.split("|", 1)
            if not cited and prefix and re.fullmatch(r"[\s\d,]*", left):
                cited = [prefix + x for x in re.findall(r"\d+", left) if prefix + x in sentences]
            text = right if len(right.strip()) > 8 else left
        elif not cited and not stamp:
            continue
        text = re.sub(r"^[\s,:\-|]+", "", text).strip()
        if len(text) <= 8 or text.lower() in seen:
            continue
        fact_words = _words(text, words) - names
        if not fact_words or not tokens:
            rejected += 1
            continue
        best = max(tokens, key=lambda k: _support(fact_words, tokens[k]))
        if _support(fact_words, tokens[best]) < MIN_SUPPORT:
            rejected += 1
            continue
        links = [t for t in cited if _support(fact_words, tokens[t]) >= MIN_SUPPORT] or [best]
        seen.add(text.lower())
        out.append(dict(text=text[:300], links=list(dict.fromkeys(links)), moved=set(links) != set(cited)))
    return context, out[:12], rejected


def fields(text, tools=None):
    """Person, place, time and figures, read from a fact by ordinary code."""
    time = "; ".join(dict.fromkeys(TIME.findall(text)))
    clean = TIME.sub(" ", text)
    number = "; ".join(dict.fromkeys(NUMBER.findall(clean)))
    person = ", ".join(dict.fromkeys(w for w in CAP.findall(text) if w.lower() not in STOP))
    place = ""
    if tools:
        place = ", ".join(dict.fromkeys(e.text for e in tools(text).ents if e.label_ in ("GPE", "LOC", "FAC")))
    return person, place, time, number
