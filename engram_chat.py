"""Engram chat memory: chats open and close, and closing moves memory to long-term storage.

During a chat: every sentence is saved to disk on arrival, the desk holds the recent conversation,
and a simple word search covers anything that has overflowed the desk.
When a chat closes: the chat joins Engram's long-term index, and a model may write a short note
that carries its state to the next chat.
Later chats: Engram's session shortlist finds the right old chats, then the right sentences.

The model only ever receives plain text, so any open model can sit on top.
"""
import datetime as dt
import math
import os
import re
import sqlite3
import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

import engram
import engram_facts as F
import engram_records as R
from engram import Config, Engrams

STOP, TOKEN = engram.STOP, engram.TOKEN


def stem(w: str) -> str:
    """Trims word endings, so pet and pets, or organize and organized, match."""
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    else:
        for suf in ("ingly", "edly", "ing", "ed"):
            if len(w) - len(suf) >= 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        else:
            if len(w) >= 5 and re.search(r"(ss|x|z|ch|sh)es$", w): w = w[:-2]
            elif len(w) >= 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")): w = w[:-1]
            elif len(w) >= 6 and w.endswith("ly"): w = w[:-2]
    if len(w) >= 4 and w[-1] == "e": w = w[:-1]
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiou": w = w[:-1]
    return w


def words(text: str) -> list[str]:
    out = []
    for t in TOKEN.findall(text.lower().replace("'s", "")):
        t = t.strip("'")
        if len(t) > 1 and t not in STOP: out.append(stem(t))
    return out


engram.words, engram.stem = words, stem  # Engram's read path uses the same word trimming


MONTHS = "January February March April May June July August September October November December".split()
WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple of": 2, "a couple of": 2}
_N = r"(\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|(?:a )?couple of)"
RELATIVE = re.compile(rf"\b(the day before yesterday|day before yesterday|yesterday|last night|today|tonight|this morning|tomorrow"
                      rf"|(?:last|next) (?:week|weekend|month|year|summer|winter|spring|fall|autumn|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
                      rf"|this (?:week|weekend|month|year)"
                      rf"|{_N} (?:days?|weeks?|months?|years?) ago|(?:a )?few (?:days|weeks|months|years) ago"
                      rf"|in {_N} (?:days?|weeks?|months?|years?))\b", re.I)


def _d(x) -> str:
    return f"{x.day} {MONTHS[x.month - 1]} {x.year}"


def _shift_month(x, n):
    m = x.year * 12 + x.month - 1 + n
    return m // 12, m % 12 + 1


def _span(a, b) -> str:
    if a.month == b.month and a.year == b.year: return f"{a.day} to {b.day} {MONTHS[a.month - 1]} {a.year}"
    return f"{_d(a)} to {_d(b)}"


def resolve_phrase(phrase: str, day) -> str | None:
    """The real date behind words like yesterday or last Friday, given the day they were said."""
    p = phrase.lower(); one = dt.timedelta(days=1)
    if p in ("today", "tonight", "this morning"): return _d(day)
    if p == "yesterday" or p == "last night": return _d(day - one)
    if "day before yesterday" in p: return _d(day - 2 * one)
    if p == "tomorrow": return _d(day + one)
    m = re.fullmatch(r"(last|next|this) (\w+)", p)
    if m:
        which, unit = m.groups()
        step = {"last": -1, "next": 1, "this": 0}[which]
        if unit in WEEKDAYS:
            target = WEEKDAYS.index(unit)
            if step < 0: return _d(day - ((day.weekday() - target - 1) % 7 + 1) * one)
            return _d(day + ((target - day.weekday() - 1) % 7 + 1) * one)
        if unit == "week":
            start = day - day.weekday() * one + 7 * step * one
            return "week of " + _span(start, start + 6 * one)
        if unit == "weekend":
            sat = day + (5 - day.weekday()) * one  # Saturday of this calendar week
            if step < 0:
                sat = sat - 7 * one
            elif step > 0 and day.weekday() >= 5:
                sat = sat + 7 * one
            return "weekend of " + _span(sat, sat + one)
        if unit == "month":
            y, mo = _shift_month(day, step)
            return f"{MONTHS[mo - 1]} {y}"
        if unit == "year": return str(day.year + step)
        season = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}[unit]
        if step < 0: y = day.year if day.month > season + 2 else day.year - 1
        else: y = day.year if day.month < season else day.year + 1
        return f"{unit} {y}"
    m = re.fullmatch(rf"(?:a )?few (days|weeks|months|years) ago", p)
    if m: return f"a few {m.group(1)} before {_d(day)}"
    m = re.fullmatch(rf"(in )?{_N} (days?|weeks?|months?|years?)( ago)?", p)
    if m:
        n = int(m.group(2)) if m.group(2).isdigit() else WORD_NUM[m.group(2)]
        sign = 1 if m.group(1) else -1
        u = m.group(3)
        if u.startswith("day"): return _d(day + sign * n * one)
        if u.startswith("week"): return "around " + _d(day + sign * 7 * n * one)
        if u.startswith("month"):
            y, mo = _shift_month(day, sign * n)
            return f"{MONTHS[mo - 1]} {y}"
        return str(day.year + sign * n)
    return None


def with_dates(text: str, date_said: str) -> str:
    """Shows the real date beside words like yesterday. The saved sentence itself never changes."""
    day = engram.parse_session_date(date_said) if date_said else None
    if not day: return text
    if hasattr(day, "date"): day = day.date()
    def add(m):
        r = resolve_phrase(m.group(0), day)
        return f"{m.group(0)} [{r}]" if r else m.group(0)
    return RELATIVE.sub(add, text)


NOTE_PROMPT = """Below is a finished conversation. Write a short handover note for whoever continues it later.
Use these headings: Goal, Decisions, Open tasks, Important facts. Under Important facts, keep names, dates, numbers, places and feelings exactly as written.
Use only what is written. Write under 150 words.
{previous}
Conversation:
{chat}

Note:"""


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Settings:
    desk_tokens: int = 2500      # live conversation on the desk
    push_tokens: int = 500       # moved off the desk together when it is full
    retrieved_tokens: int = 800  # memories that may return to the desk for one question
    note_read_tokens: int = 3200 # most of a chat the model reads when writing its note
    note_tokens: int = 300       # longest note
    fact_tokens: int = 300       # noted facts shown on top of retrieved memories, when facts are on
    fact_reply_tokens: int = 700 # longest reply the helper may write for one chat


class ChatMemory:
    def __init__(self, path: str, cfg: Config | None = None, settings: Settings | None = None, llm=None, durable: bool = True, search=None, helper=None, helper_name: str = "", use_facts: bool = False, nlp=None):
        self.s, self.cfg, self.llm = settings or Settings(), cfg or Config(), llm
        self.search = search  # optional reader of past chats, see engram_parallel.py
        self.helper, self.helper_name, self.use_facts, self.nlp = helper, helper_name, use_facts, nlp  # optional, see engram_facts.py
        self.show_dates = True
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(f"PRAGMA synchronous={'FULL' if durable else 'OFF'}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS chats(chat INTEGER PRIMARY KEY, opened_on TEXT, last_active REAL, closed INTEGER DEFAULT 0, note TEXT);
            CREATE TABLE IF NOT EXISTS sentences(sid INTEGER PRIMARY KEY, chat INTEGER, pos INTEGER, ref TEXT, speaker TEXT, date TEXT, text TEXT, caption TEXT, tokens INTEGER);
            CREATE INDEX IF NOT EXISTS sentences_chat ON sentences(chat);
            CREATE VIRTUAL TABLE IF NOT EXISTS words_index USING fts5(words, tokenize="unicode61 remove_diacritics 0");
        """)
        self.db.executescript(F.SCHEMA)
        R.prepare(self.db)
        self.index = Engrams(cfg=self.cfg)
        self.indexed_any = False
        self.chat = None
        self.desk: deque = deque()
        self.desk_used = 0
        self._reload()

    # ---------- restart ----------
    def _reload(self):
        """Rebuild what lives in RAM from the disk."""
        closed = self.db.execute("SELECT chat, opened_on FROM chats WHERE closed=1 ORDER BY chat").fetchall()
        for chat, opened in closed: self._add_to_index(chat, opened)
        if closed: self._sleep()
        open_chat = self.db.execute("SELECT chat FROM chats WHERE closed=0 ORDER BY chat DESC LIMIT 1").fetchone()
        if open_chat:
            self.chat = open_chat[0]
            for sid, text, tokens in self.db.execute("SELECT sid, text, tokens FROM sentences WHERE chat=? ORDER BY sid", (self.chat,)):
                self._to_desk(sid, tokens)

    def _add_to_index(self, chat, opened_on):
        ep = self.index.add_episode(opened_on)
        for sid, speaker, text, caption in self.db.execute("SELECT sid, speaker, text, caption FROM sentences WHERE chat=? ORDER BY sid", (chat,)):
            self.index.add_turn(str(sid), ep, speaker, text, caption or "")
        self.indexed_any = True

    def _sleep(self):
        self.index.sleep()
        if self.search is not None: self.search.rebuild(self.db)

    # ---------- during a chat ----------
    def open_chat(self, date: str = "") -> int:
        """Starts a chat. If the last one was never closed, it continues."""
        if self.chat is None:
            self.chat = self.db.execute("INSERT INTO chats(opened_on, last_active) VALUES(?,?)", (date, time.time())).lastrowid
            self.db.commit()
            self.desk.clear(); self.desk_used = 0
        return self.chat

    def carried_note(self) -> str:
        """The state left by the last closed chat."""
        r = self.db.execute("SELECT note FROM chats WHERE closed=1 AND note IS NOT NULL ORDER BY chat DESC LIMIT 1").fetchone()
        return r[0] if r else ""

    def add(self, speaker: str, text: str, date: str = "", ref: str = "", caption: str = "", commit: bool = True) -> int:
        """Saves one sentence to disk before anything else happens to it."""
        if self.chat is None: self.open_chat(date)
        body = text + (f" [image: {caption}]" if caption else "")
        tokens = count_tokens(body)
        pos = self.db.execute("SELECT count(*) FROM sentences WHERE chat=?", (self.chat,)).fetchone()[0]
        sid = self.db.execute("INSERT INTO sentences(chat,pos,ref,speaker,date,text,caption,tokens) VALUES(?,?,?,?,?,?,?,?)",
                              (self.chat, pos, ref, speaker, date, text, caption, tokens)).lastrowid
        self.db.execute("INSERT INTO words_index(rowid, words) VALUES(?,?)", (sid, " ".join(dict.fromkeys(words(body + " " + speaker)))))
        self.db.execute("UPDATE chats SET last_active=? WHERE chat=?", (time.time(), self.chat))
        if commit: self.db.commit()
        self._to_desk(sid, tokens)
        return sid

    def _to_desk(self, sid, tokens):
        self.desk.append((sid, tokens))
        self.desk_used += tokens
        if self.desk_used > self.s.desk_tokens:
            moved = 0
            while self.desk and moved < self.s.push_tokens:
                _, t = self.desk.popleft()
                self.desk_used -= t
                moved += t

    def _this_chat_off_desk(self, question: str) -> list[int]:
        """Simple word search inside the open chat, for what has left the desk."""
        if self.chat is None: return []
        terms = [t for t in (re.sub(r"[^a-z0-9]", "", w) for w in dict.fromkeys(words(question))) if t]
        if not terms: return []
        on_desk = {s for s, _ in self.desk}
        q = " OR ".join(f'"{t}"' for t in terms)
        rows = self.db.execute("""SELECT w.rowid FROM words_index w JOIN sentences s ON s.sid = w.rowid
                                  WHERE words_index MATCH ? AND s.chat = ? ORDER BY bm25(words_index) LIMIT 80""", (q, self.chat)).fetchall()
        return [r[0] for r in rows if r[0] not in on_desk]

    def _next_in_chat(self, sid: int):
        r = self.db.execute("SELECT b.sid FROM sentences a JOIN sentences b ON b.chat=a.chat AND b.pos=a.pos+1 WHERE a.sid=?", (sid,)).fetchone()
        return r[0] if r else None

    def _past_chats(self, question: str) -> list[int]:
        """Engram's read path: shortlist old chats, then choose sentences. Closed chats only."""
        if not self.indexed_any: return []
        if self.search is not None: return self.search.order(question)
        return [int(self.index.engrams[n].id) for n in self.index.retrieve(question, None, use_habit=False)]

    def find(self, question: str):
        """Sentence ids to bring to the desk. Overflow from this chat gets up to half the space, past chats the rest."""
        t0 = time.perf_counter()
        on_desk = {s for s, _ in self.desk}
        out, seen, used = [], set(on_desk), 0
        overflow = self._this_chat_off_desk(question)

        def take(sid, limit):
            nonlocal used
            if sid in seen or used >= limit: return
            seen.add(sid); out.append(sid)
            used += self._tokens(sid) + 8  # 8 tokens for the date stamp

        for sid in overflow:
            if used >= self.s.retrieved_tokens // 2: break
            take(sid, self.s.retrieved_tokens // 2)
            nxt = self._next_in_chat(sid)
            if nxt is not None: take(nxt, self.s.retrieved_tokens // 2)
        for sid in self._past_chats(question):
            if self.search is not None and used >= self.s.retrieved_tokens: break
            take(sid, self.s.retrieved_tokens)
            if self.search is not None:  # the reply after a sentence often holds the answer
                nxt = self._next_in_chat(sid)
                if nxt is not None: take(nxt, self.s.retrieved_tokens)
        return out, dict(tokens=used, ms=(time.perf_counter() - t0) * 1000, overflow_hits=len(overflow))

    def _tokens(self, sid: int) -> int:
        return self.db.execute("SELECT tokens FROM sentences WHERE sid=?", (sid,)).fetchone()[0]

    def text_of(self, sids: list[int]) -> str:
        lines = []
        for sid in sids:
            speaker, date, text, caption = self.db.execute("SELECT speaker, date, text, caption FROM sentences WHERE sid=?", (sid,)).fetchone()
            shown = with_dates(text, date) if self.show_dates else text
            lines.append(f"[{date}] {speaker}: {shown}" + (f" [image: {caption}]" if caption else ""))
        return "\n".join(lines)

    def desk_text(self) -> str:
        return self.text_of([s for s, _ in self.desk])

    def context_for(self, question: str) -> tuple[str, dict]:
        """Everything the model reads for a question: carried note, the desk, and retrieved memories."""
        sids, info = self.find(question)
        parts = []
        note = self.carried_note()
        if note: parts.append("[Handover note from the last chat]\n" + note)
        if self.use_facts:
            shown = self.facts_text(question)
            if shown: parts.append(shown)
        if sids: parts.append(self.text_of(sorted(sids) if self.search is not None else sids))
        if self.desk: parts.append(self.desk_text())
        return "\n".join(parts) or "(no excerpts)", info | dict(sids=sids)

    # ---------- closing ----------
    def close_chat(self, write_note: bool = True, sleep: bool = True, chain: bool = True):
        """Moves the chat to long-term memory. The sentences are already on disk, so nothing can be lost here."""
        if self.chat is None: return None
        chat = self.chat
        self.db.execute("UPDATE chats SET closed=1 WHERE chat=?", (chat,))
        self.db.commit()
        opened = self.db.execute("SELECT opened_on FROM chats WHERE chat=?", (chat,)).fetchone()[0]
        self._add_to_index(chat, opened)
        t0 = time.perf_counter()
        if sleep: self._sleep()
        ms = (time.perf_counter() - t0) * 1000
        self.chat = None
        self.desk.clear(); self.desk_used = 0
        note = None
        if write_note and self.llm is not None: note = self.write_note(chat, chain)
        facts = self.write_facts(chat) if self.helper is not None else None
        return dict(chat=chat, sleep_ms=ms, note=note, facts=facts)

    def close_if_idle(self, idle_seconds: float, **kw):
        """Treats a long-idle chat as closed, for chats nobody closed properly."""
        if self.chat is None: return None
        last = self.db.execute("SELECT last_active FROM chats WHERE chat=?", (self.chat,)).fetchone()[0]
        return self.close_chat(**kw) if time.time() - last >= idle_seconds else None

    def chat_text(self, chat: int) -> str:
        """The chat as the model reads it for its note: the latest part when the chat is very long."""
        rows = self.db.execute("SELECT date, speaker, text, caption, tokens FROM sentences WHERE chat=? ORDER BY sid DESC", (chat,)).fetchall()
        keep, used = [], 0
        for date, speaker, text, caption, tokens in rows:
            if used + tokens > self.s.note_read_tokens: break
            shown = with_dates(text, date) if self.show_dates else text
            keep.append(f"{speaker}: {shown}" + (f" [image: {caption}]" if caption else "")); used += tokens
        date = rows[0][0] if rows else ""
        return f"[{date}]\n" + "\n".join(reversed(keep))

    def write_note(self, chat: int, chain: bool = True) -> str:
        previous = ""
        if chain:
            r = self.db.execute("SELECT note FROM chats WHERE closed=1 AND chat<? AND note IS NOT NULL ORDER BY chat DESC LIMIT 1", (chat,)).fetchone()
            if r: previous = f"\nThe note from the chat before this one, to carry forward anything still current:\n{r[0]}\n"
        note = self.llm(NOTE_PROMPT.format(previous=previous, chat=self.chat_text(chat)), max_tokens=self.s.note_tokens)[0]
        self.db.execute("UPDATE chats SET note=? WHERE chat=?", (note, chat))
        self.db.commit()
        return note

    # ---------- facts and episode context, optional ----------
    def write_facts(self, chat: int) -> dict:
        """The helper reads a closed chat. A failure changes nothing and can be retried; running again replaces the chat's facts."""
        rows = self.db.execute("SELECT sid, ref, speaker, date, text, caption FROM sentences WHERE chat=? ORDER BY sid", (chat,)).fetchall()
        ids, lines, sentences = {}, [], {}
        for sid, ref, speaker, date, text, caption in rows:
            line_id = ref or f"S{sid}"
            ids[line_id] = sid
            body = f"{speaker}: {with_dates(text, date)}" + (f" [image: {caption}]" if caption else "")
            lines.append(f"{line_id} {body}")
            sentences[line_id] = body
        date = rows[0][3] if rows else ""
        t0 = time.perf_counter()
        try:
            reply = self.helper(F.PROMPT.format(chat=f"[{date}]\n" + "\n".join(lines)), max_tokens=self.s.fact_reply_tokens)[0]
        except Exception as e:
            self.db.execute("INSERT OR REPLACE INTO episodes(chat, status, model, seconds, raw) VALUES(?,?,?,?,?)", (chat, "failed", self.helper_name, time.perf_counter() - t0, str(e)[:500]))
            self.db.commit()
            return dict(status="failed", error=str(e)[:200])
        secs = time.perf_counter() - t0
        context, facts, rejected = F.parse(reply, sentences, words)
        self._drop_facts("chat=?", (chat,))
        for f in facts:
            person, place, when, number = F.fields(f["text"], self.nlp)
            fid = self.db.execute("INSERT INTO facts(chat,text,person,place,time,number,checked,model) VALUES(?,?,?,?,?,?,?,?)",
                                  (chat, f["text"], person, place, when, number, "moved" if f["moved"] else "linked", self.helper_name)).lastrowid
            for link in f["links"]:
                self.db.execute("INSERT INTO fact_sources(fact, sid) VALUES(?,?)", (fid, ids[link]))
            self.db.execute("INSERT INTO facts_index(rowid, words) VALUES(?,?)", (fid, " ".join(dict.fromkeys(words(f["text"])))))
        moved = sum(f["moved"] for f in facts)
        self._fact_vectors = None
        self.db.execute("INSERT OR REPLACE INTO episodes(chat, context, status, model, seconds, accepted, moved, rejected, raw) VALUES(?,?,?,?,?,?,?,?,?)",
                        (chat, context, "done" if facts else "empty", self.helper_name, secs, len(facts), moved, rejected, reply[:2500]))
        self.db.commit()
        return dict(status="done" if facts else "empty", accepted=len(facts), moved=moved, rejected=rejected, seconds=secs)

    def _drop_facts(self, where: str, args: tuple):
        fids = [r[0] for r in self.db.execute(f"SELECT fact FROM facts WHERE {where}", args).fetchall()]
        for fid in fids:
            self.db.execute("DELETE FROM facts_index WHERE rowid=?", (fid,))
            self.db.execute("DELETE FROM fact_sources WHERE fact=?", (fid,))
            self.db.execute("DELETE FROM facts WHERE fact=?", (fid,))
        return len(fids)

    def _ranked_facts(self, question: str) -> list[tuple]:
        """Meaning and word search over facts, the method used in the facts test. Needs the search layer."""
        s = self.search
        if getattr(self, "_fact_vectors", None) is None:
            rows = self.db.execute("SELECT fact, text FROM facts ORDER BY fact").fetchall()
            w = [words(r[1]) for r in rows]
            df = Counter(x for ws in w for x in set(ws))
            self._fact_vectors = (rows, w, df, sum(map(len, w)) / max(1, len(w)), s._embed([r[1] for r in rows]) if rows else None)
        rows, w, df, avg, vectors = self._fact_vectors
        if not rows: return []
        qv, qw, n = s._embed([question])[0], set(words(question)), len(rows)
        mid, st = s._curve(question)
        score = vectors @ qv
        for i, ws in enumerate(w):
            b = 0.
            for x in qw & set(ws):
                tf = ws.count(x); b += math.log(1 + (n - df[x] + .5) / (df[x] + .5)) * tf * 2.2 / (tf + 1.2 * (.5 + .5 * len(ws) / avg))
            if b > 0: score[i] += 1 / (1 + math.exp(-st * (b - mid)))
        return [rows[i] for i in np.argsort(-score)]

    def find_facts(self, question: str, limit: int = 8) -> list[dict]:
        """Noted facts for a question, each with the ids of the sentences behind it."""
        if self.search is not None and self.search.N:
            return [dict(fact=fid, text=text, sources=[r[0] for r in self.db.execute("SELECT sid FROM fact_sources WHERE fact=?", (fid,))]) for fid, text in self._ranked_facts(question)[:limit]]
        terms = [t for t in (re.sub(r"[^a-z0-9]", "", w) for w in dict.fromkeys(words(question))) if t]
        if not terms: return []
        q = " OR ".join(f'"{t}"' for t in terms)
        rows = self.db.execute("""SELECT f.fact, f.text FROM facts_index i JOIN facts f ON f.fact = i.rowid
                                  WHERE facts_index MATCH ? ORDER BY bm25(facts_index) LIMIT ?""", (q, limit)).fetchall()
        return [dict(fact=fid, text=text, sources=[r[0] for r in self.db.execute("SELECT sid FROM fact_sources WHERE fact=?", (fid,))]) for fid, text in rows]

    def facts_text(self, question: str) -> str:
        lines, used = [], 0
        ranked = self._ranked_facts(question) if self.search is not None and self.search.N else [(f["fact"], f["text"]) for f in self.find_facts(question, 40)]
        for _, text in ranked:
            t = count_tokens(text) + 6
            if used + t > self.s.fact_tokens: break
            lines.append("[Noted fact] " + text); used += t
        return "\n".join(lines)

    def episode_context(self, chat: int) -> str:
        r = self.db.execute("SELECT context FROM episodes WHERE chat=?", (chat,)).fetchone()
        return (r[0] or "") if r else ""

    def forget_sentence(self, sid: int) -> dict:
        """Deletes one sentence and every fact built on it, then rebuilds what lives in RAM."""
        fids = [r[0] for r in self.db.execute("SELECT fact FROM fact_sources WHERE sid=?", (sid,)).fetchall()]
        for fid in fids: self._drop_facts("fact=?", (fid,))
        self._fact_vectors = None
        R.forget_sentence(self.db, sid)
        n = self.db.execute("DELETE FROM sentences WHERE sid=?", (sid,)).rowcount
        self.db.execute("DELETE FROM words_index WHERE rowid=?", (sid,))
        self.db.commit()
        self.index = Engrams(cfg=self.cfg)
        self.indexed_any, self.chat = False, None
        self.desk.clear(); self.desk_used = 0
        if self.search is not None: self.search.seen.pop(sid, None)
        self._reload()
        if self.search is not None and not self.indexed_any: self.search.rebuild(self.db)
        return dict(sentences=n, facts=len(fids))

    def close(self):
        self.db.commit()
        self.db.close()
