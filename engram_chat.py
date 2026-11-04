"""Engram chat memory: chats open and close, and closing moves memory to long-term storage.

During a chat: every sentence is saved to disk on arrival, the desk holds the recent conversation,
and a simple word search covers anything that has overflowed the desk.
When a chat closes: the chat joins Engram's long-term index, and a model may write a short note
that carries its state to the next chat.
Later chats: Engram's session shortlist finds the right old chats, then the right sentences.

The model only ever receives plain text, so any open model can sit on top.
"""
import os
import re
import sqlite3
import time
from collections import deque
from dataclasses import dataclass

import engram
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


class ChatMemory:
    def __init__(self, path: str, cfg: Config | None = None, settings: Settings | None = None, llm=None, durable: bool = True):
        self.s, self.cfg, self.llm = settings or Settings(), cfg or Config(), llm
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(f"PRAGMA synchronous={'FULL' if durable else 'OFF'}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS chats(chat INTEGER PRIMARY KEY, opened_on TEXT, last_active REAL, closed INTEGER DEFAULT 0, note TEXT);
            CREATE TABLE IF NOT EXISTS sentences(sid INTEGER PRIMARY KEY, chat INTEGER, pos INTEGER, ref TEXT, speaker TEXT, date TEXT, text TEXT, caption TEXT, tokens INTEGER);
            CREATE INDEX IF NOT EXISTS sentences_chat ON sentences(chat);
            CREATE VIRTUAL TABLE IF NOT EXISTS words_index USING fts5(words, tokenize="unicode61 remove_diacritics 0");
        """)
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
            take(sid, self.s.retrieved_tokens)
        return out, dict(tokens=used, ms=(time.perf_counter() - t0) * 1000, overflow_hits=len(overflow))

    def _tokens(self, sid: int) -> int:
        return self.db.execute("SELECT tokens FROM sentences WHERE sid=?", (sid,)).fetchone()[0]

    def text_of(self, sids: list[int]) -> str:
        lines = []
        for sid in sids:
            speaker, date, text, caption = self.db.execute("SELECT speaker, date, text, caption FROM sentences WHERE sid=?", (sid,)).fetchone()
            lines.append(f"[{date}] {speaker}: {text}" + (f" [image: {caption}]" if caption else ""))
        return "\n".join(lines)

    def desk_text(self) -> str:
        return self.text_of([s for s, _ in self.desk])

    def context_for(self, question: str) -> tuple[str, dict]:
        """Everything the model reads for a question: carried note, the desk, and retrieved memories."""
        sids, info = self.find(question)
        parts = []
        note = self.carried_note()
        if note: parts.append("[Handover note from the last chat]\n" + note)
        if sids: parts.append(self.text_of(sids))
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
        return dict(chat=chat, sleep_ms=ms, note=note)

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
            keep.append(f"{speaker}: {text}" + (f" [image: {caption}]" if caption else "")); used += tokens
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

    def close(self):
        self.db.commit()
        self.db.close()
