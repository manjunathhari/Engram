"""Whole-chat filing: a closed chat's facts written once into the P1 knowledge store, checked before they are kept.

The chain, in order:
1. A writer model reads the whole chat (every line with an id and the chat's date) and proposes facts: who, kind, topic,
   the thing, one self-contained sentence, the time phrase as written, whether it says something ended, and its line ids.
2. Code checks each proposal against its cited lines: the lines exist, the thing is in them, who is in them, the P2 gate
   (P2.1 interpreter and P2.2 learning policy) does not mark every cited line as small talk, and fixed rules for passing
   remarks, wishes and negation.
3. An optional checking model, never the writer itself, sees only the fact and its lines and says whether the lines state it
   and whether it is lasting, a plan, an event or only a passing remark.
4. Fixed rules decide: stored as a P1 record, held in the P1 review queue, or dropped. A model never decides alone.
Stored records keep the kind as P1's fact type, the topic as the predicate ("topic:pet", or "ended:car" for something that
stopped), the thing as the value, the sentence as the note, and the date it happened as valid_from. So P3 works out what is
true now, P5 records decisions, and deleting a sentence removes every record resting on it, as for any P1 fact.
Nothing here runs unless a writer is given. The sentences are never changed.
"""
import datetime as dt
import json
import math
import re
import time
from collections import Counter

from .database import KnowledgeBase
from .entity_resolver import EntityResolver
from .fact_validator import HYPOTHETICAL, NEGATION, OTHER_ACTOR, SHORT_WINDOW, SourceLine
from .models import Entity, FactEvidence, FactWithEvidence, KnowledgeFact, RecordError, normalize_name, tidy
from .normalizer import content_stems, stem, stems, time_value, time_values_in, tokens
from .temporal import said_on, span
from .transition import ENDED, TOPIC, topic_key

FILING_VERSION = "1"
STORED, REVIEW, DROPPED = "stored", "review", "dropped"
KINDS = dict(attribute="durable", relationship="durable", preference="durable", event="episodic", plan="project")  # P1 fact type: P2.2 retention class
KIND_WORDS = dict(attribute="trait", relationship="relationship", preference="preference", event="event", plan="plan")
CHECK_STANDING = dict(attribute="lasting", relationship="lasting", preference="lasting", event="event", plan="plan")
ASSISTANT = {"assistant", "ai", "the assistant", "chatbot", "bot"}

SKIP = ("feelings and moods of the moment (bored, motivated, proud, excited); compliments and thanks; reactions such as "
        "'I like the sound of' or 'sounds great'; comparisons with 'like' ('it's like a nudge'); opinions about someone else's things; "
        "wishes, hopes, maybes and questions; a meal or a short break; suggestions the other speaker made that were not taken up; "
        "anything only about the assistant.")

PROMPT = """Below is one conversation. Each line starts with its id. List the facts about the people in it that are worth remembering in later conversations.
{context}
For each fact give:
- who: the person it is about, by the name the conversation uses (the speaker's name for "I").
- kind: attribute (who they are: job, home, health, skills, things they own), relationship (people in their life), preference (what they like, dislike or prefer), event (something that happened, once), or plan (something they intend to do).
- topic: one or two words naming the group it belongs to, such as pet, job, festival, instrument, book or trip.
- thing: the shortest exact words from the conversation naming the thing.
- fact: one sentence that makes sense alone, with full names and no he, she or they.
- time: the time phrase exactly as written, such as last Saturday or in 2019; empty if none.
- ended: true only if the fact says something stopped being true (sold, quit, moved out, broke up).
- lines: the ids of the lines that state it.
Write a fact only when a line states it plainly. Skip {skip}
If there are no such facts, return an empty list.

Conversation:
{chat}
"""

SCHEMA = {"type": "object", "properties": {"facts": {"type": "array", "items": {"type": "object", "properties": {
    "who": {"type": "string"}, "kind": {"type": "string", "enum": list(KINDS)}, "topic": {"type": "string"}, "thing": {"type": "string"},
    "fact": {"type": "string"}, "time": {"type": "string"}, "ended": {"type": "boolean"}, "lines": {"type": "array", "items": {"type": "string"}}},
    "required": ["who", "kind", "topic", "thing", "fact", "time", "ended", "lines"], "additionalProperties": False}}},
    "required": ["facts"], "additionalProperties": False}

CHECK_PROMPT = """Here is a statement and the lines of a conversation it was taken from.
Statement: {fact}
Lines:
{lines}
Answer two questions in JSON: {{"states": "yes", "partly" or "no", "standing": "lasting", "plan", "event" or "passing"}}.
states: do the lines plainly state the statement? partly if they state only some of it, or say it less surely.
standing: lasting if it stays true (who someone is, what they have, like or are related to); plan if it is intended; event if it happened once; passing if it is only a feeling, reaction or remark of the moment."""

CHECK_SCHEMA = {"type": "object", "properties": {"states": {"type": "string", "enum": ["yes", "partly", "no"]},
                "standing": {"type": "string", "enum": ["lasting", "plan", "event", "passing"]}},
                "required": ["states", "standing"], "additionalProperties": False}

# Passing remarks seen in the P2.6 sets. Rules taken from seen data: their effect must be measured on fresh data.
REACTION = re.compile(r"\b(like|love)s? the (sound|idea|look) of\b|\bsounds? (great|good|amazing|awesome|fun|delicious|lovely|interesting)\b", re.I)
COMPARISON = re.compile(r"\b(is|it's|its|are|was|were|be|feels?|seems?|looks?)\s+(just\s+)?like\b", re.I)
WISH = re.compile(r"\b(want(ed)? to|wanna|would love to|i'd love to|hope to|hoping to|wish|might|maybe|someday|one day)\b", re.I)
MOODS = set("bored tired happy sad excited nervous motivated stressed grateful thankful thrilled proud determined overwhelmed anxious "
            "upset angry glad relieved inspired curious worried scared lonely exhausted pumped stoked hopeful confident frustrated".split())

WEEKDAY_SHORT = dict(mon="monday", tue="tuesday", tues="tuesday", wed="wednesday", weds="wednesday", thu="thursday", thur="thursday",
                     thurs="thursday", fri="friday", sat="saturday", sun="sunday")
WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
VAGUE = ("week of", "weekend of", "around", "a few", "spring", "summer", "fall", "autumn", "winter")

SCHEMA_SQL = """CREATE TABLE IF NOT EXISTS filings(chat INTEGER PRIMARY KEY, status TEXT, writer TEXT, checker TEXT, seconds REAL,
    proposed INTEGER, stored INTEGER, linked INTEGER, review INTEGER, dropped INTEGER, unreadable INTEGER, reasons TEXT, version TEXT)"""


def resolve_time(phrase, day, plan=False):
    """A time phrase as a P1 date (YYYY, YYYY-MM or YYYY-MM-DD), given the day it was said. None when no plain date follows.
    Approximate phrases ("two weeks ago", "last week") give None rather than a date more exact than what was said."""
    p = " ".join(re.sub(r"[^a-z0-9 ]", " ", str(phrase or "").lower()).split())
    if not p or day is None: return None
    p = re.sub(r"^(on|in|at|during|this past|past)\s+", lambda m: "last " if m.group(1) in ("this past", "past") else "", p)
    stated = time_value(phrase) or time_value(p)
    if stated: return stated
    p = " ".join(WEEKDAY_SHORT.get(w, w) for w in p.split())
    if p in WEEKDAYS: p = ("next " if plan else "last ") + p
    import engram_chat as E  # the same resolver that shows dates beside sentences
    try:
        r = E.resolve_phrase(p, day)
    except (KeyError, ValueError):
        return None
    if not r or r.lower().startswith(VAGUE): return None
    return time_value(r)


def _day(date_text):
    d = said_on(date_text)
    return dt.date.fromisoformat(d) if d and len(d) == 10 else None


def _pretty(period):
    if not period: return ""
    parts = period.split("-")
    if len(parts) == 3: return dt.date.fromisoformat(period).strftime("%-d %B %Y")
    if len(parts) == 2: return dt.date(int(parts[0]), int(parts[1]), 1).strftime("%B %Y")
    return period


def _read_json(raw):
    """A model's JSON reply, allowing a code fence. None when it cannot be read."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except ValueError:
            return None


class FactFiler:
    """writer(prompt, schema) -> reply text. checker, if given, the same shape; it must be a different model from the writer."""

    def __init__(self, memory, writer, checker=None, writer_name="", checker_name="", context="", skip=SKIP, people=(), max_chat_tokens=6000):
        if checker is not None and checker is writer: raise ValueError("the checker must not be the writer itself")
        if checker is not None and checker_name and checker_name == writer_name: raise ValueError("the checker must be a different model from the writer")
        self.m, self.db = memory, memory.db
        self.kb = KnowledgeBase(self.db)
        self.resolver = EntityResolver(self.kb)
        self.writer, self.checker = writer, checker
        self.writer_name, self.checker_name = writer_name or "writer", checker_name or ""
        self.context = (f"Context: {tidy(context)}\n" if tidy(context) else "")
        self.skip, self.people, self.max_chat_tokens = skip, tuple(people), max_chat_tokens
        self.db.execute(SCHEMA_SQL)
        self.db.commit()
        self._cache = None

    # ---------- writing ----------
    def chat_lines(self, chat):
        """[(line id, sid, speaker, date, text)] for a chat, with long lines shortened for the writer only when the chat is too long."""
        rows = self.db.execute("SELECT sid, speaker, date, text, caption FROM sentences WHERE chat=? ORDER BY sid", (chat,)).fetchall()
        out = [(f"L{n}", sid, speaker, date or "", text + (f" [image: {caption}]" if caption else "")) for n, (sid, speaker, date, text, caption) in enumerate(rows, 1)]
        return out

    def prompt(self, chat):
        import engram_chat as E
        lines = self.chat_lines(chat)
        budget, shown = self.max_chat_tokens * 4, []
        total = sum(len(t) for *_, t in lines)
        for line_id, sid, speaker, date, text in lines:
            body = E.with_dates(text, date) if date else text
            if total > budget and normalize_name(speaker) in ASSISTANT and len(body) > 300: body = body[:300] + " ..."
            shown.append(f"{line_id} {speaker}: {body}")
        date = lines[0][3] if lines else ""
        return PROMPT.format(context=self.context, skip=self.skip, chat=(f"[{date}]\n" if date else "") + "\n".join(shown)), lines

    def file_chat(self, chat):
        """Files one closed chat. Running it again on the same chat adds nothing twice."""
        t0 = time.perf_counter()
        text, lines = self.prompt(chat)
        speakers = {normalize_name(sp) for _, _, sp, _, _ in lines}
        people = tuple(dict.fromkeys(self.people + tuple(sp for _, _, sp, _, _ in lines if normalize_name(sp) not in ASSISTANT)))
        try:
            reply = self.writer(text, SCHEMA)
        except Exception as e:
            self._log(chat, "failed", time.perf_counter() - t0, Counter(), dict(error=str(e)[:200]))
            return dict(status="failed", error=str(e)[:200])
        data = _read_json(reply)
        proposals = data.get("facts", []) if isinstance(data, dict) else None
        if not isinstance(proposals, list):
            self._log(chat, "unreadable", time.perf_counter() - t0, Counter(unreadable=1), {})
            return dict(status="unreadable")
        self._drop_old_reviews(chat)
        ids = {line_id: (sid, speaker, date, txt) for line_id, sid, speaker, date, txt in lines}
        counts, reasons, results = Counter(proposed=len(proposals)), Counter(), []
        for p in proposals:
            if not isinstance(p, dict) or not all(k in p for k in ("who", "kind", "topic", "thing", "fact", "lines")):
                counts["unreadable"] += 1
                continue
            outcome = self.decide(p, ids, people)
            for r in outcome["reasons"]: reasons[r] += 1
            if outcome["verdict"] == STORED:
                stored = self._store(p, outcome)
                counts["linked" if stored["linked"] else "stored"] += 1
                results.append(dict(verdict=STORED, fact_id=stored["fact_id"], linked=stored["linked"]))
            elif outcome["verdict"] == REVIEW:
                counts["review"] += 1
                results.append(dict(verdict=REVIEW, review_id=self._review(p, outcome), reasons=outcome["reasons"]))
            else:
                counts["dropped"] += 1
                results.append(dict(verdict=DROPPED, reasons=outcome["reasons"]))
        self._cache = None
        self._log(chat, "done", time.perf_counter() - t0, counts, dict(reasons))
        return dict(status="done", **{k: counts[k] for k in ("proposed", "stored", "linked", "review", "dropped", "unreadable")}, results=results)

    # ---------- checking ----------
    def decide(self, p, ids, people=()):
        """Code checks, then the checking model if there is one. Returns verdict, reasons, and what storing needs."""
        drop, unsure = [], []
        kind = str(p.get("kind", "")).strip().lower()
        topic, thing, fact = topic_key(p.get("topic")), tidy(p.get("thing")), tidy(p.get("fact"))
        cited = [ids[x] + (x,) for x in dict.fromkeys(str(i).strip() for i in (p.get("lines") or [])) if x in ids]
        if not cited: drop.append("no cited line exists")
        if kind not in KINDS: drop.append("kind not allowed")
        if not topic: drop.append("no topic")
        wanted = content_stems(thing)
        if not wanted: drop.append("thing is empty or has no meaning alone")
        if not fact: drop.append("no fact sentence")
        if drop: return dict(verdict=DROPPED, reasons=drop)
        source = [SourceLine(sid, speaker, txt, date) for sid, speaker, date, txt, _ in cited]

        # The thing must be in the cited lines.
        said = [(s, stems(s.text)) for s in source]
        everywhere = {w for _, st in said for w in st}
        if sum(w in everywhere for w in wanted) / len(wanted) < (1 if len(wanted) <= 3 else .75):
            return dict(verdict=DROPPED, reasons=["thing not in its cited lines"])
        holder = next((s for s, st in said if wanted[0] in st), source[0])
        sentence = next((x for x in re.split(r"(?<=[.!?])\s+|\n+", holder.text) if wanted[0] in stems(x)), holder.text)

        # Who: resolved from the cited lines, never guessed.
        who = self.resolver.resolve(p.get("who", ""), source)
        if who.status == "absent" or not normalize_name(p.get("who", "")): return dict(verdict=DROPPED, reasons=[who.why or "no subject"])
        if normalize_name(who.name) in ASSISTANT: return dict(verdict=DROPPED, reasons=["about the assistant"])
        if who.status in ("unresolved", "ambiguous"): unsure.append(who.why)
        if all(normalize_name(s.speaker) in ASSISTANT for s in source) and normalize_name(who.name) not in ASSISTANT:
            unsure.append("said by the assistant, not the person")

        # The P2 gate on each cited line.
        gate = self._gate(cited, people)
        if gate and all(g == "ignored" for g in gate): return dict(verdict=DROPPED, reasons=["every cited line is small talk, a question or a greeting"])
        if kind != "plan" and "uncertain_or_conditional_content" in gate: unsure.append("the source may be a wish or a maybe")

        # Passing remarks, wishes and negation, by fixed rules.
        if REACTION.search(sentence): return dict(verdict=DROPPED, reasons=["a reaction, not a fact"])
        if kind == "preference" and COMPARISON.search(sentence) and not re.search(r"\b(i|we)\s+(really\s+)?(like|love)", sentence, re.I):
            return dict(verdict=DROPPED, reasons=["'like' used as a comparison"])
        if kind == "attribute" and set(tokens(thing)) & MOODS and len(wanted) <= 2: return dict(verdict=DROPPED, reasons=["a passing feeling"])
        if kind != "plan" and WISH.search(sentence): unsure.append("the source may be a wish")
        words = tokens(sentence)
        at = next((i for i, w in enumerate(words) if stem(w) == wanted[0]), len(words))
        near = set(words[max(0, at - SHORT_WINDOW):at + 1])
        if not p.get("ended") and kind != "preference" and (NEGATION & near or "no longer" in sentence.lower() or "used to" in sentence.lower()):
            unsure.append("the source may say this is not so")
        if kind != "plan" and (HYPOTHETICAL - {"thinking", "considering"}) & near: unsure.append("the source may be hypothetical")
        if sentence.rstrip().endswith("?"): unsure.append("the source is a question")
        if normalize_name(holder.speaker) == normalize_name(who.name) and OTHER_ACTOR & set(words[:at + 1]) and not {"i", "im", "ive", "id", "we"} & set(words[:at + 1]):
            unsure.append("this may be about someone else the speaker mentions")

        # Time: a stated date must be in the source; a relative phrase is resolved against the day it was said.
        phrase = tidy(p.get("time"))
        said_day = _day(holder.date)
        when = resolve_time(phrase, said_day, plan=kind == "plan") if phrase else None
        stated = time_value(phrase) if phrase else None
        if stated and stated not in set().union(*(time_values_in(s.text, s.date) for s in source)): unsure.append("time is not in the source")

        # The checking model, if there is one: a different model from the writer.
        check = None
        if self.checker is not None:
            check = self._check(fact, source)
            if check is None: unsure.append("the check could not be made")
            elif check["states"] == "no": return dict(verdict=DROPPED, reasons=["the checker found the lines do not state it"], check=check)
            elif check["standing"] == "passing": return dict(verdict=DROPPED, reasons=["the checker found a passing remark"], check=check)
            else:
                if check["states"] == "partly": unsure.append("the checker found the lines state it only partly")
                if check["standing"] != CHECK_STANDING[kind] and not (p.get("ended") and check["standing"] == "event"): unsure.append(f"the checker read it as {check['standing']}, not {CHECK_STANDING[kind]}")
        return dict(verdict=REVIEW if unsure else STORED, reasons=unsure, who=who, kind=kind, topic=topic, thing=thing, fact=fact,
                    when=when, phrase=phrase, said_day=said_day.isoformat() if said_day else None, sids=[s.sid for s in source], check=check)

    def _gate(self, cited, people):
        from .learning_queue import prepare_stored
        out = []
        for sid, *_ in cited:
            try:
                for r in prepare_stored(self.db, sid, people=people):
                    # A short reply ("I love pizza.") is ignored by the gate because, alone, it may need the line before it.
                    # The writer read the whole chat, so only small talk, greetings, thanks and questions count as ignored here.
                    if r.status == "ignored": out.append("short" if set(r.reason_codes) <= {"short_reply"} else "ignored")
                    else: out.append(r.envelope.to_dict()["lifecycle"])
                    if r.envelope is not None and "uncertain_or_conditional_content" in r.envelope.to_dict()["learning"]["reason_codes"]:
                        out.append("uncertain_or_conditional_content")
            except Exception:
                out.append("unknown")
        return out

    def _check(self, fact, source):
        lines = "\n".join(f"{s.speaker}: {s.text}" for s in source)
        try:
            data = _read_json(self.checker(CHECK_PROMPT.format(fact=fact, lines=lines), CHECK_SCHEMA))
        except Exception:
            return None
        if not isinstance(data, dict) or data.get("states") not in ("yes", "partly", "no") or data.get("standing") not in ("lasting", "plan", "event", "passing"): return None
        return dict(states=data["states"], standing=data["standing"])

    # ---------- storing ----------
    def _store(self, p, o):
        who = o["who"].entity_id or self.kb.resolve(o["who"].name) or self.kb.add_entity(Entity.create(o["who"].name, "person")).entity_id
        predicate = (ENDED if p.get("ended") else TOPIC) + o["topic"]
        same = self.db.execute("SELECT fact_id, object_value FROM knowledge_records WHERE subject_id=? AND predicate=? AND status IN ('active','needs_review','disputed')", (who, predicate)).fetchall()
        match = next((fid for fid, value in same if normalize_name(value) == normalize_name(o["thing"]) or content_stems(value) == content_stems(o["thing"])), None)
        method, version = "filing " + FILING_VERSION, self.writer_name
        if match:  # the same thing filed again: one more sentence behind it, not a second record
            for sid in o["sids"]: self.kb.add_evidence(FactEvidence.create(match, sid, extraction_method=method, extraction_version=version))
            return dict(fact_id=match, linked=True)
        meta = dict(retention=KINDS[o["kind"]], topic=tidy(p.get("topic")), time_phrase=o["phrase"], said_on=o["said_day"], filed_by=self.writer_name,
                    checked_by="rules " + FILING_VERSION + (" + " + self.checker_name if self.checker is not None else ""), check=o["check"])
        fact = KnowledgeFact.create(who, predicate, "text", o["kind"], "explicit", object_value=o["thing"], validation_status="accepted",
                                    valid_from=o["when"], metadata=meta, note=o["fact"][:300])
        saved = self.kb.add(FactWithEvidence(fact, tuple(FactEvidence.create(fact.fact_id, sid, extraction_method=method, extraction_version=version) for sid in dict.fromkeys(o["sids"]))))
        return dict(fact_id=saved.fact.fact_id, linked=False)

    def _review(self, p, o):
        name = o["who"].name if o.get("who") is not None else tidy(p.get("who"))
        candidate = dict(kind="filed_candidate", subject=name, predicate=(ENDED if p.get("ended") else TOPIC) + o["topic"], fact_type=o["kind"], object=o["thing"],
                         fact=o["fact"], time=o["phrase"], when=o["when"], ended=bool(p.get("ended")), source_sids=o["sids"], filed_by=self.writer_name, check=o["check"])
        return self.kb.add_review(json.dumps(candidate), list(o["reasons"]), list(o["sids"]))

    def _drop_old_reviews(self, chat):
        """Filing a chat again replaces its earlier review items, so they are not queued twice."""
        sids = {r[0] for r in self.db.execute("SELECT sid FROM sentences WHERE chat=?", (chat,))}
        for item in self.kb.reviews():
            c = json.loads(item["candidate"])
            if c.get("kind") == "filed_candidate" and set(c.get("source_sids", [])) <= sids: self.kb.drop_review(item["review_id"])

    def approve(self, review_id, entity_name=None):
        """A person accepts a held proposal. The code checks run again; the checker is not asked again."""
        item = self.kb.review(review_id)
        if item is None: raise RecordError("no such review item")
        c = json.loads(item["candidate"])
        if c.get("kind") != "filed_candidate": raise RecordError("not a filed proposal")
        rows = {f"L{n}": r for n, r in enumerate(self.db.execute(f"SELECT sid, speaker, date, text FROM sentences WHERE sid IN ({','.join('?' * len(c['source_sids']))}) ORDER BY sid", c["source_sids"]).fetchall(), 1)}
        if len(rows) != len(c["source_sids"]): raise RecordError("a cited sentence no longer exists")
        p = dict(who=entity_name or c["subject"], kind=c["fact_type"], topic=c["predicate"].split(":", 1)[1], thing=c["object"], fact=c["fact"], time=c["time"], ended=c["ended"], lines=list(rows))
        checker, self.checker = self.checker, None
        try:
            o = self.decide(p, {k: tuple(v) for k, v in rows.items()})
        finally:
            self.checker = checker
        if o["verdict"] == DROPPED: raise RecordError("; ".join(o["reasons"]))
        o["check"] = c.get("check")
        stored = self._store(p, o)
        self.kb.drop_review(review_id)
        self._cache = None
        return stored

    def _log(self, chat, status, secs, counts, reasons):
        self.db.execute("INSERT OR REPLACE INTO filings VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (chat, status, self.writer_name, self.checker_name, round(secs, 3), counts["proposed"], counts["stored"],
                        counts["linked"], counts["review"], counts["dropped"], counts["unreadable"], json.dumps(reasons), FILING_VERSION))
        self.db.commit()

    # ---------- reading back ----------
    def records(self):
        """Filed records that hold: active, accepted, with their subject's name."""
        rows = self.db.execute("""SELECT r.fact_id, e.canonical_name, r.predicate, r.object_value, r.fact_type, r.valid_from, r.note, r.metadata
                                  FROM knowledge_records r JOIN entities e ON e.entity_id = r.subject_id
                                  WHERE r.status='active' AND r.validation_status='accepted' AND (r.predicate LIKE 'topic:%' OR r.predicate LIKE 'ended:%')
                                  ORDER BY r.recorded_at, r.fact_id""").fetchall()
        return [dict(fact_id=r[0], who=r[1], predicate=r[2], thing=r[3], kind=r[4], when=r[5], note=r[6], meta=json.loads(r[7])) for r in rows]

    def find(self, question, limit=10, view=None):
        """Filed records for a question, best first. Records sharing a topic with the best ones follow them, so a question
        such as 'how many instruments' sees the whole collection, not one item."""
        import engram_chat as E
        recs = self.records()
        if not recs: return []
        texts = [f"{r['who']} {r['meta'].get('topic', '')} {r['note']}" for r in recs]
        key = (len(recs), recs[-1]["fact_id"])
        s = self.m.search
        if self._cache is None or self._cache[0] != key:
            w = [E.words(t) for t in texts]
            df = Counter(x for ws in w for x in set(ws))
            vectors = s._embed(texts) if s is not None and getattr(s, "N", 0) else None
            self._cache = (key, w, df, sum(map(len, w)) / max(1, len(w)), vectors)
        _, w, df, avg, vectors = self._cache
        qw, n = set(E.words(question)), len(recs)
        score = [0.0] * n
        if vectors is not None:
            sim = vectors @ s._embed([question])[0]
            score = [float(x) for x in sim]
        for i, ws in enumerate(w):
            b = 0.0
            for x in qw & set(ws):
                tf = ws.count(x); b += math.log(1 + (n - df[x] + .5) / (df[x] + .5)) * tf * 2.2 / (tf + 1.2 * (.5 + .5 * len(ws) / avg))
            score[i] += b / (1 + b)
        if view is not None:  # P2.5: hidden for this question, or moved up a little for it
            for i, r in enumerate(recs):
                if view.is_hidden("fact:" + r["fact_id"], question): score[i] = 0.0
                elif score[i] > 0: score[i] += view.boost("fact:" + r["fact_id"], question)
        order = sorted(range(n), key=lambda i: -score[i])
        order = [i for i in order if score[i] > 0]
        out, seen = [], set()
        for i in order[:3]:
            for j in [i] + [j for j in order if (recs[j]["who"], recs[j]["predicate"].split(":", 1)[1]) == (recs[i]["who"], recs[i]["predicate"].split(":", 1)[1])]:
                if j not in seen: seen.add(j); out.append(recs[j])
        for i in order[3:]:
            if i not in seen: seen.add(i); out.append(recs[i])
        return out[:limit]

    def line(self, r, caution=""):
        label = KIND_WORDS.get(r["kind"], r["kind"])
        parts = ["Filed fact", label + (", ended" if r["predicate"].startswith(ENDED) else "")]
        if r["meta"].get("said_on"): parts.append("said " + _pretty(r["meta"]["said_on"]))
        if r["when"]: parts.append(("planned for " if r["kind"] == "plan" else "happened " if r["kind"] == "event" or r["predicate"].startswith(ENDED) else "since ") + _pretty(r["when"]))
        if caution: parts.append("caution: " + caution)
        return "[" + " | ".join(parts) + "] " + r["note"]

    def shown(self, question, tokens=300, view=None):
        """Filed facts for the answer model to read, within a token budget, and the references of those shown."""
        out, refs, used = [], [], 0
        for r in self.find(question, 40, view):
            ln = self.line(r, view.caution("fact:" + r["fact_id"]) if view is not None else "")
            t = max(1, len(ln) // 4)
            if used + t > tokens: break
            out.append(ln); refs.append("fact:" + r["fact_id"]); used += t
        return "\n".join(out), refs

    def text(self, question, tokens=300, view=None):
        return self.shown(question, tokens, view)[0]
