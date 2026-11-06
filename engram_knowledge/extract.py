"""P2.6: structured fact extraction. Writes who, what, which, qualifiers and time from one sentence.

Extraction only proposes. It never decides what is stored: every field goes to the fact validator, which checks it against the
original sentence (a separate operation). Writers are interchangeable; this file holds the fixed-rule writer, which needs no model
and covers only plain first-person sentences ("I adopted…", "I moved to…"). Anything it does not recognise gives no fact, not a guess.
"""
import calendar
import datetime
import re
from dataclasses import dataclass, field

from .normalizer import MONTHS, time_value

PREDICATES = ("owns", "lives_in", "works_at", "married_to", "parent_of", "friend_of", "member_of", "job_title", "status", "likes", "dislikes", "hobby", "plans",
              "adopted", "bought", "sold", "lost", "gave_away", "moved_to", "started_job", "left_job", "got_married", "divorced", "joined", "left_group",
              "started_hobby", "stopped_hobby", "attended", "visited", "completed")


@dataclass(frozen=True)
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    qualifiers: tuple = ()          # (role, words) pairs copied from the sentence: name, from, place, with
    time_text: str | None = None    # the time phrase exactly as said
    time_value: str | None = None   # YYYY, YYYY-MM or YYYY-MM-DD worked out from when it was said; None if it cannot be
    source_sid: int | None = None
    span: tuple | None = None       # where the sentence sits in the stored message
    writer: str = ""
    writer_version: str = ""

    def to_dict(self):
        return dict(subject=self.subject, predicate=self.predicate, object=self.object, qualifiers=dict(self.qualifiers), time_text=self.time_text, time_value=self.time_value,
                    source_sid=self.source_sid, span=list(self.span) if self.span else None, writer=self.writer, writer_version=self.writer_version)


# ---------- time ----------
_MONTH = "|".join(MONTHS)
_NUM = dict(a=1, an=1, one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10, couple=2)
TIME = re.compile(
    rf"\b(yesterday|today|tonight|this morning|this afternoon|this evening|last night|last weekend|last week|last month|last year|next week|next month|next year|next weekend|recently|"
    rf"(?:last|this) (?:spring|summer|fall|autumn|winter)|"
    rf"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|a couple of|a few) (?:days?|weeks?|months?|years?) ago|"
    rf"(?:in |on |since )?(?:\d{{1,2}}(?:st|nd|rd|th)? (?:{_MONTH})|(?:{_MONTH}) \d{{1,2}}(?:st|nd|rd|th)?)(?:,? \d{{4}})?|"
    rf"in (?:{_MONTH})(?: \d{{4}})?|in \d{{4}})\b", re.I)


def said_day(said_on):
    """The day a sentence was said, from Engram's '4:22 pm on 13 October, 2023' or YYYY-MM-DD. None if unknown."""
    if not said_on: return None
    v = time_value(re.sub(r"^.*? on ", "", said_on)) or time_value(said_on)
    return datetime.date.fromisoformat(v) if v and len(v) == 10 else None


def resolve_time(phrase, said_on):
    """A time phrase as a date, at the precision the phrase allows. Never invented: anything unclear gives None."""
    if not phrase: return None
    p = phrase.lower().strip()
    day = said_day(said_on)
    explicit = time_value(re.sub(r"^(in|on|since) ", "", p).replace(",", " "))
    if explicit and re.search(r"\d{4}", p): return explicit
    if day is None: return None
    if p in ("today", "tonight", "this morning", "this afternoon", "this evening"): return day.isoformat()
    if p in ("yesterday", "last night") or re.fullmatch(r"yesterday (morning|afternoon|evening)", p): return (day - datetime.timedelta(days=1)).isoformat()
    if p == "last month": return f"{day.year - (day.month == 1)}-{(day.month - 2) % 12 + 1:02d}"
    if p == "last year": return str(day.year - 1)
    if p == "next year": return str(day.year + 1)
    if p == "next month": return f"{day.year + (day.month == 12)}-{day.month % 12 + 1:02d}"
    m = re.fullmatch(r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|a couple of) (day|month|year)s? ago", p)
    if m:
        n = int(m[1]) if m[1].isdigit() else _NUM[m[1].split()[-1] if m[1] != "a couple of" else "couple"]
        if m[2] == "day": return (day - datetime.timedelta(days=n)).isoformat()
        if m[2] == "year": return str(day.year - n)
        total = day.year * 12 + day.month - 1 - n
        return f"{total // 12}-{total % 12 + 1:02d}"
    m = re.fullmatch(rf"(?:in |on |since )?(?:(\d{{1,2}})(?:st|nd|rd|th)? ({_MONTH})|({_MONTH}) (\d{{1,2}})(?:st|nd|rd|th)?)", p)
    if m:  # a day and month with no year: the most recent one on or before the day it was said
        month, d = MONTHS.index(m[2] or m[3]) + 1, int(m[1] or m[4])
        year = day.year if (month, d) <= (day.month, day.day) else day.year - 1
        if d > calendar.monthrange(year, month)[1]: return None
        return f"{year}-{month:02d}-{d:02d}"
    m = re.fullmatch(rf"in ({_MONTH})", p)
    if m:
        month = MONTHS.index(m[1]) + 1
        return f"{day.year if month <= day.month else day.year - 1}-{month:02d}"
    return None  # last week, recently, seasons: not a date


# ---------- the fixed-rule writer ----------
_OPENERS = re.compile(r"^(?:(?:so|well|yeah|yes|oh|actually|also|and|but|plus|anyway|honestly|btw|by the way|guess what|you know what)[,!]?\s+)+", re.I)
_ADVERB = r"(?:(?:just|recently|finally|also|really|actually|already|even|officially|now|currently|still)\s+)*"
_I = r"(?:i|we)"
_DET = r"(?:(?:a|an|the|my|our|some|this|that|these|those|another|new|first)\s+)*"
_STOP = re.compile(r"\s*(?:[,.;:!?()]|\s(?:and|but|because|so|which|that|who|when|while|since|as|after|before|until|where|if|for|with|from|in|on|at|to|by|about|called|named|"
                   + TIME.pattern[2:-2] + r")\b|$)", re.I)
_EVENT_WORDS = re.compile(r"\b(concert|festival|class|classes|workshop|conference|game|match|show|party|wedding|event|meetup|seminar|exhibition|exhibit|fair|parade|retreat|camp|course|lecture|session|tournament|race|marathon|ceremony|gala|play|musical|recital|convention|summit|webinar|talk)\b", re.I)
_PET = r"(?:dog|dogs|cat|cats|puppy|pup|kitten|pet|pets|bird|parrot|rabbit|bunny|hamster|fish|turtle|horse|snake|lizard|guinea pig)"
_OWNED = rf"(?:{_PET}|car|bike|bicycle|motorcycle|house|home|apartment|flat|boat|truck|van|camera|guitar|piano|laptop)"

# (predicate, pattern). The pattern ends where the object starts. Order matters: the first match wins for a span of text.
RULES = [
    ("adopted", rf"\b{_I} {_ADVERB}(?:have |had )?adopted {_DET}"),
    ("bought", rf"\b{_I} {_ADVERB}(?:have |had )?(?:bought|purchased|got myself) {_DET}"),
    ("sold", rf"\b{_I} {_ADVERB}(?:have )?sold {_DET}"),
    ("gave_away", rf"\b{_I} {_ADVERB}gave away {_DET}"),
    ("lost", rf"\b{_I} {_ADVERB}(?:have )?lost {_DET}"),
    ("moved_to", rf"\b{_I} {_ADVERB}(?:have |had )?(?:moved|relocated) (?:back )?to {_DET}"),
    ("lives_in", rf"\b{_I} {_ADVERB}(?:live|lived|am living|'m living|reside) in {_DET}"),
    ("started_job", rf"\b{_I} {_ADVERB}(?:started|began|got) (?:a |my )?(?:new )?(?:job|position|role|work|working) (?:at|with|for) {_DET}"),
    ("started_job", rf"\b{_I} {_ADVERB}(?:got hired|was hired|have been hired) (?:at|by) {_DET}"),
    ("left_job", rf"\b{_I} {_ADVERB}(?:quit|left|resigned from) (?:my (?:job|position|role) at )?{_DET}(?=\S+ (?:job|company)\b)"),
    ("works_at", rf"\b{_I} {_ADVERB}(?:work|am working|'m working) (?:at|for) {_DET}"),
    ("job_title", rf"\b{_I} {_ADVERB}(?:work as|am working as|'m working as) {_DET}"),
    ("married_to", rf"\b{_I}(?: am|'m) married to {_DET}"),
    ("member_of", rf"\b{_I}(?: am|'m) {_ADVERB}(?:a member of|part of|on) {_DET}(?=[\w ]*\b(?:team|club|group|band|choir|league|society|association|committee|board)\b)"),
    ("joined", rf"\b{_I} {_ADVERB}(?:have )?(?:joined|signed up for) {_DET}"),
    ("left_group", rf"\b{_I} {_ADVERB}(?:left|quit) {_DET}(?=[\w ]*\b(?:team|club|group|band|choir|league|society|association|committee)\b)"),
    ("attended", rf"\b{_I} {_ADVERB}(?:attended|went to|was at|were at|took part in|participated in) {_DET}(?=[\w' -]*{_EVENT_WORDS.pattern})"),
    ("visited", rf"\b{_I} {_ADVERB}(?:visited|went to|traveled to|travelled to|flew to|toured) {_DET}(?-i:(?=[A-Z]))"),
    ("completed", rf"\b{_I} {_ADVERB}(?:have )?(?:finished|completed|graduated from|ran|run) {_DET}(?=[\w' -]*\b(?:race|marathon|course|degree|program|programme|book|project|training|half[- ]marathon|5k|10k|triathlon|school|college|university)\b)"),
    ("started_hobby", rf"\b{_I} {_ADVERB}(?:have )?(?:started|took up|taken up|began|picked up) {_DET}(?=\w+ing\b)"),
    ("stopped_hobby", rf"\b{_I} {_ADVERB}(?:stopped|quit|gave up) {_DET}(?=\w+ing\b)"),
    ("got_married", rf"\b{_I} {_ADVERB}got married\b"),
    ("divorced", rf"\b{_I} {_ADVERB}(?:got divorced|divorced) {_DET}"),
    ("dislikes", rf"\b{_I} {_ADVERB}(?:hate|dislike|can't stand|cannot stand|don't like|do not like) {_DET}"),
    ("hobby", rf"\b{_I} {_ADVERB}(?:love|like|enjoy|adore) (?=\w+ing\b)"),
    ("likes", rf"\b{_I} {_ADVERB}(?:love|like|enjoy|adore) {_DET}(?!\w+ing\b)(?!to\b|it\b|that\b|this\b|them\b|you\b|your\b|how\b|when\b|the idea\b|being\b)"),
    ("plans", rf"\b{_I}(?: am|'m|'re| are) {_ADVERB}(?:planning|going|about) to (?=\w)"),
    ("owns", rf"\b{_I} {_ADVERB}(?:have|own|got) {_DET}(?:\w+ )?(?={_OWNED}\b)"),
]
# "my dog Toby", "my son Jake", "my friend Ana": a possessive with a name.
POSSESSIVE = [
    ("owns", rf"\b(?i:my) (?i:{_ADVERB}(?:new )?({_OWNED}))(?:,)? ([A-Z][a-z]+)\b"),
    ("parent_of", r"\b(?i:my (?:son|daughter|kid|child|baby boy|baby girl|little boy|little girl)) ([A-Z][a-z]+)\b"),
    ("friend_of", r"\b(?i:my (?:best )?friend) ([A-Z][a-z]+)\b"),
]
_QUALIFIERS = [("name", r"\b(?:called|named) ([A-Z][\w'-]*(?: [A-Z][\w'-]*)?)"), ("from", r"\bfrom (?:the |a |my )?([\w' -]+?)(?=[,.;!?]| (?:and|but|last|yesterday|in|on|which|who)\b|$)"),
               ("with", r"\bwith (?:my |our |the )?([\w' -]+?)(?=[,.;!?]| (?:and|but|last|yesterday|in|on|at)\b|$)"), ("place", r"\b(?:in|at) ((?:[A-Z][\w'-]*)(?: [A-Z][\w'-]*){0,2})")]
_BAD_OBJECT = re.compile(r"^(?:it|this|that|them|these|those|one|something|anything|everything|lot|lots|bit|way|time|thing|things|so|very|really|much|more|too|myself|yourself|you|him|her|us)$", re.I)
_NEGATED = re.compile(r"\b(?:not|never|no longer|didn't|don't|doesn't|haven't|hasn't|wasn't|won't|can't|cannot)\b", re.I)


def _object(rest):
    m = _STOP.search(rest)
    words = rest[:m.start() if m else len(rest)].strip(" '\"").split()
    words = words[:6]
    while words and words[-1].lower() in ("a", "an", "the", "my", "our", "some", "of", "to", "and", "really", "so", "very"): words.pop()
    return " ".join(words)


class RuleWriter:
    """Fixed rules, no model. Plain first-person statements only; anything else gives no fact."""
    name, version = "rules", "1"

    def extract(self, speaker, sentence, said_on="", sid=None, span=None):
        text = _OPENERS.sub("", sentence.strip())
        out, used = [], []
        times = [m.group(0) for m in TIME.finditer(sentence)]
        time_text = times[0] if len(times) == 1 else None  # two time phrases: which one belongs to the fact is not clear
        for predicate, pattern in RULES:
            for m in re.finditer(pattern, text, flags=re.I):
                if any(a <= m.start() < b for a, b in used): continue
                before = text[max(0, m.start() - 25):m.end()]
                if _NEGATED.search(before) and predicate not in ("dislikes",): continue  # "I never adopted…" is not "I adopted"
                obj = "" if predicate == "got_married" else _object(text[m.end():])
                if predicate != "got_married" and (not obj or _BAD_OBJECT.match(obj)): continue
                if predicate == "plans": obj = _object(text[m.end():])
                used.append((m.start(), m.end() + len(obj)))
                out.append(self._fact(speaker, predicate, obj or "married", text[m.end():], time_text, said_on, sid, span))
        for predicate, pattern in POSSESSIVE:
            for m in re.finditer(pattern, text):
                if any(a <= m.start() < b for a, b in used): continue
                if predicate == "owns": obj, name = m.group(1), m.group(2)
                else: obj, name = m.group(1), None
                used.append(m.span())
                fact = self._fact(speaker, predicate, obj, "", time_text, said_on, sid, span)
                if name: fact = ExtractedFact(**{**fact.__dict__, "qualifiers": (("name", name),) + tuple(q for q in fact.qualifiers if q[0] != "name")})
                out.append(fact)
        return out

    def _fact(self, speaker, predicate, obj, rest, time_text, said_on, sid, span):
        qualifiers = []
        for role, pattern in _QUALIFIERS:
            q = re.search(pattern, rest)
            if q and q.group(1).strip() and q.group(1).strip().lower() != obj.lower() and not TIME.fullmatch(q.group(1).strip()) and q.group(1).split()[0].lower() not in MONTHS:
                qualifiers.append((role, q.group(1).strip()))
        return ExtractedFact(speaker, predicate, obj, tuple(qualifiers), time_text, resolve_time(time_text, said_on), sid, tuple(span) if span else None, self.name, self.version)


# ---------- a model as the writer ----------
WRITER_PROMPT = """Read one sentence said by {speaker}. Write the lasting facts it states about {speaker}, one per line, in exactly this form:
subject | predicate | object | details | time

subject: {speaker}
predicate: exactly one of: {predicates}
object: the shortest exact words from the sentence naming the thing.
details: optional, copied from the sentence, as name=...; from=...; with=...; place=... Leave empty if none.
time: the exact time phrase from the sentence, such as yesterday or last month. Leave empty if none.

Only facts stated plainly about {speaker}, not about other people, and not questions, wishes or maybes. If there are none, write NONE. Write nothing else.

Example sentence by Asha: "I adopted a beagle called Pixel from the shelter yesterday."
Asha | adopted | beagle | name=Pixel; from=the shelter | yesterday

Sentence: "{text}"
The message it came from, for context only (take no facts from other sentences): "{message}"

Facts:
"""


def writer_prompt(speaker, text, message=""):
    return WRITER_PROMPT.format(speaker=speaker, text=text, message=message or text, predicates=", ".join(PREDICATES))


def parse_writer_reply(reply, speaker, said_on="", sid=None, span=None, writer="model", version="1"):
    """A model's reply as proposed facts. The date is worked out here from the time phrase, never taken from the model.
    Lines not in the form asked, or with a predicate not in the list, give no fact. Nothing is decided here: the validator checks every field."""
    out, unread = [], []
    for line in (reply or "").splitlines():
        line = line.strip().strip("*-• ")
        if not line or line.upper().startswith("NONE") or set(line) <= set("`") or line.lower().rstrip(":") == "facts": continue
        parts = [p.strip().strip('"') for p in line.split("|")]
        if len(parts) < 3 or parts[1].lower().replace(" ", "_") not in PREDICATES or not parts[2]:
            unread.append(line[:200]); continue
        details = []
        for item in (parts[3] if len(parts) > 3 else "").split(";"):
            role, _, words = item.partition("=")
            if role.strip().lower() in ("name", "from", "place", "with") and words.strip(): details.append((role.strip().lower(), words.strip()))
        when = parts[4] if len(parts) > 4 and parts[4] and parts[4].lower() not in ("none", "unknown", "-") else None
        out.append(ExtractedFact(parts[0] or speaker, parts[1].lower().replace(" ", "_"), parts[2], tuple(details), when, resolve_time(when, said_on), sid, tuple(span) if span else None, writer, version))
    return out, unread
