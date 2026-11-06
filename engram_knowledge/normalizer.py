"""Plain text handling for candidate facts. No model and no guessing."""
import re

import engram_chat as E

STOP = set("a an the and or of to in on at for with from by my your his her their our its this that these those some any is are was were be been am i me you he she they we it".split())
NUMBER_WORDS = dict(zero=0, one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10, eleven=11, twelve=12, twenty=20, thirty=30, forty=40, fifty=50, hundred=100)
MONTHS = "january february march april may june july august september october november december".split()
PREDICATE_TYPES = dict(
    owns="relationship", lives_in="attribute", works_at="relationship", married_to="relationship", parent_of="relationship", friend_of="relationship", member_of="relationship",
    job_title="attribute", status="attribute", likes="preference", dislikes="preference", hobby="preference", plans="plan", number="attribute",
    adopted="event", bought="event", sold="event", lost="event", gave_away="event", moved_to="event", started_job="event", left_job="event", got_married="event", divorced="event",
    joined="event", left_group="event", started_hobby="event", stopped_hobby="event", attended="event", visited="event", completed="event")
# Words a sentence would normally use if it states this. Absent words do not disprove a fact, they send it for review.
PREDICATE_CUES = dict(
    owns="own have has got my bought adopted", lives_in="live living home moved based from", works_at="work working job employed office company", married_to="married wife husband spouse wedding",
    parent_of="son daughter kid child children mom dad mother father parent", friend_of="friend pal buddy", member_of="member joined group club team part",
    job_title="work working job career profession employed title role as", status="am im is single married divorced retired student pregnant", likes="like love enjoy favorite favourite fan into passion", dislikes="dislike hate avoid not",
    hobby="hobby enjoy love play playing like free time into", plans="plan planning going will want hope thinking next soon gonna", adopted="adopt", bought="bought buy purchase got", sold="sold sell", lost="lost lose",
    gave_away="gave give donated", moved_to="moved move moving relocated", started_job="started start new job hired joined", left_job="left quit resign retired laid", got_married="married wedding", divorced="divorce",
    joined="joined join signed member", left_group="left quit", started_hobby="started start took began learning trying", stopped_hobby="stopped quit gave", attended="attended went go going was at", visited="visited visit went trip travelled traveled", completed="completed finished done graduated")


def tokens(text):
    out = []
    for w in re.findall(r"[a-z0-9$%][a-z0-9$%']*", (text or "").lower().replace("’", "'")):
        out.append(str(NUMBER_WORDS[w]) if w in NUMBER_WORDS else w.replace("'s", "").replace("'", ""))
    return out


def stem(word):
    if word.endswith("ies") and len(word) > 4: return word[:-3] + "y"
    for tail, least in (("ing", 4), ("ed", 4), ("es", 4), ("s", 4)):
        if word.endswith(tail) and len(word) - len(tail) >= least - 1:
            word = word[:-len(tail)]
            break
    return word[:-1] if len(word) > 3 and word[-1] == "e" else word


def stems(text):
    return [stem(w) for w in tokens(text)]


def content_stems(text):
    return [stem(w) for w in tokens(text) if w not in STOP]


def predicate(word):
    return re.sub(r"[\s\-]+", "_", (word or "").strip().lower()).strip("_")


def time_value(text):
    """A stated date as YYYY, YYYY-MM or YYYY-MM-DD. None when it is not a plain date."""
    t = (text or "").lower().replace(",", " ")
    m = re.search(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b", t)
    if m: return m.group(0)
    months = "|".join(MONTHS)
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({months})\s+(\d{{4}})\b", t) or None
    if m: return f"{m[3]}-{MONTHS.index(m[2]) + 1:02d}-{int(m[1]):02d}"
    m = re.search(rf"\b({months})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+(\d{{4}})\b", t)
    if m: return f"{m[3]}-{MONTHS.index(m[1]) + 1:02d}-{int(m[2]):02d}"
    m = re.search(rf"\b({months})\s+(\d{{4}})\b", t)
    if m: return f"{m[2]}-{MONTHS.index(m[1]) + 1:02d}"
    m = re.fullmatch(r"\s*(\d{4})\s*", t)
    return m[1] if m else None


def time_values_in(text, date_said):
    """Every date a sentence supports: the day it was said, and any date it states or implies."""
    found = set()
    day = time_value(re.sub(r"^.*? on ", "", date_said or ""))
    if day:
        found.add(day)
    shown = E.with_dates(text, date_said) if date_said else text
    months = "|".join(MONTHS)
    for m in re.finditer(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{months}),?\s+\d{{4}}|\b(?:{months})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\b(?:{months})\s+\d{{4}}|\b\d{{4}}-\d{{2}}(?:-\d{{2}})?|\b(?:19|20)\d{{2}}\b", shown, re.I):
        v = time_value(m.group(0))
        if v: found.add(v)
    return found
