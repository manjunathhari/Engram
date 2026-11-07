"""How each kind of observation changes a state. Written out in full, with nothing inferred from the wording.

Two kinds of slot exist. The fixed ones below come from P2.6's relation list. Topic slots come from whole-chat filing
(filing.py): a fact filed under topic "pet" has the predicate "topic:pet", and a statement that it stopped ("sold my car")
has "ended:car", which closes the slot "topic:car". A topic slot holds several values at once unless its topic is one that
can only have one value at a time (SINGLE_TOPICS), so that a new home or job ends the old one as works_at does.
"""
import re

SINGLE = ("works_at", "lives_in", "job_title", "status", "married_to")  # one value at a time, so two can conflict
MULTI = ("owns", "hobby", "member_of", "likes", "dislikes", "friend_of", "parent_of")  # several at once
OPENS = dict(works_at="works_at", started_job="works_at", lives_in="lives_in", moved_to="lives_in", job_title="job_title", status="status", married_to="married_to", got_married="married_to",
             owns="owns", bought="owns", adopted="owns", hobby="hobby", started_hobby="hobby", member_of="member_of", joined="member_of", likes="likes", dislikes="dislikes", friend_of="friend_of", parent_of="parent_of")
CLOSES = dict(left_job="works_at", divorced="married_to", sold="owns", lost="owns", gave_away="owns", stopped_hobby="hobby", left_group="member_of")

TOPIC, ENDED = "topic:", "ended:"
SINGLE_TOPICS = frozenset({"job", "employer", "workplace", "job title", "role", "home", "address", "city", "hometown", "partner", "spouse", "relationship status", "marital status"})


def topic_key(topic):
    """One spelling for a topic, so "Pets" and "pet" are the same collection. Lower case, single spaces, a plain plural dropped."""
    t = re.sub(r"[^a-z0-9 ]+", " ", str(topic or "").lower())
    t = " ".join(t.split())
    words = t.split(" ")
    last = words[-1] if t else ""
    if last.endswith("ies") and len(last) > 4: last = last[:-3] + "y"
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 3: last = last[:-1]
    return " ".join(words[:-1] + [last]) if t else ""


def slot_of(predicate):
    p = predicate or ""
    if p.startswith(TOPIC): return p if p[len(TOPIC):] else None
    if p.startswith(ENDED): return TOPIC + p[len(ENDED):] if p[len(ENDED):] else None
    return OPENS.get(p) or CLOSES.get(p)


def closes(predicate):
    """Whether this predicate says something stopped being true."""
    return predicate in CLOSES or (predicate or "").startswith(ENDED)


def is_single(slot):
    """Whether a slot holds only one value at a time."""
    return slot in SINGLE or (bool(slot) and slot.startswith(TOPIC) and slot[len(TOPIC):] in SINGLE_TOPICS)
