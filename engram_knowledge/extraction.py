"""Turns a helper model's reply into candidates. A candidate is only a proposal; FactValidator decides."""
import re

import engram_facts as F

from .fact_validator import Candidate
from .normalizer import predicate

STRUCTURED_PROMPT = """Below is one conversation. Each line starts with its id. Write the lasting knowledge in it as structured lines.

Use exactly this form, one per line, at most 12 lines:
ID | subject | word | object | time

ID: copy the id of the line that states it.
subject: a person's full name as written. Use the speaker's name for I.
word: owns, lives_in, works_at, married_to, parent_of, friend_of, member_of, job_title, status, likes, dislikes, hobby, plans, number, adopted, bought, sold, lost, gave_away, moved_to, started_job, left_job, got_married, divorced, joined, left_group, started_hobby, stopped_hobby, attended, visited, completed
object: the name, place or short value, using the same words as the line.
time: the real date if stated, otherwise unknown.

Write only what is stated. Write nothing else.

Conversation:
{chat}

Lines:
"""


def parse_reply(reply, ids, method="helper", version="1"):
    """ids: {line id: sentence id}. Returns candidates, and the lines that were not in the form asked."""
    out, unread = [], []
    for line in (reply or "").splitlines():
        line = line.strip()
        if not line: continue
        parts = [p.strip().strip('"*') for p in line.strip("*-• ").split("|")]
        cited = F.LINE_ID.findall(parts[0])
        if len(parts) < 4 or not cited:
            unread.append(line[:300])
            continue
        when = parts[4] if len(parts) > 4 and parts[4].lower() not in ("unknown", "none", "n/a", "") else ""
        out.append(Candidate(parts[1], predicate(parts[2]), parts[3], tuple(ids.get(c) for c in cited), when, method, version, line[:300]))
    return out, unread
