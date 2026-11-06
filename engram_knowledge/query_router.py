"""Decides from a question's words which path can answer it. Fixed rules, and a question it cannot read goes to ordinary memory."""
import re

from .normalizer import MONTHS, time_value, tokens

SLOT_WORDS = dict(
    works_at="work works working worked employer employers employed job jobs company hospital office".split(),
    lives_in="live lives living lived home city based reside".split(),
    job_title="title role position profession".split(),
    married_to="married spouse wife husband".split(),
    owns="own owns owned pet pets dog cat car".split(),
    hobby="hobby hobbies".split(),
    member_of="member group club".split(),
    likes="like likes favourite favorite love loves".split(),
)
MONTH = "|".join(MONTHS)
DATE = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTH}),?\s+\d{{4}}|\b(?:{MONTH})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\b(?:{MONTH})\s+\d{{4}}|\b\d{{4}}-\d{{2}}(?:-\d{{2}})?|\b(?:19|20)\d{{2}}\b", re.I)


class QueryRouter:
    def extract_date(self, question):
        m = DATE.search(question or "")
        return time_value(m.group(0).replace(",", " ")) if m else None

    def slot(self, question):
        words = set(tokens(question))
        return next((slot for slot, cues in SLOT_WORDS.items() if words & set(cues)), None)

    def classify(self, question):
        q = (question or "").lower()
        if re.search(r"\b(conflict|conflicting|contradict|disagree|disputed?)\b", q): return "conflict"
        if re.search(r"\b(evidence|source|sources|who said|how do you know|supports?|what makes you)\b", q): return "evidence"
        if self.extract_date(q): return "historical_state"
        if re.search(r"\bwhen did\b|\bsince when\b|\bwhat date\b|\bhow long ago\b", q): return "change"
        if re.search(r"\b(original|first|initial|earliest|previous|former|before)\b", q): return "original"
        if re.search(r"\b(all|history|every|over time|past)\b.*\b(job|jobs|work|employer|employers|home|homes|lived|worked)\b|\bwhere (all )?has\b", q): return "history"
        if re.search(r"\b(current|currently|now|today|present|presently|these days|still|at the moment)\b", q): return "current_state"
        if self.slot(q) and re.search(r"\b(does|do|is|are)\b", q): return "current_state"
        if self.slot(q): return "combined"
        return "general_memory"
