"""Finds which entity a mention refers to. When it is not certain, it says so and never guesses."""
import re
from dataclasses import dataclass

from engram_records import normalize_name, tidy

FIRST_PERSON = {"i", "me", "my", "myself", "user", "the user", "the speaker", "speaker"}
UNCLEAR = {"he", "she", "they", "them", "him", "her", "his", "their", "it", "we", "us", "our", "you", "your", "someone", "somebody", "friend", "my friend", "the person", "this person", "narrator", "the narrator"}


@dataclass(frozen=True)
class Resolution:
    status: str  # known, new, ambiguous, unresolved or absent
    name: str = ""
    entity_id: str | None = None
    why: str = ""


def _has_name(name, sentences):
    pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.I)
    return any(normalize_name(s.speaker) == normalize_name(name) or pattern.search(s.text) for s in sentences)


class EntityResolver:
    def __init__(self, kb):
        self.kb = kb

    def _similar(self, name):
        """Entities that share a word with this name. These could be the same person, so a person must decide."""
        words = set(normalize_name(name).split())
        rows = self.kb.db.execute("SELECT entity_id, normalized_name FROM entities").fetchall() + self.kb.db.execute("SELECT entity_id, normalized_alias FROM entity_aliases").fetchall()
        return sorted({r[0] for r in rows if words & set(r[1].split())})

    def resolve(self, mention, sentences):
        name, key = tidy(mention), normalize_name(mention)
        if not key: return Resolution("unresolved", why="no subject")
        if key in FIRST_PERSON:
            speakers = {tidy(s.speaker) for s in sentences}
            if len(speakers) != 1: return Resolution("ambiguous", why="first person across different speakers")
            return self.resolve(speakers.pop(), sentences)
        if key in UNCLEAR: return Resolution("unresolved", why=f"'{name}' does not say who")
        known = self.kb.resolve(name)
        if known: return Resolution("known", self.kb.entity(known).canonical_name, known)
        if not _has_name(name, sentences): return Resolution("absent", why=f"'{name}' is not in its source")
        if self._similar(name): return Resolution("ambiguous", name, why=f"'{name}' may be someone already known")
        return Resolution("new", name)
