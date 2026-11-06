"""Decides whether a candidate fact may be stored. Fixed rules only: a language model never judges its own output."""
import json
import re
from dataclasses import dataclass

from engram_records import Entity, FactEvidence, FactWithEvidence, KnowledgeBase, KnowledgeFact, RecordError, normalize_name, tidy

from .entity_resolver import EntityResolver, Resolution
from .normalizer import PREDICATE_CUES, PREDICATE_TYPES, content_stems, predicate, stem, stems, time_value, time_values_in, tokens

ACCEPTED, UNCERTAIN, REJECTED = "accepted", "uncertain", "rejected"
RULES_VERSION = "1"
NEGATION = set("not no never without none cannot cant dont doesnt didnt wont wasnt isnt arent werent havent hasnt hadnt neither nor".split()) | {"no longer"}
HYPOTHETICAL = set("if maybe might perhaps could would wish hope hoping thinking considering should wonder wondering".split())
OTHER_ACTOR = set("he she they his her their sister brother friend mom dad mother father son daughter husband wife cousin aunt uncle neighbor neighbour colleague boss".split())
RELATIONS = {"married_to", "parent_of", "friend_of"}
SHORT_WINDOW = 6


@dataclass(frozen=True)
class SourceLine:
    sid: int
    speaker: str
    text: str
    date: str = ""


@dataclass(frozen=True)
class Candidate:
    subject: str
    predicate: str
    object: str
    source_sids: tuple = ()
    time: str = ""
    method: str = ""
    version: str = ""
    raw: str = ""


@dataclass(frozen=True)
class Decision:
    verdict: str
    reasons: tuple = ()
    subject: Resolution | None = None
    predicate: str = ""
    time: str | None = None


class FactValidator:
    def __init__(self, kb):
        self.resolver = EntityResolver(kb)

    def validate(self, candidate, source_sentences):
        """1 entity references, 2 predicate and object, 3 source support, 4 negation and attribution. Returns a Decision."""
        reject, unsure = [], []
        word = predicate(candidate.predicate)
        lines = list(source_sentences)
        if not lines: reject.append("no source sentence")
        if any(s is None for s in candidate.source_sids) or len(lines) != len(candidate.source_sids): reject.append("a cited sentence does not exist")
        if word not in PREDICATE_TYPES: reject.append(f"predicate not allowed: {word}")
        wanted = content_stems(candidate.object)
        if not wanted: reject.append("object is empty or has no meaning alone")

        who = self.resolver.resolve(candidate.subject, lines)
        if who.status == "absent" or not normalize_name(candidate.subject): reject.append(who.why)
        elif who.status in ("unresolved", "ambiguous"): unsure.append(who.why)
        if reject: return Decision(REJECTED, tuple(reject), who, word)

        # Source support: only the cited sentences count, and every word of a short object must be there.
        said = [(s, stems(s.text)) for s in lines]
        everywhere = {w for _, st in said for w in st}
        share = sum(w in everywhere for w in wanted) / len(wanted)
        if share < (1 if len(wanted) <= 3 else .75): return Decision(REJECTED, ("object not found in its source",), who, word)
        holder = next((s for s, st in said if wanted[0] in st), lines[0])

        # A message can hold many sentences. The one holding the object is the one that must state the fact.
        sentence = next((x for x in re.split(r"(?<=[.!?])\s+|\n+", holder.text) if wanted[0] in stems(x)), holder.text)
        held = stems(sentence)

        # Predicate: does that sentence use a word for this kind of fact?
        cues = {stem(w) for w in PREDICATE_CUES.get(word, "").split()}
        if cues and not cues & set(held): unsure.append(f"nothing in the source says {word}")

        # Negation and uncertainty near the object.
        at = held.index(wanted[0])
        near = tokens(sentence)[max(0, at - SHORT_WINDOW):at + 1]
        said_plain = re.sub(r"[’']", "", sentence.lower())
        if word != "dislikes" and (NEGATION & set(near) or "no longer" in said_plain or "used to" in said_plain): unsure.append("the source may negate this")
        if word != "plans" and HYPOTHETICAL & set(near): unsure.append("the source may be hypothetical")
        if sentence.rstrip().endswith("?"): unsure.append("the source is a question")

        # Attribution: is this the subject's own statement?
        if normalize_name(holder.speaker) == "assistant" and normalize_name(who.name) != "assistant":
            unsure.append("said by the assistant, not the subject")
        elif who.status != "unresolved":
            speaker_is_subject = normalize_name(holder.speaker) == normalize_name(who.name) or self.resolver.kb.resolve(holder.speaker) == who.entity_id and who.entity_id
            first = {"i", "im", "ive", "ill", "id", "my", "me", "we", "our", "us"}
            words = set(tokens(sentence))
            names_subject = bool(set(tokens(who.name)) & words)
            if speaker_is_subject:
                before = set(tokens(sentence)[:at + 1])
                if not first & words and "you" in words: unsure.append("the speaker is talking about someone else")
                elif OTHER_ACTOR & before and not {"i", "im", "ive", "we"} & before: unsure.append("this may be about someone else the speaker mentions")
            elif not names_subject: return Decision(REJECTED, ("the speaker of the source is someone else",), who, word)
            else: unsure.append("reported by someone else, not the subject")

        # Time must come from the source.
        when = None
        if tidy(candidate.time):
            when = time_value(candidate.time)
            if when is None: unsure.append("time is not a plain date")
            elif when not in set().union(*(time_values_in(s.text, s.date) for s in lines)): unsure.append("time is not in the source")
        return Decision(UNCERTAIN if unsure else ACCEPTED, tuple(unsure), who, word, when)


class KnowledgePipeline:
    """Candidate to validator, then to the knowledge store, the review queue, or nowhere."""

    def __init__(self, db):
        self.db = db
        self.kb = KnowledgeBase(db)
        self.validator = FactValidator(self.kb)

    def lines(self, candidate):
        found = []
        for sid in candidate.source_sids:
            row = self.db.execute("SELECT sid, speaker, text, date FROM sentences WHERE sid=?", (sid,)).fetchone() if isinstance(sid, int) else None
            if row: found.append(SourceLine(*row))
        return found

    def process(self, candidate):
        decision = self.validator.validate(candidate, self.lines(candidate))
        if decision.verdict == ACCEPTED: return decision, self._store(candidate, decision, "rules " + RULES_VERSION).fact.fact_id
        if decision.verdict == UNCERTAIN: return decision, self.kb.add_review(json.dumps(candidate.__dict__ | dict(source_sids=list(candidate.source_sids))), list(decision.reasons), list(candidate.source_sids))
        return decision, None

    def _store(self, candidate, decision, checked_by, entity_id=None):
        who = entity_id or decision.subject.entity_id or self.kb.add_entity(Entity.create(decision.subject.name, "unknown")).entity_id
        same = [x for x in self.kb.about(self.kb.entity(who).canonical_name, statuses=None) if x.fact.predicate == decision.predicate and normalize_name(x.fact.object_value) == normalize_name(candidate.object)]
        if same:
            for sid in candidate.source_sids: self.kb.add_evidence(FactEvidence.create(same[0].fact.fact_id, sid, extraction_method=candidate.method or None, extraction_version=candidate.version or None))
            return self.kb.get(same[0].fact.fact_id)
        target = self.kb.resolve(candidate.object) if decision.predicate in RELATIONS else None
        fact = KnowledgeFact.create(who, decision.predicate, "entity" if target else "text", PREDICATE_TYPES[decision.predicate], "explicit", object_value=None if target else candidate.object, object_entity_id=target,
                                    validation_status="accepted", valid_from=decision.time, metadata=dict(checked_by=checked_by))
        return self.kb.add(FactWithEvidence(fact, tuple(FactEvidence.create(fact.fact_id, s, extraction_method=candidate.method or None, extraction_version=candidate.version or None) for s in candidate.source_sids)))

    def approve(self, review_id, entity_name=None):
        """A person accepts a reviewed candidate, and may say which known entity it refers to. The source checks still apply."""
        item = self.kb.review(review_id)
        if item is None: raise RecordError("no such review item")
        data = json.loads(item["candidate"])
        candidate = Candidate(**data | dict(source_sids=tuple(data["source_sids"])))
        decision = self.validator.validate(candidate, self.lines(candidate))
        if decision.verdict == REJECTED: raise RecordError("; ".join(decision.reasons))
        entity_id = self.kb.resolve(entity_name) if entity_name else decision.subject.entity_id
        if entity_name and not entity_id: raise RecordError("no entity by that name")
        if not entity_id and not decision.subject.name: raise RecordError("say which entity this refers to")
        stored = self._store(candidate, decision, "person", entity_id)
        self.kb.drop_review(review_id)
        return stored

    def decline(self, review_id):
        self.kb.drop_review(review_id)
