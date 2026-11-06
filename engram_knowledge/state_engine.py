"""What is true of an entity, now and at any past date, worked out from the knowledge store.

The state is derived and never stored, so the facts and their evidence stay the only source of truth. Reads change nothing.
"""
import json
from dataclasses import dataclass, field, replace

from .models import RecordError, normalize_name

from .conflict import Conflict
from .temporal import covers, overlaps, said_on, span, step_before
from .transition import CLOSES, OPENS, SINGLE, slot_of

COUNTED = ("active", "needs_review", "disputed")  # superseded, retracted and invalidated do not count


@dataclass
class Period:
    value: str
    start: str | None  # the stated start; None when no date was stated, and never invented
    first_said: str | None  # the earliest date of a sentence stating it
    end: str | None = None
    end_kind: str = "open"  # open, stated, closed or replaced
    fact_ids: list = field(default_factory=list)
    sources: set = field(default_factory=set)


@dataclass
class SlotState:
    slot: str
    values: list  # the Periods that hold
    conflict: bool
    conflicts: list
    waiting: list = field(default_factory=list)  # review items that may change this answer
    unmatched: list = field(default_factory=list)  # closing statements naming a value not held: kept and shown, but not a conflict


class StateEngine:
    def __init__(self, kb):
        self.kb = kb
        self.db = kb.db

    # ---------- writing ----------
    def apply_observation(self, record):
        """Adds an observation. Nothing older is changed, and what it conflicts with is reported."""
        found = self.detect_conflict(record)
        saved = self.kb.add(record)
        return dict(record=saved, conflicts=found)

    def apply_correction(self, corrected_fact_id, record, note="Corrected"):
        """Replaces what was wrong: the old fact is marked superseded and keeps its evidence."""
        old = self.kb.get(corrected_fact_id)
        if old is None: raise RecordError("no such fact to correct")
        saved = self.kb.add(replace(record, fact=replace(record.fact, metadata=dict(record.fact.metadata) | dict(corrects=corrected_fact_id))))
        self.kb.set_status(corrected_fact_id, "superseded", note=note)
        return saved

    def resolve_conflict(self, keep_fact_id, drop_fact_id, note="Settled by a person"):
        """A person decides. The dropped fact stays, marked superseded, with its evidence."""
        return self.kb.set_status(drop_fact_id, "superseded", note=note), self.kb.get(keep_fact_id)

    # ---------- reading ----------
    def _speakers(self, record):
        return self._evidence(record)[1]

    def _evidence(self, record):
        said, speakers = [], set()
        for e in record.evidence:
            if e.evidence_role != "supports" or e.source_kind != "sentence": continue
            row = self.db.execute("SELECT speaker, date FROM sentences WHERE sid=?", (int(e.source_id),)).fetchone()
            if row:
                speakers.add(normalize_name(row[0]))
                day = said_on(row[1])
                if day: said.append(day)
        return sorted(said, key=lambda d: span(d)[0]), speakers

    def _observations(self, entity_id):
        rows = self.db.execute("SELECT fact_id FROM knowledge_records WHERE subject_id=? ORDER BY recorded_at, fact_id", (entity_id,)).fetchall()
        out = []
        for r in rows:
            x = self.kb.get(r[0])
            f = x.fact
            if f.status in COUNTED and f.validation_status != "rejected" and f.source_type == "explicit" and f.fact_type != "plan" and slot_of(f.predicate):
                said, speakers = self._evidence(x)
                out.append(dict(fact=f, slot=slot_of(f.predicate), said=said, speakers=speakers, when=f.valid_from or (said[0] if said else None)))
        return out

    def _value(self, fact):
        if fact.object_entity_id:
            other = self.kb.entity(fact.object_entity_id)
            return other.canonical_name if other else ""
        return fact.object_value

    def _build(self, observations, slot):
        """Plays one slot's observations in date order. Returns all periods and the conflicts."""
        single = slot in SINGLE
        seen = [o for o in observations if o["slot"] == slot]
        dated = sorted((o for o in seen if o["when"]), key=lambda o: (span(o["when"])[0], o["fact"].predicate in CLOSES))
        undated = [o for o in seen if not o["when"]]
        periods, conflicts = [], []

        def add_period(o):
            f = o["fact"]
            p = Period(self._value(f), f.valid_from, o["said"][0] if o["said"] else None, fact_ids=[f.fact_id], sources=set(o["speakers"]))
            if f.valid_to: p.end, p.end_kind = f.valid_to, "stated"
            periods.append(p)
            return p

        for o in dated:
            f, value = o["fact"], self._value(o["fact"])
            key = normalize_name(value)
            still = [p for p in periods if p.end_kind == "open" or (p.end_kind == "stated" and span(p.end)[1] >= span(o["when"])[0])]
            if f.predicate in CLOSES:
                hit = [p for p in still if normalize_name(p.value) == key]
                for p in hit:
                    p.end, p.end_kind = o["when"], "closed"
                    p.fact_ids.append(f.fact_id)
                if not hit: conflicts.append(Conflict("ended", slot, (value,), (f.fact_id,), tuple(sorted(o["speakers"]))))
                continue
            if single:  # the same speaker giving a new value ends their earlier account
                for p in still:
                    when_old = p.start or p.first_said
                    if normalize_name(p.value) != key and when_old and not overlaps(when_old, o["when"]) and p.sources & o["speakers"] and p.end_kind == "open":
                        p.end, p.end_kind = step_before(o["when"]), "replaced"
            same = next((p for p in still if normalize_name(p.value) == key), None)
            if same:
                if f.valid_from and same.start and not overlaps(f.valid_from, same.start) and not o["speakers"] <= same.sources:
                    conflicts.append(Conflict("date", slot, (value,), (same.fact_ids[0], f.fact_id), tuple(sorted(same.sources | o["speakers"]))))
                if not same.start: same.start = f.valid_from
                elif f.valid_from and span(f.valid_from)[0] < span(same.start)[0]: same.start = f.valid_from
                same.fact_ids.append(f.fact_id); same.sources |= o["speakers"]
                if o["said"] and (not same.first_said or span(o["said"][0])[0] < span(same.first_said)[0]): same.first_said = o["said"][0]
                if f.valid_to: same.end, same.end_kind = f.valid_to, "stated"
                continue
            add_period(o)

        for o in undated:  # no date of any kind: it cannot be placed in order
            f, key = o["fact"], normalize_name(self._value(o["fact"]))
            if f.predicate in CLOSES: conflicts.append(Conflict("ended", slot, (self._value(f),), (f.fact_id,), tuple(sorted(o["speakers"]))))
            elif not any(normalize_name(p.value) == key for p in periods): add_period(o)
        return periods, conflicts

    def _overlap_conflicts(self, slot, held, conflicts):
        """Different values held at once in a single-valued slot."""
        if slot in SINGLE and len({normalize_name(p.value) for p in held}) > 1:
            return conflicts + [Conflict("value", slot, tuple(p.value for p in held), tuple(i for p in held for i in p.fact_ids), tuple(sorted({s for p in held for s in p.sources})))]
        return conflicts

    def _slots(self, observations):
        return sorted({o["slot"] for o in observations})

    def get_current_state(self, entity_id):
        """{slot: SlotState}. Holds what the latest accepted observations say; a conflict keeps every side."""
        observations = self._observations(entity_id)
        out = {}
        for slot in self._slots(observations):
            periods, conflicts = self._build(observations, slot)
            held = [p for p in periods if p.end_kind == "open"]
            current = self._overlap_conflicts(slot, held, [c for c in conflicts if c.kind == "date" and any(set(c.fact_ids) & set(p.fact_ids) for p in held)])
            out[slot] = SlotState(slot, held, bool(current), current, self._waiting(entity_id, slot), [c for c in conflicts if c.kind == "ended"])
        for slot, items in self._all_waiting(entity_id).items():  # known only from statements still in review
            out.setdefault(slot, SlotState(slot, [], False, [], items))
        return out

    def _all_waiting(self, entity_id):
        """Candidates about this entity still in review. A newer statement may be among them, so a stale answer must say so."""
        found = {}
        for item in self.kb.reviews():
            c = json.loads(item["candidate"])
            if "predicate" not in c: continue  # a sentence waiting for extraction, not yet a fact candidate
            slot = slot_of(c["predicate"])
            if slot and self.kb.resolve(c["subject"]) == entity_id and c["predicate"] != "plans": found.setdefault(slot, []).append(item["review_id"])
        return found

    def _waiting(self, entity_id, slot):
        return self._all_waiting(entity_id).get(slot, [])

    def get_state_at(self, entity_id, date):
        """{slot: SlotState} for a past date. A year gives everything true during any of that year."""
        observations = self._observations(entity_id)
        out = {}
        for slot in self._slots(observations):
            periods, conflicts = self._build(observations, slot)
            held = [p for p in periods if covers(p.start or p.first_said, p.end, date)]
            if not held: continue
            out[slot] = SlotState(slot, held, False, [])
            found = self._at_conflicts(slot, held, date) + [c for c in conflicts if c.kind == "date" and any(set(c.fact_ids) & set(p.fact_ids) for p in held)]
            out[slot] = SlotState(slot, held, bool(found), found)
        return out

    def _at_conflicts(self, slot, held, date):
        """Two values during one year may simply be one after the other, so only true overlaps count."""
        if slot not in SINGLE: return []
        pairs = []
        for i, a in enumerate(held):
            for b in held[i + 1:]:
                sa, sb = a.start or a.first_said, b.start or b.first_said
                first, second = (a, b) if span(sa)[0] <= span(sb)[0] else (b, a)
                if normalize_name(a.value) != normalize_name(b.value) and (first.end is None or span(first.end)[1] >= span(second.start or second.first_said)[0]):
                    pairs.append(Conflict("value", slot, (a.value, b.value), tuple(a.fact_ids + b.fact_ids), tuple(sorted(a.sources | b.sources))))
        return pairs

    def get_history(self, entity_id, predicate):
        """Every period of a slot in date order, the conflicts, and what was not counted and why, so nothing is lost silently."""
        slot = slot_of(predicate) or predicate
        observations = self._observations(entity_id)
        periods, conflicts = self._build(observations, slot)
        periods.sort(key=lambda p: (span(p.start or p.first_said)[0] if (p.start or p.first_said) else span("9999")[0]))
        counted = {o["fact"].fact_id for o in observations}
        other = []
        for r in self.db.execute("SELECT fact_id FROM knowledge_records WHERE subject_id=? ORDER BY recorded_at, fact_id", (entity_id,)).fetchall():
            x = self.kb.get(r[0])
            if r[0] not in counted and slot_of(x.fact.predicate) == slot: other.append(x)
        return dict(slot=slot, periods=periods, conflicts=conflicts, not_counted=other)

    def get_plans(self, entity_id):
        """Plans are kept apart. A plan is never a state or a completed event."""
        rows = self.db.execute("SELECT fact_id FROM knowledge_records WHERE subject_id=? AND fact_type='plan' AND status IN ('active','needs_review') ORDER BY recorded_at, fact_id", (entity_id,)).fetchall()
        return [self.kb.get(r[0]) for r in rows]

    def detect_conflict(self, record):
        """What this observation would conflict with, if saved. Changes nothing."""
        f = record.fact
        slot = slot_of(f.predicate)
        if f.fact_type == "plan" or not slot or f.source_type != "explicit": return []
        said, speakers = self._evidence(record)
        trial = self._observations(f.subject_id) + [dict(fact=f, slot=slot, said=said, speakers=speakers, when=f.valid_from or (said[0] if said else None))]
        periods, conflicts = self._build(trial, slot)
        held = [p for p in periods if p.end_kind == "open"]
        found = self._overlap_conflicts(slot, held, conflicts)
        return [c for c in found if f.fact_id in c.fact_ids]
