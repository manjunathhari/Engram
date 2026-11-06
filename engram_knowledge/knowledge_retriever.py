"""Answers structured questions from the knowledge store, alongside ordinary memory. Engrams.retrieve and ChatMemory.find are used as they are."""
import re

from engram_records import KnowledgeBase, normalize_name

from .evidence_assembler import EvidenceAssembler
from .query_router import QueryRouter
from .state_engine import StateEngine
from .temporal import span


class KnowledgeRetriever:
    def __init__(self, memory):
        self.memory = memory
        self.kb = KnowledgeBase(memory.db)
        self.state_engine = StateEngine(self.kb)
        self.router = QueryRouter()
        self.evidence = EvidenceAssembler(self.kb)

    def find_entity(self, question):
        """A known name or alias named in the question, longest first. Two different people named means no single answer."""
        names = [r[0] for r in self.kb.db.execute("SELECT canonical_name FROM entities").fetchall()] + [r[0] for r in self.kb.db.execute("SELECT alias FROM entity_aliases").fetchall()]
        text = normalize_name(question)
        found, covered = [], []
        for name in sorted(names, key=len, reverse=True):
            m = re.search(r"(?<!\w)" + re.escape(normalize_name(name)) + r"(?!\w)", text)
            if m and not any(a <= m.start() < b for a, b in covered):
                found.append(self.kb.resolve(name)); covered.append(m.span())
        found = list(dict.fromkeys(found))
        return found[0] if len(found) == 1 else None

    def _memory(self, question):
        sids, _ = self.memory.find(question)
        return sids

    def _result(self, kind, question, entity_id, slot, facts=(), conflicts=(), note="", memory=()):
        return dict(query_type=kind, question=question, entity_id=entity_id, slot=slot, facts=list(facts), conflicts=list(conflicts), note=note, memory_sids=list(memory),
                    sentences_used=len({s for f in facts for s in f["evidence"]} | set(memory)))

    def retrieve(self, question, entity_id=None):
        kind = self.router.classify(question)
        slot = self.router.slot(question)
        entity_id = entity_id or self.find_entity(question)
        if kind == "general_memory": return self._result(kind, question, entity_id, slot, memory=self._memory(question))
        if not entity_id: return self._result(kind, question, None, slot, note="no single known entity named", memory=self._memory(question))
        if kind == "conflict": return self._conflicts(question, entity_id, slot)
        if not slot: return self._result(kind, question, entity_id, None, note="could not tell which kind of fact", memory=self._memory(question))
        return getattr(self, "_" + kind)(question, entity_id, slot)

    def _state(self, kind, question, entity_id, slot, state, note):
        here = state.get(slot)
        waiting = here.waiting if here else []
        warning = "newer statements may be waiting for review" if waiting else ""
        if not here or not here.values: return self._result(kind, question, entity_id, slot, note=" ; ".join(x for x in (note, warning) if x), memory=self._memory(question)) | dict(waiting_review=waiting)
        return self._result(kind, question, entity_id, slot, [self.evidence.period(slot, p, here.conflict) for p in here.values], here.conflicts, note=warning) | dict(waiting_review=waiting)

    def _current_state(self, question, entity_id, slot):
        return self._state("current_state", question, entity_id, slot, self.state_engine.get_current_state(entity_id), "no current record")

    def _historical_state(self, question, entity_id, slot):
        return self._state("historical_state", question, entity_id, slot, self.state_engine.get_state_at(entity_id, self.router.extract_date(question)), "no record for that time")

    def _history_periods(self, entity_id, slot):
        return self.state_engine.get_history(entity_id, slot)

    def _history(self, question, entity_id, slot):
        h = self._history_periods(entity_id, slot)
        if not h["periods"]: return self._result("history", question, entity_id, slot, note="no record", memory=self._memory(question))
        return self._result("history", question, entity_id, slot, [self.evidence.period(slot, p, bool(h["conflicts"])) for p in h["periods"]], h["conflicts"])

    def _original(self, question, entity_id, slot):
        """The earliest value. If the earliest date cannot be told apart, every candidate is returned."""
        h = self._history_periods(entity_id, slot)
        placed = [p for p in h["periods"] if p.start or p.first_said]
        if not placed: return self._result("original", question, entity_id, slot, note="no dated record", memory=self._memory(question))
        first = min(span(p.start or p.first_said)[0] for p in placed)
        keep = [p for p in placed if span(p.start or p.first_said)[0] == first]
        return self._result("original", question, entity_id, slot, [self.evidence.period(slot, p, len(keep) > 1) for p in keep], h["conflicts"], note="" if not [p for p in h["periods"] if p not in placed] else "some records have no date")

    def _change(self, question, entity_id, slot):
        """When the current value began. The start is unknown if none was stated; first_said then gives when it was first mentioned."""
        result = self._current_state(question, entity_id, slot)
        result["query_type"] = "change"
        return result

    def _evidence(self, question, entity_id, slot):
        result = self._current_state(question, entity_id, slot)
        result["query_type"] = "evidence"
        return result

    def _conflicts(self, question, entity_id, slot):
        state = self.state_engine.get_current_state(entity_id)
        slots = [slot] if slot else sorted(state)
        facts, conflicts = [], []
        for s in slots:
            if s in state and state[s].conflict:
                facts += [self.evidence.period(s, p, True) for p in state[s].values]
                conflicts += state[s].conflicts
        return self._result("conflict", question, entity_id, slot, facts, conflicts, note="" if conflicts else "no conflict on record")

    def combined_retrieval(self, question, entity_id=None):
        """Both paths: the record, and the sentences ordinary memory finds."""
        entity_id = entity_id or self.find_entity(question)
        slot = self.router.slot(question)
        result = self._current_state(question, entity_id, slot) if entity_id and slot else self._result("combined", question, entity_id, slot)
        result = result | dict(query_type="combined", memory_sids=self._memory(question))
        result["sentences_used"] = len({s for f in result["facts"] for s in f["evidence"]} | set(result["memory_sids"]))
        return result

    def _combined(self, question, entity_id, slot):
        return self.combined_retrieval(question, entity_id)
