"""P6.2 Context Evaluator.

Read-only evaluator over P6.1 experience events. It estimates context-specific
retrieval usefulness from explicit feedback. It does not modify facts or learn
retrieval parameters; P6.3 owns adaptation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json, sqlite3
from collections import defaultdict
from typing import Any

class ContextEvaluationError(ValueError):
    pass

POSITIVE = {"helpful", "confirm", "successful"}
NEGATIVE = {"irrelevant", "wrong", "outdated", "not_about_me",
            "corrected", "unsuccessful", "no_result"}
NEUTRAL = {"needs_context", "selected", "dismissed"}

@dataclass(frozen=True)
class ContextAssessment:
    target_ref: str
    context_key: str | None
    query_kinds: tuple[str, ...]
    exposure_count: int
    selection_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    evidence_count: int
    usefulness_mean: float
    evidence_coverage: float
    assessment: str

    def to_dict(self) -> dict[str, Any]:
        d=asdict(self)
        d["query_kinds"]=list(self.query_kinds)
        return d

class ContextEvaluator:
    """Evaluates memory usefulness in context using P6.1's event ledger."""

    def __init__(self, db: sqlite3.Connection, *, prior_alpha: float=1.0,
                 prior_beta: float=1.0, coverage_saturation: int=5):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ContextEvaluationError("Beta prior parameters must be positive")
        if coverage_saturation < 1:
            raise ContextEvaluationError("coverage_saturation must be >= 1")
        self.db=db
        self.alpha=float(prior_alpha)
        self.beta=float(prior_beta)
        self.coverage_saturation=coverage_saturation
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "p6_experience_events" not in tables:
            raise ContextEvaluationError("P6.1 table p6_experience_events is missing")
        required={"event_id","event_type","occurred_at","context_key","query_kind",
                  "candidate_refs","selected_refs","target_ref","signal",
                  "source_event_ref","metadata"}
        columns={r[1] for r in db.execute("PRAGMA table_info(p6_experience_events)")}
        if not required.issubset(columns):
            raise ContextEvaluationError("p6_experience_events schema does not match P6.1 contract")

    @staticmethod
    def _load_refs(value: str) -> list[str]:
        try: parsed=json.loads(value or "[]")
        except (TypeError,ValueError): raise ContextEvaluationError("invalid reference JSON in P6.1 event") from None
        if not isinstance(parsed,list) or any(not isinstance(x,str) for x in parsed):
            raise ContextEvaluationError("P6.1 references must be JSON arrays of strings")
        return parsed

    def _events(self, target_ref: str, context_key: str | None):
        if not target_ref or not target_ref.strip():
            raise ContextEvaluationError("target_ref is required")
        sql="""SELECT event_id,event_type,occurred_at,context_key,query_kind,
                 candidate_refs,selected_refs,target_ref,signal,signal_value,
                 source_event_ref,metadata
               FROM p6_experience_events"""
        params=[]
        if context_key is None:
            sql+=" WHERE target_ref=? OR event_type='retrieval'"
            params.append(target_ref)
        else:
            sql+=" WHERE context_key=?"
            params.append(context_key)
        sql+=" ORDER BY occurred_at,event_id"
        return self.db.execute(sql,params).fetchall()

    def evaluate(self, target_ref: str, context_key: str | None=None) -> ContextAssessment:
        """Assess one memory overall or within one context bucket.

        Retrievals contribute exposure/selection counts only. Only explicit
        feedback, confirmation, correction and outcome signals affect usefulness.
        """
        events=self._events(target_ref,context_key)
        exposure=selected=positive=negative=neutral=0
        kinds=set()
        for row in events:
            (event_id,event_type,occurred_at,ctx,query_kind,candidate_json,
             selected_json,event_target,signal,signal_value,source_ref,metadata)=row
            if query_kind: kinds.add(query_kind)
            if event_type=="retrieval":
                candidates=self._load_refs(candidate_json)
                picks=self._load_refs(selected_json)
                if target_ref in candidates: exposure+=1
                if target_ref in picks: selected+=1
            # Direct event references are exact; don't infer relevance from selection.
            if event_target != target_ref:
                continue
            sig=signal
            if event_type=="correction" and not sig: sig="corrected"
            if event_type=="confirmation" and not sig: sig="confirm"
            if event_type=="recall_failure" and not sig: sig="no_result"
            if event_type=="outcome" and not sig: sig="successful"
            if sig in POSITIVE: positive+=1
            elif sig in NEGATIVE: negative+=1
            elif sig in NEUTRAL: neutral+=1

        n=positive+negative
        mean=(self.alpha+positive)/(self.alpha+self.beta+n)
        coverage=min(1.0,n/self.coverage_saturation)
        if n==0:
            label="insufficient_evidence"
        elif mean >= 0.70 and n >= 2:
            label="useful_in_context"
        elif mean <= 0.30 and n >= 2:
            label="low_usefulness_in_context"
        else:
            label="mixed_or_early_evidence"
        return ContextAssessment(target_ref,context_key,tuple(sorted(kinds)),
            exposure,selected,positive,negative,neutral,n,round(mean,4),
            round(coverage,4),label)

    def evaluate_all_contexts(self, target_ref: str) -> list[ContextAssessment]:
        if not target_ref or not target_ref.strip():
            raise ContextEvaluationError("target_ref is required")
        contexts={r[0] for r in self.db.execute("""SELECT DISTINCT context_key
          FROM p6_experience_events WHERE target_ref=? AND context_key IS NOT NULL""",(target_ref,))}
        # Also include contexts where the memory appeared as a retrieval candidate.
        for (ctx,candidates) in self.db.execute("""SELECT context_key,candidate_refs
          FROM p6_experience_events WHERE event_type='retrieval' AND context_key IS NOT NULL"""):
            if target_ref in self._load_refs(candidates): contexts.add(ctx)
        return [self.evaluate(target_ref,ctx) for ctx in sorted(contexts)]

    def evaluate_many(self, target_refs: list[str]) -> list[ContextAssessment]:
        if not isinstance(target_refs,list) or any(not isinstance(x,str) or not x.strip() for x in target_refs):
            raise ContextEvaluationError("target_refs must be a list of non-empty strings")
        return [self.evaluate(ref) for ref in dict.fromkeys(target_refs)]

    def sync_p2_feedback(self) -> int:
        """Bridge existing P2 feedback rows into P6.1 without copying comments.

        Requires P2's documented p2_feedback columns. The source P2 table is
        read-only here. Each event is imported idempotently into P6.1.
        """
        tables={r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "p2_feedback" not in tables:
            raise ContextEvaluationError("P2 table p2_feedback is missing")
        cols={r[1] for r in self.db.execute("PRAGMA table_info(p2_feedback)")}
        required={"event_id","candidate_id","feedback","query_key"}
        if not required.issubset(cols):
            raise ContextEvaluationError("p2_feedback schema does not match supported P2 integration contract")
        rows=self.db.execute("""SELECT event_id,candidate_id,feedback,query_key
          FROM p2_feedback ORDER BY created_at,event_id""").fetchall()
        allowed=POSITIVE|NEGATIVE|{"needs_context"}
        imported=0
        for event_id,candidate_id,feedback,query_key in rows:
            if feedback not in allowed:
                continue
            existing=self.db.execute("SELECT 1 FROM p6_experience_events WHERE idempotency_key=?",
                ("p2-feedback:"+event_id,)).fetchone()
            if existing: continue
            # Lazy import avoids a hard package dependency: P6.1 is the event writer.
            from .experience_compat import write_feedback
            write_feedback(self.db,event_id,candidate_id,feedback,query_key)
            imported+=1
        return imported
