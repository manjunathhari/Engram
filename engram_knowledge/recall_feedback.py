"""P2.5 connected: feedback changes what is shown at once, and P6 records what was shown, in shadow.

Feedback on something shown ("fact:<id>" for a filed fact, "sentence:<sid>" for a sentence) is stored once, in P6.1's event log,
by store_feedback, as before. What is new is that it is read back when memory is shown:
- wrong, outdated, not about me, needs context: the item is still shown, with a caution beside it, until a later "confirm".
  wrong, outdated and not about me also put the item in the review queue, where P3 counts it as waiting and a person can keep
  or retract it (P5's rule: a person settles a change; nothing is retracted by feedback alone).
- irrelevant: hidden for the same question only.
- helpful: a filed fact moves up a little for the same question only.
P6 is shadow only: each showing is recorded as a retrieval event (references, never text), so feedback has something to attach
to and P6 can learn; P6's learned scores are never read when memory is shown, until P6.5's check shows they help.
Context for P6 is a content-free category (P6.1's scheme: the question's kind and slot from the router), not the question.
"""
import json
import uuid

from .feedback import ALLOWED_FEEDBACK, query_key, store_feedback
from .models import RecordError
from .query_router import QueryRouter

CAUTIONS = dict(wrong="marked wrong", outdated="marked outdated", not_about_me="marked not about the user", needs_context="marked as needing context")
TO_REVIEW = ("wrong", "outdated", "not_about_me")
HELPFUL_STEP = 0.1  # how far one "helpful" moves a filed fact up, for the same question; at most three count


def category(question):
    """The content-free bucket P6 learns in: the question's kind and slot, never its words."""
    from engram_p6 import ExperienceRecorder
    router = QueryRouter()
    return ExperienceRecorder.context_key(router.classify(question), router.slot(question) or "none")


def give_feedback(db, target_ref, signal, question, retrieval_event=None, idempotency_key=None):
    """Stores one piece of feedback; returns the event and, for wrong, outdated or not about me, the review item."""
    from engram_p6 import ExperienceRecorder
    from .database import KnowledgeBase
    if signal not in ALLOWED_FEEDBACK: raise RecordError(f"feedback must be one of {ALLOWED_FEEDBACK}")
    qk = query_key(question)[:24]
    key = idempotency_key or "fb_" + uuid.uuid4().hex  # give the same key to make a retried click count once
    from .feedback import _check_target
    _check_target(db, target_ref)
    event = ExperienceRecorder(db).record_feedback(idempotency_key=key, target_ref=target_ref, signal=signal, context_key=category(question),
                                                   source_event_ref=retrieval_event, metadata=dict(question_key=qk))
    review = None
    if signal in TO_REVIEW and not event.replayed:
        kb = KnowledgeBase(db)
        kind, _, ident = target_ref.partition(":")
        if kind == "fact":
            rec = kb.get(ident)
            who = kb.entity(rec.fact.subject_id)
            sids = [int(e.source_id) for e in rec.evidence]
            candidate = dict(kind="feedback_review", target=target_ref, signal=signal, subject=who.canonical_name if who else "", predicate=rec.fact.predicate, object=rec.fact.object_value)
        else:
            sids = [int(ident)]
            candidate = dict(kind="feedback_review", target=target_ref, signal=signal)
        review = kb.add_review(json.dumps(candidate), [f"feedback: {CAUTIONS[signal]}"], sids)
    return dict(event=event.event_id, review=review)


def settle(db, review_id, keep, reviewer, reason):
    """A person decides a feedback review: keep the item as it is, or retract a filed fact. The decision is kept in the fact's note."""
    from .database import KnowledgeBase
    kb = KnowledgeBase(db)
    item = kb.review(review_id)
    if item is None: raise RecordError("no such review item")
    c = json.loads(item["candidate"])
    if c.get("kind") != "feedback_review": raise RecordError("not a feedback review")
    if not str(reviewer or "").strip() or not str(reason or "").strip(): raise RecordError("a decision needs a reviewer and a reason")
    kind, _, ident = c["target"].partition(":")
    if not keep:
        if kind != "fact": raise RecordError("a sentence is never retracted; it can only be deleted")
        kb.set_status(ident, "retracted", note=f"Retracted by {reviewer.strip()} after feedback ({c['signal']}): {reason.strip()}"[:300])
    else:  # kept after a look: the caution is cleared by a confirmation in the same log
        from engram_p6 import ExperienceRecorder
        ExperienceRecorder(db).record_feedback(idempotency_key="settle_" + review_id, target_ref=c["target"], signal="confirm", metadata=dict(review=review_id))
    kb.drop_review(review_id)
    return dict(target=c["target"], kept=bool(keep))


class FeedbackView:
    """What feedback says about each item, read once per showing."""

    def __init__(self, db):
        self.cautions, self.hidden, self.helpful = {}, {}, {}
        have = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='p6_experience_events'").fetchone()
        if not have: return
        rows = db.execute("SELECT target_ref, signal, metadata FROM p6_experience_events WHERE event_type='feedback' AND target_ref IS NOT NULL ORDER BY occurred_at, event_id").fetchall()
        for target, signal, meta in rows:
            qk = json.loads(meta or "{}").get("question_key")
            if signal in CAUTIONS: self.cautions[target] = CAUTIONS[signal]
            elif signal == "confirm": self.cautions.pop(target, None)
            elif signal == "irrelevant" and qk: self.hidden.setdefault(target, set()).add(qk)
            elif signal == "helpful" and qk: self.helpful.setdefault(target, {}).setdefault(qk, 0); self.helpful[target][qk] += 1

    def caution(self, target):
        return self.cautions.get(target, "")

    def is_hidden(self, target, question):
        return query_key(question)[:24] in self.hidden.get(target, ())

    def boost(self, target, question):
        return HELPFUL_STEP * min(3, self.helpful.get(target, {}).get(query_key(question)[:24], 0))


def record_showing(db, question, refs):
    """P6 shadow: one retrieval event with the references shown. Returns its id, for feedback to point back to."""
    from engram_p6 import ExperienceRecorder
    from .query_router import QueryRouter
    if not refs: return None
    e = ExperienceRecorder(db).record_retrieval(idempotency_key="ret_" + uuid.uuid4().hex, context_key=category(question), query_kind=QueryRouter().classify(question),
                                                 candidate_refs=refs, selected_refs=refs, metadata=dict(shown=len(refs)))
    return e.event_id


def shadow_scores(db, question, refs):
    """What P6 would change for these items, in this question's category. Read only, and never used to show memory."""
    from engram_p6 import AdaptiveScorer
    scorer, ctx = AdaptiveScorer(db), category(question)
    return {r: scorer.score(r, ctx).multiplier for r in refs}
