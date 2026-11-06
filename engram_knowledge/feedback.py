"""Feedback on a remembered fact or sentence (from P2.5), stored in one place: P6.1's event log.

A target is "fact:<fact_id>" or "sentence:<sid>" and must exist. Free-text comments are not stored, as P6.1 asks, so the log holds
references and a feedback word, never conversation text. Feedback changes no fact: "wrong" or "outdated" asks a person to look.
Deleting a sentence removes feedback and learned scores that point at it, or at a fact erased with it.
"""
import hashlib
import json
import re

from .models import RecordError

ALLOWED_FEEDBACK = ("helpful", "irrelevant", "wrong", "outdated", "not_about_me", "needs_context", "confirm")
P6_TABLES = ("p6_experience_events", "p6_scoring_policies")


def query_key(query):
    return hashlib.sha256(" ".join(re.findall(r"[a-z0-9]+", str(query).casefold())).encode()).hexdigest()


def _check_target(db, target_ref):
    kind, _, ident = str(target_ref).partition(":")
    if kind == "fact":
        row = db.execute("SELECT status FROM knowledge_records WHERE fact_id=?", (ident,)).fetchone()
        if row is None or row[0] == "invalidated": raise RecordError(f"no such fact to give feedback on: {ident}")
    elif kind == "sentence":
        if not ident.isdigit() or db.execute("SELECT 1 FROM sentences WHERE sid=?", (int(ident),)).fetchone() is None: raise RecordError(f"no such sentence: {ident}")
    else:
        raise RecordError("feedback targets are 'fact:<id>' or 'sentence:<sid>'")


def store_feedback(db, target_ref, feedback, query, idempotency_key=None, context_key=None):
    """Writes one feedback event. The same idempotency key twice writes it once."""
    from engram_p6 import ExperienceRecorder
    if feedback not in ALLOWED_FEEDBACK: raise RecordError(f"feedback must be one of {ALLOWED_FEEDBACK}")
    if not str(query or "").strip(): raise RecordError("feedback needs the question it was about")
    _check_target(db, target_ref)
    key = idempotency_key or "fb_" + hashlib.sha256(json.dumps([target_ref, feedback, query_key(query)]).encode()).hexdigest()[:24]
    return ExperienceRecorder(db).record_feedback(idempotency_key=key, target_ref=target_ref, signal=feedback, context_key=context_key or query_key(query)[:24])


def forget_refs(db, refs):
    """Removes feedback and learned scores for these references. Called when a sentence is deleted."""
    refs = set(refs)
    have = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not refs or "p6_experience_events" not in have: return 0
    gone = 0
    for event_id, target, cand, sel in db.execute("SELECT event_id, target_ref, candidate_refs, selected_refs FROM p6_experience_events").fetchall():
        if target in refs:
            db.execute("DELETE FROM p6_experience_events WHERE event_id=?", (event_id,)); gone += 1
            continue
        c, s = json.loads(cand or "[]"), json.loads(sel or "[]")
        if refs & set(c + s):  # a retrieval that showed it: keep the event, drop the reference
            db.execute("UPDATE p6_experience_events SET candidate_refs=?, selected_refs=? WHERE event_id=?", (json.dumps([x for x in c if x not in refs]), json.dumps([x for x in s if x not in refs]), event_id))
    if "p6_scoring_policies" in have:
        for r in refs: gone += db.execute("DELETE FROM p6_scoring_policies WHERE target_ref=?", (r,)).rowcount
    return gone
