"""Joins the sentence pipeline (interpreter, learning gate, candidate envelope) to the P1 store.

Reads a stored sentence by its sid, splits it into sentences, and builds one envelope per sentence worth keeping. Review envelopes
go into P1's one review queue. Nothing here creates a fact: a fact needs who, what and which, which extraction must supply.
"""
import hashlib
import json

from .candidate import build_candidate
from .interpreter import interpret, split_message
from .learning import evaluate

# P2 content kinds onto P1 fact types. Hints for extraction, not facts.
FACT_TYPE = dict(preference="preference", plan="plan", goal="plan", decision="event", experience="event", instruction="preference", claim="observation", belief="observation", hypothesis="observation")


def prepare_stored(db, sid, people=(), context=None):
    """Envelopes for one stored sentence row. A greeting at the start no longer hides the fact after it."""
    row = db.execute("SELECT sid, chat, speaker, date, ref, caption, text FROM sentences WHERE sid=?", (sid,)).fetchone()
    if row is None: raise ValueError(f"no stored sentence {sid}")
    sid, chat, speaker, date, ref, caption, text = row
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    results = []
    for start, end, body in split_message(text):
        source = dict(sid=sid, chat=chat, speaker=speaker, date=date or "", ref=ref or "", caption=caption or "", text=body, span=(start, end), message_sha256=digest)
        parsed = interpret(speaker, body, date=date or "", context=context, people=people)
        results.append(build_candidate(source, parsed, evaluate(parsed, has_context=context is not None), keep_assistant=True))
    return results


def p1_hint(envelope):
    """What P1 would need, so extraction can be pointed at it. P2's 'uncertain' becomes P1's review queue, never a stored fact."""
    e = envelope.to_dict() if hasattr(envelope, "to_dict") else envelope
    kind = e["interpretation"]["content_kind"]
    return dict(fact_type=FACT_TYPE.get(kind), source_type="explicit" if e["lifecycle"] == "proposed" else None,
                about="assistant" if e["lifecycle"] == "assistant_statement" else "speaker", goes_to="review_queue" if e["lifecycle"] == "needs_review" else "extraction")


def queue_for_review(kb, envelope):
    """Puts a review envelope in P1's review queue, tied to its sentence so deleting the sentence removes it."""
    e = envelope.to_dict() if hasattr(envelope, "to_dict") else envelope
    if e["lifecycle"] != "needs_review": raise ValueError("only review envelopes go to the review queue")
    return kb.add_review(json.dumps(dict(kind="sentence_envelope", envelope=e)), list(e["learning"]["reason_codes"]), [int(e["source"]["sid"])])
