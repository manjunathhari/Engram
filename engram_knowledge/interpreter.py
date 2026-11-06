"""Lightweight, deterministic P2.1 conversational interpreter.

The interpreter labels one already-stored Engram sentence. It deliberately does
not decide whether the content should be learned; that is P2.2's responsibility.
It does not resolve entities/pronouns, infer facts, or write to any database.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Interpretation:
    speaker: str
    speaker_role: str
    speech_act: str
    content_kind: str
    epistemic_stance: str
    attribution: str
    polarity: str
    temporal_markers: tuple[str, ...]
    signals: tuple[str, ...]
    confidence: float
    needs_review: bool
    word_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (tuples become lists)."""
        out = asdict(self)
        out["temporal_markers"] = list(self.temporal_markers)
        out["signals"] = list(self.signals)
        return out


_PATTERNS: dict[str, tuple[str, ...]] = {
    "greeting": (r"^(hi|hello|hey|good morning|good afternoon|good evening)\b",),
    "thanks": (r"^(thank you|thanks|thank u|many thanks|much appreciated)\b",),
    "acknowledgement": (r"^(ok|okay|understood|got it|right|noted|sure)[.! ]*$",),
    "correction": (
        r"\b(i mean|i meant|what i meant|correction|to correct that|that's not what i meant)\b",
        r"\b(no,? actually)\b",
    ),
    "request": (
        r"\b(can|could|would|will) you\b",
        r"^(please|kindly)\b|\bplease[.!?]?$",
        r"^(please\s+|kindly\s+)?(tell me|show me|give me|write|draft|summari[sz]e|explain|analyse|analyze|review|compare|help me|list|suggest|recommend|add|make|create|convert|rewrite|translate|generate|provide|break ?down|format|calculate|find)\b",
        r"\b(tell me|show me|give me|help me)\b",
    ),
    "commitment": (
        r"\b(i will|i'll|we will|we'll|i am going to|we are going to|i'm going to|we're going to)\b",
        r"\b(i commit to|we commit to)\b",
    ),
    "proposal": (r"\b(let's|we could|we should consider|i suggest|i propose|what if)\b",),
    "decision": (r"\b(we decided|i decided|decision is|we have chosen|i have chosen|we will use|we are using)\b",),
    "preference": (
        r"\b(i prefer|i like|i dislike|i don't like|i do not like|i would rather|my preference)\b",
    ),
    "goal": (r"\b(my goal|our goal|i aim to|we aim to|i want to build|i want to learn|i hope to achieve)\b",),
    "plan": (r"\b(plan|planning|planned|roadmap|next step|milestone|schedule)\b",),
    "belief": (r"\b(i think|i believe|in my view|my understanding|i feel that)\b",),
    "hypothesis": (r"\b(might|maybe|perhaps|possibly|could be|i suspect|hypothesis|it is possible that)\b", r"(?-i:\bmay\b)(?!\s+\d)"),
    "instruction": (r"^(always|never|make sure|remember to|do not|don't|please (?:always|never|remember))\b", r"\bfrom now on\b"),
    "emotion": (r"\b(i am happy|i'm happy|i am worried|i'm worried|i am frustrated|i'm frustrated|i feel|i am excited|i'm excited)\b",),
    "question": (r"\?\s*$", r"^(what|why|when|where|who|which|how|is|are|am|do|does|did|has|have|had)\b"),
}

_TIME = re.compile(
    r"\b(?:today|yesterday|tomorrow|tonight|last night|this morning|next week|last week|"
    r"next month|last month|next year|last year|\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
    r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?|\b\d{4}-\d{2}-\d{2})\b",
    re.I,
)
_UNCERTAIN = re.compile(r"\b(might|maybe|perhaps|possibly|probably|i suspect|not sure|uncertain|could be)\b|(?<!\b[Ii]n )(?<!\b[Oo]f )(?<!\b[Ss]ince )(?<!\b[Uu]ntil )(?<!\b[Ll]ast )(?<!\b[Nn]ext )(?<!\b[Tt]his )(?<!\b[Ee]arly )(?<!\b[Ll]ate )(?<!^)(?-i:\bmay\b)(?!\s+\d)", re.I)
_NEGATION = re.compile(r"\b(not|never|no|don't|doesn't|didn't|cannot|can't|won't|isn't|aren't|wasn't|weren't)\b", re.I)
_QUOTED = re.compile(r"(?:\"[^\"]+\"|(?<![\w])'[^']+'(?![\w])|“[^”]+”|(?<![\w])‘[^’]+’(?![\w]))")
_THIRD_PARTY = re.compile(
    r"\b(?:according to|as reported by)\b|"
    r"\b(?i:he|she|they)\s+(?i:said|reported|believes|thinks|decided|stated)\b|"
    # Require a multi-token proper name so sentence-initial "We decided" or
    # "I said" is not mistaken for a named third-party source.
    r"\b[A-Z][a-z][\w.-]*(?:\s+[A-Z][a-z][\w.-]*)+\s+"
    r"(?i:said|reported|believes|thinks|decided|stated)\b"
)
_CONDITIONAL = re.compile(r"\b(if|unless|assuming|suppose|in case|as long as|i wish|i'd love to|i would love to)\b", re.I)
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'll|i'd|me|my|mine|myself|we|we're|we've|we'll|us|our|ours)\b", re.I)
_BARE_START = re.compile(r"^(it|it's|its|they|they're|them|this|that|these|those|he|she|her|his|there|most of them|some of them)\b", re.I)
_BARE_OBJECT = re.compile(r"\b(it|them|that one|this one|those \w+|these \w+|the same)\b[.!?]*$|\b(do|did|recommend|love|loved|like|liked|try|tried|check out|use|used|got|get|buy|bought) (it|them|that|this|those|these)\b", re.I)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def _speaker_role(speaker: str, people=()) -> str:
    """user and assistant by their usual labels. A named person counts as a user of their own memory when listed in people,
    since in a conversation between people (as in LoCoMo) each speaker is talking about their own life."""
    s = speaker.strip().lower()
    if s in {"user", "human", "customer", "client"} or s in {p.strip().lower() for p in people}:
        return "user"
    if s in {"assistant", "ai", "chatgpt", "model", "bot"}:
        return "assistant"
    if s in {"system", "developer"}:
        return "system"
    if s.startswith(("tool:", "function:", "tool_")):
        return "tool"
    return "other"


def interpret(speaker: str, text: str, date: str = "", context: str | None = None, people=()) -> Interpretation:
    """Interpret one Engram sentence without mutating memory.

    Args:
        speaker: Existing ``sentences.speaker`` value.
        text: Original sentence text; never modified.
        date: Existing sentence date. Reserved for downstream temporal anchoring;
              P2.1 returns only temporal expressions found in the sentence.
        context: Optional adjacent conversational context. Currently used only
                 to flag context-dependent utterances for review, not to invent
                 missing meaning.
    """
    del date  # The sentence date is preserved by the caller; no date resolution in P2.1.
    raw = text or ""
    clean = raw.strip()
    role = _speaker_role(speaker, people)
    signals: list[str] = []

    if not clean:
        return Interpretation(speaker, role, "unclear", "unknown", "unknown", "unknown", "unknown", (), (), 0.0, True)

    # Speech act: the order is intentional; explicit corrections and requests
    # must not be swallowed by broad question/assertion patterns.
    if _matches(clean, _PATTERNS["correction"]):
        speech_act = "correction"; signals.append("correction_marker")
    elif _matches(clean, _PATTERNS["greeting"]):
        speech_act = "greeting"
    elif _matches(clean, _PATTERNS["thanks"]):
        speech_act = "thanks"
    elif _matches(clean, _PATTERNS["acknowledgement"]):
        speech_act = "acknowledgement"
    elif _matches(clean, _PATTERNS["request"]):
        speech_act = "request"; signals.append("request_marker")
    elif _matches(clean, _PATTERNS["proposal"]):
        speech_act = "proposal"; signals.append("proposal_marker")
    elif _matches(clean, _PATTERNS["commitment"]):
        speech_act = "commitment"; signals.append("commitment_marker")
    elif _matches(clean, _PATTERNS["question"]):
        speech_act = "question"
    else:
        speech_act = "assertion"

    # Content kind is a separate axis from speech act.
    priority = ("decision", "preference", "goal", "instruction", "plan", "emotion", "hypothesis", "belief")
    content_kind = next((k for k in priority if _matches(clean, _PATTERNS[k])), "unknown")
    if content_kind == "unknown":
        if speech_act in {"question", "request"}:
            content_kind = "question" if speech_act == "question" else "instruction_request"
        elif speech_act in {"greeting", "thanks", "acknowledgement"}:
            content_kind = "social"
        elif speech_act == "correction":
            content_kind = "correction"
        elif re.search(r"\b(i|we)\s+(did|have done|had|went|underwent|performed|completed|experienced)\b", clean, re.I):
            content_kind = "experience"
        elif speech_act in {"commitment", "proposal"}:
            content_kind = "plan"
        elif speech_act == "assertion":
            content_kind = "claim"

    if _UNCERTAIN.search(clean):
        epistemic = "uncertain"; signals.append("uncertainty_marker")
    elif _CONDITIONAL.search(clean):
        epistemic = "conditional"; signals.append("conditional_marker")
    elif _QUOTED.search(clean):
        epistemic = "quoted"; signals.append("quoted_text")
    else:
        epistemic = "asserted"

    polarity = "negative" if _NEGATION.search(clean) else "positive"
    if polarity == "negative": signals.append("negation_marker")

    # Keep the actual speaker separate from the source of a reported claim.
    # E.g. a user saying "Dr Rao said X" is still a user utterance, but the
    # claim source is reported as a third party.
    if _THIRD_PARTY.search(clean):
        attribution = "third_party_reported"
        signals.append("reported_source_marker")
    elif _QUOTED.search(clean):
        attribution = "quoted_or_external"
    elif role in {"user", "assistant"}:
        attribution = "speaker"
    else:
        attribution = "unknown"

    # Shape of the sentence, for the learning gate: small talk has no first person, and a bare "it" needs the sentence before it.
    n_words = len(clean.split())
    if not _FIRST_PERSON.search(clean): signals.append("no_first_person")
    if n_words <= 3: signals.append("short_reply")
    expletive = re.match(r"^(it|it's|it is|it was)\s+(\w+\s+){0,3}?(to|that|when|how)\b", clean, re.I)  # "It's tough to pick" points at nothing earlier
    if (_BARE_START.search(clean) and not expletive) or _BARE_OBJECT.search(clean): signals.append("bare_reference")
    if role == "assistant" and (clean.endswith(":") or re.fullmatch(r"\*\*[^*]+\*\*[.:]?", clean) or n_words < 5
                                or re.match(r"^(good luck|have fun|and most importantly|i apologi[sz]e|here (is|are)|now,? let's|i am always|i'm always|i hope|feel free|let me know|happy to help|you're welcome|enjoy)\b", clean, re.I)):
        signals.append("assistant_filler")

    temporal = tuple(dict.fromkeys(m.group(0) for m in _TIME.finditer(clean)))
    if temporal: signals.append("temporal_expression")
    if context is not None and re.search(r"\b(it|this|that|they|he|she|there|same|other one)\b", clean, re.I):
        signals.append("context_dependent_reference")

    # Confidence describes the strength of this classification, not truth.
    strong = speech_act in {"greeting", "thanks", "acknowledgement", "correction", "request", "question"} or content_kind != "unknown"
    confidence = 0.88 if strong else 0.66
    needs_review = confidence < 0.75 or epistemic in {"uncertain", "conditional", "quoted"} or "context_dependent_reference" in signals
    if role == "other":
        needs_review = True

    return Interpretation(
        speaker=speaker,
        speaker_role=role,
        speech_act=speech_act,
        content_kind=content_kind,
        epistemic_stance=epistemic,
        attribution=attribution,
        polarity=polarity,
        temporal_markers=temporal,
        signals=tuple(signals),
        confidence=confidence,
        needs_review=needs_review,
        word_count=n_words,
    )


_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+[\"'”’)]*|\n|$)")


def split_message(text: str) -> list[tuple[int, int, str]]:
    """A stored message into sentences, as (start, end, sentence) inside the exact original text. Nothing is rewritten."""
    out = []
    for m in _SENTENCE.finditer(text or ""):
        piece = m.group(0)
        lead = len(piece) - len(piece.lstrip())
        body = piece.strip()
        if body: out.append((m.start() + lead, m.start() + lead + len(body), body))
    return out


def interpret_message(speaker: str, text: str, date: str = "", people=()) -> list[tuple[int, int, Interpretation]]:
    """Each sentence of a message on its own, so a greeting does not hide the fact that follows it."""
    return [(a, b, interpret(speaker, body, date=date, people=people)) for a, b, body in split_message(text)]
