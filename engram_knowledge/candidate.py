"""P2.3: package interpretation and learning policy into a provenance-safe envelope.

No database writes occur here. The original sentence is copied verbatim and
identified by its existing Engram sentence ID (sid). Candidate identity is
deterministic for idempotent downstream processing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .interpreter import interpret
from .learning import evaluate

ENVELOPE_SCHEMA = "engram.learning_candidate"
ENVELOPE_VERSION = 1
_ALLOWED_DECISIONS = {"candidate", "review"}


@dataclass(frozen=True)
class CandidateEnvelope:
    candidate_id: str
    schema: str
    schema_version: int
    source: dict[str, Any]
    original_text: str
    interpretation: dict[str, Any]
    learning: dict[str, Any]
    lifecycle: str

    def to_dict(self) -> dict[str, Any]:
        # Return fresh nested structures so callers cannot mutate the envelope.
        return json.loads(json.dumps(asdict(self), ensure_ascii=False))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent)


@dataclass(frozen=True)
class CandidateBuildResult:
    status: str                 # created | ignored
    envelope: CandidateEnvelope | None
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "envelope": self.envelope.to_dict() if self.envelope else None,
            "reason_codes": list(self.reason_codes),
        }


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("interpretation and learning decision must be mappings or expose to_dict()")


def _validate_source(source: Mapping[str, Any]) -> dict[str, Any]:
    required = ("sid", "speaker", "text")
    missing = [k for k in required if k not in source]
    if missing:
        raise ValueError(f"source is missing required field(s): {', '.join(missing)}")
    sid = source["sid"]
    if sid is None or isinstance(sid, bool) or not str(sid).strip():
        raise ValueError("source.sid must be a non-empty existing Engram sentence ID")
    if not isinstance(source["speaker"], str) or not source["speaker"].strip():
        raise ValueError("source.speaker must be a non-empty string")
    if not isinstance(source["text"], str):
        raise TypeError("source.text must be the original sentence string")
    # Preserve only known provenance fields. sid is kept in its supplied type.
    out = {
        "sid": sid,
        "chat": source.get("chat"),
        "speaker": source["speaker"],
        "text": source["text"],
        "date": source.get("date", ""),
        "ref": source.get("ref", ""),
        "caption": source.get("caption", ""),
    }
    if source.get("span") is not None:  # where the sentence sits inside the stored message, and that message's digest
        start, end = source["span"]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end): raise ValueError("source.span must be (start, end) inside the stored message")
        out["span"] = [start, end]
        out["message_sha256"] = str(source.get("message_sha256", ""))
    for key in ("date", "ref", "caption"):
        if out[key] is None:
            out[key] = ""
        if not isinstance(out[key], str):
            raise TypeError(f"source.{key} must be a string")
    if out["chat"] is not None and (isinstance(out["chat"], bool) or not isinstance(out["chat"], (int, str))):
        raise TypeError("source.chat must be an integer, string, or None")
    return out


def _stable_id(source: Mapping[str, Any], original_text: str, interp: Mapping[str, Any],
               learning: Mapping[str, Any]) -> str:
    # Deterministic across runs; includes the sentence identity and policy result.
    payload = {
        "schema": ENVELOPE_SCHEMA,
        "version": ENVELOPE_VERSION,
        "sid": str(source["sid"]),
        **({"span": list(source["span"])} if source.get("span") else {}),
        "text": original_text,
        "interpretation": interp,
        "learning": learning,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "ec_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def build_candidate(source: Mapping[str, Any], interpretation: Any,
                    learning_decision: Any, *, keep_assistant: bool = False) -> CandidateBuildResult:
    """Build an envelope for candidate/review outcomes; ignore outcomes are excluded.

    The builder validates that interpretation speaker matches the stored source
    speaker. It does not reinterpret or rewrite the sentence.
    """
    src = _validate_source(source)
    original = source["text"]
    interp = _as_dict(interpretation)
    learning = _as_dict(learning_decision)

    if interp.get("speaker") != src["speaker"]:
        raise ValueError("interpretation.speaker does not match source.speaker")
    if learning.get("content_kind") != interp.get("content_kind"):
        raise ValueError("learning.content_kind does not match interpretation.content_kind")

    decision = learning.get("decision")
    if decision == "ignore" and keep_assistant and learning.get("keep_as") == "assistant_statement":
        decision = "assistant_statement"
    elif decision == "ignore":
        return CandidateBuildResult(
            status="ignored",
            envelope=None,
            reason_codes=tuple(learning.get("reason_codes", ())) or ("learning_policy_ignore",),
        )
    if decision not in _ALLOWED_DECISIONS | {"assistant_statement"}:
        raise ValueError("learning.decision must be candidate, review, or ignore")

    # P2.3 is packaging, not a second policy engine. Carry P2.1/P2.2 outputs
    # as-is, with a source-text digest for integrity checking.
    src["text_sha256"] = hashlib.sha256(original.encode("utf-8")).hexdigest()
    lifecycle = {"candidate": "proposed", "review": "needs_review", "assistant_statement": "assistant_statement"}[decision]
    envelope_id = _stable_id(src, original, interp, learning)
    envelope = CandidateEnvelope(
        candidate_id=envelope_id,
        schema=ENVELOPE_SCHEMA,
        schema_version=ENVELOPE_VERSION,
        source=src,
        original_text=original,
        interpretation=interp,
        learning=learning,
        lifecycle=lifecycle,
    )
    return CandidateBuildResult(
        status="created",
        envelope=envelope,
        reason_codes=tuple(learning.get("reason_codes", ())),
    )


def prepare_sentence(source: Mapping[str, Any], *, context: str | None = None) -> CandidateBuildResult:
    """Run P2.1 -> P2.2 -> P2.3 for one already-persisted Engram sentence."""
    src = _validate_source(source)
    parsed = interpret(src["speaker"], source["text"], date=src["date"], context=context)
    decision = evaluate(parsed, has_context=context is not None)
    return build_candidate(src, parsed, decision)
