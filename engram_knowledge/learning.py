"""P2.2: deterministic learning-policy evaluator for Engram P2.1 outputs.

This module decides whether an interpretation is worth proposing to memory.
It does NOT create facts, write to P1, resolve entities, or verify truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LearningDecision:
    decision: str                 # candidate | review | ignore
    reason_codes: tuple[str, ...]
    content_kind: str
    retention_hint: str            # durable | project | episodic | none
    proposed_source_type: str      # explicit | derived | inferred | none
    proposed_validation: str       # pending | uncertain | rejected
    requires_context: bool
    requires_human_review: bool
    score: float                   # policy priority, NOT truth probability
    keep_as: str = "none"          # "assistant_statement": kept as said by the assistant, never as a fact about the user

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["reason_codes"] = list(self.reason_codes)
        return out


DURABLE = {"preference", "decision", "goal", "instruction"}
PROJECT = {"plan"}
EPISODIC = {"experience"}
CLAIM_LIKE = {"claim", "belief"}
SUPPORTED = DURABLE | PROJECT | EPISODIC | CLAIM_LIKE | {"hypothesis"}
SOCIAL = {"social"}
NON_LEARNING_ACTS = {"question", "request", "greeting", "thanks", "acknowledgement", "unclear"}


def _get(x: Any, key: str, default: Any = None) -> Any:
    if isinstance(x, Mapping):
        return x.get(key, default)
    return getattr(x, key, default)


def evaluate(interpretation: Any, *, has_context: bool = False,
             min_interpreter_confidence: float = 0.75) -> LearningDecision:
    """Apply P2.2 policy to a P2.1 Interpretation or its ``to_dict()`` output.

    The result is a *proposal decision*, never a database write. ``score`` is
    an explainable prioritisation heuristic and must not be read as truth or
    probability. The original sentence and its speaker remain authoritative.
    """
    reasons: list[str] = []
    role = _get(interpretation, "speaker_role", "other")
    act = _get(interpretation, "speech_act", "unclear")
    kind = _get(interpretation, "content_kind", "unknown")
    stance = _get(interpretation, "epistemic_stance", "unknown")
    attribution = _get(interpretation, "attribution", "unknown")
    confidence = float(_get(interpretation, "confidence", 0.0) or 0.0)
    needs_review = bool(_get(interpretation, "needs_review", True))
    signals = set(_get(interpretation, "signals", ()) or ())
    temporal = tuple(_get(interpretation, "temporal_markers", ()) or ())

    def result(decision: str, *codes: str, retention: str = "none",
               source: str = "none", validation: str = "rejected",
               context: bool = False, human: bool = False, score: float = 0.0):
        return LearningDecision(decision, tuple(dict.fromkeys(codes)), kind,
                                retention, source, validation, context, human,
                                round(max(0.0, min(1.0, score)), 3))

    # The system must never learn the assistant's own suggestions as user memory.
    if role in {"assistant", "system", "tool"}:
        out = result("ignore", "non_user_speaker", "not_user_memory")
        content = kind not in SOCIAL | {"question", "instruction_request", "unknown"} and act not in NON_LEARNING_ACTS | {"request"}
        if role == "assistant" and content:  # what the assistant told the user can be asked about later
            from dataclasses import replace
            return replace(out, reason_codes=out.reason_codes + ("kept_as_assistant_statement",), keep_as="assistant_statement", retention_hint="episodic")
        return out
    if role != "user":
        return result("review", "speaker_role_unknown", human=True)

    if kind in SOCIAL or act in {"greeting", "thanks", "acknowledgement"}:
        return result("ignore", "social_or_acknowledgement")
    if act in NON_LEARNING_ACTS or kind in {"question", "instruction_request", "unknown"}:
        return result("ignore", "interaction_not_memory_content")

    # Corrections are important but need a target. P2.2 must not guess which
    # prior fact is being corrected; later context/consolidation can resolve it.
    if act == "correction" or kind == "correction" or "correction_marker" in signals:
        return result("review", "correction_requires_target_resolution",
                      context=True, human=True, validation="uncertain", score=0.75)

    if "context_dependent_reference" in signals:
        return result("review", "unresolved_context_reference", context=True,
                      human=True, validation="uncertain", score=0.45)
    if attribution in {"third_party_reported", "quoted_or_external"} or stance == "quoted":
        return result("review", "externally_attributed_statement",
                      context=True, human=True, validation="uncertain", score=0.35)
    if stance in {"uncertain", "conditional", "unknown"} or kind == "hypothesis":
        return result("review", "uncertain_or_conditional_content",
                      human=True, validation="uncertain", score=0.3)
    if attribution != "speaker":
        return result("review", "attribution_not_confirmed", human=True,
                      validation="uncertain", score=0.3)
    if confidence < min_interpreter_confidence or needs_review:
        return result("review", "interpreter_flagged_for_review",
                      human=True, validation="uncertain", score=0.25)
    if kind not in SUPPORTED:
        return result("review", "unsupported_content_kind", human=True,
                      validation="uncertain", score=0.2)

    if kind in DURABLE:
        retention, score = "durable", 0.9
    elif kind in PROJECT:
        retention, score = "project", 0.8
    elif kind in EPISODIC:
        retention, score = "episodic", 0.65
    else:
        retention, score = "durable", 0.6

    if temporal and kind == "plan":
        reasons.append("time_bounded_plan")
    if _get(interpretation, "polarity", "positive") == "negative":
        reasons.append("explicit_negative_statement")

    reasons.insert(0, "user_authored_explicit_content")
    reasons.append("eligible_for_learning_candidate")
    return result("candidate", *reasons, retention=retention,
                  source="explicit", validation="pending", score=score)
