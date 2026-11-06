"""P6.1 append-only experience recorder.

Stores interaction *events*, not copied conversation text or learned weights.
Uses the existing Engram SQLite connection. Event payloads contain references and
structured signals; callers should not pass raw user/assistant text.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib, json, sqlite3, uuid
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS p6_experience_events (
    event_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL CHECK(event_type IN (
      'retrieval','memory_selected','memory_dismissed','feedback',
      'correction','confirmation','outcome','recall_failure'
    )),
    occurred_at TEXT NOT NULL,
    context_key TEXT,
    query_kind TEXT,
    candidate_refs TEXT NOT NULL DEFAULT '[]',
    selected_refs TEXT NOT NULL DEFAULT '[]',
    target_ref TEXT,
    signal TEXT,
    signal_value REAL,
    source_event_ref TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    schema_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS p6_events_context_time
 ON p6_experience_events(context_key, occurred_at);
CREATE INDEX IF NOT EXISTS p6_events_target_time
 ON p6_experience_events(target_ref, occurred_at);
CREATE INDEX IF NOT EXISTS p6_events_type_time
 ON p6_experience_events(event_type, occurred_at);
"""

ALLOWED_SIGNALS = {
    "helpful", "irrelevant", "wrong", "outdated", "not_about_me",
    "needs_context", "confirm", "corrected", "selected", "dismissed",
    "successful", "unsuccessful", "no_result"
}

class ExperienceError(ValueError):
    pass

@dataclass(frozen=True)
class ExperienceEvent:
    event_id: str
    event_type: str
    occurred_at: str
    context_key: str | None
    target_ref: str | None
    signal: str | None
    replayed: bool = False

def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ExperienceError(f"metadata must be JSON serializable: {e}") from None

def _refs(values: Iterable[str] | None, field: str) -> list[str]:
    result=[]
    for value in values or ():
        if not isinstance(value,str) or not value.strip():
            raise ExperienceError(f"{field} must contain non-empty string references")
        if value not in result: result.append(value)
    return result

class ExperienceRecorder:
    """Append-only recorder bound to Engram's existing SQLite connection."""

    def __init__(self, db: sqlite3.Connection):
        self.db=db
        self.db.executescript(SCHEMA)

    @staticmethod
    def context_key(query_kind: str | None, route: str | None = None,
                    scope: str | None = None) -> str | None:
        """Stable, content-free context bucket; intentionally excludes raw query text."""
        parts=[x.strip().lower() for x in (query_kind,route,scope) if isinstance(x,str) and x.strip()]
        if not parts: return None
        raw="|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def record(self, event_type: str, *, idempotency_key: str,
               context_key: str | None = None, query_kind: str | None = None,
               candidate_refs: Iterable[str] | None = None,
               selected_refs: Iterable[str] | None = None,
               target_ref: str | None = None, signal: str | None = None,
               signal_value: float | None = None, source_event_ref: str | None = None,
               metadata: dict[str, Any] | None = None,
               occurred_at: str | None = None) -> ExperienceEvent:
        allowed={"retrieval","memory_selected","memory_dismissed","feedback",
                 "correction","confirmation","outcome","recall_failure"}
        if event_type not in allowed: raise ExperienceError("unsupported event_type")
        if not idempotency_key or not idempotency_key.strip():
            raise ExperienceError("idempotency_key is required")
        if signal is not None and signal not in ALLOWED_SIGNALS:
            raise ExperienceError("unsupported signal")
        if signal_value is not None and not isinstance(signal_value,(int,float)):
            raise ExperienceError("signal_value must be numeric")
        if context_key is not None and (not isinstance(context_key,str) or not context_key.strip()):
            raise ExperienceError("context_key must be a non-empty string or None")
        if target_ref is not None and (not isinstance(target_ref,str) or not target_ref.strip()):
            raise ExperienceError("target_ref must be a non-empty reference or None")
        candidates=_refs(candidate_refs,"candidate_refs")
        selected=_refs(selected_refs,"selected_refs")
        meta=metadata or {}
        if not isinstance(meta,dict): raise ExperienceError("metadata must be a dictionary")
        # Prevent accidental transcript storage. The recorder stores references and small metadata only.
        forbidden={"text","raw_text","transcript","utterance","prompt","response","message"}
        if forbidden.intersection(k.lower() for k in meta):
            raise ExperienceError("metadata must not contain raw conversation text fields; store source references instead")
        meta_json=_json(meta)
        ts=occurred_at or datetime.now(timezone.utc).isoformat()
        try: datetime.fromisoformat(ts.replace("Z","+00:00"))
        except ValueError: raise ExperienceError("occurred_at must be ISO-8601") from None
        prior=self.db.execute("SELECT event_id,event_type,occurred_at,context_key,target_ref,signal FROM p6_experience_events WHERE idempotency_key=?",(idempotency_key,)).fetchone()
        if prior:
            return ExperienceEvent(*prior,replayed=True)
        eid="p6e_"+uuid.uuid4().hex
        try:
            self.db.execute("""INSERT INTO p6_experience_events
              (event_id,idempotency_key,event_type,occurred_at,context_key,query_kind,
               candidate_refs,selected_refs,target_ref,signal,signal_value,
               source_event_ref,metadata,schema_version)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
              (eid,idempotency_key,event_type,ts,context_key,query_kind,
               _json(candidates),_json(selected),target_ref,signal,signal_value,
               source_event_ref,meta_json))
            self.db.commit()
        except sqlite3.IntegrityError:
            self.db.rollback()
            prior=self.db.execute("SELECT event_id,event_type,occurred_at,context_key,target_ref,signal FROM p6_experience_events WHERE idempotency_key=?",(idempotency_key,)).fetchone()
            if prior: return ExperienceEvent(*prior,replayed=True)
            raise
        except Exception:
            self.db.rollback()
            raise
        return ExperienceEvent(eid,event_type,ts,context_key,target_ref,signal,False)

    def record_retrieval(self, *, idempotency_key: str, context_key: str | None,
                         query_kind: str | None, candidate_refs: Iterable[str],
                         selected_refs: Iterable[str] = (), metadata: dict | None = None):
        return self.record("retrieval",idempotency_key=idempotency_key,
            context_key=context_key,query_kind=query_kind,candidate_refs=candidate_refs,
            selected_refs=selected_refs,metadata=metadata)

    def record_feedback(self, *, idempotency_key: str, target_ref: str,
                        signal: str, context_key: str | None = None,
                        source_event_ref: str | None = None, metadata: dict | None = None):
        if signal not in {"helpful","irrelevant","wrong","outdated","not_about_me",
                          "needs_context","confirm"}:
            raise ExperienceError("feedback signal must be a recognised P2 feedback value")
        return self.record("feedback",idempotency_key=idempotency_key,
            target_ref=target_ref,signal=signal,context_key=context_key,
            source_event_ref=source_event_ref,metadata=metadata)

    def record_correction(self, *, idempotency_key: str, target_ref: str,
                          source_event_ref: str | None = None,
                          context_key: str | None = None, metadata: dict | None = None):
        return self.record("correction",idempotency_key=idempotency_key,
            target_ref=target_ref,signal="corrected",context_key=context_key,
            source_event_ref=source_event_ref,metadata=metadata)

    def events_for(self, target_ref: str, limit: int = 100) -> list[dict]:
        if limit < 1 or limit > 10000: raise ExperienceError("limit must be 1..10000")
        rows=self.db.execute("""SELECT event_id,event_type,occurred_at,context_key,
          query_kind,candidate_refs,selected_refs,target_ref,signal,signal_value,
          source_event_ref,metadata FROM p6_experience_events
          WHERE target_ref=? ORDER BY occurred_at DESC,event_id DESC LIMIT ?""",(target_ref,limit))
        keys=("event_id","event_type","occurred_at","context_key","query_kind",
              "candidate_refs","selected_refs","target_ref","signal","signal_value",
              "source_event_ref","metadata")
        out=[]
        for row in rows:
            d=dict(zip(keys,row))
            for k in ("candidate_refs","selected_refs","metadata"): d[k]=json.loads(d[k])
            out.append(d)
        return out

    def count(self, event_type: str | None = None) -> int:
        if event_type is None:
            return self.db.execute("SELECT count(*) FROM p6_experience_events").fetchone()[0]
        return self.db.execute("SELECT count(*) FROM p6_experience_events WHERE event_type=?",(event_type,)).fetchone()[0]
