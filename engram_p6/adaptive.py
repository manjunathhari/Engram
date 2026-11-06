"""P6.3 Context-conditioned Adaptive Scorer.

Learns bounded retrieval adjustments from explicit P6.1 feedback events.
It writes versioned policy snapshots, not changes to facts or retrieval code.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json, math, sqlite3, uuid
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS p6_scoring_policies (
 policy_id TEXT PRIMARY KEY,
 version INTEGER NOT NULL,
 context_key TEXT NOT NULL,
 target_ref TEXT NOT NULL,
 positive_count INTEGER NOT NULL,
 negative_count INTEGER NOT NULL,
 posterior_mean REAL NOT NULL,
 confidence REAL NOT NULL,
 adjustment REAL NOT NULL,
 multiplier REAL NOT NULL,
 evidence_count INTEGER NOT NULL,
 source_event_count INTEGER NOT NULL,
 assessment TEXT NOT NULL,
 created_at TEXT NOT NULL,
 algorithm TEXT NOT NULL,
 config_json TEXT NOT NULL,
 UNIQUE(version, context_key, target_ref)
);
CREATE INDEX IF NOT EXISTS p6_policy_context_version
 ON p6_scoring_policies(context_key, version);
CREATE INDEX IF NOT EXISTS p6_policy_target
 ON p6_scoring_policies(target_ref, version);
CREATE TABLE IF NOT EXISTS p6_policy_versions (
 version INTEGER PRIMARY KEY,
 context_key TEXT NOT NULL,
 created_at TEXT NOT NULL,
 algorithm TEXT NOT NULL,
 config_json TEXT NOT NULL,
 source_max_event_id TEXT,
 policy_count INTEGER NOT NULL,
 note TEXT NOT NULL DEFAULT ''
);
"""

POSITIVE={"helpful","confirm","successful"}
NEGATIVE={"irrelevant","wrong","outdated","not_about_me","corrected","unsuccessful","no_result"}

class AdaptiveScorerError(ValueError):
    pass

@dataclass(frozen=True)
class AdaptiveScore:
    target_ref: str
    context_key: str
    positive_count: int
    negative_count: int
    posterior_mean: float
    confidence: float
    adjustment: float
    multiplier: float
    evidence_count: int
    assessment: str

    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class PolicySnapshot:
    version: int
    context_key: str
    policy_count: int
    created_at: str
    algorithm: str

class AdaptiveScorer:
    """Builds immutable, versioned per-memory/per-context retrieval policies."""

    ALGORITHM="beta_shrinkage_v1"

    def __init__(self, db: sqlite3.Connection, *, prior_alpha: float=1.0,
                 prior_beta: float=1.0, confidence_scale: float=4.0,
                 max_adjustment: float=0.20, min_multiplier: float=0.80,
                 max_multiplier: float=1.20, min_evidence: int=2):
        if prior_alpha<=0 or prior_beta<=0: raise AdaptiveScorerError("prior parameters must be positive")
        if confidence_scale<=0: raise AdaptiveScorerError("confidence_scale must be positive")
        if not 0<max_adjustment<1: raise AdaptiveScorerError("max_adjustment must be between 0 and 1")
        if min_multiplier<=0 or max_multiplier<min_multiplier: raise AdaptiveScorerError("invalid multiplier bounds")
        if min_evidence<1: raise AdaptiveScorerError("min_evidence must be >= 1")
        self.db=db
        self.alpha=float(prior_alpha); self.beta=float(prior_beta)
        self.confidence_scale=float(confidence_scale)
        self.max_adjustment=float(max_adjustment)
        self.min_multiplier=float(min_multiplier); self.max_multiplier=float(max_multiplier)
        self.min_evidence=int(min_evidence)
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "p6_experience_events" not in tables:
            raise AdaptiveScorerError("P6.1 table p6_experience_events is missing")
        cols={r[1] for r in db.execute("PRAGMA table_info(p6_experience_events)")}
        required={"event_id","event_type","occurred_at","context_key","candidate_refs",
                  "selected_refs","target_ref","signal","source_event_ref"}
        if not required.issubset(cols):
            raise AdaptiveScorerError("P6.1 event schema is incomplete")
        self.db.executescript(SCHEMA)

    def _config(self):
        return {"prior_alpha":self.alpha,"prior_beta":self.beta,
          "confidence_scale":self.confidence_scale,"max_adjustment":self.max_adjustment,
          "min_multiplier":self.min_multiplier,"max_multiplier":self.max_multiplier,
          "min_evidence":self.min_evidence}

    @staticmethod
    def _refs(raw):
        try: vals=json.loads(raw or "[]")
        except Exception: raise AdaptiveScorerError("invalid P6.1 reference JSON") from None
        if not isinstance(vals,list) or any(not isinstance(v,str) for v in vals):
            raise AdaptiveScorerError("P6.1 references must be arrays of strings")
        return vals

    def _feedback_counts(self, context_key: str):
        rows=self.db.execute("""SELECT target_ref,signal,event_type
          FROM p6_experience_events
          WHERE context_key=? AND target_ref IS NOT NULL
          ORDER BY occurred_at,event_id""",(context_key,))
        counts={}
        source_events=set()
        for target,signal,event_type in rows:
            if not target: continue
            sig=signal
            if event_type=="correction" and not sig: sig="corrected"
            if event_type=="confirmation" and not sig: sig="confirm"
            if event_type=="recall_failure" and not sig: sig="no_result"
            if event_type=="outcome" and not sig: sig="successful"
            if sig not in POSITIVE|NEGATIVE: continue
            item=counts.setdefault(target,{"positive":0,"negative":0,"events":0})
            if sig in POSITIVE: item["positive"]+=1
            else: item["negative"]+=1
            item["events"]+=1
            source_events.add(target)
        return counts

    def score(self, target_ref: str, context_key: str) -> AdaptiveScore:
        if not target_ref or not target_ref.strip(): raise AdaptiveScorerError("target_ref is required")
        if not context_key or not context_key.strip(): raise AdaptiveScorerError("context_key is required")
        counts=self._feedback_counts(context_key).get(target_ref,{"positive":0,"negative":0,"events":0})
        p,n=counts["positive"],counts["negative"]
        total=p+n
        posterior=(self.alpha+p)/(self.alpha+self.beta+total)
        confidence=total/(total+self.confidence_scale) if total else 0.0
        if total < self.min_evidence:
            adjustment=0.0
            assessment="insufficient_evidence"
        else:
            adjustment=self.max_adjustment*(2*posterior-1)*confidence
            assessment=("positive_adaptation" if adjustment>0 else
                        "negative_adaptation" if adjustment<0 else "neutral")
        multiplier=min(self.max_multiplier,max(self.min_multiplier,1.0+adjustment))
        return AdaptiveScore(target_ref,context_key,p,n,round(posterior,6),
          round(confidence,6),round(adjustment,6),round(multiplier,6),total,assessment)

    def build_policy(self, context_key: str, *, note: str="") -> PolicySnapshot:
        """Create a new immutable version for every memory with explicit feedback in this context."""
        if not context_key or not context_key.strip(): raise AdaptiveScorerError("context_key is required")
        counts=self._feedback_counts(context_key)
        scores=[self.score(ref,context_key) for ref in sorted(counts)]
        now=datetime.now(timezone.utc).isoformat()
        config=self._config()
        version=self.db.execute("SELECT COALESCE(MAX(version),0)+1 FROM p6_policy_versions").fetchone()[0]
        max_event=self.db.execute("SELECT event_id FROM p6_experience_events WHERE context_key=? ORDER BY occurred_at DESC,event_id DESC LIMIT 1",(context_key,)).fetchone()
        source_max=max_event[0] if max_event else None
        policy_id_prefix=f"p6p{version}_"
        sp="p6_policy_build"
        self.db.execute(f"SAVEPOINT {sp}")
        try:
            self.db.execute("""INSERT INTO p6_policy_versions
              (version,context_key,created_at,algorithm,config_json,source_max_event_id,policy_count,note)
              VALUES(?,?,?,?,?,?,?,?)""",(version,context_key,now,self.ALGORITHM,json.dumps(config,sort_keys=True),
              source_max,len(scores),note))
            for s in scores:
                self.db.execute("""INSERT INTO p6_scoring_policies
                  (policy_id,version,context_key,target_ref,positive_count,negative_count,
                   posterior_mean,confidence,adjustment,multiplier,evidence_count,
                   source_event_count,assessment,created_at,algorithm,config_json)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (policy_id_prefix+uuid.uuid4().hex[:12],version,context_key,s.target_ref,
                   s.positive_count,s.negative_count,s.posterior_mean,s.confidence,
                   s.adjustment,s.multiplier,s.evidence_count,s.evidence_count,
                   s.assessment,now,self.ALGORITHM,json.dumps(config,sort_keys=True)))
            self.db.execute(f"RELEASE {sp}")
            self.db.commit()
        except Exception:
            self.db.execute(f"ROLLBACK TO {sp}")
            self.db.execute(f"RELEASE {sp}")
            self.db.rollback()
            raise
        return PolicySnapshot(version,context_key,len(scores),now,self.ALGORITHM)

    def get_policy(self, target_ref: str, context_key: str, version: int | None=None):
        if version is None:
            row=self.db.execute("""SELECT version,multiplier,adjustment,posterior_mean,
              confidence,evidence_count,positive_count,negative_count,assessment
              FROM p6_scoring_policies WHERE target_ref=? AND context_key=?
              ORDER BY version DESC LIMIT 1""",(target_ref,context_key)).fetchone()
        else:
            row=self.db.execute("""SELECT version,multiplier,adjustment,posterior_mean,
              confidence,evidence_count,positive_count,negative_count,assessment
              FROM p6_scoring_policies WHERE target_ref=? AND context_key=? AND version=?""",
              (target_ref,context_key,version)).fetchone()
        if not row: return None
        return dict(zip(("version","multiplier","adjustment","posterior_mean","confidence",
          "evidence_count","positive_count","negative_count","assessment"),row))

    def policy_for_context(self, context_key: str, version: int | None=None) -> dict[str,dict]:
        if version is None:
            version=self.db.execute("SELECT MAX(version) FROM p6_policy_versions WHERE context_key=?",(context_key,)).fetchone()[0]
        if version is None: return {}
        rows=self.db.execute("""SELECT target_ref,version,multiplier,adjustment,posterior_mean,
          confidence,evidence_count,positive_count,negative_count
          FROM p6_scoring_policies WHERE context_key=? AND version=?""",(context_key,version))
        keys=("version","multiplier","adjustment","posterior_mean","confidence",
              "evidence_count","positive_count","negative_count")
        return {r[0]:dict(zip(keys,r[1:])) for r in rows}

    def apply_multiplier(self, base_score: float, target_ref: str,
                         context_key: str, version: int | None=None) -> float:
        """Pure scoring helper. Caller decides how/where to blend into retrieval rank."""
        if not isinstance(base_score,(int,float)) or not math.isfinite(base_score):
            raise AdaptiveScorerError("base_score must be finite numeric")
        policy=self.get_policy(target_ref,context_key,version)
        return float(base_score) if policy is None else float(base_score)*policy["multiplier"]

    def versions(self, context_key: str | None=None):
        if context_key is None:
            rows=self.db.execute("SELECT version,created_at,algorithm,policy_count,note FROM p6_policy_versions ORDER BY version")
        else:
            rows=self.db.execute("SELECT version,created_at,algorithm,policy_count,note FROM p6_policy_versions WHERE context_key=? ORDER BY version",(context_key,))
        # context_key is stored in policy rows; versions are globally numbered.
        return [dict(zip(("version","created_at","algorithm","policy_count","note"),r)) for r in rows]
