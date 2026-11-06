"""P6.4 Temporal Adaptation.

Re-weights explicit feedback by age and detects context-specific usefulness drift.
It produces a bounded temporal multiplier; it never changes facts or validity dates.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import json, math, sqlite3
from typing import Optional

POSITIVE={"helpful","confirm","successful"}
NEGATIVE={"irrelevant","wrong","outdated","not_about_me","corrected","unsuccessful","no_result"}

class TemporalAdaptationError(ValueError):
    pass

@dataclass(frozen=True)
class TemporalScore:
    target_ref: str
    context_key: str
    as_of: str
    weighted_positive: float
    weighted_negative: float
    posterior_mean: float
    evidence_confidence: float
    temporal_multiplier: float
    recent_positive: int
    recent_negative: int
    previous_positive: int
    previous_negative: int
    trend_delta: float
    trend_label: str
    evidence_age_days: Optional[float]
    assessment: str

    def to_dict(self): return asdict(self)

class TemporalAdapter:
    """Time-decayed, context-conditioned feedback evaluator."""

    def __init__(self, db: sqlite3.Connection, *, half_life_days: float=90.0,
                 prior_alpha: float=1.0, prior_beta: float=1.0,
                 confidence_scale: float=4.0, max_adjustment: float=0.20,
                 min_multiplier: float=0.80, max_multiplier: float=1.20,
                 min_weighted_evidence: float=1.0,
                 trend_window_days: int=30, trend_threshold: float=0.15):
        if half_life_days<=0: raise TemporalAdaptationError("half_life_days must be positive")
        if prior_alpha<=0 or prior_beta<=0: raise TemporalAdaptationError("prior parameters must be positive")
        if confidence_scale<=0: raise TemporalAdaptationError("confidence_scale must be positive")
        if not 0<max_adjustment<1: raise TemporalAdaptationError("max_adjustment must be between 0 and 1")
        if min_multiplier<=0 or max_multiplier<min_multiplier: raise TemporalAdaptationError("invalid multiplier bounds")
        if min_weighted_evidence<0: raise TemporalAdaptationError("min_weighted_evidence cannot be negative")
        if trend_window_days<1: raise TemporalAdaptationError("trend_window_days must be >= 1")
        self.db=db
        self.half_life=float(half_life_days)
        self.alpha=float(prior_alpha); self.beta=float(prior_beta)
        self.confidence_scale=float(confidence_scale)
        self.max_adjustment=float(max_adjustment)
        self.min_multiplier=float(min_multiplier); self.max_multiplier=float(max_multiplier)
        self.min_weight=float(min_weighted_evidence)
        self.window=int(trend_window_days); self.trend_threshold=float(trend_threshold)
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "p6_experience_events" not in tables:
            raise TemporalAdaptationError("P6.1 table p6_experience_events is missing")
        cols={r[1] for r in db.execute("PRAGMA table_info(p6_experience_events)")}
        required={"event_id","event_type","occurred_at","context_key","target_ref","signal"}
        if not required.issubset(cols):
            raise TemporalAdaptationError("P6.1 event schema is incomplete")

    @staticmethod
    def _datetime(value: str) -> datetime:
        try:
            dt=datetime.fromisoformat(value.replace("Z","+00:00"))
        except (AttributeError,ValueError):
            raise TemporalAdaptationError(f"invalid ISO-8601 timestamp: {value!r}") from None
        if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _events(self,target_ref,context_key):
        return self.db.execute("""SELECT event_id,event_type,occurred_at,signal
          FROM p6_experience_events
          WHERE target_ref=? AND context_key=?
          ORDER BY occurred_at,event_id""",(target_ref,context_key)).fetchall()

    def evaluate(self,target_ref: str,context_key: str,*,as_of: str|None=None) -> TemporalScore:
        if not target_ref or not target_ref.strip(): raise TemporalAdaptationError("target_ref is required")
        if not context_key or not context_key.strip(): raise TemporalAdaptationError("context_key is required")
        now=self._datetime(as_of) if as_of else datetime.now(timezone.utc)
        rows=self._events(target_ref,context_key)
        wp=wn=0.0
        recent_p=recent_n=previous_p=previous_n=0
        ages=[]
        recent_start=now-timedelta(days=self.window)
        previous_start=now-timedelta(days=2*self.window)
        ln2=math.log(2)
        for event_id,event_type,occurred_at,signal in rows:
            if event_type=="correction" and not signal: signal="corrected"
            if event_type=="confirmation" and not signal: signal="confirm"
            if event_type=="recall_failure" and not signal: signal="no_result"
            if event_type=="outcome" and not signal: signal="successful"
            if signal not in POSITIVE|NEGATIVE: continue
            dt=self._datetime(occurred_at)
            age=(now-dt).total_seconds()/86400.0
            # Future-dated records are excluded rather than given >1 weight.
            if age<0: continue
            ages.append(age)
            weight=math.exp(-ln2*age/self.half_life)
            if signal in POSITIVE: wp+=weight
            else: wn+=weight
            if dt>=recent_start:
                if signal in POSITIVE: recent_p+=1
                else: recent_n+=1
            elif previous_start<=dt<recent_start:
                if signal in POSITIVE: previous_p+=1
                else: previous_n+=1

        total=wp+wn
        posterior=(self.alpha+wp)/(self.alpha+self.beta+total)
        confidence=total/(total+self.confidence_scale) if total else 0.0
        if total < self.min_weight:
            adjustment=0.0
            assessment="insufficient_temporal_evidence"
        else:
            adjustment=self.max_adjustment*(2*posterior-1)*confidence
            assessment="temporally_positive" if adjustment>0 else "temporally_negative" if adjustment<0 else "temporally_neutral"

        # Compare two adjacent windows. Laplace smoothing prevents empty-window extremes.
        recent_mean=(1+recent_p)/(2+recent_p+recent_n)
        previous_mean=(1+previous_p)/(2+previous_p+previous_n)
        trend=recent_mean-previous_mean
        if recent_p+recent_n+previous_p+previous_n < 2:
            trend_label="insufficient_trend_evidence"
        elif trend>=self.trend_threshold: trend_label="improving"
        elif trend<=-self.trend_threshold: trend_label="declining"
        else: trend_label="stable"

        # Trend contributes only a small bounded correction and only with enough observations.
        if recent_p+recent_n+previous_p+previous_n>=2:
            trend_adjustment=max(-0.05,min(0.05,trend*0.10))
        else:
            trend_adjustment=0.0
        multiplier=min(self.max_multiplier,max(self.min_multiplier,1.0+adjustment+trend_adjustment))
        return TemporalScore(target_ref,context_key,now.isoformat(),
            round(wp,6),round(wn,6),round(posterior,6),round(confidence,6),
            round(multiplier,6),recent_p,recent_n,previous_p,previous_n,
            round(trend,6),trend_label,round(min(ages),2) if ages else None,assessment)

    def combine_with_p63(self,base_multiplier: float,temporal_score: TemporalScore) -> float:
        """Combine P6.3's stable policy with temporal adjustment, enforcing global bounds."""
        if not isinstance(base_multiplier,(int,float)) or not math.isfinite(base_multiplier) or base_multiplier<=0:
            raise TemporalAdaptationError("base_multiplier must be finite and positive")
        combined=float(base_multiplier)*temporal_score.temporal_multiplier
        return round(min(self.max_multiplier,max(self.min_multiplier,combined)),6)
