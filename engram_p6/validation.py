"""P6.5 offline validation for baseline and adaptive Engram retrieval.

Consumes held-out query traces and independently judged relevance labels.
It never reads or writes the live Engram database.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import json, math, random

SYSTEMS=("baseline","p63","p63_p64")
METRICS=("precision","recall","mrr","ndcg","rare_recall","correction_rate")

class ValidationError(ValueError):
    pass

@dataclass(frozen=True)
class Comparison:
    metric: str
    system: str
    baseline: str
    mean_difference: float
    ci_low: float
    ci_high: float
    permutation_p: float
    n_queries: int

@dataclass(frozen=True)
class ValidationReport:
    query_count: int
    k: int
    metrics: dict
    comparisons: list[dict]
    warnings: list[str]
    protocol: dict

    def to_dict(self):
        return asdict(self)

def _unique_strings(value, field, qid):
    if not isinstance(value,list) or any(not isinstance(x,str) or not x for x in value):
        raise ValidationError(f"{qid}: {field} must be a list of non-empty strings")
    if len(value)!=len(set(value)):
        raise ValidationError(f"{qid}: {field} contains duplicate refs")
    return value

def validate_records(records: list[dict]) -> list[dict]:
    if not records: raise ValidationError("validation dataset is empty")
    seen=set()
    cleaned=[]
    for r in records:
        qid=r.get("query_id")
        if not isinstance(qid,str) or not qid: raise ValidationError("each record needs query_id")
        if qid in seen: raise ValidationError(f"duplicate query_id: {qid}")
        seen.add(qid)
        query_time=r.get("query_time")
        if not isinstance(query_time,str) or not query_time:
            raise ValidationError(f"{qid}: query_time is required")
        relevant=_unique_strings(r.get("relevant_refs"),"relevant_refs",qid)
        rare=_unique_strings(r.get("rare_important_refs",[]),"rare_important_refs",qid)
        corrected=_unique_strings(r.get("corrected_refs",[]),"corrected_refs",qid)
        if not set(rare).issubset(relevant):
            raise ValidationError(f"{qid}: rare_important_refs must be a subset of relevant_refs")
        rankings=r.get("rankings")
        if not isinstance(rankings,dict): raise ValidationError(f"{qid}: rankings is required")
        for system in SYSTEMS:
            _unique_strings(rankings.get(system),f"rankings.{system}",qid)
        # Adaptive policy provenance must predate the query. This is a leakage guard,
        # not proof that the upstream policy builder used only permitted events.
        policy_meta=r.get("policy_provenance",{})
        for system in ("p63","p63_p64"):
            meta=policy_meta.get(system)
            if not isinstance(meta,dict):
                raise ValidationError(f"{qid}: policy_provenance.{system} is required")
            for field in ("policy_built_at","source_max_event_at"):
                value=meta.get(field)
                if not isinstance(value,str) or not value:
                    raise ValidationError(f"{qid}: {system}.{field} is required")
                if value > query_time:
                    raise ValidationError(f"{qid}: temporal leakage: {system}.{field} is after query_time")
        cleaned.append({
            "query_id":qid,"query_time":query_time,
            "relevant_refs":relevant,"rare_important_refs":rare,
            "corrected_refs":corrected,"rankings":rankings,
            "policy_provenance":policy_meta
        })
    return cleaned

def _dcg(ranked, relevant, k):
    rel=set(relevant)
    return sum((1.0 if ref in rel else 0.0)/math.log2(i+2)
               for i,ref in enumerate(ranked[:k]))

def query_metrics(record: dict, system: str, k: int) -> dict:
    ranked=record["rankings"][system][:k]
    relevant=set(record["relevant_refs"])
    rare=set(record["rare_important_refs"])
    corrected=set(record["corrected_refs"])
    hits=sum(ref in relevant for ref in ranked)
    precision=hits/k if k else 0.0
    recall=hits/len(relevant) if relevant else float("nan")
    rr=0.0
    for i,ref in enumerate(ranked):
        if ref in relevant:
            rr=1.0/(i+1); break
    ideal=min(len(relevant),k)
    idcg=sum(1.0/math.log2(i+2) for i in range(ideal))
    ndcg=_dcg(ranked,relevant,k)/idcg if idcg else float("nan")
    rare_recall=(sum(ref in rare for ref in ranked)/len(rare)) if rare else float("nan")
    correction_rate=(sum(ref in corrected for ref in ranked)/len(ranked)) if ranked else 0.0
    return {"precision":precision,"recall":recall,"mrr":rr,"ndcg":ndcg,
            "rare_recall":rare_recall,"correction_rate":correction_rate}

def _mean(values):
    values=[x for x in values if not math.isnan(x)]
    return sum(values)/len(values) if values else float("nan")

def _bootstrap_ci(diffs, samples, seed):
    rng=random.Random(seed)
    n=len(diffs)
    if not n: return (float("nan"),float("nan"))
    means=[]
    for _ in range(samples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n))/n)
    means.sort()
    return means[int(.025*(samples-1))],means[int(.975*(samples-1))]

def _sign_flip_p(diffs, samples, seed):
    diffs=[d for d in diffs if not math.isnan(d)]
    n=len(diffs)
    if not n: return float("nan")
    observed=abs(sum(diffs)/n)
    rng=random.Random(seed+17)
    if n<=18:
        signs=[]
        for mask in range(1<<n):
            val=sum((d if (mask>>i)&1 else -d) for i,d in enumerate(diffs))/n
            signs.append(abs(val))
        return sum(v>=observed-1e-12 for v in signs)/len(signs)
    extreme=0
    for _ in range(samples):
        val=sum(d if rng.random()<.5 else -d for d in diffs)/n
        if abs(val)>=observed-1e-12: extreme+=1
    return (extreme+1)/(samples+1)

def evaluate(records: list[dict], *, k: int=5, bootstrap_samples: int=2000,
             seed: int=2026) -> ValidationReport:
    if k<1: raise ValidationError("k must be >= 1")
    if bootstrap_samples<100: raise ValidationError("bootstrap_samples must be >= 100")
    rows=validate_records(records)
    per_system={}
    for system in SYSTEMS:
        metric_rows=[query_metrics(r,system,k) for r in rows]
        per_system[system]={m:_mean([x[m] for x in metric_rows]) for m in METRICS}
    comparisons=[]
    for system in ("p63","p63_p64"):
        for metric in METRICS:
            diffs=[]
            for r in rows:
                a=query_metrics(r,system,k)[metric]
                b=query_metrics(r,"baseline",k)[metric]
                if not math.isnan(a) and not math.isnan(b): diffs.append(a-b)
            lo,hi=_bootstrap_ci(diffs,bootstrap_samples,seed+len(comparisons))
            comparisons.append(asdict(Comparison(
                metric=metric,system=system,baseline="baseline",
                mean_difference=_mean(diffs),ci_low=lo,ci_high=hi,
                permutation_p=_sign_flip_p(diffs,bootstrap_samples,seed+len(comparisons)),
                n_queries=len(diffs))))
    warnings=[]
    if not any(r["rare_important_refs"] for r in rows):
        warnings.append("No rare_important_refs supplied; rare-memory protection cannot be evaluated.")
    if not any(r["corrected_refs"] for r in rows):
        warnings.append("No corrected_refs supplied; correction_rate is zero but not meaningfully assessed.")
    if len(rows)<30:
        warnings.append("Small held-out query set; uncertainty estimates may be unstable.")
    if any(not r["relevant_refs"] for r in rows):
        warnings.append("Queries with no relevant labels are excluded from recall/MRR/nDCG averages.")
    protocol={
        "systems":list(SYSTEMS),"ranking_cutoff":k,
        "split":"caller-supplied held-out query set; this module does not create the split",
        "leakage_guard":"adaptive policy build and latest source event timestamps must not be after query time",
        "significance":"paired query-level sign-flip permutation test; bootstrap percentile 95% CI",
        "interpretation":"positive difference is better except correction_rate, where lower is better",
    }
    return ValidationReport(len(rows),k,per_system,comparisons,warnings,protocol)

def load_jsonl(path: str) -> list[dict]:
    records=[]
    with open(path,encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            try: records.append(json.loads(line))
            except json.JSONDecodeError as e: raise ValidationError(f"invalid JSON on line {line_no}: {e}") from e
    return records

def save_report(report: ValidationReport,path: str):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(report.to_dict(),f,indent=2,allow_nan=False)
