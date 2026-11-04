"""Two follow-ups. A better wake signal, and whether refresh-on-recall beats a fixed importance score at equal awake size."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, pickle, random, time
from collections import defaultdict
from dataclasses import replace
import numpy as np
import engram
from bench import Conv, Flat, tok
from engram import Config, words
import final_lib as L
convs = pickle.load(open("convs.pkl", "rb")); dev, test = convs[:5], convs[5:]
engram.CRITIC_WEIGHTS = list(json.load(open("results_v3.json"))["critic"]["weights"].values())
cfg = Config(**json.load(open("results_v4.json"))["config"])
R = {}
def pair(c, frac):
    full = L.build_with_notes(c, cfg, None); hot = L.build_with_notes(c, cfg, None)
    idx = list(range(len(hot.engrams)))
    hot.forget(set(sorted(idx, key=lambda n: hot.engrams[n].importance, reverse=True)[: int(frac * len(idx))]))
    return full, hot
def dormant_cue(hot, full, q, df_max):
    """Wake only if a cue in the question is rare and has pointers lying in the dormant index."""
    for t in dict.fromkeys(words(q)):
        p = full.postings.get(t)
        if p and len(p) <= df_max and len(hot.postings.get(t, ())) < len(p): return True
    return False
def tier(cs, frac, df_max):
    rec, ms, touched, woke, toks = [], [], [], [], []
    for c in cs:
        full, hot = pair(c, frac)
        for q, qv in zip(c.qs, c.qvecs):
            tr = {}; t0 = time.perf_counter()
            ns = hot.retrieve(q["q"], qv, use_habit=False, trace=tr); t = tr.get("touched", 0); w = dormant_cue(hot, full, q["q"], df_max)
            if w:
                tr2 = {}; ns = full.retrieve(q["q"], qv, use_habit=False, trace=tr2); t += tr2.get("touched", 0)
            ms.append((time.perf_counter() - t0) * 1e3)
            got = {full.engrams[n].id for n in ns}; rec.append(len(got & q["ev"]) / len(q["ev"])); touched.append(t); woke.append(w); toks.append(tok(full.reinstate(ns)) if ns else 0)
    return dict(recall=float(np.mean(rec)), tokens=float(np.mean(toks)), p50=float(np.percentile(ms, 50)), touched=float(np.mean(touched)), woke=float(np.mean(woke)))
for frac in (.5, .25):
    d = {df: tier(dev, frac, df) for df in (3, 10, 30, 100, 1000)}
    print("dev", frac, {k: (round(v["recall"], 3), round(v["woke"], 2)) for k, v in d.items()}, flush=True)
    target = max(v["recall"] for v in d.values()) - .01
    df = min(k for k, v in d.items() if v["recall"] >= target)
    R[f"tier_{frac}"] = dict(df=df, dev={str(k): v for k, v in d.items()}, test=tier(test, frac, df)); print("test", frac, df, R[f"tier_{frac}"]["test"], flush=True)

def refresh(cs, frac, seed):
    out = defaultdict(list); sizes = []
    for c in cs:
        pairs = list(zip(c.qs, c.qvecs)); random.Random(seed).shuffle(pairs); first, second = pairs[: len(pairs) // 2], pairs[len(pairs) // 2:]
        full, hot = pair(c, frac)
        for q, qv in first:
            for n in full.retrieve(q["q"], qv, use_habit=False)[:3]: hot.promote(full, n)
        awake = len(hot.engrams) - len(hot.forgotten); N = len(hot.engrams); sizes.append(awake / N)
        def score(m, name):
            for q, qv in second:
                got = {m.engrams[n].id for n in m.retrieve(q["q"], qv, use_habit=False)}; out[name].append(len(got & q["ev"]) / len(q["ev"]))
        score(hot, "refreshed")
        _, st = pair(c, frac); score(st, "starting index, no refresh")
        _, eq = pair(c, awake / N); score(eq, "importance only, same size")
        score(full, "whole index")
    return {k: float(np.mean(v)) for k, v in out.items()}, {k: v for k, v in out.items()}, float(np.mean(sizes))
for frac in (.25, .5):
    m, raw, size = refresh(test, frac, 5)
    R[f"refresh_{frac}"] = dict(means=m, awake_share=size, vs_importance=L.boot(raw["refreshed"], raw["importance only, same size"]), n=len(raw["refreshed"])); print(frac, R[f"refresh_{frac}"], flush=True)
json.dump(R, open("results_v4b.json", "w"), indent=1, default=float); print("done")
