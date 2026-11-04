"""Harder recall test on real labels: all ten conversations in one memory, so the shortlist must pick 4 episodes out of 272."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, pickle, types, time
from dataclasses import replace
import numpy as np
from bench import Conv, Flat, run_eng, DATA
from engram import Config
import final_lib as L

convs = pickle.load(open("convs.pkl", "rb"))
M = types.SimpleNamespace(turns=[], bodies=[], qs=[], vecs=None, qvecs=None)
vs, qvs = [], []
for ci, c in enumerate(convs):
    for t in c.turns: M.turns.append(dict(t, id=f"c{ci}:{t['id']}", session=ci * 1000 + t["session"]))
    M.bodies += c.bodies; vs.append(c.vecs); qvs.append(c.qvecs)
    for q in c.qs: M.qs.append(dict(q, ev={f"c{ci}:{e}" for e in q["ev"]}))
M.vecs = np.concatenate(vs); M.qvecs = np.concatenate(qvs)
M.ids = {t["id"] for t in M.turns}
cfg = Config(**json.load(open("results_v2.json"))["config"])
print("turns", len(M.turns), "episodes", len({t["session"] for t in M.turns}), "questions", len(M.qs))
R = {}
for ep in (4, 8, 16, 32):
    x = run_eng([M], replace(cfg, episodes=ep))
    R[f"engram_ep{ep}"] = {k: x[k] for k in ("recall", "tokens", "p50", "p95", "touched")}; print(ep, R[f"engram_ep{ep}"], flush=True)
x = run_eng([M], replace(cfg, shortlist=False)); R["engram_noshort"] = {k: x[k] for k in ("recall", "tokens", "p50", "touched")}; print(R["engram_noshort"], flush=True)
for k in (5, 6, 7):
    rows, ms = L.flat_rows([M], k, True)
    R[f"bm25next{k}"] = dict(recall=float(np.mean([r[0] for r in rows])), tokens=float(np.mean([r[1] for r in rows])), p50=float(np.percentile(ms, 50))); print(k, R[f"bm25next{k}"], flush=True)
tok0 = R["engram_ep4"]["tokens"]; best = min((5, 6, 7), key=lambda k: abs(R[f"bm25next{k}"]["tokens"] - tok0))
e = L.per_q([M], cfg); b, _ = L.flat_rows([M], best, True)
R["paired_ep4"] = dict(k=best, diff=L.boot([r[0] for r in e], [r[0] for r in b])); print(R["paired_ep4"])
json.dump(R, open("merged.json", "w"), indent=1, default=float)
n0 = sum(len(c.qs) for c in convs[:5])
er, br = [r[0] for r in e][n0:], [r[0] for r in b][n0:]
b7, _ = L.flat_rows([M], 7, True)
print("test half", len(er), np.mean(er), np.mean([r[1] for r in e][n0:]), np.mean(br), np.mean([r[1] for r in b][n0:]), L.boot(er, br))
print("vs k7 test half", np.mean([r[0] for r in b7][n0:]), np.mean([r[1] for r in b7][n0:]), L.boot(er, [r[0] for r in b7][n0:]))
print("vs k7 all", L.boot([r[0] for r in e], [r[0] for r in b7]))
