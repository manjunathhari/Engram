"""Final selection for the third design, on development conversations, after the first v3 run showed the world table and importance weight did not help ranking.
This is the fourth look at the test conversations, so test figures here are mildly optimistic."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, pickle, types
from dataclasses import replace, asdict
import numpy as np
import engram
from bench import run_eng, CATS, DATA, Conv, Flat
from engram import Config
import final_lib as L
convs = pickle.load(open("convs.pkl", "rb")); dev, test = convs[:5], convs[5:]
R0 = json.load(open("results_v3.json")); engram.CRITIC_WEIGHTS = list(R0["critic"]["weights"].values())
base = Config(**{**R0["config"], "world_gate": True})
opts = {}
for imp in (0, .5):
    for world, gate in ((False, True), (True, True), (True, False)):
        cfg = replace(base, w_imp=imp, world=world, world_gate=gate); r = run_eng(dev, cfg)
        opts[(imp, world, gate)] = (r["recall"], r["tokens"], cfg); print(imp, world, gate, round(r["recall"], 4), round(r["tokens"]), flush=True)
pick = max((v for v in opts.values() if v[1] <= 500), key=lambda v: v[0]); cfg = pick[2]
print("picked", cfg.w_imp, cfg.world, cfg.world_gate, pick[0], pick[1], flush=True)
R = dict(config=asdict(cfg), dev=dict(recall=pick[0], tokens=pick[1]), options={str(k): (v[0], v[1]) for k, v in opts.items()})
v2 = Config(**json.load(open("results_v2.json"))["config"])
x = run_eng(test, cfg); R["test"] = {k: x[k] for k in ("recall", "complete", "tokens", "p50", "p95", "touched", "escalated")}
rows = L.per_q(test, cfg); v2r = L.per_q(test, v2); rec = [r[0] for r in rows]; tok = float(np.mean([r[1] for r in rows]))
k = min(range(1, 30), key=lambda k: abs(np.mean([r[1] for r in L.flat_rows(test, k, True)[0]]) - tok)); b, _ = L.flat_rows(test, k, True)
R["vs_baseline"] = dict(k=k, tokens=float(np.mean([r[1] for r in b])), recall=float(np.mean([r[0] for r in b])), diff=L.boot(rec, [r[0] for r in b]))
R["vs_v2"] = dict(recall=float(np.mean([r[0] for r in v2r])), tokens=float(np.mean([r[1] for r in v2r])), diff=L.boot(rec, [r[0] for r in v2r]))
tmp = [i for i, r in enumerate(rows) if r[2] == 2]
R["temporal"] = dict(n=len(tmp), v3=float(np.mean([rec[i] for i in tmp])), v2=float(np.mean([v2r[i][0] for i in tmp])), baseline=float(np.mean([b[i][0] for i in tmp])),
                     vs_v2=L.boot([rec[i] for i in tmp], [v2r[i][0] for i in tmp]), vs_baseline=L.boot([rec[i] for i in tmp], [b[i][0] for i in tmp]))
R["bycat"] = {CATS[c]: dict(v3=float(np.mean([r[0] for r in rows if r[2] == c])), baseline=float(np.mean([r[0] for r in b if r[2] == c])), n=sum(r[2] == c for r in rows)) for c in CATS}
n = L.per_q(test, cfg, notes=[s["observation"] for s in DATA[5:]])
R["notes"] = dict(recall=float(np.mean([r[0] for r in n])), tokens=float(np.mean([r[1] for r in n])), diff=L.boot([r[0] for r in n], rec))
M = types.SimpleNamespace(turns=[], bodies=[], qs=[]); vs, qvs = [], []
for ci, c in enumerate(convs):
    for t in c.turns: M.turns.append(dict(t, id=f"c{ci}:{t['id']}", session=ci * 1000 + t["session"]))
    M.bodies += c.bodies; vs.append(c.vecs); qvs.append(c.qvecs)
    for q in c.qs: M.qs.append(dict(q, ev={f"c{ci}:{e}" for e in q["ev"]}))
M.vecs = np.concatenate(vs); M.qvecs = np.concatenate(qvs); M.ids = {t["id"] for t in M.turns}
n0 = sum(len(c.qs) for c in dev)
e = L.per_q([M], cfg)[n0:]; tm = float(np.mean([r[1] for r in e]))
res = {}
for kk in (6, 7):
    bk = L.flat_rows([M], kk, True)[0][n0:]; res[kk] = dict(recall=float(np.mean([r[0] for r in bk])), tokens=float(np.mean([r[1] for r in bk])), diff=L.boot([r[0] for r in e], [r[0] for r in bk]))
R["merged"] = dict(recall=float(np.mean([r[0] for r in e])), tokens=tm, baseline=res)
json.dump(R, open("results_v3_final.json", "w"), indent=1, default=float)
for k in ("test", "vs_baseline", "vs_v2", "temporal", "notes", "merged", "bycat"): print(k, R[k], flush=True)
