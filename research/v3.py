"""Third design: time, learned importance, world-state updates. Tuned on development conversations only.
The test conversations had already been scored with the two earlier designs."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import itertools, json, pickle, time, types, random
from dataclasses import replace, asdict
import numpy as np
import engram
from bench import run_eng, CATS, DATA, Conv, Flat, build
from engram import Config, Engrams, importance_features
import final_lib as L

convs = pickle.load(open("convs.pkl", "rb"))
dev, test = convs[:5], convs[5:]
R = {}

def dataset(cs):
    X, y = [], []
    for c in cs:
        ev = set().union(*[q["ev"] for q in c.qs])
        ses = {}
        for t in c.turns: ses.setdefault(t["session"], []).append(t)
        for s in ses.values():
            for p, t in enumerate(s):
                X.append(importance_features(t["text"], t["cap"], s[p - 1]["text"] if p else "", p, len(s))); y.append(t["id"] in ev)
    return np.array(X, float), np.array(y, float)

def fit(X, y, l2=1e-3, it=3000, lr=0.3):
    w = np.zeros(X.shape[1]); mu = X.mean(0); 
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) / len(y) + l2 * w; w -= lr * g
    return w

def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

Xd, yd = dataset(dev); Xt, yt = dataset(test)
w = fit(Xd, yd)
engram.CRITIC_WEIGHTS = [float(x) for x in w]
names = "bias log_len ends_q has_q entities digit timeword emotion plan image first_person ack after_question position".split()
R["critic"] = dict(weights=dict(zip(names, map(float, w))), auc_test=auc(Xt @ w, yt), base_rate_test=float(yt.mean()))
print("critic auc test", R["critic"]["auc_test"], "base rate", R["critic"]["base_rate_test"], flush=True)

v2 = Config(**json.load(open("results_v2.json"))["config"])
grid = []
for imp, rng, ep, kmin, kmax, mg in itertools.product((0, .5, 1, 2, 4), (0, .5, 1), (4, 6), (3, 5), (5, 8), (.25, .4)):
    cfg = replace(v2, w_imp=imp, w_range=rng, episodes=ep, k_min=kmin, k_max=kmax, margin=mg, world=True)
    r = run_eng(dev, cfg); grid.append((r["recall"], r["tokens"], asdict(cfg)))
json.dump(grid, open("grid_v3.json", "w"))
best = max((g for g in grid if g[1] <= 500), key=lambda g: g[0])
cfg = Config(**best[2])
print("best dev", best[0], best[1], {k: best[2][k] for k in ("w_imp", "w_range", "episodes", "k_min", "k_max", "margin")}, flush=True)
R["config"] = best[2]
R["dev"] = dict(recall=best[0], tokens=best[1])

# held-out conversations
r = run_eng(test, cfg); R["v3"] = {k: r[k] for k in ("recall", "complete", "tokens", "p50", "p95", "touched", "escalated")}
rows = L.per_q(test, cfg); rec = [x[0] for x in rows]; tok = float(np.mean([x[1] for x in rows]))
v2rows = L.per_q(test, v2)
k = min(range(1, 30), key=lambda k: abs(np.mean([x[1] for x in L.flat_rows(test, k, True)[0]]) - tok))
b, ms = L.flat_rows(test, k, True)
R["vs_baseline"] = dict(k=k, tokens=float(np.mean([x[1] for x in b])), recall=float(np.mean([x[0] for x in b])), diff=L.boot(rec, [x[0] for x in b]))
R["vs_v2"] = dict(recall=float(np.mean([x[0] for x in v2rows])), tokens=float(np.mean([x[1] for x in v2rows])), diff=L.boot(rec, [x[0] for x in v2rows]))
R["bycat"] = {CATS[c]: dict(v3=float(np.mean([x[0] for x in rows if x[2] == c])), v2=float(np.mean([x[0] for x in v2rows if x[2] == c])), baseline=float(np.mean([x[0] for x in b if x[2] == c])), n=sum(x[2] == c for x in rows)) for c in CATS}
print(R["v3"], R["vs_baseline"], R["vs_v2"], R["bycat"], flush=True)
abl = {}
for n, c in {"all on": cfg, "no importance": replace(cfg, w_imp=0), "no time range": replace(cfg, w_range=0), "no world state": replace(cfg, world=False)}.items():
    x = run_eng(test, c); abl[n] = {q: x[q] for q in ("recall", "tokens", "p50")}; print(n, abl[n], flush=True)
R["ablation"] = abl
nrows = L.per_q(test, cfg, notes=[s["observation"] for s in DATA[5:]])
R["with_notes"] = dict(recall=float(np.mean([x[0] for x in nrows])), tokens=float(np.mean([x[1] for x in nrows])), diff_vs_no_notes=L.boot([x[0] for x in nrows], rec))
print(R["with_notes"], flush=True)

# ten conversations in one memory
M = types.SimpleNamespace(turns=[], bodies=[], qs=[], ids=set())
vs, qvs = [], []
for ci, c in enumerate(convs):
    for t in c.turns: M.turns.append(dict(t, id=f"c{ci}:{t['id']}", session=ci * 1000 + t["session"]))
    M.bodies += c.bodies; vs.append(c.vecs); qvs.append(c.qvecs)
    for q in c.qs: M.qs.append(dict(q, ev={f"c{ci}:{e}" for e in q["ev"]}))
M.vecs = np.concatenate(vs); M.qvecs = np.concatenate(qvs); M.ids = {t["id"] for t in M.turns}
n0 = sum(len(c.qs) for c in dev)
e = L.per_q([M], cfg)[n0:]; e2 = L.per_q([M], v2)[n0:]
tm = float(np.mean([x[1] for x in e]))
kk = min(range(3, 12), key=lambda k: abs(np.mean([x[1] for x in L.flat_rows([M], k, True)[0][n0:]]) - tm))
bk = L.flat_rows([M], kk, True)[0][n0:]
R["merged"] = dict(v3=dict(recall=float(np.mean([x[0] for x in e])), tokens=tm), v2=dict(recall=float(np.mean([x[0] for x in e2])), tokens=float(np.mean([x[1] for x in e2]))),
                   baseline=dict(k=kk, recall=float(np.mean([x[0] for x in bk])), tokens=float(np.mean([x[1] for x in bk]))), diff_vs_baseline=L.boot([x[0] for x in e], [x[0] for x in bk]), diff_vs_v2=L.boot([x[0] for x in e], [x[0] for x in e2]))
print(R["merged"], flush=True)

# forgetting: which memories should the index keep when it can only hold part of them?
def forget_test(frac, how, seed=1):
    out = []
    rng = random.Random(seed)
    for c in test:
        m = L.build_with_notes(c, cfg, None)
        idx = list(range(len(m.engrams)))
        key = {"critic": lambda n: m.engrams[n].importance, "emotion": lambda n: (m.engrams[n].intrinsic, rng.random()), "random": lambda n: rng.random(),
               "newest": lambda n: m.engrams[n].day, "longest": lambda n: m.doclen[n]}[how]
        keep = set(sorted(idx, key=key, reverse=True)[: int(frac * len(idx))])
        m.forget(keep)
        for q, qv in zip(c.qs, c.qvecs):
            ns = m.retrieve(q["q"], qv, use_habit=False)
            got = {m.engrams[n].id for n in ns}
            out.append(len(got & q["ev"]) / len(q["ev"]))
    return out
F = {}
for frac in (.5, .25):
    ref = forget_test(frac, "random")
    for how in ("critic", "emotion", "random", "newest", "longest"):
        x = forget_test(frac, how); F[f"{how}@{frac}"] = dict(recall=float(np.mean(x)), vs_random=L.boot(x, ref)); print(how, frac, F[f"{how}@{frac}"], flush=True)
R["forget"] = F
json.dump(R, open("results_v3.json", "w"), indent=1, default=float)
print("done")
