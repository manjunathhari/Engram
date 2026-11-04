"""Second design. Tuned on development conversations only, then run once on test.
Note: the test conversations had already been scored once with the first design."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import itertools, json, pickle, time
from dataclasses import replace, asdict
import numpy as np
from bench import run_eng, CATS, DATA, Conv, Flat
from engram import Config
import final_lib as L

convs = pickle.load(open("convs.pkl", "rb"))
dev, test = convs[:5], convs[5:]
grid = []
t0 = time.time()
base = Config(semantic=False, salience=False, gate=False, router=True, w_entity=0.0)
for ep, kmin, kmax, mg, rp, terms, wt, back, tb, wd in itertools.product((4, 8), (3, 5), (5, 8), (.25, .4), (0, 3), (4, 8), (.3, .6), (0, 1), (0, .3), (0, 6)):
    if rp == 0 and (terms, wt) != (4, .3): continue
    cfg = replace(base, episodes=ep, k_min=kmin, k_max=kmax, margin=mg, replay=rp, replay_terms=terms, replay_weight=wt, spread_back=back, time_boost=tb, widen_set=wd)
    r = run_eng(dev, cfg)
    grid.append((r["recall"], r["tokens"], asdict(cfg)))
json.dump(grid, open("grid_v2.json", "w"))
print("grid", len(grid), round(time.time() - t0), flush=True)
ok = [g for g in grid if g[1] <= 500]
best = max(ok, key=lambda g: g[0])
print("best dev", best[0], best[1], {k: best[2][k] for k in ("episodes", "k_min", "k_max", "margin", "replay", "replay_terms", "replay_weight", "spread_back", "time_boost", "widen_set")}, flush=True)
# best with each new layer off, on dev, to see what each adds
for name, f in (("no replay", lambda c: c["replay"] == 0), ("no previous-turn", lambda c: c["spread_back"] == 0), ("no time boost", lambda c: c["time_boost"] == 0), ("no widening", lambda c: c["widen_set"] == 0)):
    b = max((g for g in ok if f(g[2])), key=lambda g: g[0])
    print("dev", name, round(b[0], 4), round(b[1]))
cfg = Config(**best[2])
print("bm25+next dev at matched tokens", [(k, round(np.mean([r[0] for r in L.flat_rows(dev, k, True)[0]]), 4), round(np.mean([r[1] for r in L.flat_rows(dev, k, True)[0]]))) for k in (5, 6)], flush=True)

# one run on test
R = {}
r = run_eng(test, cfg)
R["v2"] = {k: r[k] for k in ("recall", "complete", "tokens", "p50", "p95", "touched", "escalated")}
rows = L.per_q(test, cfg)
rec = [x[0] for x in rows]; tok = float(np.mean([x[1] for x in rows]))
k = min(range(1, 30), key=lambda k: abs(np.mean([x[1] for x in L.flat_rows(test, k, True)[0]]) - tok))
b, ms = L.flat_rows(test, k, True)
R["vs_bm25_next"] = dict(k=k, tokens=float(np.mean([x[1] for x in b])), recall=float(np.mean([x[0] for x in b])), p50=float(np.percentile(ms, 50)), diff=L.boot(rec, [x[0] for x in b]))
R["bycat"] = {CATS[c]: dict(engram=float(np.mean([x[0] for x in rows if x[2] == c])), baseline=float(np.mean([x[0] for x in b if x[2] == c])), n=sum(x[2] == c for x in rows)) for c in CATS}
nrows = L.per_q(test, cfg, notes=[s["observation"] for s in DATA[5:]])
R["with_notes"] = dict(recall=float(np.mean([x[0] for x in nrows])), tokens=float(np.mean([x[1] for x in nrows])), diff_vs_no_notes=L.boot([x[0] for x in nrows], rec), diff_vs_baseline=L.boot([x[0] for x in nrows], [x[0] for x in b]))
# notes on the plain baseline: same offline cues, one flat index
bn = L.flat_notes(test, k, [s["observation"] for s in DATA[5:]])
R["baseline_with_notes"] = dict(recall=float(np.mean([x[0] for x in bn])), tokens=float(np.mean([x[1] for x in bn])), diff_engram_notes_vs=L.boot([x[0] for x in nrows], [x[0] for x in bn]))
# ablation on test, chosen setting
abl = {}
variants = {"all on": cfg, "no replay": replace(cfg, replay=0), "no previous turn": replace(cfg, spread_back=0), "no next turn": replace(cfg, spread_each=0), "no time boost": replace(cfg, time_boost=0), "no widening": replace(cfg, widen_set=0),
            "no router": replace(cfg, router=False), "no shortlist": replace(cfg, shortlist=False), "no early stop": replace(cfg, cingulate=False),
            "add amygdala salience": replace(cfg, salience=True), "add sparse semantic": replace(cfg, semantic=True), "add thalamic gate": replace(cfg, gate=True), "add entity boost": replace(cfg, w_entity=0.5)}
for n, c in variants.items():
    x = run_eng(test, c); abl[n] = {q: x[q] for q in ("recall", "tokens", "p50", "touched")}; print(n, round(x["recall"], 4), round(x["tokens"]), round(x["p50"], 3), flush=True)
R["ablation"] = abl
R["config"] = best[2]
json.dump(R, open("results_v2.json", "w"), indent=1, default=float)
print({k: R[k] for k in ("v2", "vs_bm25_next", "with_notes", "baseline_with_notes")}); print(R["bycat"])
