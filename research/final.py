"""Final evaluation on the held-out test conversations. Settings come only from the development grid."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, pickle, time, re
from dataclasses import replace
from collections import defaultdict
import numpy as np
import bench
from bench import Conv, Flat, build, run_flat, run_eng, score, tok, CATS, load_encoder, DATA, norm
from engram import Config, Engrams, words

enc = load_encoder()
convs = pickle.load(open("convs.pkl", "rb"))
dev, test = convs[:5], convs[5:]
obs_all = [s["observation"] for s in DATA]
grid = json.load(open("grid_dev.json"))
R = {}


def pick(limit):
    rows = [g for g in grid if g[1] <= limit]
    g = max(rows, key=lambda x: x[0])
    return Config(episodes=g[2], k_min=g[3], k_max=g[4], spread_each=g[5], margin=g[6], w_sem=g[7]), g


def per_q_eng(convs, cfg, notes=None):
    rows = []
    for ci, c in enumerate(convs):
        m = build_with_notes(c, cfg, notes[ci] if notes else None)
        for q, qv in zip(c.qs, c.qvecs):
            ns = m.retrieve(q["q"], qv, use_habit=False)
            got = {m.engrams[n].id for n in ns}
            rows.append((len(got & q["ev"]) / len(q["ev"]), tok(m.reinstate(ns)) if ns else 0, q["cat"]))
    return rows


def build_with_notes(c, cfg, obs=None):
    m = Engrams(cfg=cfg); eps = {}
    for t, v in zip(c.turns, c.vecs):
        if t["session"] not in eps: eps[t["session"]] = m.add_episode(t["date"])
        m.add_turn(t["id"], eps[t["session"]], t["speaker"], t["text"], t["cap"], v)
    if obs:
        n_notes = 0
        for sess in obs.values():
            for speaker_facts in sess.values():
                for fact in speaker_facts:
                    text, ids = fact[0], fact[1]
                    for i in re.split(r"[;,\s]+", ids if isinstance(ids, str) else " ".join(ids)):
                        i = re.sub(r"^D:?(\d+):0*(\d+)$", r"D\1:\2", i)
                        if i in m.by_id:
                            add_cues(m, m.by_id[i], text); n_notes += 1
        m.stats["notes"] = n_notes
    m.sleep()
    return m


def add_cues(m, n, text):
    """Sleep-time cue: a note written offline adds index terms to the pointer. The note text is not delivered."""
    cnt = defaultdict(int)
    for t in words(text): cnt[t] += 1
    have = {}
    for t, c in cnt.items():
        p = m.postings[t]
        if p and p[-1][0] == n: p[-1] = (n, p[-1][1] + c)
        else:
            idx = next((j for j, x in enumerate(p) if x[0] == n), None)
            if idx is None: p.append((n, c))
            else: p[idx] = (n, p[idx][1] + c)


def flat_rows(convs, k, neighbor=False, kind="bm25"):
    rows = []; ms = []
    for c in convs:
        f = Flat(c)
        order = defaultdict(list)
        for i, t in enumerate(c.turns): order[t["session"]].append(i)
        for q, qv in zip(c.qs, c.qvecs):
            t0 = time.perf_counter()
            r = f.bm25(q["q"]) if kind == "bm25" else f.dense(qv) if kind == "dense" else f.hybrid(q["q"], qv)
            top = r[:k].tolist()
            if neighbor:
                out, seen = [], set()
                for i in top:
                    if i not in seen: out.append(i); seen.add(i)
                    if i + 1 < len(c.turns) and c.turns[i + 1]["session"] == c.turns[i]["session"] and i + 1 not in seen:
                        out.append(i + 1); seen.add(i + 1)
                top = out
            ms.append((time.perf_counter() - t0) * 1e3)
            got = {c.turns[i]["id"] for i in top}
            rows.append((len(got & q["ev"]) / len(q["ev"]), sum(tok(c.bodies[i]) for i in top), q["cat"]))
    return rows, ms


def boot(a, b, n=4000, seed=1):
    rng = np.random.default_rng(seed); d = np.array(a) - np.array(b)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------- operating points chosen on dev only ----------
cfg_lean, g1 = pick(300)
cfg_std, g2 = pick(500)
cfg_wide, g3 = pick(900)
R["chosen"] = dict(lean=g1, std=g2, wide=g3)
print("chosen", g1, g2, g3)

curve = {}
for kind in ("bm25", "dense", "hybrid"):
    for k in (2, 3, 5, 8, 10, 15, 20, 30):
        r = run_flat(test, kind, k)
        curve[f"{kind}@{k}"] = {x: r[x] for x in ("recall", "complete", "tokens", "turns", "p50", "p95")}
for k in (1, 2, 3, 5, 8, 10, 15):
    rows, ms = flat_rows(test, k, neighbor=True)
    curve[f"bm25+next@{k}"] = dict(recall=float(np.mean([r[0] for r in rows])), complete=float(np.mean([r[0] == 1 for r in rows])), tokens=float(np.mean([r[1] for r in rows])), p50=float(np.percentile(ms, 50)), p95=float(np.percentile(ms, 95)))
for name, cfg in (("lean", cfg_lean), ("standard", cfg_std), ("wide", cfg_wide)):
    curve[f"engram-{name}"] = run_eng(test, cfg)
    print(name, {k: curve[f"engram-{name}"][k] for k in ("recall", "complete", "tokens", "p50", "p95", "touched", "escalated")})
R["curve"] = curve

# ---------- which layers earn their place ----------
abl = {}
base = cfg_std
variants = {"all layers on": base, "no thalamic gate": replace(base, gate=False), "no router": replace(base, router=False),
            "no episode shortlist": replace(base, shortlist=False), "no amygdala salience": replace(base, salience=False),
            "no spreading": replace(base, spread=False), "no cingulate early stop": replace(base, cingulate=False),
            "no sparse semantic cues": replace(base, semantic=False), "no entity boost": replace(base, w_entity=0.0)}
for name, cfg in variants.items():
    r = run_eng(test, cfg)
    abl[name] = {k: r[k] for k in ("recall", "complete", "tokens", "p50", "p95", "touched", "escalated")}
    print(name, round(r["recall"], 4), round(r["tokens"]), round(r["p50"], 3), round(r["touched"]))
R["ablation"] = abl

# ---------- paired comparisons at matched tokens ----------
eng_rows = per_q_eng(test, cfg_std)
eng_rec = [r[0] for r in eng_rows]; eng_tok = np.mean([r[1] for r in eng_rows])
cmp = {}
for label, nb in (("bm25", False), ("bm25+next", True)):
    best_k = min(range(1, 40), key=lambda k: abs(np.mean([r[1] for r in flat_rows(test, k, nb)[0]]) - eng_tok))
    rows, _ = flat_rows(test, best_k, nb)
    cmp[label] = dict(k=best_k, tokens=float(np.mean([r[1] for r in rows])), recall=float(np.mean([r[0] for r in rows])), diff=boot(eng_rec, [r[0] for r in rows]))
R["paired"] = dict(engram_tokens=float(eng_tok), engram_recall=float(np.mean(eng_rec)), cmp=cmp)
print(R["paired"])

# ---------- by question type ----------
bycat = defaultdict(dict)
k_std = cmp["bm25"]["k"]
b_rows, _ = flat_rows(test, k_std)
for cat in CATS:
    bycat[CATS[cat]]["engram"] = float(np.mean([r[0] for r in eng_rows if r[2] == cat]))
    bycat[CATS[cat]]["bm25"] = float(np.mean([r[0] for r in b_rows if r[2] == cat]))
    bycat[CATS[cat]]["n"] = sum(r[2] == cat for r in eng_rows)
R["bycat"] = bycat
print(dict(bycat))

# ---------- sleep with offline LLM notes (the dataset's own observations) ----------
notes_rows = per_q_eng(test, cfg_std, notes=obs_all[5:])
R["sleep_notes"] = dict(recall=float(np.mean([r[0] for r in notes_rows])), tokens=float(np.mean([r[1] for r in notes_rows])), diff=boot([r[0] for r in notes_rows], eng_rec))
print("notes", R["sleep_notes"])

# ---------- costs ----------
c0 = test[0]
t0 = time.perf_counter(); m = build(c0, cfg_std); t_build = time.perf_counter() - t0
t0 = time.perf_counter(); m.sleep(); t_sleep = time.perf_counter() - t0
sz = m.sizes()
dense_bytes = c0.vecs.nbytes
R["costs"] = dict(turns=len(c0.turns), write_ms_per_turn=t_build / len(c0.turns) * 1e3, sleep_ms=t_sleep * 1e3, sizes=sz, dense_vectors_bytes=dense_bytes, gated=m.stats["gated"])
qs = [q["q"] for q in c0.qs[:100]]
t0 = time.perf_counter()
for q in qs: enc.embed(q)
R["costs"]["query_encode_ms"] = (time.perf_counter() - t0) / len(qs) * 1e3
# stage 0
for q, qv in zip(c0.qs[:100], c0.qvecs[:100]): m.learn_habit(q["q"], m.retrieve(q["q"], qv, use_habit=False))
ts = []
for q, qv in zip(c0.qs[:100], c0.qvecs[:100]):
    t0 = time.perf_counter(); m.retrieve(q["q"], qv); ts.append((time.perf_counter() - t0) * 1e3)
R["costs"]["habit_ms_p50"] = float(np.percentile(ts, 50))
print(R["costs"])
R["all_tokens_per_conv"] = float(np.mean([sum(tok(b) for b in c.bodies) for c in test]))
json.dump(R, open("results.json", "w"), indent=1, default=float)
print("done")
