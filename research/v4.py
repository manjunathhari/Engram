"""Fourth design: nothing is deleted. An awake index answers first; a dormant index wakes on a weak match.
Dials are tuned on development conversations. Test conversations have been seen in earlier rounds."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, pickle, random, time, itertools, types, copy
from dataclasses import replace, asdict
from collections import defaultdict
import numpy as np
import engram
from bench import Conv, Flat, run_eng, CATS, DATA, tok
from engram import Config, Engrams, words
import final_lib as L
convs = pickle.load(open("convs.pkl", "rb")); dev, test = convs[:5], convs[5:]
engram.CRITIC_WEIGHTS = list(json.load(open("results_v3.json"))["critic"]["weights"].values())
base = Config(**json.load(open("results_v3_final.json"))["config"])
R = {}

# ---- flat routes on the single index
def ev(cs, cfg, cat=None):
    rows = L.per_q(cs, cfg); return float(np.mean([r[0] for r in rows])), float(np.mean([r[1] for r in rows])), rows
g = []
for df, cs in itertools.product((0, 1, 2, 3, 5), (0, 1)):
    c = replace(base, direct_df=df, cue_slots=cs); a, b, _ = ev(dev, c); g.append((a, b, df, cs)); print("dev", df, cs, round(a, 4), round(b), flush=True)
best = max((x for x in g if x[1] <= 500), key=lambda x: x[0])
cfg = replace(base, direct_df=best[2], cue_slots=best[3]); R["routes_dev"] = g; R["routes_pick"] = best
print("pick", best, flush=True)
a, b, rows = ev(test, cfg); a0, b0, r0 = ev(test, base)
R["routes_test"] = dict(recall=a, tokens=b, base_recall=a0, base_tokens=b0, diff=L.boot([r[0] for r in rows], [r[0] for r in r0]),
                        multihop=float(np.mean([r[0] for r in rows if r[2] == 1])), multihop_base=float(np.mean([r[0] for r in r0 if r[2] == 1])))
print(R["routes_test"], flush=True)
# every setting on test, for the record, using cue slots forced on: how much do reserved cue places give multi-hop?
c1 = replace(base, cue_slots=1); a1, b1, r1 = ev(test, c1)
R["cue_slots_test"] = dict(recall=a1, tokens=b1, multihop=float(np.mean([r[0] for r in r1 if r[2] == 1])), diff=L.boot([r[0] for r in r1], [r[0] for r in r0]))
print(R["cue_slots_test"], flush=True)

# ---- awake and dormant tiers
def coverage(m, q, ns):
    qt = [t for t in dict.fromkeys(words(q)) if t in m.idf]
    if not qt: return 0.0
    have = set()
    for n in ns[:3]:
        e = m.engrams[n]; have |= set(words(m.cortex[e.store][e.id]))
    tot = sum(m.idf[t] for t in qt)
    return sum(m.idf[t] for t in qt if t in have) / tot

def tiered(cs, cfg, frac, thr, how="critic", refresh=False, order_seed=None, split=False):
    rec, toks, ms, touched, woke = [], [], [], [], []
    for c in cs:
        full = L.build_with_notes(c, cfg, None); hot = L.build_with_notes(c, cfg, None)
        idx = list(range(len(hot.engrams)))
        hot.forget(set(sorted(idx, key=lambda n: hot.engrams[n].importance, reverse=True)[: int(frac * len(idx))]))
        for q, qv in zip(c.qs, c.qvecs):
            tr = {}; t0 = time.perf_counter()
            ns = hot.retrieve(q["q"], qv, use_habit=False, trace=tr); t = tr.get("touched", 0); w = False
            if coverage(hot, q["q"], ns) < thr:
                tr2 = {}; ns = full.retrieve(q["q"], qv, use_habit=False, trace=tr2); t += tr2.get("touched", 0); w = True
            ms.append((time.perf_counter() - t0) * 1e3)
            got = {full.engrams[n].id for n in ns}
            rec.append(len(got & q["ev"]) / len(q["ev"])); toks.append(tok(full.reinstate(ns)) if ns else 0); touched.append(t); woke.append(w)
    return dict(recall=float(np.mean(rec)), tokens=float(np.mean(toks)), p50=float(np.percentile(ms, 50)), p95=float(np.percentile(ms, 95)), touched=float(np.mean(touched)), woke=float(np.mean(woke))), rec

T = {}
for frac in (.5, .25):
    gg = []
    for thr in (.3, .5, .7, .85, 1.01):
        x, _ = tiered(dev, cfg, frac, thr); gg.append((x["recall"], x["woke"], thr)); print("dev tier", frac, thr, round(x["recall"], 4), round(x["woke"], 3), flush=True)
    # lowest threshold that stays within one point of the best dev recall
    top = max(a for a, _, _ in gg); thr = min(t for a, _, t in gg if a >= top - .01)
    x, rx = tiered(test, cfg, frac, thr); x["thr"] = thr; x["vs_full"] = L.boot(rx, [r[0] for r in rows])
    T[str(frac)] = x; print("test tier", frac, x, flush=True)
fullx = run_eng(test, cfg); T["full_index"] = {k: fullx[k] for k in ("recall", "tokens", "p50", "p95", "touched")}
print(T["full_index"], flush=True)
R["tiers"] = T

# ---- refresh on recall: does waking a memory make later questions cheaper?
def refresh_test(cs, cfg, frac, thr, seed=5):
    out = defaultdict(list)
    for c in cs:
        for mode in ("no refresh", "refresh"):
            full = L.build_with_notes(c, cfg, None); hot = L.build_with_notes(c, cfg, None)
            idx = list(range(len(hot.engrams)))
            hot.forget(set(sorted(idx, key=lambda n: hot.engrams[n].importance, reverse=True)[: int(frac * len(idx))]))
            pairs = list(zip(c.qs, c.qvecs)); random.Random(seed).shuffle(pairs)
            first, second = pairs[: len(pairs) // 2], pairs[len(pairs) // 2:]
            for q, qv in first + first:  # first half asked, then the same questions asked again
                ns = hot.retrieve(q["q"], qv, use_habit=False)
                if coverage(hot, q["q"], ns) < thr:
                    ns = full.retrieve(q["q"], qv, use_habit=False)
                    if mode == "refresh":
                        for n in ns[:3]: hot.promote(full, n)
            for tag, group in (("repeat", first), ("new", second)):
                for q, qv in group:
                    ns = hot.retrieve(q["q"], qv, use_habit=False)
                    wake = coverage(hot, q["q"], ns) < thr
                    got = {hot.engrams[n].id for n in ns}
                    out[(mode, tag, "awake_recall")].append(len(got & q["ev"]) / len(q["ev"])); out[(mode, tag, "woke")].append(wake)
    return {f"{a}|{b}|{k}": float(np.mean(v)) for (a, b, k), v in out.items()}
R["refresh"] = refresh_test(test, cfg, .25, T[".25" if False else "0.25"]["thr"]); print(R["refresh"], flush=True)

# ---- compartments: find the right episode, then open it
comp = defaultdict(list)
for c in test:
    m = L.build_with_notes(c, cfg, None)
    ep_of = {m.engrams[n].id: m.engrams[n].episode for n in range(len(m.engrams))}
    ep_tok = {i: sum(tok(m.reinstate([n])) for n in mem) for i, mem in enumerate(m.ep_members)}
    for q, qv in zip(c.qs, c.qvecs):
        ns = m.retrieve(q["q"], qv, use_habit=False)
        order = list(dict.fromkeys(m.engrams[n].episode for n in ns))
        want = {ep_of[e] for e in q["ev"]}
        for k in (1, 2, 4):
            pick = order[:k]; got = {e for e in q["ev"] if ep_of[e] in pick}
            comp[f"open{k}_recall"].append(len(got) / len(q["ev"])); comp[f"open{k}_tokens"].append(sum(ep_tok[i] for i in pick))
        comp["episodes_needed"].append(len(want))
R["compartments"] = {k: float(np.mean(v)) for k, v in comp.items()}; print(R["compartments"], flush=True)

# ---- visual cues: can an image's search term bring back the turn and what was said?
vis = {}
for name, use_cap in (("captions indexed", True), ("captions not indexed", False)):
    hits, hits_ctx, n = 0, 0, 0
    for ci, c in enumerate(test):
        s = DATA[5 + ci]["conversation"]; qmap = {t["dia_id"]: t.get("query", "") for k, v in s.items() if isinstance(v, list) for t in v if t.get("img_url")}
        m = Engrams(cfg=replace(cfg, semantic=False, direct_df=0, cue_slots=0)); eps = {}
        for t in c.turns:
            if t["session"] not in eps: eps[t["session"]] = m.add_episode(t["date"])
            m.add_turn(t["id"], eps[t["session"]], t["speaker"], t["text"], t["cap"] if use_cap else "")
        m.sleep()
        for i, cue in qmap.items():
            if not cue or i not in m.by_id: continue
            ns = m.retrieve(cue, None, use_habit=False); ids = [m.engrams[x].id for x in ns]; n += 1
            hits += i in ids
    vis[name] = dict(hit=hits / max(1, n), n=n)
R["visual"] = vis; print(vis, flush=True)
R["config"] = asdict(cfg)
json.dump(R, open("results_v4.json", "w"), indent=1, default=float); print("done")
