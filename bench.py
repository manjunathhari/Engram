"""Benchmark Engram against flat search on LoCoMo. Development conversations 0-4, test conversations 5-9.
Metric: of the turns that hold the answer (the dataset's evidence labels), how many came back, and at what token cost."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, re, sys, time, itertools, math, pickle, logging
from collections import defaultdict
from dataclasses import replace
import numpy as np
from engram import Engrams, Config, words
logging.disable(logging.CRITICAL)

DATA = json.load(open("locomo/data/locomo10.json"))
CATS = {1: "multi-hop", 2: "temporal", 3: "inference", 4: "single-hop", 5: "adversarial"}
tok = lambda s: max(1, len(s) // 4)


def load_encoder():
    from wordllama import WordLlama
    return WordLlama.load()


def evidence(q, ids):
    out = set()
    for e in q["evidence"]:
        for p in re.split(r"[;\s]+", e):
            m = re.match(r"D:?(\d+):0*(\d+)$", p.strip())
            if m and f"D{m[1]}:{m[2]}" in ids: out.add(f"D{m[1]}:{m[2]}")
    return out


class Conv:
    def __init__(self, s, enc, prefix=""):
        c = s["conversation"]
        self.turns = []
        self.ep = {}
        ns = sorted((int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k)))
        for i in ns:
            for t in c[f"session_{i}"]:
                self.turns.append(dict(id=prefix + t["dia_id"], session=i, date=c[f"session_{i}_date_time"], speaker=t["speaker"],
                                       text=t["text"], cap=t.get("blip_caption", "")))
        self.ids = {t["id"] for t in self.turns}
        self.qs = []
        for q in s["qa"]:
            raw = {x[len(prefix):] if prefix else x for x in self.ids}
            ev = evidence(q, raw)
            if ev: self.qs.append(dict(q=q["question"], ev={prefix + e for e in ev}, cat=q["category"]))
        self.bodies = [f"[{t['date']}] {t['speaker']}: {t['text']}" + (f" [image: {t['cap']}]" if t["cap"] else "") for t in self.turns]
        self.vecs = norm(enc.embed([b for b in self.bodies]))
        self.qvecs = norm(enc.embed([q["q"] for q in self.qs]))


def norm(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9, None)


class Flat:
    """Single-layer retrieval over every turn: dense cosine, BM25, or both fused."""
    def __init__(self, conv):
        self.c = conv
        self.post = defaultdict(list)
        self.dl = []
        for i, b in enumerate(conv.bodies):
            ws = words(b); self.dl.append(len(ws))
            cnt = defaultdict(int)
            for w in ws: cnt[w] += 1
            for w, n in cnt.items(): self.post[w].append((i, n))
        self.N = len(self.dl); self.avg = sum(self.dl) / max(1, self.N)
        self.idf = {t: math.log(1 + (self.N - len(p) + .5) / (len(p) + .5)) for t, p in self.post.items()}

    def dense(self, qv): return np.argsort(-(self.c.vecs @ qv))

    def bm25(self, q):
        s = np.zeros(self.N, dtype=np.float32)
        k1, b = 1.2, .75
        for t in set(words(q)):
            if t in self.idf:
                for i, tf in self.post[t]: s[i] += self.idf[t] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.dl[i] / self.avg))
        return np.argsort(-s)

    def hybrid(self, q, qv):
        rr = np.zeros(self.N, dtype=np.float32)
        for r in (self.dense(qv), self.bm25(q)):
            rr[r] += 1 / (60 + np.arange(self.N))
        return np.argsort(-rr)


def build(conv, cfg, prefix_ok=True):
    m = Engrams(cfg=cfg)
    eps = {}
    for t, v in zip(conv.turns, conv.vecs):
        key = t["session"] if "group" not in t else (t["group"], t["session"])
        if key not in eps: eps[key] = m.add_episode(t["date"])
        m.add_turn(t["id"], eps[key], t["speaker"], t["text"], t["cap"], v)
    m.sleep()
    return m


def score(rows):
    if not rows: return {}
    return dict(recall=float(np.mean([r[0] for r in rows])), complete=float(np.mean([r[1] for r in rows])),
                tokens=float(np.mean([r[2] for r in rows])), turns=float(np.mean([r[3] for r in rows])))


def run_flat(convs, kind, k):
    rows = []; ms = []; per = defaultdict(list)
    for c in convs:
        f = Flat(c)
        for q, qv in zip(c.qs, c.qvecs):
            t0 = time.perf_counter()
            r = f.dense(qv) if kind == "dense" else f.bm25(q["q"]) if kind == "bm25" else f.hybrid(q["q"], qv)
            top = r[:k].tolist()
            ms.append((time.perf_counter() - t0) * 1e3)
            got = {c.turns[i]["id"] for i in top}
            rec = len(got & q["ev"]) / len(q["ev"])
            row = (rec, rec == 1.0, sum(tok(c.bodies[i]) for i in top), len(top))
            rows.append(row); per[q["cat"]].append(row)
    return dict(score(rows), p50=float(np.percentile(ms, 50)), p95=float(np.percentile(ms, 95)), by={CATS[k]: score(v) for k, v in per.items()})


def run_eng(convs, cfg, models=None):
    rows = []; ms = []; per = defaultdict(list); touched = []; esc = []
    for ci, c in enumerate(convs):
        m = build(c, cfg)
        for q, qv in zip(c.qs, c.qvecs):
            tr = {}
            t0 = time.perf_counter()
            ns = m.retrieve(q["q"], qv, use_habit=False, trace=tr)
            ctx = m.reinstate(ns)
            ms.append((time.perf_counter() - t0) * 1e3)
            got = {m.engrams[n].id for n in ns}
            rec = len(got & q["ev"]) / len(q["ev"])
            row = (rec, rec == 1.0, tok(ctx) if ns else 0, len(ns))
            rows.append(row); per[q["cat"]].append(row); touched.append(tr.get("touched", 0)); esc.append(tr.get("escalated", False))
    return dict(score(rows), p50=float(np.percentile(ms, 50)), p95=float(np.percentile(ms, 95)), touched=float(np.mean(touched)), escalated=float(np.mean(esc)),
                by={CATS[k]: score(v) for k, v in per.items()})


if __name__ == "__main__":
    t0 = time.time(); enc = load_encoder()
    cache = "convs.pkl"
    if os.path.exists(cache):
        convs = pickle.load(open(cache, "rb"))
    else:
        convs = [Conv(s, enc) for s in DATA]; pickle.dump(convs, open(cache, "wb"))
    dev, test = convs[:5], convs[5:]
    print("turns", sum(len(c.turns) for c in convs), "questions", sum(len(c.qs) for c in convs), "dev q", sum(len(c.qs) for c in dev), "test q", sum(len(c.qs) for c in test), "setup s", round(time.time() - t0, 1))
    print("mean tokens all turns per conv", np.mean([sum(tok(b) for b in c.bodies) for c in convs]))
    # tuning grid on development conversations only
    best = []
    for ep, kmin, kmax, sp, mg, sem in itertools.product((4, 8, 12), (3, 5), (5, 8, 12), (0, 1), (.15, .25, .4), (0.5, 1.0)):
        if kmin > kmax: continue
        cfg = Config(episodes=ep, k_min=kmin, k_max=kmax, spread_each=sp, margin=mg, w_sem=sem)
        r = run_eng(dev, cfg)
        best.append((r["recall"], r["tokens"], ep, kmin, kmax, sp, mg, sem))
    json.dump(best, open("grid_dev.json", "w"))
    print("grid", len(best), "elapsed", round(time.time() - t0))
    for row in sorted(best, key=lambda x: -x[0])[:5]: print(row)
    print('flat dev', {f"{kind}@{k}": (round(run_flat(dev, kind, k)["recall"], 3), round(run_flat(dev, kind, k)["tokens"])) for kind in ("dense", "bm25", "hybrid") for k in (5, 10, 20)})
