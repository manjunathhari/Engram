import os,re,time
from collections import defaultdict
import numpy as np
from bench import Flat, tok
from engram import Engrams, words

def per_q(convs, cfg, notes=None):
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



def flat_notes(convs, k, notes):
    rows = []
    for c, obs in zip(convs, notes):
        f = Flat(c); index = {t["id"]: i for i, t in enumerate(c.turns)}
        for sess in obs.values():
            for facts in sess.values():
                for fact in facts:
                    ids = fact[1] if isinstance(fact[1], str) else " ".join(fact[1])
                    for i in re.split(r"[;,\\s]+", ids):
                        i = re.sub(r"^D:?(\\d+):0*(\\d+)$", r"D\\1:\\2", i)
                        if i in index:
                            cnt = defaultdict(int)
                            for w in words(fact[0]): cnt[w] += 1
                            for w, n in cnt.items():
                                p = f.post[w]
                                j = next((a for a, x in enumerate(p) if x[0] == index[i]), None)
                                if j is None: p.append((index[i], n))
                                else: p[j] = (index[i], p[j][1] + n)
        f.idf = {t: __import__("math").log(1 + (f.N - len(p) + .5) / (len(p) + .5)) for t, p in f.post.items()}
        for q in c.qs:
            top = f.bm25(q["q"])[:k].tolist(); out, seen = [], set()
            for i in top:
                if i not in seen: out.append(i); seen.add(i)
                if i + 1 < len(c.turns) and c.turns[i + 1]["session"] == c.turns[i]["session"] and i + 1 not in seen:
                    out.append(i + 1); seen.add(i + 1)
            got = {c.turns[i]["id"] for i in out}
            rows.append((len(got & q["ev"]) / len(q["ev"]), sum(tok(c.bodies[i]) for i in out), q["cat"]))
    return rows
