"""Mechanism check for changing facts. Scenarios are written by me, not a public benchmark, with real LoCoMo chat mixed in as noise.
'seen' wordings use verbs the rules know; 'unseen' wordings do not."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, pickle, random, types
import numpy as np
from dataclasses import replace
from bench import Conv, Flat
from engram import Config, Engrams
rng = random.Random(11)
convs = pickle.load(open("convs.pkl", "rb"))
noise = [t for c in convs[:5] for t in c.turns if not t["cap"]]
PEOPLE = "Asha Ravi Meera Karan Nisha Arjun Divya Suresh Priya Vikram Lakshmi Rohan Anita Sanjay Kavya Mohan Deepa Naveen Rekha Tarun".split()
CITIES = "Paris Rome Tokyo Madrid Oslo Lima Cairo Seoul Vienna Dublin Delhi Sydney".split()
COS = "Acme Globex Initech Umbrella Soylent Hooli Stark Wayne Tyrell Wonka".split()
T = {"home": dict(seen=["I live in {v} now.", "We moved to {v} last month.", "I relocated to {v} for good."], unseen=["{v} is home for us these days.", "We've made our base in {v}.", "Our new address is in {v}."], q="Where does {p} live now?", qp="Where did {p} live in {y}?", vals=CITIES),
     "job": dict(seen=["I work at {v} now.", "I joined {v} this month.", "I was hired at {v}."], unseen=["{v} is my employer these days.", "My new office is at {v}.", "I took a role with {v}."], q="Where does {p} work now?", qp="Where did {p} work in {y}?", vals=COS)}
def scenario(kind):
    turns, qs = [], []
    ses = []
    for y, mon in ((2022, "March"), (2023, "June"), (2024, "September")):
        for s in range(3): ses.append((f"1:00 pm on {5+s} {mon}, {y}", y))
    for i, p in enumerate(PEOPLE):
        for slot, d in T.items():
            a, b = rng.sample(d["vals"], 2)
            sa, sb = rng.randrange(0, 3), rng.randrange(6, 9)
            turns.append((sa, p, rng.choice(d[kind]).format(v=a), f"{p}-{slot}-old"))
            turns.append((sb, p, rng.choice(d[kind]).format(v=b), f"{p}-{slot}-new"))
            qs.append(dict(q=d["q"].format(p=p), want=f"{p}-{slot}-new", stale=f"{p}-{slot}-old", kind="current"))
            qs.append(dict(q=d["qp"].format(p=p, y=ses[sa][1]), want=f"{p}-{slot}-old", stale=f"{p}-{slot}-new", kind="past"))
    by = {}
    for t in turns: by.setdefault(t[0], []).append(t)
    for s in range(len(ses)):
        for x in rng.sample(noise, 40): by.setdefault(s, []).append((s, x["speaker"], x["text"], f"n{rng.random()}"))
        rng.shuffle(by[s])
    return ses, by, qs
def run(kind, world):
    ses, by, qs = scenario(kind)
    cfg = replace(Config(**json.load(open("results_v3_final.json"))["config"]), world=True, world_gate=True, semantic=False, w_imp=0) if world else replace(Config(**json.load(open("results_v3_final.json"))["config"]), world=False, semantic=False, w_imp=0)
    m = Engrams(cfg=cfg); eps = {}
    for s, (d, y) in enumerate(ses):
        eps[s] = m.add_episode(d)
        for _, sp, tx, id_ in by[s]: m.add_turn(id_, eps[s], sp, tx)
    m.sleep()
    res = {"current": [], "past": []}
    for q in qs:
        ns = m.retrieve(q["q"], None, use_habit=False)
        ids = [m.engrams[n].id for n in ns]
        pw = ids.index(q["want"]) if q["want"] in ids else 99
        ps = ids.index(q["stale"]) if q["stale"] in ids else 99
        res[q["kind"]].append((pw < ps, ids[0] == q["want"] if ids else False))
    return {k: dict(right_before_stale=float(np.mean([a for a, _ in v])), top1=float(np.mean([b for _, b in v])), n=len(v)) for k, v in res.items()}
def baseline(kind):
    ses, by, qs = scenario(kind)
    c = types.SimpleNamespace(turns=[], bodies=[], vecs=None)
    for s, (d, y) in enumerate(ses):
        for _, sp, tx, id_ in by[s]:
            c.turns.append(dict(id=id_, session=s)); c.bodies.append(f"[{d}] {sp}: {tx}")
    f = Flat(c); res = {"current": [], "past": []}
    for q in qs:
        ids = [c.turns[i]["id"] for i in f.bm25(q["q"])[:12].tolist()]
        pw = ids.index(q["want"]) if q["want"] in ids else 99
        ps = ids.index(q["stale"]) if q["stale"] in ids else 99
        res[q["kind"]].append((pw < ps, ids[0] == q["want"]))
    return {k: dict(right_before_stale=float(np.mean([a for a, _ in v])), top1=float(np.mean([b for _, b in v])), n=len(v)) for k, v in res.items()}
out = {}
for kind in ("seen", "unseen"):
    rng.seed(11); out[f"baseline_{kind}"] = baseline(kind)
    rng.seed(11); out[f"nowworld_{kind}"] = run(kind, False)
    rng.seed(11); out[f"world_{kind}"] = run(kind, True)
    print(kind, {k: v for k, v in out.items() if k.endswith(kind)}, flush=True)
json.dump(out, open("results_updates.json", "w"), indent=1)
