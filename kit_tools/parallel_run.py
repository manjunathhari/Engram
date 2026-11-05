"""Replays benchmark conversations through ChatMemory with the search layer attached and reports evidence found.
Benchmark data only. Each conversation is searched by networks trained on the others. No language model."""
import json, os, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.environ.get("ENGRAM", "."))
import vps_kit as K
import engram_chat as E
from engram import Config
import engram_parallel as P

ids = [0, 1, 2, 3, 4]
SEEDS = [1, 2, 3, 4, 5]
TYPES = {4: "single", 1: "multi", 2: "date", 3: "inference"}
cfg = Config(**json.load(open("final_config.json")))
tools = P._load_tools()
tmp = tempfile.mkdtemp()


def replay(conv, name, search):
    t0 = time.perf_counter()
    m = E.ChatMemory(f"{tmp}/{name}.db", cfg=cfg, durable=False, search=search)
    last = None
    for t in conv.turns:
        if t["session"] != last:
            if last is not None: m.close_chat(write_note=False, sleep=False)
            m.open_chat(t["date"]); last = t["session"]
        m.add(t["speaker"], t["text"], date=t["date"], ref=t["id"], caption=t["cap"], commit=False)
    m.db.commit()
    m.close_chat(write_note=False)
    return m, time.perf_counter() - t0


convs = {c: K.load_conv(K.DATA[c]) for c in ids}
mems, plain_mem, qs, build = {}, {}, {}, {}
for c in ids:
    mems[c], build[c] = replay(convs[c], f"s{c}", P.ChannelSearch("plain", tools=tools))
    plain_mem[c], _ = replay(convs[c], f"p{c}", None)
    ref = dict(mems[c].db.execute("SELECT ref, sid FROM sentences").fetchall())
    qs[c] = [(q["q"], {ref[e] for e in q["ev"] if e in ref}, q["cat"]) for q in convs[c].qs if q["cat"] != 5 and q["ev"]]
    qs[c] = [x for x in qs[c] if x[1]]
    print("replayed", c, len(mems[c].search.sids), "sentences", round(build[c], 1), "s", flush=True)
keys = [(c, n) for c in ids for n in range(len(qs[c]))]
cats = np.array([qs[c][n][2] for c, n in keys])
print("questions", len(keys), flush=True)


def evaluate(fn):
    rec, tok, ms = [], [], []
    for c, n in keys:
        q, gold, _ = qs[c][n]
        out, info = fn(c, q)
        rec.append(len(set(out) & gold) / len(gold)); tok.append(info["tokens"]); ms.append(info["ms"])
    return np.array(rec), float(np.mean(tok)), float(np.mean(ms))


def with_mode(mode):
    def f(c, q):
        s = mems[c].search; s.mode = mode
        return mems[c].find(q)
    return f


R = {"current": evaluate(lambda c, q: plain_mem[c].find(q)), "plain": evaluate(with_mode("plain")), "cabinet": evaluate(with_mode("cabinet"))}
for seed in SEEDS:
    for c in ids:
        ex = [(mems[d].search, q, gold) for d in ids if d != c for q, gold, _ in qs[d]]
        mems[c].search.model = P.train(ex, seed)
    R[f"network{seed}"] = evaluate(with_mode("network"))
    print("seed", seed, round(R[f"network{seed}"][0].mean() * 100, 1), flush=True)
net = np.mean([R[f"network{s}"][0] for s in SEEDS], axis=0)
R["network"] = (net, np.mean([R[f"network{s}"][1] for s in SEEDS]), np.mean([R[f"network{s}"][2] for s in SEEDS]))
rng = np.random.default_rng(0)
base = R["plain"][0]
out = {}
for k in ("current", "plain", "cabinet", "network"):
    r, tok, ms = R[k]
    d = r - base
    bo = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(4000)]
    out[k] = dict(evidence=round(float(r.mean() * 100), 1), tokens=round(tok), ms=round(float(ms), 1), vs_plain=round(float(d.mean() * 100), 1),
                  ci=[round(float(np.percentile(bo, 2.5)), 1), round(float(np.percentile(bo, 97.5)), 1)], **{TYPES[t]: round(float(r[cats == t].mean() * 100), 1) for t in TYPES})
    print(k, out[k], flush=True)
out["seeds"] = [round(float(R[f"network{s}"][0].mean() * 100), 1) for s in SEEDS]
print("seeds", out["seeds"], flush=True)

# Checks on the joined system
c = 0
m = mems[c]
before = m.db.execute("SELECT count(*), sum(length(text)) FROM sentences").fetchone()
q = qs[c][0][0]
first = m.find(q)[0]
path = f"{tmp}/s{c}.db"
model = m.search.model
m.close()
m2 = E.ChatMemory(path, cfg=cfg, durable=False, search=P.ChannelSearch("network", model=model, tools=tools))
assert m2.db.execute("SELECT count(*), sum(length(text)) FROM sentences").fetchone() == before, "record changed"
assert m2.find(q)[0] == first, "search differs after restart"
e = E.ChatMemory(f"{tmp}/empty.db", cfg=cfg, durable=False, search=P.ChannelSearch("plain", tools=tools))
assert e.find("anything at all")[0] == [], "empty memory should find nothing"
e.open_chat("1 May 2023"); e.add("A", "My dog is called Rex.", date="1 May 2023")
assert e.find("dog")[0] == [] , "sentence on the desk should not return twice"
e.close_chat(write_note=False)
assert e.find("What is the dog called?")[0], "closed chat not found"
print("checks passed: record unchanged, restart gives the same search, empty and open chats behave", flush=True)
json.dump(out, open("joined_results.json", "w"), indent=1)
print("Done", flush=True)
