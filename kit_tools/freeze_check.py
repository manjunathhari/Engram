"""Proves the real ChatMemory path gives Gemma exactly the text used in the winning facts test.
Development conversations only: conversations 5 to 9 are never loaded here. No language model."""
import json, os, sys, tempfile
import numpy as np
sys.argv = sys.argv + []
sys.path.insert(0, os.environ.get("ENGRAM", ".")); sys.path.insert(0, os.environ["CHECK_DIR"])
import vps_kit as K
import engram_chat as E
from engram import Config
import engram_parallel as P
cfg = Config(**json.load(open("final_config.json")))
tools = P._load_tools()
ids = [0, 1, 2, 3, 4]
tmp = tempfile.mkdtemp()
convs = {c: K.load_conv(K.DATA[c]) for c in ids}
saved = json.load(open("results/facts_dev_gemma3-4b.json"))
def replay(c):
    state = {}
    def helper(prompt, max_tokens=0): return saved[state["key"]]["raw"], {}, 0.
    m = E.ChatMemory(f"{tmp}/{c}.db", cfg=cfg, durable=False, search=P.ChannelSearch("network", model=[{}], tools=tools), helper=helper, helper_name="saved 4B replies", use_facts=True)
    last = None
    for t in convs[c].turns:
        if t["session"] != last:
            if last is not None: state["key"] = f"{c}:{last}"; m.close_chat(write_note=False, sleep=False)
            m.open_chat(t["date"]); last = t["session"]
        m.add(t["speaker"], t["text"], date=t["date"], ref=t["id"], caption=t["cap"], commit=False)
    m.db.commit(); state["key"] = f"{c}:{last}"; m.close_chat(write_note=False)
    return m
M = {c: replay(c) for c in ids}
print("facts stored", sum(m.db.execute("SELECT count(*) FROM facts").fetchone()[0] for m in M.values()), flush=True)
REF = {c: dict(M[c].db.execute("SELECT ref, sid FROM sentences").fetchall()) for c in ids}
GOLD = {c: [(q, {REF[c][e] for e in q["ev"] if e in REF[c]}) for q in convs[c].qs if q["cat"] != 5 and q["ev"]] for c in ids}
for c in ids:
    M[c].search.model = [P.train([(M[d].search, q["q"], g) for d in ids if d != c for q, g in GOLD[d] if g], s) for s in (1, 2, 3)]
plan = [(c, i) for c, i in K.balanced(convs, ids, 240) if convs[c].qs[i]["cat"] != 5]
plan = [(c, i) for c, i in plan if any(e in REF[c] for e in convs[c].qs[i]["ev"])]
out = {}
for c, i in plan:
    q = convs[c].qs[i]
    ctx, info = M[c].context_for(q["q"])
    gold = {REF[c][e] for e in q["ev"] if e in REF[c]}
    out[f"{c}:{i}"] = dict(ctx=ctx, recall=len(set(info["sids"]) & gold) / len(gold), tokens=info["tokens"])
json.dump(out, open(os.environ["CHECK_DIR"] + "/real_path.json", "w"))
print("questions", len(out), "evidence", round(np.mean([v["recall"] for v in out.values()]) * 100, 1), "tokens", round(np.mean([v["tokens"] for v in out.values()])), flush=True)
