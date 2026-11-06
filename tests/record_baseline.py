"""Measures retrieval on development conversations 0-4 and, when run directly, records it as the baseline.
Run from the repository root: python3 tests/record_baseline.py"""
import hashlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CODE = ["engram.py", "engram_parallel.py", "final_config.json"]


def measure():
    os.chdir(ROOT)
    import numpy as np
    import engram_chat as E
    import engram_parallel as P
    import vps_kit as K
    from engram import Config
    cfg = Config(**json.load(open("final_config.json")))
    tools = P._load_tools()
    tmp = tempfile.mkdtemp()
    out = {}

    def replay(conv, name, search):
        m = E.ChatMemory(f"{tmp}/{name}.db", cfg=cfg, durable=False, search=search)
        last = None
        for t in conv.turns:
            if t["session"] != last:
                if last is not None: m.close_chat(write_note=False, sleep=False)
                m.open_chat(t["date"]); last = t["session"]
            m.add(t["speaker"], t["text"], date=t["date"], ref=t["id"], caption=t["cap"], commit=False)
        m.db.commit(); m.close_chat(write_note=False)
        return m
    convs = {c: K.load_conv(K.DATA[c]) for c in range(5)}
    for system, make in (("current", lambda: None), ("plain", lambda: P.ChannelSearch("plain", tools=tools)), ("cabinet", lambda: P.ChannelSearch("cabinet", tools=tools))):
        recall, tokens = [], []
        for c, conv in convs.items():
            m = replay(conv, f"{system}{c}", make())
            ref = dict(m.db.execute("SELECT ref, sid FROM sentences").fetchall())
            for q in conv.qs:
                if q["cat"] == 5 or not q["ev"]: continue
                gold = {ref[e] for e in q["ev"] if e in ref}
                if not gold: continue
                sids, _ = m.find(q["q"])
                recall.append(len(set(sids) & gold) / len(gold)); tokens.append(sum(m._tokens(s) + 8 for s in sids))
            m.db.close()
        out[system] = dict(evidence=round(float(np.mean(recall)) * 100, 2), tokens=round(float(np.mean(tokens))), per_question=[round(x, 4) for x in recall])
    out["questions"] = len(out["current"]["per_question"])
    return out


if __name__ == "__main__":
    result = measure()
    result["code"] = {n: hashlib.sha256(open(os.path.join(ROOT, n), "rb").read()).hexdigest() for n in CODE}
    json.dump(result, open(os.path.join(ROOT, "tests", "fixtures", "retrieval_baseline.json"), "w"))
    print({k: (v["evidence"], v["tokens"]) for k, v in result.items() if k in ("current", "plain", "cabinet")}, result["questions"])
