"""Original P2.3 package against the fixed version. Run once with ORIGINAL=1 (package path first) and once without."""
import sys, json, collections, os
ORIG = os.environ.get("ORIGINAL")
if ORIG:
    sys.path.insert(0, ORIG)
    from engram_interpreter import interpret
    from engram_learning import evaluate
    msg = lambda sp, text, people: [(0, 0, interpret("user", text))]   # best case for the original: every speaker treated as user, whole message
else:
    sys.path.insert(0, "/home/claude/engram")
    from engram_knowledge.interpreter import interpret_message
    from engram_knowledge.learning import evaluate
    msg = lambda sp, text, people: interpret_message(sp, text, people=people)
def decide(parts):
    ds = [evaluate(x) for _, _, x in parts]
    names = {d.decision if d.decision != "ignore" or getattr(d, "keep_as", "none") == "none" else "assistant" for d in ds}
    for k in ("candidate", "review", "assistant"):
        if k in names: return k
    return "ignore"
mode = sys.argv[1]
if mode == "locomo":
    D = json.load(open("/home/claude/lm/locomo/data/locomo10.json")); c2 = collections.Counter()
    for c in D[:5]:
        turns = {t["dia_id"]: t for k, v in c["conversation"].items() if k.startswith("session_") and isinstance(v, list) for t in v}
        gold = {e for q in c["qa"] for e in q.get("evidence", []) if q.get("category") != 5}
        people = tuple({t["speaker"] for t in turns.values()})
        for i in gold:
            if i in turns: c2[decide(msg(turns[i]["speaker"], turns[i]["text"], people))] += 1
    print(dict(c2))
else:
    rows = [json.loads(l) for l in open(sys.argv[2])]
    people = tuple({r["speaker"] for r in rows if r["speaker"] not in ("user", "assistant")})
    out = {}
    for r in rows:
        sp = r["speaker"]
        if ORIG: parts = [(0, 0, interpret("user" if sp != "assistant" else "assistant", r["text"]))]
        else: parts = msg(sp, r["text"], people)
        d = decide(parts)
        out[r["id"]] = "learn" if d == "candidate" else d
    json.dump(out, open(sys.argv[3], "w"))
