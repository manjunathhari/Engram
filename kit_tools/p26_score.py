"""Scores the P2.6 rule writer and the validator against blind labels. Usage: p26_score.py sample.jsonl labels.json out.json"""
import json, os, re, sys, tempfile
sys.path.insert(0, os.environ.get("ENGRAM", "."))
import engram_chat as E
from engram_knowledge.extract import RuleWriter
from engram_knowledge.fact_validator import ACCEPTED, REJECTED, UNCERTAIN, KnowledgePipeline

DET = {"a", "an", "the", "my", "our", "some", "his", "her", "their", "new", "first"}
words = lambda t: [w for w in re.findall(r"[a-z0-9']+", str(t).lower()) if w not in DET]


def same_object(a, b, strict):
    x, y = words(a), words(b)
    if strict: return x == y
    return bool(x) and bool(y) and x[-1].rstrip("s") == y[-1].rstrip("s")  # the same head word


rows = [json.loads(l) for l in open(sys.argv[1])]
labels = json.load(open(sys.argv[2]))
m = E.ChatMemory(tempfile.mkdtemp() + "/p26.db")
pipe = KnowledgePipeline(m.db)
writer = RuleWriter()
gold_n = written = right_strict = right_head = time_ok = time_n = 0
verdicts = {ACCEPTED: [0, 0], UNCERTAIN: [0, 0], REJECTED: [0, 0]}  # [count, right by head word]
misses, wrong = [], []
for r in rows:
    date = r["said_on"] or "1:00 pm on 1 January, 2000"
    m.open_chat(date); sid = m.add(r["speaker"], r["text"], date=r["said_on"] or "", commit=True); m.close_chat(write_note=False, sleep=False)
    gold = labels[r["id"]]["facts"]
    facts = writer.extract(r["speaker"], r["text"], r["said_on"], sid=sid)
    gold_n += len(gold); written += len(facts)
    for f in facts:
        match = next((g for g in gold if g["predicate"] == f.predicate and same_object(g["object"], f.object, False)), None)
        right_head += match is not None; right_strict += bool(match and same_object(match["object"], f.object, True))
        if match and match.get("time_value") is not None:
            time_n += 1; time_ok += match["time_value"] == f.time_value
        if not match: wrong.append((r["id"], f.predicate, f.object, r["text"][:80]))
        d, _ = pipe.process_extracted(f)
        verdicts[d.verdict][0] += 1; verdicts[d.verdict][1] += match is not None
    for g in gold:
        if not any(g["predicate"] == f.predicate and same_object(g["object"], f.object, False) for f in facts): misses.append((r["id"], g["predicate"], g["object"]))
out = dict(sentences=len(rows), labelled_facts=gold_n, written=written, right_head_word=right_head, right_exact_object=right_strict, found_of_labelled=gold_n - len(misses),
           dated_and_right=[time_ok, time_n], validator={k: dict(count=v[0], right=v[1]) for k, v in verdicts.items()}, wrong=wrong, missed=misses)
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("wrong", "missed")}, indent=1))
