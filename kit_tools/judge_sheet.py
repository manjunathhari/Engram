"""Make a blinded scoring sheet from answers_*.jsonl: same questions, system names hidden, order shuffled. Then join scores back."""
import json, random, sys
if sys.argv[1] == "make":
    rows = [json.loads(x) for x in open(sys.argv[2])]
    random.Random(99).shuffle(rows)
    key = {f"a{n}": (r["conv"], r["i"], r["system"]) for n, r in enumerate(rows)}
    json.dump(key, open(sys.argv[2] + ".key.json", "w"))
    with open(sys.argv[2] + ".sheet.jsonl", "w") as o:
        for n, r in enumerate(rows): o.write(json.dumps(dict(id=f"a{n}", cat=r["cat"], q=r["q"], gold=r["gold"], pred=r["pred"])) + "\n")
else:  # score: answers.jsonl labels.json, labels = {id: {"generous": 0/1, "strict": 0/1}}
    key = json.load(open(sys.argv[2] + ".key.json")); lab = json.load(open(sys.argv[3])); t = {}
    for k, v in lab.items():
        sysname = key[k][2]; t.setdefault(sysname, []).append(v)
    print({s: dict(n=len(v), generous=sum(x["generous"] for x in v) / len(v), strict=sum(x["strict"] for x in v) / len(v)) for s, v in t.items()})
