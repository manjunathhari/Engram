"""Scores the research-memory test from two blind markings. Usage: python3 score.py key.json marks_A.json marks_B.json out.json
A row's mark is the average of the two markers (fixed before any mark was seen). Primary: both minus search_more,
95% interval by paired bootstrap over the 40 questions (10,000 resamples, seed 7)."""
import json, random, sys
key, A, B = (json.load(open(x)) for x in sys.argv[1:4])
mark = {}
for row, k in key.items():
    mark[(k["id"], k["way"])] = (A[row]["mark"] + B[row]["mark"]) / 2
ids = sorted({k["id"] for k in key.values()})
kind = {json.loads(l)["id"]: json.loads(l)["kind"] for l in open("research_memory/questions.jsonl")}
WAYS = ("none", "search", "search_more", "cabinets", "both")
per = {w: [mark[(i, w)] for i in ids] for w in WAYS}
agree = sum(abs(A[r]["mark"] - B[r]["mark"]) < 0.25 for r in key) / len(key)
def boot(x, y, n=10000):
    rnd = random.Random(7); d = [a - b for a, b in zip(x, y)]
    m = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(n))
    return round(100 * sum(d) / len(d), 1), round(100 * m[int(.025 * n)], 1), round(100 * m[int(.975 * n)], 1)
out = dict(questions=len(ids), markers_agree=round(agree, 3),
           score={w: round(100 * sum(per[w]) / len(ids), 1) for w in WAYS},
           primary_both_minus_search_more=boot(per["both"], per["search_more"]),
           cabinets_minus_search=boot(per["cabinets"], per["search"]),
           search_more_minus_search=boot(per["search_more"], per["search"]),
           by_kind={k: {w: round(100 * sum(mark[(i, w)] for i in ids if kind[i] == k) / sum(kind[i] == k for i in ids), 1) for w in WAYS} for k in sorted(set(kind.values()))},
           won_lost_both_vs_search_more=[sum(a > b for a, b in zip(per["both"], per["search_more"])), sum(a < b for a, b in zip(per["both"], per["search_more"]))])
json.dump(out, open(sys.argv[4], "w"), indent=1)
print(json.dumps(out, indent=1))
