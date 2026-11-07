"""Checks questions.jsonl against the corpus. Usage: python3 check_quotes.py questions.jsonl [corpus folder]

For every question: each cited paper is in the corpus, the quote is in that paper's text word for word (spacing ignored),
and the answer's numbers appear in the quote. A quote may be one string or a list of strings; it may be spread over the cited
papers. Not-answerable questions need no quote but must cite no paper. Prints one line per question and a summary."""
import json, os, re, sys
qfile = sys.argv[1]; corpus = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
texts = {f[:-4]: open(os.path.join(corpus, "text", f), encoding="utf-8").read() for f in os.listdir(os.path.join(corpus, "text"))}
tight = lambda t: re.sub(r"\s+", " ", t).strip()
loose = lambda t: re.sub(r"[\s  ]+", "", t).replace("−", "-").replace("–", "-").replace("—", "-").replace("‐", "-").replace("‑", "-").replace("’", "'").replace("“", '"').replace("”", '"').lower()
nums = lambda t: set(re.findall(r"\d+(?:[.,]\d+)?", t.replace("−", "-")))
rows = [json.loads(l) for l in open(qfile, encoding="utf-8") if l.strip()]
counts = {"ok": 0, "loose": 0, "fail": 0}
kinds = {}
for q in rows:
    kinds[q.get("kind", "?")] = kinds.get(q.get("kind", "?"), 0) + 1
    studies = q.get("study") or []
    studies = [studies] if isinstance(studies, str) else studies
    quotes = q.get("quote") or []
    quotes = [quotes] if isinstance(quotes, str) else quotes
    problems, level = [], "ok"
    unanswerable = str(q.get("answer", "")).strip().lower() == "not reported"
    if unanswerable:
        if studies or quotes: problems.append("a not-answerable question should cite no paper and no quote")
    else:
        missing = [s for s in studies if s not in texts]
        if missing: problems.append(f"not in the corpus: {missing}")
        if not quotes: problems.append("no quote")
        pool = [texts[s] for s in studies if s in texts]
        for part in quotes:
            if any(tight(part) in tight(t) for t in pool): continue
            if any(loose(part) in loose(t) for t in pool): level = "loose"; continue
            problems.append(f"quote not found: {part[:90]!r}")
        want = nums(str(q.get("answer", ""))) - nums(str(q.get("question", "")))
        absent = want - nums(" ".join(quotes)) if quotes else set()
        if absent: problems.append(f"answer numbers not in the quote: {sorted(absent)}")
    status = "fail" if problems else level
    counts[status] += 1
    print(f"{q.get('id')}: {status.upper()}" + ("" if status != "loose" else " (matches only if spaces, dashes and quote marks are ignored)") + "".join(f"\n    - {p}" for p in problems))
print(f"\n{len(rows)} questions: {counts['ok']} exact, {counts['loose']} loose, {counts['fail']} failed. Kinds: {kinds}")
