"""Builds every prompt for the research-memory test, so the run on Kaggle only asks Gemma. Usage:
    python3 build_prompts.py CORPUS_FOLDER OUT.jsonl
CORPUS_FOLDER is what fetch_corpus.py wrote (xml/ and text/). Each line of OUT holds one question asked one way, with the
prompt, the text tokens given, and 'evidence': the share of the answer's numbers found in what the model is given.
Nothing here calls a model. The search is Engram's tested design, unchanged."""
import json, os, re, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from engram_memory import open_memory
import cabinets as C

# The answer prompt of the earlier tests (vps_kit.ANSWER), checked against vps_kit.py below.
ANSWER = """Below are excerpts from past conversations. Answer the question using only them, in as few words as you can. Resolve relative dates using the date in brackets. If they do not contain the answer, reply exactly: not mentioned

Excerpts:
{ctx}

Question: {q}
Answer:"""
LONGEST = 600  # characters per stored sentence, as in the LongMemEval kit
WAYS = ("none", "search", "search_more", "cabinets", "both")


def sentences(text):
    """Table rows (lines with ' | ') stay whole; other lines are split into sentences. No words are changed."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        pieces = [line] if " | " in line or line.startswith("[") else re.split(r"(?<=[.!?])\s+", line)
        for piece in pieces:
            piece = piece.strip()
            while len(piece) > LONGEST:
                cut = piece.rfind(" ", 0, LONGEST); cut = cut if cut > 0 else LONGEST
                out.append(piece[:cut].strip()); piece = piece[cut:].strip()
            if piece: out.append(piece)
    return out


def numbers(t):
    return re.findall(r"\d+(?:\.\d+)?", t.replace(",", ""))


def evidence(answer, ctx):
    want = numbers(answer)
    if not want: return None
    have = set(numbers(ctx))
    return round(sum(n in have for n in want) / len(want), 3)


def main(corpus, out_path):
    vps = open(os.path.join(ROOT, "vps_kit.py")).read()
    assert ANSWER in vps, "answer prompt differs from the tested one"
    papers = [json.loads(l) for l in open(os.path.join(HERE, "papers.jsonl"))]
    questions = [json.loads(l) for l in open(os.path.join(HERE, "questions.jsonl"))]
    m = open_memory(os.path.join(tempfile.mkdtemp(), "papers.db"))
    for n, p in enumerate(papers, 1):
        label = C.label_of(p)
        date = f"12:00 pm on 1 July, {p.get('year') or 2000}"  # the year is all the papers give here
        m.open_chat(date)
        for k, s in enumerate(sentences(open(os.path.join(corpus, "text", p["pmcid"] + ".txt")).read())):
            m.add(label, s, date=date, ref=f"{p['pmcid']}:{k}", commit=False)
        m.db.commit()
        m.close_chat(write_note=False, sleep=n == len(papers))
    print("sentences stored:", m.db.execute("SELECT count(*) FROM sentences").fetchone()[0], flush=True)
    index = C.Index(C.build(os.path.join(corpus, "xml")), papers)
    with open(out_path, "w") as out:
        for q in questions:
            m.s.retrieved_tokens = 800
            search, _ = m.context_for(q["question"])
            m.s.retrieved_tokens = 1100
            more, _ = m.context_for(q["question"])
            m.s.retrieved_tokens = 800
            cab, _ = index.pick(q["question"], 800)
            small, _ = index.pick(q["question"], 300)
            ctx = {"none": "(none)", "search": search, "search_more": more, "cabinets": cab, "both": small + "\n" + search}
            for way in WAYS:
                c = ctx[way]
                out.write(json.dumps(dict(id=q["id"], kind=q["kind"], way=way, prompt=ANSWER.format(ctx=c, q=q["question"]), tokens=len(c) // 4,
                                          evidence=None if way == "none" else evidence(q["answer"], c)), ensure_ascii=False) + "\n")
    print("prompts written:", len(questions) * len(WAYS), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
