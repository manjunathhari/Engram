"""Filing cabinets for research papers: one record per table row, built by fixed code from the PMC XML. No model is involved.

Each record keeps the paper (first author and year), the table's label and caption, the row's label, and every cell under
its full column heading (stacked headings are joined, spanning cells are expanded). Each column is also marked robotic,
laparoscopic or other by fixed label rules, so a reader can tell the groups apart.

    records = build(xml_folder)                 # list of dicts
    text = pick(records, question, budget=800)  # the best-matching records as text, within a token budget
"""
import json, math, os, re
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = re.compile(r"\d")
N_ONLY = re.compile(r"^[^\d]*(\(?\s*[nN]\s*=\s*[\d,. ]+\s*\)?)?[^\d]*$")  # a heading such as "RS (n = 214)" holds no data


def text(e):
    return re.sub(r"\s+", " ", "".join(e.itertext())).strip() if e is not None else ""


def group_of(heading):
    """robotic, laparoscopic, p or other, from a column heading, by fixed rules. A stacked heading
    ("Overall / RALS group (n = 120) / n (%)") is read from the bottom line up until one names a group."""
    parts = [x for x in (heading or "").split(" / ") if x]
    if not parts: return "other"
    if re.search(r"\bp[\s-]*values?\b|^p\b|\bp\s*[*‡¶†§]|^p$", parts[-1].lower()): return "p"
    for h in reversed(parts):
        low = h.lower()
        if "robot" in low: return "robotic"
        if "laparoscop" in low or "conventional" in low: return "laparoscopic"
        for tok in re.findall(r"\b[A-Z][A-Za-z]*(?:-[A-Z][A-Za-z]*)?\b", h):
            if tok.upper() != tok or len(tok) < 2: continue
            if tok.startswith("R"): return "robotic"
            if tok.startswith(("L", "CL")): return "laparoscopic"
    return "other"


def grid(rows):
    """Expands colspan and rowspan into a plain grid of cell texts."""
    out, carry = [], {}
    for tr in rows:
        cells, col = [], 0
        for c in [c for c in tr if c.tag in ("td", "th")]:
            while (len(out), col) in carry: cells.append(carry.pop((len(out), col))); col += 1
            t, span, down = text(c), int(c.get("colspan", "1") or 1), int(c.get("rowspan", "1") or 1)
            for k in range(span):
                cells.append(t)
                for d in range(1, down): carry[(len(out) + d, col + k)] = t
            col += span
        while (len(out), col) in carry: cells.append(carry.pop((len(out), col))); col += 1
        out.append(cells)
    return out


def is_data(row):
    return any(NUM.search(c) and not N_ONLY.match(c) for c in row[1:])


def table_records(tw, paper):
    label, caption = text(tw.find("label")), text(tw.find("caption"))
    head = [tr for th in tw.iter("thead") for tr in th.iter("tr")]
    body = [tr for tb in tw.iter("tbody") for tr in tb.iter("tr")] or ([tr for tr in tw.iter("tr")] if not head else [])
    hg, bg = grid(head), grid(body)
    if not hg:  # no marked heading: the leading rows without data are the heading
        while bg and not is_data(bg[0]): hg.append(bg.pop(0))
    width = max([len(r) for r in hg + bg] or [0])
    headings = []
    for j in range(width):
        parts = []
        for r in hg:
            c = r[j] if j < len(r) else ""
            if c and c not in parts: parts.append(c)
        headings.append(" / ".join(parts))
    out, section = [], ""
    for r in bg:
        if not r or not any(r): continue
        first = r[0]
        if not is_data(r):
            if first and not any(r[1:]): section = first  # a row naming a group of rows, such as "Complications"
            continue
        cells = []
        for j, c in enumerate(r[1:], 1):
            if not c: continue
            h = headings[j] if j < len(headings) else ""
            cells.append(dict(heading=h, group=group_of(h), value=c))
        row = f"{section}: {first}" if section and section != first else first
        out.append(dict(paper=paper["label"], pmcid=paper["pmcid"], table=label, caption=caption, row=row, cells=cells))
    return out


def label_of(p):
    return f"{(p.get('first_author') or '?').split()[0]} {p.get('year') or ''}".strip()


def build(xml_folder):
    papers = [json.loads(l) for l in open(os.path.join(HERE, "papers.jsonl"))]
    records = []
    for p in papers:
        path = os.path.join(xml_folder, p["pmcid"] + ".xml")
        if p["text"] == "abstract only" or not os.path.exists(path): continue
        p = dict(p, label=label_of(p))
        art = ET.parse(path).getroot().find(".//article")
        for tw in art.iter("table-wrap"):
            records += table_records(tw, p)
    return records


def as_text(r):
    cells = " | ".join(f"{c['heading'] or 'column'}{' (' + c['group'] + ')' if c['group'] in ('robotic', 'laparoscopic') else ''}: {c['value']}" for c in r["cells"])
    return f"[{r['paper']}, {r['table']}: {r['caption']}] {r['row']} | {cells}"


WORD = re.compile(r"[a-z0-9]+")


def words(t): return [w for w in WORD.findall(t.lower()) if len(w) > 1]


class Index:
    """Scores records by the rarity of the question's words they share, with the paper's own words (author, year,
    title) counted twice so a question naming a study finds that study's rows."""
    def __init__(self, records, papers):
        self.records, self.texts = records, [as_text(r) for r in records]
        title = {p["pmcid"]: f"{label_of(p)} {p.get('title') or ''}" for p in papers}
        self.bags = [set(words(t)) for t in self.texts]
        self.paper_bags = [set(words(title.get(r["pmcid"], ""))) for r in records]
        n = len(records)
        df = {}
        for b in self.bags:
            for w in b: df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + n / d) for w, d in df.items()}

    def pick(self, question, budget=800):
        q = set(words(question))
        scored = []
        for i, (b, pb) in enumerate(zip(self.bags, self.paper_bags)):
            s = sum(self.idf.get(w, 0) for w in q & b) + sum(self.idf.get(w, 0) for w in q & pb)
            if s > 0: scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out, used = [], 0
        for _, i in scored:
            t = self.texts[i]
            size = len(t) // 4 + 1
            if used + size > budget: continue
            out.append(t); used += size
            if used >= budget - 20: break
        return "\n".join(out), used


if __name__ == "__main__":
    import sys
    recs = build(sys.argv[1])
    json.dump(recs, open(sys.argv[2], "w"), ensure_ascii=False)
    print(len(recs), "records")
