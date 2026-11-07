"""Fetches the 35 papers of research_memory/papers.jsonl from PubMed Central as plain text with their tables.
Usage: python3 fetch_corpus.py OUT_FOLDER. Writes OUT_FOLDER/xml/ and OUT_FOLDER/text/. The papers are not kept in this
repository because several licences allow only non-commercial sharing. Where a publisher blocks the full text, the
PubMed abstract is used and the file says so; where the PMC text has no abstract, the PubMed abstract is added."""
import json, os, re, sys, time, urllib.request
import xml.etree.ElementTree as ET
HERE = os.path.dirname(os.path.abspath(__file__))
E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"


def fetch(url, path=None):
    for i in range(5):
        try:
            data = urllib.request.urlopen(url, timeout=90).read()
            if path: open(path, "wb").write(data)
            time.sleep(0.4)
            return data
        except Exception as e:
            err = e; time.sleep(3 + 3 * i)
    raise err


def txt(e): return re.sub(r"\s+", " ", "".join(e.itertext())).strip() if e is not None else ""
def table(tw):
    rows = []
    label, cap = txt(tw.find("label")), txt(tw.find("caption"))
    for tr in tw.iter("tr"): rows.append(" | ".join(txt(c) for c in tr if c.tag in ("td", "th")))
    foot = txt(tw.find("table-wrap-foot"))
    return f"[{label}. {cap}]\n" + "\n".join(rows) + (f"\n[Table notes: {foot}]" if foot else "")
def section(sec, depth=0):
    parts = []
    t = txt(sec.find("title"))
    if t: parts.append("#" * (depth + 2) + " " + t)
    for ch in sec:
        if ch.tag == "p":
            inner = [x for x in ch.iter("table-wrap")]
            for x in inner: ch.remove(x) if x in list(ch) else None
            parts.append(txt(ch)); parts += [table(x) for x in inner]
        elif ch.tag == "sec": parts.append(section(ch, depth + 1))
        elif ch.tag == "table-wrap": parts.append(table(ch))
        elif ch.tag == "table-wrap-group": parts += [table(x) for x in ch.iter("table-wrap")]
    return "\n\n".join(p for p in parts if p)


def build(out_dir):
    os.makedirs(f"{out_dir}/xml", exist_ok=True); os.makedirs(f"{out_dir}/text", exist_ok=True)
    papers = [json.loads(l) for l in open(os.path.join(HERE, "papers.jsonl"))]
    for p in papers:
        pmc, path = p["pmcid"], f"{out_dir}/xml/{p['pmcid']}.xml"
        if not os.path.exists(path): fetch(f"{E}db=pmc&id={pmc[3:]}&retmode=xml", path)
        art = ET.parse(path).getroot().find(".//article")
        front, body = art.find("front"), art.find("body")
        if p["text"] == "abstract only":
            t = fetch(f"{E}db=pubmed&id={p['pmid']}&rettype=abstract&retmode=text").decode()
            open(f"{out_dir}/text/{pmc}.txt", "w").write(f"# Abstract only: the publisher does not allow the full text to be downloaded\n\nPMCID: {pmc}  PMID: {p['pmid']}\n\n{t}")
            continue
        title = txt(front.find(".//article-title"))
        abstract = "\n\n".join(section(a) if a.find("sec") is not None else txt(a) for a in front.iter("abstract") if not a.get("abstract-type"))
        body_text = "\n\n".join(section(s) if s.tag == "sec" else (txt(s) if s.tag == "p" else (table(s) if s.tag == "table-wrap" else "")) for s in body) if body is not None else ""
        floats = art.find("floats-group")
        extra = [table(t) for t in floats.iter("table-wrap")] if floats is not None else []
        if len(abstract.strip()) <= 200:
            t = fetch(f"{E}db=pubmed&id={p['pmid']}&rettype=abstract&retmode=text").decode()
            abstract = "(Taken from PubMed, because the PMC full text has no abstract.)\n\n" + t.strip() + "\n\n" + abstract
        full = f"# {title}\n\nPMCID: {pmc}\n\n## Abstract\n\n{abstract}\n\n{body_text}" + ("\n\n## Tables\n\n" + "\n\n".join(extra) if extra else "")
        open(f"{out_dir}/text/{pmc}.txt", "w").write(full)
    print(len(papers), "papers written to", out_dir, flush=True)


if __name__ == "__main__":
    build(sys.argv[1])
