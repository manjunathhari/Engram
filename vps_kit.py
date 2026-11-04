#!/usr/bin/env python3
"""Run on your own server, next to Gemma. Needs numpy; talks to any OpenAI-compatible endpoint.

  check    Measure this machine and model, then estimate how long each test will take. Run this first.
  answers  Do correct answers come back when the LLM reads Engram's context, compared with the baseline?
  notes    Let Gemma write sleep-time notes per session, and measure the change in recall.
  updates  Let Gemma extract changing facts, and test them on sentences hand-written rules could not read.

Work resumes where it stopped: finished items are kept in results/ as they complete.
"""
import argparse, json, os, re, string, sys, time, types, urllib.request
from collections import Counter, defaultdict
from dataclasses import replace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
from bench import Flat, evidence, tok, DATA, CATS  # noqa: E402
from engram import Config, Engrams, words  # noqa: E402


def load_conv(s):
    c = s["conversation"]; turns = []
    for i in sorted(int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k)):
        for t in c[f"session_{i}"]:
            turns.append(dict(id=t["dia_id"], session=i, date=c[f"session_{i}_date_time"], speaker=t["speaker"], text=t["text"], cap=t.get("blip_caption", "")))
    ids = {t["id"] for t in turns}
    qs = []
    for q in s["qa"]:
        ev = evidence(q, ids)
        if ev: qs.append(dict(q=q["question"], ev=ev, cat=q["category"], answer=str(q.get("answer", "")), adv=q.get("adversarial_answer")))
    bodies = [f"[{t['date']}] {t['speaker']}: {t['text']}" + (f" [image: {t['cap']}]" if t["cap"] else "") for t in turns]
    return types.SimpleNamespace(turns=turns, ids=ids, qs=qs, bodies=bodies, vecs=None, qvecs=None)


class LLM:
    def __init__(self, url, model, key, timeout):
        self.url, self.model, self.key, self.timeout = url.rstrip("/") + "/chat/completions", model, key, timeout

    def __call__(self, prompt, max_tokens=120, temperature=0.0):
        body = json.dumps(dict(model=self.model, messages=[dict(role="user", content=prompt)], max_tokens=max_tokens, temperature=temperature, seed=7)).encode()
        for attempt in range(3):
            try:
                req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
                t0 = time.time()
                r = json.load(urllib.request.urlopen(req, timeout=self.timeout))
                return r["choices"][0]["message"]["content"].strip(), r.get("usage", {}), time.time() - t0
            except Exception as e:
                err = e; time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after 3 tries: {err}")


def norm(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w not in ("a", "an", "the", "and")]


def f1(pred, gold):
    p, g = norm(pred), norm(gold)
    same = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not same: return 0.0
    a, b = same / len(p), same / len(g)
    return 2 * a * b / (a + b)


def build(c, cfg, notes=None):
    m = Engrams(cfg=cfg); eps = {}
    for t in c.turns:
        if t["session"] not in eps: eps[t["session"]] = m.add_episode(t["date"])
        m.add_turn(t["id"], eps[t["session"]], t["speaker"], t["text"], t["cap"])
    for text, tid in (notes or []):
        if tid in m.by_id:
            n = m.by_id[tid]; cnt = Counter(words(text))
            for w, k in cnt.items():
                p = m.postings[w]; j = next((a for a, x in enumerate(p) if x[0] == n), None)
                if j is None: p.append((n, k))
                else: p[j] = (n, p[j][1] + k)
    m.sleep()
    return m


def engram_ctx(m, q):
    ns = m.retrieve(q, None, use_habit=False)
    return m.reinstate(ns), {m.engrams[n].id for n in ns}


def baseline_ctx(f, c, q, k):
    out, seen = [], set()
    for i in f.bm25(q)[:k].tolist():
        for j in (i, i + 1):
            if j < len(c.turns) and c.turns[j]["session"] == c.turns[i]["session"] and j not in seen:
                out.append(j); seen.add(j)
    return "\n".join(c.bodies[i] for i in out), {c.turns[i]["id"] for i in out}


ANSWER = """Below are excerpts from past conversations. Answer the question using only them, in as few words as you can. Resolve relative dates using the date in brackets. If they do not contain the answer, reply exactly: not mentioned

Excerpts:
{ctx}

Question: {q}
Answer:"""
JUDGE = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. The gold answer will usually be concise. The generated answer might be much longer, but be generous: as long as it touches on the same topic as the gold answer, count it as CORRECT. For time questions, count it as CORRECT if it refers to the same date or time period, whatever the format.

Question: {q}
Gold answer: {gold}
Generated answer: {pred}

Reply with only CORRECT or WRONG."""


def balanced(convs, ids, n, adversarial=False):
    """A fixed question order that interleaves question types, so stopping at any point leaves a balanced set."""
    import random
    groups = defaultdict(list)
    for ci in ids:
        for i, q in enumerate(convs[ci].qs):
            if q["cat"] != 5 or adversarial: groups[q["cat"]].append((ci, i))
    rng = random.Random(2026)
    for g in groups.values(): rng.shuffle(g)
    order = []
    while any(groups.values()):
        for c in sorted(groups):
            if groups[c]: order.append(groups[c].pop())
    return order[: n] if n else order


def jsonl(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def run_answers(a, llm, convs, cfg):
    path = f"results/answers_{a.tag}.jsonl"; done = {(r["conv"], r["i"], r["system"]) for r in jsonl(path)}
    models = {ci: (build(convs[ci], cfg), Flat(convs[ci])) for ci in a.convs}
    order = balanced(convs, a.convs, a.sample, a.adversarial)
    t0 = time.time()
    with open(path, "a") as out:
        for n, (ci, i) in enumerate(order, 1):
            c = convs[ci]; q = c.qs[i]; m, f = models[ci]
            for system in a.systems.split(","):
                if (ci, i, system) in done: continue
                if system == "engram": ctx, got = engram_ctx(m, q["q"])
                elif system == "baseline": ctx, got = baseline_ctx(f, c, q["q"], a.k)
                elif system == "oracle":
                    ns = [m.by_id[e] for e in sorted(q["ev"], key=lambda x: m.by_id[x])]
                    ctx, got = m.reinstate(ns), set(q["ev"])
                elif system in ("open1", "open2"):
                    eps = list(dict.fromkeys(m.engrams[x].episode for x in m.retrieve(q["q"], None, use_habit=False)))[: int(system[-1])]
                    ns = [x for ep in sorted(eps) for x in m.ep_members[ep]]
                    ctx, got = m.reinstate(ns), {m.engrams[x].id for x in ns}
                elif system == "none": ctx, got = "(no excerpts)", set()
                elif system == "full": ctx, got = "\n".join(c.bodies), c.ids
                else: sys.exit(f"Unknown system {system}")
                prompt = ANSWER.format(ctx=ctx, q=q["q"])
                if tok(prompt) + a.answer_tokens > a.ctx:
                    print(f"skipped {system}: prompt is about {tok(prompt)} tokens, over the {a.ctx} limit", flush=True); continue
                pred, use, secs = llm(prompt, max_tokens=a.answer_tokens)
                gold = q["answer"] or "not mentioned"; judged = None
                if a.judge:
                    judged = llm(JUDGE.format(q=q["q"], gold=gold, pred=pred), max_tokens=4)[0].upper().startswith("CORRECT")
                out.write(json.dumps(dict(conv=ci, i=i, system=system, cat=CATS[q["cat"]], q=q["q"], gold=gold, pred=pred, f1=f1(pred, gold), judged=judged,
                                          recall=(len(got & q["ev"]) / len(q["ev"])), ctx_tokens=tok(ctx), prompt_tokens=use.get("prompt_tokens"), answer_tokens=use.get("completion_tokens"), secs=secs)) + "\n"); out.flush()
            left = (time.time() - t0) / n * (len(order) - n)
            print(f"question {n}/{len(order)}; about {left / 3600:.1f} h left", flush=True)
    rows = jsonl(path); s = {}
    for system in sorted({r["system"] for r in rows}):
        x = [r for r in rows if r["system"] == system]
        s[system] = dict(n=len(x), f1=float(np.mean([r["f1"] for r in x])), recall=float(np.mean([r["recall"] for r in x])), ctx_tokens=float(np.mean([r["ctx_tokens"] for r in x])), secs=float(np.mean([r["secs"] for r in x])),
                         f1_by_type={k: float(np.mean([r["f1"] for r in x if r["cat"] == k])) for k in sorted({r["cat"] for r in x})})
        if a.judge: s[system]["local_judge_correct"] = float(np.mean([bool(r["judged"]) for r in x]))
    json.dump(s, open(f"results/answers_{a.tag}_summary.json", "w"), indent=1); print(json.dumps(s, indent=1))


def run_check(a, llm, convs, cfg):
    mem = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith(("MemTotal", "MemAvailable"))} if os.path.exists("/proc/meminfo") else {}
    print(f"CPU cores: {os.cpu_count()}; RAM total {mem.get('MemTotal', '?')} MB, available {mem.get('MemAvailable', '?')} MB", flush=True)
    ci = a.convs[0]; c = convs[ci]; m = build(c, cfg); f = Flat(c)
    order = balanced(convs, [ci], 3)
    print("Retrieval speed and first answers:", flush=True)
    times = []
    for _, i in order:
        q = c.qs[i]; t = time.perf_counter(); ctx, _ = engram_ctx(m, q["q"]); ms = (time.perf_counter() - t) * 1e3
        pred, use, secs = llm(ANSWER.format(ctx=ctx, q=q["q"]), max_tokens=a.answer_tokens); times.append(secs)
        print(f"  read {ms:.2f} ms; model {secs:.0f} s; prompt tokens {use.get('prompt_tokens')}, answer tokens {use.get('completion_tokens')}\n  Q: {q['q']}\n  gold: {q['answer']}\n  answer: {pred}", flush=True)
    per = float(np.mean(times[1:] or times)); sys_n = len(a.systems.split(","))
    print(f"\nAbout {per:.0f} s per answer (the first call may include loading the model).")
    for n in (60, 120, 300, 776): print(f"  {n} questions, {sys_n} systems: about {n * sys_n * per / 3600:.1f} h")
    json.dump(dict(cores=os.cpu_count(), ram=mem, model=a.model, secs_per_answer=per), open(f"results/check_{a.tag}.json", "w"), indent=1)


NOTE = """Below is one session of a conversation. Write the lasting facts it contains: who did what, plans, preferences, relationships, dates. Each line must be one plain sentence, then a tab, then the id in square brackets of the turn it came from. Write nothing else.

Session date: {date}
{text}"""


def run_notes(a, llm, convs, cfg):
    path = f"results/notes_{a.tag}.json"; store = json.load(open(path)) if os.path.exists(path) else {}
    secs = []
    for ci in a.convs:
        c = convs[ci]; ses = defaultdict(list)
        for t in c.turns: ses[t["session"]].append(t)
        for s, ts in ses.items():
            key = f"{ci}:{s}"
            if key in store: continue
            text = "\n".join(f"[{t['id']}] {t['speaker']}: {t['text']}" for t in ts)
            r, _, sec = llm(NOTE.format(date=ts[0]["date"], text=text), max_tokens=a.note_tokens); secs.append(sec)
            store[key] = [[m[0].strip(), m[1]] for m in re.findall(r"^(.+?)\s*\[(D\d+:\d+)\]\s*$", r, re.M)]
            json.dump(store, open(path, "w")); print(f"notes {key}: {len(store[key])} facts, {sec:.0f} s", flush=True)
    res = {}
    for name, use in (("no notes", False), ("gemma notes", True)):
        rec, toks = [], []
        for ci in a.convs:
            c = convs[ci]; notes = [x for k, v in store.items() if k.startswith(f"{ci}:") for x in v] if use else None
            m = build(c, cfg, notes)
            for q in c.qs:
                ctx, got = engram_ctx(m, q["q"]); rec.append(len(got & q["ev"]) / len(q["ev"])); toks.append(tok(ctx))
        res[name] = dict(recall=float(np.mean(rec)), tokens=float(np.mean(toks)), n=len(rec))
    res["seconds_per_session"] = float(np.mean(secs)) if secs else None
    json.dump(res, open(f"results/notes_{a.tag}_summary.json", "w"), indent=1); print(json.dumps(res, indent=1))


EXTRACT = """Does this sentence, said by {who}, state where {who} lives or works? Reply with exactly one word: home, job, or neither.
Sentence: {text}"""


def run_updates(a, llm, cfg):
    import random
    rng = random.Random(11)
    PEOPLE = "Asha Ravi Meera Karan Nisha Arjun Divya Suresh Priya Vikram Lakshmi Rohan Anita Sanjay Kavya Mohan Deepa Naveen Rekha Tarun".split()
    CITIES = "Paris Rome Tokyo Madrid Oslo Lima Cairo Seoul Vienna Dublin Delhi Sydney".split(); COS = "Acme Globex Initech Umbrella Soylent Hooli Stark Wayne Tyrell Wonka".split()
    T = {"home": (["{v} is home for us these days.", "We've made our base in {v}.", "Our new address is in {v}."], "Where does {p} live now?", "Where did {p} live in {y}?", CITIES),
         "job": (["{v} is my employer these days.", "My new office is at {v}.", "I took a role with {v}."], "Where does {p} work now?", "Where did {p} work in {y}?", COS)}
    ses = [(f"1:00 pm on {5 + s} {mon}, {y}", y) for y, mon in ((2022, "March"), (2023, "June"), (2024, "September")) for s in range(3)]
    noise = [t for s in DATA[:5] for k, v in s["conversation"].items() if isinstance(v, list) for t in v if "blip_caption" not in t]
    by, qs = defaultdict(list), []
    for p in PEOPLE:
        for slot, (forms, qn, qp, vals) in T.items():
            x, y2 = rng.sample(vals, 2); sa, sb = rng.randrange(0, 3), rng.randrange(6, 9)
            by[sa].append((p, rng.choice(forms).format(v=x), f"{p}-{slot}-old")); by[sb].append((p, rng.choice(forms).format(v=y2), f"{p}-{slot}-new"))
            qs.append(dict(q=qn.format(p=p), want=f"{p}-{slot}-new", stale=f"{p}-{slot}-old", kind="current")); qs.append(dict(q=qp.format(p=p, y=ses[sa][1]), want=f"{p}-{slot}-old", stale=f"{p}-{slot}-new", kind="past"))
    for s in range(len(ses)):
        for x in rng.sample(noise, 40): by[s].append((x["speaker"], x["text"], f"n{rng.random()}"))
        rng.shuffle(by[s])
    m = Engrams(cfg=replace(cfg, world=True, world_gate=True, semantic=False)); eps = {}; correct = n = 0
    for s, (d, _) in enumerate(ses):
        eps[s] = m.add_episode(d)
        for sp, tx, id_ in by[s]:
            m.add_turn(id_, eps[s], sp, tx)
    m.world_state.clear()  # empty the rule-built table; Gemma alone fills it
    for e in m.engrams:
        if e.id.startswith("n") and not a.check_noise: continue
        text = m.cortex[e.store][e.id].split(": ", 1)[-1]; who = e.speaker.title()
        r, _, _ = llm(EXTRACT.format(who=who, text=text), max_tokens=3); r = r.lower().strip(" .")
        if r in ("home", "job"): m.world_state[(e.speaker, r)].append(m.by_id[e.id])
        if not e.id.startswith("n"):
            n += 1; correct += r == e.id.split("-")[1]
    m.sleep()
    res = defaultdict(list)
    for q in qs:
        ids = [m.engrams[x].id for x in m.retrieve(q["q"], None, use_habit=False)]
        a_, b_ = (ids.index(q["want"]) if q["want"] in ids else 99), (ids.index(q["stale"]) if q["stale"] in ids else 99)
        res[q["kind"]].append(a_ < b_)
    out = dict(extraction_accuracy=correct / max(1, n), **{k: dict(right_before_stale=float(np.mean(v)), n=len(v)) for k, v in res.items()})
    json.dump(out, open(f"results/updates_{a.tag}.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["check", "answers", "notes", "updates"])
    p.add_argument("--url", default="http://localhost:11434/v1"); p.add_argument("--model", required=True)
    p.add_argument("--key", default=os.environ.get("LLM_API_KEY", "local")); p.add_argument("--tag", default="run1")
    p.add_argument("--convs", default="5,6,7,8,9", help="conversations to use; 5-9 are the held-out set")
    p.add_argument("--sample", type=int, default=120, help="questions in total, balanced by type; 0 means all. Safe to stop early")
    p.add_argument("--judge", action="store_true", help="also have the local model score answers; slower, and small models score poorly")
    p.add_argument("--ctx", type=int, default=4096, help="the model's context limit, in tokens")
    p.add_argument("--answer-tokens", type=int, default=60); p.add_argument("--note-tokens", type=int, default=300)
    p.add_argument("--systems", default="engram,baseline,oracle", help="also none, open1, open2. oracle uses only the labelled answer turns, to show the model's ceiling")
    p.add_argument("--k", type=int, default=6); p.add_argument("--adversarial", action="store_true")
    p.add_argument("--check-noise", action="store_true", help="updates: also classify the noise turns; slower and stricter")
    p.add_argument("--timeout", type=int, default=1200); p.add_argument("--config", default="final_config.json")
    a = p.parse_args(); a.convs = [int(x) for x in a.convs.split(",")]
    os.makedirs("results", exist_ok=True)
    cfg = Config(**json.load(open(a.config)))
    llm = LLM(a.url, a.model, a.key, a.timeout)
    print("Checking the model:", llm("Reply with the single word ready.", max_tokens=5)[0], flush=True)
    convs = {i: load_conv(DATA[i]) for i in a.convs}
    if a.mode == "check": run_check(a, llm, convs, cfg)
    elif a.mode == "answers": run_answers(a, llm, convs, cfg)
    elif a.mode == "notes": run_notes(a, llm, convs, cfg)
    else: run_updates(a, llm, cfg)
