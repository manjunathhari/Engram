"""Handover test for Engram chat memory. Run it from this folder on the machine that can reach your model.

A chat closes and the model writes a note. A new chat then opens and asks about the closed chat.
The model answers four ways: with no memory, the note only, retrieved memories only, or both.

  check  One note and one answer, to show timing and memory.
  run    The handover test. Safe to stop and restart. Results go to the results folder.
  locomo The full LoCoMo benchmark on all 10 conversations. Safe to stop and restart.
"""
import argparse, json, os, random, sys, tempfile, time
from collections import defaultdict

import vps_kit as K  # also moves into this folder
from engram import Config
import engram_chat as E

SYSTEMS = ("none", "note", "memory", "note+memory")


def jsonl(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def choose(convs, n, used):
    """Questions whose evidence all sits in one chat, not used in the earlier test, balanced by type."""
    groups = defaultdict(list)
    for ci, c in convs.items():
        order = {t["id"]: t["session"] for t in c.turns}
        for i, q in enumerate(c.qs):
            if q["cat"] == 5 or (ci, i) in used: continue
            if len({order[e] for e in q["ev"]}) == 1: groups[q["cat"]].append((ci, i, order[next(iter(q["ev"]))]))
    rng = random.Random(2027)
    for g in groups.values(): rng.shuffle(g)
    out = []
    while any(groups.values()) and len(out) < n:
        for c in sorted(groups):
            if groups[c] and len(out) < n: out.append(groups[c].pop())
    return out


def memory_up_to(c, session, cfg, llm=None):
    """A memory holding this conversation's chats up to and including the given one, all closed."""
    path = tempfile.mktemp(suffix=".db")
    m = E.ChatMemory(path, cfg, durable=False, llm=llm); ref = {}; last = None
    for t in c.turns:
        if t["session"] > session: break
        if t["session"] != last:
            if last is not None: m.close_chat(write_note=False, sleep=False)
            last = t["session"]; m.open_chat(t["date"])
        ref[t["id"]] = m.add(t["speaker"], t["text"], t["date"], t["id"], t["cap"], commit=False)
    chat = m.chat
    m.close_chat(write_note=False, sleep=True)
    return m, ref, chat, path


def cleanup(path):
    for x in ("", "-wal", "-shm"):
        if os.path.exists(path + x): os.remove(path + x)


def note_for(notes, key, m, chat, llm, path):
    if key not in notes:
        t0 = time.time()
        text = llm(E.NOTE_PROMPT.format(previous="", chat=m.chat_text(chat)), max_tokens=m.s.note_tokens)[0]
        notes[key] = dict(text=text, secs=time.time() - t0)
        json.dump(notes, open(path, "w"), indent=0)
    return notes[key]["text"]


def run(a, llm, convs, cfg):
    answers = f"results/handover_{a.tag}.jsonl"; notes_path = f"results/handover_notes_{a.tag}.json"
    notes = json.load(open(notes_path)) if os.path.exists(notes_path) else {}
    done = {(r["conv"], r["i"], r["system"]) for r in jsonl(answers)}
    used = {(r["conv"], r["i"]) for r in jsonl("results/answers_g3.jsonl")}
    plan = choose(convs, a.sample, used)
    print(f"{len(plan)} questions, {len(SYSTEMS)} ways, {len({(c, s) for c, _, s in plan})} notes", flush=True)
    t0 = time.time()
    with open(answers, "a") as out:
        for n, (ci, i, session) in enumerate(plan, 1):
            todo = [s for s in SYSTEMS if (ci, i, s) not in done]
            if not todo: continue
            c = convs[ci]; q = c.qs[i]
            m, ref, chat, path = memory_up_to(c, session, cfg)
            note = note_for(notes, f"{ci}:{session}", m, chat, llm, notes_path)
            sids, info = m.find(q["q"])
            recall = len({e for e in q["ev"] if ref[e] in set(sids)}) / len(q["ev"])
            retrieved = m.text_of(sids)
            for system in todo:
                parts = []
                if "note" in system: parts.append("[Handover note from the last chat]\n" + note)
                if "memory" in system and retrieved: parts.append(retrieved)
                ctx = "\n".join(parts) or "(no excerpts)"
                prompt = K.ANSWER.format(ctx=ctx, q=q["q"])
                pred, use, secs = llm(prompt, max_tokens=a.answer_tokens)
                out.write(json.dumps(dict(conv=ci, i=i, system=system, cat=K.CATS[q["cat"]], session=session, q=q["q"], gold=q["answer"], pred=pred, f1=K.f1(pred, q["answer"]),
                                          recall=recall if "memory" in system else 0.0, ctx_tokens=K.tok(ctx), prompt_tokens=use.get("prompt_tokens"), answer_tokens=use.get("completion_tokens"), secs=secs)) + "\n"); out.flush()
            m.close(); cleanup(path)
            left = (time.time() - t0) / n * (len(plan) - n)
            print(f"question {n}/{len(plan)}; about {left / 3600:.1f} h left", flush=True)
    rows = jsonl(answers); S = {}
    for s in SYSTEMS:
        r = [x for x in rows if x["system"] == s]
        if r: S[s] = dict(n=len(r), f1=round(sum(x["f1"] for x in r) / len(r), 3), not_mentioned=round(sum("not mentioned" in x["pred"].lower() for x in r) / len(r), 3), ctx_tokens=round(sum(x["ctx_tokens"] for x in r) / len(r)), secs=round(sum(x["secs"] for x in r) / len(r), 1))
    S["notes"] = dict(n=len(notes), secs=round(sum(v["secs"] for v in notes.values()) / max(1, len(notes)), 1), tokens=round(sum(K.tok(v["text"]) for v in notes.values()) / max(1, len(notes))))
    print(json.dumps(S, indent=1)); json.dump(S, open(f"results/handover_{a.tag}_summary.json", "w"), indent=1)
    print("Done. f1 is only a rough signal; the answers still need scoring blind.")


def locomo(a, llm, convs, cfg):
    """The full benchmark: each conversation is its own memory, closed chat by chat, with every question asked at the end.
    Date questions are answered a second time without the resolved dates, as a control."""
    answers = f"results/locomo_{a.tag}.jsonl"
    done = {(r["conv"], r["i"], r["system"]) for r in jsonl(answers)}
    order = K.balanced(convs, list(convs), 0)
    print(f"{len(order)} questions from {len(convs)} conversations", flush=True)
    mems = {}
    t0 = time.time(); count = 0
    with open(answers, "a") as out:
        for n, (ci, i) in enumerate(order, 1):
            q = convs[ci].qs[i]
            systems = ["engram"] + (["no_dates"] if q["cat"] == 2 else [])
            systems = [s for s in systems if (ci, i, s) not in done]
            if not systems: continue
            if ci not in mems: mems[ci] = memory_up_to(convs[ci], 10 ** 9, cfg)
            m, ref, _, _ = mems[ci]
            sids, info = m.find(q["q"])
            recall = len({e for e in q["ev"] if ref[e] in set(sids)}) / len(q["ev"])
            for system in systems:
                m.show_dates = system == "engram"
                ctx = m.text_of(sids) or "(no excerpts)"
                pred, use, secs = llm(K.ANSWER.format(ctx=ctx, q=q["q"]), max_tokens=a.answer_tokens)
                out.write(json.dumps(dict(conv=ci, i=i, system=system, cat=K.CATS[q["cat"]], q=q["q"], gold=q["answer"], pred=pred, f1=K.f1(pred, q["answer"]), recall=recall, ctx_tokens=K.tok(ctx), prompt_tokens=use.get("prompt_tokens"), answer_tokens=use.get("completion_tokens"), secs=secs, lookup_ms=info["ms"])) + "\n"); out.flush()
                count += 1
            left = (time.time() - t0) / max(1, count) * ((len(order) - n) * 1.2)
            print(f"question {n}/{len(order)}; about {left / 3600:.1f} h left", flush=True)
    for m, _, _, path in mems.values():
        m.close(); cleanup(path)
    rows = jsonl(answers); S = {}
    for s in ("engram", "no_dates"):
        r = [x for x in rows if x["system"] == s]
        if r: S[s] = dict(n=len(r), f1=round(sum(x["f1"] for x in r) / len(r), 3), recall=round(sum(x["recall"] for x in r) / len(r), 3), not_mentioned=round(sum("not mentioned" in x["pred"].lower() for x in r) / len(r), 3), ctx_tokens=round(sum(x["ctx_tokens"] for x in r) / len(r)), secs=round(sum(x["secs"] for x in r) / len(r), 1))
    print(json.dumps(S, indent=1)); json.dump(S, open(f"results/locomo_{a.tag}_summary.json", "w"), indent=1)
    print("Done. f1 is only a rough signal; the answers still need scoring blind.")


def check(a, llm, convs, cfg):
    plan = choose(convs, a.sample, {(r["conv"], r["i"]) for r in jsonl("results/answers_g3.jsonl")})
    ci, i, session = plan[0]; c = convs[ci]
    m, ref, chat, path = memory_up_to(c, session, cfg)
    t0 = time.time(); note = llm(E.NOTE_PROMPT.format(previous="", chat=m.chat_text(chat)), max_tokens=m.s.note_tokens)[0]; t1 = time.time() - t0
    print(f"Note, {t1:.0f} s, about {K.tok(note)} tokens:\n{note}\n")
    t0 = time.time(); p = llm(K.ANSWER.format(ctx="[Handover note from the last chat]\n" + note, q=c.qs[i]["q"]), max_tokens=a.answer_tokens)[0]; t2 = time.time() - t0
    print(f"Question: {c.qs[i]['q']}\nAnswer from the note, {t2:.0f} s: {p}\nBenchmark answer: {c.qs[i]['answer']}")
    m.close(); cleanup(path)
    notes = len({(x, s) for x, _, s in plan})
    print(f"\nFull test: {len(plan)} questions and {notes} notes. Estimate: {(notes * t1 + len(plan) * 4 * max(t2, 5)) / 3600:.1f} h")
    json.dump(dict(note_secs=t1, answer_secs=t2, questions=len(plan), notes=notes), open(f"results/check_handover_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["check", "run", "locomo"])
    p.add_argument("--url", default="http://localhost:11434/v1"); p.add_argument("--model", required=True)
    p.add_argument("--key", default=os.environ.get("LLM_API_KEY", "local")); p.add_argument("--tag", default="g3")
    p.add_argument("--convs", default="5,6,7,8,9"); p.add_argument("--sample", type=int, default=60)
    p.add_argument("--answer-tokens", type=int, default=60); p.add_argument("--timeout", type=int, default=1200)
    a = p.parse_args(); a.convs = [int(x) for x in a.convs.split(",")]
    os.makedirs("results", exist_ok=True)
    cfg = Config(**json.load(open("final_config.json")))
    llm = K.LLM(a.url, a.model, a.key, a.timeout)
    print("Checking the model:", llm("Reply with the single word ready.", max_tokens=5)[0], flush=True)
    if a.mode == "locomo": a.convs = list(range(10))
    convs = {i: K.load_conv(K.DATA[i]) for i in a.convs}
    {"check": check, "run": run, "locomo": locomo}[a.mode](a, llm, convs, cfg)
