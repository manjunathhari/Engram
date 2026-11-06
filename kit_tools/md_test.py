"""Engram against a plain .md memory file, on LongMemEval. Benchmark data only. Rule fixed in frozen_md_baseline.json.

The .md memory works like the memory in everyday chat apps: after each conversation the same model (Gemma 3 4B) writes a short
note, and the notes, dated and in order, form one memory file. To answer, the model reads that file and the question, with the
same answer prompt Engram used. Engram's answers are the ones already recorded in results/lme_answers.jsonl (networks, no
facts); nothing in Engram is run or changed here.

  1. timing: notes for 24 sessions from the development questions, which never count
  2. plan:   the number of test questions that fit the time budget, written to md_plan.json before any test note exists
  3. notes:  one note per session, written once however many questions hold it; resumes where it stopped
  4. answers: md_full (the whole file, trimmed only if over 3,000 tokens) and md_equal (the latest notes within the same
     number of tokens Engram gave the model for that question)
"""
import argparse, concurrent.futures, datetime, hashlib, json, os, re, time, urllib.request

DEV_QUESTIONS = 30
NOTE_INPUT_TOKENS = 3000
FULL_FILE_TOKENS = 3000
LONGEST_SENTENCE = 600

NOTE = """Below is one conversation between a user and an AI assistant, held on {date}. Write a memory note for the user's personal memory file, to be read in later conversations. Keep the specific facts worth remembering: about the user (events, people, places, possessions, preferences, plans, numbers, dates) and anything specific the assistant gave (names, recommendations, figures). Write dates as calendar dates, worked out from {date}. At most 40 words. If nothing is worth remembering, write NONE.

Conversation:
{chat}

Memory note:"""

# Exactly the answer prompt of the Engram runs (vps_kit.ANSWER), checked against vps_kit.py at start.
ANSWER = """Below are excerpts from past conversations. Answer the question using only them, in as few words as you can. Resolve relative dates using the date in brackets. If they do not contain the answer, reply exactly: not mentioned

Excerpts:
{ctx}

Question: {q}
Answer:"""


def tokens(text): return len(text) // 4  # the same rough count the earlier kit used for prompt limits


def split_sentences(text):
    out = []
    for piece in re.split(r"(?<=[.!?])\s+|\s*\n+\s*", text or ""):
        piece = piece.strip()
        while len(piece) > LONGEST_SENTENCE:
            cut = piece.rfind(" ", 0, LONGEST_SENTENCE); cut = cut if cut > 0 else LONGEST_SENTENCE
            out.append(piece[:cut].strip()); piece = piece[cut:].strip()
        if piece: out.append(piece)
    return out


def session_date(text):
    d = datetime.datetime.strptime(re.sub(r"\s*\(\w+\)\s*", " ", text.strip()), "%Y/%m/%d %H:%M")
    return f"{d.hour % 12 or 12}:{d.minute:02d} {'am' if d.hour < 12 else 'pm'} on {d.day} {d.strftime('%B')}, {d.year}"


def hash_order(qid): return hashlib.sha256(qid.encode()).hexdigest()


def split_questions(data, dev=DEV_QUESTIONS):
    ordered = sorted(data, key=lambda q: hash_order(q["question_id"]))  # the same split as kit_tools/lme_test.py
    return ordered[:dev], ordered[dev:]


def session_key(session_id, session):
    return session_id + ":" + hashlib.md5(json.dumps(session, sort_keys=True).encode()).hexdigest()[:8]


def chat_text(session, limit=NOTE_INPUT_TOKENS):
    """The conversation for the note prompt. If too long, the user's sentences are kept first, then the assistant's, in order,
    as in the earlier kit's fact prompts. Returns the text and how many sentences were left out."""
    lines = [(t["role"].capitalize(), s) for t in session for s in split_sentences(t["content"])]
    if tokens("\n".join(f"{r}: {s}" for r, s in lines)) <= limit: return "\n".join(f"{r}: {s}" for r, s in lines), 0
    keep, used = set(), 0
    for wanted in ("User", "Assistant"):
        for n, (r, s) in enumerate(lines):
            size = len(r) + len(s) + 3
            if r == wanted and used + size <= limit * 4: keep.add(n); used += size
    return "\n".join(f"{r}: {s}" for n, (r, s) in enumerate(lines) if n in keep), len(lines) - len(keep)


def memory_file(notes, budget):
    """The .md file: one dated line per note, oldest first, NONE left out. Over budget, the oldest notes go first."""
    lines = [f"- [{date}] {text}" for date, text in notes if text and not text.strip().upper().startswith("NONE")]
    kept, used = [], 0
    for line in reversed(lines):
        if used + tokens(line) + 1 > budget: break
        kept.append(line); used += tokens(line) + 1
    return "\n".join(reversed(kept)), len(kept), len(lines)


class LLM:
    def __init__(self, url, model): self.url, self.model = url.rstrip("/") + "/chat/completions", model

    def __call__(self, prompt, max_tokens):
        body = json.dumps(dict(model=self.model, messages=[dict(role="user", content=prompt)], max_tokens=max_tokens, temperature=0.0, seed=7)).encode()
        err = None
        for attempt in range(3):
            try:
                t0 = time.time()
                r = json.load(urllib.request.urlopen(urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"}), timeout=1200))
                return r["choices"][0]["message"]["content"].strip(), time.time() - t0
            except Exception as e:
                err = e; time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"model call failed after 3 tries: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--save", required=True)
    ap.add_argument("--engram-answers", default="results/lme_answers.jsonl"); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    rule = json.load(open("frozen_md_baseline.json"))
    vps = open("vps_kit.py").read()
    assert ANSWER in vps, "the answer prompt differs from the one Engram's answers were made with"
    data = json.load(open(a.data))
    dev, test = split_questions(data)
    engram = {r["id"]: r for r in map(json.loads, open(a.engram_answers)) if r["system"] == "no_facts"}
    assert all(q["question_id"] in engram for q in test), "Engram's recorded answers do not cover the test questions"
    llm = LLM(a.url, a.model)
    print("first reply:", repr(llm("What is 2 + 2? Reply with just the number.", 5)[0]), flush=True)

    note_path = f"{a.save}/md_notes.json"
    notes = json.load(open(note_path)) if os.path.exists(note_path) else {}

    def sessions_of(questions):
        out = {}
        for q in questions:
            for i, s, d in zip(q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]):
                out.setdefault(session_key(i, s), (s, session_date(d)))
        return out

    def write(todo, label):
        todo = {k: v for k, v in todo.items() if k not in notes}
        print(f"{label}: {len(todo)} sessions to note", flush=True)

        def one(item):
            key, (session, date) = item
            text, dropped = chat_text(session)
            note, secs = llm(NOTE.format(date=date, chat=text), 100)
            return key, dict(note=note, date=date, secs=round(secs, 1), dropped=dropped)
        t0, n = time.time(), 0
        with concurrent.futures.ThreadPoolExecutor(a.threads) as pool:
            for key, v in pool.map(one, todo.items()):
                notes[key] = v; n += 1
                if n % 100 == 0 or n == len(todo):
                    json.dump(notes, open(note_path, "w"))
                    print(f"{label}: {n}/{len(todo)} notes, {(time.time() - t0) / n:.2f} s each", flush=True)
        json.dump(notes, open(note_path, "w"))
        return (time.time() - t0) / max(1, n)

    # 1 and 2: timing on development sessions, then the plan, fixed before any test note is written
    plan_path = f"{a.save}/md_plan.json"
    if os.path.exists(plan_path):
        plan = json.load(open(plan_path))
    else:
        each = write(dict(list(sessions_of(dev).items())[:24]), "timing")
        seen, fits = set(), 0
        for n, q in enumerate(test, 1):
            seen |= {session_key(i, s) for i, s in zip(q["haystack_session_ids"], q["haystack_sessions"])}
            if len(seen) * each <= rule["note_hours_budget"] * 3600: fits = n
        n_questions = min(len(test), max(rule["min_questions"], fits))
        plan = dict(seconds_per_note=round(each, 2), questions=n_questions, sessions=len(sessions_of(test[:n_questions])),
                    estimated_hours=round(len(sessions_of(test[:n_questions])) * each / 3600, 1), fixed_at=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
        json.dump(plan, open(plan_path, "w"), indent=1)
    print("plan:", json.dumps(plan), flush=True)
    questions = test[:plan["questions"]]

    # 3 notes
    write(sessions_of(questions), "notes")

    # 4 answers
    path = f"{a.save}/md_answers.jsonl"
    done = {(r["id"], r["system"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    with open(path, "a") as out:
        for n, q in enumerate(questions, 1):
            ordered = sorted(zip(q["haystack_dates"], q["haystack_session_ids"], q["haystack_sessions"]), key=lambda x: x[0])
            mine = [(notes[k]["date"], notes[k]["note"]) for k in (session_key(i, s) for _, i, s in ordered)]
            asked = f"{q['question']} (Today is {session_date(q['question_date'])}.)"
            for system, budget in (("md_full", FULL_FILE_TOKENS), ("md_equal", engram[q["question_id"]]["tokens"])):
                if (q["question_id"], system) in done: continue
                ctx, kept, total = memory_file(mine, budget)
                pred, secs = llm(ANSWER.format(ctx=ctx or "(no notes)", q=asked), 60)
                out.write(json.dumps(dict(id=q["question_id"], type=q["question_type"], not_answerable=q["question_id"].endswith("_abs"), system=system, q=q["question"],
                                          gold=str(q["answer"]), pred=pred, tokens=tokens(ctx), notes_kept=kept, notes_total=total, secs=round(secs, 1))) + "\n")
                out.flush()
            if n % 25 == 0 or n == len(questions): print(f"answers: {n}/{len(questions)} questions", flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
