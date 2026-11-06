"""Tests the fixed Engram design on LongMemEval, data it has never seen. Benchmark data only.

Nothing is tuned on this data. The networks train on LoCoMo conversations 0 to 4, as in frozen_design.json.

  --stage stats   counts only, no content
  --stage pilot   30 development questions, to measure speed and check the adapter; these are never part of the test
  --stage test    every other question: current Engram against the networks, then facts on the subset named in frozen_design_lme.json

Work resumes where it stopped.
"""
import argparse, concurrent.futures, datetime, hashlib, json, os, pickle, re, shutil, sys, tempfile, time

DEV_QUESTIONS = 30
FACT_PROMPT_TOKENS = 3000  # 4,096 context less the 700 reply tokens, with a margin: too long a prompt would be cut silently
LONGEST_SENTENCE = 600  # characters


def split_sentences(text):
    """One message into sentences. No words are changed or dropped, only the spaces between pieces."""
    out = []
    for piece in re.split(r"(?<=[.!?])\s+|\s*\n+\s*", text or ""):
        piece = piece.strip()
        while len(piece) > LONGEST_SENTENCE:
            cut = piece.rfind(" ", 0, LONGEST_SENTENCE)
            cut = cut if cut > 0 else LONGEST_SENTENCE
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece: out.append(piece)
    return out


def session_date(text):
    """'2023/05/20 (Sat) 02:21' into the form the rest of Engram reads: '2:21 am on 20 May, 2023'."""
    d = datetime.datetime.strptime(re.sub(r"\s*\(\w+\)\s*", " ", text.strip()), "%Y/%m/%d %H:%M")
    return f"{d.hour % 12 or 12}:{d.minute:02d} {'am' if d.hour < 12 else 'pm'} on {d.day} {d.strftime('%B')}, {d.year}"


def hash_order(question_id):
    return hashlib.sha256(question_id.encode()).hexdigest()


def split_questions(data, dev=DEV_QUESTIONS):
    """Decided by the question id alone, before anything is known about the results."""
    ordered = sorted(data, key=lambda q: hash_order(q["question_id"]))
    return ordered[:dev], ordered[dev:]


def session_key(session_id, session):
    return session_id + ":" + hashlib.md5(json.dumps(session, sort_keys=True).encode()).hexdigest()[:8]


def session_numbers(data):
    """A fixed number per session, so its line ids and saved facts are the same in every question that holds it."""
    keys = sorted({session_key(i, s) for q in data for i, s in zip(q["haystack_session_ids"], q["haystack_sessions"])})
    return {k: n for n, k in enumerate(keys)}


def session_lines(session, number):
    """[(line id, speaker, sentence, turn number, whether the turn holds the answer)]"""
    lines, n = [], 0
    for t, turn in enumerate(session):
        for sentence in split_sentences(turn["content"]):
            n += 1
            lines.append((f"D{number}:{n}", turn["role"].capitalize(), sentence, t, bool(turn.get("has_answer"))))
    return lines


def fit_prompt(prompt, limit=FACT_PROMPT_TOKENS):
    """Keeps a fact prompt inside the context. The user's lines come first, then the assistant's, in order."""
    if len(prompt) // 4 <= limit: return prompt, 0
    head, rest = prompt.split("Conversation:\n", 1)
    body, tail = rest.rsplit("\n\nFacts:", 1)
    first, *lines = body.split("\n")
    room = limit * 4 - len(head) - len(first) - len(tail) - 40
    keep, used = set(), 0
    for wanted in (" User: ", " Assistant: "):
        for n, line in enumerate(lines):
            if wanted in line[:30] and n not in keep and used + len(line) + 1 <= room:
                keep.add(n); used += len(line) + 1
    kept = [x for n, x in enumerate(lines) if n in keep]
    return head + "Conversation:\n" + "\n".join([first] + kept) + "\n\nFacts:" + tail, len(lines) - len(kept)


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))] if values else 0


def stats(data, numbers, save):
    dev, test = split_questions(data)
    tokens = {}
    for q in data:
        for i, s in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            tokens[session_key(i, s)] = sum(len(t["content"]) for t in s) // 4
    per_q = [len(q["haystack_sessions"]) for q in data]
    seen, sizes = set(), {}
    for n, q in enumerate(test, 1):
        seen |= {session_key(i, s) for i, s in zip(q["haystack_session_ids"], q["haystack_sessions"])}
        if n in (25, 50, 75, 100, 150, 200, 300, len(test)): sizes[n] = len(seen)
    types = {}
    for q in test: types[q["question_type"]] = types.get(q["question_type"], 0) + 1
    out = dict(questions=len(data), dev=len(dev), test=len(test), test_types=types, not_answerable=sum(q["question_id"].endswith("_abs") for q in test),
               sessions_per_question=[min(per_q), percentile(per_q, .5), max(per_q)], unique_sessions=len(tokens),
               session_tokens=[percentile(list(tokens.values()), p) for p in (.1, .5, .9, .99)],
               sessions_over_fact_limit=sum(v > FACT_PROMPT_TOKENS for v in tokens.values()), unique_sessions_by_questions=sizes,
               turns_marked_as_answer=sum(bool(t.get("has_answer")) for q in dev for s in q["haystack_sessions"] for t in s))
    json.dump(out, open(f"{save}/lme_stats.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("stats", "pilot", "test"))
    ap.add_argument("--data", required=True); ap.add_argument("--url", default=""); ap.add_argument("--model", default=""); ap.add_argument("--save", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    data = json.load(open(a.data))
    numbers = session_numbers(data)
    if a.stage == "stats":
        stats(data, numbers, a.save)
        print("Done", flush=True)
        return

    sys.path.insert(0, os.getcwd())
    import vps_kit as K
    import engram_chat as E
    import engram_facts as F
    import engram_parallel as P
    from engram import Config
    D = json.load(open("frozen_design.json"))
    assert D["sentence_tokens"] == 800 and D["fact_tokens"] == 300
    cfg = Config(**json.load(open("final_config.json")))
    tools = P._load_tools()
    llm = K.LLM(a.url, a.model, "local", 1200)
    tmp = tempfile.mkdtemp()

    # The networks learn from LoCoMo only.
    net_path = f"{a.save}/networks.pkl"
    if os.path.exists(net_path):
        NET = pickle.load(open(net_path, "rb"))
    else:
        print("training networks on LoCoMo conversations 0 to 4", flush=True)
        examples = []
        for c in [0, 1, 2, 3, 4]:
            conv = K.load_conv(K.DATA[c])
            m = E.ChatMemory(f"{tmp}/train{c}.db", cfg=cfg, durable=False, search=P.ChannelSearch("plain", tools=tools))
            last = None
            for t in conv.turns:
                if t["session"] != last:
                    if last is not None: m.close_chat(write_note=False, sleep=False)
                    m.open_chat(t["date"]); last = t["session"]
                m.add(t["speaker"], t["text"], date=t["date"], ref=t["id"], caption=t["cap"], commit=False)
            m.db.commit(); m.close_chat(write_note=False)
            ref = dict(m.db.execute("SELECT ref, sid FROM sentences").fetchall())
            examples += [(m.search, q["q"], {ref[e] for e in q["ev"] if e in ref}) for q in conv.qs if q["cat"] != 5 and q["ev"]]
        NET = [P.train([x for x in examples if x[2]], s) for s in (1, 2, 3)]
        pickle.dump(NET, open(net_path, "wb"))

    fact_path = f"{a.save}/lme_facts.json"
    facts = json.load(open(fact_path)) if os.path.exists(fact_path) else {}

    def fact_prompt(key, session):
        lines = []
        for line_id, speaker, sentence, _, _ in session_lines(session["turns"], numbers[key]):
            lines.append(f"{line_id} {speaker}: {E.with_dates(sentence, session['date'])}")
        return fit_prompt(F.PROMPT.format(chat=f"[{session['date']}]\n" + "\n".join(lines)))

    def write_facts(questions):
        """Writes each session's facts once, however many questions hold it."""
        todo = {}
        for q in questions:
            for i, s, d in zip(q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]):
                key = session_key(i, s)
                if key not in facts: todo[key] = dict(turns=s, date=session_date(d))
        print(len(todo), "sessions to read for facts", flush=True)

        def one(item):
            key, session = item
            prompt, dropped = fact_prompt(key, session)
            t0 = time.time()
            return key, dict(raw=llm(prompt, max_tokens=cfg_reply)[0], secs=round(time.time() - t0, 1), dropped=dropped, prompt=hashlib.md5(prompt.encode()).hexdigest())
        t0, n = time.time(), 0
        with concurrent.futures.ThreadPoolExecutor(a.threads) as pool:
            for key, value in pool.map(one, todo.items()):
                facts[key] = value; n += 1
                if n % 50 == 0 or n == len(todo):
                    json.dump(facts, open(fact_path, "w"))
                    print(f"facts {n}/{len(todo)}, {(time.time() - t0) / n:.1f} s per session", flush=True)
        json.dump(facts, open(fact_path, "w"))
        return (time.time() - t0) / max(1, n)

    cfg_reply = E.Settings().fact_reply_tokens

    def replay(q, search, use_helper=False):
        state = {}

        def helper(prompt, max_tokens=0):
            fitted = fit_prompt(prompt)[0]
            saved = facts[state["key"]]
            assert saved["prompt"] == hashlib.md5(fitted.encode()).hexdigest(), "fact prompt changed"
            return saved["raw"], {}, saved["secs"]
        path = f"{tmp}/{hash_order(q['question_id'])[:12]}_{id(state)}.db"
        m = E.ChatMemory(path, cfg=cfg, durable=False, search=search, helper=helper if use_helper else None, helper_name=a.model if use_helper else "")
        where = {}
        total = len(q["haystack_sessions"])
        for n, (i, s, d) in enumerate(zip(q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]), 1):
            date = session_date(d)
            key = session_key(i, s)
            state["key"] = key
            m.open_chat(date)
            for line_id, speaker, sentence, turn, answer in session_lines(s, numbers[key]):
                where[line_id] = (i, turn, answer)
                m.add(speaker, sentence, date=date, ref=line_id, commit=False)
            m.db.commit()
            m.close_chat(write_note=False, sleep=n == total)  # sleeping once at the end, as in the earlier tests
        return m, where, path

    def found(m, where, sids, q):
        """How much of the evidence the search reached: answer sessions, and turns marked as holding the answer."""
        refs = {r[0] for s in sids for r in m.db.execute("SELECT ref FROM sentences WHERE sid=?", (s,))}
        hit = {where[r] for r in refs}
        wanted = set(q["answer_session_ids"])
        turns = {(w[0], w[1]) for w in where.values() if w[2]}
        got = {(h[0], h[1]) for h in hit if h[2]}
        return (len(wanted & {h[0] for h in hit}) / len(wanted) if wanted else None, len(got) / len(turns) if turns else None)

    def ask(q, m, where, name, out):
        m.use_facts = name == "with_facts"
        ctx, info = m.context_for(q["question"])
        asked = f"{q['question']} (Today is {session_date(q['question_date'])}.)"
        pred, _, secs = llm(K.ANSWER.format(ctx=ctx, q=asked), max_tokens=60)
        sessions, turns = found(m, where, info["sids"], q)
        out.write(json.dumps(dict(id=q["question_id"], type=q["question_type"], not_answerable=q["question_id"].endswith("_abs"), system=name, q=q["question"], gold=str(q["answer"]), pred=pred, answer_sessions=sessions, answer_turns=turns, tokens=info["tokens"], secs=secs)) + "\n")
        out.flush()

    def forget(*items):
        for m, path in items:
            m.db.close()
            for tail in ("", "-wal", "-shm"):
                if os.path.exists(path + tail): os.remove(path + tail)

    def part_one(questions, path):
        done = {(r["id"], r["system"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
        t0, n = time.time(), 0
        with open(path, "a") as out:
            for q in questions:
                if all((q["question_id"], s) in done for s in ("no_facts", "current")): continue
                net, where, p1 = replay(q, P.ChannelSearch("network", model=NET, tools=tools))
                cur, where2, p2 = replay(q, None)
                ask(q, net, where, "no_facts", out)
                ask(q, cur, where2, "current", out)
                forget((net, p1), (cur, p2))
                n += 1
                if n % 5 == 0 or n == len(questions): print(f"answers: {n} more questions, {(time.time() - t0) / n:.0f} s each", flush=True)
        return (time.time() - t0) / max(1, n)

    def part_two(questions, path):
        write_facts(questions)
        done = {(r["id"], r["system"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
        with open(path, "a") as out:
            for n, q in enumerate(questions, 1):
                if (q["question_id"], "with_facts") in done: continue
                net, where, p1 = replay(q, P.ChannelSearch("network", model=NET, tools=tools), use_helper=True)
                ask(q, net, where, "with_facts", out)
                forget((net, p1))
                if n % 5 == 0: print(f"with facts: question {n}/{len(questions)}", flush=True)

    dev, test = split_questions(data)
    if a.stage == "pilot":
        print("pilot on", len(dev), "development questions", flush=True)
        path = f"{a.save}/lme_pilot_answers.jsonl"
        each = part_one(dev, path)
        t0 = time.time()
        part_two(dev[:4], path)
        s = stats(data, numbers, a.save)
        rows = [r for r in map(json.loads, open(path))]
        read = list(facts.values())
        secs = [v["secs"] for v in read]
        summary = dict(seconds_per_question=round(each), hours_for_all_test_questions=round(each * len(test) / 3600, 1), fact_sessions=len(read), fact_seconds_each_one_at_a_time=round(sum(secs) / max(1, len(secs)), 1),
                       fact_sessions_cut=sum(v["dropped"] > 0 for v in read), facts_part_seconds=round(time.time() - t0), hours_for_facts={n: round(u * (time.time() - t0) / max(1, len(read)) / 3600, 1) for n, u in s["unique_sessions_by_questions"].items()},
                       answer_seconds=round(sum(r["secs"] for r in rows) / max(1, len(rows)), 1), tokens={x: round(sum(r["tokens"] for r in rows if r["system"] == x) / max(1, sum(r["system"] == x for r in rows))) for x in ("current", "no_facts", "with_facts")})
        json.dump(summary, open(f"{a.save}/lme_pilot.json", "w"), indent=1)
        print(json.dumps(summary, indent=1), flush=True)
    else:
        design = json.load(open("frozen_design_lme.json"))
        if not isinstance(design["facts_questions"], int): raise SystemExit("STOPPED: frozen_design_lme.json does not yet fix the number of questions with facts.")
        path = f"{a.save}/lme_answers.jsonl"
        print(len(test), "test questions", flush=True)
        part_one(test, path)
        print("part one finished", flush=True)
        part_two(test[:design["facts_questions"]], path)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
