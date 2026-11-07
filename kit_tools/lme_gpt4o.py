"""Engram on LongMemEval-S with GPT-4o answering and the official LongMemEval GPT-4o judge.

Same answering model and judge as the published Supermemory, Zep and full-context figures, so the score can sit beside theirs.
Search is the tested design from main (open_memory, saved networks trained on LoCoMo only); nothing is tuned on this data.

  python kit_tools/lme_gpt4o.py --data longmemeval_s_cleaned.json --save results_dir --n 100

--n 100 is the fast run (first 100 test questions in hash order, about 20 minutes); --n 0 runs every question.
Work resumes where it stopped. Needs OPENAI_API_KEY.
"""
import argparse, concurrent.futures, datetime, hashlib, json, math, os, random, re, sys, tempfile, time

ANSWER_MODEL = "gpt-4o-2024-08-06"
JUDGE_MODEL = "gpt-4o-2024-08-06"  # the official evaluate_qa.py maps "gpt-4o" to this snapshot
DEV_QUESTIONS = 30  # the same 30 development questions as the Gemma test; never counted
LONGEST_SENTENCE = 600

# ---------- data adapter, identical to kit_tools/lme_test.py at d25a8cd ----------

def split_sentences(text):
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
    d = datetime.datetime.strptime(re.sub(r"\s*\(\w+\)\s*", " ", text.strip()), "%Y/%m/%d %H:%M")
    return f"{d.hour % 12 or 12}:{d.minute:02d} {'am' if d.hour < 12 else 'pm'} on {d.day} {d.strftime('%B')}, {d.year}"


def hash_order(question_id):
    return hashlib.sha256(question_id.encode()).hexdigest()


def split_questions(data, dev=DEV_QUESTIONS):
    ordered = sorted(data, key=lambda q: hash_order(q["question_id"]))
    return ordered[:dev], ordered[dev:]


def session_key(session_id, session):
    return session_id + ":" + hashlib.md5(json.dumps(session, sort_keys=True).encode()).hexdigest()[:8]


def session_numbers(data):
    keys = sorted({session_key(i, s) for q in data for i, s in zip(q["haystack_session_ids"], q["haystack_sessions"])})
    return {k: n for n, k in enumerate(keys)}


def session_lines(session, number):
    lines, n = [], 0
    for t, turn in enumerate(session):
        for sentence in split_sentences(turn["content"]):
            n += 1
            lines.append((f"D{number}:{n}", turn["role"].capitalize(), sentence, t, bool(turn.get("has_answer"))))
    return lines

# ---------- official LongMemEval judge prompts, verbatim from src/evaluation/evaluate_qa.py ----------

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt

# ---------- OpenAI calls ----------

def make_chat(fake=False):
    if fake:  # dry run without a key: answers "not mentioned", judge says no
        return lambda model, prompt, max_tokens: "not mentioned" if max_tokens > 10 else "no"
    from openai import OpenAI
    client = OpenAI()

    def chat(model, prompt, max_tokens):
        for attempt in range(6):
            try:
                r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], n=1, temperature=0, max_tokens=max_tokens, seed=7)
                return r.choices[0].message.content.strip()
            except Exception as e:
                err = e
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"OpenAI call failed after 6 tries: {err}")
    return chat


def bootstrap(values, reps=10000, seed=7):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return round(100 * means[int(.025 * reps)], 1), round(100 * means[int(.975 * reps) - 1], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--n", type=int, default=100, help="test questions to run; 0 = all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fake", action="store_true", help="dry run without OpenAI")
    a = ap.parse_args()
    os.makedirs(a.save, exist_ok=True)

    sys.path.insert(0, os.getcwd())
    import vps_kit as K
    import engram_chat as E
    from engram_memory import open_memory

    data = json.load(open(a.data))
    numbers = session_numbers(data)
    _, test = split_questions(data)
    questions = test[:a.n] if a.n else test
    chat = make_chat(a.fake)
    tmp = tempfile.mkdtemp()
    out_path = f"{a.save}/lme_gpt4o_answers.jsonl"
    done = {r["question_id"] for r in map(json.loads, open(out_path))} if os.path.exists(out_path) else set()
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"{len(questions)} questions, {len(todo)} still to run; answer model {ANSWER_MODEL}, judge {JUDGE_MODEL}", flush=True)

    def retrieve(q):
        """Replays the question's history into a fresh memory, closing each session as a chat, then searches."""
        path = f"{tmp}/{hash_order(q['question_id'])[:12]}.db"
        m = open_memory(path, durable=False)
        where, total = {}, len(q["haystack_sessions"])
        for n, (i, s, d) in enumerate(zip(q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]), 1):
            date = session_date(d)
            m.open_chat(date)
            for line_id, speaker, sentence, turn, answer in session_lines(s, numbers[session_key(i, s)]):
                where[line_id] = (i, turn, answer)
                m.add(speaker, sentence, date=date, ref=line_id, commit=False)
            m.db.commit()
            m.close_chat(write_note=False, sleep=n == total)
        ctx, info = m.context_for(q["question"])
        refs = {r[0] for sid in info["sids"] for r in m.db.execute("SELECT ref FROM sentences WHERE sid=?", (sid,))}
        hit = {where[r][0] for r in refs if r in where}
        wanted = set(q["answer_session_ids"])
        reached = len(wanted & hit) / len(wanted) if wanted else None
        m.db.close()
        for tail in ("", "-wal", "-shm"):
            if os.path.exists(path + tail): os.remove(path + tail)
        return ctx, info["tokens"], reached

    def answer_and_judge(q, ctx, tokens, reached, secs):
        asked = f"{q['question']} (Today is {session_date(q['question_date'])}.)"
        hyp = chat(ANSWER_MODEL, K.ANSWER.format(ctx=ctx, q=asked), 60)
        prompt = get_anscheck_prompt(q["question_type"], q["question"], q["answer"], hyp, abstention="_abs" in q["question_id"])
        verdict = chat(JUDGE_MODEL, prompt, 10)
        return dict(question_id=q["question_id"], question_type=q["question_type"], abstention="_abs" in q["question_id"], question=q["question"], answer=str(q["answer"]),
                    hypothesis=hyp, judge_reply=verdict, label="yes" in verdict.lower(), context_tokens=tokens, answer_sessions_reached=reached, retrieval_secs=round(secs, 2))

    t0 = time.time()
    with open(out_path, "a") as out, concurrent.futures.ThreadPoolExecutor(a.threads) as pool:
        futures = []
        for n, q in enumerate(todo, 1):
            s0 = time.time()
            ctx, tokens, reached = retrieve(q)  # search runs here, one question at a time; the API calls run in parallel
            futures.append(pool.submit(answer_and_judge, q, ctx, tokens, reached, time.time() - s0))
            if n % 10 == 0: print(f"searched {n}/{len(todo)}, {(time.time() - t0) / n:.1f} s each", flush=True)
            for f in [f for f in futures if f.done()]:
                out.write(json.dumps(f.result()) + "\n"); out.flush(); futures.remove(f)
        for f in futures:
            out.write(json.dumps(f.result()) + "\n"); out.flush()

    rows = [r for r in map(json.loads, open(out_path)) if r["question_id"] in {q["question_id"] for q in questions}]
    labels = [int(r["label"]) for r in rows]
    by_type = {}
    for r in rows: by_type.setdefault(r["question_type"], []).append(int(r["label"]))
    reached = [r["answer_sessions_reached"] for r in rows if r["answer_sessions_reached"] is not None]
    summary = dict(
        questions=len(rows), answer_model=ANSWER_MODEL, judge=JUDGE_MODEL, search="Engram tested design (open_memory, main)",
        accuracy=round(100 * sum(labels) / max(1, len(labels)), 1), interval_95=bootstrap(labels) if labels else None,
        by_type={k: dict(n=len(v), accuracy=round(100 * sum(v) / len(v), 1)) for k, v in sorted(by_type.items())},
        not_answerable=dict(n=sum(r["abstention"] for r in rows), right=sum(r["label"] for r in rows if r["abstention"])),
        all_answer_sessions_reached=round(100 * sum(x == 1 for x in reached) / max(1, len(reached)), 1),
        mean_context_tokens=round(sum(r["context_tokens"] for r in rows) / max(1, len(rows))),
        published_for_comparison="Supermemory report, LongMemEval-S, GPT-4o judge: Zep 71.2, full context 60.2; Supermemory 84.6 (GPT-5 answers), 85.2 (Gemini 3 Pro). Their 97 (GPT-4o row) equals their Recall@20.",
        caveats="Fast run is a sample of the 470 test questions, so its interval is wide; the published figures use all 500. Errors are not dropped: every question is in the score.")
    json.dump(summary, open(f"{a.save}/lme_gpt4o_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("Done", flush=True)


if __name__ == "__main__":
    main()
