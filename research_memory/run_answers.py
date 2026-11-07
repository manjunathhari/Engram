"""Asks Gemma every prompt in prompts.jsonl and saves the replies. Usage:
    python3 run_answers.py --url URL --model NAME --save FOLDER
Temperature 0, seed 7, 200 reply tokens, as fixed in frozen_rule.json. Resumes where it stopped."""
import argparse, concurrent.futures, json, os, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--save", required=True); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
rows = [json.loads(l) for l in open(os.path.join(HERE, "prompts.jsonl"))]
path = os.path.join(a.save, "research_answers.jsonl")
done = {(r["id"], r["way"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()


def llm(prompt, max_tokens):
    body = json.dumps(dict(model=a.model, messages=[dict(role="user", content=prompt)], max_tokens=max_tokens, temperature=0.0, seed=7)).encode()
    err = None
    for attempt in range(3):
        try:
            t0 = time.time()
            r = json.load(urllib.request.urlopen(urllib.request.Request(a.url.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=900))
            c = r["choices"][0]
            return c["message"]["content"].strip(), c.get("finish_reason"), time.time() - t0
        except Exception as e:
            err = e; time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"model call failed after 3 tries: {err}")


print("first reply:", repr(llm("What is 2 + 2? Reply with just the number.", 5)[0]), flush=True)
todo = [r for r in rows if (r["id"], r["way"]) not in done]


def one(r):
    pred, finish, secs = llm(r["prompt"], 200)
    return dict(id=r["id"], kind=r["kind"], way=r["way"], pred=pred, finish=finish, secs=round(secs, 1), tokens=r["tokens"])


with open(path, "a") as out, concurrent.futures.ThreadPoolExecutor(a.threads) as pool:
    for n, row in enumerate(pool.map(one, todo), 1):
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
        if n % 25 == 0 or n == len(todo): print(f"{n}/{len(todo)} answers", flush=True)
print("Done", flush=True)
