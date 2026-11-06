"""Asks a model to write facts for the P2.6 sample sentences. Saves only the raw replies; scoring is done separately.
Usage: p26_gemma.py --url URL --model NAME --save FOLDER. Benchmark data only."""
import argparse, concurrent.futures, json, os, sys, time, urllib.request
sys.path.insert(0, os.getcwd())
from engram_knowledge.extract import writer_prompt

ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--save", required=True); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
rows = [json.loads(l) for l in open("results/p26/sample.jsonl")]
path = f"{a.save}/p26_replies.json"
done = json.load(open(path)) if os.path.exists(path) else {}
def llm(prompt, max_tokens=200):
    """One call to an OpenAI-style endpoint such as Ollama's, at temperature 0, retried twice."""
    body = json.dumps(dict(model=a.model, messages=[dict(role="user", content=prompt)], max_tokens=max_tokens, temperature=0.0, seed=7)).encode()
    for attempt in range(3):
        try:
            t0 = time.time()
            req = urllib.request.Request(a.url.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=600))
            c = r["choices"][0]
            return c["message"]["content"].strip(), c.get("finish_reason"), time.time() - t0
        except Exception as e:
            err = e; time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"model call failed after 3 tries: {err}")
print("first reply:", repr(llm("What is 2 + 2? Reply with just the number.", max_tokens=10)[0]), flush=True)


def one(r):
    reply, finish, secs = llm(writer_prompt(r["speaker"], r["text"], r["message"]), max_tokens=400)
    return r["id"], dict(raw=reply, finish=finish, secs=round(secs, 2))


todo = [r for r in rows if r["id"] not in done]
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(a.threads) as pool:
    for n, (i, v) in enumerate(pool.map(one, todo), 1):
        done[i] = v
        if n % 25 == 0 or n == len(todo):
            json.dump(done, open(path, "w"), indent=0)
            print(f"{n}/{len(todo)} sentences, {(time.time() - t0) / n:.1f} s each", flush=True)
json.dump(done, open(path, "w"), indent=0)
print("replies cut off at the length limit:", sum(v.get("finish") == "length" for v in done.values()))
print("Done", flush=True)
