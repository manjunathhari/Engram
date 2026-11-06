"""Asks a model to write facts for P2.6 sentences. Saves only the raw replies; scoring is done separately.
Usage: p26_gemma.py --url URL --model NAME --save FOLDER [--sample FILE] [--form free|json]
  free: the line form (subject | predicate | object | details | time), read by parse_writer_reply.
  json: the forced form; Ollama holds the reply to writer_schema, so the predicate can only come from the list.
Benchmark data only."""
import argparse, concurrent.futures, json, os, sys, time, urllib.request
sys.path.insert(0, os.getcwd())
from engram_knowledge.extract import writer_prompt, writer_prompt_json, writer_schema

ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--save", required=True)
ap.add_argument("--sample", default="results/p26/sample.jsonl"); ap.add_argument("--form", choices=("free", "json"), default="free"); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.sample)]
name = os.path.basename(os.path.dirname(os.path.abspath(a.sample)))
path = f"{a.save}/{name}_{a.form}_replies.json"
done = json.load(open(path)) if os.path.exists(path) else {}
base = a.url.rstrip("/").removesuffix("/v1")


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def llm(prompt, max_tokens=400, schema=None):
    """One call at temperature 0, retried twice. Free form uses the OpenAI-style endpoint; the forced form uses Ollama's own,
    which takes the JSON schema as "format"."""
    for attempt in range(3):
        try:
            t0 = time.time()
            if schema is None:
                c = post(base + "/v1/chat/completions", dict(model=a.model, messages=[dict(role="user", content=prompt)], max_tokens=max_tokens, temperature=0.0, seed=7))["choices"][0]
                return c["message"]["content"].strip(), c.get("finish_reason"), time.time() - t0
            r = post(base + "/api/chat", dict(model=a.model, messages=[dict(role="user", content=prompt)], stream=False, format=schema,
                                              options=dict(temperature=0.0, seed=7, num_predict=max_tokens)))
            return r["message"]["content"].strip(), r.get("done_reason"), time.time() - t0
        except Exception as e:
            err = e; time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"model call failed after 3 tries: {err}")


print("form:", a.form, "| sample:", a.sample, "| first reply:", repr(llm("What is 2 + 2? Reply with just the number.", max_tokens=10)[0]), flush=True)


def one(r):
    if a.form == "free": reply, finish, secs = llm(writer_prompt(r["speaker"], r["text"], r["message"]))
    else: reply, finish, secs = llm(writer_prompt_json(r["speaker"], r["text"], r["message"]), schema=writer_schema(r["speaker"]))
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
print("saved", path)
print("Done", flush=True)
