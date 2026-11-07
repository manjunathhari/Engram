"""P2.6 writer through the OpenAI API: the same prompts and forms as p26_gemma.py, with only the model changed.
Saves only the raw replies, in the same file format; scoring is done separately by p26_score.py.
Usage: p26_openai.py --model gpt-4o-2024-08-06 --save FOLDER [--sample FILE] [--form free|json] [--tag gpt4o]
  free: the line form, read by parse_writer_reply.
  json: the forced form. OpenAI's structured outputs hold the reply to writer_schema (strict), so the predicate can only come
        from the list and the subject only the speaker, as Ollama's "format" did for Gemma.
Needs OPENAI_API_KEY. Benchmark data only."""
import argparse, concurrent.futures, copy, json, os, sys, time
sys.path.insert(0, os.getcwd())
from engram_knowledge.extract import writer_prompt, writer_prompt_json, writer_schema

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gpt-4o-2024-08-06"); ap.add_argument("--save", required=True)
ap.add_argument("--sample", default="results/p26b/sample.jsonl"); ap.add_argument("--form", choices=("free", "json"), default="free"); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--tag", default="gpt4o", help="added to the file name")
ap.add_argument("--fake", action="store_true", help="dry run without the API: every reply is an empty list")
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.sample)]
name = os.path.basename(os.path.dirname(os.path.abspath(a.sample)))
path = f"{a.save}/{name}{'_' + a.tag if a.tag else ''}_{a.form}_replies.json"
os.makedirs(a.save, exist_ok=True)
done = json.load(open(path)) if os.path.exists(path) else {}


def strict(schema):
    """writer_schema as OpenAI's strict mode needs it: every object closed to extra fields. Nothing else changes."""
    s = copy.deepcopy(schema)
    def close(node):
        if isinstance(node, dict):
            if node.get("type") == "object": node["additionalProperties"] = False
            for v in node.values(): close(v)
        elif isinstance(node, list):
            for v in node: close(v)
    close(s)
    return s


if a.fake:
    client = None
else:
    from openai import OpenAI
    client = OpenAI()


def llm(prompt, max_tokens=400, schema=None):
    """One call at temperature 0, retried; the same settings as the Gemma runs (400 reply tokens, seed 7)."""
    if a.fake: return ('{"facts": []}' if schema else ""), "stop", 0.0
    for attempt in range(6):
        try:
            t0 = time.time()
            kw = dict(model=a.model, messages=[dict(role="user", content=prompt)], max_tokens=max_tokens, temperature=0.0, seed=7)
            if schema is not None: kw["response_format"] = dict(type="json_schema", json_schema=dict(name="writer_facts", strict=True, schema=strict(schema)))
            c = client.chat.completions.create(**kw).choices[0]
            return (c.message.content or "").strip(), c.finish_reason, time.time() - t0
        except Exception as e:
            err = e; time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"model call failed after 6 tries: {err}")


print("model:", a.model, "| form:", a.form, "| sample:", a.sample, "| first reply:", repr(llm("What is 2 + 2? Reply with just the number.", max_tokens=10)[0]), flush=True)


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
