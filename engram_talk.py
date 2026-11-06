"""Talk to Engram in a terminal, with the tested design: four-network search, the handover note, and noted facts written
when a chat closes. Gemma writes the replies through Ollama (or any OpenAI-style server).

    python3 engram_talk.py --model gemma3:4b

The memory and its log are kept in a private folder outside this repository (~/.engram_private by default), readable only
by you, so nothing you say can be synced to GitHub. Never put patient information in it: Engram has no encryption or
access control beyond your server account.

Commands while talking:
  /recall words   show what memory finds in past chats for these words, without calling the model
  /good  /bad     mark the last reply, for later measurement
  /close          close this chat: the model writes a handover note and noted facts (slow on a CPU)
  /bye            close the chat, then leave
  /quit           leave without closing; the chat continues next time
A chat left idle longer than --idle-hours is closed the next time you start.

What is tested and what is not: the search, notes and noted facts are the design measured on LoCoMo and LongMemEval. The reply
prompt below is new: the tests asked questions with short answers, not open conversation.
"""
import argparse, datetime, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REPLY = """You are {name}'s assistant, with a long-term memory of your past conversations. Below is what memory found for this message: a handover note from the last chat, noted facts, excerpts from past chats with the date each was said, and the current chat.
Use the memory when it helps. Never invent a memory: if {name} asks about something the memory does not hold, say you do not remember it. Reply naturally and briefly.

Memory:
{ctx}

{name}: {msg}
Assistant:"""


class Model:
    """One OpenAI-style chat endpoint. Returns (text, usage, seconds), as the memory expects."""
    def __init__(self, url, model, timeout=600):
        self.url, self.model, self.timeout = url.rstrip("/") + "/chat/completions", model, timeout

    def __call__(self, prompt, max_tokens=300, temperature=0.0):
        body = json.dumps(dict(model=self.model, messages=[dict(role="user", content=prompt)], max_tokens=max_tokens, temperature=temperature, seed=7)).encode()
        err = None
        for attempt in range(3):
            try:
                t0 = time.time()
                r = json.load(urllib.request.urlopen(urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"}), timeout=self.timeout))
                return r["choices"][0]["message"]["content"].strip(), r.get("usage", {}), time.time() - t0
            except Exception as e:
                err = e; time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"model call failed after 3 tries: {err}")


def said_on(now=None):
    """The date in the form Engram's date handling reads, such as 4:05 pm on 7 November, 2026."""
    now = now or datetime.datetime.now()
    return f"{now.hour % 12 or 12}:{now.minute:02d} {'am' if now.hour < 12 else 'pm'} on {now.day} {now.strftime('%B')}, {now.year}"


def private_folder(path):
    """Creates the folder readable only by its owner, and refuses one inside this repository."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.commonpath([path, HERE]) == HERE:
        raise SystemExit(f"Refusing to keep the memory inside the repository ({path}): it could be synced to GitHub. Choose a folder outside {HERE}.")
    os.makedirs(path, exist_ok=True); os.chmod(path, 0o700)
    return path


class Talk:
    """One person talking to one memory. Kept apart from the terminal so it can be tested."""
    def __init__(self, memory, model, name="Me", log_path=None):
        self.m, self.model, self.name, self.log_path = memory, model, name, log_path
        self.last = None

    def log(self, **row):
        if self.log_path:
            with open(self.log_path, "a") as f: f.write(json.dumps(dict(at=time.time(), **row)) + "\n")
            os.chmod(self.log_path, 0o600)

    def say(self, msg, now=None):
        date = said_on(now)
        if self.m.chat is None: self.m.open_chat(date)
        t0 = time.perf_counter()
        ctx, info = self.m.context_for(msg)  # what memory finds, before the new message joins the desk
        found_ms = (time.perf_counter() - t0) * 1000
        user_sid = self.m.add(self.name, msg, date=date)
        try:
            reply, _, secs = self.model(REPLY.format(name=self.name, ctx=ctx, msg=msg))
        except Exception as e:
            self.log(kind="turn", chat=self.m.chat, user_sid=user_sid, found=len(info.get("sids", [])), found_ms=round(found_ms, 1), error=str(e)[:200])
            return f"(The model did not answer: {e}. Your message is saved.)"
        reply_sid = self.m.add("Assistant", reply, date=said_on(now))
        self.last = dict(chat=self.m.chat, user_sid=user_sid, reply_sid=reply_sid)
        self.log(kind="turn", chat=self.m.chat, user_sid=user_sid, reply_sid=reply_sid, found=len(info.get("sids", [])), found_sids=list(info.get("sids", []))[:40],
                 found_ms=round(found_ms, 1), context_tokens=len(ctx.split()), model_secs=round(secs, 1))
        return reply

    def recall(self, words):
        sids, _ = self.m.find(words)
        return self.m.text_of(sorted(sids)) if sids else "(nothing in past chats for that; the open chat is searched once it closes)"

    def rate(self, good):
        if not self.last: return "Nothing to mark yet."
        self.log(kind="rating", good=bool(good), **self.last)
        return "Marked."

    def close(self):
        if self.m.chat is None: return "No open chat."
        chat = self.m.chat
        r = self.m.close_chat(write_note=True, sleep=True)
        facts = (r or {}).get("facts") or {}
        self.log(kind="close", chat=chat, note=bool((r or {}).get("note")), facts=facts.get("accepted"), facts_status=facts.get("status"))
        return f"Chat {chat} closed: note {'written' if (r or {}).get('note') else 'not written'}, noted facts: {facts.get('accepted', 0)} ({facts.get('status', 'none')})."


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:11434/v1", help="the model server; keep its port closed to the internet")
    p.add_argument("--model", default="gemma3:4b")
    p.add_argument("--name", default="Me", help="how you appear in the memory")
    p.add_argument("--folder", default="~/.engram_private", help="private folder for the memory and its log, outside this repository")
    p.add_argument("--design", default="tested", choices=("tested", "light"))
    p.add_argument("--no-facts", action="store_true", help="skip noted facts at chat close (faster closing; not the tested design)")
    p.add_argument("--idle-hours", type=float, default=6)
    a = p.parse_args()
    os.umask(0o077)  # the memory file and its journal are readable only by you
    folder = private_folder(a.folder)
    from engram_memory import open_memory
    model = Model(a.url, a.model)
    print("Loading memory...", flush=True)
    m = open_memory(os.path.join(folder, "memory.db"), design=a.design, llm=model, helper=None if a.no_facts or a.design == "light" else model, helper_name=a.model)
    talk = Talk(m, model, a.name, os.path.join(folder, "talk_log.jsonl"))
    if m.chat is not None:
        r = m.close_if_idle(a.idle_hours * 3600)
        print("Closed the last chat, idle too long." if r else "Continuing your open chat.")
    print("Ready. /recall, /good, /bad, /close, /bye, /quit. Never type patient information.\n")
    while True:
        try: msg = input(f"{a.name}: ").strip()
        except (EOFError, KeyboardInterrupt): print(); msg = "/quit"
        if not msg: continue
        if msg == "/quit": print("Bye. The chat stays open for next time."); break
        if msg == "/bye": print("Closing (the model writes a note and facts; this can take a few minutes on a CPU)..."); print(talk.close()); break
        if msg == "/close": print("Closing..."); print(talk.close()); continue
        if msg in ("/good", "/bad"): print(talk.rate(msg == "/good")); continue
        if msg.startswith("/recall"): print(talk.recall(msg[7:].strip() or "")); continue
        t0 = time.time()
        print(f"Assistant: {talk.say(msg)}  [{time.time() - t0:.0f} s]\n", flush=True)
    m.close()


if __name__ == "__main__":
    main()
