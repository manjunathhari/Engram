"""Checks the optional facts layer with a stand-in helper. No model and no personal data."""
import hashlib, json, os, sqlite3, sys, tempfile
sys.path.insert(0, os.environ.get("ENGRAM", "."))
import engram_chat as E
from engram import Config
tmp = tempfile.mkdtemp()
CHAT = [("Caroline", "I joined a group called Connected LGBTQ Activists last Tuesday.", "D10:1"), ("Melanie", "How nice! I took my 2 kids to the beach in Florida yesterday.", "D10:2"), ("Caroline", "Thanks, talk soon!", "D10:3")]
DATE = "1:56 pm on 8 May, 2023"
GOOD = "CONTEXT: Two friends catch up about activism and a beach trip.\nD10:3 | D10:1 Caroline joined an activist group called Connected LGBTQ Activists.\n10:37 am on 8 May, 2023 | D10:2 Melanie took her 2 kids to the beach in Florida on [7 May 2023].\nD10:1 | Caroline owns a spaceship factory on Jupiter.\nHere are the facts:\n"
def helper(reply=GOOD):
    def h(prompt, max_tokens=0): return reply, {}, .01
    return h
def fill(m):
    m.open_chat(DATE)
    for who, text, ref in CHAT: m.add(who, text, date=DATE, ref=ref)
def record(m): return hashlib.md5(json.dumps(m.db.execute("SELECT sid,chat,pos,ref,speaker,date,text,caption,tokens FROM sentences ORDER BY sid").fetchall()).encode()).hexdigest()
n = 0
def ok(c, name):
    global n
    assert c, "FAILED: " + name
    n += 1; print("ok", name)

a = E.ChatMemory(f"{tmp}/off.db", durable=False); fill(a); a.close_chat(write_note=False)
ok(a.db.execute("SELECT count(*) FROM facts").fetchone()[0] == 0 and "Noted fact" not in a.context_for("beach")[0], "off by default: no facts written or shown")
b = E.ChatMemory(f"{tmp}/on.db", durable=False, helper=helper(), helper_name="stand-in"); fill(b); before = record(b); r = b.close_chat(write_note=False)
ok(record(b) == before, "helper does not change the sentences")
ok(r["facts"]["accepted"] == 2 and r["facts"]["rejected"] == 1, "two facts accepted, invented fact rejected")
rows = b.db.execute("SELECT f.text, s.ref FROM facts f JOIN fact_sources x ON x.fact=f.fact JOIN sentences s ON s.sid=x.sid ORDER BY f.fact").fetchall()
ok(rows[0][1] == "D10:1" and rows[1][1] == "D10:2", "facts linked to the sentences that support them")
f = b.db.execute("SELECT person,time,number FROM facts WHERE text LIKE '%beach%'").fetchone()
ok("Melanie" in f[0] and "7 May 2023" in f[1] and "2" in f[2], "person, time and figure stored")
ok("beach trip" in b.episode_context(1), "episode context stored")
ok(b.find_facts("beach")[0]["sources"], "fact search returns its sources")
b.use_facts = True
ctx = b.context_for("Where did Melanie go to the beach?")[0]
ok("Noted fact" in ctx and "the beach in Florida yesterday" in ctx, "with facts on, exact sentences still come too")
b.use_facts = False
ok("Noted fact" not in b.context_for("beach")[0], "facts can be switched off")
count = b.db.execute("SELECT count(*) FROM facts").fetchone()[0]; b.write_facts(1)
ok(b.db.execute("SELECT count(*) FROM facts").fetchone()[0] == count and b.db.execute("SELECT count(*) FROM facts_index").fetchone()[0] == count, "running again does not duplicate facts")
b.close()
c = E.ChatMemory(f"{tmp}/on.db", durable=False)
ok(c.db.execute("SELECT count(*) FROM facts").fetchone()[0] == count, "facts survive a restart")
sid = c.db.execute("SELECT sid FROM sentences WHERE ref='D10:2'").fetchone()[0]
out = c.forget_sentence(sid)
ok(out == dict(sentences=1, facts=1) and not c.find_facts("beach") and c.db.execute("SELECT count(*) FROM fact_sources WHERE sid=?", (sid,)).fetchone()[0] == 0, "deleting a sentence removes its facts")
ok(sid not in c.find("beach Florida")[0], "deleted sentence is no longer found")
c.close()
def boom(prompt, max_tokens=0): raise RuntimeError("model down")
d = E.ChatMemory(f"{tmp}/fail.db", durable=False, helper=boom, helper_name="stand-in"); fill(d); r = d.close_chat(write_note=False)
ok(r["facts"]["status"] == "failed" and d.db.execute("SELECT count(*) FROM sentences").fetchone()[0] == 3 and d.db.execute("SELECT closed FROM chats").fetchone()[0] == 1, "helper failure keeps the chat and closes it")
d.helper = helper(); ok(d.write_facts(1)["accepted"] == 2, "failed chat can be retried")
e = E.ChatMemory(f"{tmp}/junk.db", durable=False, helper=helper("sorry, I cannot help"), helper_name="stand-in"); fill(e); r = e.close_chat(write_note=False)
ok(r["facts"]["status"] == "empty" and "sorry" in e.db.execute("SELECT raw FROM episodes").fetchone()[0], "unusable reply is kept for review and adds no facts")
old = sqlite3.connect(f"{tmp}/old.db")
old.executescript("""CREATE TABLE chats(chat INTEGER PRIMARY KEY, opened_on TEXT, last_active REAL, closed INTEGER DEFAULT 0, note TEXT);
CREATE TABLE sentences(sid INTEGER PRIMARY KEY, chat INTEGER, pos INTEGER, ref TEXT, speaker TEXT, date TEXT, text TEXT, caption TEXT, tokens INTEGER);
CREATE VIRTUAL TABLE words_index USING fts5(words, tokenize="unicode61 remove_diacritics 0");
INSERT INTO chats VALUES(1,'1 May 2023',0,1,NULL); INSERT INTO sentences VALUES(1,1,0,'','A','1 May 2023','My dog is called Rex.','',6); INSERT INTO words_index(rowid,words) VALUES(1,'dog called rex');""")
old.commit(); old.close()
g = E.ChatMemory(f"{tmp}/old.db", durable=False)
ok(g.db.execute("SELECT count(*) FROM sentences").fetchone()[0] == 1 and g.db.execute("SELECT count(*) FROM facts").fetchone()[0] == 0 and g.find("dog")[0], "an older database opens and upgrades")
print("all", n, "checks passed")
