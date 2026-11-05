"""Checks Knowledge State v0.1 on hand-written cases. No model and no personal data."""
import json, os, sys, tempfile
sys.path.insert(0, os.environ.get("ENGRAM", "."))
import engram_chat as E
import engram_knowledge as K
from engram import Config
tmp = tempfile.mkdtemp()
n = 0
def ok(c, name):
    global n
    assert c, "FAILED: " + name
    n += 1; print("ok", name)
LINES = [
 ("1 March 2023", "Melanie", "I adopted a dog called Oliver last week.", "a1"),
 ("1 March 2023", "Caroline", "Mel told me Oliver is sweet.", "a2"),
 ("5 April 2023", "Melanie", "I moved to Boston on 2 April 2023.", "b1"),
 ("9 June 2023", "Melanie", "I moved to Denver in June 2023.", "c1"),
 ("20 June 2023", "Melanie", "Before Boston I lived in Austin from January 2020.", "c2"),
 ("1 July 2023", "Melanie", "I started work at Acme as a nurse on 1 July 2023.", "d1"),
 ("3 August 2023", "Melanie", "Sadly I gave away Oliver to my sister on 1 August 2023.", "e1"),
 ("9 August 2023", "Melanie", "Correction: I never worked at Acme, it was Apex.", "f1"),
 ("9 August 2023", "Melanie", "I work at Apex.", "f2"),
 ("10 August 2023", "Caroline", "Melanie is also known as Mel.", "g1"),
 ("11 August 2023", "Caroline", "Some people say she likes jazz.", "h1"),
 ("12 August 2023", "Melanie", "I love painting landscapes.", "i1"),
 ("13 August 2023", "Sam", "I think Mel might be a Boston fan.", "j1"),
]
m = E.ChatMemory(f"{tmp}/k.db", cfg=Config(), durable=False)
sid = {}
last = None
for date, who, text, ref in LINES:
    if date != last:
        if last: m.close_chat(write_note=False, sleep=False)
        m.open_chat(date); last = date
    sid[ref] = m.add(who, text, date=date, ref=ref)
m.close_chat(write_note=False)
before = m.db.execute("SELECT sid, text FROM sentences ORDER BY sid").fetchall()
k = K.KnowledgeState(m.db)

# An event produces a relationship; its date is metadata.
r = k.learn(dict(subject="Melanie", event="adopted", object="Oliver", time="2023-02-22", sources=[sid["a1"]]))
ok(r["accepted"] and r["kind"] == "relationship" and r["event"], "adoption stored as an event and an ownership")
s = k.get_state("Melanie")
ok(s["state"]["owns"][0]["value"] == "oliver" and s["state"]["owns"][0]["since"] == "2023-02-22" and s["state"]["owns"][0]["time_status"] == "explicit", "ownership holds the adoption date")
ok(m.db.execute("SELECT count(*) FROM events WHERE type='adopted'").fetchone()[0] == 1 and m.db.execute("SELECT count(*) FROM relationships WHERE predicate='owns'").fetchone()[0] == 1, "event and relationship are separate records")

# Rejections
for name, o, why in [
 ("no source", dict(subject="Melanie", predicate="hobby", value="painting"), "no source"),
 ("invented source", dict(subject="Melanie", predicate="hobby", value="painting", sources=[99999]), "does not exist"),
 ("pronoun subject", dict(subject="she", predicate="hobby", value="painting", sources=[sid["i1"]]), "does not resolve"),
 ("predicate not allowed", dict(subject="Melanie", predicate="vibes_with", value="painting", sources=[sid["i1"]]), "not allowed"),
 ("event not allowed", dict(subject="Melanie", event="vibed", object="painting", sources=[sid["i1"]]), "not allowed"),
 ("subject not in source", dict(subject="Bob", predicate="hobby", value="painting", sources=[sid["i1"]]), "subject not found"),
 ("object not in source", dict(subject="Melanie", predicate="hobby", value="fencing", sources=[sid["i1"]]), "object not found"),
 ("vague object", dict(subject="Melanie", predicate="friend_of", object="my friend", sources=[sid["i1"]]), "object does not resolve"),
 ("bad status", dict(subject="Melanie", predicate="hobby", value="painting", status="verified", sources=[sid["i1"]]), "status must be"),
]:
    r = k.learn(o)
    ok(not r["accepted"] and why in r["reason"], "rejected: " + name)
ok(k.resolve("Bob") is None, "a rejected observation creates nothing")

# Moves, in the wrong order, give the right history.
k.learn(dict(subject="Melanie", event="moved_to", object="Denver", time="June 2023", sources=[sid["c1"]]))
k.learn(dict(subject="Melanie", event="moved_to", object="Boston", time="2 April 2023", sources=[sid["b1"]]))
k.learn(dict(subject="Melanie", predicate="lives_in", object="Austin", time="January 2020", sources=[sid["c2"]]))
ok([e["value"] for e in k.get_state("Melanie")["state"]["lives_in"]] == ["denver"], "current home is the latest, whatever the learning order")
ok(k.get_state("Melanie", as_of="2021-05-01")["state"]["lives_in"][0]["value"] == "austin", "as_of reads 2021 correctly")
ok(k.get_state("Melanie", as_of="2023-05-01")["state"]["lives_in"][0]["value"] == "boston", "as_of reads May 2023 correctly")
ok("lives_in" not in k.get_state("Melanie", as_of="2019-01-01")["state"], "nothing known before the first date")
ch = [(c["old"], c["new"]) for c in k.explain_state("Melanie")["changes"] if c["predicate"] == "lives_in"]
ok(ch == [(None, "austin"), ("austin", "boston"), ("boston", "denver")], "changes listed in true order")

# Ending and correcting
r = k.learn(dict(subject="Melanie", event="gave_away", object="Oliver", time="1 August 2023", sources=[sid["e1"]]))
ok(r.get("ended") and "owns" not in k.get_state("Melanie")["state"], "giving away ends ownership")
ok(k.get_state("Melanie", as_of="2023-06-01")["state"]["owns"][0]["value"] == "oliver", "past ownership is still known")
r = k.learn(dict(subject="Melanie", event="sold", object="Boston", time="1 August 2023", sources=[sid["c2"]]))
ok(r["accepted"] and r.get("note") == "no earlier state to end", "ending something never held changes no state")
j = k.learn(dict(subject="Melanie", event="started_job", object="Acme", time="1 July 2023", sources=[sid["d1"]]))
k.learn(dict(subject="Melanie", predicate="job_title", value="nurse", time="1 July 2023", sources=[sid["d1"]]))
ok(not k.retract("relationship", j["item"], "no reason", [])["accepted"], "a correction needs a source")
ok(k.retract("relationship", j["item"], "She said she never worked there", [sid["f1"]])["accepted"], "correction accepted")
ok("works_at" not in k.get_state("Melanie")["state"], "corrected item leaves the state")
ex = k.explain_state("Melanie")
ok(any(c["value"] == "acme" and any("Correction" in q["text"] for q in c["evidence"]) for c in ex["corrections"]), "correction stays visible with its evidence")
k.learn(dict(subject="Melanie", predicate="works_at", object="Apex", sources=[sid["f2"]]))
w = k.get_state("Melanie")["state"]["works_at"][0]
ok(w["value"] == "apex" and w["time_status"] == "said" and w["since"] == "2023-08-09", "no date given: date said is used and labelled")
ok(any("when it was said" in u for u in k.explain_state("Melanie")["uncertainties"]), "that uncertainty is reported")

# Names
ok(k.resolve("Mel") is None, "a similar name is not merged by itself")
r = k.learn(dict(subject="Melanie", predicate="also_known_as", object="Mel", sources=[sid["g1"]]))
ok(r["accepted"] and k.resolve("mel") == k.resolve("Melanie"), "a stated alias merges")
k.learn(dict(subject="Mel", predicate="friend_of", object="Caroline", sources=[sid["a2"]]))
ok(k.get_state("Melanie")["state"]["friend_of"][0]["value"] == "caroline", "alias reaches the same entity")
k.learn(dict(subject="Sam", predicate="also_known_as", object="Sam", sources=[sid["j1"]]))
ok(not k.learn(dict(subject="Sam", predicate="also_known_as", object="Mel", sources=[sid["j1"]]))["accepted"], "an alias cannot take a name from another entity")

# Inference
own = k.learn(dict(subject="Melanie", predicate="hobby", value="painting", sources=[sid["i1"]]))
ok(not k.learn(dict(subject="Melanie", predicate="likes", value="art", status="inferred", sources=[sid["i1"]]))["accepted"], "an inference must name its basis")
inf = k.learn(dict(subject="Melanie", predicate="likes", value="art", status="inferred", basis=[("fact", own["item"])]))
s = k.get_state("Melanie")
ok(inf["accepted"] and "likes" not in s["state"] and s["inferred"][0]["value"] == "art" and s["inferred"][0]["status"] == "inferred", "an inference stays out of the stated state")
ok(any("inferred, not stated" in u for u in k.explain_state("Melanie")["uncertainties"]), "an inference is labelled as uncertain")
ok(k.explain_state("Melanie")["inferred"][0]["evidence"][0]["sid"] == sid["i1"], "an inference carries its evidence")

# Evidence
ex = k.explain_state("Melanie")
ok(ex["state"]["lives_in"][0]["evidence"][0]["text"] == "I moved to Denver in June 2023." and ex["checked"] == "structure only", "explanation quotes the exact sentence and admits its limit")
k.learn(dict(subject="Melanie", predicate="hobby", value="painting", sources=[sid["i1"]]))
ok(m.db.execute("SELECT count(*) FROM knowledge_facts WHERE predicate='hobby'").fetchone()[0] == 1, "learning the same thing twice does not duplicate it")
ok(k.get_state("Nobody")["known"] is False and k.explain_state("Nobody")["uncertainties"], "unknown entity is reported as unknown")

# No conversation text is copied
tables = ["entities", "entity_aliases", "relationships", "knowledge_facts", "events", "fact_evidence", "state_transitions"]
stored = json.dumps([m.db.execute(f"SELECT * FROM {t}").fetchall() for t in tables])
ok(not any(text in stored for _, text in before), "no sentence is copied into the knowledge tables")
ok(m.db.execute("SELECT sid, text FROM sentences ORDER BY sid").fetchall() == before, "the sentences are unchanged")

# Restart
m.close()
m = E.ChatMemory(f"{tmp}/k.db", cfg=Config(), durable=False)
k = K.KnowledgeState(m.db)
ok(k.get_state("Melanie")["state"]["lives_in"][0]["value"] == "denver", "state survives a restart")

# Deletion
out = m.forget_sentence(sid["c1"])
ok(k.get_state("Melanie")["state"]["lives_in"][0]["value"] == "boston" and k.get_state("Denver")["known"] is False, "deleting a sentence removes what rests on it and restores the earlier state")
ok(("boston", "denver") not in [(c["old"], c["new"]) for c in k.explain_state("Melanie")["changes"]], "the change is gone from history")
m.forget_sentence(sid["i1"])
s = k.get_state("Melanie")
ok("hobby" not in s["state"] and not s["inferred"], "deleting a sentence removes the inference built on it")
m.forget_sentence(sid["e1"])
ok(k.get_state("Melanie")["state"]["owns"][0]["value"] == "oliver", "deleting an ending reopens the state")
m.forget_sentence(sid["g1"])
ok(k.resolve("Mel") is None, "deleting the sentence that stated an alias removes the alias")
m.forget_sentence(sid["f1"])
s = k.get_state("Melanie")
ok(any(e["value"] == "acme" for e in s["state"].get("works_at", []) + [x for x in s["time_unknown"]]) or any(c["new"] == "acme" for c in k.explain_state("Melanie")["changes"]), "deleting a correction brings the item back")
ok(not k.explain_state("Melanie")["corrections"], "the deleted correction and its reason are gone")
ok(m.db.execute("SELECT count(*) FROM fact_evidence WHERE sid=?", (sid["c1"],)).fetchone()[0] == 0, "no evidence points to a deleted sentence")
stored = m.db.execute("SELECT count(*) FROM relationships WHERE predicate='lives_in' AND object NOT IN (SELECT entity FROM entities)").fetchone()[0]
ok(stored == 0, "no item points to a missing entity")
print("all", n, "checks passed")
