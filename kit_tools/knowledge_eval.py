"""Measures the state engine and knowledge retrieval on made-up people, with real LoCoMo chat mixed in as noise. No language model.
The facts are supplied as if a perfect writer had read each sentence, so this does not measure how well a model would write them."""
import json, os, random, statistics, sys, tempfile, time
sys.path.insert(0, os.environ.get("ENGRAM", "."))
import engram_chat as E
import vps_kit as K
from engram import Config
from engram_knowledge import ACCEPTED, UNCERTAIN, Candidate, KnowledgePipeline, KnowledgeRetriever, StateEngine
from engram_knowledge.normalizer import PREDICATE_TYPES
from engram_records import Entity, FactEvidence, FactWithEvidence, KnowledgeBase, KnowledgeFact

PEOPLE = "Asha Ravi Meera Karan Nisha Arjun Divya Suresh Priya Vikram Lakshmi Rohan Anita Sanjay Kavya Mohan Deepa Naveen Rekha Tarun".split()
CITIES = "Paris Rome Tokyo Madrid Oslo Lima Cairo Seoul Vienna Dublin Delhi Sydney".split()
COS = "Acme Globex Initech Umbrella Soylent Hooli Stark Wayne Tyrell Wonka".split()
T = {"lives_in": dict(seen=["I live in {v} now.", "We moved to {v} last month.", "I relocated to {v} for good."], unseen=["{v} is home for us these days.", "We've made our base in {v}.", "Our new address is in {v}."], q="Where does {p} live now?", qp="Where did {p} live in {y}?", vals=CITIES),
     "works_at": dict(seen=["I work at {v} now.", "I joined {v} this month.", "I was hired at {v}."], unseen=["{v} is my employer these days.", "My new office is at {v}.", "I took a role with {v}."], q="Where does {p} work now?", qp="Where did {p} work in {y}?", vals=COS)}
noise = [t for c in range(5) for t in K.load_conv(K.DATA[c]).turns if not t["cap"]]
cfg = Config(**json.load(open("final_config.json")))
tmp = tempfile.mkdtemp()


def updates(kind, seed=11):
    """The scenario of research/updates.py: same wordings, same numbers, same questions."""
    rng = random.Random(seed)
    ses = [(f"1:00 pm on {5 + s} {mon}, {y}", y) for y, mon in ((2022, "March"), (2023, "June"), (2024, "September")) for s in range(3)]
    plan, qs = [], []
    for p in PEOPLE:
        for slot, d in T.items():
            a, b = rng.sample(d["vals"], 2)
            sa, sb = rng.randrange(0, 3), rng.randrange(6, 9)
            plan += [(sa, p, rng.choice(d[kind]).format(v=a), slot, a), (sb, p, rng.choice(d[kind]).format(v=b), slot, b)]
            qs += [dict(q=d["q"].format(p=p), want=b, stale=a, kind="current", p=p), dict(q=d["qp"].format(p=p, y=ses[sa][1]), want=a, stale=b, kind="past", p=p)]
    for validate in (False, True):
        m = E.ChatMemory(f"{tmp}/u{kind}{validate}.db", cfg=cfg, durable=False)
        pipe, kb = KnowledgePipeline(m.db), KnowledgeBase(m.db)
        engine = StateEngine(kb)
        verdicts = dict(accepted=0, uncertain=0, rejected=0)
        cand = []
        for s, (date, _) in enumerate(ses):
            m.open_chat(date)
            for x in rng.sample(noise, 40): m.add(x["speaker"], x["text"], date=date, commit=False)
            for pl in plan:
                if pl[0] == s: cand.append((m.add(pl[1], pl[2], date=date, commit=False), pl))
            m.db.commit(); m.close_chat(write_note=False, sleep=s == len(ses) - 1)
        for sid, (_, p, _, slot, value) in cand:
            if validate:
                verdicts[pipe.process(Candidate(p, slot, value, (sid,), "", "perfect writer", "1"))[0].verdict] += 1
            else:
                who = kb.resolve(p) or kb.add_entity(Entity.create(p, "person")).entity_id
                f = KnowledgeFact.create(who, slot, "text", PREDICATE_TYPES[slot], "explicit", object_value=value, validation_status="accepted")
                engine.apply_observation(FactWithEvidence(f, (FactEvidence.create(f.fact_id, sid),)))
        r = KnowledgeRetriever(m)
        score = {"current": [0, 0, 0], "past": [0, 0, 0]}  # right, wrong, no record
        wrong_flagged = wrong_all = 0
        for q in qs:
            res = r.retrieve(q["q"])
            values = [f["value"] for f in res["facts"]]
            row = score[q["kind"]]
            n = 0 if values == [q["want"]] else 2 if not values else 1
            row[n] += 1
            if n == 1 and q["kind"] == "current":
                wrong_all += 1; wrong_flagged += bool(res.get("waiting_review"))
        yield ("after_p2_checks" if validate else "facts_given", dict(wrong_with_warning=wrong_flagged, wrong_total=wrong_all, verdicts=verdicts if validate else None, current=dict(zip(("right", "wrong", "no_record"), score["current"])), past=dict(zip(("right", "wrong", "no_record"), score["past"])), n=len(qs) // 2))
        m.close()


ASK = dict(
    current=["What is the current employer of {p}?", "Where does {p} work now?"],
    past=["Where did {p} work in {y}?"],
    change=["When did {p} change jobs?"],
    evidence=["What evidence supports the current employer of {p}?"],
    original=["What was the original employer of {p}?"],
    conflict=["Are there conflicting observations about {p}'s employer?"],
    unfamiliar=["Which outfit pays {p} at the moment?", "Who is {p} with these days, jobwise?", "Has {p} stayed at the same firm?", "Where was {p} a few years back?"])


def question_set(seed=5):
    rng = random.Random(seed)
    m = E.ChatMemory(f"{tmp}/q.db", cfg=cfg, durable=False)
    kb = KnowledgeBase(m.db)
    engine = StateEngine(kb)
    months = [("January", "01"), ("July", "07"), ("October", "10")]
    gold = {}
    for n, p in enumerate(PEOPLE):
        chain = rng.sample(COS, 3)
        who = kb.add_entity(Entity.create(p, "person")).entity_id
        sids = []
        for (name, num), y, v in zip(months, (2022, 2024, 2025), chain):
            date = f"1:00 pm on 5 {name}, {y}"
            m.open_chat(date)
            for x in rng.sample(noise, 15): m.add(x["speaker"], x["text"], date=date, commit=False)
            sid = m.add(p, f"I work at {v} now.", date=date, commit=False)
            m.db.commit(); m.close_chat(write_note=False, sleep=False)
            f = KnowledgeFact.create(who, "works_at", "text", "relationship", "explicit", object_value=v, valid_from=f"{y}-{num}", validation_status="accepted")
            engine.apply_observation(FactWithEvidence(f, (FactEvidence.create(f.fact_id, sid),)))
            sids.append(sid)
        rival = None
        if n % 4 == 0:  # someone else gives a different employer
            date = "1:00 pm on 9 November, 2025"
            m.open_chat(date)
            rival = rng.choice([c for c in COS if c not in chain])
            sid = m.add("Bystander", f"{p} works at {rival}, I heard.", date=date)
            m.close_chat(write_note=False, sleep=False)
            f = KnowledgeFact.create(who, "works_at", "text", "relationship", "explicit", object_value=rival, valid_from="2025-11", validation_status="accepted")
            engine.apply_observation(FactWithEvidence(f, (FactEvidence.create(f.fact_id, sid),)))
            rival = (rival, sid)
        gold[p] = dict(chain=chain, sids=sids, rival=rival)
    m._sleep()
    r = KnowledgeRetriever(m)
    rows = []
    for p, g in gold.items():
        now = {g["chain"][2]} | ({g["rival"][0]} if g["rival"] else set())
        now_ev = {g["sids"][2]} | ({g["rival"][1]} if g["rival"] else set())
        wanted = dict(current=(now, now_ev), past=({g["chain"][0], g["chain"][1]}, {g["sids"][0], g["sids"][1]}), change=(now, now_ev), evidence=(now, now_ev), original=({g["chain"][0]}, {g["sids"][0]}), conflict=(now if g["rival"] else set(), now_ev if g["rival"] else set()), unfamiliar=(None, None))
        for kind, wordings in ASK.items():
            for w in wordings:
                t0 = time.perf_counter()
                res = r.retrieve(w.format(p=p, y=2024))
                ms = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                plain = set(m.find(w.format(p=p, y=2024))[0])
                plain_ms = (time.perf_counter() - t0) * 1000
                values, ev = {f["value"] for f in res["facts"]}, {s for f in res["facts"] for s in f["evidence"]}
                want, want_ev = wanted[kind]
                row = dict(kind=kind, q=w, routed=res["query_type"], ms=ms, plain_ms=plain_ms, sentences=res["sentences_used"], plain_sentences=len(plain))
                if want is not None:
                    row |= dict(right=values == want, evidence_found=len(ev & want_ev) / len(want_ev) if want_ev else 1.0, plain_found=len(plain & want_ev) / len(want_ev) if want_ev else 1.0, extra_records=len(ev - want_ev))
                else:
                    row |= dict(answered=bool(res["facts"]), values=sorted(values))
                rows.append(row)
    m.close()
    out = {}
    for kind in ASK:
        v = [x for x in rows if x["kind"] == kind]
        d = dict(n=len(v), ms=round(statistics.median(x["ms"] for x in v), 1), plain_ms=round(statistics.median(x["plain_ms"] for x in v), 1), sentences=statistics.mean(x["sentences"] for x in v), plain_sentences=statistics.mean(x["plain_sentences"] for x in v))
        if kind == "unfamiliar":
            d |= dict(routes=sorted({(x["q"], x["routed"]) for x in v}), answered_from_records=sum(x["answered"] for x in v))
        else:
            d |= dict(right=sum(x["right"] for x in v), evidence_recall=round(statistics.mean(x["evidence_found"] for x in v), 3), plain_evidence_recall=round(statistics.mean(x["plain_found"] for x in v), 3))
        out[kind] = d
    return out


if __name__ == "__main__":
    out = dict(questions=question_set())
    for kind in ("seen", "unseen"):
        for name, res in updates(kind): out[f"updates_{kind}_{name}"] = res
    out["old_world_state"] = {k: v for k, v in json.load(open("research/results_updates.json")).items() if k.startswith("world_")}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out, indent=1, default=str))
