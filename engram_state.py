"""Knowledge State v0.1: a persistent record of entities, relationships, events and changing facts.

The fourth part of Engram, beside the record, the index and sleep. It stores no conversation text: every item links to
sentence ids in ChatMemory's database, and explain_state reads the sentences only when asked.

  learn(observation)               validates one structured observation and updates the state
  get_state(entity, as_of=None)    the current or historical state
  explain_state(entity, as_of=None) the state with its evidence, changes and uncertainties

Rules: nothing is accepted without a real source. A stated item and an inferred item are kept apart, and an inference
never changes the stated state. Names merge only on an exact name or a stated alias. Checks here are structural: they do
not prove that a source means what an item says, so every item reports checked="structure only".
"""
import re

SCHEMA = """
    CREATE TABLE IF NOT EXISTS entities(entity INTEGER PRIMARY KEY, name TEXT UNIQUE, kind TEXT);
    CREATE TABLE IF NOT EXISTS entity_aliases(alias TEXT PRIMARY KEY, entity INTEGER, sid INTEGER);
    CREATE TABLE IF NOT EXISTS relationships(item INTEGER PRIMARY KEY, subject INTEGER, predicate TEXT, object INTEGER, valid_from TEXT, valid_to TEXT, ended_at TEXT, time_status TEXT, status TEXT, retracted INTEGER DEFAULT 0, reason TEXT, basis TEXT, event INTEGER);
    CREATE TABLE IF NOT EXISTS knowledge_facts(item INTEGER PRIMARY KEY, subject INTEGER, predicate TEXT, value TEXT, valid_from TEXT, valid_to TEXT, ended_at TEXT, time_status TEXT, status TEXT, retracted INTEGER DEFAULT 0, reason TEXT, basis TEXT, event INTEGER);
    CREATE TABLE IF NOT EXISTS events(event INTEGER PRIMARY KEY, type TEXT, subject INTEGER, object TEXT, time TEXT, time_status TEXT, status TEXT, retracted INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS fact_evidence(kind TEXT, item INTEGER, sid INTEGER, PRIMARY KEY(kind, item, sid));
    CREATE INDEX IF NOT EXISTS fact_evidence_sid ON fact_evidence(sid);
    CREATE TABLE IF NOT EXISTS state_transitions(transition INTEGER PRIMARY KEY, entity INTEGER, predicate TEXT, old TEXT, new TEXT, time TEXT, time_status TEXT);
"""

# predicate: one value at a time?
RELATIONS = {"owns": False, "lives_in": True, "works_at": True, "married_to": True, "parent_of": False, "friend_of": False, "member_of": False}
ATTRIBUTES = {"job_title": True, "status": True, "likes": False, "dislikes": False, "hobby": False, "plans": False, "number": False}
# event type: (what it does, predicate)
EVENTS = {"adopted": ("start", "owns"), "bought": ("start", "owns"), "sold": ("end", "owns"), "lost": ("end", "owns"), "gave_away": ("end", "owns"),
          "moved_to": ("start", "lives_in"), "started_job": ("start", "works_at"), "left_job": ("end", "works_at"), "got_married": ("start", "married_to"),
          "divorced": ("end", "married_to"), "joined": ("start", "member_of"), "left_group": ("end", "member_of"), "started_hobby": ("start", "hobby"),
          "stopped_hobby": ("end", "hobby"), "attended": (None, None), "visited": (None, None), "completed": (None, None)}
VAGUE = {"i", "me", "my", "he", "she", "they", "him", "her", "them", "we", "us", "it", "someone", "somebody", "friend", "my friend", "the kids", "kids", "family"}
MONTHS = {m: i + 1 for i, m in enumerate("january february march april may june july august september october november december".split())}
MONTHS.update({m[:3]: n for m, n in list(MONTHS.items())})
WORD = re.compile(r"[a-z0-9]+")
CHECKED = "structure only"


def norm(name):
    return re.sub(r"^(the|a|an)\s+", "", " ".join(WORD.findall((name or "").lower())))


def tokens(text):
    return set(WORD.findall((text or "").lower()))


def parse_date(text):
    """A date to YYYY-MM-DD, or None. A month or a year means its first day."""
    if not text: return None
    t = str(text).lower().strip()
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m: return f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}"
    m = re.search(r"\b(\d{1,2})\s+(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r"),?\s+(\d{4})\b", t)
    if m: return f"{m[3]}-{MONTHS[m[2]]:02d}-{int(m[1]):02d}"
    m = re.search(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\s+(\d{1,2}),?\s+(\d{4})\b", t)
    if m: return f"{m[3]}-{MONTHS[m[1]]:02d}-{int(m[2]):02d}"
    m = re.search(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r"),?\s+(\d{4})\b", t)
    if m: return f"{m[2]}-{MONTHS[m[1]]:02d}-01"
    m = re.fullmatch(r"(\d{4})", t)
    return f"{m[1]}-01-01" if m else None


def forget_sentence(db, sid):
    """Removes a sentence's evidence, then anything left with none, including inferences built on it."""
    touched = set()
    gone = db.execute("SELECT kind, item FROM fact_evidence WHERE sid=?", (sid,)).fetchall()
    db.execute("DELETE FROM fact_evidence WHERE sid=?", (sid,))
    db.execute("DELETE FROM entity_aliases WHERE sid=?", (sid,))
    table = dict(relationship="relationships", fact="knowledge_facts")
    removed = set()
    for kind, item in gone:
        if db.execute("SELECT 1 FROM fact_evidence WHERE kind=? AND item=?", (kind, item)).fetchone(): continue
        if kind.endswith("_correction"):  # the correction has gone, so the item returns and its reason is removed
            base = kind.removesuffix("_correction")
            r = db.execute(f"SELECT subject, predicate FROM {table[base]} WHERE item=?", (item,)).fetchone()
            if r:
                db.execute(f"UPDATE {table[base]} SET retracted=0, reason=NULL WHERE item=?", (item,))
                touched.add((base, r[0], r[1]))
            continue
        if kind == "event":
            db.execute("DELETE FROM events WHERE event=?", (item,)); removed.add(("event", item)); continue
        r = db.execute(f"SELECT subject, predicate FROM {table[kind]} WHERE item=?", (item,)).fetchone()
        if r: touched.add((kind, r[0], r[1]))
        db.execute(f"DELETE FROM {table[kind]} WHERE item=?", (item,)); removed.add((kind, item))
    # An end event that has gone reopens what it closed, and an inference falls with its basis.
    for t in ("relationships", "knowledge_facts"):
        kind = "relationship" if t == "relationships" else "fact"
        for item, subject, predicate, event, basis in db.execute(f"SELECT item, subject, predicate, event, basis FROM {t}").fetchall():
            if event and ("event", event) in removed:
                db.execute(f"UPDATE {t} SET ended_at=NULL, event=NULL WHERE item=?", (item,)); touched.add((kind, subject, predicate))
            if basis and any(f"{k}:{i}" in basis.split(",") for k, i in removed):
                db.execute(f"DELETE FROM {t} WHERE item=?", (item,))
                db.execute("DELETE FROM fact_evidence WHERE kind=? AND item=?", (kind, item))
    return touched


class KnowledgeState:
    def __init__(self, db):
        self.db = db
        db.executescript(SCHEMA)

    # ---------- names ----------
    def resolve(self, name):
        key = norm(name)
        if not key: return None
        r = self.db.execute("SELECT entity FROM entities WHERE name=?", (key,)).fetchone() or self.db.execute("SELECT entity FROM entity_aliases WHERE alias=?", (key,)).fetchone()
        return r[0] if r else None

    def _entity(self, name, kind=""):
        found = self.resolve(name)
        if found: return found
        return self.db.execute("INSERT INTO entities(name, kind) VALUES(?,?)", (norm(name), kind)).lastrowid

    def _names(self, entity):
        rows = [r[0] for r in self.db.execute("SELECT name FROM entities WHERE entity=?", (entity,))] + [r[0] for r in self.db.execute("SELECT alias FROM entity_aliases WHERE entity=?", (entity,))]
        return rows

    def _sentence(self, sid):
        return self.db.execute("SELECT speaker, date, text, caption FROM sentences WHERE sid=?", (sid,)).fetchone()

    @staticmethod
    def _no(reason):
        return dict(accepted=False, reason=reason, checked=CHECKED)

    def _mentioned(self, name, sources, known=None):
        """Every word of the name, or a known alias, appears in a source or its speaker."""
        names = [norm(name)] + (self._names(known) if known else [])
        for sid in sources:
            r = self._sentence(sid)
            words = tokens(f"{r[0]} {r[2]} {r[3] or ''}")
            if any(tokens(n) and tokens(n) <= words for n in names): return True
        return False

    # ---------- learning ----------
    def learn(self, o):
        """o: subject, predicate or event, object or value, sources (sentence ids); optional time, status ('stated' or 'inferred'), basis.
        Nothing partly applies: a rejected observation changes nothing and says why."""
        result = self._learn(o)
        self.db.commit()
        return result

    def _learn(self, o):
        status = o.get("status", "stated")
        if status not in ("stated", "inferred"): return self._no("status must be stated or inferred")
        sources = list(dict.fromkeys(o.get("sources") or []))
        basis = o.get("basis") or []
        if status == "inferred":
            if not basis: return self._no("an inference must name what it was inferred from")
            for kind, item in basis:
                sources += [r[0] for r in self.db.execute("SELECT sid FROM fact_evidence WHERE kind=? AND item=?", (kind, item))]
            sources = list(dict.fromkeys(sources))
        if not sources: return self._no("no source")
        if any(self._sentence(s) is None for s in sources): return self._no("source sentence does not exist")
        subject = o.get("subject", "")
        if norm(subject) in VAGUE or not norm(subject): return self._no("subject does not resolve to an entity")
        known = self.resolve(subject)
        if status == "stated" and not self._mentioned(subject, sources, known): return self._no("subject not found in its source")

        if o.get("predicate") == "also_known_as":
            return self._alias(subject, o.get("object"), sources)
        event, predicate = o.get("event"), o.get("predicate")
        if event and event not in EVENTS: return self._no(f"event type not allowed: {event}")
        if not event and predicate not in RELATIONS and predicate not in ATTRIBUTES: return self._no(f"predicate not allowed: {predicate}")
        target = o.get("object", o.get("value"))
        action, implied = EVENTS[event] if event else ("start", predicate)
        if event and action is None and not target: return self._no("event needs what it happened to")
        if implied and not (target or "").strip(): return self._no("object is missing")
        if implied and norm(target) in VAGUE: return self._no("object does not resolve to an entity")
        if target and status == "stated" and not self._mentioned(target, sources, self.resolve(target)): return self._no("object not found in its source")

        when, time_status = parse_date(o.get("time")), "explicit"
        if not when:
            said = [parse_date(self._sentence(s)[1]) for s in sources]
            said = sorted(x for x in said if x)
            when, time_status = (said[-1], "said") if said else (None, "unknown")

        subject_id = known or self._entity(subject, o.get("subject_kind", ""))
        event_id = None
        if event:
            event_id = self.db.execute("INSERT INTO events(type, subject, object, time, time_status, status) VALUES(?,?,?,?,?,?)", (event, subject_id, target, when, time_status, status)).lastrowid
            self._evidence("event", event_id, sources)
        out = dict(accepted=True, checked=CHECKED, status=status, time=when, time_status=time_status, event=event_id)
        if implied is None: return out
        is_relation = implied in RELATIONS
        kind, table = ("relationship", "relationships") if is_relation else ("fact", "knowledge_facts")
        column = "object" if is_relation else "value"
        value = self._entity(target, o.get("object_kind", "")) if is_relation else target.strip()
        if action == "end":
            if status == "stated":
                row = self._open_item(table, column, subject_id, implied, value, when)
                if row:
                    self.db.execute(f"UPDATE {table} SET ended_at=?, event=? WHERE item=?", (when, event_id, row))
                    self._evidence(kind, row, sources)
                    out.update(kind=kind, item=row, ended=True)
                else:
                    out.update(note="no earlier state to end")
                self._rebuild(kind, subject_id, implied)
            return out
        basis_text = ",".join(f"{k}:{i}" for k, i in basis) or None
        same = self.db.execute(f"SELECT item FROM {table} WHERE subject=? AND predicate=? AND {column}=? AND status=? AND retracted=0 AND ended_at IS NULL AND valid_from IS ?", (subject_id, implied, value, status, when)).fetchone()
        if same:
            item = same[0]
        else:
            item = self.db.execute(f"INSERT INTO {table}(subject, predicate, {column}, valid_from, time_status, status, basis, event) VALUES(?,?,?,?,?,?,?,?)", (subject_id, implied, value, when, time_status, status, basis_text, None)).lastrowid
        self._evidence(kind, item, sources)
        if status == "stated": self._rebuild(kind, subject_id, implied)
        out.update(kind=kind, item=item)
        return out

    def _alias(self, name, alias, sources):
        if not alias or norm(alias) in VAGUE: return self._no("alias is vague")
        if not self._mentioned(alias, sources): return self._no("alias not found in its source")
        entity = self._entity(name)
        taken = self.resolve(alias)
        if taken and taken != entity: return self._no("alias already belongs to another entity; not merged")
        if not taken: self.db.execute("INSERT INTO entity_aliases(alias, entity, sid) VALUES(?,?,?)", (norm(alias), entity, sources[0]))
        return dict(accepted=True, kind="alias", checked=CHECKED)

    def _evidence(self, kind, item, sources):
        for sid in sources:
            self.db.execute("INSERT OR IGNORE INTO fact_evidence(kind, item, sid) VALUES(?,?,?)", (kind, item, sid))

    def _open_item(self, table, column, subject, predicate, value, when):
        rows = self.db.execute(f"SELECT item, valid_from FROM {table} WHERE subject=? AND predicate=? AND {column}=? AND retracted=0 AND status='stated' AND ended_at IS NULL ORDER BY valid_from DESC", (subject, predicate, value)).fetchall()
        for item, start in rows:
            if start is None or when is None or start <= when: return item
        return None

    def _label(self, predicate, value):
        if predicate in RELATIONS:
            r = self.db.execute("SELECT name FROM entities WHERE entity=?", (value,)).fetchone()
            return r[0] if r else str(value)
        return value

    def _rebuild(self, kind, subject, predicate):
        """Recomputes dates and history from the stated items, so order of learning does not matter."""
        table, column = ("relationships", "object") if kind == "relationship" else ("knowledge_facts", "value")
        single = (RELATIONS if kind == "relationship" else ATTRIBUTES)[predicate]
        rows = self.db.execute(f"SELECT item, {column}, valid_from, ended_at, time_status FROM {table} WHERE subject=? AND predicate=? AND status='stated' AND retracted=0", (subject, predicate)).fetchall()
        dated = sorted((r for r in rows if r[2]), key=lambda r: (r[2], r[0]))
        for r in rows:
            self.db.execute(f"UPDATE {table} SET valid_to=? WHERE item=?", (r[3], r[0]))
        self.db.execute("DELETE FROM state_transitions WHERE entity=? AND predicate=?", (subject, predicate))
        add = lambda old, new, when, ts: self.db.execute("INSERT INTO state_transitions(entity, predicate, old, new, time, time_status) VALUES(?,?,?,?,?,?)", (subject, predicate, old, new, when, ts))
        if single:
            dates = sorted({r[2] for r in dated})
            following = dict(zip(dates, dates[1:]))
            for a in dated:  # every value from one date ends at the next date, even when several share a date
                b = following.get(a[2])
                if b and (a[3] is None or a[3] > b):
                    self.db.execute(f"UPDATE {table} SET valid_to=? WHERE item=?", (b, a[0]))
            previous = None
            for r in dated:
                label = self._label(predicate, r[1])
                if label != previous: add(previous, label, r[2], r[4])
                previous = label
                if r[3]:
                    add(label, None, r[3], "explicit"); previous = None
        else:
            for r in dated:
                label = self._label(predicate, r[1])
                add(None, label, r[2], r[4])
                if r[3]: add(label, None, r[3], "explicit")

    # ---------- corrections ----------
    def retract(self, kind, item, reason, sources):
        """A correction: the item leaves the state but stays visible in explain_state with the reason."""
        table = dict(relationship="relationships", fact="knowledge_facts").get(kind)
        if not table: return self._no("unknown kind")
        if not sources or any(self._sentence(s) is None for s in sources): return self._no("a correction needs a real source")
        r = self.db.execute(f"SELECT subject, predicate FROM {table} WHERE item=?", (item,)).fetchone()
        if not r: return self._no("no such item")
        self.db.execute(f"UPDATE {table} SET retracted=1, reason=? WHERE item=?", (reason, item))
        self._evidence(kind + "_correction", item, sources)
        self._rebuild(kind, r[0], r[1])
        self.db.commit()
        return dict(accepted=True, checked=CHECKED)

    def rebuild_after_forget(self, touched):
        for kind, subject, predicate in touched: self._rebuild(kind, subject, predicate)
        known = [r[0] for r in self.db.execute("SELECT entity FROM entities")]
        for e in known:
            used = any(self.db.execute(q, (e, e) if q.count("?") == 2 else (e,)).fetchone() for q in (
                "SELECT 1 FROM relationships WHERE subject=? OR object=?", "SELECT 1 FROM knowledge_facts WHERE subject=?", "SELECT 1 FROM events WHERE subject=?", "SELECT 1 FROM entity_aliases WHERE entity=?"))
            if not used: self.db.execute("DELETE FROM entities WHERE entity=?", (e,))
        self.db.commit()

    # ---------- reading ----------
    def _rows(self, entity):
        rel = [("relationship", r) for r in self.db.execute("SELECT item, predicate, object, valid_from, valid_to, time_status, status, retracted, reason FROM relationships WHERE subject=?", (entity,))]
        fact = [("fact", r) for r in self.db.execute("SELECT item, predicate, value, valid_from, valid_to, time_status, status, retracted, reason FROM knowledge_facts WHERE subject=?", (entity,))]
        return rel + fact

    def get_state(self, entity, as_of=None):
        """What is held about an entity now, or on a past date. Inferences and undated items are listed apart from the state."""
        eid = self.resolve(entity)
        if not eid: return dict(entity=entity, known=False, state={}, inferred=[], time_unknown=[], checked=CHECKED)
        when = parse_date(as_of) if as_of else None
        if as_of and not when: raise ValueError(f"cannot read date: {as_of}")
        state, inferred, unknown = {}, [], []
        for kind, (item, predicate, value, start, end, ts, status, retracted, _) in self._rows(eid):
            if retracted: continue
            entry = dict(kind=kind, item=item, predicate=predicate, value=self._label(predicate, value) if kind == "relationship" else value, since=start, until=end, time_status=ts, status=status)
            if status == "inferred":
                inferred.append(entry); continue
            if start is None:
                unknown.append(entry); continue
            if when:
                if start > when or (end and end <= when): continue
            elif end:
                continue
            state.setdefault(predicate, []).append(entry)
        for items in state.values(): items.sort(key=lambda e: e["since"])
        name = self.db.execute("SELECT name FROM entities WHERE entity=?", (eid,)).fetchone()[0]
        return dict(entity=name, known=True, as_of=when, state=state, inferred=inferred, time_unknown=unknown, checked=CHECKED)

    def _quote(self, kind, item):
        rows = self.db.execute("SELECT e.sid, s.speaker, s.date, s.text FROM fact_evidence e JOIN sentences s ON s.sid=e.sid WHERE e.kind=? AND e.item=? ORDER BY e.sid", (kind, item)).fetchall()
        return [dict(sid=r[0], speaker=r[1], date=r[2], text=r[3]) for r in rows]

    def explain_state(self, entity, as_of=None):
        """The state with the exact sentences behind it, the changes over time, and what remains uncertain."""
        out = self.get_state(entity, as_of)
        if not out["known"]: return dict(out, changes=[], corrections=[], events=[], uncertainties=["nothing is known about this entity"])
        eid = self.resolve(entity)
        when, unsure = out["as_of"], []
        for items in out["state"].values():
            for e in items:
                e["evidence"] = self._quote(e["kind"], e["item"])
                if e["time_status"] == "said":
                    unsure.append(f"{e['predicate']} {e['value']}: date is when it was said, not when it happened")
        for e in out["inferred"]:
            e["evidence"] = self._quote(e["kind"], e["item"])
            unsure.append(f"{e['predicate']} {e['value']}: inferred, not stated")
        for e in out["time_unknown"]:
            e["evidence"] = self._quote(e["kind"], e["item"])
            unsure.append(f"{e['predicate']} {e['value']}: time unknown, so it is left out of the dated state")
        for predicate, items in out["state"].items():
            if len(items) > 1 and (RELATIONS.get(predicate) or ATTRIBUTES.get(predicate)):
                unsure.append(f"{predicate}: more than one value on the same date")
        changes = [dict(predicate=r[0], old=r[1], new=r[2], time=r[3], time_status=r[4]) for r in self.db.execute("SELECT predicate, old, new, time, time_status FROM state_transitions WHERE entity=? ORDER BY time, transition", (eid,)) if not when or (r[3] and r[3] <= when)]
        for c in changes:
            if c["time_status"] == "said" and c["old"]:
                unsure.append(f"{c['predicate']} from {c['old']} to {c['new']}: order rests on when it was said")
        corrections = []
        for kind, (item, predicate, value, start, end, ts, status, retracted, reason) in self._rows(eid):
            if retracted:
                corrections.append(dict(kind=kind, item=item, predicate=predicate, value=self._label(predicate, value) if kind == "relationship" else value, reason=reason, evidence=self._quote(kind, item) + self._quote(kind + "_correction", item)))
        events = [dict(type=r[0], object=r[1], time=r[2], time_status=r[3], evidence=self._quote("event", r[4])) for r in self.db.execute("SELECT type, object, time, time_status, event FROM events WHERE subject=? AND retracted=0 ORDER BY time, event", (eid,)) if not when or (r[2] and r[2] <= when)]
        return dict(out, changes=changes, corrections=corrections, events=events, uncertainties=list(dict.fromkeys(unsure)))
