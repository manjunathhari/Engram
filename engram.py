"""
Engram: a layered memory store modelled on how the brain separates storage from retrieval.

Runs on one CPU core. Reads use no LLM and no GPU.

  Write path (fast, online)
    thalamic_gate   decide whether a turn earns index cues at all
    tag_salience    amygdala: emotional weight from a small lexicon
    classify_store  which cortical store holds the content
    bind_engram     hippocampus: cues -> pointer record, not a copy of the text

  Sleep (slow, offline, batch)
    sleep()         builds episode gists and term statistics

  Read path (staged, cheapest first)
    stage 0  basal ganglia   exact-repeat cue -> stored answer pointers
    stage 1  thalamic router question type -> which store to favour
    stage 2  episode shortlist: search gists, not the whole memory
    stage 3  hippocampal completion: partial cue -> engrams inside those episodes
    stage 4  spreading: pull in the neighbouring turns of each engram
    stage 5  cingulate: if the top matches are not clearly separated, widen; else stop early
"""
from __future__ import annotations
import math, re, sys
from datetime import date, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

STOP = set("""a an the and or but if then else of to in on at for from by with about as into over after before between
is are was were be been being am do does did doing done have has had having i me my mine we us our you your he him his she her
it its they them their this that these those there here what which who whom whose when where why how will would can could
should shall may might must not no yes so too very just also than more most much many some any all each both few other own
same s t don didn doesn isn wasn weren aren couldn wouldn shouldn won ve ll re d m oh hey hi hello thanks thank wow yeah yep
really like get got go going went gone know think thought make made one two good great nice lot lots things thing way
say said tell told see saw seen let lets well sure still even now always never ever""".split())

EMOTION = set("""afraid scared terrified fear fears fearful anxious anxiety worried worry panic nervous stress stressed stressful
angry anger furious upset mad frustrated frustrating annoyed sad sadness cried crying cry tears grief grieving lonely depressed
heartbroken hurt hurting pain painful accident crash danger dangerous emergency hospital died death funeral lost loss
happy happiness joy joyful thrilled excited exciting proud pride grateful thankful love loved overjoyed amazed relieved
ashamed guilty embarrassed disappointed devastated overwhelmed shocked terrifying nightmare""".split())

PLAN = re.compile(r"\b(plan|plans|planning|going to|gonna|goal|goals|hope to|want to|wants to|next (week|month|year)|intend|dream of|looking forward|will be|decided to|thinking of|thinking about)\b", re.I)

ROUTES = {
    "visual": re.compile(r"\b(photo|photos|picture|pictures|image|images|painting|paintings|painted|paint|drew|drawing|sketch|mural|look like|looks like|colou?r|poster|sign|tattoo|shown|show)\b", re.I),
    "emotion": re.compile(r"\b(feel|feels|felt|feeling|feelings|afraid|scared|happy|sad|upset|proud|excited|angry|worried|emotion|emotional|reaction|react|reacted)\b", re.I),
    "plan": re.compile(r"\b(plan|plans|planning|goal|goals|future|going to|want to|intend|hope|next|dream|aspire|career|pursue)\b", re.I),
    "time": re.compile(r"\b(when|date|how long|what year|which year|what month|first time|last time|recently|since|before|after|ago)\b", re.I),
}
TIMEWORD = re.compile(r"\b(yesterday|today|tomorrow|tonight|last|next|ago|week|weekend|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december|recently|soon|since|\d{4})\b", re.I)
SETQ = re.compile(r"\b(both|common|what (are|kinds?|types?)|which .*s\b|how many|activities|hobbies|all)\b", re.I)
TOKEN = re.compile(r"[a-z0-9']+")
CAP = re.compile(r"\b[A-Z][a-z]{2,}\b")
MONTHS = set("january february march april may june july august september october november december".split())


def stem(w: str) -> str:
    """A very small suffix stripper. Deliberately crude: no dictionary, no model."""
    if len(w) > 5 and w.endswith("ies"): return w[:-3] + "y"
    for suf in ("ingly", "edly", "ing", "ed", "es", "s", "ly"):
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 4 and w[-1] == w[-2] and w[-1] not in "aeiou": w = w[:-1]
    return w


def words(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower().replace("'s", "")) if t not in STOP and len(t) > 1]


MONTHS_L = "january february march april may june july august september october november december".split()
WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
_M = "|".join(MONTHS_L)
Q_MY = re.compile(rf"\b(?:in\s+|of\s+|during\s+)?({_M})\s*,?\s*(\d{{4}})\b", re.I)
Q_Y = re.compile(r"\b(?:in|during)\s+(\d{4})\b", re.I)
PAST_Q = re.compile(r"\b(used to|before|previously|earlier|former|originally|at first|first|back in|in \d{4}|did .* (live|work))\b", re.I)
REL = [(re.compile(r"\byesterday\b", re.I), 1, 1), (re.compile(r"\blast night\b", re.I), 1, 1), (re.compile(r"\b(today|this morning|tonight)\b", re.I), 0, 0),
       (re.compile(r"\blast week(end)?\b", re.I), 7, 14), (re.compile(r"\blast month\b", re.I), 28, 45), (re.compile(r"\blast year\b", re.I), 300, 430),
       (re.compile(r"\b(a|one) week ago\b", re.I), 7, 7), (re.compile(r"\btwo weeks ago\b", re.I), 14, 14), (re.compile(r"\b(a|one) month ago\b", re.I), 30, 30)]


def parse_session_date(s: str):
    m = re.search(r"(\d{1,2}) (\w+),? (\d{4})", s or "")
    if not m or m[2].lower() not in MONTHS_L: return None
    return date(int(m[3]), MONTHS_L.index(m[2].lower()) + 1, int(m[1]))


def mention_range(text: str, d):
    """Resolve 'last week' and similar against the date of the episode. Returns ordinal range or None."""
    if d is None: return None
    for r, lo, hi in REL:
        if r.search(text): return (d.toordinal() - hi, d.toordinal() - lo)
    m = re.search(r"\blast (" + "|".join(WEEKDAYS) + r")\b", text, re.I)
    if m:
        back = (d.weekday() - WEEKDAYS.index(m[1].lower())) % 7 or 7
        return (d.toordinal() - back, d.toordinal() - back)
    m = re.search(rf"\b({_M})\s+(\d{{4}})\b", text, re.I)
    if m:
        a = date(int(m[2]), MONTHS_L.index(m[1].lower()) + 1, 1)
        return (a.toordinal(), a.toordinal() + 30)
    return None


def query_range(q: str):
    m = Q_MY.search(q)
    if m:
        a = date(int(m[2]), MONTHS_L.index(m[1].lower()) + 1, 1)
        return (a.toordinal(), a.toordinal() + 30)
    m = Q_Y.search(q)
    if m: return (date(int(m[1]), 1, 1).toordinal(), date(int(m[1]), 12, 31).toordinal())
    return None


# World-state slots, after LeCun 4.9: only the entry an event touches is rewritten.
SLOT_WRITE = {
    "home": re.compile(r"\b(live|lives|lived|living|moved|moving|relocated|settled)\b.*\b(in|to)\b", re.I),
    "job": re.compile(r"\b(work|works|worked|working|joined|job|employed|hired)\b.*\b(at|for|as)\b", re.I),
    "school": re.compile(r"\b(study|studies|studying|enrolled|attending|attend|attends|student)\b.*\b(at|in)\b", re.I),
    "pet": re.compile(r"\b(adopted|got|have|has|bought)\b.*\b(dog|cat|puppy|kitten|rabbit|parrot|pet)\b", re.I),
    "car": re.compile(r"\b(drive|drives|driving|bought|sold|got)\b.*\b(car|truck|suv|tesla|toyota|honda|bmw|ford)\b", re.I),
}
SLOT_READ = {
    "home": re.compile(r"\b(live|lives|lived|living|home|city|town|reside|resides|moved)\b", re.I),
    "job": re.compile(r"\b(work|works|worked|job|employer|company|occupation)\b", re.I),
    "school": re.compile(r"\b(study|studies|studied|school|college|university)\b", re.I),
    "pet": re.compile(r"\b(pet|pets|dog|cat|puppy|kitten)\b", re.I),
    "car": re.compile(r"\b(car|drive|drives|vehicle)\b", re.I),
}
WORLD_GATE = re.compile(r"\b(now|currently|current|these days|still|at present|used to|previously|former|originally|in \d{4})\b", re.I)
ACK = re.compile(r"^(wow|thanks|thank you|awesome|great|cool|nice|that'?s|sounds|yeah|yes|oh|haha|lol|congrats|congratulations|good luck)\b", re.I)


def importance_features(text: str, caption: str, prev_text: str, pos: int, n_in_ep: int) -> list[float]:
    """Cheap features for the learned critic. No model is called."""
    toks = TOKEN.findall(text.lower())
    return [1.0, math.log1p(len(toks)), float(text.strip().endswith("?")), float("?" in text), min(3, len(CAP.findall(text))) / 3,
            float(bool(re.search(r"\d", text))), float(bool(TIMEWORD.search(text))), min(2, sum(t in EMOTION for t in toks)) / 2,
            float(bool(PLAN.search(text))), float(bool(caption)), float(bool(re.search(r"\b(i|my|we|our)\b", text, re.I))),
            float(bool(ACK.search(text))), float(prev_text.strip().endswith("?")), pos / max(1, n_in_ep)]


CRITIC_WEIGHTS: list[float] | None = None


@dataclass
class Engram:
    id: str
    episode: int
    pos: int
    store: str            # cortical store: temporal | occipital | frontal
    salience: float
    gated: bool           # True if the thalamic gate gave it no index cues
    speaker: str = ""
    has_time: bool = False
    day: int = 0            # episode date as an ordinal
    event: tuple | None = None  # resolved date range of what the turn mentions
    importance: float = 0.5     # learned: how likely this memory is to be needed later
    intrinsic: float = 0.0      # fixed: emotion score, kept separate as in LeCun's intrinsic cost


@dataclass
class Config:
    # which layers are on (for ablation)
    gate: bool = True
    router: bool = True
    shortlist: bool = True
    salience: bool = True
    spread: bool = True
    cingulate: bool = True
    semantic: bool = True     # sparse fly-hash cue channel
    # dials, tuned on the development conversations only
    episodes: int = 8
    k_min: int = 3
    k_max: int = 8
    spread_each: int = 1
    margin: float = 0.25
    w_sem: float = 1.0
    w_route: float = 0.25
    w_sal: float = 0.15
    w_entity: float = 0.5
    bm25_k1: float = 1.2
    replay: int = 0            # second pass: cues taken from the top matches
    replay_terms: int = 6
    replay_weight: float = 0.5
    spread_back: int = 0       # also pull the turn before
    time_boost: float = 0.0    # for 'when' questions, favour turns that mention time
    widen_set: int = 0         # extra episodes for questions asking about several things
    w_imp: float = 0.0         # weight of learned importance in ranking
    w_range: float = 0.0       # boost when a memory falls in the time the question asks about
    world: bool = False        # use the world-state table
    direct_df: int = 0         # flat route: a cue this rare goes straight to its pointers, past the shortlist
    cue_slots: int = 0         # each of the rarest cues reserves this many places in the result
    world_boost: float = 1.5
    world_gate: bool = True    # only when the question asks about now or a past period
    stale_penalty: float = 0.5
    ep_cap: int = 2000         # each word keeps pointers to at most this many episodes, strongest first
    bm25_b: float = 0.5


class FlyHash:
    """Dasgupta et al. 2017: random sparse expansion then winner-take-all.
    Turns a dense vector into a short list of active units."""
    def __init__(self, dim: int, m: int = 4096, density: float = 0.1, active: int = 32, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.proj = (rng.random((m, dim)) < density).astype(np.float32)
        self.active = active

    def encode(self, v: np.ndarray) -> np.ndarray:
        a = v @ self.proj.T
        return np.argpartition(a, -self.active, axis=-1)[..., -self.active:].astype(np.uint16)


class Engrams:
    def __init__(self, encoder=None, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.encoder = encoder
        self.fly = None
        # cortical stores: full content lives here and nowhere else
        self.cortex: dict[str, dict[str, str]] = {"temporal": {}, "occipital": {}, "frontal": {}}
        # hippocampus: light records and inverted indexes
        self.engrams: list[Engram] = []
        self.by_id: dict[str, int] = {}
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.entity_post: dict[str, list[int]] = defaultdict(list)
        self.sem_post: dict[int, list[int]] = defaultdict(list)
        self.doclen: list[int] = []
        # episodes
        self.ep_dates: list[str] = []
        self.ep_members: list[list[int]] = []
        self.ep_post: dict[str, list[tuple[int, int]]] = {}
        self.ep_sem: dict[int, list[int]] = {}
        self.ep_len: list[int] = []
        # basal ganglia
        self.habit: dict[str, list[int]] = {}
        self.ep_day: list = []
        self.world_state: dict = defaultdict(list)   # (entity, slot) -> pointers, oldest first
        self.forgotten: set[int] = set()
        self.prev_text: dict[int, str] = {}
        self.names: set[str] = set()
        self.idf: dict[str, float] = {}
        self.avg = 1.0
        self.stats = defaultdict(int)

    # ---------- write path ----------
    def thalamic_gate(self, tokens: list[str], ents: list[str]) -> bool:
        """True = pass. A turn with too few content words earns no cues. It is still kept in cortex
        and can come back through spreading from its neighbour."""
        return (not self.cfg.gate) or len(set(tokens)) >= 3 or bool(ents)

    @staticmethod
    def tag_salience(text: str, tokens_raw: list[str]) -> float:
        hits = sum(t in EMOTION for t in tokens_raw)
        return min(1.0, 0.35 * hits + 0.1 * text.count("!"))

    @staticmethod
    def classify_store(text: str, caption: str) -> str:
        if caption: return "occipital"
        if PLAN.search(text): return "frontal"
        return "temporal"

    def add_episode(self, date: str) -> int:
        self.ep_dates.append(date)
        self.ep_day.append(parse_session_date(date))
        self.ep_members.append([])
        return len(self.ep_dates) - 1

    def add_turn(self, id: str, episode: int, speaker: str, text: str, caption: str = "", vec=None):
        body = f"{speaker}: {text}" + (f" [image: {caption}]" if caption else "")
        store = self.classify_store(text, caption)
        self.cortex[store][id] = body                        # storage, cheap and unprocessed
        raw = TOKEN.findall(body.lower())
        toks = words(body)
        ents = [e.lower() for e in CAP.findall(text)]
        self.names.add(speaker.lower())
        n = len(self.engrams)
        passed = self.thalamic_gate(toks, ents)
        self.engrams.append(Engram(id, episode, len(self.ep_members[episode]), store, self.tag_salience(text, raw), not passed, speaker.lower()))
        rec = self.engrams[-1]
        rec.has_time = bool(TIMEWORD.search(text))
        d = self.ep_day[episode]
        rec.day = d.toordinal() if d else 0
        rec.event = mention_range(text, d)
        rec.intrinsic = rec.salience
        members = self.ep_members[episode]
        prev = self.cortex[self.engrams[members[-1]].store][self.engrams[members[-1]].id] if members else ""
        rec.feats_prev = prev.split(": ", 1)[-1]
        rec.feats_text, rec.feats_cap = text, caption
        for slot, r in SLOT_WRITE.items():
            if r.search(text):
                who = {speaker.lower()} if re.search(r"\b(i|we|i'm|i've|we've)\b", text, re.I) else {x.lower() for x in CAP.findall(text)}
                for w in who: self.world_state[(w, slot)].append(n)
        self.by_id[id] = n
        self.ep_members[episode].append(n)
        self.doclen.append(len(toks) if passed else 0)
        if not passed:
            self.stats["gated"] += 1
            return
        counts = defaultdict(int)
        for t in toks: counts[t] += 1
        for t, c in counts.items(): self.postings[t].append((n, c))
        for e in set(ents + [speaker.lower()]): self.entity_post[e].append(n)
        if vec is not None:
            if self.fly is None: self.fly = FlyHash(len(vec))
            for u in self.fly.encode(vec).tolist(): self.sem_post[u].append(n)

    # ---------- sleep ----------
    def sleep(self):
        """Offline consolidation. No LLM in this version: builds term statistics and episode gists.
        Costly work belongs here so that reads stay cheap."""
        N = max(1, len(self.engrams))
        if CRITIC_WEIGHTS is not None:
            for ep_m in self.ep_members:
                for pos, n in enumerate(ep_m):
                    e = self.engrams[n]
                    f = importance_features(e.feats_text, e.feats_cap, e.feats_prev, pos, len(ep_m))
                    e.importance = 1 / (1 + math.exp(-sum(a * b for a, b in zip(CRITIC_WEIGHTS, f))))
        self.idf = {t: math.log(1 + (N - len(p) + .5) / (len(p) + .5)) for t, p in self.postings.items()}
        self.avg = max(1.0, sum(self.doclen) / N)
        ep_tf = defaultdict(lambda: defaultdict(int))
        for t, p in self.postings.items():
            for n, c in p: ep_tf[t][self.engrams[n].episode] += c
        cap = self.cfg.ep_cap
        self.ep_post = {t: (list(d.items()) if len(d) <= cap else sorted(d.items(), key=lambda x: -x[1])[:cap]) for t, d in ep_tf.items()}
        self.by_ep = {}
        for t, p in self.postings.items():
            g = defaultdict(list)
            for n, c in p: g[self.engrams[n].episode].append((n, c))
            self.by_ep[t] = dict(g)
        self.ep_len = [sum(self.doclen[n] for n in m) for m in self.ep_members]
        sem = defaultdict(set)
        for u, p in self.sem_post.items():
            for n in p: sem[u].add(self.engrams[n].episode)
        self.ep_sem = {u: sorted(s) for u, s in sem.items()}

    # ---------- read path ----------
    def route(self, q: str) -> set[str]:
        return {k for k, r in ROUTES.items() if r.search(q)} if self.cfg.router else set()

    def retrieve(self, q: str, qvec=None, use_habit: bool = True, trace: dict | None = None) -> list[int]:
        c = self.cfg
        key = " ".join(sorted(set(words(q))))
        if use_habit and key in self.habit:                  # stage 0
            if trace is not None: trace["stage"] = 0
            return self.habit[key]
        routes = self.route(q)                               # stage 1
        qt = list(dict.fromkeys(words(q)))
        ents = {e.lower() for e in CAP.findall(q)}
        qsem = self.fly.encode(qvec).tolist() if (c.semantic and qvec is not None and self.fly is not None) else []

        allowed = None
        touched = 0
        if c.shortlist and self.ep_dates:                    # stage 2
            scores = defaultdict(float)
            k1, b = c.bm25_k1, 0.75
            avg_ep = max(1.0, sum(self.ep_len) / len(self.ep_len))
            for t in qt:
                idf = self.idf.get(t)
                if idf is None: continue
                for ep, tf in self.ep_post[t]:
                    scores[ep] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.ep_len[ep] / avg_ep))
                    touched += 1
            if qsem and c.w_sem:
                top_lex = max(scores.values(), default=1.0) or 1.0
                sem = defaultdict(int)
                for u in qsem:
                    for ep in self.ep_sem.get(u, ()): sem[ep] += 1; touched += 1
                peak = max(sem.values(), default=1)
                for ep, s in sem.items(): scores[ep] += c.w_sem * 0.3 * top_lex * s / peak
            allowed = set(sorted(scores, key=scores.get, reverse=True)[: c.episodes + (c.widen_set if SETQ.search(q) else 0)])

        scores = defaultdict(float)                          # stage 3
        k1, b = c.bm25_k1, c.bm25_b
        for t in qt:
            idf = self.idf.get(t)
            if idf is None: continue
            mult = c.w_entity + 1 if t in ents or t in self.names else 1
            for n, tf in self._entries(t, allowed):
                touched += 1
                scores[n] += mult * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.doclen[n] / self.avg))
        if qsem and c.w_sem:
            top_lex = max(scores.values(), default=1.0) or 1.0
            sem = defaultdict(int)
            for u in qsem:
                for n in self.sem_post.get(u, ()):
                    touched += 1
                    if allowed is None or self.engrams[n].episode in allowed: sem[n] += 1
            for n, s in sem.items(): scores[n] += c.w_sem * top_lex * s / len(qsem)
        if not scores:
            if trace is not None: trace.update(stage=3, touched=touched, k=0, escalated=False)
            return []
        if c.replay and scores:                              # replay: completed matches supply a second cue
            first = sorted(scores, key=scores.get, reverse=True)[: c.replay]
            cue = defaultdict(float)
            for n in first:
                e = self.engrams[n]
                for t in words(self.cortex[e.store][e.id]):
                    if t not in qt and t in self.idf: cue[t] = self.idf[t]
            extra = sorted(cue, key=cue.get, reverse=True)[: c.replay_terms]
            peak = scores[first[0]] or 1.0
            acc = defaultdict(float)
            for t in extra:
                idf = self.idf[t]
                for n, tf in self._entries(t, allowed):
                    touched += 1
                    acc[n] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.doclen[n] / self.avg))
            if acc:
                top2 = max(acc.values())
                for n, v in acc.items(): scores[n] += c.replay_weight * peak * v / top2
        qrange = query_range(q) if c.w_range else None
        for n in list(scores):                               # store and salience weighting
            e = self.engrams[n]
            m = 1.0
            if "visual" in routes and e.store == "occipital": m += c.w_route
            if "plan" in routes and e.store == "frontal": m += c.w_route
            if c.salience and "emotion" in routes: m += c.w_sal * 2 * e.salience
            elif c.salience: m += c.w_sal * e.salience
            if c.time_boost and "time" in routes and e.has_time: m += c.time_boost
            if c.w_imp: m += c.w_imp * (e.importance - 0.5)
            if qrange:
                in_ep = qrange[0] <= e.day <= qrange[1]
                in_ev = e.event is not None and e.event[0] <= qrange[1] and e.event[1] >= qrange[0]
                if in_ep or in_ev: m += c.w_range
            scores[n] *= m
        if c.world and (not c.world_gate or WORLD_GATE.search(q)):
            lowered = {x.lower() for x in CAP.findall(q)}
            for slot, r in SLOT_READ.items():
                if not r.search(q): continue
                for who in lowered:
                    chain = [n for n in self.world_state.get((who, slot), ()) if n not in self.forgotten]
                    if not chain: continue
                    top = max(scores.values(), default=1.0) or 1.0
                    if qrange or PAST_Q.search(q):
                        qr = query_range(q)
                        if qr:
                            hit = [n for n in chain if self.engrams[n].day <= qr[1]]
                            pick = hit[-1] if hit else chain[0]
                        else:
                            pick = chain[0] if re.search(r"\b(first|originally|at first)\b", q, re.I) else (chain[-2] if len(chain) > 1 else chain[0])
                    else:
                        pick = chain[-1]
                        for old in chain[:-1]:
                            if old in scores: scores[old] *= c.stale_penalty
                    scores[pick] = top * c.world_boost
        ranked = sorted(scores, key=scores.get, reverse=True)

        routed = []
        if c.direct_df:
            for t in qt:
                p = self.postings.get(t)
                if p and len(p) <= c.direct_df: routed += [n for n, _ in p]
        if c.cue_slots:
            rare = sorted((t for t in qt if t in self.idf), key=lambda t: -self.idf[t])[:3]
            for t in rare:
                hits = sorted(self._entries(t, allowed), key=lambda x: -x[1] / max(1, self.doclen[x[0]]))
                routed += [n for n, _ in hits[: c.cue_slots]]
        if routed: ranked = list(dict.fromkeys(routed + ranked))
        k = c.k_max                                          # stage 5: decide how much to pull
        escalated = False
        if c.cingulate:
            top = scores[ranked[0]]
            nxt = scores[ranked[min(c.k_min, len(ranked) - 1)]]
            clear = top > 0 and (top - nxt) / top >= c.margin
            k = c.k_min if clear else c.k_max
            escalated = not clear
        out, seen = [], set()
        for n in ranked[:k]:                                 # stage 4: spreading
            if n not in seen: out.append(n); seen.add(n)
            if c.spread and c.spread_each:
                e = self.engrams[n]
                m = self.ep_members[e.episode]
                for d in range(1, c.spread_each + 1):
                    if e.pos + d < len(m) and m[e.pos + d] not in seen:
                        out.append(m[e.pos + d]); seen.add(m[e.pos + d])
            if c.spread and c.spread_back:
                e = self.engrams[n]
                m = self.ep_members[e.episode]
                if e.pos >= 1 and m[e.pos - 1] not in seen:
                    out.append(m[e.pos - 1]); seen.add(m[e.pos - 1])
        if trace is not None: trace.update(stage=3, touched=touched, k=k, escalated=escalated, routes=routes)
        return out

    def promote(self, full, n: int):
        """Refresh on recall: bring a dormant memory back into this awake index, from the full index."""
        if n not in self.forgotten: return
        self.forgotten.discard(n)
        ep = self.engrams[n].episode
        for t, d in full.by_ep.items():
            for x in d.get(ep, ()):
                if x[0] == n:
                    self.postings[t].append(x)
                    self.by_ep.setdefault(t, {}).setdefault(ep, []).append(x)
                    if t not in self.idf: self.idf[t] = full.idf[t]
                    g = dict(self.ep_post.get(t, []))
                    g[ep] = g.get(ep, 0) + x[1]
                    self.ep_post[t] = list(g.items())
                    break

    def forget(self, keep: set[int]):
        """Remove index entries for everything not kept. Content stays in the store."""
        self.forgotten = set(range(len(self.engrams))) - keep
        self.postings = defaultdict(list, {t: [x for x in p if x[0] in keep] for t, p in self.postings.items()})
        self.postings = defaultdict(list, {t: p for t, p in self.postings.items() if p})
        self.sem_post = defaultdict(list)
        self.sleep_stats_only = True
        self.sleep()

    def _entries(self, t, allowed):
        if allowed is None:
            yield from self.postings[t]
            return
        d = self.by_ep.get(t, {})
        for ep in allowed:
            yield from d.get(ep, ())

    def learn_habit(self, q: str, result: list[int]):
        self.habit[" ".join(sorted(set(words(q))))] = result

    def reinstate(self, ns: list[int]) -> str:
        """Fetch only the selected pieces from cortex. Each is stamped with its episode's date."""
        lines = []
        for n in ns:
            e = self.engrams[n]
            lines.append(f"[{self.ep_dates[e.episode]}] {self.cortex[e.store][e.id]}")
        return "\n".join(lines)

    # ---------- accounting ----------
    def sizes(self) -> dict[str, int]:
        """Rough serialised sizes in bytes: 4 bytes per integer, plus term text once."""
        store = {k: sum(len(v.encode()) for v in d.values()) for k, d in self.cortex.items()}
        hip = 12 * len(self.engrams)                                  # episode, position, store+flags
        hip += sum(8 * len(p) for p in self.postings.values()) + sum(len(t) + 4 for t in self.postings)
        ents = sum(4 * len(p) for p in self.entity_post.values())
        sem = sum(4 * len(p) for p in self.sem_post.values())
        gist = sum(8 * len(p) for p in self.ep_post.values()) + sum(4 * len(p) for p in self.ep_sem.values()) + 4 * len(self.ep_len)
        return dict(cortex_total=sum(store.values()), **{f"cortex_{k}": v for k, v in store.items()},
                    hippocampal_lexical=hip, entity_cues=ents, sparse_semantic=sem, episode_gists=gist)
