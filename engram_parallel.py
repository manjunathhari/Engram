"""Optional search layer for ChatMemory: person, place, time and context channels.

It only decides the order in which past sentences are read. The exact sentences stay in
ChatMemory's database, so this layer can be removed or rebuilt without losing anything.

Three modes:
  plain    word and meaning search
  cabinet  plain, plus fixed credit for matching person, place, date and episode; no training
  network  four small trained networks, one per channel, averaged

Needs: numpy, scikit-learn, wordllama, spacy with en_core_web_sm.
Train only on benchmark data with known answers. Never put personal data in a shared model.
"""
import math
import re
from collections import Counter

import numpy as np

from engram import CAP, ROUTES, TIMEWORD
from engram_chat import STOP, stem, with_dates, words

CHANNELS = ["time", "place", "person", "context"]
MONTHS = "january february march april may june july august september october november december".split()
DATEWORD = re.compile(r"\b(" + "|".join(MONTHS) + r"|20\d\d)\b", re.I)
WHERE = re.compile(r"\b(where|place|city|country|location|visit|visited|travel|travelled|trip|state|town|areas?)\b", re.I)
HOWMANY = re.compile(r"\bhow (many|much|long|often)\b", re.I)
NUMBER = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|twice|once)\b", re.I)
FEEL = set("happy excited proud love loved sad angry scared afraid worried stress lost died cried amazing great awesome wonderful terrible hate fun thrilled grateful".split())
PLACE_LABELS = {"GPE", "LOC", "FAC"}


def _load_tools():
    import spacy
    from wordllama import WordLlama
    return spacy.load("en_core_web_sm", disable=["lemmatizer"]), WordLlama.load()


class ChannelSearch:
    def __init__(self, mode="plain", model=None, tools=None):
        if mode not in ("plain", "cabinet", "network"):
            raise ValueError(f"unknown search mode: {mode}")
        if mode == "network" and model is None:
            raise ValueError("network mode needs a trained model")
        self.mode, self.model = mode, model
        self.nlp, self.encoder = tools or _load_tools()
        self.seen = {}  # sid -> what was read from that sentence, so a new chat does not reread old ones
        self.N = 0

    def _embed(self, texts):
        v = np.array(self.encoder.embed(texts), dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    # ---------- reading the record ----------
    def rebuild(self, db):
        """Called when chats close. Reads closed chats from ChatMemory's own database."""
        rows = db.execute("""SELECT s.sid, s.chat, s.speaker, s.date, s.text, s.caption, s.tokens FROM sentences s
                             JOIN chats c ON c.chat = s.chat WHERE c.closed = 1 ORDER BY s.sid""").fetchall()
        fresh = [r for r in rows if r[0] not in self.seen]
        if fresh:
            bodies = [f"{r[2]}: {r[4]}" + (f" [image: {r[5]}]" if r[5] else "") for r in fresh]
            vectors = self._embed(bodies)
            docs = self.nlp.pipe([r[4] for r in fresh], batch_size=64)
            for r, body, vec, doc in zip(fresh, bodies, vectors, docs):
                people, places = {stem(r[2].lower())}, set()
                for e in doc.ents:
                    name = e.text.lower().replace("the ", "").strip()
                    if e.label_ == "PERSON" and name:
                        people.add(stem(name.split()[0]))
                    if e.label_ in PLACE_LABELS and name:
                        places.add(name)
                self.seen[r[0]] = dict(body=body, vec=vec, people=people, places=places, w=words(body),
                                       dates={x.lower() for x in DATEWORD.findall(with_dates(r[4], r[3]))},
                                       said={x.lower() for x in DATEWORD.findall(r[3])})
        self.rows = rows
        self.N = N = len(rows)
        if not N:
            return
        info = [self.seen[r[0]] for r in rows]
        self.sids = [r[0] for r in rows]
        self.chats = np.array([r[1] for r in rows])
        self.vec = np.stack([x["vec"] for x in info])
        self.w = [x["w"] for x in info]
        self.people, self.places, self.dates, self.said = [x["people"] for x in info], [x["places"] for x in info], [x["dates"] for x in info], [x["said"] for x in info]
        self.speakers = [r[2] for r in rows]
        self.tok = [r[6] + 8 for r in rows]
        self.avg = sum(map(len, self.w)) / N
        self.postings = {}
        for i, w in enumerate(self.w):
            for k, v in Counter(w).items():
                self.postings.setdefault(k, []).append((i, v))
        seen, self.feel, self.nov, self.fact = Counter(), np.zeros(N), np.zeros(N), np.zeros(N)
        for i, r in enumerate(rows):
            ws = set(self.w[i])
            self.nov[i] = sum(seen[w] == 0 for w in ws) / max(1, len(ws))
            for w in ws:
                seen[w] += 1
            self.feel[i] = min(1, sum(y in FEEL for y in re.findall(r"[a-z']+", r[4].lower())) / 2)
            self.fact[i] = min(1, (len(CAP.findall(r[4])) + len(re.findall(r"\b\d", r[4]))) / 3)
        self.says_time = np.array([float(bool(TIMEWORD.search(r[4]))) for r in rows])
        self.has_place = np.array([float(bool(p)) for p in self.places])
        self.has_number = np.array([float(bool(NUMBER.search(r[4]))) for r in rows])
        order = {c: k for k, c in enumerate(sorted(set(self.chats.tolist())))}
        self.chat_pos = np.array([order[c] / max(1, len(order) - 1) for c in self.chats.tolist()])
        self.by_chat = {c: np.where(self.chats == c)[0] for c in order}
        self.around = [[j for j in range(max(0, i - 2), min(N, i + 3)) if self.chats[j] == self.chats[i]] for i in range(N)]
        self.window = np.stack([self.vec[a].mean(0) for a in self.around])
        self.window /= np.linalg.norm(self.window, axis=1, keepdims=True) + 1e-9
        self.chat_vec = np.zeros_like(self.vec)
        for c, k in self.by_chat.items():
            v = self.vec[k].mean(0)
            self.chat_vec[k] = v / (np.linalg.norm(v) + 1e-9)

    # ---------- reading the question ----------
    def _bm25(self, q):
        r = np.zeros(self.N)
        for w in set(words(q)):
            pl = self.postings.get(w)
            if not pl:
                continue
            idf = math.log(1 + (self.N - len(pl) + .5) / (len(pl) + .5))
            for i, tf in pl:
                r[i] += idf * tf * 2.2 / (tf + 1.2 * (.5 + .5 * len(self.w[i]) / self.avg))
        return r

    @staticmethod
    def _curve(q):
        n = len(set(words(q)))
        return (5., .7) if n <= 3 else (7., .6) if n <= 6 else (9., .5) if n <= 9 else (10., .5) if n <= 15 else (12., .5)

    def features(self, q):
        """One table of clues per channel, with the plain and cabinet scores."""
        N, qv = self.N, self._embed([q])[0]
        b, cos = self._bm25(q), self.vec @ qv
        mid, st = self._curve(q)
        bn = np.where(b > 0, 1 / (1 + np.exp(-st * (b - mid))), 0)
        h = cos + bn
        doc = self.nlp(q)
        qp = {stem(e.lower()) for e in CAP.findall(q) if e.lower() not in STOP}
        qpl = {e.text.lower().replace("the ", "").strip() for e in doc.ents if e.label_ in PLACE_LABELS}
        qd = {x.lower() for x in DATEWORD.findall(q)}
        when, where, many = float(bool(ROUTES["time"].search(q))), float(bool(WHERE.search(q))), float(bool(HOWMANY.search(q)))
        qw = set(words(q))
        speaker = np.array([float(stem(s.lower()) in qp) for s in self.speakers])
        pmatch = np.array([len(qp & p) / max(1, len(qp)) for p in self.people])
        pprev = np.array([pmatch[i - 1] if i and self.chats[i] == self.chats[i - 1] else 0 for i in range(N)])
        dm = np.array([float(bool(qd & x)) for x in self.dates])
        ds = np.array([float(bool(qd & x)) for x in self.said])
        plm = np.array([float(any(a in w or w in a for a in p for w in qpl)) for p in self.places])
        ps, pls, sb = np.zeros(N), np.zeros(N), np.zeros(N)
        for k in self.by_chat.values():
            ps[k], pls[k], sb[k] = pmatch[k].mean(), plm[k].max(), np.sort(h[k])[-3:].mean()
        wb = np.array([h[a].max() for a in self.around])
        wcos, scos = self.window @ qv, self.chat_vec @ qv
        over = np.array([len(qw & set(w)) / max(1, len(qw)) for w in self.w])
        length = np.array([len(r[4]) / 200 for r in self.rows])
        full = lambda x: np.full(N, float(x))
        base = [cos, bn]
        tables = dict(
            time=np.column_stack(base + [self.says_time, dm, ds, self.chat_pos, full(when), when * self.says_time, when * dm, full(bool(qd)), self.fact]),
            place=np.column_stack(base + [self.has_place, plm, pls, full(where), where * self.has_place, full(bool(qpl)), self.fact]),
            person=np.column_stack(base + [speaker, pmatch, pprev, ps, full(len(qp)), speaker * cos, pmatch * cos, full(len(qw) / 10)]),
            context=np.column_stack(base + [wcos, scos, sb, wb, over, self.feel, self.nov, length, full(len(qw) / 10), self.has_number, many * self.has_number, full(many)]),
        )
        rules = [when * self.says_time + dm + .5 * ds, plm + .5 * pls + where * self.has_place, pmatch + .5 * speaker, wcos + sb]
        cabinet = h + .25 * pmatch + .15 * speaker + .25 * dm + .1 * ds + .15 * when * self.says_time + .25 * plm + .1 * where * self.has_place + .3 * sb / (sb.max() + 1e-9)
        pool = set(np.argsort(-h)[:120].tolist())
        for r in rules:
            pool |= set(np.argsort(-(r + .2 * h))[:40].tolist())
        pool = np.array(sorted(pool))
        return dict(pool=pool, tables={k: v[pool] for k, v in tables.items()}, plain=h, cabinet=cabinet)

    # ---------- the search ----------
    def order(self, q, mode=None):
        """Every closed sentence id, best first."""
        if not self.N:
            return []
        mode, f = mode or self.mode, self.features(q)
        if mode in ("plain", "cabinet"):
            return [self.sids[i] for i in np.argsort(-f[mode])]
        s = np.mean([self.model[k][1].predict_proba(self.model[k][0].transform(f["tables"][k]))[:, 1] for k in CHANNELS], axis=0)
        first = f["pool"][np.argsort(-s)].tolist()
        have = set(first)
        return [self.sids[i] for i in first + [x for x in np.argsort(-f["plain"]).tolist() if x not in have]]


def train(examples, seed=1):
    """examples: (ChannelSearch, question, set of correct sentence ids). Returns one small network per channel."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    X, y = {k: [] for k in CHANNELS}, []
    for search, q, gold in examples:
        f = search.features(q)
        for k in CHANNELS:
            X[k].append(f["tables"][k])
        y.append(np.array([int(search.sids[i] in gold) for i in f["pool"]]))
    y = np.concatenate(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    neg = np.random.default_rng(seed).choice(neg, min(len(neg), len(pos) * 8), replace=False)
    pick = np.concatenate([pos, neg])
    model = {}
    for k in CHANNELS:
        table = np.vstack(X[k])[pick]
        scale = StandardScaler().fit(table)
        model[k] = (scale, MLPClassifier((32, 16), max_iter=300, random_state=seed, early_stopping=True).fit(scale.transform(table), y[pick]))
    return model
