"""Latency as memory grows. Text is resampled from real LoCoMo turns; this measures speed and memory only, not recall."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import pickle, time, json, sys, resource
import numpy as np
from bench import Conv, norm
from engram import Engrams, Config, FlyHash, words

convs = pickle.load(open("convs.pkl", "rb"))
pool = [(t, v) for c in convs for t, v in zip(c.turns, c.vecs)]
qs = [(q["q"], qv) for c in convs for q, qv in zip(c.qs, c.qvecs)]
rng = np.random.default_rng(3)
cfg = Config()
cfg = Config(**json.load(open("results_v2.json"))["config"])
out = []
for N in (6000, 30000, 100000, 300000):
    t0 = time.perf_counter()
    m = Engrams(cfg=cfg)
    idx = rng.integers(0, len(pool), N)
    vecs = np.empty((N, 256), dtype=np.float32)
    ep = -1
    fly = FlyHash(256); m.fly = fly
    B = 20000
    for s in range(0, N, B):
        chunk = idx[s:s + B]
        V = np.stack([pool[i][1] for i in chunk]) + rng.normal(0, .02, (len(chunk), 256)).astype(np.float32)
        V = norm(V); vecs[s:s + len(chunk)] = V
        codes = fly.encode(V)
        for j, i in enumerate(chunk):
            n = s + j
            if n % 30 == 0: ep = m.add_episode(pool[i][0]["date"])
            t = pool[i][0]
            # bypass the per-turn hash, the code is already computed in batch
            m_fly, m.fly = m.fly, None
            m.add_turn(f"x{n}", ep, t["speaker"], t["text"], t["cap"], None)
            m.fly = m_fly
            if cfg.semantic and not m.engrams[n].gated:
                for u in codes[j].tolist(): m.sem_post[u].append(n)
    t_w = time.perf_counter() - t0
    t0 = time.perf_counter(); m.sleep(); t_s = time.perf_counter() - t0
    sample = rng.integers(0, len(qs), 100)
    te, td, tou, tb = [], [], [], []
    dl = np.array(m.doclen, dtype=np.float32); avg = max(1.0, dl.mean())
    for i in sample:
        q, qv = qs[i]; tr = {}
        t0 = time.perf_counter(); ns = m.retrieve(q, qv, use_habit=False, trace=tr); m.reinstate(ns); te.append((time.perf_counter() - t0) * 1e3); tou.append(tr.get("touched", 0))
        t0 = time.perf_counter(); r = vecs @ qv; np.argpartition(-r, 20)[:20]; td.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter(); s_ = np.zeros(N, dtype=np.float32)
        for t in set(words(q)):
            if t in m.idf:
                for n, tf in m.postings[t]: s_[n] += m.idf[t] * tf * 2.2 / (tf + 1.2 * (.25 + .75 * dl[n] / avg))
        np.argpartition(-s_, 20)[:20]; tb.append((time.perf_counter() - t0) * 1e3)
    sz = m.sizes(); idx_b = sz["hippocampal_lexical"] + sz["entity_cues"] + sz["sparse_semantic"] + sz["episode_gists"]
    row = dict(N=N, write_s=t_w, sleep_s=t_s, engram_p50=float(np.percentile(te, 50)), engram_p95=float(np.percentile(te, 95)), flat_dense_p50=float(np.percentile(td, 50)), flat_dense_p95=float(np.percentile(td, 95)), flat_bm25_p50=float(np.percentile(tb, 50)), flat_bm25_p95=float(np.percentile(tb, 95)),
               touched=float(np.mean(tou)), index_bytes=idx_b, cortex_bytes=sz["cortex_total"], dense_bytes=int(vecs.nbytes), rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    print(row, flush=True); out.append(row)
    json.dump(out, open("scale.json", "w"), indent=1)
    del m, vecs
    if row["rss_gb"] > 5.5: break
