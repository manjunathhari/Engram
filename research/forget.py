import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, pickle, random
import numpy as np
import engram
from bench import Conv, Flat
from engram import Config
import final_lib as L
convs = pickle.load(open("convs.pkl", "rb")); test = convs[5:]
engram.CRITIC_WEIGHTS = list(json.load(open("results_v3.json"))["critic"]["weights"].values())
cfg = Config(**json.load(open("results_v3_final.json"))["config"])
def run(frac, how, seed=1):
    out = []; rng = random.Random(seed)
    for c in test:
        m = L.build_with_notes(c, cfg, None); idx = list(range(len(m.engrams)))
        key = {"critic": lambda n: m.engrams[n].importance, "emotion": lambda n: (m.engrams[n].intrinsic, rng.random()), "random": lambda n: rng.random(), "newest": lambda n: m.engrams[n].day, "longest": lambda n: m.doclen[n]}[how]
        m.forget(set(sorted(idx, key=key, reverse=True)[: int(frac * len(idx))]))
        for q, qv in zip(c.qs, c.qvecs):
            got = {m.engrams[n].id for n in m.retrieve(q["q"], qv, use_habit=False)}
            out.append(len(got & q["ev"]) / len(q["ev"]))
    return out
R = {}
for f in (1.0, .5, .25):
    r = {h: run(f, h) for h in ("critic", "emotion", "random", "newest", "longest")}
    R[str(f)] = {h: float(np.mean(v)) for h, v in r.items()}
    R[str(f)]["critic_vs_random"] = L.boot(r["critic"], r["random"]); R[str(f)]["critic_vs_longest"] = L.boot(r["critic"], r["longest"]); R[str(f)]["emotion_vs_random"] = L.boot(r["emotion"], r["random"])
    print(f, R[str(f)], flush=True)
json.dump(R, open("results_forget.json", "w"), indent=1)
