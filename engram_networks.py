"""Saved channel networks for engram_parallel: plain numbers with a checksum, never a pickle, so loading cannot run code."""
import numpy as np

from engram_parallel import CHANNELS

NETWORKS_FILE = "engram_networks.npz"


class _Scale:
    def __init__(self, mean, scale): self.mean_, self.scale_ = mean, scale
    def transform(self, x): return (np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_


class _Net:
    """The same sums as scikit-learn's MLPClassifier.predict_proba for these networks: relu layers, then a logistic output."""
    def __init__(self, coefs, intercepts): self.coefs_, self.intercepts_ = coefs, intercepts
    def predict_proba(self, x):
        h = np.asarray(x, dtype=np.float64)
        for w, b in zip(self.coefs_[:-1], self.intercepts_[:-1]): h = np.maximum(h @ w + b, 0)
        p = 1 / (1 + np.exp(-(h @ self.coefs_[-1] + self.intercepts_[-1])))
        return np.hstack([1 - p, p])


def save_networks(models, path, note=""):
    """Writes trained networks as plain arrays, with a description and a checksum of the numbers."""
    import hashlib, json
    arrays = {}
    for i, m in enumerate(models):
        for k in CHANNELS:
            scale, net = m[k]
            if net.activation != "relu" or net.out_activation_ != "logistic": raise ValueError("only relu networks with a logistic output can be saved")
            arrays[f"{i}.{k}.mean"], arrays[f"{i}.{k}.scale"] = scale.mean_, scale.scale_
            for j, (w, b) in enumerate(zip(net.coefs_, net.intercepts_)): arrays[f"{i}.{k}.w{j}"], arrays[f"{i}.{k}.b{j}"] = w, b
    digest = hashlib.sha256(b"".join(arrays[n].astype("<f8").tobytes() for n in sorted(arrays))).hexdigest()
    arrays["about"] = np.array(json.dumps(dict(starts=len(models), channels=CHANNELS, layers=len(models[0][CHANNELS[0]][1].coefs_), sha256=digest, note=note)))
    np.savez(path, **arrays)
    return digest


def load_networks(path):
    """Reads networks written by save_networks, checking the numbers against their checksum."""
    import hashlib, json
    with np.load(path, allow_pickle=False) as z:
        about = json.loads(str(z["about"]))
        arrays = {n: z[n] for n in z.files if n != "about"}
    if hashlib.sha256(b"".join(arrays[n].astype("<f8").tobytes() for n in sorted(arrays))).hexdigest() != about["sha256"]: raise ValueError("the saved networks do not match their checksum")
    return [{k: (_Scale(arrays[f"{i}.{k}.mean"], arrays[f"{i}.{k}.scale"]), _Net([arrays[f"{i}.{k}.w{j}"] for j in range(about["layers"])], [arrays[f"{i}.{k}.b{j}"] for j in range(about["layers"])])) for k in about["channels"]} for i in range(about["starts"])]
