"""open_memory gives the design that passed both tests; the saved networks are plain numbers checked by a checksum."""
import os
import shutil

import numpy as np
import pytest

import engram_chat as E
import engram_parallel as P
from conftest import ROOT, fill, load
from engram_memory import open_memory, tested_config

NETS = os.path.join(ROOT, P.NETWORKS_FILE)


def test_the_shipped_networks_load_and_have_the_tested_shape():
    nets = P.load_networks(NETS)
    assert len(nets) == 3 and all(set(m) == set(P.CHANNELS) for m in nets)
    assert [w.shape for w in nets[0]["time"][1].coefs_] == [(11, 32), (32, 16), (16, 1)]


def test_plain_arithmetic_matches_scikit_learn():
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 6)); y = (X[:, 0] + X[:, 1] ** 2 > 1).astype(int)
    scale = StandardScaler().fit(X)
    net = MLPClassifier((32, 16), max_iter=300, random_state=1).fit(scale.transform(X), y)
    mine = P._Net(net.coefs_, net.intercepts_), P._Scale(scale.mean_, scale.scale_)
    probe = rng.normal(size=(200, 6)) * 4
    assert np.abs(mine[0].predict_proba(mine[1].transform(probe)) - net.predict_proba(scale.transform(probe))).max() < 1e-12


def test_changed_numbers_are_refused(tmp_path):
    with np.load(NETS, allow_pickle=False) as z: arrays = {n: z[n] for n in z.files}
    arrays["0.time.w0"] = arrays["0.time.w0"] + 1e-6
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="checksum"): P.load_networks(tmp_path / "bad.npz")


def test_a_file_needing_pickle_is_refused(tmp_path):
    np.savez(tmp_path / "pickled.npz", about=np.array([{"code": 1}], dtype=object))
    with pytest.raises(ValueError): P.load_networks(tmp_path / "pickled.npz")


def test_the_light_design_needs_no_extra_libraries(path):
    m = open_memory(path, design="light")
    assert m.search is None and m.cfg == tested_config()
    m.close()
    with pytest.raises(ValueError): open_memory(path, design="banana")


@pytest.mark.slow
def test_the_tested_design_finds_the_answer(tmp_path):
    data = load("basic_conversation.json")
    m = open_memory(str(tmp_path / "t.db"))
    assert m.search is not None and m.search.mode == "network" and not m.use_facts
    ids = fill(m, data)
    hits = [ids[q["expect"]] in set(m.find(q["q"])[0]) for q in data["questions"]]
    assert sum(hits) >= len(hits) - 1, hits
    m.close()
