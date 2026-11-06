"""The one place to open an Engram memory with the design that passed both tests.

    from engram_memory import open_memory
    memory = open_memory("my_memory.db")

"tested" (the default) is the search that passed the LoCoMo final test and the LongMemEval test: four small channel networks
(time, place, person, context), trained only on LoCoMo conversations 0 to 4, read from engram_networks.npz, with the settings in
final_config.json. Noted facts are added only when a helper model is given, as in the LoCoMo test.

"light" is the older search with no extra libraries. It does worse (41.7% against 47.4% on LongMemEval) and is kept for small machines.

ChatMemory itself is unchanged, so existing code and the regression tests behave exactly as before.
"""
import json
import os

import engram_chat as E
from engram import Config

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGNS = ("tested", "light")


def tested_config():
    return Config(**json.load(open(os.path.join(HERE, "final_config.json"))))


def open_memory(path, design="tested", helper=None, helper_name="", tools=None, **kw):
    if design not in DESIGNS: raise ValueError(f"design must be one of {DESIGNS}")
    if design == "light":
        return E.ChatMemory(path, cfg=tested_config(), **kw)
    try:
        import engram_networks as N
        import engram_parallel as P
        search = P.ChannelSearch("network", model=N.load_networks(os.path.join(HERE, N.NETWORKS_FILE)), tools=tools)
    except ImportError as e:
        raise ImportError(f"the tested design needs numpy, wordllama and spacy with en_core_web_sm ({e}). Install them, or open_memory(path, design='light').") from None
    return E.ChatMemory(path, cfg=tested_config(), search=search, helper=helper, helper_name=helper_name, use_facts=helper is not None, **kw)
