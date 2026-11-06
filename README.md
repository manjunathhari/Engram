# Engram

A layered memory architecture modelled on how the brain separates storage from retrieval. Content stays in a permanent record, a small index holds pointers, and reads call no model until the final answer.

This is a research prototype. Do not put patient or other personal information in it: it has no access control or deletion.

## Layout

- `engram.py`: the memory system.
- `vps_kit.py`, `bench.py`, `final_config.json`: the test kit, run on the server next to Gemma.
- `sync_results.sh`: sends `results/` to GitHub and fetches kit updates.
- `results/`: test outputs, written by the server.
- `research/`: earlier benchmark scripts and their outputs. Run them from a folder holding `bench.py`, `engram.py` and the LoCoMo data.
- `kit_tools/`: a stand-in model for checking the kit, and the blind scoring sheet.

## Set up on the server

    sudo apt install -y python3-numpy git
    ssh-keygen -t ed25519 -f ~/.ssh/engram_deploy -N "" -C engram-vps
    cat ~/.ssh/engram_deploy.pub

Add that public key at GitHub: repository Settings, Deploy keys, Add deploy key, and tick "Allow write access". The key reaches this repository only.

    GIT_SSH_COMMAND="ssh -i ~/.ssh/engram_deploy" git clone git@github.com:manjunathhari/Engram.git
    cd Engram
    git config core.sshCommand "ssh -i ~/.ssh/engram_deploy"
    git clone --depth 1 https://github.com/snap-research/locomo.git

## Run the tests

If Ollama runs on the server itself, leave out `--url`. If it runs in Docker, find its internal address first, using your container's name:

    OLLAMA=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' YOUR_OLLAMA_CONTAINER)

Then:

    python3 vps_kit.py check --url http://$OLLAMA:11434/v1 --model gemma3:4b --tag g3
    ./sync_results.sh
    nohup python3 vps_kit.py answers --url http://$OLLAMA:11434/v1 --model gemma3:4b --tag g3 --sample 120 > answers.log 2>&1 &

To send results every 15 minutes while it runs:

    (crontab -l 2>/dev/null; echo "*/15 * * * * $PWD/sync_results.sh >> $PWD/sync.log 2>&1") | crontab -

Question types are interleaved, so a run can stop at any time and still leave a balanced set. Repeat the same command to continue.

Keep Ollama's port (11434) closed to the internet.

## Data and licence

Tests use [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., 2024), licensed CC BY-NC 4.0. The data is not stored here, but `results/` contains questions and short excerpts from it, under that licence. This repository's own code has no licence yet, so its rights stay with the author unless one is added.

## Regression tests

From the repository root:

    python3 -m pytest -m "not slow"

This takes under a second and needs no model. To also replay the benchmark and compare retrieval with the recorded baseline, clone `locomo` into the repository and install wordllama, spacy and scikit-learn, then run:

    python3 -m pytest

The baseline is in `tests/fixtures/retrieval_baseline.json`. If you deliberately change retrieval, rerun `python3 tests/record_baseline.py` and commit the new baseline with the change. Test data is made up or from the benchmark; never put personal data in the fixtures.

## Knowledge records

`engram_records.py` is a standard, typed record for knowledge, stored beside the sentences and always traceable to them. It is not used by search or answers, and nothing writes records automatically yet.

- **Four tables:** `entities`, `entity_aliases`, `knowledge_records`, `record_evidence`.
- **Names:** an entity is found only by its exact name or a stated alias, ignoring case and spacing. Similar names are never treated as the same. An alias needs a source sentence or must be marked manual.
- **Stated or inferred:** every fact must say which (`source_type`). There is no default, and answers about an entity show stated facts unless inferred ones are asked for. Accepting an inference does not turn it into a statement.
- **Status and checking are separate:** `status` is the record's life (active, superseded, disputed, retracted, needs_review, invalidated); `validation_status` is whether it was checked (pending, accepted, rejected).
- **Time:** `valid_from` and `valid_to` say when it was true; `recorded_at` says when Engram learned it.
- **Version check:** a database whose knowledge records are a different version is refused before anything in it changes.
- **Deletion:** if a source sentence is deleted, a fact with another supporting sentence is marked `needs_review`. A fact with no support left is `invalidated` and its content erased. Aliases from that sentence go too, and an entity nothing uses any more is removed.

## Entity resolution and fact validation

`engram_knowledge/` decides which entity a statement refers to and whether a candidate fact may be stored. Fixed rules decide; a language model can only propose. `engram_facts.py` is unchanged, and these records are not used by search or answers.

- **Accepted** goes to the knowledge store. **Uncertain** waits in the review queue, where it is not treated as knowledge. **Rejected** is not saved.
- **Names:** exact name or stated alias. "I" means the speaker. He, she and they are never guessed, and a similar name goes to review instead of being merged.
- **Checks:** the sentence holding the object must exist, contain its words, and use a word for that kind of fact. Possible negation, questions, hypotheticals, things said by the assistant, things reported by someone else and dates not in the source all go to review.
- **Deletion:** deleting a source sentence removes any review item that cites it.

These rules read words and cannot understand meaning, so accepted facts can still be wrong.

## Temporal state and knowledge retrieval

`StateEngine` works out what is true of an entity, now and at any past date, from the stored facts each time. The state is never stored, so the facts and their sentences stay the only source of truth. The old `world_state` in `engram.py` is unchanged.

- A later value from the same speaker ends the earlier one. A different value from another speaker is a conflict with both kept, until one speaker updates or a person settles it.
- A correction marks the old fact superseded and keeps its evidence. A plan never becomes a state.
- An unstated date stays unknown; the date the sentence was said is shown separately as `first_said`.
- If a newer statement is waiting for review, the answer says so, because it may be out of date.

`KnowledgeRetriever` sits beside ordinary memory. Its router sends a question to the records, to `ChatMemory.find`, or to both, using fixed word rules, and every returned fact carries its original sentences. `context_for` does not use it, so answers do not change. `kit_tools/knowledge_eval.py` holds the measurements.
