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

    python3 vps_kit.py check --model gemma3:4b --tag g3
    ./sync_results.sh
    nohup python3 vps_kit.py answers --model gemma3:4b --tag g3 --sample 120 > answers.log 2>&1 &

To send results every 15 minutes while it runs:

    (crontab -l 2>/dev/null; echo "*/15 * * * * $PWD/sync_results.sh >> $PWD/sync.log 2>&1") | crontab -

Question types are interleaved, so a run can stop at any time and still leave a balanced set. Repeat the same command to continue.

Keep Ollama's port (11434) closed to the internet.

## Data and licence

Tests use [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., 2024), licensed CC BY-NC 4.0. The data is not stored here, but `results/` contains questions and short excerpts from it, under that licence. This repository's own code has no licence yet, so its rights stay with the author unless one is added.
