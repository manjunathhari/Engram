# Gemma 3 12B as the P2.6 writer: rule fixed before the run (7 November 2026)

Same 200 sentences, blind labels, prompts, parser, validator and scorer as the Gemma 3 4B run (commit 77fa6d4 code; validator unchanged).
Only the model changes: hf.co/bartowski/google_gemma-3-12b-it-GGUF:Q4_K_M in place of the 4B.

Primary: the forced form. 12B passes only if both hold, scored once with kit_tools/p26_score.py:
1. Of the facts the validator accepts, at least 3 right for every 1 wrong (75% or more right by head word).
2. It finds at least 19 of the 56 labelled facts (one third).

If it fails, P2.6 is parked: extraction stays off and unmerged.
The free form is run too and reported for the record; it does not decide.
This set has been seen in scoring the 4B run, but nothing in the code was changed after seeing it.
