# GPT-4o as the P2.6 writer: rule fixed before the run (7 November 2026)

Same 200 sentences, blind labels, prompts, parser, validator and scorer as the Gemma 3 4B and 12B runs (validator unchanged since 77fa6d4).
Only the model changes: gpt-4o-2024-08-06 through the OpenAI API (kit_tools/p26_openai.py), temperature 0, seed 7, 400 reply tokens.
In the forced form OpenAI's structured outputs (strict JSON schema) hold the reply to writer_schema, as Ollama's "format" did for Gemma;
the only change to the schema is closing each object to extra fields, which strict mode requires.

Primary: the forced form. GPT-4o passes only if both hold, scored once with kit_tools/p26_score.py:
1. Of the facts the validator accepts, at least 3 right for every 1 wrong (75% or more right by head word).
2. It finds at least 19 of the 56 labelled facts (one third).

The free form is run too and reported for the record; it does not decide.
This set has been seen in scoring the 4B and 12B runs, but nothing in the prompts, parser, validator or scorer was changed after seeing it.

What each result means for the filing plan, decided now:
- Passes: a strong writer can file into the current design; next, wire the chain behind a switch and test it on answers.
- Finds at least 19 but fails condition 1: the writer is good enough to find facts, and the validator or the relation list is letting wrong ones in; fix the checking and the relations before wiring.
- Finds fewer than 19: even a strong writer cannot file well into 30 sentence-by-sentence relations; change the filing format (whole chats, fewer relations or free-text facts with fields) before anything else.
In every case the validator's handling of right facts (how many right facts it holds back as uncertain) is reported too.
