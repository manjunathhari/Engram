# Engram as a research memory: filing cabinets against search (7 November 2026)

35 papers on robotic versus laparoscopic colorectal surgery (PubMed Central; `fetch_corpus.py` rebuilds them; not stored here
because of their licences) and 40 questions written by another AI from the papers and checked by Claude for meaning.
Rule fixed beforehand in `frozen_rule.json`. Gemma 3 4B on Kaggle, 200 answers, marked blind by two Claude agents (agreement
199 of 200 rows within 0.25); a row's mark is their average.

| Way | Text (tokens) | Answer numbers present (before answering) | Score |
|---|---|---|---|
| No memory | 0 | - | 15.0 (the 6 not-answerable questions) |
| Search | 878 | 45% | 41.1 |
| Search, 1,100 tokens | 1,198 | 53% | 45.9 |
| Cabinets alone | 789 | 56% | 41.1 |
| Cabinets + search | 1,170 | 72% | 53.7 |

**Primary, cabinets + search minus search at equal text: +7.8 points (95% interval -2.5 to +18.5; 5 questions won, 1 lost).
The interval crosses zero, so by the rule fixed beforehand no difference is shown at this size.**

By kind (search at 1,100 tokens / cabinets + search): single lookups 40.0 / 60.0; within-study comparisons 12.5 / 25.0;
time and design 100 / 100 (cabinets alone 0, as these are in sentences, not tables); across the meta-analyses 5.8 / 8.1
(no way could list several studies' results); not answerable 100 / 83.3 (the cabinet records twice led Gemma to quote a
number for a question nothing in the papers answers).

Limits: 40 questions; Claude read the questions before writing the cabinet code (the code treats every table alike and was
not changed after any answer); one model; Claude marks.
