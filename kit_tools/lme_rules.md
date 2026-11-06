# Scoring rules for the LongMemEval test

The wording follows `src/evaluation/evaluate_qa.py` in github.com/xiaowu0162/LongMemEval. Score 1 or 0. The scorer sees only the question, the correct answer and the response, never which system wrote it.

**Single-session user, single-session assistant, multi-session:** 1 if the response contains the correct answer, is equivalent to it, or contains all the intermediate steps to get it. 0 if it holds only part of the information required.

**Temporal reasoning:** as above. Do not penalise an off-by-one error in a count of days, weeks or months: 18 days counts as correct when the answer is 19.

**Knowledge update:** as above. A response holding some earlier information along with the updated answer is correct as long as the updated answer is the required answer.

**Preference:** the correct answer is a rubric. 1 if the response satisfies it. It need not reflect every point; it must recall and use the user's personal information correctly.

**Not answerable (id ends in `_abs`):** 1 if the response correctly identifies the question as unanswerable, for example that the information is incomplete or was not mentioned, or that other information is given but not the one asked.

A response of "not mentioned" scores 0 on every type except not answerable.
