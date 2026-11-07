# Marking rules

Each row has a question about published surgical research, the reference answer (checked against the papers), and a reply to mark. Judge the reply against the reference answer only. Do not use your own knowledge of the papers.

- **kind single, within or time_design**: mark 1 if the reply gives the reference answer's value(s) for the right group(s), allowing different formatting (8.1% for 19/236 (8.1%); "2017–2019" for "January 2017 to December 2019"; a number without its unit when the question asks only for that number). For `within`, both groups' values must be given and attributable to the right group. Mark 0 if any required value is wrong, missing, attached to the wrong group, or the reply says not mentioned.
- **kind across**: the reference answer lists several studies or values. Mark the share (0 to 1, two decimals) of the reference's distinct study results that the reply gives correctly and attributes to the right study. Values given for the wrong study, or not in the reference, earn nothing. A reply that only says "not mentioned" scores 0.
- **kind not_answerable**: mark 1 only if the reply says the information is not mentioned or not reported, without giving numbers as if they answered the question. Otherwise 0.

Write one JSON object mapping every row id to {"mark": number, "why": "under 12 words"}.
