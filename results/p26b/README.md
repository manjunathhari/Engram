# P2.6 fresh set: 200 gated sentences, labelled blind

Drawn after the forced form and the wish rules were committed (7968a01): 150 LoCoMo (conversations 0 to 4) and 50 LongMemEval
user sentences that the P2 gate passes, none in any earlier set. Labels by a separate agent from the same written instructions
as results/p26, without the code. 49 of 200 sentences hold a listed fact; 56 facts.

Before Gemma's run, on these labels:
- Fixed-rule writer: 11 facts written, 2 right; found 2 of 56. Validator accepted 10.
- The labelled facts themselves (a perfect writer): validator accepted 34, uncertain 22, rejected 0. The wish rules turned
  away no correct fact. The 22 uncertain lack the relation's word in the sentence ("my pup" for owns).
- Dates: 4 of 6 labelled dates match the fixed rules. Not covered: "this month", "last Saturday". Left as they are for this test.

## Gemma 3 4B as the writer (Kaggle T4, 7 November 2026), scored once against these labels

| | Free line form | Forced form (JSON held to the list) |
|---|---|---|
| Sentences given facts | 99 replies with lines, 170 lines unreadable | 167 of 200 (labels: 49) |
| Facts read | 14 | 187 |
| Right (same person, relation, head word) | 6 | 11 |
| Labelled facts found | 6 of 56 | 11 of 56 |
| Validator accepted | 8, of which 5 right | 32, of which 8 right |

The forced form ends unreadable replies but writes a fact for almost every sentence, so it finds more and stores more wrong.
Wrong facts the validator accepted are mostly "like" used as a comparison ("It's like a nudge", "like goldmines"),
"I like the sound of", wants and ideas not caught by the frozen rules ("One idea I had was to join", "I want to give something
back"), and wrong relations ("give those news sources a try" as gave_away). Several more are the right thing under a neighbouring
relation (started_job for owns a store, visited for attended a parade, lost for left_job). Neither form is a usable writer.
Any rule added from these examples must be measured on another fresh set.
