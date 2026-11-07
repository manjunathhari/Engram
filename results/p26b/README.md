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

## Gemma 3 12B on the same 200 sentences (7 November 2026), against the rule in rule_12b.md

| | 4B free | 4B forced | 12B free | 12B forced (decides) |
|---|---|---|---|---|
| Facts read | 14 | 187 | 108 | 172 |
| Labelled found (of 56) | 6 | 11 | 22 | 17 |
| Stored by the validator | 8 | 32 | 39 | 46 |
| Of those right | 5 | 8 | 16 | 10 |

Rule: forced form needs at least 3 right per wrong among stored facts, and at least 19 found. 12B forced: 10 right to 36 wrong,
17 found. **Failed both. P2.6 is parked**: extraction stays off and unmerged. The 12B free form found more (22) but stored 16 right
to 23 wrong, so it would also fail the first condition. In the forced form the 12B twice copied the prompt's example
("adopted beagle, Pixel, from the shelter, yesterday") for sentences that do not mention it; the validator rejected both.

## GPT-4o on the same 200 sentences (7 November 2026), against the rule in rule_gpt4o.md

Replies from the Kaggle run; rescored here from the replies with the scorer at e87d52d, identical to the notebook's scores.

| | 12B forced | GPT-4o free | GPT-4o forced (decides) |
|---|---|---|---|
| Sentences given any fact (labels: 49) | | 13 | 106 |
| Facts written | 172 | 14 | 135 |
| Labelled found (of 56) | 17 | 10 | 31 |
| Stored by the validator | 46 | 11 | 68 |
| Of those right | 10 | 9 | 21 |
| Right facts the validator held back as uncertain | | 1 | 10 |

GPT-4o forced: found 31 (condition 2 met), stored 21 right to 47 wrong (condition 1 not met). **Failed.** Under the outcomes fixed
beforehand this is the middle case: the writer finds facts well enough, and the checking and the relation list let wrong ones in.

Where the 104 wrong forced-form facts came from (exploratory, after seeing the result):
- 76 from sentences that hold no listed fact: feelings and states ("motivated", "bored", "proud of who I am"), opinions
  ("likes color", "likes your opinion"), one-off events (a break, lunch, a curry made) and vague plans ("list", "workout plan").
  GPT-4o gave facts to 106 sentences where the labels have 49.
- 12 from sentences that hold a different fact.
- 10 the right thing under a neighbouring relation: started_job for owns (3), attended for plans or completed, member_of for joined,
  lost for left_job, likes against hobby (3).
- 6 the right relation with different words for the thing.
The free form is the opposite: GPT-4o wrote facts for only 13 sentences, 9 of 11 stored right, but found only 10.
The validator passed 47 wrong facts and held back 10 of 31 right ones. Nothing here changes code; any fix must be measured on a fresh set.
