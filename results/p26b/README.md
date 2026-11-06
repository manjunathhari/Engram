# P2.6 fresh set: 200 gated sentences, labelled blind

Drawn after the forced form and the wish rules were committed (7968a01): 150 LoCoMo (conversations 0 to 4) and 50 LongMemEval
user sentences that the P2 gate passes, none in any earlier set. Labels by a separate agent from the same written instructions
as results/p26, without the code. 49 of 200 sentences hold a listed fact; 56 facts.

Before Gemma's run, on these labels:
- Fixed-rule writer: 11 facts written, 2 right; found 2 of 56. Validator accepted 10.
- The labelled facts themselves (a perfect writer): validator accepted 34, uncertain 22, rejected 0. The wish rules turned
  away no correct fact. The 22 uncertain lack the relation's word in the sentence ("my pup" for owns).
- Dates: 4 of 6 labelled dates match the fixed rules. Not covered: "this month", "last Saturday". Left as they are for this test.
