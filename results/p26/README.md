# P2.6 fixed-rule writer on 200 gated sentences, labelled blind

Sentences: 150 LoCoMo (conversations 5 to 9) and 50 LongMemEval user sentences that the P2 gate passes as "learn", none used before.
Labels (who, what, which, qualifiers, time) by a separate agent from written instructions, without the code. Rules written before the labels were seen; scored once with `kit_tools/p26_score.py`.

- 48 of 200 sentences state a fact in the predicate list; 55 facts in all.
- The writer wrote 13 facts. 2 were right (same predicate, same head word); 1 had the exact object.
- It found 2 of the 55 labelled facts.
- The validator accepted all 13, including the 11 wrong ones: it checks that the words are in the sentence, not what they mean.

Why: real speech states facts indirectly ("Yesterday I took my puppy to the clinic" means owns puppy), and "I love X" is mostly not a lasting fact ("I love the idea of..."). Fixed rules are not a usable writer. Not tuned on these labels; a new labelled set is needed to measure any change.
