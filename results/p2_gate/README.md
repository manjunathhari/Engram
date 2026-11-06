# P2 learning gate on 200 real sentences, labelled blind

Sample: 120 LoCoMo sentences (conversations 0 to 4) and 80 LongMemEval sentences, labelled by a separate agent from the written policy without seeing code.
Fixes were chosen before the labels existed. Run with `kit_tools/p2_compare.py`.

| | original P2.3 (speakers as user) | fixed |
|---|---|---|
| agrees with label | 88 of 200 | 102 of 200 |
| should learn (32): learned | 27 | 29 |
| should review (26): reviewed | 3 | 2 |
| should ignore (116): learned anyway | 56 | 57 |
| assistant content (26): kept as assistant | 0 | 26 |

The fixes did what they targeted. The gate still treats small talk ("That's awesome!") as something to learn, and sends almost nothing to review.
Any further change must be measured on a new labelled set, not this one.
