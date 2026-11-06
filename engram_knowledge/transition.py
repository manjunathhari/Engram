"""How each kind of observation changes a state. Written out in full, with nothing inferred from the wording."""

SINGLE = ("works_at", "lives_in", "job_title", "status", "married_to")  # one value at a time, so two can conflict
MULTI = ("owns", "hobby", "member_of", "likes", "dislikes", "friend_of", "parent_of")  # several at once
OPENS = dict(works_at="works_at", started_job="works_at", lives_in="lives_in", moved_to="lives_in", job_title="job_title", status="status", married_to="married_to", got_married="married_to",
             owns="owns", bought="owns", adopted="owns", hobby="hobby", started_hobby="hobby", member_of="member_of", joined="member_of", likes="likes", dislikes="dislikes", friend_of="friend_of", parent_of="parent_of")
CLOSES = dict(left_job="works_at", divorced="married_to", sold="owns", lost="owns", gave_away="owns", stopped_hobby="hobby", left_group="member_of")


def slot_of(predicate):
    return OPENS.get(predicate) or CLOSES.get(predicate)
