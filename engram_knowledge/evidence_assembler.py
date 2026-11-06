"""Puts each returned fact with the original sentences behind it. The sentences come from the permanent record, not from any summary."""
LABELS = dict(works_at="Works at", lives_in="Lives in", job_title="Job title", status="Status", married_to="Married to", owns="Owns", hobby="Hobby", member_of="Member of", likes="Likes", dislikes="Dislikes", friend_of="Friend of", parent_of="Parent of")


class EvidenceAssembler:
    def __init__(self, kb):
        self.kb = kb

    def sentences(self, fact_ids):
        seen, out = set(), []
        for fid in fact_ids:
            for s in self.kb.trace(fid):
                if s.sid not in seen:
                    seen.add(s.sid)
                    out.append(dict(sid=s.sid, speaker=s.speaker, date=s.date, text=s.text, role=s.evidence_role))
        return sorted(out, key=lambda x: x["sid"])

    def period(self, slot, p, conflict=False):
        """One value, in the form every path returns. Confidence stays None: nothing here measures it."""
        statuses = [self.kb.get(i).fact.status for i in p.fact_ids]
        lines = self.sentences(p.fact_ids)
        return dict(fact=f"{LABELS.get(slot, slot)} {p.value}", slot=slot, value=p.value, status="needs_review" if "needs_review" in statuses else "active", valid_from=p.start, first_said=p.first_said,
                    valid_to=p.end, conflict=conflict, evidence=[x["sid"] for x in lines], sentences=lines, confidence=None)

    def text(self, result):
        """What a model would read: each fact with its original sentences beneath it. Not used for answers yet."""
        out = []
        for f in result["facts"]:
            when = f"{f['valid_from'] or 'start unknown'} to {f['valid_to'] or 'now'}"
            out.append(f"[Record] {f['fact']} ({when}" + ("; sources disagree" if f["conflict"] else "") + ")")
            out += [f"  [{s['date']}] {s['speaker']}: {s['text']}" for s in f["sentences"]]
        return "\n".join(out)
