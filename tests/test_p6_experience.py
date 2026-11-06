import sqlite3, unittest
from engram_p6 import ExperienceRecorder, ExperienceError

class TestExperienceRecorder(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(":memory:")
        self.r=ExperienceRecorder(self.db)
    def test_retrieval_records_refs_not_text(self):
        e=self.r.record_retrieval(idempotency_key="q1",context_key="abc",query_kind="career",
          candidate_refs=["fact:1","fact:2"],selected_refs=["fact:2"])
        self.assertEqual(e.event_type,"retrieval")
        rows=self.r.events_for("fact:2")
        self.assertEqual(rows,[])
        raw=self.db.execute("SELECT candidate_refs,selected_refs FROM p6_experience_events").fetchone()
        self.assertEqual(raw,('["fact:1","fact:2"]','["fact:2"]'))
    def test_feedback(self):
        e=self.r.record_feedback(idempotency_key="f1",target_ref="fact:2",signal="irrelevant")
        self.assertEqual(e.signal,"irrelevant")
        self.assertEqual(self.r.events_for("fact:2")[0]["signal"],"irrelevant")
    def test_correction(self):
        e=self.r.record_correction(idempotency_key="c1",target_ref="fact:3")
        self.assertEqual(e.signal,"corrected")
    def test_idempotency(self):
        a=self.r.record_feedback(idempotency_key="same",target_ref="x",signal="helpful")
        b=self.r.record_feedback(idempotency_key="same",target_ref="x",signal="helpful")
        self.assertEqual(a.event_id,b.event_id)
        self.assertTrue(b.replayed)
        self.assertEqual(self.r.count(),1)
    def test_reject_raw_text_fields(self):
        with self.assertRaises(ExperienceError):
            self.r.record("outcome",idempotency_key="x",metadata={"transcript":"hello"})
    def test_invalid_signal(self):
        with self.assertRaises(ExperienceError):
            self.r.record_feedback(idempotency_key="x",target_ref="f",signal="maybe")
    def test_context_key_is_content_free_stable(self):
        a=self.r.context_key("career","plan","personal")
        b=self.r.context_key("career","plan","personal")
        self.assertEqual(a,b)
        self.assertNotEqual(a,self.r.context_key("medical","plan","personal"))
    def test_append_only_no_update_api(self):
        self.r.record_feedback(idempotency_key="f1",target_ref="f",signal="helpful")
        self.assertEqual(self.r.count("feedback"),1)
        self.assertEqual(self.db.execute("SELECT count(*) FROM p6_experience_events").fetchone()[0],1)
if __name__=="__main__": unittest.main()
