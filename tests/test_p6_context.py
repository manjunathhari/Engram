import sqlite3, unittest
from engram_p6 import ContextEvaluator, ContextEvaluationError
from engram_p6.experience_compat import write_feedback

SCHEMA="""CREATE TABLE p6_experience_events(
event_id TEXT PRIMARY KEY,idempotency_key TEXT NOT NULL UNIQUE,event_type TEXT NOT NULL,
occurred_at TEXT NOT NULL,context_key TEXT,query_kind TEXT,candidate_refs TEXT NOT NULL DEFAULT '[]',
selected_refs TEXT NOT NULL DEFAULT '[]',target_ref TEXT,signal TEXT,signal_value REAL,
source_event_ref TEXT,metadata TEXT NOT NULL DEFAULT '{}',schema_version INTEGER NOT NULL DEFAULT 1);
CREATE TABLE p2_feedback(event_id TEXT PRIMARY KEY,candidate_id TEXT,source_sid TEXT,
feedback TEXT,query_key TEXT,comment TEXT,event_json TEXT,created_at REAL);"""
def add(db,eid,typ,ctx=None,qkind=None,candidates="[]",selected="[]",target=None,signal=None,source=None):
    db.execute("INSERT INTO p6_experience_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
      (eid,eid,typ,"2026-01-01T00:00:00+00:00",ctx,qkind,candidates,selected,target,signal,None,source,"{}"))

class TestContextEvaluator(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(":memory:")
        self.db.executescript(SCHEMA)
        add(self.db,"r1","retrieval","career","career",'["f1","f2"]','["f1"]')
        add(self.db,"f1","feedback","career",target="f1",signal="helpful")
        add(self.db,"f2","feedback","career",target="f1",signal="confirm")
        add(self.db,"f3","feedback","career",target="f1",signal="irrelevant")
        add(self.db,"f4","feedback","medical",target="f1",signal="wrong")
        self.ev=ContextEvaluator(self.db)
    def test_exposure_and_selection_not_outcome(self):
        a=self.ev.evaluate("f2","career")
        self.assertEqual((a.exposure_count,a.selection_count),(1,0))
        self.assertEqual(a.evidence_count,0)
        self.assertEqual(a.assessment,"insufficient_evidence")
    def test_context_specific_score(self):
        a=self.ev.evaluate("f1","career")
        self.assertEqual((a.positive_count,a.negative_count),(2,1))
        self.assertAlmostEqual(a.usefulness_mean,0.6)
        self.assertEqual(a.assessment,"mixed_or_early_evidence")
    def test_separate_contexts(self):
        results=self.ev.evaluate_all_contexts("f1")
        self.assertEqual({x.context_key for x in results},{"career","medical"})
    def test_no_evidence_neutral_prior(self):
        a=self.ev.evaluate("unseen")
        self.assertEqual(a.usefulness_mean,0.5)
        self.assertEqual(a.assessment,"insufficient_evidence")
    def test_p2_feedback_sync(self):
        self.db.execute("INSERT INTO p2_feedback VALUES(?,?,?,?,?,?,?,?)",
          ("p2e1","candidate-1","77","helpful","queryhash","","{}",1.0))
        self.assertEqual(self.ev.sync_p2_feedback(),1)
        self.assertEqual(self.ev.sync_p2_feedback(),0)
        a=self.ev.evaluate("candidate-1","queryhash")
        self.assertEqual(a.positive_count,1)
    def test_wrong_schema_refused(self):
        db=sqlite3.connect(":memory:")
        db.execute("CREATE TABLE p6_experience_events(x)")
        with self.assertRaises(ContextEvaluationError): ContextEvaluator(db)
    def test_does_not_write_p1_or_update_events(self):
        before=self.db.execute("SELECT event_id,signal FROM p6_experience_events ORDER BY event_id").fetchall()
        self.ev.evaluate("f1")
        after=self.db.execute("SELECT event_id,signal FROM p6_experience_events ORDER BY event_id").fetchall()
        self.assertEqual(before,after)
    def test_missing_p2_is_explicit(self):
        db=sqlite3.connect(":memory:")
        db.executescript(SCHEMA.split("CREATE TABLE p2_feedback",1)[0])
        evaluator=ContextEvaluator(db)
        with self.assertRaises(ContextEvaluationError): evaluator.sync_p2_feedback()
if __name__=="__main__": unittest.main()
