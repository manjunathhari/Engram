import sqlite3, unittest
from engram_p6 import AdaptiveScorer, AdaptiveScorerError

SCHEMA="""CREATE TABLE p6_experience_events(
event_id TEXT PRIMARY KEY,idempotency_key TEXT UNIQUE,event_type TEXT,occurred_at TEXT,
context_key TEXT,query_kind TEXT,candidate_refs TEXT,selected_refs TEXT,target_ref TEXT,
signal TEXT,signal_value REAL,source_event_ref TEXT,metadata TEXT,schema_version INTEGER);"""
def add(db,eid,ctx,target,signal):
    db.execute("INSERT INTO p6_experience_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
      (eid,eid,"feedback","2026-01-01T00:00:00Z",ctx,None,"[]","[]",target,signal,None,None,"{}"))

class TestAdaptiveScorer(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(":memory:")
        self.db.executescript(SCHEMA)
        add(self.db,"e1","career","f1","helpful")
        add(self.db,"e2","career","f1","confirm")
        add(self.db,"e3","career","f1","irrelevant")
        add(self.db,"e4","medical","f1","wrong")
        add(self.db,"e5","career","f2","irrelevant")
        add(self.db,"e6","career","f2","wrong")
        self.s=AdaptiveScorer(self.db)
    def test_positive_moderate_adjustment(self):
        x=self.s.score("f1","career")
        self.assertGreater(x.multiplier,1)
        self.assertLessEqual(x.multiplier,1.2)
    def test_negative_adjustment(self):
        x=self.s.score("f2","career")
        self.assertLess(x.multiplier,1)
        self.assertGreaterEqual(x.multiplier,.8)
    def test_no_evidence_neutral(self):
        x=self.s.score("unknown","career")
        self.assertEqual(x.multiplier,1.0)
        self.assertEqual(x.assessment,"insufficient_evidence")
    def test_contexts_separate(self):
        self.assertNotEqual(self.s.score("f1","career").multiplier,
                            self.s.score("f1","medical").multiplier)
    def test_policy_versions_immutable(self):
        a=self.s.build_policy("career")
        old=self.s.get_policy("f1","career",a.version)
        add(self.db,"e7","career","f1","helpful")
        b=self.s.build_policy("career")
        self.assertGreater(b.version,a.version)
        self.assertEqual(old,self.s.get_policy("f1","career",a.version))
        self.assertNotEqual(old["multiplier"],self.s.get_policy("f1","career",b.version)["multiplier"])
    def test_apply_multiplier(self):
        self.s.build_policy("career")
        self.assertGreater(self.s.apply_multiplier(10,"f1","career"),10)
    def test_unknown_policy_leaves_base_score(self):
        self.assertEqual(self.s.apply_multiplier(5,"missing","career"),5)
    def test_policy_bounded(self):
        for i in range(100):
            add(self.db,f"pos{i}","x","pos","helpful")
            add(self.db,f"neg{i}","x","neg","wrong")
        pos=self.s.score("pos","x")
        neg=self.s.score("neg","x")
        self.assertLessEqual(pos.multiplier,1.2)
        self.assertGreaterEqual(neg.multiplier,.8)
    def test_invalid_config(self):
        with self.assertRaises(AdaptiveScorerError):
            AdaptiveScorer(self.db,max_adjustment=2)
    def test_no_event_mutation(self):
        before=self.db.execute("SELECT event_id,signal FROM p6_experience_events ORDER BY event_id").fetchall()
        self.s.build_policy("career")
        after=self.db.execute("SELECT event_id,signal FROM p6_experience_events ORDER BY event_id").fetchall()
        self.assertEqual(before,after)
if __name__=="__main__": unittest.main()
