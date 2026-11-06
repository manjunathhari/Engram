import sqlite3, unittest
from datetime import datetime, timezone, timedelta
from engram_p6 import TemporalAdapter, TemporalAdaptationError

SCHEMA="""CREATE TABLE p6_experience_events(
event_id TEXT PRIMARY KEY,idempotency_key TEXT UNIQUE,event_type TEXT,occurred_at TEXT,
context_key TEXT,query_kind TEXT,candidate_refs TEXT,selected_refs TEXT,target_ref TEXT,
signal TEXT,signal_value REAL,source_event_ref TEXT,metadata TEXT,schema_version INTEGER);"""
def add(db,eid,when,target="f1",ctx="career",signal="helpful",typ="feedback"):
    db.execute("INSERT INTO p6_experience_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
      (eid,eid,typ,when,ctx,None,"[]","[]",target,signal,None,None,"{}"))
class TestTemporalAdapter(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(":memory:")
        self.db.executescript(SCHEMA)
        self.now=datetime(2026,6,1,tzinfo=timezone.utc)
        self.adapter=TemporalAdapter(self.db,half_life_days=30,trend_window_days=30)
    def ts(self,days): return (self.now-timedelta(days=days)).isoformat()
    def test_recent_feedback_has_more_weight(self):
        add(self.db,"old",self.ts(120),signal="helpful")
        add(self.db,"new",self.ts(1),signal="helpful")
        x=self.adapter.evaluate("f1","career",as_of=self.now.isoformat())
        self.assertGreater(x.weighted_positive,1.0)
        self.assertLess(x.weighted_positive,2.0)
    def test_no_evidence_neutral(self):
        x=self.adapter.evaluate("missing","career",as_of=self.now.isoformat())
        self.assertEqual(x.temporal_multiplier,1.0)
        self.assertEqual(x.assessment,"insufficient_temporal_evidence")
    def test_contexts_isolated(self):
        add(self.db,"a",self.ts(1),ctx="career",signal="helpful")
        add(self.db,"b",self.ts(1),ctx="medical",signal="wrong")
        a=self.adapter.evaluate("f1","career",as_of=self.now.isoformat())
        b=self.adapter.evaluate("f1","medical",as_of=self.now.isoformat())
        self.assertGreater(a.posterior_mean,b.posterior_mean)
    def test_future_events_excluded(self):
        add(self.db,"future",(self.now+timedelta(days=3)).isoformat())
        x=self.adapter.evaluate("f1","career",as_of=self.now.isoformat())
        self.assertEqual(x.weighted_positive,0)
    def test_declining_trend(self):
        for i in range(3): add(self.db,f"old{i}",self.ts(45+i),signal="helpful")
        for i in range(3): add(self.db,f"new{i}",self.ts(1+i),signal="wrong")
        x=self.adapter.evaluate("f1","career",as_of=self.now.isoformat())
        self.assertEqual(x.trend_label,"declining")
        self.assertLess(x.temporal_multiplier,1)
    def test_improving_trend(self):
        for i in range(3): add(self.db,f"old{i}",self.ts(45+i),signal="wrong")
        for i in range(3): add(self.db,f"new{i}",self.ts(1+i),signal="helpful")
        x=self.adapter.evaluate("f1","career",as_of=self.now.isoformat())
        self.assertEqual(x.trend_label,"improving")
        self.assertGreater(x.temporal_multiplier,1)
    def test_decay_moves_score_towards_neutral(self):
        add(self.db,"old",self.ts(365),signal="helpful")
        x=self.adapter.evaluate("f1","career",as_of=self.now.isoformat())
        self.assertAlmostEqual(x.temporal_multiplier,1.0,places=2)
    def test_combine_with_p63_bounded(self):
        add(self.db,"a",self.ts(1),signal="helpful")
        x=self.adapter.evaluate("f1","career",as_of=self.now.isoformat())
        self.assertLessEqual(self.adapter.combine_with_p63(1.2,x),1.2)
    def test_invalid_timestamp_rejected(self):
        add(self.db,"bad","not-a-date")
        with self.assertRaises(TemporalAdaptationError):
            self.adapter.evaluate("f1","career",as_of=self.now.isoformat())
    def test_event_ledger_not_mutated(self):
        add(self.db,"a",self.ts(1))
        before=self.db.execute("SELECT event_id,signal FROM p6_experience_events").fetchall()
        self.adapter.evaluate("f1","career",as_of=self.now.isoformat())
        after=self.db.execute("SELECT event_id,signal FROM p6_experience_events").fetchall()
        self.assertEqual(before,after)
if __name__=="__main__": unittest.main()
