import unittest
from engram_p6 import evaluate, validate_records, ValidationError, query_metrics

def rec(qid,base,p63,p64,relevant,rare=None,corrected=None):
    return {"query_id":qid,"query_time":"2026-05-01T10:00:00+00:00",
      "relevant_refs":relevant,"rare_important_refs":rare or [],
      "corrected_refs":corrected or [],
      "rankings":{"baseline":base,"p63":p63,"p63_p64":p64},
      "policy_provenance":{
        "p63":{"policy_built_at":"2026-04-30T10:00:00+00:00","source_max_event_at":"2026-04-29T10:00:00+00:00"},
        "p63_p64":{"policy_built_at":"2026-04-30T10:00:00+00:00","source_max_event_at":"2026-04-29T10:00:00+00:00"}}}
class TestP65(unittest.TestCase):
    def setUp(self):
        self.rows=[
          rec("q1",["a","x"],["b","a"],["b","a"],["a","b"],["b"]),
          rec("q2",["x","c"],["c","d"],["d","c"],["c","d"],["d"],["x"]),
          rec("q3",["z","y"],["y","z"],["y","z"],["y","z"],["z"]),
        ]
    def test_metrics_and_comparison(self):
        report=evaluate(self.rows,k=2,bootstrap_samples=200)
        self.assertEqual(report.query_count,3)
        self.assertGreater(report.metrics["p63"]["recall"],report.metrics["baseline"]["recall"])
        self.assertEqual(len(report.comparisons),12)
    def test_rare_recall_reported(self):
        report=evaluate(self.rows,k=2,bootstrap_samples=200)
        self.assertIn("rare_recall",report.metrics["p63"])
    def test_correction_rate(self):
        m=query_metrics(self.rows[1],"baseline",2)
        self.assertEqual(m["correction_rate"],0.5)
    def test_future_policy_rejected(self):
        bad=[dict(self.rows[0])]
        bad[0]["policy_provenance"]=dict(self.rows[0]["policy_provenance"])
        bad[0]["policy_provenance"]["p63"]["source_max_event_at"]="2026-05-02T00:00:00+00:00"
        with self.assertRaisesRegex(ValidationError,"temporal leakage"):
            evaluate(bad,bootstrap_samples=100)
    def test_duplicate_query_rejected(self):
        with self.assertRaisesRegex(ValidationError,"duplicate query_id"):
            validate_records([self.rows[0],self.rows[0]])
    def test_rare_must_be_relevant(self):
        bad=[dict(self.rows[0],rare_important_refs=["not-relevant"])]
        with self.assertRaisesRegex(ValidationError,"subset"):
            validate_records(bad)
    def test_duplicate_rank_rejected(self):
        bad=[dict(self.rows[0])]
        bad[0]["rankings"]=dict(self.rows[0]["rankings"])
        bad[0]["rankings"]["p63"]=["a","a"]
        with self.assertRaisesRegex(ValidationError,"duplicate"):
            validate_records(bad)
    def test_k_validation(self):
        with self.assertRaises(ValidationError): evaluate(self.rows,k=0)
    def test_empty_relevant_is_nan_safe(self):
        r=rec("empty",[],[],[],[])
        report=evaluate([r],bootstrap_samples=100)
        self.assertTrue(report.metrics["baseline"]["recall"] != report.metrics["baseline"]["recall"])
if __name__=="__main__": unittest.main()
