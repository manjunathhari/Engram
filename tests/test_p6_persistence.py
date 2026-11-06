import sqlite3, unittest, json
from engram_p6 import ModelPersistence, ModelPersistenceError

def setup_db():
    db=sqlite3.connect(":memory:")
    db.executescript("""
    CREATE TABLE p6_policy_versions(version INTEGER PRIMARY KEY,context_key TEXT,created_at TEXT,
      algorithm TEXT,config_json TEXT,source_max_event_id TEXT,policy_count INTEGER,note TEXT);
    CREATE TABLE p6_scoring_policies(policy_id TEXT PRIMARY KEY,version INTEGER,context_key TEXT,target_ref TEXT);
    """)
    db.execute("INSERT INTO p6_policy_versions VALUES(1,'career','2026-01-01T00:00:00Z','beta_v1','{\"min_evidence\":2}','e9',1,'test')")
    db.execute("INSERT INTO p6_scoring_policies VALUES('p1',1,'career','memory-a')")
    db.commit()
    return db

class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.db=setup_db(); self.store=ModelPersistence(self.db)
    def make(self):
        return self.store.create_bundle(p63_policy_version=1,p63_context_key="career",
          p64_config={"half_life_days":90,"trend_window_days":30},
          retrieval_config={"max_adjustment":0.2},note="test")
    def test_create_and_read(self):
        b=self.make(); self.assertEqual(self.store.get_bundle(b.bundle_id),b)
    def test_activate_and_read_active(self):
        b=self.make(); active=self.store.activate(b.bundle_id,activated_by="operator",reason="validated")
        self.assertEqual(active["bundle"].bundle_id,b.bundle_id)
    def test_export_import_roundtrip(self):
        b=self.make(); doc=self.store.export_bundle(b.bundle_id)
        db2=setup_db(); s2=ModelPersistence(db2)
        imported=s2.import_bundle(doc)
        self.assertEqual(imported.bundle_sha256,b.bundle_sha256)
    def test_checksum_tampering_rejected(self):
        b=self.make(); doc=self.store.export_bundle(b.bundle_id)
        doc["bundle"]["p64_config"]["half_life_days"]=7
        with self.assertRaisesRegex(ModelPersistenceError,"checksum"):
            ModelPersistence(setup_db()).import_bundle(doc)
    def test_wrong_context_rejected(self):
        with self.assertRaisesRegex(ModelPersistenceError,"context"):
            self.store.create_bundle(p63_policy_version=1,p63_context_key="medical",p64_config={})
    def test_missing_policy_rejected(self):
        with self.assertRaises(ModelPersistenceError):
            self.store.create_bundle(p63_policy_version=9,p63_context_key="career",p64_config={})
    def test_manifest_count_mismatch_rejected(self):
        self.db.execute("DELETE FROM p6_scoring_policies"); self.db.commit()
        with self.assertRaisesRegex(ModelPersistenceError,"manifest"):
            self.make()
    def test_required_activation_audit(self):
        b=self.make()
        with self.assertRaisesRegex(ModelPersistenceError,"activated_by"):
            self.store.activate(b.bundle_id,activated_by="",reason="x")
    def test_idempotent_import(self):
        b=self.make(); doc=self.store.export_bundle(b.bundle_id)
        self.assertEqual(self.store.import_bundle(doc).bundle_id,b.bundle_id)
    def test_bundle_immutable_conflict(self):
        b=self.make(); doc=self.store.export_bundle(b.bundle_id)
        doc["bundle"]["note"]="altered"
        # checksum catches changed content before id collision.
        with self.assertRaises(ModelPersistenceError): self.store.import_bundle(doc)
if __name__=="__main__": unittest.main()
