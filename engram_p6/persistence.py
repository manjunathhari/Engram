"""P6.6 Model Persistence: durable, portable snapshots of adaptive policy configuration.

P6.3 already stores per-memory policy rows in p6_scoring_policies and version
metadata in p6_policy_versions. P6.6 does not duplicate those rows. It stores
an immutable deployment bundle that pins a P6.3 policy version and the P6.4
temporal configuration, plus checksummed export/import and an explicit active
bundle pointer.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib, json, sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS p6_model_bundles (
  bundle_id TEXT PRIMARY KEY,
  bundle_version INTEGER NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  p63_policy_version INTEGER NOT NULL,
  p63_context_key TEXT NOT NULL,
  p63_algorithm TEXT NOT NULL,
  p63_config_json TEXT NOT NULL,
  p64_config_json TEXT NOT NULL,
  retrieval_config_json TEXT NOT NULL,
  bundle_sha256 TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS p6_model_bundle_active (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  bundle_id TEXT NOT NULL REFERENCES p6_model_bundles(bundle_id),
  activated_at TEXT NOT NULL,
  activated_by TEXT NOT NULL,
  reason TEXT NOT NULL
);
"""

class ModelPersistenceError(ValueError): pass

@dataclass(frozen=True)
class ModelBundle:
    bundle_id: str
    bundle_version: int
    created_at: str
    p63_policy_version: int
    p63_context_key: str
    p63_algorithm: str
    p63_config: dict
    p64_config: dict
    retrieval_config: dict
    bundle_sha256: str
    note: str

    def to_dict(self): return asdict(self)

class ModelPersistence:
    FORMAT_VERSION=1
    def __init__(self, db: sqlite3.Connection):
        self.db=db
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required={"p6_policy_versions","p6_scoring_policies"}
        if not required.issubset(tables):
            raise ModelPersistenceError("P6.3 policy tables are required before P6.6")
        self.db.executescript(SCHEMA)

    @staticmethod
    def _json_object(value, name):
        if not isinstance(value,dict): raise ModelPersistenceError(f"{name} must be a JSON object")
        try: return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
        except (TypeError,ValueError) as e: raise ModelPersistenceError(f"{name} is not JSON-safe: {e}") from None

    @staticmethod
    def _hash(payload):
        raw=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def create_bundle(self, *, p63_policy_version: int, p63_context_key: str,
                      p64_config: dict, retrieval_config: dict|None=None,
                      note: str="", bundle_id: str|None=None) -> ModelBundle:
        if not isinstance(p63_policy_version,int) or p63_policy_version<1:
            raise ModelPersistenceError("p63_policy_version must be a positive integer")
        if not isinstance(p63_context_key,str) or not p63_context_key.strip():
            raise ModelPersistenceError("p63_context_key is required")
        row=self.db.execute("""SELECT context_key,algorithm,config_json,policy_count
          FROM p6_policy_versions WHERE version=?""",(p63_policy_version,)).fetchone()
        if not row: raise ModelPersistenceError("P6.3 policy version does not exist")
        context,algorithm,config_json,policy_count=row
        if context!=p63_context_key: raise ModelPersistenceError("P6.3 policy context does not match")
        if policy_count<1: raise ModelPersistenceError("cannot persist an empty P6.3 policy")
        # Ensure the policy rows still match the version manifest.
        actual=self.db.execute("SELECT count(*) FROM p6_scoring_policies WHERE version=? AND context_key=?",
                               (p63_policy_version,p63_context_key)).fetchone()[0]
        if actual!=policy_count: raise ModelPersistenceError("P6.3 policy rows do not match version manifest")
        try: p63_config=json.loads(config_json)
        except Exception: raise ModelPersistenceError("stored P6.3 config is invalid JSON") from None
        p64_json=self._json_object(p64_config,"p64_config")
        retrieval_json=self._json_object(retrieval_config or {},"retrieval_config")
        try: p63_json=self._json_object(p63_config,"p63_config")
        except ModelPersistenceError: raise
        version=self.db.execute("SELECT COALESCE(MAX(bundle_version),0)+1 FROM p6_model_bundles").fetchone()[0]
        created=datetime.now(timezone.utc).isoformat()
        bid=bundle_id or f"p6bundle_{version:06d}"
        payload={"format_version":self.FORMAT_VERSION,"bundle_id":bid,"bundle_version":version,
          "created_at":created,"p63_policy_version":p63_policy_version,"p63_context_key":p63_context_key,
          "p63_algorithm":algorithm,"p63_config":json.loads(p63_json),
          "p64_config":json.loads(p64_json),"retrieval_config":json.loads(retrieval_json),"note":note}
        digest=self._hash(payload)
        self.db.execute("""INSERT INTO p6_model_bundles
          (bundle_id,bundle_version,created_at,p63_policy_version,p63_context_key,p63_algorithm,
           p63_config_json,p64_config_json,retrieval_config_json,bundle_sha256,note)
          VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
          (bid,version,created,p63_policy_version,p63_context_key,algorithm,p63_json,p64_json,
           retrieval_json,digest,note))
        self.db.commit()
        return ModelBundle(bid,version,created,p63_policy_version,p63_context_key,algorithm,
          payload["p63_config"],payload["p64_config"],payload["retrieval_config"],digest,note)

    def _row_to_payload(self,row):
        (bid,version,created,p63v,ctx,algo,p63,p64,ret,digest,note)=row
        return {"format_version":self.FORMAT_VERSION,"bundle_id":bid,"bundle_version":version,
          "created_at":created,"p63_policy_version":p63v,"p63_context_key":ctx,
          "p63_algorithm":algo,"p63_config":json.loads(p63),"p64_config":json.loads(p64),
          "retrieval_config":json.loads(ret),"note":note}

    def get_bundle(self,bundle_id: str) -> ModelBundle:
        row=self.db.execute("""SELECT bundle_id,bundle_version,created_at,p63_policy_version,
          p63_context_key,p63_algorithm,p63_config_json,p64_config_json,retrieval_config_json,
          bundle_sha256,note FROM p6_model_bundles WHERE bundle_id=?""",(bundle_id,)).fetchone()
        if not row: raise ModelPersistenceError("bundle not found")
        payload=self._row_to_payload(row)
        if self._hash(payload)!=row[9]: raise ModelPersistenceError("bundle checksum mismatch")
        return ModelBundle(payload["bundle_id"],payload["bundle_version"],payload["created_at"],
          payload["p63_policy_version"],payload["p63_context_key"],payload["p63_algorithm"],
          payload["p63_config"],payload["p64_config"],payload["retrieval_config"],row[9],payload["note"])

    def activate(self,bundle_id: str, *, activated_by: str, reason: str):
        if not activated_by or not activated_by.strip(): raise ModelPersistenceError("activated_by is required")
        if not reason or not reason.strip(): raise ModelPersistenceError("reason is required")
        bundle=self.get_bundle(bundle_id)
        # Do not activate a bundle whose referenced P6.3 policy is missing.
        exists=self.db.execute("""SELECT 1 FROM p6_policy_versions WHERE version=? AND context_key=?""",
          (bundle.p63_policy_version,bundle.p63_context_key)).fetchone()
        if not exists: raise ModelPersistenceError("referenced P6.3 policy version is unavailable")
        now=datetime.now(timezone.utc).isoformat()
        self.db.execute("""INSERT INTO p6_model_bundle_active(singleton,bundle_id,activated_at,activated_by,reason)
          VALUES(1,?,?,?,?) ON CONFLICT(singleton) DO UPDATE SET
          bundle_id=excluded.bundle_id,activated_at=excluded.activated_at,
          activated_by=excluded.activated_by,reason=excluded.reason""",
          (bundle_id,now,activated_by,reason))
        self.db.commit()
        return self.active_bundle()

    def active_bundle(self):
        row=self.db.execute("SELECT bundle_id,activated_at,activated_by,reason FROM p6_model_bundle_active WHERE singleton=1").fetchone()
        if not row: return None
        bundle=self.get_bundle(row[0])
        return {"bundle":bundle,"activated_at":row[1],"activated_by":row[2],"reason":row[3]}

    def export_bundle(self,bundle_id: str) -> dict:
        bundle=self.get_bundle(bundle_id)
        return {"format":"engram-p6-model-bundle","format_version":self.FORMAT_VERSION,
                "bundle":bundle.to_dict()}

    def import_bundle(self,document: dict, *, allow_policy_reference_only: bool=True) -> ModelBundle:
        if not isinstance(document,dict) or document.get("format")!="engram-p6-model-bundle":
            raise ModelPersistenceError("unrecognised model bundle format")
        if document.get("format_version")!=self.FORMAT_VERSION:
            raise ModelPersistenceError("unsupported model bundle format version")
        b=document.get("bundle")
        if not isinstance(b,dict): raise ModelPersistenceError("bundle payload is missing")
        required={"bundle_id","bundle_version","created_at","p63_policy_version","p63_context_key",
          "p63_algorithm","p63_config","p64_config","retrieval_config","bundle_sha256","note"}
        if not required.issubset(b): raise ModelPersistenceError("bundle payload is incomplete")
        payload={"format_version":self.FORMAT_VERSION,**{k:b[k] for k in (
          "bundle_id","bundle_version","created_at","p63_policy_version","p63_context_key",
          "p63_algorithm","p63_config","p64_config","retrieval_config","note")}}
        if self._hash(payload)!=b["bundle_sha256"]: raise ModelPersistenceError("import checksum mismatch")
        if not allow_policy_reference_only:
            raise ModelPersistenceError("P6.3 policy rows must be migrated separately; this bundle contains a version reference, not the policy rows")
        # A bundle imported on a different database may be archived before its policy is present.
        self._json_object(b["p63_config"],"p63_config")
        p64=self._json_object(b["p64_config"],"p64_config")
        ret=self._json_object(b["retrieval_config"],"retrieval_config")
        p63=self._json_object(b["p63_config"],"p63_config")
        existing=self.db.execute("SELECT bundle_sha256 FROM p6_model_bundles WHERE bundle_id=?",(b["bundle_id"],)).fetchone()
        if existing:
            if existing[0]!=b["bundle_sha256"]: raise ModelPersistenceError("bundle_id already exists with different content")
            return self.get_bundle(b["bundle_id"])
        version=self.db.execute("SELECT COALESCE(MAX(bundle_version),0)+1 FROM p6_model_bundles").fetchone()[0]
        self.db.execute("""INSERT INTO p6_model_bundles VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
          (b["bundle_id"],version,b["created_at"],b["p63_policy_version"],b["p63_context_key"],
           b["p63_algorithm"],p63,p64,ret,b["bundle_sha256"],b["note"]))
        self.db.commit()
        return self.get_bundle(b["bundle_id"])
