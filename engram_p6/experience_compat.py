"""Internal P6.1 event writer for the P2 feedback bridge."""
import json, sqlite3
from datetime import datetime, timezone

def write_feedback(db: sqlite3.Connection, source_event_id: str,
                   candidate_id: str, feedback: str, query_key: str):
    allowed={"helpful","irrelevant","wrong","outdated","not_about_me","needs_context","confirm"}
    if feedback not in allowed: return False
    now=datetime.now(timezone.utc).isoformat()
    db.execute("""INSERT OR IGNORE INTO p6_experience_events
      (event_id,idempotency_key,event_type,occurred_at,context_key,query_kind,
       candidate_refs,selected_refs,target_ref,signal,signal_value,
       source_event_ref,metadata,schema_version)
      VALUES(?,?, 'feedback', ?, ?, 'p2_feedback', '[]','[]',?,?,NULL,?,'{}',1)""",
      ("p6e_p2_"+source_event_id,"p2-feedback:"+source_event_id,now,
       query_key,candidate_id,feedback,source_event_id))
    db.commit()
    return True
