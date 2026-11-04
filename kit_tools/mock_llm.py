"""Stand-in for an OpenAI-compatible model, used only to check vps_kit.py runs end to end."""
import json, re, sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        p = b["messages"][0]["content"]
        if "Reply with exactly one word" in p:
            s = p.split("Sentence:")[-1].lower()
            r = "home" if re.search(r"home|base|address", s) else "job" if re.search(r"employer|office|role", s) else "neither"
        elif "Write the lasting facts" in p:
            r = "\n".join(f"{m[1]} said something. [{m[0]}]" for m in re.findall(r"^\[(D\d+:\d+)\] (\w+):", p, re.M)[:5])
        elif "mean the same" in p: r = "yes"
        elif "handover note for whoever" in p: r = "Goal: test.\nImportant facts: mock note."
        elif "Excerpts:" in p: r = "7 May 2023"
        else: r = "ready"
        o = json.dumps(dict(choices=[dict(message=dict(content=r))], usage=dict(prompt_tokens=len(p) // 4, completion_tokens=len(r) // 4))).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(o)


HTTPServer(("127.0.0.1", 18998), H).serve_forever()
