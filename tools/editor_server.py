#!/usr/bin/env python3
"""Serves the game folder AND accepts chart saves from the editor.

    .venv/bin/python tools/editor_server.py     # http://localhost:8043

GET  — static files (game + editor both work from this port)
PUT  /charts/<name>.json — writes the chart (charts/ only, .json only)
"""
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


class H(SimpleHTTPRequestHandler):
    def do_PUT(self):
        path = self.path.split("?")[0]
        name = os.path.basename(path)
        draft = path.startswith("/charts/drafts/")
        if not ((draft or path.startswith("/charts/")) and name.endswith(".json")):
            self.send_error(403)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            json.loads(body)                      # refuse to save broken JSON
        except ValueError:
            self.send_error(400, "invalid JSON")
            return
        sub = os.path.join("charts", "drafts") if draft else "charts"
        os.makedirs(os.path.join(ROOT, sub), exist_ok=True)
        with open(os.path.join(ROOT, sub, name), "wb") as f:
            f.write(body)
        self.send_response(200)
        self.end_headers()
        print("saved", os.path.join(sub, name), n, "bytes", flush=True)

    def log_message(self, *a):
        pass


print("editor server: http://localhost:8043/editor.html")
ThreadingHTTPServer(("127.0.0.1", 8043), H).serve_forever()
