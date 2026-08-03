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
        name = os.path.basename(self.path.split("?")[0])
        if not (self.path.startswith("/charts/") and name.endswith(".json")):
            self.send_error(403)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            json.loads(body)                      # refuse to save broken JSON
        except ValueError:
            self.send_error(400, "invalid JSON")
            return
        with open(os.path.join(ROOT, "charts", name), "wb") as f:
            f.write(body)
        self.send_response(200)
        self.end_headers()
        print("saved", name, n, "bytes", flush=True)

    def log_message(self, *a):
        pass


print("editor server: http://localhost:8043/editor.html")
ThreadingHTTPServer(("127.0.0.1", 8043), H).serve_forever()
