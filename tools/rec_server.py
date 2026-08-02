#!/usr/bin/env python3
"""Tiny PUT receiver for the game's self-recorder. Saves bodies to promo/recordings/."""
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

OUT = os.path.join(os.path.dirname(__file__), "..", "promo", "recordings")
os.makedirs(OUT, exist_ok=True)


class H(BaseHTTPRequestHandler):
    def do_PUT(self):
        name = parse_qs(urlparse(self.path).query).get("name", ["out.webm"])[0]
        name = os.path.basename(name)
        n = int(self.headers.get("Content-Length", 0))
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(self.rfile.read(n))
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        print("saved", name, n, "bytes", flush=True)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", 8042), H).serve_forever()
