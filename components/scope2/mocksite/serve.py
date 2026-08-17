"""Serve the mock portal with caching disabled.

`python -m http.server` lets the browser cache portal.js, so editing the portal
and reloading can leave the old script running - the page looks unchanged, and
an automation built against it fails with "element not found" for a reason that
is nowhere on screen. That cost real time twice while setting up the RPA
comparison, once for me and once for the operator.

Every response here carries no-store, so a reload is always the current file.

Usage:
    python mocksite/serve.py            # http://127.0.0.1:8765
    python mocksite/serve.py --port 9000
"""

import argparse
import functools
import http.server
import socketserver
from pathlib import Path

MOCKSITE = Path(__file__).resolve().parent


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # the request log is noise while driving a browser by hand


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    handler = functools.partial(NoCacheHandler, directory=str(MOCKSITE))
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer((args.host, args.port), handler) as server:
        print(f"mock portal on http://{args.host}:{args.port}  (caching disabled)")
        print(f"  v0_base   http://{args.host}:{args.port}/v0_base/index.html")
        print("Ctrl+C to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
