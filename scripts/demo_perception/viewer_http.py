"""HTTP adapter over the viewer snapshot; never commands a robot."""

import json
from http.server import BaseHTTPRequestHandler

from .viewer_page import HTML


def make_handler(viewer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                content, mime = HTML.encode(), "text/html; charset=utf-8"
            elif path == "/status":
                content, mime = json.dumps(viewer.state()).encode(), "application/json"
            elif path == "/depth/left_rim.jpg":
                with viewer.lock:
                    content = viewer.rim["jpeg"]
                if content is None:
                    self.send_error(503, "Waiting for rim image")
                    return
                mime = "image/jpeg"
            elif path == "/depth/left_candidate.jpg":
                with viewer.lock:
                    content = viewer.candidate["jpeg"]
                if content is None:
                    self.send_error(503, "Waiting for candidate image")
                    return
                mime = "image/jpeg"
            elif path == "/depth/left_filtered.jpg":
                with viewer.lock:
                    content = viewer.filtered["jpeg"]
                if content is None:
                    self.send_error(503, "Waiting for numeric depth")
                    return
                mime = "image/jpeg"
            elif path in ("/depth/head.jpg", "/depth/left.jpg", "/depth/right.jpg"):
                camera = path.split("/")[-1].removesuffix(".jpg")
                with viewer.lock:
                    content = viewer.depth[camera]["jpeg"]
                if content is None:
                    self.send_error(503, "Waiting for depth image")
                    return
                mime = "image/jpeg"
            elif path == "/image.jpg":
                with viewer.lock:
                    content = viewer.jpeg
                if content is None:
                    self.send_error(503, "Waiting for image")
                    return
                mime = "image/jpeg"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if self.path != "/filter":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                viewer.set_filter(float(body["min_m"]), float(body["max_m"]))
                status, data = 200, {"ok": True}
            except (ValueError, TypeError, KeyError) as error:
                status, data = 400, {"error": str(error)}
            content = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_args):
            pass

    return Handler
