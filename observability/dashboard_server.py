# observability/dashboard_server.py
# Simple HTTP server to serve a built SPA and proxy /metrics to a Prometheus metrics endpoint.

from __future__ import annotations
import http.server
import socketserver
import threading
import webbrowser
import os
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from typing import Tuple, Optional

class _ProxyAndStaticHandler(http.server.SimpleHTTPRequestHandler):
    """Serves static files from `static_dir` and proxies GET /metrics to metrics_upstream."""
    static_dir: str = "."
    metrics_upstream: str = "http://127.0.0.1:9100/metrics"

    # Silence default noisy logging
    def log_message(self, fmt, *args):  # noqa: N802
        pass

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/metrics":
            self._proxy_metrics()
            return
        return super().do_GET()

    def _proxy_metrics(self):
        try:
            req = Request(self.metrics_upstream, headers={"User-Agent": "orrin-dashboard/1.0"})
            with urlopen(req, timeout=5) as resp:
                data = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", resp.headers.get_content_type() or "text/plain")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except HTTPError as e:
            body = (e.read() or b"").decode("utf-8", "replace")
            payload = f"upstream HTTP {e.code}\n{body}".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except URLError as e:
            msg = f"upstream not reachable: {e}".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
        except Exception as e:
            msg = f"proxy error: {e}".encode()
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

def start_dashboard_server(
    dist_dir: str,
    port: int = 9300,
    metrics_upstream: str = "http://127.0.0.1:9100/metrics",
    open_browser: bool = True,
    host: str = "127.0.0.1",
) -> Tuple[threading.Thread, socketserver.TCPServer, str]:
    """
    Serve a built SPA from `dist_dir` and proxy /metrics to `metrics_upstream`.

    Returns: (thread, httpd, url)
    """
    # If dist_dir doesn’t exist, provide a one-file fallback that links to /metrics
    if not os.path.isdir(dist_dir):
        os.makedirs(dist_dir, exist_ok=True)
        index_html = os.path.join(dist_dir, "index.html")
        if not os.path.isfile(index_html):
            with open(index_html, "w", encoding="utf-8") as f:
                f.write(
                    """<!doctype html><meta charset="utf-8">
<title>orrin metrics</title>
<h1>orrin dashboard</h1>
<p>Build the UI to see graphs, or check raw <a href="/metrics">/metrics</a>.</p>
<p>To build the UI:</p>
<pre>cd UI/metrics-dashboard
npm install
npm run build</pre>
"""
                )

    # chdir for SimpleHTTPRequestHandler to serve from dist_dir
    # we do this in a thread-local way by setting class variable and using directory=...
    Handler = _ProxyAndStaticHandler
    Handler.metrics_upstream = metrics_upstream

    # Python 3.7+: SimpleHTTPRequestHandler supports 'directory' kw
    def handler_factory(*args, **kwargs):
        return Handler(*args, directory=dist_dir, **kwargs)

    httpd = socketserver.TCPServer((host, port), handler_factory)
    url = f"http://{host}:{port}/"

    def serve():
        with httpd:
            httpd.serve_forever(poll_interval=0.5)

    t = threading.Thread(target=serve, name="dashboard-server", daemon=True)
    t.start()

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return t, httpd, url
