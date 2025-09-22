# tests/observability/test_dashboard_server.py
from __future__ import annotations

import http.server
import socketserver
import threading
import time
import socket
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from pathlib import Path

import pytest

from observability.dashboard_server import start_dashboard_server


# ---------- helpers ----------

def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    body: bytes = b"# HELP demo Demo metrics\n# TYPE demo counter\ndemo 1\n"
    status: int = 200
    content_type: str = "text/plain; version=0.0.4"

    # keep quiet
    def log_message(self, fmt, *args):  # noqa: N802
        pass

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/metrics":
            self.send_response(self.status)
            self.send_header("Content-Type", self.content_type)
            self.send_header("Content-Length", str(len(self.body)))
            self.end_headers()
            self.wfile.write(self.body)
        else:
            self.send_response(404)
            self.end_headers()


class _ThreadedHTTP:
    """Context manager for a tiny metrics upstream server."""
    def __init__(self, handler_cls=_MetricsHandler, host="127.0.0.1", port: int | None = None):
        self.handler_cls = handler_cls
        self.host = host
        self.port = port or _get_free_port()
        self.httpd: socketserver.TCPServer | None = None
        self.t: threading.Thread | None = None
        self.url = f"http://{self.host}:{self.port}"

    def __enter__(self):
        self.httpd = socketserver.TCPServer((self.host, self.port), self.handler_cls)
        def serve():
            with self.httpd:
                self.httpd.serve_forever(poll_interval=0.2)
        self.t = threading.Thread(target=serve, name="fake-metrics", daemon=True)
        self.t.start()
        # tiny wait so the socket is actually listening
        time.sleep(0.05)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.httpd:
            self.httpd.shutdown()
        if self.t:
            self.t.join(timeout=2.0)


def _fetch(url: str) -> tuple[int, bytes, dict]:
    """Return (status_code, body, headers) using urllib."""
    try:
        with urlopen(url, timeout=3) as r:
            return (200, r.read(), dict(r.headers.items()))
    except HTTPError as e:
        # read body for diagnostics
        body = e.read() or b""
        return (e.code, body, dict(e.headers.items()))


# ---------- tests ----------

def test_serves_static_index(tmp_path: Path):
    # Create a fake built SPA
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><h1>Hello Dashboard</h1>", encoding="utf-8")

    port = _get_free_port()
    # Upstream can be anything; we won't hit /metrics in this test
    _, httpd, url = start_dashboard_server(
        dist_dir=str(dist),
        port=port,
        metrics_upstream="http://127.0.0.1:9/metrics",  # guaranteed closed port
        open_browser=False,
    )
    try:
        status, body, _ = _fetch(url)
        assert status == 200
        assert b"Hello Dashboard" in body
    finally:
        httpd.shutdown()

def test_fallback_index_when_dist_missing(tmp_path: Path):
    dist = tmp_path / "does_not_exist"

    port = _get_free_port()
    _, httpd, url = start_dashboard_server(
        dist_dir=str(dist),
        port=port,
        metrics_upstream="http://127.0.0.1:9/metrics",
        open_browser=False,
    )
    try:
        status, body, _ = _fetch(url)
        assert status == 200
        assert b"orrin dashboard" in body  # fallback page text
        # fallback index.html should be created
        assert (dist / "index.html").exists()
    finally:
        httpd.shutdown()

def test_metrics_proxy_happy_path(tmp_path: Path):
    # Start fake upstream that serves /metrics 200 with a tiny payload
    with _ThreadedHTTP(_MetricsHandler) as upstream:
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")

        port = _get_free_port()
        _, httpd, url = start_dashboard_server(
            dist_dir=str(dist),
            port=port,
            metrics_upstream=f"{upstream.url}/metrics",
            open_browser=False,
        )
        try:
            status, body, headers = _fetch(url + "metrics")
            assert status == 200
            assert b"# HELP demo" in body
            # should preserve content-type if set upstream
            assert "text/plain" in headers.get("Content-Type", "")
        finally:
            httpd.shutdown()

def test_metrics_proxy_returns_502_if_upstream_unreachable(tmp_path: Path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")

    port = _get_free_port()
    # Point to a closed port for upstream
    _, httpd, url = start_dashboard_server(
        dist_dir=str(dist),
        port=port,
        metrics_upstream="http://127.0.0.1:9/metrics",
        open_browser=False,
    )
    try:
        status, body, _ = _fetch(url + "metrics")
        assert status == 502
        assert b"upstream" in body or b"proxy error" in body
    finally:
        httpd.shutdown()

def test_shutdown_is_clean(tmp_path: Path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("ok", encoding="utf-8")

    port = _get_free_port()
    thread, httpd, url = start_dashboard_server(
        dist_dir=str(dist),
        port=port,
        metrics_upstream="http://127.0.0.1:9/metrics",
        open_browser=False,
    )
    # quick sanity
    status, _, _ = _fetch(url)
    assert status == 200

    httpd.shutdown()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
