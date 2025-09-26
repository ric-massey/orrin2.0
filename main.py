# main.py
from __future__ import annotations

import os
import time
import webbrowser
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
import threading
from dataclasses import asdict, is_dataclass  # ← added
import platform, shutil, subprocess  # ← added

from watchdogs import Pulse, start_watchdogs
from observability.metrics import serve_metrics
from observability.dashboard_server import start_dashboard_server
from observability import metrics  # Gauge: lifespan_cycles
# from reaper.errors import make_event_from_key  # optional

# --- Orrin memory subsystem imports ---
from memory.store.inmem import InMemoryStore
from memory.memory_daemon import MemoryDaemon
from memory.health import snapshot as memory_snapshot  # use the rich snapshot
from memory.wal import flush as wal_flush

# --- Goals subsystem imports (for the goals dashboard JSON feed) ---
from goals.api import GoalsAPI  # ← added
from goals.model import Goal  # ← added
from goals.store import FileGoalsStore  # ← added

# ---------- Metrics endpoint (Prometheus) ----------
METRICS_PORT = 9100  # http://127.0.0.1:9100/metrics
serve_metrics(port=METRICS_PORT)
print(f"[metrics] Prometheus exporter on http://127.0.0.1:{METRICS_PORT}/metrics")

# ---------- Dist paths (PORTABLE: no user-specific hard-coding) ----------
def _repo_root_from_file(this_file: str) -> Path:
    # If main.py lives in the repo root, .parent is correct.
    # If you later move main.py into a subdir, adjust to .parent.parent, etc.
    return Path(this_file).resolve().parent

REPO_ROOT = Path(os.environ.get("ORRIN_REPO_ROOT", _repo_root_from_file(__file__)))

# Allow environment overrides (useful for packaging/CI), otherwise resolve from repo structure
METRICS_DIST_DIR = Path(
    os.environ.get(
        "ORRIN_METRICS_DIST",
        REPO_ROOT / "UI" / "metrics-dashboard" / "dist",
    )
).resolve()

MEMORY_DIST_DIR = Path(
    os.environ.get(
        "ORRIN_MEMORY_DIST",
        REPO_ROOT / "UI" / "memory-dashboard" / "dist",
    )
).resolve()

GOALS_DIST_DIR = Path(  # ← added
    os.environ.get(
        "ORRIN_GOALS_DIST",
        REPO_ROOT / "UI" / "goals-dashboard" / "dist",
    )
).resolve()

def _require_dist(dist_dir: Path, ui_name: str) -> None:
    """Verify dist_dir/index.html exists, otherwise print a build hint relative to REPO_ROOT."""
    idx = dist_dir / "index.html"
    print(f"[{ui_name}] using dist_dir: {dist_dir}")
    print(f"[{ui_name}] dist_dir exists: {dist_dir.exists()}  index.html exists: {idx.exists()}")
    if not dist_dir.exists() or not idx.exists():
        # Build hint: if dist_dir is inside REPO_ROOT, show a clean relative path
        try:
            ui_rel = dist_dir.relative_to(REPO_ROOT)
            build_cd = REPO_ROOT / ui_rel.parent
        except ValueError:
            # dist_dir is outside REPO_ROOT or unrelated; just use its parent
            build_cd = dist_dir.parent
        raise SystemExit(
            f"[{ui_name}] UI build not found.\n"
            "Build it, then run again:\n"
            f"  cd {build_cd}\n"
            "  npm install\n"
            "  npm run build\n"
        )

# ---------- NEW: resilient builder (auto npm/brew install + build, non-crashing) ----------
def _ensure_ui_build(ui_name: str, dist_dir: Path) -> bool:
    """
    Ensure a Vite UI 'dist/index.html' exists.
    If missing, attempt: npm ci (or npm install) + npm run build.
    On macOS, if npm is missing, try 'brew install node' once.
    Returns True if dist is present (already or after build), else False.
    """
    idx = dist_dir / "index.html"
    print(f"[{ui_name}] using dist_dir: {dist_dir}")
    print(f"[{ui_name}] dist_dir exists: {dist_dir.exists()}  index.html exists: {idx.exists()}")

    if idx.exists():
        return True

    # Try to build
    ui_src_dir = dist_dir.parent  # e.g., UI/goals-dashboard
    print(f"[{ui_name}] dist missing → attempting local build in {ui_src_dir}")

    def _run(cmd: list[str]) -> bool:
        try:
            print(f"[{ui_name}] $ {' '.join(cmd)}")
            subprocess.run(cmd, cwd=str(ui_src_dir), check=True)
            return True
        except Exception as e:
            print(f"[{ui_name}] command failed: {e}")
            return False

    npm = shutil.which("npm")

    # If npm absent, try to install Node via Homebrew on macOS, then re-detect npm
    if npm is None and platform.system() == "Darwin":
        brew = shutil.which("brew")
        if brew:
            print(f"[{ui_name}] npm not found → trying 'brew install node'")
            _run([brew, "install", "node"])
            npm = shutil.which("npm")

    if npm is None:
        print(f"[{ui_name}] npm not found and could not auto-install. "
              f"Please install Node.js, then run: cd {ui_src_dir} && npm install && npm run build")
        return False

    # Install deps (prefer ci; fall back to install)
    if not _run([npm, "ci"]):
        if not _run([npm, "install"]):
            print(f"[{ui_name}] npm install failed.")
            return False

    # Build
    if not _run([npm, "run", "build"]):
        print(f"[{ui_name}] build failed.")
        return False

    ok = idx.exists()
    print(f"[{ui_name}] build result index.html exists: {ok}")
    return ok

# ---------- Ports ----------
DASH_PORT       = int(os.environ.get("ORRIN_DASH_PORT", "9310"))      # metrics UI
MEM_DASH_PORT   = int(os.environ.get("ORRIN_MEM_DASH_PORT", "9400"))  # memory UI
GOALS_DASH_PORT = int(os.environ.get("ORRIN_GOALS_DASH_PORT", "9500"))  # ← goals UI

# ---------- Helper: simple memory dashboard server (fallback if needed) ----------
def start_memory_server(dist_dir: str, port: int, memory_health_provider):
    """Serve static SPA from dist_dir and expose GET /memory."""
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=dist_dir, **kwargs)

        def do_GET(self):
            if self.path == "/memory":
                try:
                    data = memory_health_provider() or {}
                    body = json.dumps(data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as e:
                    msg = json.dumps({"error": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                return
            return super().do_GET()

    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    t = threading.Thread(target=httpd.serve_forever, name=f"memory-ui:{port}", daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}/"
    return t, httpd, url

# ---------- Helper: goals JSON encoding + provider (for the goals dashboard) ----------
# Where goals data (JSONL/WAL) lives — matches CLI default unless overridden
GOALS_DATA_DIR = Path(os.environ.get("ORRIN_GOALS_DIR", REPO_ROOT / "data" / "goals")).resolve()
GOALS_DATA_DIR.mkdir(parents=True, exist_ok=True)

_goal_store = FileGoalsStore(data_dir=GOALS_DATA_DIR)  # ← fixed
_goals_api = GoalsAPI(store=_goal_store)

def _goal_to_jsonable(g: Goal) -> dict:
    d = g.__dict__.copy()
    # enums -> names
    d["status"] = getattr(g.status, "name", str(g.status))
    d["priority"] = getattr(g.priority, "name", str(g.priority))
    # datetimes -> ISO
    if d.get("deadline"):    d["deadline"]    = g.deadline.isoformat()
    if d.get("created_at"):  d["created_at"]  = g.created_at.isoformat()
    if d.get("updated_at"):  d["updated_at"]  = g.updated_at.isoformat()
    # dataclass progress -> dict
    pr = d.get("progress")
    if is_dataclass(pr):
        d["progress"] = asdict(pr)
    # normalize mapping types
    if d.get("acceptance") is not None:
        d["acceptance"] = dict(d["acceptance"])
    if d.get("spec") is not None:
        d["spec"] = dict(d["spec"])
    return d

def get_goals_json():
    # Extend with filters later if needed (?status=, ?priority=, etc.)
    goals = _goals_api.list_goals()
    return [_goal_to_jsonable(g) for g in goals]

# ---------- Helper: simple goals dashboard server ----------
def start_goals_server(dist_dir: str, port: int, goals_provider):
    """Serve static SPA from dist_dir and expose GET /goals and /goals.json."""
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=dist_dir, **kwargs)

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/goals", "/goals.json"):
                try:
                    data = goals_provider() or []
                    self._send_json(data, 200)
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
                return
            return super().do_GET()

    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    t = threading.Thread(target=httpd.serve_forever, name=f"goals-ui:{port}", daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}/"
    return t, httpd, url

# ---------- Start Metrics Dashboard ----------
dash_thread = dash_httpd = None
try:
    if _ensure_ui_build("dashboard", METRICS_DIST_DIR):
        dash_thread, dash_httpd, dash_url = start_dashboard_server(
            dist_dir=str(METRICS_DIST_DIR),
            port=DASH_PORT,
            metrics_upstream=f"http://127.0.0.1:{METRICS_PORT}/metrics",
            open_browser=True,  # auto-open metrics tab
        )
        print(f"[dashboard] serving at {dash_url}")
    else:
        print("[dashboard] skipped (dist not available)")
except Exception as e:
    print(f"[dashboard] not started: {e}")

# ---------- Memory subsystem ----------
store = InMemoryStore()
daemon = MemoryDaemon(store)
daemon.start()
print("[memory] MemoryDaemon started with InMemoryStore")

# Robust /memory provider (rich fields for the UI)
def get_memory_health():
    """
    Returns a dict with common keys the UI expects, plus the rich snapshot:
    - daemon_alive, store_type, items, bytes, wal_enabled, wal_queue, wal_lag_s
    - status, signals, notes, store_stats (flattened)
    """
    # Helpers
    def _getattr(obj, name, default=None):
        try:
            val = getattr(obj, name, default)
            return val() if callable(val) else val
        except Exception:
            return default

    # Daemon basics
    try:
        alive = daemon.is_alive() if hasattr(daemon, "is_alive") else bool(getattr(daemon, "thread", None) and daemon.thread.is_alive())
    except Exception:
        alive = False

    # Optional daemon hints for snapshot
    working_cache_size = int(_getattr(daemon, "working_cache_size", 0) or 0)
    last_compaction_ts = float(_getattr(daemon, "last_compaction_ts", 0.0) or 0.0)
    flush_failures     = int(_getattr(daemon, "flush_failures", 0) or 0)

    # Rich snapshot (status, signals, notes, store_stats dataclass)
    snap = {}
    try:
        snap = memory_snapshot(
            store,
            working_cache_size=working_cache_size,
            last_compaction_ts=last_compaction_ts,
            flush_failures=flush_failures,
        ) or {}
    except Exception as e:
        snap = {"status": "snapshot_error", "error": str(e)}

    # Flatten store_stats if it is a dataclass / namedtuple
    ss = snap.get("store_stats")
    if ss is not None:
        try:
            if hasattr(ss, "__dict__"):
                snap["store_stats"] = ss.__dict__
            elif hasattr(ss, "_asdict"):
                snap["store_stats"] = ss._asdict()
        except Exception:
            pass

    # Derive simple top-level fields
    items = None
    bytes_total = None
    if isinstance(snap.get("store_stats"), dict):
        items = snap["store_stats"].get("items_total")
        # Prefer a total byte-like field if present; fall back to 0
        for key in ("vector_bytes_total", "bytes_total", "approx_bytes", "mem_bytes"):
            if key in snap["store_stats"] and isinstance(snap["store_stats"][key], (int, float)):
                bytes_total = int(snap["store_stats"][key])
                break

    # WAL-ish hints from store/daemon if available
    wal_enabled = bool(_getattr(store, "wal_enabled", False) or _getattr(daemon, "wal_enabled", False) or False)
    wal_queue   = _getattr(store, "wal_queue", None)
    if isinstance(wal_queue, bool):  # bad type guard
        wal_queue = None
    try:
        wal_queue = int(wal_queue) if wal_queue is not None else None
    except Exception:
        wal_queue = None

    # last flush age (seconds) if available
    wal_lag_s = _getattr(store, "wal_last_flush_age_s", None)
    if wal_lag_s is None:
        # derive from last timestamp if present
        ts = _getattr(store, "wal_last_flush_ts", None)
        if ts:
            try:
                import time as _t
                wal_lag_s = max(0.0, _t.time() - float(ts))
            except Exception:
                wal_lag_s = None

    out = {
        "daemon_alive": bool(alive),
        "store_type": type(store).__name__,
        "items": items,
        "bytes": bytes_total,
        "wal_enabled": wal_enabled,
        "wal_queue": wal_queue,
        "wal_lag_s": wal_lag_s,
        **snap,  # keep original rich fields
    }
    return out

# ---------- Start Memory Dashboard (with fallback if server lacks memory_health_provider) ----------
mem_dash_thread = mem_dash_httpd = None
try:
    if _ensure_ui_build("memory-dashboard", MEMORY_DIST_DIR):
        try:
            # Preferred: server supports /memory via memory_health_provider
            mem_dash_thread, mem_dash_httpd, mem_dash_url = start_dashboard_server(
                dist_dir=str(MEMORY_DIST_DIR),
                port=MEM_DASH_PORT,
                metrics_upstream=f"http://127.0.0.1:{METRICS_PORT}/metrics",
                memory_health_provider=get_memory_health,  # serve /memory here
                open_browser=True,
            )
            print("[memory-dashboard] /memory endpoint wired via memory_health_provider")
        except TypeError:
            # Fallback: our small server that always exposes /memory
            print("[memory-dashboard] start_dashboard_server lacks memory_health_provider → using fallback server")
            mem_dash_thread, mem_dash_httpd, mem_dash_url = start_memory_server(
                dist_dir=str(MEMORY_DIST_DIR),
                port=MEM_DASH_PORT,
                memory_health_provider=get_memory_health,
            )
            webbrowser.open(mem_dash_url)
        print(f"[memory-dashboard] serving at {mem_dash_url}")
    else:
        print("[memory-dashboard] skipped (dist not available)")
except Exception as e:
    print(f"[memory-dashboard] not started: {e}")

# ---------- Start Goals Dashboard ----------
goals_thread = goals_httpd = None
try:
    if _ensure_ui_build("goals-dashboard", GOALS_DIST_DIR):
        goals_thread, goals_httpd, goals_url = start_goals_server(
            dist_dir=str(GOALS_DIST_DIR),
            port=GOALS_DASH_PORT,
            goals_provider=get_goals_json,
        )
        try:
            webbrowser.open(goals_url)
        except Exception:
            pass
        print(f"[goals-dashboard] serving at {goals_url}")
    else:
        print("[goals-dashboard] skipped (dist not available)")
except Exception as e:
    print(f"[goals-dashboard] not started: {e}")

# ---------- Watchdogs ----------
pulse = Pulse()

# Call start_watchdogs with a fallback if it doesn't accept the newer args
try:
    tup = start_watchdogs(
        pulse,
        per_key_limits={"llm_timeout": (10, 15.0)},
        get_memory_health=get_memory_health,   # newer versions
        # ↓ enable 5 Hz nervous-system sampling + compact summaries into memory
        memory_daemon=daemon,
        ns_sample_interval_s=0.2,              # sample every 0.2s
        ns_summary_interval_s=5.0,             # write summary events every 5s
    )
except TypeError:
    # Older signature: call without the new args (sampler won’t run in that build)
    tup = start_watchdogs(
        pulse,
        per_key_limits={"llm_timeout": (10, 15.0)},
    )


(
    reaper,        # kill switch (reaper.trigger("reason"))
    detector,      # heartbeat guard (period too fast/slow)
    errors,        # error repetition + rate-limit guard
    liveness,      # section freshness-by-cycles (register/touch sections)
    lifespan,      # random hard cutoff by cycles
    no_goals,      # goals/saturation guard (None unless providers were passed)
    mem_guard,     # memory/FD/CPU guard (may be None on older versions)
    repeat_guard,  # repetition loop guard
    stop_evt,      # event to stop the watchdog thread
) = tup

def run() -> None:
    last_log = 0
    try:
        while True:
            # heartbeat/liveness
            pulse.tick()

            # publish current cycle so metrics UI can display it
            n = pulse.read()
            try:
                metrics.lifespan_cycles.set(float(n))  # Gauge("orrin_lifespan_cycles", ...)
            except Exception:
                pass
            try:
                # Optional backup gauge if you added one
                metrics.cycle_gauge.set(float(n))  # ignore if not defined
            except Exception:
                pass

            # --- your main loop work goes here ---
            time.sleep(0.02)

            # light heartbeat log every ~2s
            last_log += 1
            if last_log >= 100:
                print(f"[main] cycle={n}")
                last_log = 0

    except KeyboardInterrupt:
        print("\n[main] Ctrl+C received; shutting down…")
    finally:
        # stop the watchdog thread
        try:
            stop_evt.set()
        except Exception:
            pass

        # stop memory daemon + flush WAL
        try:
            daemon.stop(join=True)
        except Exception:
            pass
        try:
            wal_flush()
        except Exception:
            pass

        # stop the dashboard servers
        if dash_httpd is not None:
            try:
                dash_httpd.shutdown()
            except Exception:
                pass
        if mem_dash_httpd is not None:
            try:
                mem_dash_httpd.shutdown()
            except Exception:
                pass
        if goals_httpd is not None:  # ← added
            try:
                goals_httpd.shutdown()
            except Exception:
                pass

        print("[main] shutdown complete.")

if __name__ == "__main__":
    run()
