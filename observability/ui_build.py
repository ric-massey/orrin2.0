# observability/ui_build.py
from __future__ import annotations
from pathlib import Path
import platform, shutil, subprocess

def ensure_ui_build(ui_name: str, dist_dir: Path) -> bool:
    """
    Ensure a Vite UI 'dist/index.html' exists.
    If missing, attempt npm ci/install + npm run build.
    On macOS, if npm is missing, try 'brew install node'.
    Returns True if dist is present (already or after build), else False.
    """
    idx = dist_dir / "index.html"
    print(f"[{ui_name}] using dist_dir: {dist_dir}")
    print(f"[{ui_name}] dist_dir exists: {dist_dir.exists()}  index.html exists: {idx.exists()}")

    if idx.exists():
        return True

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
