"""Safe-ish Python sandbox.

This is NOT a security boundary — the LLM is trusted code running on
your machine. It IS a stability boundary: timeouts + memory limits keep
a runaway loop from freezing your laptop.

On Linux we use resource.setrlimit. On Windows we rely on subprocess
timeout + optional psutil memory polling.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_py(
    code: str,
    *,
    timeout_s: int = 60,
    mem_mb: int = 2048,
    workdir: str | None = None,
    extra_env: dict | None = None,
) -> dict:
    """Execute `code` in a subprocess and capture stdout/stderr.

    Returns {stdout, stderr, returncode, timed_out}.
    """
    cwd = Path(workdir) if workdir else Path.cwd()
    cwd.mkdir(parents=True, exist_ok=True)

    # Save code to temp file for better tracebacks
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=str(cwd), delete=False) as f:
        f.write(code)
        script_path = f.name

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    preexec = None
    if os.name == "posix":
        def _limit():
            import resource
            resource.setrlimit(
                resource.RLIMIT_AS,
                (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024),
            )
        preexec = _limit

    timed_out = False
    try:
        r = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd),
            env=env,
            preexec_fn=preexec,
        )
        result = {
            "stdout": r.stdout,
            "stderr": r.stderr,
            "returncode": r.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        timed_out = True
        result = {
            "stdout": (e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            "stderr": f"TIMEOUT after {timeout_s}s\n"
                      + ((e.stderr or b"").decode(errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or "")),
            "returncode": -1,
            "timed_out": True,
        }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
    return result


# Alias used by task_conversation
run_code_sync = run_py
