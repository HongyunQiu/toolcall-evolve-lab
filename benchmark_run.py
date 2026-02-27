#!/usr/bin/env python3
"""Benchmark runner for toolcall-evolve-lab.

- Loads a task set (JSON array) from tasks/tasks_v1.json (or --tasks).
- For each task, runs oss120b_toolcall_run.py in a fresh sandbox (runner default).
- Verifies outputs by inspecting the saved run_log and reading files from the sandbox.

This benchmark is intentionally simple and self-contained.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text("utf-8"))


def _run_cmd(cmd: List[str], cwd: Path, timeout_sec: int | None) -> Tuple[int, str, str, bool]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_sec)
        return p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else ""
        return 124, out, err + "\n[TIMEOUT]", True


def _extract_run_log_path(stderr: str) -> str | None:
    for line in (stderr or "").splitlines():
        if line.startswith("[run_log]"):
            return line.split("[run_log]", 1)[1].strip()
    return None


def _load_sandbox(run_log_path: Path) -> Path:
    run = _read_json(run_log_path)
    sb = run.get("sandbox")
    if not sb:
        raise RuntimeError("missing sandbox in run log")
    return Path(sb)


def _file_text(sb: Path, rel: str) -> str:
    return (sb / rel).read_text("utf-8", errors="replace")


def _verify(task_obj: Dict[str, Any], run_log_path: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    run = _read_json(run_log_path)
    sb = _load_sandbox(run_log_path)

    # Basic: should have a final tool call unless TOOL_FAIL
    final = (run.get("final") or "").strip()
    if not final:
        errors.append("empty final")

    required_files = task_obj.get("required_files") or []
    for f in required_files:
        fp = sb / f
        if not fp.exists():
            errors.append(f"missing file: {f}")
            continue
        if fp.stat().st_size <= 0:
            errors.append(f"empty file: {f}")

    must_include = task_obj.get("must_include") or {}
    for f, needles in must_include.items():
        try:
            txt = _file_text(sb, f)
        except Exception:
            errors.append(f"cannot read: {f}")
            continue
        for n in needles:
            if n not in txt:
                errors.append(f"{f} missing substring: {n}")

    content_regex = task_obj.get("content_regex") or {}
    for f, pat in content_regex.items():
        try:
            txt = _file_text(sb, f)
        except Exception:
            errors.append(f"cannot read: {f}")
            continue
        if not re.search(pat, txt, flags=re.M):
            errors.append(f"{f} regex not match: {pat}")

    json_keys = task_obj.get("json_keys") or {}
    for f, keys in json_keys.items():
        try:
            obj = json.loads(_file_text(sb, f))
        except Exception as e:
            errors.append(f"{f} invalid json: {e}")
            continue
        for k in keys:
            if k not in obj:
                errors.append(f"{f} missing key: {k}")

    # If model produced TOOL_FAIL, treat as fail.
    if final == "TOOL_FAIL":
        errors.append("final is TOOL_FAIL")

    return (len(errors) == 0), errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks" / "tasks_v1.json"))
    ap.add_argument("--base-url", default=os.environ.get("OSS120B_BASE_URL", "http://172.24.168.225:8389/v1"))
    ap.add_argument("--model", default=os.environ.get("OSS120B_MODEL", "openai/gpt-oss-120b"))
    ap.add_argument("--allow-any-cli", action="store_true")
    ap.add_argument("--auto-fix", action="store_true", default=True)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).parent / "benchmark_results.json"))
    ap.add_argument("--timeout-sec", type=int, default=600, help="Per-task subprocess timeout (default 600s).")
    ap.add_argument("--local-retry", action="store_true", help="Forward --local-retry to the runner.")
    ap.add_argument("--local-retry-max", type=int, default=1, help="Forward --local-retry-max to the runner.")
    args = ap.parse_args()

    repo = Path(__file__).parent
    tasks = _read_json(Path(args.tasks))

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    for t in tasks:
        tid = t["id"]
        cmd = [
            sys.executable,
            str(repo / "oss120b_toolcall_run.py"),
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--task",
            t["task"],
            "--auto-fix",
            "--max-rounds",
            str(int(t.get("max_rounds", args.max_rounds))),
            "--max-steps",
            str(int(t.get("max_steps", 12))),
        ]
        if args.allow_any_cli or t.get("allow_any_cli"):
            cmd.append("--allow-any-cli")
        if args.local_retry:
            cmd.append("--local-retry")
            cmd.extend(["--local-retry-max", str(int(args.local_retry_max))])

        rc, out, err, timed_out = _run_cmd(cmd, cwd=repo, timeout_sec=int(args.timeout_sec) if args.timeout_sec else None)
        run_log = _extract_run_log_path(err)

        item: Dict[str, Any] = {
            "id": tid,
            "category": t.get("category"),
            "rc": rc,
            "timed_out": timed_out,
            "stdout": (out or "").strip()[:5000],
            "stderr_tail": "\n".join((err or "").splitlines()[-30:]),
            "run_log": run_log,
        }

        if run_log:
            ok, verr = _verify(t, Path(run_log))
            item["pass"] = ok
            item["verify_errors"] = verr
        else:
            item["pass"] = False
            item["verify_errors"] = ["missing run_log in stderr"]

        results.append(item)
        print(f"{tid}: {'PASS' if item['pass'] else 'FAIL'}")

    summary = {
        "tasks": len(results),
        "pass": sum(1 for r in results if r.get("pass")),
        "fail": sum(1 for r in results if not r.get("pass")),
        "elapsed_sec": round(time.time() - t0, 2),
        "results": results,
    }

    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nWrote: {args.out}")
    print(f"Pass rate: {summary['pass']}/{summary['tasks']}")


if __name__ == "__main__":
    main()
