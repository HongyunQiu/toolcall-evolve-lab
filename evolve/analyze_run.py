#!/usr/bin/env python3
"""Analyze a single oss120b toolcall run log and produce failure tags + summary.

Usage:
  python3 evolve/analyze_run.py evolve/runs/<run>.json

Outputs JSON to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _get_errmsg(run: Dict[str, Any]) -> str:
    raw = run.get("raw") or run.get("raw_last") or {}
    if isinstance(raw, dict):
        err = raw.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str):
                return msg
    return ""


def analyze(run: Dict[str, Any]) -> Dict[str, Any]:
    tags: List[str] = []

    ok = bool(run.get("ok"))
    final = (run.get("final") or "").strip()
    saw_final = bool(run.get("saw_final_answer"))

    if not saw_final and final:
        tags.append("noncompliant_no_final_tool_but_has_text")
    if not saw_final and not final:
        tags.append("noncompliant_no_final_tool")

    if final == "TOOL_FAIL":
        tags.append("tool_fail")

    err = (run.get("error") or "")
    if isinstance(err, str) and err.startswith("HTTP_"):
        tags.append("http_error")

    msg = _get_errmsg(run)
    if "Unknown channel" in msg:
        tags.append("vllm_unknown_channel")
    if "unexpected tokens remaining" in msg:
        tags.append("vllm_parser_glitch")

    # Tool-level errors
    executed = run.get("executed") or []
    if isinstance(executed, list):
        for e in executed:
            res = (e or {}).get("result") or {}
            if isinstance(res, dict) and res.get("ok") is False:
                et = (e or {}).get("tool")
                if et == "run_cli":
                    tags.append("run_cli_failed")
                    er = res.get("error")
                    if isinstance(er, str) and er.startswith("command_not_allowed"):
                        tags.append("run_cli_command_not_allowed")
                    if er == "shell_metacharacters_forbidden":
                        tags.append("run_cli_shell_metacharacters_forbidden")
                elif et == "fetch_url":
                    tags.append("fetch_url_failed")
                elif et == "write_file":
                    tags.append("write_file_failed")
                elif et == "read_file":
                    tags.append("read_file_failed")

    # De-dupe
    tags = sorted(set(tags))

    return {
        "ok": ok,
        "final": final,
        "saw_final_answer": saw_final,
        "tags": tags,
        "error": run.get("error"),
        "errmsg": msg,
    }


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: analyze_run.py <run.json>", file=sys.stderr)
        raise SystemExit(2)

    p = Path(sys.argv[1])
    run = json.loads(p.read_text("utf-8"))
    out = analyze(run)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
