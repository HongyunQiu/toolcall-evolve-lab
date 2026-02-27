#!/usr/bin/env python3
"""LLM-as-judge verifier for toolcall-evolve-lab.

Reads a saved run log JSON (from oss120b_toolcall_run.py) and asks the *same model*
(as configured by OSS120B_BASE_URL/OSS120B_MODEL) to judge PASS vs REDO.

Output: prints the judge response text (brief analysis + DECISION + PATCH_JSON line).

This script is intentionally lightweight and only depends on httpx.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx


def _post_response(client: httpx.Client, base_url: str, api_key: str | None, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/responses"
    r = client.post(url, headers=headers, json=payload, timeout=180)
    try:
        j = r.json()
    except Exception:
        j = {"_raw": r.text}
    return r.status_code, j


def _extract_output_text(resp_json: Dict[str, Any]) -> str:
    out_parts: List[str] = []
    for item in resp_json.get("output", []) or []:
        itype = item.get("type")
        if itype in ("output_text", "text") and isinstance(item.get("text"), str):
            out_parts.append(item["text"])
            continue
        if itype != "message":
            continue
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                out_parts.append(c["text"])
    return "\n".join(out_parts).strip()


def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + "…(truncated)"


def _summarize_evidence(run: Dict[str, Any], limit_tools: int = 10) -> Dict[str, Any]:
    executed = run.get("executed") or []
    executed_tail = executed[-limit_tools:] if isinstance(executed, list) else []

    tool_calls = []
    errors: List[str] = []

    for e in executed_tail:
        if not isinstance(e, dict):
            continue
        tool = e.get("tool")
        args = e.get("arguments") or {}
        res = e.get("result") or {}
        ok = res.get("ok")
        err = res.get("error")
        if ok is False and err:
            errors.append(f"{tool}: {err}")

        entry = {"name": tool, "ok": ok, "error": err}
        if tool == "run_cli":
            entry["command"] = (args or {}).get("command")
            entry["stdout_tail"] = _truncate(str(res.get("stdout") or ""), 800)
        elif tool == "fetch_url":
            entry["url"] = (args or {}).get("url")
            entry["status"] = res.get("status")
        elif tool in ("read_file", "write_file", "list_dir"):
            entry["path"] = (args or {}).get("path")
        tool_calls.append(entry)

    # If the run itself failed at HTTP level
    if run.get("ok") is False and run.get("error"):
        errors.append(str(run.get("error")))

    return {
        "tool_calls": tool_calls,
        "errors": errors,
        "saw_final_answer": run.get("saw_final_answer"),
        "final_empty": not bool((run.get("final") or "").strip()),
    }


def _load_judge_skill(path: str) -> str:
    return Path(path).read_text("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_log", help="Path to evolve/runs/run-*.json")
    ap.add_argument("--base-url", default=os.environ.get("OSS120B_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("OSS120B_MODEL", "openai/gpt-oss-120b"))
    ap.add_argument(
        "--judge-skill",
        default=str(Path(__file__).resolve().parents[1] / "recipes" / "judge_skill_v1.md"),
        help="Path to judge skill prompt markdown",
    )
    ap.add_argument("--spec", default="", help="Optional extra acceptance spec text")
    args = ap.parse_args()

    run_path = Path(args.run_log)
    run = json.loads(run_path.read_text("utf-8"))

    task = run.get("task") or run.get("task_original") or ""
    final_output = run.get("final") or ""

    evidence = _summarize_evidence(run)

    judge_skill = _load_judge_skill(args.judge_skill)

    # Assemble judge input as plain text to keep vLLM compatibility.
    blocks = [
        "TASK:\n" + (task.strip() or "(missing task in run log)"),
        "FINAL_OUTPUT:\n" + (final_output.strip() or "(empty)"),
        "EVIDENCE(JSON):\n" + json.dumps(evidence, ensure_ascii=False, indent=2),
    ]
    if args.spec.strip():
        blocks.append("SPEC:\n" + args.spec.strip())

    prompt = "\n\n".join(blocks)

    api_key = os.environ.get("OPENAI_API_KEY")

    payload = {
        "model": args.model,
        "instructions": judge_skill,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "temperature": 0.2,
        "top_p": 1.0,
        "max_output_tokens": 600,
    }

    with httpx.Client() as client:
        status, j = _post_response(client, args.base_url, api_key, payload)
        if status != 200:
            print(f"TOOL_FAIL\nHTTP_{status}", file=sys.stderr)
            print(json.dumps(j, ensure_ascii=False, indent=2), file=sys.stderr)
            raise SystemExit(2)

    out = _extract_output_text(j)
    if not out:
        out = "ANALYSIS: (empty judge output)\nDECISION: REDO\nPATCH_JSON: {\"decision\":\"REDO\",\"analysis\":\"empty judge output\",\"reasons\":[{\"tag\":\"empty_output\",\"because\":\"judge returned empty\",\"evidence\":\"n/a\"}],\"add_constraints\":[]}"  # noqa: E501

    # ensure decision line exists (best-effort)
    if "DECISION:" not in out:
        out = out.strip() + "\nDECISION: REDO"

    # ensure PATCH_JSON exists (best-effort)
    if "PATCH_JSON:" not in out:
        decision = "REDO"
        m = re.search(r"DECISION:\s*(PASS|REDO)", out)
        if m:
            decision = m.group(1)
        patch = {
            "decision": decision,
            "analysis": _truncate(out.strip(), 400),
            "reasons": [{"tag": "format_mismatch", "because": "missing PATCH_JSON", "evidence": "judge_output"}],
            "add_constraints": [],
        }
        out = out.strip() + "\nPATCH_JSON: " + json.dumps(patch, ensure_ascii=False)

    print(out)


if __name__ == "__main__":
    main()
