#!/usr/bin/env python3
"""Run a single real task against gpt-oss-120b with native tool-calling (Responses API).

This is the "practical use" entrypoint (as opposed to the benchmark).

Safety model
- Default: run inside a fresh sandbox directory with a few fixture files.
- Tools are restricted (allowlist) and forbid pipes/redirects/chaining.
- The model MUST finish by calling final_answer({text}).
- If ANY tool returns ok=false, the model MUST final_answer('TOOL_FAIL') (no guessing).

Usage
  python3 oss120b_toolcall_run.py --task "Read notes.txt and return the 2nd TODO item"

  # interactive
  python3 oss120b_toolcall_run.py

Env
  OSS120B_BASE_URL, OSS120B_MODEL, OPENAI_API_KEY(optional)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except Exception:  # pragma: no cover
    print("Missing dependency: httpx. Install with: pip install httpx", file=sys.stderr)
    raise


# -------------------------
# Responses API helpers
# -------------------------

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
    """Extract best-effort text from a Responses API JSON.

    Different backends may return either:
    - output: [{type:"message", content:[{type:"output_text", text:"..."}]}]
    - output: [{type:"output_text", text:"..."}]
    """
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


def _extract_tool_calls(resp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    for item in resp_json.get("output", []) or []:
        if item.get("type") in ("function_call", "tool_call"):
            name = item.get("name") or item.get("tool_name")
            args = item.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_unparsed": args}
            if isinstance(name, str):
                calls.append({"name": name, "arguments": args or {}, "raw": item})

        if item.get("type") == "message":
            tool_calls = item.get("tool_calls") or []
            for tc in tool_calls:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                name = fn.get("name")
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"_unparsed": args}
                if isinstance(name, str):
                    calls.append({"name": name, "arguments": args or {}, "raw": tc})

    uniq: List[Dict[str, Any]] = []
    seen = set()
    for c in calls:
        key = (c.get("name"), json.dumps(c.get("arguments", {}), sort_keys=True, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _slim_function_call(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return a minimal function_call item to avoid backend parser quirks."""
    keep = {"type": raw.get("type"), "name": raw.get("name"), "arguments": raw.get("arguments"), "call_id": raw.get("call_id")}
    # some backends prefer id instead of call_id
    if not keep.get("call_id") and raw.get("id"):
        keep["call_id"] = raw.get("id")
    return {k: v for k, v in keep.items() if v is not None}


# -------------------------
# Sandbox tools
# -------------------------

# Default allowlist: execution-style, but still constrained.
# You can expand with --allow-any-cli (see tool_run_cli).
ALLOWED_CMDS = {
    "ls",
    "cat",
    "wc",
    "head",
    "tail",
    "grep",
    "echo",
    "sort",
    "uniq",
    "sed",
    "awk",
    "cut",
    "tr",
    "find",
    "python3",
}

# Dangerous interpreters/shells that would trivially bypass restrictions.
DENY_CMDS = {"sh", "bash", "zsh", "fish", "python", "node", "perl", "ruby"}


def _safe_relpath(p: str) -> str:
    p = (p or "").strip().lstrip("/")
    if ".." in Path(p).parts:
        raise ValueError("path_traversal")
    return p or "."


def tool_read_file(root: Path, path: str) -> Dict[str, Any]:
    rel = _safe_relpath(path)
    fp = (root / rel).resolve()
    if root.resolve() not in fp.parents and fp != root.resolve():
        raise ValueError("path_escape")
    if not fp.exists() or not fp.is_file():
        return {"ok": False, "error": "not_found"}
    text = fp.read_text("utf-8", errors="replace")
    if len(text) > 12000:
        text = text[:12000] + "\n...TRUNCATED..."
    return {"ok": True, "path": rel, "content": text}


def tool_list_dir(root: Path, path: str = ".") -> Dict[str, Any]:
    rel = _safe_relpath(path)
    dp = (root / rel).resolve()
    if root.resolve() not in dp.parents and dp != root.resolve():
        raise ValueError("path_escape")
    if not dp.exists() or not dp.is_dir():
        return {"ok": False, "error": "not_found"}
    items = []
    for c in sorted(dp.iterdir(), key=lambda x: x.name):
        if c.name.startswith("."):
            continue
        items.append({"name": c.name, "type": "dir" if c.is_dir() else "file", "size": c.stat().st_size})
    return {"ok": True, "path": rel, "items": items}


def tool_run_cli(
    root: Path,
    command: str,
    allow_any_cli: bool = False,
    allow_shell: bool = False,
    allow_abs_paths: bool = False,
    timeout_sec: int = 8,
) -> Dict[str, Any]:
    """Run a command inside the sandbox directory.

    Default mode (recommended):
    - No shell; forbids metacharacters (pipes/redirects/chaining).
    - Allowlist of executables unless allow_any_cli=True.

    Unsafe mode (explicit):
    - allow_shell=True runs through zsh and allows metacharacters.

    NOTE: This is NOT a real OS sandbox. Even with cwd set to sandbox root,
    a shell command can access outside files/network under your user permissions.
    Use only for experiments you trust.
    """

    cmd = (command or "").strip()
    if not cmd:
        return {"ok": False, "error": "empty_command"}

    if not allow_shell:
        if re.search(r"[;&|><`$()]", cmd):
            return {"ok": False, "error": "shell_metacharacters_forbidden"}

        parts = shlex.split(cmd)
        if not parts:
            return {"ok": False, "error": "empty_command"}

        exe = parts[0]
        if exe in DENY_CMDS:
            return {"ok": False, "error": f"command_denied:{exe}"}

        if not allow_any_cli and exe not in ALLOWED_CMDS:
            return {"ok": False, "error": f"command_not_allowed:{exe}", "allowed": sorted(ALLOWED_CMDS)}

        if not allow_abs_paths:
            for a in parts[1:]:
                if a.startswith("/"):
                    return {"ok": False, "error": "absolute_paths_forbidden"}
                if ".." in Path(a).parts:
                    return {"ok": False, "error": "path_traversal"}

        try:
            r = subprocess.run(parts, cwd=str(root), capture_output=True, text=True, timeout=timeout_sec, check=False)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}

    else:
        # Unsafe: allow pipes/redirects etc by running through a shell.
        try:
            r = subprocess.run(
                cmd,
                cwd=str(root),
                shell=True,
                executable="/bin/zsh",
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}

    out = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
    out = out.strip()
    if len(out) > 12000:
        out = out[:12000] + "\n...TRUNCATED..."

    return {"ok": r.returncode == 0, "returncode": r.returncode, "stdout": out}


TOOLS_SPEC = [
    {
        "type": "function",
        "name": "list_dir",
        "description": "List files in a sandbox directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a UTF-8 text file from the sandbox directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write a UTF-8 text file into the sandbox directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path under sandbox root"},
                "content": {"type": "string", "description": "File content"}
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "fetch_url",
        "description": "Fetch an HTTP(S) URL and return status/headers/body (truncated).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_bytes": {"type": "integer", "description": "Max bytes of body to return (default 20000)"}
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_cli",
        "description": "Run a restricted CLI command inside the sandbox root. No pipes/redirects/chaining.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "final",
        "description": "Return the final answer. Must be called exactly once at the end.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
]


SYSTEM = (
    "You are a tool-using assistant running in a sandbox.\n"
    "Hard rules (non-negotiable):\n"
    "- You MUST use tools to read/write files and run CLI; never guess.\n"
    "- If ANY tool result has ok=false (or indicates an error), you MUST call final with text exactly 'TOOL_FAIL'.\n"
    "- You MUST NOT produce a normal text answer directly. Your final response MUST be via the tool final({text}).\n"
    "- Do NOT fabricate file contents or command outputs.\n"
    "- Keep final text minimal and follow the user's requested output format exactly.\n"
)


def tool_write_file(root: Path, path: str, content: str) -> Dict[str, Any]:
    rel = _safe_relpath(path)
    fp = (root / rel).resolve()
    if root.resolve() not in fp.parents and fp != root.resolve():
        raise ValueError("path_escape")

    # Keep it simple + safe: only allow .py/.txt/.md by default
    if not (rel.endswith(".py") or rel.endswith(".txt") or rel.endswith(".md")):
        return {"ok": False, "error": "extension_not_allowed"}

    if len(content) > 50_000:
        return {"ok": False, "error": "content_too_large"}

    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, "utf-8")
    return {"ok": True, "path": rel, "bytes": len(content.encode('utf-8'))}


def tool_fetch_url(url: str, max_bytes: int = 20000) -> Dict[str, Any]:
    url = (url or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"ok": False, "error": "scheme_not_allowed"}

    if max_bytes is None or max_bytes <= 0:
        max_bytes = 20000
    max_bytes = min(int(max_bytes), 200000)  # hard cap

    api_key = os.environ.get("OPENAI_API_KEY")  # unused but keep env pattern consistent
    _ = api_key

    try:
        with httpx.Client(follow_redirects=True, timeout=20) as client:
            r = client.get(url, headers={"User-Agent": "oss120b-toolcall-run/1.0"})
            content = r.content[:max_bytes]
            # try decode as utf-8; otherwise base64 would be needed (skip)
            try:
                text = content.decode("utf-8", errors="replace")
            except Exception:
                text = "".join(chr(b) if 32 <= b < 127 else "?" for b in content)

            # headers to simple dict (truncate)
            headers = {k: v for k, v in list(r.headers.items())[:50]}
            return {
                "ok": True,
                "url": url,
                "status": r.status_code,
                "headers": headers,
                "body": text,
                "truncated": len(r.content) > max_bytes,
            }
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{type(e).__name__}:{e}"}


def _exec_tool(
    root: Path,
    name: str,
    args: Dict[str, Any],
    allow_any_cli: bool = False,
    allow_shell: bool = False,
    allow_abs_paths: bool = False,
    cli_timeout_sec: int = 8,
) -> Dict[str, Any]:
    # If backend returned truncated/unparsed arguments, treat as a hard tool error.
    if isinstance(args, dict) and args.get("_unparsed"):
        return {"ok": False, "error": "bad_arguments_unparsed", "_unparsed": str(args.get("_unparsed"))[:500]}

    if name == "list_dir":
        return tool_list_dir(root, path=str(args.get("path", ".")))
    if name == "read_file":
        return tool_read_file(root, path=str(args.get("path")))
    if name == "write_file":
        return tool_write_file(root, path=str(args.get("path")), content=str(args.get("content")))
    if name == "fetch_url":
        return tool_fetch_url(url=str(args.get("url")), max_bytes=int(args.get("max_bytes", 20000)))
    if name == "run_cli":
        return tool_run_cli(
            root,
            command=str(args.get("command")),
            allow_any_cli=allow_any_cli,
            allow_shell=allow_shell,
            allow_abs_paths=allow_abs_paths,
            timeout_sec=cli_timeout_sec,
        )
    if name == "final":
        return {"ok": True, "captured": True, "text": str(args.get("text", ""))}
    return {"ok": False, "error": f"unknown_tool:{name}"}


# -------------------------
# Fixture dir
# -------------------------

def make_default_sandbox() -> Path:
    root = Path(tempfile.mkdtemp(prefix="oss120b_run_"))
    (root / "alpha.txt").write_text("apple\nbanana\ncarrot\n", "utf-8")
    (root / "beta.txt").write_text("one\ntwo\nthree\nfour\n", "utf-8")
    (root / "notes.txt").write_text("TODO: buy milk\nTODO: write probe\nDONE: sleep\n", "utf-8")
    return root


# -------------------------
# Runner
# -------------------------

def run_task_once(
    base_url: str,
    model: str,
    task: str,
    root: Path,
    temperature: float,
    top_p: float,
    max_steps: int = 12,
    verbose: bool = False,
    allow_any_cli: bool = False,
    allow_shell: bool = False,
    allow_abs_paths: bool = False,
    cli_timeout_sec: int = 8,
    extra_system: str = "",
) -> Dict[str, Any]:
    # IMPORTANT: include the original task in the returned run log so downstream
    # verifiers/judges can evaluate completion.
    task_original = task
    api_key = os.environ.get("OPENAI_API_KEY")

    def _to_output_item(tc_raw: Dict[str, Any], result_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        call_id = tc_raw.get("call_id") or tc_raw.get("id")
        if not isinstance(call_id, str) or not call_id:
            return None
        return {"type": "function_call_output", "call_id": call_id, "output": json.dumps(result_obj, ensure_ascii=False)}

    convo: List[Dict[str, Any]] = [{"role": "user", "content": [{"type": "input_text", "text": task}]}]
    steps: List[Dict[str, Any]] = []
    executed: List[Dict[str, Any]] = []

    final_text = ""
    saw_final_answer = False
    saw_any_tool_fail = False

    with httpx.Client() as client:
        for step_i in range(max_steps):
            payload = {
                "model": model,
                "instructions": (SYSTEM + ("\n" + extra_system.strip() if extra_system.strip() else "")),
                "input": convo,
                "tools": TOOLS_SPEC,
                "tool_choice": "auto",
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": 900,
            }

            status, j = _post_response(client, base_url, api_key, payload)

            # Self-heal: vLLM Responses API occasionally returns 400 like
            # "Unknown channel: final_answer" / similar parser glitches.
            # If we detect this class of error, retry the SAME request once.
            if status == 400:
                try:
                    msg = ((j.get("error") or {}).get("message") or "") if isinstance(j, dict) else ""
                except Exception:
                    msg = ""
                if "Unknown channel" in msg or "unexpected tokens remaining" in msg:
                    status, j = _post_response(client, base_url, api_key, payload)

            if status != 200:
                return {
                    "ok": False,
                    "task": task_original,
                    "error": f"HTTP_{status}",
                    "raw": j,
                    "steps": steps,
                    "executed": executed,
                }

            tool_calls = _extract_tool_calls(j)
            text = _extract_output_text(j)
            steps.append({"step": step_i, "tool_calls": tool_calls, "text": text})

            if verbose and text.strip():
                print(f"[model_text@step{step_i}] {text[:200]}")

            if tool_calls:
                # If model calls final, capture and stop.
                fa_calls = [tc for tc in tool_calls if tc.get("name") == "final"]
                if fa_calls:
                    saw_final_answer = True
                    args = fa_calls[0].get("arguments") or {}
                    final_text = str(args.get("text", "")).strip()
                    executed.append({"tool": "final", "arguments": args, "result": {"ok": True}})
                    break

                # Append minimal function_call items
                for tc in tool_calls:
                    raw = tc.get("raw")
                    if isinstance(raw, dict) and raw.get("type") in ("function_call", "tool_call"):
                        convo.append(_slim_function_call(raw))

                # Execute tools + feed outputs
                any_fail = False
                for tc in tool_calls:
                    try:
                        res = _exec_tool(
                            root,
                            tc["name"],
                            tc.get("arguments") or {},
                            allow_any_cli=allow_any_cli,
                            allow_shell=allow_shell,
                            allow_abs_paths=allow_abs_paths,
                            cli_timeout_sec=cli_timeout_sec,
                        )
                    except Exception as e:
                        res = {"ok": False, "error": f"exception:{type(e).__name__}:{e}"}

                    if res.get("ok") is False:
                        any_fail = True

                    executed.append({"tool": tc["name"], "arguments": tc.get("arguments") or {}, "result": res})
                    out_item = _to_output_item(tc.get("raw") or {}, res)
                    if out_item:
                        convo.append(out_item)

                # Nudge to close
                if any_fail:
                    saw_any_tool_fail = True
                    convo.append({"role": "user", "content": [{"type": "input_text", "text": "A tool returned ok=false. You MUST now call final with text exactly TOOL_FAIL."}]})
                else:
                    convo.append({"role": "user", "content": [{"type": "input_text", "text": "You now have sufficient information. You MUST call final({text: ...}) now."}]})
                continue

            # No tool calls:
            # Rescue: some backends/models sometimes output TOOL_FAIL as plain text or as JSON
            # instead of calling final(). If we already saw a tool failure, accept it.
            tstr = text.strip()
            if tstr:
                if saw_any_tool_fail and (tstr == "TOOL_FAIL" or '"text": "TOOL_FAIL"' in tstr):
                    final_text = "TOOL_FAIL"
                    break

                # Practical fallback: if the model outputs plain text after we already executed
                # a successful command (e.g. python3 task.py), accept it as final.
                last = next((e for e in reversed(executed) if e.get("tool") == "run_cli"), None)
                if last and (last.get("result") or {}).get("ok") is True:
                    final_text = tstr
                    break

            # Otherwise demand a final() call.
            convo.append({"role": "user", "content": [{"type": "input_text", "text": "You MUST now call final({text: ...}). Do not output normal text."}]})

    # Final safety net: if the model never called final() but we observed a tool failure,
    # force a deterministic TOOL_FAIL so the outer loop can recover.
    if (not final_text.strip()) and saw_any_tool_fail:
        final_text = "TOOL_FAIL"

    # If we still have no final at all, treat as a failed run (useful when max_steps is hit).
    if (not saw_final_answer) and (not final_text.strip()):
        return {
            "ok": False,
            "task": task_original,
            "sandbox": str(root),
            "error": "no_final",
            "final": "",
            "saw_final_answer": False,
            "steps": steps,
            "executed": executed,
            "temperature": temperature,
            "top_p": top_p,
        }

    return {
        "ok": True,
        "task": task_original,
        "sandbox": str(root),
        "final": final_text,
        "saw_final_answer": saw_final_answer,
        "steps": steps,
        "executed": executed,
        "temperature": temperature,
        "top_p": top_p,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("OSS120B_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("OSS120B_MODEL", "openai/gpt-oss-120b"))
    ap.add_argument("--task", help="Natural language task to execute (quoted). If omitted, reads from stdin.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--keep-sandbox", action="store_true", help="Keep sandbox dir path printed for inspection.")
    ap.add_argument("--dump-json", action="store_true", help="Print full run JSON (debug).")
    ap.add_argument("--recipe", default="", help="Path to a recipe markdown file; content is injected as extra system instructions.")
    ap.add_argument(
        "--save-run",
        default="",
        help="Write run log JSON to evolve/runs/<timestamp>.json (or to the provided path).",
    )
    ap.add_argument("--no-save-run", action="store_true", help="Disable auto run logging.")
    ap.add_argument(
        "--auto-propose",
        action="store_true",
        help="After the run, analyze and (if relevant) generate an auto recipe under recipes/auto/.",
    )
    ap.add_argument(
        "--allow-any-cli",
        action="store_true",
        help=(
            "Allow any CLI executable (still no pipes/redirects/chaining; still denies shells/interpreters). "
            "Useful for real workflows, but increases risk."
        ),
    )
    ap.add_argument(
        "--allow-shell",
        action="store_true",
        help=(
            "UNSAFE: allow running run_cli through /bin/zsh (pipes/redirects/chaining enabled). "
            "This is not a real OS sandbox; commands may read outside files under your user."
        ),
    )
    ap.add_argument(
        "--allow-abs-paths",
        action="store_true",
        help="UNSAFE: allow absolute paths in CLI arguments.",
    )
    ap.add_argument(
        "--cli-timeout",
        type=int,
        default=8,
        help="CLI timeout seconds (default 8).",
    )
    ap.add_argument("--retries", type=int, default=0, help="Retry the whole task N additional times if it fails (legacy).")
    ap.add_argument(
        "--retry-temps",
        default="",
        help="Comma-separated temperature schedule for attempts (e.g. '1.0,0.7,0.2'). Default: use requested temp then 0.7 then 0.2.",
    )
    ap.add_argument(
        "--auto-fix",
        action="store_true",
        help="Enable verifier-driven repair loop: run -> judge PASS/REDO -> if REDO, apply patch constraints and rerun.",
    )
    ap.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Max rounds for --auto-fix (default 3).",
    )
    ap.add_argument(
        "--judge-skill",
        default=str(Path(__file__).parent / "recipes" / "judge_skill_v1.md"),
        help="Judge skill prompt (used by --auto-fix).",
    )
    args = ap.parse_args()

    task = args.task
    if not task:
        print("Enter task (end with Ctrl-D):", file=sys.stderr)
        task = sys.stdin.read().strip()

    if not task:
        print("Empty task.", file=sys.stderr)
        raise SystemExit(2)

    sandbox = make_default_sandbox()

    recipe_text = ""
    if args.recipe:
        try:
            recipe_text = Path(args.recipe).read_text("utf-8")
        except Exception as e:
            print(f"Failed to read recipe: {args.recipe}: {e}", file=sys.stderr)
            raise SystemExit(2)

    def summarize_attempt(executed: List[Dict[str, Any]], limit: int = 8) -> str:
        tail = executed[-limit:] if executed else []
        parts = []
        for e in tail:
            tool = e.get("tool")
            args = e.get("arguments") or {}
            res = e.get("result") or {}
            ok = res.get("ok")
            if tool == "run_cli":
                parts.append(f"run_cli({args.get('command')!r}) -> ok={ok} err={res.get('error')}")
            elif tool == "fetch_url":
                parts.append(f"fetch_url({args.get('url')!r}) -> status={res.get('status')} ok={ok} err={res.get('error')}")
            else:
                parts.append(f"{tool}({args}) -> ok={ok} err={res.get('error')}")
        return "\n".join(parts)

    def _judge_run(run: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """Return (decision, brief_analysis, patch_json). Decision is PASS or REDO.

        We do a *structural* pre-check first (deterministic) to avoid the judge
        hallucinating a PASS when obvious hard requirements are missing.
        """

        run_task = (run.get("task") or task or "").strip()
        final_out = (run.get("final") or "").strip()
        saw_final = bool(run.get("saw_final_answer"))

        # -------------------------
        # 1) Structural pre-checks
        # -------------------------
        add_constraints: List[str] = []
        reasons: List[Dict[str, str]] = []

        # (a) Must end via final() tool (otherwise output may be empty even if work was done)
        if not saw_final or not final_out:
            reasons.append({
                "tag": "empty_output",
                "because": "Run did not end with a non-empty final() answer.",
                "evidence": f"saw_final_answer={saw_final} final_len={len(final_out)}",
            })
            add_constraints.append("You MUST call final({text: ...}) exactly once at the end, and the text must be non-empty.")

        # (b) If task mentions explicit .md filenames, ensure they were written successfully.
        required_files = sorted(set(re.findall(r"\b[\w.-]+\.md\b", run_task)))
        if required_files:
            wrote_ok: Dict[str, bool] = {fn: False for fn in required_files}
            executed = run.get("executed") or []
            if isinstance(executed, list):
                for e in executed:
                    if not isinstance(e, dict):
                        continue
                    if e.get("tool") != "write_file":
                        continue
                    args2 = e.get("arguments") or {}
                    res2 = e.get("result") or {}
                    path2 = str(args2.get("path") or "")
                    if path2 in wrote_ok and res2.get("ok") is True:
                        # basic non-empty check
                        b = res2.get("bytes")
                        if isinstance(b, int) and b > 0:
                            wrote_ok[path2] = True
                        else:
                            wrote_ok[path2] = True  # best-effort

            missing = [fn for fn, ok_ in wrote_ok.items() if not ok_]
            if missing:
                reasons.append({
                    "tag": "missing_requirements",
                    "because": f"Missing required output files: {', '.join(missing)}",
                    "evidence": "executed[*].tool=write_file",
                })
                add_constraints.append("You MUST create ALL required files listed in the task via write_file().")
                # Skeleton-first strategy to avoid long tool arguments getting truncated.
                add_constraints.append(
                    "Write files in TWO phases: (1) create ALL 7 files as SHORT skeletons first (<= 25 lines each, include at least 1 link and 2-3 bullets), then (2) optionally enrich each file in a second pass if you still have steps left."
                )
                add_constraints.append("Keep each write_file content under 2000 characters to avoid tool argument truncation.")
                add_constraints.append("After writing files, call list_dir('.') and verify the files exist, then call final() with a short summary of what you created.")

        # If structural checks fail, do not call LLM judge (save cost + avoid confusion).
        if reasons:
            patch = {
                "decision": "REDO",
                "analysis": "Structural verification failed; redo required.",
                "reasons": reasons,
                "add_constraints": add_constraints,
            }
            brief = "\n".join([
                "Structural check:",
                *(f"- {r['tag']}: {r['because']} ({r['evidence']})" for r in reasons),
                "DECISION: REDO",
            ])
            return "REDO", brief, patch

        # -------------------------
        # 2) LLM-as-judge (semantic)
        # -------------------------
        judge_skill = Path(args.judge_skill).read_text("utf-8")

        # Compact evidence summary (keep judge cheap + evidence-based)
        executed = run.get("executed") or []
        tail = executed[-10:] if isinstance(executed, list) else []
        evidence = {
            "ok": run.get("ok"),
            "error": run.get("error"),
            "saw_final_answer": run.get("saw_final_answer"),
            "final_len": len(final_out),
            "executed_tail": [],
        }
        for e in tail:
            if not isinstance(e, dict):
                continue
            tool = e.get("tool")
            a = e.get("arguments") or {}
            r = e.get("result") or {}
            item = {"tool": tool, "ok": r.get("ok"), "error": r.get("error")}
            if tool == "run_cli":
                item["command"] = a.get("command")
                item["stdout_tail"] = (r.get("stdout") or "")[-800:]
            if tool == "fetch_url":
                item["url"] = a.get("url")
                item["status"] = r.get("status")
            if tool in ("read_file", "write_file", "list_dir"):
                item["path"] = a.get("path")
            evidence["executed_tail"].append(item)

        prompt = (
            "TASK:\n" + run_task + "\n\n"
            "FINAL_OUTPUT:\n" + (final_out or "(empty)") + "\n\n"
            "EVIDENCE(JSON):\n" + json.dumps(evidence, ensure_ascii=False, indent=2)
        )

        api_key = os.environ.get("OPENAI_API_KEY")
        payload = {
            "model": args.model,
            "instructions": judge_skill,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "temperature": 0.2,
            "top_p": 1.0,
            "max_output_tokens": 650,
        }

        with httpx.Client() as client:
            status, j = _post_response(client, args.base_url, api_key, payload)
            if status != 200:
                return "REDO", f"judge_http_error: HTTP_{status}", {
                    "decision": "REDO",
                    "analysis": f"judge_http_error HTTP_{status}",
                    "reasons": [{"tag": "tool_error", "because": "judge http error", "evidence": str(j)[:2000]}],
                    "add_constraints": [],
                }

        out = _extract_output_text(j) or ""
        m = re.search(r"DECISION:\s*(PASS|REDO)", out)
        decision = m.group(1) if m else "REDO"

        patch: Dict[str, Any] = {}
        m2 = re.search(r"PATCH_JSON:\s*(\{.*\})\s*$", out.strip(), flags=re.S)
        if m2:
            try:
                patch = json.loads(m2.group(1))
            except Exception:
                patch = {}

        brief_lines = []
        for line in out.splitlines():
            if line.strip().startswith("PATCH_JSON:"):
                break
            brief_lines.append(line)
        brief = "\n".join(brief_lines).strip() or "(no analysis)"

        if not patch:
            patch = {
                "decision": decision,
                "analysis": brief[:500],
                "reasons": [{"tag": "format_mismatch", "because": "missing/invalid PATCH_JSON", "evidence": "judge_output"}],
                "add_constraints": [],
            }
        patch["decision"] = decision
        return decision, brief, patch

    # Attempt schedule (used for both legacy retries and auto-fix rounds)
    temps = getattr(args, "retry_temps", None)
    if temps:
        temp_schedule = [float(x) for x in temps.split(",") if x.strip()]
    else:
        temp_schedule = [args.temperature, 0.7, 0.2]

    best: Dict[str, Any] | None = None

    if args.auto_fix:
        rounds = max(1, int(args.max_rounds))
        # Start from the original task; later rounds may shrink it.
        task_i = task
        extra_constraints: List[str] = []

        for r_i in range(rounds):
            t = temp_schedule[min(r_i, len(temp_schedule) - 1)]

            extra_system = recipe_text.strip()
            if extra_system:
                extra_system += "\n\n"
            extra_system += (
                f"ROUND {r_i+1}/{rounds}. "
                "Your goal is to COMPLETE the user TASK. If the previous round was REDO, you MUST change strategy and satisfy the verifier constraints."
            )
            if extra_constraints:
                extra_system += "\n\nVERIFIER_CONSTRAINTS (must follow):\n- " + "\n- ".join(extra_constraints)

            # Provide last attempt summary to avoid repeating failures.
            if best is not None:
                task_i = task_i + "\n\nPrevious round tool summary (avoid repeating failures):\n" + summarize_attempt(best.get("executed") or [])

            res = run_task_once(
                base_url=args.base_url,
                model=args.model,
                task=task_i,
                root=sandbox,
                temperature=t,
                top_p=args.top_p,
                max_steps=args.max_steps,
                verbose=args.verbose,
                allow_any_cli=args.allow_any_cli,
                allow_shell=args.allow_shell,
                allow_abs_paths=args.allow_abs_paths,
                cli_timeout_sec=args.cli_timeout,
                extra_system=extra_system,
            )

            best = res

            decision, brief, patch = _judge_run(res)
            res["judge"] = {"decision": decision, "brief": brief, "patch": patch}

            if decision == "PASS":
                break

            # Apply patch for next round
            add_constraints = patch.get("add_constraints") or []
            if isinstance(add_constraints, list):
                extra_constraints.extend([str(x) for x in add_constraints if str(x).strip()])

            shrink_task = patch.get("shrink_task")
            if isinstance(shrink_task, str) and shrink_task.strip():
                task_i = shrink_task.strip()

            output_template = patch.get("output_template")
            if isinstance(output_template, str) and output_template.strip():
                extra_constraints.append("Output template (must follow):\n" + output_template.strip())

        res = best or {"ok": False, "task": task, "error": "no_rounds"}

    else:
        # Legacy simple retries: temperature annealing + 'try different method'
        retries = getattr(args, "retries", 0)

        for attempt in range(retries + 1):
            t = temp_schedule[min(attempt, len(temp_schedule) - 1)]

            extra_system = (
                recipe_text.strip()
                + ("\n\n" if recipe_text.strip() else "")
                + (
                    f"ATTEMPT {attempt+1}/{retries+1}. "
                    "If you already tried one approach before, try a DIFFERENT method now. "
                    "Do not repeat the same failed tool call; use alternative tools if possible (read_file vs write_file vs run_cli vs fetch_url)."
                )
            )

            task_i = task
            if best is not None:
                task_i += "\n\nPrevious attempt tool summary (do NOT repeat the same failed steps):\n" + summarize_attempt(best.get("executed") or [])

            res = run_task_once(
                base_url=args.base_url,
                model=args.model,
                task=task_i,
                root=sandbox,
                temperature=t,
                top_p=args.top_p,
                max_steps=args.max_steps,
                verbose=args.verbose,
                allow_any_cli=args.allow_any_cli,
                allow_shell=args.allow_shell,
                allow_abs_paths=args.allow_abs_paths,
                cli_timeout_sec=args.cli_timeout,
                extra_system=extra_system,
            )

            best = res

            # Success criteria: got final_answer and it's not TOOL_FAIL
            if res.get("ok") and res.get("saw_final_answer") and (res.get("final") or "").strip() not in ("", "TOOL_FAIL"):
                break

        res = best or {"ok": False, "task": task, "error": "no_attempts"}

    # Auto-save run log (enabled by default unless --no-save-run)
    save_path = None
    if not args.no_save_run:
        if args.save_run and args.save_run.strip():
            save_path = Path(args.save_run).expanduser()
        else:
            ts = time.strftime("%Y%m%d-%H%M%S")
            save_path = Path(__file__).parent / "evolve" / "runs" / f"run-{ts}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            save_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), "utf-8")
        except Exception as e:
            print(f"Failed to save run log: {save_path}: {e}", file=sys.stderr)

    if args.dump_json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        if res.get("ok"):
            print(res.get("final", ""))
        else:
            print("TOOL_FAIL")
            print(json.dumps(res.get("raw"), ensure_ascii=False, indent=2), file=sys.stderr)

    if args.keep_sandbox:
        print(f"[sandbox] {sandbox}", file=sys.stderr)
    if save_path:
        print(f"[run_log] {save_path}", file=sys.stderr)

    if args.auto_propose and save_path:
        try:
            import subprocess as _sp
            # 1) analyze
            analysis_json = _sp.check_output(
                [
                    sys.executable,
                    str(Path(__file__).parent / "evolve" / "analyze_run.py"),
                    str(save_path),
                ],
                text=True,
            )
            analysis = json.loads(analysis_json)
            tags = analysis.get("tags") or []

            if any(t in ("run_cli_command_not_allowed", "run_cli_shell_metacharacters_forbidden") for t in tags):
                # 2) propose recipe
                tmp_analysis = Path(tempfile.mkdtemp(prefix="oss120b_analysis_")) / "analysis.json"
                tmp_analysis.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), "utf-8")
                out_json = _sp.check_output(
                    [
                        sys.executable,
                        str(Path(__file__).parent / "evolve" / "propose_recipe.py"),
                        str(save_path),
                        "--analysis",
                        str(tmp_analysis),
                    ],
                    text=True,
                )
                out = json.loads(out_json)
                if out.get("ok"):
                    print(f"[auto_recipe] {out.get('recipe_path')}", file=sys.stderr)
        except Exception as e:
            print(f"[auto_propose_error] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
