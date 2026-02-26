#!/usr/bin/env python3
"""Benchmark gpt-oss-120b native tool calling (Responses API) for execution-style tasks.

What this is
- A reproducible, programmatic benchmark you can run yourself (no OpenClaw routing required).
- Uses OpenAI-compatible Responses API (vLLM) with native `tools` and multi-step tool loop.
- Enforces a strict protocol: model MUST end by calling `final_answer({text})`.
- If any tool returns ok=false, model MUST final_answer('TOOL_FAIL') (no guessing).

Design choices (per Dr.Q requirements)
- Cases are grouped in 4 levels (L1-L4), default 20 cases/level.
- Output format is "B": strict string match, including multi-line outputs.
- "Execution style": includes run_cli with a strict allowlist; no pipes/redirects/chaining.

Usage
  python3 oss120b_toolcall_benchmark.py --trials 3 --per-level 20

Outputs
- JSON: memory/oss120b_runs/toolcall_benchmark-latest.json
- MD:   memory/oss120b_runs/toolcall_benchmark-latest.md

Env
  OSS120B_BASE_URL, OSS120B_MODEL, OPENAI_API_KEY(optional)
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
    out_parts: List[str] = []
    for item in resp_json.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                out_parts.append(c["text"])
    return "\n".join(out_parts).strip()


def _extract_tool_calls(resp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Best-effort extraction across backends.

    Returns list of: {name:str, arguments:dict, raw:dict}
    """
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


# -------------------------
# Sandbox tools
# -------------------------

ALLOWED_CMDS = {"ls", "cat", "wc", "head", "tail", "grep", "echo", "sort", "uniq"}


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
    if len(text) > 8000:
        text = text[:8000] + "\n...TRUNCATED..."
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


def tool_run_cli(root: Path, command: str) -> Dict[str, Any]:
    cmd = (command or "").strip()
    if not cmd:
        return {"ok": False, "error": "empty_command"}

    # no chaining, pipelines, redirects, subshell.
    if re.search(r"[;&|><`$()]", cmd):
        return {"ok": False, "error": "shell_metacharacters_forbidden"}

    parts = shlex.split(cmd)
    if not parts:
        return {"ok": False, "error": "empty_command"}

    exe = parts[0]
    if exe not in ALLOWED_CMDS:
        return {"ok": False, "error": f"command_not_allowed:{exe}", "allowed": sorted(ALLOWED_CMDS)}

    for a in parts[1:]:
        if a.startswith("/"):
            return {"ok": False, "error": "absolute_paths_forbidden"}
        if ".." in Path(a).parts:
            return {"ok": False, "error": "path_traversal"}

    try:
        r = subprocess.run(parts, cwd=str(root), capture_output=True, text=True, timeout=3, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}

    out = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
    out = out.strip()
    if len(out) > 8000:
        out = out[:8000] + "\n...TRUNCATED..."

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
        "name": "final_answer",
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
    "- You MUST use tools to read files / run CLI; never guess.\n"
    "- If ANY tool result has ok=false (or indicates an error), you MUST call final_answer with text exactly 'TOOL_FAIL'.\n"
    "- You MUST NOT produce a normal text answer directly. Your final response MUST be via the tool final_answer(text).\n"
    "- Do NOT fabricate file contents or command outputs.\n"
    "- Keep final_answer text minimal and follow the requested output format exactly.\n"
)


def _exec_tool(root: Path, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "list_dir":
        return tool_list_dir(root, path=str(args.get("path", ".")))
    if name == "read_file":
        return tool_read_file(root, path=str(args.get("path")))
    if name == "run_cli":
        return tool_run_cli(root, command=str(args.get("command")))
    if name == "final_answer":
        return {"ok": True, "captured": True, "text": str(args.get("text", ""))}
    return {"ok": False, "error": f"unknown_tool:{name}"}


# -------------------------
# Fixtures
# -------------------------

def _make_fixture_dir(seed: int = 0) -> Tuple[Path, Dict[str, Any]]:
    """Create sandbox files; return (root, meta) where meta is used for expected outputs."""
    rng = random.Random(seed)
    root = Path(tempfile.mkdtemp(prefix="oss120b_bench_"))

    alpha = ["apple", "banana", "carrot"]
    beta = ["one", "two", "three", "four"]
    notes = ["TODO: buy milk", "TODO: write probe", "DONE: sleep"]

    # a dup-heavy file
    dup = ["z", "a", "a", "b", "b", "b", "c", "c", "x"]

    # a small numeric file for parsing tasks
    nums = [str(rng.randint(1, 9)) for _ in range(12)]

    (root / "alpha.txt").write_text("\n".join(alpha) + "\n", "utf-8")
    (root / "beta.txt").write_text("\n".join(beta) + "\n", "utf-8")
    (root / "notes.txt").write_text("\n".join(notes) + "\n", "utf-8")
    (root / "dup.txt").write_text("\n".join(dup) + "\n", "utf-8")
    (root / "nums.txt").write_text("\n".join(nums) + "\n", "utf-8")

    meta = {
        "alpha": alpha,
        "beta": beta,
        "notes": notes,
        "dup": dup,
        "nums": nums,
        "files": ["alpha.txt", "beta.txt", "dup.txt", "notes.txt", "nums.txt"],
    }
    return root, meta


# -------------------------
# Cases (levels)
# -------------------------

@dataclass
class Case:
    id: str
    level: str  # L1-L4
    prompt: str
    expected: str


def _norm(s: str) -> str:
    return (s or "").replace("\r\n", "\n").strip()


def build_cases(meta: Dict[str, Any], per_level: int, seed: int = 0) -> List[Case]:
    rng = random.Random(seed)
    cases: List[Case] = []

    alpha = meta["alpha"]
    beta = meta["beta"]
    notes = meta["notes"]
    dup = meta["dup"]
    nums = list(map(int, meta["nums"]))

    # ---------- L1: basic ----------
    l1_templates = []

    l1_templates.append(lambda k: Case(
        id=f"L1_wc_beta_lines_{k}",
        level="L1",
        prompt="In the sandbox, count lines in beta.txt. Use tools. Return ONLY the number.",
        expected=str(len(beta)),
    ))

    l1_templates.append(lambda k: Case(
        id=f"L1_alpha_first_{k}",
        level="L1",
        prompt="Read alpha.txt and return ONLY the first line. Use tools.",
        expected=alpha[0],
    ))

    l1_templates.append(lambda k: Case(
        id=f"L1_grep_todo_count_{k}",
        level="L1",
        prompt="Count lines in notes.txt that start with 'TODO:'. Use tools. Return ONLY the number.",
        expected=str(sum(1 for x in notes if x.startswith("TODO:"))),
    ))

    l1_templates.append(lambda k: Case(
        id=f"L1_nums_sum_{k}",
        level="L1",
        prompt="Compute the sum of all integers in nums.txt. Use tools; do not guess. Return ONLY the sum as a number.",
        expected=str(sum(nums)),
    ))

    for i in range(per_level):
        cases.append(rng.choice(l1_templates)(i))

    # ---------- L2: multi-step chains ----------
    l2_templates = []

    l2_templates.append(lambda k: Case(
        id=f"L2_second_todo_{k}",
        level="L2",
        prompt=(
            "Find the SECOND TODO item text in notes.txt. Use tools (read_file or run_cli). "
            "Return ONLY the TODO text without the leading 'TODO: '."
        ),
        expected=notes[1].split("TODO: ", 1)[1],
    ))

    l2_templates.append(lambda k: Case(
        id=f"L2_dup_unique_sorted_{k}",
        level="L2",
        prompt=(
            "List all unique lines in dup.txt in alphabetical order, one per line. "
            "Use tools; do not guess."
        ),
        expected="\n".join(sorted(set(dup))),
    ))

    l2_templates.append(lambda k: Case(
        id=f"L2_list_then_read_{k}",
        level="L2",
        prompt=(
            "First, list the sandbox directory. Then read the file whose name comes last alphabetically. "
            "Return ONLY the last line of that file. Use tools."
        ),
        expected=(lambda last_name: _norm((Path(last_name).name,)))(""),
    ))

    # Fill expected for L2_list_then_read deterministically based on known files
    files_sorted = sorted(meta["files"])
    last_file = files_sorted[-1]
    content_map = {
        "alpha.txt": alpha,
        "beta.txt": beta,
        "dup.txt": dup,
        "notes.txt": notes,
        "nums.txt": list(map(str, meta["nums"])),
    }
    l2_templates[-1] = lambda k, lf=last_file: Case(
        id=f"L2_list_then_read_{k}",
        level="L2",
        prompt=(
            "First, list the sandbox directory. Then read the file whose name comes last alphabetically. "
            "Return ONLY the last line of that file. Use tools."
        ),
        expected=content_map[lf][-1],
    )

    for i in range(per_level):
        cases.append(rng.choice(l2_templates)(i))

    # ---------- L3: failure discipline ----------
    l3_templates = []

    l3_templates.append(lambda k: Case(
        id=f"L3_missing_file_{k}",
        level="L3",
        prompt=(
            "Read gamma.txt and return its first line. You MUST use tools. "
            "If the file doesn't exist, you MUST final_answer TOOL_FAIL."
        ),
        expected="TOOL_FAIL",
    ))

    l3_templates.append(lambda k: Case(
        id=f"L3_forbidden_rm_{k}",
        level="L3",
        prompt=(
            "Delete notes.txt using rm and then return ONLY the word DONE. Use tools. "
            "If deletion is not allowed, output TOOL_FAIL."
        ),
        expected="TOOL_FAIL",
    ))

    l3_templates.append(lambda k: Case(
        id=f"L3_forbidden_pipe_{k}",
        level="L3",
        prompt=(
            "Run `ls | wc -l` exactly as written and return ONLY the number. Use tools. "
            "If it cannot be executed exactly, output TOOL_FAIL."
        ),
        expected="TOOL_FAIL",
    ))

    for i in range(per_level):
        cases.append(rng.choice(l3_templates)(i))

    # ---------- L4: constrained workaround (no guessing) ----------
    l4_templates = []

    l4_templates.append(lambda k: Case(
        id=f"L4_pipe_workaround_count_files_{k}",
        level="L4",
        prompt=(
            "Try to run `ls | wc -l` in the sandbox. If rejected, do NOT guess. "
            "Compute the same result using allowed tools/commands (e.g., list_dir then count). "
            "Return ONLY the final number."
        ),
        expected=str(len(meta["files"])),
    ))

    l4_templates.append(lambda k: Case(
        id=f"L4_head_workaround_{k}",
        level="L4",
        prompt=(
            "Try to run `cat alpha.txt | head -n 1`. If rejected, do NOT guess. "
            "Use other allowed tools/commands to get the first line of alpha.txt. Return ONLY that line."
        ),
        expected=alpha[0],
    ))

    for i in range(per_level):
        cases.append(rng.choice(l4_templates)(i))

    # de-dup ids (should already be unique)
    return cases


# -------------------------
# Runner
# -------------------------

def _score(expected: str, final_text: str, saw_tool_call: bool, saw_final_answer: bool) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not saw_tool_call:
        errs.append("no_tool_call")
    if not saw_final_answer:
        errs.append("no_final_answer_call")

    if _norm(final_text) != _norm(expected):
        errs.append("wrong_answer")
    if _norm(expected) == "TOOL_FAIL" and _norm(final_text) != "TOOL_FAIL":
        errs.append("expected_tool_fail")

    return (len(errs) == 0, errs)


def run_case(
    client: httpx.Client,
    base_url: str,
    model: str,
    root: Path,
    case: Case,
    temperature: float,
    top_p: float,
    max_steps: int = 10,
) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")

    def _to_output_item(tc_raw: Dict[str, Any], result_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        call_id = tc_raw.get("call_id") or tc_raw.get("id")
        if not isinstance(call_id, str) or not call_id:
            return None
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result_obj, ensure_ascii=False),
        }

    convo: List[Dict[str, Any]] = [{"role": "user", "content": [{"type": "input_text", "text": case.prompt}]}]
    steps: List[Dict[str, Any]] = []
    executed: List[Dict[str, Any]] = []

    saw_tool_call = False
    saw_final_answer = False
    final_text = ""

    last_status: int | None = None
    last_json: Dict[str, Any] | None = None

    for step_i in range(max_steps):
        payload = {
            "model": model,
            "instructions": SYSTEM,
            "input": convo,
            "tools": TOOLS_SPEC,
            "tool_choice": "auto",
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": 900,
        }

        status, j = _post_response(client, base_url, api_key, payload)
        last_status, last_json = status, j
        if status != 200:
            break

        tool_calls = _extract_tool_calls(j)
        text = _extract_output_text(j)
        steps.append({"step": step_i, "tool_calls": tool_calls, "text": text})

        if tool_calls:
            saw_tool_call = True

            fa_calls = [tc for tc in tool_calls if tc.get("name") == "final_answer"]
            if fa_calls:
                saw_final_answer = True
                args = fa_calls[0].get("arguments") or {}
                final_text = str(args.get("text", "")).strip()
                executed.append({"tool": "final_answer", "arguments": args, "result": {"ok": True}})
                break

            # append raw function_call items for pairing
            for tc in tool_calls:
                raw = tc.get("raw")
                if isinstance(raw, dict) and raw.get("type") in ("function_call", "tool_call"):
                    convo.append(raw)

            # execute + output
            any_fail = False
            for tc in tool_calls:
                try:
                    res = _exec_tool(root, tc["name"], tc.get("arguments") or {})
                except Exception as e:
                    res = {"ok": False, "error": f"exception:{type(e).__name__}:{e}"}

                if res.get("ok") is False:
                    any_fail = True

                executed.append({"tool": tc["name"], "arguments": tc.get("arguments") or {}, "result": res})
                out_item = _to_output_item(tc.get("raw") or {}, res)
                if out_item:
                    convo.append(out_item)

            # If a tool failed, explicitly demand final_answer('TOOL_FAIL') to test protocol compliance.
            if any_fail:
                convo.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": "A tool returned ok=false. Per the rules, you MUST now call final_answer with text exactly TOOL_FAIL."}],
                })
            else:
                # General nudge (option B): after we have at least one successful "informational" tool
                # output (read_file/run_cli), and the expected answer is single-line, the model
                # sometimes forgets to close with final_answer at high temperature.
                saw_info = any(e.get("tool") in ("read_file", "run_cli") and (e.get("result") or {}).get("ok") is True for e in executed)
                expected_single_line = ("\n" not in case.expected) and (len(case.expected) <= 120)

                if saw_info and expected_single_line:
                    convo.append({
                        "role": "user",
                        "content": [{"type": "input_text", "text": "You now have sufficient information. You MUST call final_answer({text: ...}) now."}],
                    })
                else:
                    convo.append({"role": "user", "content": [{"type": "input_text", "text": "Continue."}]})
            continue

        # no tool calls: force final_answer
        convo.append({
            "role": "user",
            "content": [{"type": "input_text", "text": "You MUST now call final_answer({text: ...}). Do not output normal text."}],
        })
        continue

    # last resort: ask again with tools enabled and executed context
    fallback_used = False
    if not final_text.strip():
        fallback_used = True
        exec_blob = json.dumps(executed, ensure_ascii=False, indent=2)
        payload_fallback = {
            "model": model,
            "instructions": SYSTEM,
            "input": [
                {
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": (
                            f"Task: {case.prompt}\n\n"
                            "Tool executions already performed (JSON):\n"
                            f"{exec_blob}\n\n"
                            "Now you MUST call final_answer({text: ...}). "
                            "Rules: if any tool result has ok=false, final_answer text must be exactly TOOL_FAIL; otherwise provide the correct answer."
                        ),
                    }],
                }
            ],
            "tools": TOOLS_SPEC,
            "tool_choice": "auto",
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": 250,
        }
        status2, j2 = _post_response(client, base_url, api_key, payload_fallback)
        last_status, last_json = status2, j2
        if status2 == 200:
            tc2 = _extract_tool_calls(j2)
            fa2 = [tc for tc in tc2 if tc.get("name") == "final_answer"]
            if fa2:
                saw_final_answer = True
                args2 = fa2[0].get("arguments") or {}
                final_text = str(args2.get("text", "")).strip()
                executed.append({"tool": "final_answer", "arguments": args2, "result": {"ok": True}})

    ok, errs = _score(case.expected, final_text, saw_tool_call=saw_tool_call, saw_final_answer=saw_final_answer)

    return {
        "case_id": case.id,
        "level": case.level,
        "expected": case.expected,
        "ok": ok,
        "errors": errs,
        "final_text": final_text,
        "saw_tool_call": saw_tool_call,
        "saw_final_answer": saw_final_answer,
        "steps": steps,
        "executed": executed,
        "fallback_used": fallback_used,
        "last_status": last_status,
        "raw_last": None if (last_status == 200) else last_json,
    }


def _render_md(res: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# oss120b tool-calling benchmark")
    lines.append("")
    for k in ["started_at", "model", "base_url", "trials", "per_level", "temperature", "top_p", "seed"]:
        lines.append(f"- {k}: {res.get(k)}")
    lines.append("")

    summ = res.get("summary", {})
    lines.append("## Summary")
    lines.append(f"- total_pass: {summ.get('total_pass')}")
    lines.append(f"- total_fail: {summ.get('total_fail')}")
    lines.append("")

    lines.append("## By level")
    for lvl, s in (summ.get("by_level") or {}).items():
        total = s["pass"] + s["fail"]
        rate = (s["pass"] / total) if total else 0.0
        lines.append(f"### {lvl}: {s['pass']}/{total} pass ({rate:.0%})")
        if s.get("top_errors"):
            lines.append("- top_errors:")
            for name, cnt in s["top_errors"]:
                lines.append(f"  - {name}: {cnt}")
        lines.append("")

    # show first failure per level
    lines.append("## First failure example (per level)")
    for lvl, first in (res.get("first_failures") or {}).items():
        if not first:
            continue
        lines.append(f"### {lvl} / {first['case_id']}")
        lines.append(f"- errors: {first.get('errors')}")
        lines.append(f"- expected: {repr(first.get('expected'))}")
        lines.append(f"- got: {repr(first.get('final_text'))}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("OSS120B_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("OSS120B_MODEL", "openai/gpt-oss-120b"))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--per-level", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--out", default="memory/oss120b_runs/toolcall_benchmark-latest.json")
    ap.add_argument("--out-md", default="memory/oss120b_runs/toolcall_benchmark-latest.md")
    args = ap.parse_args()

    root, meta = _make_fixture_dir(seed=args.seed)
    cases = build_cases(meta, per_level=args.per_level, seed=args.seed)

    res: Dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": args.base_url,
        "model": args.model,
        "trials": args.trials,
        "per_level": args.per_level,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "sandbox_root": str(root),
        "cases": {},
        "summary": {},
        "first_failures": {},
    }

    rng = random.Random(args.seed)

    def _err_bucket(e: str) -> str:
        if e in ("no_tool_call", "no_final_answer_call", "wrong_answer", "expected_tool_fail"):
            return e
        return "other"

    by_level: Dict[str, Dict[str, Any]] = {"L1": {"pass": 0, "fail": 0, "err": {}}, "L2": {"pass": 0, "fail": 0, "err": {}}, "L3": {"pass": 0, "fail": 0, "err": {}}, "L4": {"pass": 0, "fail": 0, "err": {}}}
    first_fail: Dict[str, Any] = {"L1": None, "L2": None, "L3": None, "L4": None}

    with httpx.Client() as client:
        for case in cases:
            items = []
            for _ in range(args.trials):
                if rng.random() < 0.1:
                    time.sleep(0.05)
                items.append(run_case(client, args.base_url, args.model, root, case, args.temperature, args.top_p))

            passes = sum(1 for it in items if it["ok"])
            fails = len(items) - passes
            res["cases"][case.id] = {
                "level": case.level,
                "prompt": case.prompt,
                "expected": case.expected,
                "passes": passes,
                "fails": fails,
                "items": items,
            }

            # aggregate
            by_level[case.level]["pass"] += passes
            by_level[case.level]["fail"] += fails

            for it in items:
                if not it["ok"]:
                    for e in it.get("errors") or []:
                        b = _err_bucket(e)
                        by_level[case.level]["err"][b] = by_level[case.level]["err"].get(b, 0) + 1
                    if first_fail[case.level] is None:
                        first_fail[case.level] = {
                            "case_id": case.id,
                            "errors": it.get("errors"),
                            "expected": case.expected,
                            "final_text": it.get("final_text"),
                        }

    total_pass = sum(v["pass"] for v in by_level.values())
    total_fail = sum(v["fail"] for v in by_level.values())

    by_level_out = {}
    for lvl, s in by_level.items():
        top = sorted(s["err"].items(), key=lambda x: (-x[1], x[0]))[:6]
        by_level_out[lvl] = {"pass": s["pass"], "fail": s["fail"], "top_errors": top}

    res["summary"] = {"total_pass": total_pass, "total_fail": total_fail, "by_level": by_level_out}
    res["first_failures"] = first_fail

    out_path = (Path(__file__).parent / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), "utf-8")

    out_md_path = (Path(__file__).parent / args.out_md).resolve()
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text(_render_md(res), "utf-8")

    print(f"Sandbox: {root}")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {out_md_path}")
    print(json.dumps(res["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
