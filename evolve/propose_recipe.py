#!/usr/bin/env python3
"""Propose (generate) a recipe from an analyzed run.

Current focus: tag family (2)
- run_cli_command_not_allowed
- run_cli_shell_metacharacters_forbidden

Usage:
  python3 evolve/propose_recipe.py evolve/runs/<run>.json --analysis evolve/reports/<analysis>.json

Or (most common):
  python3 evolve/analyze_run.py evolve/runs/<run>.json > /tmp/analysis.json
  python3 evolve/propose_recipe.py evolve/runs/<run>.json --analysis /tmp/analysis.json

It writes a new recipe under recipes/auto/ and prints the path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


WORKSPACE = Path(__file__).resolve().parents[1]


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text("utf-8"))


def _slug(ts: str) -> str:
    return ts.replace(":", "").replace("-", "")


def propose_recipe(run: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    tags: List[str] = analysis.get("tags") or []

    relevant = [t for t in tags if t in ("run_cli_command_not_allowed", "run_cli_shell_metacharacters_forbidden")]
    if not relevant:
        return {"ok": False, "error": "no_relevant_tags"}

    ts = time.strftime("%Y%m%d-%H%M%S")
    name = "cli_fallback_no_pipes_v1"
    if "run_cli_command_not_allowed" in relevant and "run_cli_shell_metacharacters_forbidden" in relevant:
        name = "cli_fallback_allowlist_and_no_pipes_v1"
    elif "run_cli_shell_metacharacters_forbidden" in relevant:
        name = "cli_fallback_no_pipes_v1"
    else:
        name = "cli_fallback_allowlist_v1"

    recipe_dir = WORKSPACE / "recipes" / "auto"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = recipe_dir / f"{name}-{ts}.md"

    # Extract a hint of what command failed
    failed_cmds: List[str] = []
    for e in (run.get("executed") or []):
        if (e or {}).get("tool") != "run_cli":
            continue
        res = (e or {}).get("result") or {}
        if isinstance(res, dict) and res.get("ok") is False:
            args = (e or {}).get("arguments") or {}
            c = args.get("command")
            if isinstance(c, str):
                failed_cmds.append(c)

    failed_cmds = failed_cmds[:5]

    failed_cmd_lines = "\n".join(["- " + c for c in failed_cmds]) if failed_cmds else "- (none captured)"

    content = f"""# Recipe (AUTO): CLI fallback when run_cli is restricted

Generated: {ts}
Tags: {', '.join(relevant)}

## When to apply
Apply this recipe when a task requires shell-like pipelines/redirects or a disallowed command, and `run_cli` fails with:
- `shell_metacharacters_forbidden` (pipes/redirects/chaining are blocked)
- `command_not_allowed:*` (command not in allowlist)

Observed failing commands (examples):
{failed_cmd_lines}

## Tools available
- list_dir(path)
- read_file(path)
- write_file(path, content)
- run_cli(command)  # restricted: NO pipes/redirects/chaining
- fetch_url(url, max_bytes)
- final({{text}})

## Discipline
- Do NOT try to bypass restrictions.
- If a specific command cannot be executed exactly as written, switch to an alternative approach.
- Prefer using structured tools (`list_dir`, `read_file`) to replace pipelines.
- Only call `final` once you have sufficient info.

## Common substitutions (pipeline-free)
### 1) `ls | wc -l`
- Use `list_dir("")`, then count `items`.

### 2) `cat FILE | head -n 1`
- Use `read_file(FILE)`, return first line.

### 3) `grep PATTERN FILE | wc -l`
- Prefer `run_cli("grep -c '<pattern>' FILE")` if grep is allowed and no pipes.
- Or `read_file(FILE)` and count matches yourself.

### 4) Redirects like `cmd > out.txt`
- Do NOT use redirects.
- Instead, compute the content in the model, then `write_file("out.txt", content)`.

## Output
- If the user demands the forbidden form EXACTLY (must run the exact pipeline), and it is rejected: `final({{"text":"TOOL_FAIL"}})`.
- If the user allows equivalent result: use substitutions and return the equivalent output.
"""

    recipe_path.write_text(content, "utf-8")
    return {"ok": True, "recipe_path": str(recipe_path), "name": name, "tags": relevant}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json", help="Path to evolve/runs/run-*.json")
    ap.add_argument("--analysis", required=True, help="Path to analysis JSON (output of analyze_run.py)")
    args = ap.parse_args()

    run_p = Path(args.run_json)
    analysis_p = Path(args.analysis)

    run = _load_json(run_p)
    analysis = _load_json(analysis_p)

    out = propose_recipe(run, analysis)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
