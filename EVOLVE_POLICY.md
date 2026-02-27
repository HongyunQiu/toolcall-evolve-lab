# EVOLVE_POLICY

This repo is a lab for evaluating and *evolving* tool-calling behavior.

## Core principle (default)
**Evolve behavior without auto-modifying core code.**

- The model/agent is allowed to *generate assets* (recipes, task sets, reports).
- The model/agent is **not** allowed to automatically modify the runner/benchmark/evolve core programs.
- Any changes to core programs are done by a human, or by an agent in “suggest-only” mode with explicit human review + apply.

Rationale:
- Prevents bricking the system ("改死掉").
- Prevents silent regressions ("改得更差").
- Prevents overfitting to a single task and losing generality ("失去通用性").

## What may be auto-written (assets)
Allowed auto-write targets (safe, reviewable, revertible):
- `recipes/` (prompt/strategy plugins)
- `tasks/` (benchmark task sets)
- `evolve/runs/` (run logs)
- Optional future: `reports/` (analysis summaries)

## What is core code (manual change only)
Manual-change-only targets (core):
- `oss120b_toolcall_run.py`
- `benchmark_run.py`
- `oss120b_toolcall_benchmark.py`
- `evolve/analyze_run.py`
- `evolve/propose_recipe.py`
- `evolve/verify_run_llm.py`

## Optional future: controlled self-modification mode (not default)
If we later want to experiment with self-modifying core code, we MUST add guardrails:
1) Changes occur only in a branch/copy (never directly on main)
2) Mandatory regression tests (py_compile + a representative benchmark subset)
3) Generality check (multi-task improvements, not single-task)
4) Diff + rationale + risk notes for human review

---
Owner intent: keep the system stable while allowing continuous improvement via externalized assets.
