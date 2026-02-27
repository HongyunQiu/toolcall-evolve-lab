# Recipe: checkpointing_v1 (stateful, step-level checkpoints)

Goal: Make multi-step tasks resumable by writing explicit checkpoints to the sandbox.

## Required behavior
1) Plan and label steps explicitly (STEP 1/2/3...).
2) After completing each step, write/update `state.json` with:
   - `task`: short string
   - `steps`: array of objects, each:
     - `id`: e.g. "step_1_fetch"
     - `status`: "done" | "todo" | "failed"
     - `artifacts`: list of filenames created/updated in this step
     - `notes`: 1-3 lines, include source URLs when relevant
   - `artifacts_index`: map filename -> {step_id, brief}
   - `updated_at`: ISO timestamp (best effort)

3) Before starting any step, read `state.json` if it exists and SKIP steps already marked `done`.
4) If a tool fails (ok=false) and LOCAL_RETRY is enabled, first consult `state.json` and attempt a minimal repair.
5) At the end, call `list_dir('.')` and ensure required artifacts exist, then call `final()`.

## Practical tips
- Keep each `write_file` content reasonably small; split large outputs into multiple files.
- Prefer storing intermediate results in files rather than in chat text.

## Minimal `state.json` example
```json
{
  "task": "...",
  "steps": [
    {"id": "step_1", "status": "done", "artifacts": ["a.txt"], "notes": "..."},
    {"id": "step_2", "status": "todo", "artifacts": [], "notes": ""}
  ],
  "artifacts_index": {"a.txt": {"step_id": "step_1", "brief": "..."}},
  "updated_at": "2026-02-28T00:00:00+08:00"
}
```
