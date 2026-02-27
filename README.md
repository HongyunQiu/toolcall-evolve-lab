# toolcall-evolve-lab

A small lab to evaluate and *evolve* tool-calling behavior for a model served via an OpenAI-compatible **Responses API** (e.g. vLLM).

Core idea:
- Keep a stable runner.
- Grow capability via **tools** and **recipes**.
- Persist runs to disk, analyze failures, and auto-propose new recipes.

Project policy:
- See `EVOLVE_POLICY.md` (default: evolve via assets; do NOT auto-modify core code).

## Contents
- `oss120b_toolcall_run.py` — single-task runner (agent-like)
- `oss120b_toolcall_benchmark.py` — benchmark suite (L1–L4)
- `recipes/` — strategy “plugins”
- `evolve/` — run log analysis + recipe proposer

## Setup

### 1) Configure environment

```bash
cp .env.example .env
# load env vars into current shell
set -a; source .env; set +a
```

### 2) Run a single task (runner)

Minimal example:

```bash
python3 oss120b_toolcall_run.py \
  --task "Read notes.txt and return ONLY the second TODO item." \
  --temperature 1.0
```

Recommended (retry + anneal temperatures):

```bash
python3 oss120b_toolcall_run.py \
  --task "Read notes.txt and return ONLY the second TODO item." \
  --temperature 1.0 \
  --retries 2 \
  --retry-temps 1.0,0.7,0.2
```

Use a recipe (strategy plugin):

```bash
python3 oss120b_toolcall_run.py \
  --recipe recipes/codegen_and_run_python.md \
  --task "Write task.py, run it, and return ONLY stdout. The script prints hello then world." \
  --temperature 1.0
```

Checkpointing (resumable multi-step tasks):

```bash
python3 oss120b_toolcall_run.py \
  --recipe recipes/checkpointing_v1.md \
  --task "..." \
  --auto-fix --max-rounds 3
```

Web fetch example:

```bash
python3 oss120b_toolcall_run.py \
  --task "Fetch https://example.com and return ONLY the HTTP status code." \
  --temperature 1.0
```

Allow more CLI freedom (unsafe):

```bash
python3 oss120b_toolcall_run.py \
  --allow-any-cli --allow-shell \
  --task "Use run_cli to execute: ls | wc -l . Return ONLY the number." \
  --temperature 1.0
```

Runs are saved by default to `evolve/runs/run-*.json`.

### 3) Benchmarks

#### a) Built-in L1–L4 benchmark

```bash
python3 oss120b_toolcall_benchmark.py --trials 3 --per-level 20 --temperature 1.0
```

#### b) Realistic task-set benchmark (programming/web/docs/mixed)

Task sets:
- `tasks/tasks_v1.json` (10 tasks)
- `tasks/tasks_v1_50.json` (50 tasks)

Run (50 tasks + per-task timeout):
```bash
python3 benchmark_run.py --tasks tasks/tasks_v1_50.json --allow-any-cli --timeout-sec 600
```

Outputs:
- `benchmark_results.json` — per-task PASS/FAIL + run_log paths + verifier errors

## CLI arguments (runner)

Common arguments for `oss120b_toolcall_run.py`:

- `--local-retry`: allow limited *in-round* recovery when a tool fails (instead of immediate TOOL_FAIL).
- `--local-retry-max <n>`: max number of in-round recoveries (default 1).

- `--task <text>`: the task to perform (natural language). If omitted, reads from stdin.
- `--recipe <path>`: inject a recipe markdown file as extra system instructions.
- `--temperature <float>` / `--top-p <float>`: sampling params for the model.
- `--max-steps <int>`: maximum tool-loop turns for a single attempt.
- `--retries <int>`: retry the whole task N additional times if it fails.
- `--retry-temps <csv>`: temperature schedule per attempt (e.g. `1.0,0.7,0.2`).
- `--verbose`: print model plain-text outputs (debug).
- `--dump-json`: print the full run JSON (debug).
- `--keep-sandbox`: print sandbox path to stderr for inspection.

Run logging / evolution:
- `--save-run <path>`: write run log JSON to this path (default: `evolve/runs/run-<timestamp>.json`).
- `--no-save-run`: disable auto run logging.
- `--auto-propose`: after the run, analyze the run log and (if relevant) auto-generate a recipe under `recipes/auto/`.

CLI sandbox looseners (unsafe; not a real OS sandbox):
- `--allow-any-cli`: allow any executable name for `run_cli` (still no shell unless `--allow-shell`).
- `--allow-shell`: run `run_cli` through `/bin/zsh` (enables pipes/redirects/chaining).
- `--allow-abs-paths`: allow absolute paths in CLI args.
- `--cli-timeout <sec>`: timeout for a single `run_cli` call.

## Safety notes
This project uses a *working-directory sandbox* (temp directory). It is **not** a container/VM.

- `write_file` writes **UTF-8 text** into the sandbox using a **relative path**. Any filename/extension is allowed, but size is capped.
- `--allow-shell` is UNSAFE: it runs commands through `/bin/zsh` and enables pipes/redirects.
- Use a VM/Docker for maximal freedom (e.g. apt-get) and better isolation.

## License
TBD (pick MIT/Apache-2.0/etc).
