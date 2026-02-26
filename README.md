# toolcall-evolve-lab

A small lab to evaluate and *evolve* tool-calling behavior for a model served via an OpenAI-compatible **Responses API** (e.g. vLLM).

Core idea:
- Keep a stable runner.
- Grow capability via **tools** and **recipes**.
- Persist runs to disk, analyze failures, and auto-propose new recipes.

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

### 2) Run a single task

```bash
python3 oss120b_toolcall_run.py --task "Read notes.txt and return ONLY the second TODO item." \
  --temperature 1.0 --retries 2 --retry-temps 1.0,0.7,0.2
```

Use a recipe (strategy plugin):

```bash
python3 oss120b_toolcall_run.py \
  --recipe recipes/codegen_and_run_python.md \
  --task "Write task.py, run it, and return ONLY stdout." \
  --temperature 1.0
```

Runs are saved by default to `evolve/runs/run-*.json`.

### 3) Benchmark

```bash
python3 oss120b_toolcall_benchmark.py --trials 3 --per-level 20 --temperature 1.0
```

## Safety notes
This project uses a *working-directory sandbox* (temp directory). It is **not** a container/VM.

- `--allow-shell` is UNSAFE: it runs commands through `/bin/zsh` and enables pipes/redirects.
- Use a VM/Docker for maximal freedom (e.g. apt-get) and better isolation.

## License
TBD (pick MIT/Apache-2.0/etc).
