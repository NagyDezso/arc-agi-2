<div align="center">

# Unified ARC-AGI CLI Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ARC-AGI-2](https://img.shields.io/badge/Task-ARC--AGI--2-red.svg)](https://arcprize.org/)

**Solves [ARC-AGI-2](https://arcprize.org/) tasks by dispatching coding-agent CLIs into isolated sandboxes.**

</div>

---

## Overview

This project solves ARC-AGI tasks by running AI coding-agent CLIs as solvers.
For each task, multiple agents run in parallel inside isolated sandboxes; each
agent writes and iteratively refines a Python `transform()` function against the
training examples, then applies it to the test inputs.

It started as a fork of Confluence Labs' ARC-AGI-2 solver and has been
generalized into a unified harness with pluggable CLIs and sandboxes.

**Supported solver CLIs:** `gemini`, `opencode`, `junie`, `antigravity`, `claude`
**Supported sandboxes:** `docker` (local), `e2b` (remote)

## Quick Start

```bash
# Install dependencies
uv sync

# Configure credentials
cp .env.example .env
# Fill in the keys for the CLI/sandbox you intend to use

# Run directly
uv run python main.py --tasks all --cli gemini --sandbox docker

# Smoke test a single task
uv run python main.py --tasks <task_id> --cli gemini --sandbox docker
```

Convenience wrappers reproduce full-run configurations with a 12-hour
wall-clock circuit breaker and automatic submission building:

```bash
./run.sh                      # Gemini CLI solver
./run.sh --smoke <task_id>    # single-task smoke test
./run-opencode.sh             # OpenCode CLI solver
```

## Configuration

`main.py` options (see `--help` for the full list):

| Option | Default | Description |
|--------|---------|-------------|
| `--tasks` | `all` | `all` or comma-separated task IDs |
| `--cli` | `gemini` | Solver CLI: `gemini`, `opencode`, `junie`, `antigravity`, or `claude` |
| `--sandbox` | `docker` | Execution sandbox: `docker` or `e2b` |
| `--model` | `gemini-2.5-flash` | Model name passed to the CLI |
| `--dataset` | `arc-prize-2025/arc-agi_evaluation_challenges.json` | Challenges JSON, relative to `data/` or an absolute path |
| `--num-agents` | 2 | Agents per test input |
| `--max-iterations` | 5 | Max refinement loops per agent |
| `--concurrency` | 2 | Max simultaneous agents |
| `--limit` | — | Cap the number of tasks to run |
| `--whole-task` | off | One `transform()` shared across all test inputs |
| `--soft-training-feedback` | off | Use a softer training-failure message |
| `--name` | — | Name prefix for the results directory |
| `--resume` | — | Resume a previous run directory |

The bundled datasets live under `data/arc-prize-2024/` and
`data/arc-prize-2025/`. Pass `--dataset arc-prize-2024/arc-agi_evaluation_challenges.json`
to solve the 2024 evaluation set instead of the 2025 default.

Credentials are read from `.env` (see `.env.example`). Each CLI/sandbox needs
its own keys — e.g. `E2B_API_KEY` for the E2B sandbox, `GEMINI_API_KEY` for the
Gemini CLI, `KILO_API_KEY` for OpenCode.

## Building a Submission

After a run, aggregate the results into a Kaggle-style `submission.json`:

```bash
# Aggregate the most recent run (results/latest) against the 2025 solutions
python3 submission.py

# Score a specific run directory against a chosen solutions file
python3 submission.py \
    --results-dir results/<run-dir> \
    --dataset arc-prize-2024/arc-agi_evaluation_solutions.json \
    --output submission.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--results-dir` | `results/latest` | Run directory to aggregate |
| `--dataset` | `arc-prize-2025/arc-agi_evaluation_solutions.json` | Ground-truth solutions JSON, relative to `data/` or an absolute path |
| `--output` | `submission.json` | Output path for the submission file |

## Project Layout

```
main.py            CLI entry point (typer)
submission.py      Builds submission.json from run results
run.sh             Full-run wrapper for the Gemini solver
run-opencode.sh    Full-run wrapper for the OpenCode solver
src/
  orchestrator.py  Task loading, dispatch, logging, resume logic
  agent_runner.py  Per-agent run loop
  cli_impl/        CLI adapters: gemini, opencode, junie, antigravity, claude
  sandboxes/       Sandbox runners: docker, e2b
  dockerfiles/     Container images per CLI (Dockerfile.<cli>)
data/
  arc-prize-2024/  ARC-AGI-1 evaluation challenges and solutions
  arc-prize-2025/  ARC-AGI-2 evaluation challenges and solutions
docs/              Documentation
tests/             unit / functional / integration tests
```

## Testing

```bash
uv run pytest
```

- **unit/** — single function or class in isolation; fast, no side effects
- **functional/** — component workflows with mocked dependencies; fast
- **integration/** — real external dependencies (Docker, APIs); have side effects

## License

MIT — see [LICENSE](LICENSE).
