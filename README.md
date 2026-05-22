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

**Supported solver CLIs:** `gemini`, `opencode`, `junie`, `antigravity`
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
| `--cli` | `gemini` | Solver CLI: `gemini`, `opencode`, `junie`, or `antigravity` |
| `--sandbox` | `docker` | Execution sandbox: `docker` or `e2b` |
| `--model` | `gemini-2.5-flash-lite` | Model name passed to the CLI |
| `--num-agents` | 2 | Agents per test input |
| `--max-iterations` | 5 | Max refinement loops per agent |
| `--concurrency` | 2 | Max simultaneous agents |
| `--whole-task` | off | One `transform()` shared across all test inputs |
| `--name` | — | Name prefix for the results directory |
| `--resume` | — | Resume a previous run directory |

Credentials are read from `.env` (see `.env.example`). Each CLI/sandbox needs
its own keys — e.g. `E2B_API_KEY` for the E2B sandbox, `GEMINI_API_KEY` for the
Gemini CLI, `KILO_API_KEY` for OpenCode.

## Building a Submission

After a run, aggregate the results into a Kaggle-style `submission.json`:

```bash
python3 submission.py                    # gemini results
python3 submission.py --solver opencode  # opencode results
```

## Project Layout

```
main.py            CLI entry point (typer)
submission.py      Builds submission.json from run results
run.sh             Full-run wrapper for the Gemini solver
run-opencode.sh    Full-run wrapper for the OpenCode solver
src/
  orchestrator.py  Task loading, dispatch, logging, resume logic
  agent_runner.py  Per-agent run loop
  cli_impl/        CLI adapters: gemini, opencode, junie, antigravity
  sandboxes/       Sandbox runners: docker, e2b
  Dockerfile.*     Container images per CLI
data/              ARC-AGI evaluation challenges and solutions
docs/              Thesis documentation
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
</content>
</invoke>
