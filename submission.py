#!/usr/bin/env python3
"""Build a Kaggle submission from ARC-AGI solver results.

For each task and each test index:
  - Pool ALL grids from all agents into a single list
  - Rank by frequency count across the pool
  - attempt_1 = most common grid (rank 1)
  - attempt_2 = second most common grid (rank 2), or [[0]] fallback

Writes submission.json and cost_breakdown.json to project root,
and scores against ground truth.

Usage:
  python3 submission.py          # reads from src/results/latest
  python3 submission.py --results-dir path/to/results  # custom path
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.orchestrator import TRANSCRIPT_FILENAME

Grid = list[list[int]]

logger = logging.getLogger(__name__)


def canonicalize_grid(grid: Grid) -> str:
    """Stable JSON string for a grid (for hashing / counting)."""
    return json.dumps(grid, separators=(",", ":"))


def top_k_vote(grids: list[Grid], k: int = 2) -> list[Grid]:
    """Return the top-k most common grids from a pool, ordered by frequency.

    If the pool is empty, returns an empty list. Ties are broken by
    whichever grid was seen first (Counter.most_common is stable).
    """
    if not grids:
        return []
    counts: Counter[str] = Counter()
    lookup: dict[str, Grid] = {}
    for g in grids:
        key = canonicalize_grid(g)
        counts[key] += 1
        lookup[key] = g
    return [lookup[key] for key, _ in counts.most_common(k)]


# ── Solver result loading ─────────────────────────────────────────────────


def load_solver_grids(results_dir: Path) -> dict[str, dict[int, list[Grid]]]:
    """Load solver results, preferring attempts.jsonl logs for proper test_index.

    In whole-task mode, each agent solves ALL test inputs and the attempts.jsonl
    log files contain per-attempt test_index fields. The task_results JSON only
    stores a single test_index per agent (always 0 in whole-task mode), so we
    prefer reading from log files when available.

    Falls back to task_results/*.json when log files are missing.

    Returns: {task_id: {test_index: [grid, grid, ...]}}
    """
    out: dict[str, dict[int, list[Grid]]] = {}

    # Try loading from attempts.jsonl log files first (has proper test_index)
    logs_dir = results_dir / "logs"
    if logs_dir.is_dir():
        for attempts_file in sorted(logs_dir.rglob("attempts.jsonl")):
            # Extract task_id from path: logs/<task_id>/...
            parts = attempts_file.relative_to(logs_dir).parts
            if len(parts) < 2:
                continue
            task_id = parts[0]
            if task_id not in out:
                out[task_id] = {}
            for line in attempts_file.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    attempt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ti: int = attempt.get("test_index", 0)
                grid: Grid | None = attempt.get("grid")
                if grid is not None:
                    out.setdefault(task_id, {}).setdefault(ti, []).append(grid)

    # Fall back to task_results/*.json for any tasks not found in logs
    task_results_dir = results_dir / "task_results"
    if task_results_dir.is_dir():
        for f in sorted(task_results_dir.glob("*.json")):
            task_id = f.stem
            if task_id in out:
                continue  # already loaded from logs
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"skipping corrupt {f}: {e}")
                continue
            agents: dict[str, dict] = data.get("agents", {})
            per_test: dict[int, list[Grid]] = {}

            for agent_info in agents.values():
                ti = agent_info.get("test_index", 0)
                attempts: list[Grid] = agent_info.get("attempts", [])
                if ti not in per_test:
                    per_test[ti] = []
                per_test[ti].extend(attempts)

            out[task_id] = per_test

    if not out:
        logger.warning(f"no results found in {results_dir}")

    return out


# ── Summary loading ───────────────────────────────────────────────────────


def load_summary(results_dir: Path) -> dict[str, Any] | None:
    """Load summary.json from a solver results directory."""
    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        return None
    return json.loads(summary_file.read_text())


# ── Ground truth loading ──────────────────────────────────────────────────


def load_ground_truth(data_dir: Path) -> dict[str, list[Grid]]:
    """Load ground truth from arc-agi_evaluation_solutions.json.

    Returns: {task_id: [gt_output_for_test_0, gt_output_for_test_1, ...]}
    """
    solutions_file = data_dir / "arc-agi_evaluation_solutions.json"
    if not solutions_file.exists():
        logger.warning(f"ground truth not found: {solutions_file}")
        return {}
    return json.loads(solutions_file.read_text())


# ── Scoring ───────────────────────────────────────────────────────────────


def score_submission(
    submission: dict[str, list[dict[str, Grid | None]]],
    ground_truth: dict[str, list[Grid]],
) -> tuple[float, int, int]:
    """Score a Kaggle submission against ground truth.

    Returns: (arc_mean_score, num_tasks_scored, total_correct_tasks)
    """
    per_task_scores: dict[str, float] = {}

    for task_id, gt_outputs in ground_truth.items():
        if task_id not in submission:
            per_task_scores[task_id] = 0.0
            continue

        preds = submission[task_id]
        correct = 0
        for ti, gt in enumerate(gt_outputs):
            if ti >= len(preds):
                continue
            pack = preds[ti] or {}
            a1 = pack.get("attempt_1")
            a2 = pack.get("attempt_2")
            if (a1 is not None and a1 == gt) or (a2 is not None and a2 == gt):
                correct += 1
        per_task_scores[task_id] = correct / max(len(gt_outputs), 1)

    if not per_task_scores:
        return 0.0, 0, 0

    arc_mean = sum(per_task_scores.values()) / len(per_task_scores)
    perfect_tasks = sum(1 for s in per_task_scores.values() if s == 1.0)
    return arc_mean, len(per_task_scores), perfect_tasks


# ── Cost breakdown ────────────────────────────────────────────────────────


def _aggregate_costs_from_task_results(results_dir: Path) -> dict[str, dict]:
    """Fallback: aggregate per-agent costs from task_results/*.json files.

    Used when summary.json is missing (e.g. process killed before writing it).
    Returns: {task_id: {"api_cost": float, "backend_cost": float, "total_cost": float, "usage": dict}}
    """
    task_results_dir = results_dir / "task_results"
    if not task_results_dir.is_dir():
        return {}

    out: dict[str, dict] = {}
    for f in sorted(task_results_dir.glob("*.json")):
        task_id = f.stem
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Prefer score section if available (new format)
        score = data.get("score", {})
        if score:
            out[task_id] = {
                "api_cost": score.get("api_cost", 0),
                "backend_cost": score.get("backend_cost", 0),
                "total_cost": score.get("total_cost", 0),
                "usage": score.get("usage", {}),
            }
        else:
            # Fallback: aggregate from agents
            agents = data.get("agents", {})
            api_cost = 0.0
            backend_cost = 0.0
            total_cost = 0.0
            usage: dict[str, int] = {}
            for agent_info in agents.values():
                api_cost += agent_info.get("cost", 0)
                backend_cost += agent_info.get("backend_cost", 0)
                total_cost += agent_info.get("total_cost", 0)
                for k, v in agent_info.get("usage", {}).items():
                    usage[k] = usage.get(k, 0) + (v if isinstance(v, int) else 0)
            out[task_id] = {
                "api_cost": api_cost,
                "backend_cost": backend_cost,
                "total_cost": total_cost,
                "usage": usage,
            }
    return out


def extract_cost_breakdown(results_dir: Path, num_tasks: int) -> dict:
    """Extract cost breakdown from the solver results.

    Falls back to aggregating from task_results/*.json when summary.json
    is missing (e.g. process killed before writing it).
    """
    summary = load_summary(results_dir)

    # Get CLI and backend info from summary
    cli = "unknown"
    backend = "unknown"
    model = "unknown"
    if summary:
        cli = summary.get("cli", "unknown")
        backend = summary.get("backend", "unknown")
        model = summary.get("model", "unknown")

    # Determine backend pricing description
    if backend == "e2b":
        backend_pricing = {
            "vcpus": 2,
            "cost_per_vcpu_hour_usd": 0.05,
            "total_hourly_rate_usd": 0.10,
            "formula": "(duration_seconds / 3600) * vcpus * cost_per_vcpu_hour",
        }
    elif backend == "docker":
        backend_pricing = {
            "note": "Local Docker execution - no cloud infrastructure costs",
            "cost": 0.0,
        }
    else:
        backend_pricing = {"note": f"Unknown backend: {backend}"}

    breakdown = {
        "_documentation": {
            "purpose": "Cost breakdown",
            "generated_by": "submission.py",
            "cli": cli,
            "backend": backend,
            "backend_pricing": backend_pricing,
            "note": f"API costs calculated from {cli} CLI token reports using published pricing",
        },
        "solver": {},
        "metadata": {"cli": cli, "backend": backend, "num_tasks": num_tasks},
    }

    if summary:
        tasks = summary.get("tasks", {})
        api_cost = sum(task.get("api_cost", 0.0) for task in tasks.values())
        backend_cost = sum(task.get("backend_cost", 0.0) for task in tasks.values())
        total = api_cost + backend_cost

        total_usage = {
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
        }
        for task in tasks.values():
            usage = task.get("usage", {})
            total_usage["input_tokens"] += usage.get("input_tokens", 0)
            total_usage["cached_tokens"] += usage.get("cached_tokens", 0)
            total_usage["output_tokens"] += usage.get("output_tokens", 0)

        breakdown["solver"] = {
            "cli": cli,
            "backend": backend,
            "model": model,
            "api_cost": round(api_cost, 4),
            "backend_cost": round(backend_cost, 4),
            "total_cost": round(total, 4),
            "usage": total_usage,
            "per_task": {
                task_id: {
                    "api_cost": round(task.get("api_cost", 0.0), 4),
                    "backend_cost": round(task.get("backend_cost", 0.0), 4),
                    "total_cost": round(task.get("total_cost", 0.0), 4),
                    "elapsed_seconds": round(task.get("elapsed", 0.0), 2),
                    "usage": task.get("usage", {}),
                }
                for task_id, task in tasks.items()
            },
        }
    else:
        # Fallback: aggregate from task_results/*.json
        task_costs = _aggregate_costs_from_task_results(results_dir)
        if task_costs:
            api_cost = sum(t["api_cost"] for t in task_costs.values())
            backend_cost = sum(t["backend_cost"] for t in task_costs.values())
            total = sum(t["total_cost"] for t in task_costs.values())
            total_usage: dict[str, int] = {}
            for t in task_costs.values():
                for k, v in t["usage"].items():
                    total_usage[k] = total_usage.get(k, 0) + v
            breakdown["solver"] = {
                "cli": cli,
                "backend": backend,
                "model": f"{model} (from task_results)",
                "api_cost": round(api_cost, 4),
                "backend_cost": round(backend_cost, 4),
                "total_cost": round(total, 4),
                "usage": total_usage,
                "per_task": {
                    task_id: {
                        "api_cost": round(t["api_cost"], 4),
                        "backend_cost": round(t["backend_cost"], 4),
                        "total_cost": round(t["total_cost"], 4),
                        "usage": t["usage"],
                    }
                    for task_id, t in task_costs.items()
                },
            }
        else:
            breakdown["solver"] = {
                "cli": cli,
                "backend": backend,
                "model": "N/A",
                "api_cost": 0.0,
                "backend_cost": 0.0,
                "total_cost": 0.0,
                "usage": {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0},
                "per_task": {},
            }

    return breakdown


def print_cost_report(cost_breakdown: dict) -> None:
    """Print human-readable cost breakdown to console."""
    logger.info(f"\n{'=' * 60}")
    logger.info("COST BREAKDOWN")
    logger.info(f"{'=' * 60}")

    s = cost_breakdown["solver"]
    s_usage = s.get("usage", {})
    logger.info(f"\nCLI:      {s.get('cli', 'unknown')}")
    logger.info(f"Backend:  {s.get('backend', 'unknown')}")
    logger.info(f"Model:    {s['model']}")
    logger.info(f"  API Cost:      ${s['api_cost']:.4f}")
    logger.info(f"  Backend Cost:  ${s['backend_cost']:.4f}")
    logger.info(f"  TOTAL:         ${s['total_cost']:.4f}")
    if s_usage:
        logger.info(
            f"  Tokens:        input={s_usage.get('input_tokens', 0):,}, "
            f"cached={s_usage.get('cached_tokens', 0):,}, "
            f"output={s_usage.get('output_tokens', 0):,}"
        )
    logger.info(f"{'=' * 60}")


def write_cost_breakdown_file(cost_breakdown: dict, output_dir: Path) -> None:
    """Write cost breakdown to cost_breakdown.json."""
    cost_file = output_dir / "cost_breakdown.json"
    cost_file.write_text(json.dumps(cost_breakdown, indent=2))
    logger.info(f"Wrote cost breakdown to: {cost_file}")


# ── Transcript security check ────────────────────────────────────────────


_SUSPICIOUS_STRINGS = [
    # API key env var names, network access
    r"GEMINI_API_KEY",
    r"GOOGLE_API_KEY",
    r"KILO_API_KEY",
    r"GITHUB_TOKEN",
    r"https://",
    r"http://",
    # Env inspection
    r"printenv",
    r"os\.environ",
    r"/proc/self/environ",
    r"/proc/\d+/environ",
]


def check_transcripts(results_dir: Path) -> list[str]:
    """Scan transcript logs for API key access or env inspection attempts.

    Returns a list of warning strings (empty if clean).
    """
    warnings: list[str] = []

    logs_dir = results_dir / "logs"
    if not logs_dir.is_dir():
        return warnings

    for transcript_file in sorted(logs_dir.rglob(TRANSCRIPT_FILENAME)):
        rel = transcript_file.relative_to(results_dir)
        lines = transcript_file.read_text()
        for string in _SUSPICIOUS_STRINGS:
            if (index := lines.find(string)) != -1:
                warnings.append(f'[{rel}] suspicious string "{string}" found at index {index}')
                break
    return warnings


# ── Main ──────────────────────────────────────────────────────────────────


def build_submission(
    solver_grids: dict[str, dict[int, list[Grid]]],
    ground_truth: dict[str, list[Grid]],
) -> dict[str, list[dict[str, Grid | None]]]:
    """Build Kaggle submission from solver output.

    For each task/test, grids are ranked by frequency:
      attempt_1 = most common grid (rank 1)
      attempt_2 = second most common grid (rank 2), or [[0]] fallback
    """
    all_task_ids = sorted(set(ground_truth.keys()) | set(solver_grids.keys()))
    submission: dict[str, list[dict[str, Grid | None]]] = {}

    for task_id in all_task_ids:
        num_tests = len(ground_truth.get(task_id, []))
        if num_tests == 0:
            # Infer from solver outputs
            tests = solver_grids.get(task_id, {})
            all_indices = set(tests.keys())
            num_tests = max(all_indices, default=-1) + 1

        preds: list[dict[str, Grid | None]] = []
        tests = solver_grids.get(task_id, {})

        for ti in range(num_tests):
            pool = tests.get(ti, [])
            top = top_k_vote(pool, k=2)

            preds.append(
                {
                    "attempt_1": top[0] if len(top) >= 1 else [[0]],
                    "attempt_2": top[1] if len(top) >= 2 else [[0]],
                }
            )

        submission[task_id] = preds

    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Kaggle submission from ARC-AGI solver results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Path to results directory (default: src/results/latest)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to data directory with ground truth (default: data/ next to script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for submission.json (default: submission.json in script dir)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir / "data"

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = script_dir / "src" / "results" / "latest"

    output_path = args.output or script_dir / "submission.json"

    # Resolve symlinks (e.g. results/latest -> actual run dir)
    if results_dir.is_symlink():
        results_dir = results_dir.resolve()

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    logger.info(f"Results dir: {results_dir}")

    # Load results
    solver_grids = load_solver_grids(results_dir)
    logger.info(f"Loaded {len(solver_grids)} tasks from solver results")

    # Load ground truth
    ground_truth = load_ground_truth(data_dir)

    # Build submission
    submission = build_submission(solver_grids, ground_truth)
    logger.info(f"Built submission with {len(submission)} tasks")

    # Score
    if ground_truth:
        arc_mean, num_scored, perfect = score_submission(submission, ground_truth)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ARC-mean score: {arc_mean * 100:.2f}% ({arc_mean * num_scored:.2f}/{num_scored})")
        logger.info(f"Perfect tasks:  {perfect}/{num_scored}")
        logger.info(f"{'=' * 60}")

    # Cost breakdown
    logger.info(f"\n{'=' * 60}")
    logger.info("Extracting cost breakdown...")
    cost_breakdown = extract_cost_breakdown(results_dir, num_tasks=len(ground_truth))
    print_cost_report(cost_breakdown)
    write_cost_breakdown_file(cost_breakdown, output_path.parent)

    # Security: check transcripts for API key access attempts
    logger.info(f"\n{'=' * 60}")
    logger.info("Checking transcripts for suspicious patterns...")
    security_warnings = check_transcripts(results_dir)
    if security_warnings:
        logger.info(f"\n  WARNING: {len(security_warnings)} suspicious pattern(s) found:")
        for w in security_warnings[:20]:  # Limit output
            logger.info(f"    {w}")
        if len(security_warnings) > 20:
            logger.info(f"    ... and {len(security_warnings) - 20} more")
    else:
        logger.info("  Clean — no suspicious patterns found.")
    logger.info(f"{'=' * 60}")

    # Write submission
    output_path.write_text(json.dumps(submission))
    logger.info(f"\nWrote submission to: {output_path}")


if __name__ == "__main__":
    main()
