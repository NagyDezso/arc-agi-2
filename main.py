import asyncio
import logging

import typer
from dotenv import load_dotenv

from src.models import CliArgs
from src.orchestrator import run_all

load_dotenv()

logger = logging.getLogger(__name__)
app = typer.Typer(help="ARC-AGI CLI Solver (Unified)")


@app.command()
def main(
    tasks: str = typer.Option("all", "--tasks", help="'all' (default) | comma-separated IDs"),
    num_agents: int = typer.Option(2, "--num-agents", help="Agents per test input (default: 2)"),
    max_iterations: int = typer.Option(
        5, "--max-iterations", help="Max transform loop iterations per agent (default: 5)"
    ),
    model: str = typer.Option(
        "gemini-2.5-flash-lite",
        "--model",
        help="Model name (e.g., gemini-3.1-pro-preview, kilo/minimax/minimax-m2.5:free)",
    ),
    name: str | None = typer.Option(None, "--name", help="Name prefix for results directory"),
    resume: str | None = typer.Option(None, "--resume", help="Resume a previous run directory"),
    soft_training_feedback: bool = typer.Option(
        False,
        "--soft-training-feedback",
        help="Use softer training failure message",
    ),
    whole_task: bool = typer.Option(
        False,
        "--whole-task",
        help="Each agent sees ALL test inputs and writes one transform() applied to all",
    ),
    concurrency: int = typer.Option(2, "--concurrency", help="Max simultaneous agents (default: 2)."),
    limit: int | None = typer.Option(None, "--limit", help="Limit the number of tasks to run"),
    cli: str = typer.Option(
        "gemini",
        "--cli",
        help="CLI to use: opencode, gemini, or junie (default: gemini)",
    ),
    backend: str = typer.Option(
        "docker",
        "--backend",
        help="Execution backend (docker or e2b, default: docker)",
    ),
) -> None:
    """ARC-AGI CLI Solver - solves ARC-AGI tasks using AI agents."""
    args = CliArgs(
        tasks=tasks,
        num_agents=num_agents,
        max_iterations=max_iterations,
        model=model,
        name=name,
        resume=resume,
        soft_training_feedback=soft_training_feedback,
        whole_task=whole_task,
        concurrency=concurrency,
        limit=limit,
        cli=cli,
        backend=backend,
    )
    asyncio.run(run_all(args))


if __name__ == "__main__":
    app()
