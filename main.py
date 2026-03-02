import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

from src.orchestrator import run_all

load_dotenv()

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI CLI Solver (Unified)")
    parser.add_argument(
        "--tasks", default="all", help="'all' (default) | comma-separated IDs"
    )
    parser.add_argument(
        "--num-agents", type=int, default=1, help="Agents per test input (default: 1)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Max transform loop iterations per agent (default: 10)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="Model name (e.g., gemini-3.1-pro-preview, kilo/minimax/minimax-m2.5:free)",
    )
    parser.add_argument(
        "--name", default=None, help="Name prefix for results directory"
    )
    parser.add_argument(
        "--resume", default=None, help="Resume a previous run directory"
    )
    parser.add_argument(
        "--soft-training-feedback",
        action="store_true",
        default=False,
        help="Use softer training failure message",
    )
    parser.add_argument(
        "--whole-task",
        action="store_true",
        default=False,
        help="Each agent sees ALL test inputs and writes one transform() applied to all",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max simultaneous agents (default: 5). Set to 0 for unlimited.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tasks to run",
    )
    parser.add_argument(
        "--cli",
        choices=["opencode", "gemini"],
        default="gemini",
        help="CLI to use (opencode or gemini, default: gemini)",
    )
    parser.add_argument(
        "--backend",
        choices=["docker", "e2b"],
        default="docker",
        help="Execution backend (docker or e2b, default: docker)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_all(args))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, exiting...")
        sys.exit(130)


if __name__ == "__main__":
    main()
