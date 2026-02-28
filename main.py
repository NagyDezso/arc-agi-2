import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.orchestrator import run_all, cleanup_opencode_containers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI OpenCode CLI Solver (Docker)")
    parser.add_argument(
        "--tasks", default="all", help="'all' (default) | comma-separated IDs"
    )
    parser.add_argument(
        "--num-agents", type=int, default=1, help="Agents per test input (default: 1)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Max transform loop iterations per agent (default: 10)",
    )
    parser.add_argument(
        "--model",
        default="kilo/minimax/minimax-M2.5:free",
        help="Model in provider/model format",
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
        "--concurrency",
        type=int,
        default=5,
        help="Max simultaneous Docker containers (default: 5). Set to 0 for unlimited.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tasks to run",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_all(args))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, cleaning up...")
        cleanup_opencode_containers()
        sys.exit(130)


if __name__ == "__main__":
    main()
