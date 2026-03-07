import asyncio
import json
import logging
import os
import time
from pathlib import Path

from src.orchestrator import _EVENT_FORMATTERS

logger = logging.getLogger(__name__)

E2B_CPU_COUNT = 2
E2B_COST_PER_VCPU_HOUR = 0.05


async def setup(root_path: Path, cli_type: str):
    pass


async def run_agent(
    task_id: str,
    agent_id: str,
    raw_task: dict,
    test_index: int,
    model: str,
    max_iterations: int,
    soft_training_feedback: bool,
    whole_task: bool,
    cli_type: str,
    root_path: Path,
    log_dir: Path,
) -> dict:
    from e2b import ALL_TRAFFIC, AsyncSandbox

    # ... (envs setup remains same)
    envs: dict[str, str] = {}
    if cli_type == "opencode":
        kilo_key = os.environ.get("KILO_API_KEY")
        if kilo_key:
            envs["KILO_API_KEY"] = kilo_key
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            envs["GITHUB_TOKEN"] = github_token
    elif cli_type == "gemini":
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            envs["GEMINI_API_KEY"] = gemini_key
        gemini_oauth_access = os.environ.get("GEMINI_OAUTH_ACCESS_TOKEN")
        if gemini_oauth_access:
            envs["GEMINI_OAUTH_ACCESS_TOKEN"] = gemini_oauth_access
        gemini_oauth_refresh = os.environ.get("GEMINI_OAUTH_REFRESH_TOKEN")
        if gemini_oauth_refresh:
            envs["GEMINI_OAUTH_REFRESH_TOKEN"] = gemini_oauth_refresh
        gemini_oauth_id = os.environ.get("GEMINI_OAUTH_ID_TOKEN")
        if gemini_oauth_id:
            envs["GEMINI_OAUTH_ID_TOKEN"] = gemini_oauth_id

    config = {
        "task_id": task_id,
        "agent_id": agent_id,
        "raw_task": raw_task,
        "test_index": test_index,
        "model": model,
        "max_iterations": max_iterations,
        "soft_training_feedback": soft_training_feedback,
        "whole_task": whole_task,
        "cli_type": cli_type,
    }

    network = {
        "deny_out": [ALL_TRAFFIC],
        "allow_out": [
            "generativelanguage.googleapis.com",
            "api.github.com",
            "opencode.ai",
        ],
    }

    log_dir.mkdir(parents=True, exist_ok=True)
    raw_stream_path = log_dir / "raw_stream.jsonl"

    sandbox = None
    sandbox_start = time.time()
    for attempt in range(5):
        try:
            sandbox = await AsyncSandbox.create(
                template="arc-solver",
                envs=envs,
                network=network,
                timeout=43500,
            )
            sandbox_start = time.time()
            break
        except Exception as e:
            if attempt == 4:
                logger.error(f"  [e2b] {agent_id}: sandbox create failed permanently: {e}")
                raise
            wait = 2**attempt * 5
            logger.warning(
                f"  [e2b] {agent_id}: sandbox create failed (attempt {attempt + 1}/5), retrying in {wait}s: {e}"
            )
            await asyncio.sleep(wait)

    try:
        await sandbox.files.write("/root/config.json", json.dumps(config))
        await sandbox.files.write("/app/agent_runner.py", (root_path / "agent_runner.py").read_text())
        await sandbox.files.make_dir("/app/cli_impl")
        for f in (root_path / "cli_impl").glob("*.py"):
            await sandbox.files.write(f"/app/cli_impl/{f.name}", f.read_text())

        raw_f = open(raw_stream_path, "a")

        def on_stdout(output) -> None:
            line = output.line if hasattr(output, "line") else str(output)
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    if "event" in event:
                        evt_type = event.get("event", "?")
                        aid = event.get("agent_id", "?")
                        formatter = _EVENT_FORMATTERS.get(evt_type)
                        if formatter:
                            detail = formatter(event)
                            logger.info(f"  [status] {aid}: {detail}")
                        raw_f.write(line + "\n")
                        raw_f.flush()
            except (json.JSONDecodeError, TypeError):
                pass

        def on_stderr(output) -> None:
            line = output.line if hasattr(output, "line") else str(output)
            if line.strip():
                logger.error(f"  [e2b-stderr] {agent_id}: {line[:200]}")

        try:
            await sandbox.commands.run(
                "python3 /app/agent_runner.py",
                user="root",
                timeout=43200 + 120,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )
        finally:
            raw_f.close()

        results_content = await sandbox.files.read("/workspace/results.json")
        result = json.loads(results_content)

        sandbox_duration = time.time() - sandbox_start
        e2b_cost = (sandbox_duration / 3600) * E2B_CPU_COUNT * E2B_COST_PER_VCPU_HOUR
        result["backend_cost"] = e2b_cost
        result["backend_duration"] = sandbox_duration
        result["total_cost"] = result.get("cost", 0) + e2b_cost

        logger.info(
            f"  [e2b-cost] {agent_id}: API=${result.get('cost', 0):.4f}, "
            f"E2B=${e2b_cost:.4f}, Total=${result['total_cost']:.4f}, "
            f"Duration={sandbox_duration:.1f}s"
        )
        return result

    except Exception as e:
        err_msg = f"E2B sandbox error: {e}"
        logger.error(f"  [e2b-error] {agent_id}: {err_msg}", exc_info=True)
        sandbox_duration = time.time() - sandbox_start
        e2b_cost = (sandbox_duration / 3600) * E2B_CPU_COUNT * E2B_COST_PER_VCPU_HOUR
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "test_index": test_index,
            "attempts": [],
            "elapsed": 0,
            "cost": 0,
            "backend_cost": e2b_cost,
            "backend_duration": sandbox_duration,
            "total_cost": e2b_cost,
            "turns": 0,
            "error": err_msg,
            "raw_lines": [],
            "stderr": "",
            "usage": {},
        }
    finally:
        if sandbox:
            await sandbox.kill()
