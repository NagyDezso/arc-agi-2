"""Orchestrator: dispatches OpenCode CLI agents to local Docker containers.

The orchestrator runs locally and handles:
- Task loading from ARC-AGI data files
- Dispatching agents to Docker containers with OpenCode CLI installed
- Writing logs (raw_stream, transcript, readable, attempts)
- Results aggregation and summary.json
- Resume logic (skip completed tasks)

Each agent runs in its own Docker container with network disabled and a
workspace containing only task.json + transform.py loop artifacts.

Usage:
  uv run python orchestrator.py --tasks 0934a4d8 --num-agents 1
  uv run python orchestrator.py --tasks all --num-agents 3
"""

import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import docker
from docker.errors import BuildError, DockerException, ImageNotFound

ROOT = Path(__file__).resolve().parent
CHALLENGES_FILE = ROOT.parent / "data" / "arc-agi_evaluation_challenges.json"
RESULTS = ROOT / "results"
AGENT_RUNNER_PATH = ROOT / "agent_runner.py"


DOCKER_IMAGE = os.environ.get("OPENCODE_DOCKER_IMAGE", "arc-opencode-solver:latest")
DOCKERFILE_PATH = ROOT / "Dockerfile"
DOCKER_CPU_COUNT = os.environ.get("OPENCODE_DOCKER_CPUS", "2")
DOCKER_MEMORY = os.environ.get("OPENCODE_DOCKER_MEMORY", "4g")

_EVENT_FORMATTERS: dict[str, Callable[[dict], str]] = {
    "started": lambda e: f"started (model={e.get('model', '?')})",
    "iteration": lambda e: (
        f"iteration {e.get('iteration', '?')}/{e.get('max_iterations', '?')}"
    ),
    "transform_validation": lambda e: (
        f"transform {'PASS' if e.get('all_pass') else 'FAIL'} (iter {e.get('iteration', '?')})"
    ),
    "submitted": lambda e: f"submit #{e.get('attempt', '?')}",
    "done": lambda e: (
        f"done — {e.get('attempts', 0)} attempts, {e.get('elapsed', '?')}s"
    ),
    "results_written": lambda e: "results written",
    "error": lambda e: f"ERROR: {e.get('msg', '')}",
}


_DOCKER_CLIENT: docker.DockerClient | None = None


logger = logging.getLogger(__name__)


def _sanitize_container_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "-", name)
    return cleaned[:63] if cleaned else "opencode-agent"


def _get_docker_client() -> docker.DockerClient:
    global _DOCKER_CLIENT
    if _DOCKER_CLIENT is None:
        _DOCKER_CLIENT = docker.from_env()
    return _DOCKER_CLIENT


def cleanup_opencode_containers() -> None:
    """Stop and remove all opencode-* Docker containers. Call on Ctrl+C."""
    try:
        client = _get_docker_client()
        containers = client.containers.list(all=True, filters={"name": "opencode"})
        for c in containers:
            try:
                c.stop(timeout=2)
            except DockerException:
                pass
            try:
                c.remove(force=True)
            except DockerException:
                pass
        if containers:
            logger.info(f"Cleaned up {len(containers)} opencode container(s)")
    except DockerException as e:
        logger.warning(f"Cleanup failed: {e}")


def _cpu_to_nano_cpus(cpu_value: str) -> int:
    cpus = float(cpu_value)
    return int(cpus * 1_000_000_000)


def _handle_status_stdout_line(line: str) -> None:
    try:
        event = json.loads(line)
        if isinstance(event, dict):
            evt_type = event.get("event", "?")
            aid = event.get("agent_id", "?")
            formatter = _EVENT_FORMATTERS.get(evt_type)
            if formatter:
                detail = formatter(event)
                logger.info(f"[status] {aid}: {detail}")
    except (json.JSONDecodeError, TypeError):
        pass


def _ensure_docker_image_sync() -> None:
    client = _get_docker_client()
    try:
        client.images.get(DOCKER_IMAGE)
        return
    except ImageNotFound:
        pass

    logger.info(f"Building Docker image '{DOCKER_IMAGE}' from {DOCKERFILE_PATH} ...")
    try:
        # Build image synchronously
        client.images.build(
            path=str(ROOT),
            dockerfile=str(DOCKERFILE_PATH),
            tag=DOCKER_IMAGE,
            rm=True,
        )
    except BuildError as e:
        tail_lines: list[str] = []
        build_log = list(e.build_log)
        for chunk in build_log[-50:]:
            if isinstance(chunk, dict):
                stream = chunk.get("stream")
                if isinstance(stream, str) and stream:
                    tail_lines.append(stream)
                    continue
                err_detail = chunk.get("errorDetail")
                if isinstance(err_detail, dict):
                    detail_msg = err_detail.get("message")
                    if isinstance(detail_msg, str) and detail_msg:
                        tail_lines.append(detail_msg)
                        continue
                err = chunk.get("error")
                if isinstance(err, str) and err:
                    tail_lines.append(err)
        raise RuntimeError(
            f"Docker image build failed for {DOCKER_IMAGE}:\n{''.join(tail_lines)[-4000:]}"
        ) from e
    except DockerException as e:
        raise RuntimeError(f"Docker image build failed for {DOCKER_IMAGE}: {e}") from e

    logger.info(f"Docker image ready: {DOCKER_IMAGE}")


async def _ensure_docker_image() -> None:
    """Ensure Docker image exists, building it if necessary. Runs once at startup."""
    await asyncio.to_thread(_ensure_docker_image_sync)


def _run_agent_container_sync(
    *,
    run_root: Path,
    container_name: str,
    envs: dict[str, str],
) -> tuple[dict, list[str], int]:
    client = _get_docker_client()
    stderr_lines: list[str] = []
    output_buffer = ""
    container = None
    exit_code = -1

    command = (
        "cp /workspace/config.json /root/config.json && "
        "cp /workspace/agent_runner.py /app/agent_runner.py && "
        "rm -f /workspace/config.json /workspace/agent_runner.py && "
        "python3 /app/agent_runner.py"
    )

    create_kwargs = {
        "image": DOCKER_IMAGE,
        "name": container_name,
        "command": ["bash", "-lc", command],
        "detach": True,
        "working_dir": "/workspace",
        "volumes": {str(run_root.resolve()): {"bind": "/workspace", "mode": "rw"}},
        "environment": envs or None,
        "mem_limit": DOCKER_MEMORY,
        "log_config": {"type": "json-file", "config": {}},
    }
    try:
        create_kwargs["nano_cpus"] = _cpu_to_nano_cpus(DOCKER_CPU_COUNT)
    except ValueError:
        pass

    try:
        container = client.containers.create(**create_kwargs)
        container.start()

        output_buffer = ""
        for chunk in container.logs(stream=True, follow=True, tail=0):
            output_buffer += chunk.decode(errors="replace")
            while "\n" in output_buffer:
                line, output_buffer = output_buffer.split("\n", 1)
                line = line.rstrip("\r")
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if isinstance(event, dict) and "event" in event:
                        _handle_status_stdout_line(line)
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                stderr_lines.append(line)

        if stderr_lines:
            logger.error(
                f"[docker-stderr] {container_name}: {''.join(stderr_lines)}"
            )

        if output_buffer.strip():
            line = output_buffer.strip()
            try:
                event = json.loads(line)
                if isinstance(event, dict) and "event" in event:
                    _handle_status_stdout_line(line)
                else:
                    stderr_lines.append(line)
                    logger.error(f"[docker-stderr] {container_name}: {line[:200]}")
            except (json.JSONDecodeError, TypeError):
                stderr_lines.append(line)

        if stderr_lines:
            logger.error(
                f"[docker-stderr] {container_name}: {' | '.join(stderr_lines[:10])}"
            )

        wait_result = container.wait(timeout=3600)
        exit_code = int(wait_result.get("StatusCode", -1))

        results_path = run_root / "results.json"
        if not results_path.exists():
            raise RuntimeError(
                f"Docker run finished with code {exit_code}, no results.json. "
                f"stderr tail: {' | '.join(stderr_lines[-8:])}"
            )
        result = json.loads(results_path.read_text())
        return result, stderr_lines, exit_code
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except DockerException:
                pass


async def run_agent_in_docker(
    task_id: str,
    agent_id: str,
    raw_task: dict,
    test_index: int,
    model: str,
    max_iterations: int,
    soft_training_feedback: bool,
) -> dict:
    """Run an OpenCode CLI agent inside an isolated local Docker container."""
    envs: dict[str, str] = {}
    kilo_key = os.environ.get("KILO_API_KEY")
    if kilo_key:
        envs["KILO_API_KEY"] = kilo_key
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        envs["GITHUB_TOKEN"] = github_token

    config = {
        "task_id": task_id,
        "agent_id": agent_id,
        "raw_task": raw_task,
        "test_index": test_index,
        "model": model,
        "max_iterations": max_iterations,
        "soft_training_feedback": soft_training_feedback,
    }

    run_root = Path(tempfile.mkdtemp(prefix=f"opencode-{agent_id[:24]}-"))
    container_name = _sanitize_container_name(f"opencode-{agent_id}-{int(time.time())}")
    run_start = time.time()

    try:
        (run_root / "config.json").write_text(json.dumps(config))
        (run_root / "agent_runner.py").write_text(AGENT_RUNNER_PATH.read_text())

        result, stderr_lines, exit_code = await asyncio.to_thread(
            _run_agent_container_sync,
            run_root=run_root,
            container_name=container_name,
            envs=envs,
        )

        container_duration = time.time() - run_start

        if stderr_lines:
            existing_stderr = result.get("stderr", "")
            joined = "\n".join(stderr_lines)
            result["stderr"] = f"{existing_stderr}\n{joined}".strip()

        if exit_code != 0 and "error" not in result:
            result["error"] = f"Docker container exited with code {exit_code}"

        result["container_duration"] = container_duration
        result["total_cost"] = result.get("cost", 0)

        logger.info(
            f"[docker-cost] {agent_id}: API=${result.get('cost', 0):.4f}, "
            f"Infra=$0.0000, Total=${result['total_cost']:.4f}, "
            f"Duration={container_duration:.1f}s"
        )
        return result
    except Exception as e:
        container_duration = time.time() - run_start
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "test_index": test_index,
            "attempts": [],
            "elapsed": 0,
            "cost": 0,
            "container_duration": container_duration,
            "total_cost": 0.0,
            "turns": 0,
            "error": f"Docker container error: {e}",
            "raw_lines": [],
            "stderr": "",
        }
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


_ALL_TASKS: dict[str, dict] | None = None


def _load_all_tasks() -> dict[str, dict]:
    """Load challenges into {task_id: {train, test}} (cached)."""
    global _ALL_TASKS
    if _ALL_TASKS is None:
        if not CHALLENGES_FILE.exists():
            raise FileNotFoundError(f"Challenges file not found: {CHALLENGES_FILE}")
        challenges = json.loads(CHALLENGES_FILE.read_text())
        _ALL_TASKS = challenges
    return _ALL_TASKS


def load_task_ids(tasks_arg: str) -> list[str]:
    """Parse --tasks argument into list of task IDs."""
    if tasks_arg == "all":
        return sorted(_load_all_tasks().keys())
    return [t.strip() for t in tasks_arg.split(",") if t.strip()]


def load_task_json(task_id: str) -> dict:
    """Load a single task from challenges."""
    all_tasks = _load_all_tasks()
    if task_id not in all_tasks:
        raise KeyError(f"Task {task_id} not found")
    return all_tasks[task_id]


_TOOL_NAME_MAP: dict[str, str] = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "list": "Glob",
    "task": "Task",
}


def _map_tool_params(tool_name: str, params: dict) -> dict:
    """Map OpenCode tool parameters to viewer-compatible format."""
    tool_lower = tool_name.lower()
    if tool_lower == "bash":
        return {
            "command": params.get("command", ""),
            "description": params.get("description", ""),
        }
    if tool_lower == "read":
        return {"file_path": params.get("file_path", "")}
    if tool_lower in ("write", "edit"):
        return {
            "file_path": params.get("file_path", ""),
            "content": params.get("content", ""),
        }
    if tool_lower == "glob":
        return {"pattern": params.get("pattern", "")}
    if tool_lower == "grep":
        return {"pattern": params.get("pattern", ""), "path": params.get("path", "")}
    return params


def parse_opencode_stream_json(raw_lines: list[str], task_id: str) -> list[dict]:
    """Transform OpenCode JSON output into viewer-compatible transcript entries.

    OpenCode format:
    - type: "step_start" - start of a turn
    - type: "text" - text content in part.text
    - type: "tool_use" - tool use with part.tool, part.state.input/output
    - type: "step_finish" - end of turn with tokens in part.tokens
    """
    entries: list[dict] = []
    turn_counter = 0
    current_blocks: list[dict] = []
    total_tokens = {"input": 0, "output": 0, "cache_read": 0}

    def flush_assistant():
        nonlocal current_blocks, turn_counter
        if current_blocks:
            turn_counter += 1
            entries.append(
                {
                    "type": "assistant",
                    "turn": turn_counter,
                    "content": current_blocks,
                }
            )
            current_blocks = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        evt_type = obj.get("type", "")
        part = obj.get("part", {})

        if evt_type == "text":
            text = part.get("text", "")
            if text.strip():
                current_blocks.append({"type": "text", "text": text.strip()})

        elif evt_type == "tool_use":
            tool_name = part.get("tool", "")
            call_id = part.get("callID", "")
            state = part.get("state", {})
            inp = state.get("input", {})
            output = state.get("output", "")

            viewer_name = _TOOL_NAME_MAP.get(tool_name.lower(), tool_name)
            viewer_params = _map_tool_params(tool_name, inp)

            current_blocks.append(
                {
                    "type": "tool_use",
                    "name": viewer_name,
                    "id": call_id,
                    "input": viewer_params,
                }
            )

            if output and len(output) > 10:
                flush_assistant()
                truncated = output[:5000] if len(output) > 5000 else output
                entries.append(
                    {
                        "type": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": truncated,
                            }
                        ],
                    }
                )

        elif evt_type == "step_finish":
            flush_assistant()
            tokens = part.get("tokens", {})
            total_tokens["input"] += tokens.get("input", 0)
            total_tokens["output"] += tokens.get("output", 0)
            total_tokens["cache_read"] += tokens.get("cache", {}).get("read", 0)

    flush_assistant()

    if total_tokens["input"] > 0 or total_tokens["output"] > 0:
        entries.append(
            {
                "type": "result",
                "cost": 0,
                "num_turns": turn_counter,
                "usage": {
                    "input_tokens": total_tokens["input"],
                    "output_tokens": total_tokens["output"],
                    "total_tokens": total_tokens["input"] + total_tokens["output"],
                    "cached_tokens": total_tokens["cache_read"],
                },
            }
        )

    return entries


def write_agent_logs(
    result: dict,
    task_id: str,
    log_dir: Path,
) -> None:
    """Write log files from result's raw_lines."""
    log_dir.mkdir(parents=True, exist_ok=True)

    raw_lines: list[str] = result.get("raw_lines", [])

    raw_stream_path = log_dir / "raw_stream.jsonl"
    with open(raw_stream_path, "w") as f:
        for line in raw_lines:
            f.write(line + "\n")

    transcript_entries = parse_opencode_stream_json(raw_lines, task_id)
    transcript_path = log_dir / "transcript.jsonl"
    with open(transcript_path, "w") as f:
        for entry in transcript_entries:
            f.write(json.dumps(entry) + "\n")

    readable_path = log_dir / "readable.md"
    with open(readable_path, "w") as rf:
        agent_id = result.get("agent_id", "unknown")
        test_index = result.get("test_index", 0)
        rf.write(f"# Agent Log: {agent_id} (task: {task_id}, test: {test_index})\n\n")

        for line in raw_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                rf.write(f"[raw] {line}\n")
                continue

            evt_type = obj.get("type", "")
            part = obj.get("part", {})

            if evt_type == "text":
                text = part.get("text", "")
                if text.strip():
                    rf.write(f"\n**Assistant:**\n{text.strip()}\n\n")

            elif evt_type == "tool_use":
                tool_name = part.get("tool", "")
                state = part.get("state", {})
                inp = state.get("input", {})
                output = state.get("output", "")

                if tool_name.lower() == "bash":
                    rf.write(
                        f"\n\n**Tool: {tool_name}**\n```\n$ {inp.get('command', '')}\n```\n\n"
                    )
                else:
                    input_str = json.dumps(inp, indent=2)[:500]
                    rf.write(f"\n\n**Tool: {tool_name}**\n```\n{input_str}\n```\n\n")

                if output:
                    truncated = output[:2000] if len(output) > 2000 else output
                    rf.write(f"**Tool Result:**\n```\n{truncated}\n```\n\n")

            elif evt_type == "step_finish":
                tokens = part.get("tokens", {})
                rf.write(
                    f"---\n**Step:** "
                    f"tokens={tokens.get('input', 0) + tokens.get('output', 0)}, "
                    f"reason={part.get('reason', '?')}\n"
                )

    attempts_path = log_dir / "attempts.jsonl"
    with open(attempts_path, "w") as f:
        for attempt in result.get("attempts", []):
            f.write(json.dumps(attempt) + "\n")

    stderr = result.get("stderr", "")
    if stderr:
        (log_dir / "stderr.log").write_text(stderr)

    if "error" in result:
        (log_dir / "error.log").write_text(result["error"])


MAX_RETRIES = 10
INITIAL_BACKOFF_S = 2.0
MAX_BACKOFF_S = 600.0

logger = logging.getLogger(__name__)


async def _retry_backend_call(coro_fn, *, agent_id: str) -> dict:
    """Call an async function with exponential backoff + jitter on transient failures."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await coro_fn()
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(
                kw in err_str
                for kw in (
                    "deadline exceeded",
                    "unavailable",
                    "connection",
                    "timeout",
                    "reset by peer",
                    "broken pipe",
                    "eof",
                    "transport",
                    "503",
                    "502",
                    "429",
                    "rate limit",
                    "resource_exhausted",
                    "overloaded",
                    "too many requests",
                    "stopped or disabled",
                )
            )
            if not is_transient or attempt == MAX_RETRIES:
                raise
            backoff = min(INITIAL_BACKOFF_S * (2 ** (attempt - 1)), MAX_BACKOFF_S)
            jitter = random.uniform(0, backoff * 0.5)
            wait = backoff + jitter
            logger.warning(
                f"[{agent_id}] Attempt {attempt}/{MAX_RETRIES} failed: {e} — "
                f"retrying in {wait:.1f}s"
            )
            logger.warning(
                f"retry {agent_id} attempt {attempt}/{MAX_RETRIES} failed "
                f"({type(e).__name__}), retrying in {wait:.0f}s..."
            )
            await asyncio.sleep(wait)

    raise RuntimeError(f"[{agent_id}] All {MAX_RETRIES} retries exhausted")


def _write_agent_result(
    run_dir: Path, task_id: str, agent_id: str, agent_data: dict
) -> None:
    """Atomically write/update a single agent's result into the task file."""
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    task_file = task_results_dir / f"{task_id}.json"
    tmp_file = task_results_dir / f"{task_id}.json.tmp"

    if task_file.exists():
        try:
            data = json.loads(task_file.read_text())
        except (json.JSONDecodeError, OSError):
            data = {"agents": {}}
    else:
        data = {"agents": {}}

    data.setdefault("agents", {})[agent_id] = agent_data

    tmp_file.write_text(json.dumps(data, indent=2))
    os.rename(str(tmp_file), str(task_file))


async def process_task(
    task_id: str,
    args: argparse.Namespace,
    run_dir: Path,
    backend_queue: asyncio.Queue[str] | None,
) -> dict:
    """Orchestrate N agents per test input via Docker, save results independently."""
    raw_task = load_task_json(task_id)
    num_tests = len(raw_task["test"])

    agent_metas: list[tuple[str, int, Path]] = []

    async def _dispatch(
        agent_id: str, kwargs: dict, test_index: int, log_dir: Path
    ) -> dict:
        if backend_queue is None:
            result = await _retry_backend_call(
                lambda kw=kwargs: run_agent_in_docker(**kw),
                agent_id=agent_id,
            )
        else:
            token = await backend_queue.get()
            try:
                result = await _retry_backend_call(
                    lambda kw=kwargs: run_agent_in_docker(**kw),
                    agent_id=agent_id,
                )
            finally:
                backend_queue.put_nowait(token)

        if not isinstance(result, BaseException):
            write_agent_logs(result, task_id, log_dir)

            attempts = result.get("attempts", [])
            agent_data = {
                "test_index": test_index,
                "attempts": [a["grid"] for a in attempts],
                "cost": result.get("cost", 0),
                "container_duration": result.get("container_duration", 0),
                "total_cost": result.get("total_cost", 0),
                "turns": result.get("turns", 0),
                "usage": result.get("usage", {}),
            }
            _write_agent_result(run_dir, task_id, agent_id, agent_data)

        return result

    agent_coros: list = []

    for ti in range(num_tests):
        for ei in range(args.num_agents):
            agent_id = f"{task_id}_ens{ei}_t{ti}"
            agent_log_dir = run_dir / "logs" / task_id / f"t{ti}" / f"agent{ei}"
            agent_metas.append((agent_id, ti, agent_log_dir))

            _kwargs = dict(
                task_id=task_id,
                agent_id=agent_id,
                raw_task=raw_task,
                test_index=ti,
                model=args.model,
                max_iterations=args.max_iterations,
                soft_training_feedback=args.soft_training_feedback,
            )
            agent_coros.append(_dispatch(agent_id, _kwargs, ti, agent_log_dir))

    agent_results = await asyncio.gather(*agent_coros, return_exceptions=True)

    per_agent: dict[str, dict] = {}
    submitted_tests: set[int] = set()

    for (agent_id, ti, log_dir), result in zip(agent_metas, agent_results):
        if isinstance(result, BaseException):
            per_agent[agent_id] = {
                "test_index": ti,
                "attempts": [],
                "error": str(result),
            }
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "error.log").write_text(str(result))
            continue

        attempts = result.get("attempts", [])
        has_grid = any(a.get("grid") is not None for a in attempts)
        if has_grid:
            submitted_tests.add(ti)
        per_agent[agent_id] = {
            "test_index": ti,
            "attempts": [a["grid"] for a in attempts],
            "cost": result.get("cost", 0),
            "container_duration": result.get("container_duration", 0),
            "total_cost": result.get("total_cost", 0),
            "turns": result.get("turns", 0),
            "usage": result.get("usage", {}),
        }

    submitted = len(submitted_tests)
    total = num_tests

    valid_results = [r for r in agent_results if isinstance(r, dict)]
    total_cost = sum(r.get("cost", 0) for r in valid_results)
    elapsed = max((r.get("elapsed", 0) for r in valid_results), default=0)

    total_usage = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
    }
    for r in valid_results:
        usage = r.get("usage", {})
        total_usage["input_tokens"] += usage.get("input_tokens", 0)
        total_usage["cached_tokens"] += usage.get("cached_tokens", 0)
        total_usage["output_tokens"] += usage.get("output_tokens", 0)

    score_data = {
        "submitted": submitted,
        "total": total,
        "elapsed": round(elapsed, 1),
        "api_cost": round(total_cost, 4),
        "total_cost": round(total_cost, 4),
        "usage": total_usage,
    }

    task_result = {
        "score": score_data,
        "agents": per_agent,
    }
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    (task_results_dir / f"{task_id}.json").write_text(json.dumps(task_result, indent=2))

    return {
        "task_id": task_id,
        "score": score_data,
    }


async def run_all(args: argparse.Namespace):
    try:
        await _ensure_docker_image()
    except DockerException as e:
        logger.error(f"Failed to create Docker image: {e}")
        return
    task_ids = load_task_ids(args.tasks)
    logger.info(f"Loaded {len(task_ids)} tasks")

    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.is_absolute():
            run_dir = RESULTS / args.resume
        if not run_dir.exists():
            raise RuntimeError(f"Resume directory not found: {run_dir}")
        logger.info(f"Resuming run: {run_dir}")
    else:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{args.name}_{run_stamp}" if args.name else run_stamp
        run_dir = RESULTS / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    latest = RESULTS / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    logger.info(f"Run directory: {run_dir}")

    completed_tasks: dict[str, dict] = {}
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    for f in task_results_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            completed_tasks[f.stem] = data
        except Exception:
            pass
    if completed_tasks:
        logger.info(
            f"Found {len(completed_tasks)} already-completed tasks, skipping them"
        )

    remaining_ids = [tid for tid in task_ids if tid not in completed_tasks]

    all_scores: dict[str, dict] = {}
    total_submitted = 0
    total_tests = 0
    total_cost = 0.0

    for tid, data in completed_tasks.items():
        score = data.get("score", {})
        all_scores[tid] = score
        total_submitted += score.get("submitted", 0)
        total_tests += score.get("total", 0)
        total_cost += score.get("total_cost", score.get("cost", 0))

    completed = len(completed_tasks)

    backend_queue: asyncio.Queue[str] | None = None
    if args.concurrency > 0:
        backend_queue = asyncio.Queue()
        for _ in range(args.concurrency):
            backend_queue.put_nowait("opencode")

    async def _process_and_report(task_id: str):
        nonlocal completed, total_submitted, total_tests, total_cost
        try:
            result = await process_task(task_id, args, run_dir, backend_queue)
        except Exception as e:
            completed += 1
            logger.error(f"[{completed}/{len(task_ids)}] ERROR {task_id}: {e}")
            return

        score = result["score"]
        total_submitted += score["submitted"]
        total_tests += score["total"]
        total_cost += score.get("total_cost", 0)
        all_scores[task_id] = score

        completed += 1
        s = score["submitted"]
        t = score["total"]
        logger.info(
            f"[{completed}/{len(task_ids)}] "
            f"{'ok' if s == t else 'XX'} {task_id}  "
            f"{s}/{t} submitted  "
            f"({score.get('elapsed', 0):.0f}s)"
        )

    random.shuffle(remaining_ids)
    nr_remaining = len(remaining_ids)
    if args.limit:
        remaining_ids = remaining_ids[: args.limit]
    logger.info(
        f"Running {len(remaining_ids)} tasks (limit: {args.limit}) ({len(task_ids) - nr_remaining} skipped)"
    )

    await asyncio.gather(
        *[_process_and_report(tid) for tid in remaining_ids],
        return_exceptions=True,
    )

    summary = {
        "model": args.model,
        "num_agents": args.num_agents,
        "max_iterations": args.max_iterations,
        "soft_training_feedback": args.soft_training_feedback,
        "num_tasks": len(task_ids),
        "total_tests": total_tests,
        "total_cost": round(total_cost, 2),
        "tasks": all_scores,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(f"Done! {len(task_ids)} tasks, {total_tests} test inputs")
    logger.info("Score with majority voting + pass@2 post-hoc")
    logger.info(f"Summary: {run_dir / 'summary.json'}")
