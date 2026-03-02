import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import docker
from docker.errors import BuildError, DockerException, ImageNotFound

from src.orchestrator import _EVENT_FORMATTERS

logger = logging.getLogger(__name__)

DOCKER_IMAGE = os.environ.get("ARC_SOLVER_DOCKER_IMAGE", "arc-solver:latest")
DOCKER_CPU_COUNT = os.environ.get("ARC_SOLVER_DOCKER_CPUS", "2")
DOCKER_MEMORY = os.environ.get("ARC_SOLVER_DOCKER_MEMORY", "4g")

_DOCKER_CLIENT = None


def _sanitize_container_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "-", name)
    return cleaned[:63] if cleaned else "arc-agent"


def _get_docker_client() -> docker.DockerClient:
    global _DOCKER_CLIENT
    if _DOCKER_CLIENT is None:
        _DOCKER_CLIENT = docker.from_env()
    return _DOCKER_CLIENT


def cleanup_containers() -> None:
    try:
        client = _get_docker_client()
        containers = client.containers.list(all=True, filters={"name": "arc-agent"})
        for c in containers:
            try:
                c.stop(timeout=2)
            except DockerException:
                pass
            try:
                c.remove(force=True)
            except DockerException:
                pass
    except DockerException as e:
        logger.warning(f"Cleanup failed: {e}")


def _cpu_to_nano_cpus(cpu_value: str) -> int:
    return int(float(cpu_value) * 1_000_000_000)


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


def _ensure_docker_image_sync(root_path: Path, cli_type: str) -> None:
    client = _get_docker_client()
    image_tag = f"arc-solver-{cli_type}:latest"
    try:
        client.images.get(image_tag)
        return
    except ImageNotFound:
        pass

    logger.info(f"Building Docker image '{image_tag}' ...")
    dockerfile = f"Dockerfile.{cli_type}"
    try:
        client.images.build(
            path=str(root_path),
            dockerfile=str(root_path / dockerfile),
            tag=image_tag,
            rm=True,
        )
    except BuildError as e:
        raise RuntimeError(f"Docker image build failed for {image_tag}: {e}") from e
    except DockerException as e:
        raise RuntimeError(f"Docker image build failed for {image_tag}: {e}") from e


async def setup(root_path: Path, cli_type: str):
    await asyncio.to_thread(_ensure_docker_image_sync, root_path, cli_type)


def _run_agent_container_sync(
    *,
    run_root: Path,
    container_name: str,
    envs: dict[str, str],
    cli_type: str,
    log_dir: Path,
) -> tuple[dict, list[str], int]:
    client = _get_docker_client()
    stderr_lines: list[str] = []
    container = None
    exit_code = -1
    image_tag = f"arc-solver-{cli_type}:latest"

    log_dir.mkdir(parents=True, exist_ok=True)
    raw_stream_path = log_dir / "raw_stream.jsonl"

    command = (
        "cp /workspace/config.json /root/config.json && "
        "cp -r /workspace/app/* /app/ && "
        "rm -rf /workspace/config.json /workspace/app && "
        "python3 /app/agent_runner.py"
    )

    create_kwargs = {
        "image": image_tag,
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
        with open(raw_stream_path, "a") as raw_f:
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
                            raw_f.write(line + "\n")
                            raw_f.flush()
                            continue
                    except (json.JSONDecodeError, TypeError):
                        pass
                    logger.error(f"[docker-stderr] {container_name}: {line}")
                    stderr_lines.append(line)

        wait_result = container.wait(timeout=3600)
        exit_code = int(wait_result.get("StatusCode", -1))
        results_path = run_root / "results.json"
        if not results_path.exists():
            return {
                "error": f"Docker run finished with code {exit_code}, no results.json.",
                "attempts": [],
                "turns": 0,
            }, stderr_lines, exit_code
        result = json.loads(results_path.read_text())
        return result, stderr_lines, exit_code
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except DockerException:
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
    envs: dict[str, str] = {}
    # ... (envs setup remains same)
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

    run_root = Path(tempfile.mkdtemp(prefix=f"arc-agent-{agent_id[:24]}-"))
    container_name = _sanitize_container_name(
        f"arc-agent-{agent_id}-{int(time.time())}"
    )
    run_start = time.time()

    try:
        (run_root / "config.json").write_text(json.dumps(config))
        app_dir = run_root / "app"
        app_dir.mkdir()
        shutil.copy(root_path / "agent_runner.py", app_dir / "agent_runner.py")
        shutil.copytree(root_path / "cli_impl", app_dir / "cli_impl")

        result, stderr_lines, exit_code = await asyncio.to_thread(
            _run_agent_container_sync,
            run_root=run_root,
            container_name=container_name,
            envs=envs,
            cli_type=cli_type,
            log_dir=log_dir,
        )

        container_duration = time.time() - run_start
        if stderr_lines:
            existing_stderr = result.get("stderr", "")
            joined = "\n".join(stderr_lines)
            result["stderr"] = f"{existing_stderr}\n{joined}".strip()
        if exit_code != 0 and "error" not in result:
            result["error"] = f"Docker exited with code {exit_code}"
            logger.error(f"[docker-error] {agent_id}: {result['error']}")

        result["backend_duration"] = container_duration
        result["backend_cost"] = 0.0
        result["total_cost"] = result.get("cost", 0)

        logger.info(
            f"[docker-cost] {agent_id}: API=${result.get('cost', 0):.4f}, "
            f"Total=${result['total_cost']:.4f}, Duration={container_duration:.1f}s"
        )
        return result
    except Exception as e:
        err_msg = f"Docker error: {e}"
        logger.error(f"[docker-error] {agent_id}: {err_msg}", exc_info=True)
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "test_index": test_index,
            "attempts": [],
            "elapsed": 0,
            "cost": 0,
            "backend_cost": 0,
            "backend_duration": time.time() - run_start,
            "total_cost": 0,
            "turns": 0,
            "error": err_msg,
            "raw_lines": [],
            "stderr": "",
            "usage": {},
        }
    finally:
        shutil.rmtree(run_root, ignore_errors=True)
