import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import docker
from docker.errors import DockerException

from src.backends.base import BackendRunner
from src.log_protocol import SESSION_LOG_FILENAME, TRANSCRIPT_FILENAME, decode_stream_event
from src.models import AgentRunSpec

logger = logging.getLogger(__name__)

DOCKER_IMAGE = os.environ.get("ARC_SOLVER_DOCKER_IMAGE", "arc-solver:latest")
DOCKER_CPU_COUNT = int(os.environ.get("ARC_SOLVER_DOCKER_CPUS", "1"))
DOCKER_MEMORY = os.environ.get("ARC_SOLVER_DOCKER_MEMORY", "1g")

_IGNORED_CONTAINER_LOG_SUBSTRINGS = (
    "Performing one time database migration, may take a few minutes",
    "sqlite-migration:done",
    "Database migration complete.",
)


class DockerRunner(BackendRunner):
    client = docker.from_env()

    def _sanitize_container_name(self, name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "-", name)
        return cleaned[:63] if cleaned else "arc-agent"

    def _route_agent_output_line(self, line: str, session_file, transcript_file) -> None:
        if any(s in line for s in _IGNORED_CONTAINER_LOG_SUBSTRINGS):
            return

        event = decode_stream_event(line)
        if event is None:
            logger.info(line)
            session_file.write(line + "\n")
            session_file.flush()
            return

        if event["type"] == "status":
            message = str(event.get("message", ""))
            level = str(event.get("level", "info")).lower()
            if message:
                for part in message.splitlines() or [""]:
                    if level == "error":
                        logger.error(part)
                    elif level == "warning":
                        logger.warning(part)
                    else:
                        logger.info(part)
                session_file.write(message + "\n")
                session_file.flush()
            return

        entry = event.get("entry")
        if isinstance(entry, dict):
            transcript_file.write(json.dumps(entry) + "\n")
            transcript_file.flush()

    def _ensure_docker_image(self, root_path: Path, cli_type: str) -> None:
        image_tag = f"arc-solver-{cli_type}:latest"
        logger.info(f"Building Docker image '{image_tag}' ...")
        dockerfile = f"Dockerfile.{cli_type}"
        self.client.images.build(
            path=str(root_path),
            dockerfile=str(root_path / dockerfile),
            tag=image_tag,
            rm=True,
        )

    def setup(self, root_path: Path, cli_type: str) -> None:
        self._ensure_docker_image(root_path, cli_type)

    def _run_agent_container_sync(
        self,
        run_root: Path,
        container_name: str,
        envs: dict[str, str],
        cli_type: str,
        log_dir: Path,
    ) -> tuple[dict, list[str], int]:
        stderr_lines: list[str] = []
        container = None
        exit_code = -1
        image_tag = f"arc-solver-{cli_type}:latest"

        log_dir.mkdir(parents=True, exist_ok=True)
        session_log_path = log_dir / SESSION_LOG_FILENAME
        transcript_path = log_dir / TRANSCRIPT_FILENAME

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
            "cpu_count": DOCKER_CPU_COUNT,
            "mem_limit": DOCKER_MEMORY,
            "log_config": {"type": "json-file", "config": {}},
        }

        try:
            container = self.client.containers.create(**create_kwargs)
            container.start()
            output_buffer = ""
            with session_log_path.open("a") as session_f, transcript_path.open("a") as transcript_f:
                for chunk in container.logs(stream=True, follow=True):
                    output_buffer += chunk.decode(errors="replace")
                    while "\n" in output_buffer:
                        line, output_buffer = output_buffer.split("\n", 1)
                        line = line.rstrip("\r")
                        if not line:
                            continue
                        self._route_agent_output_line(line, session_f, transcript_f)

                if output_buffer.strip():
                    final_line = output_buffer.rstrip("\r")
                    self._route_agent_output_line(final_line, session_f, transcript_f)

            wait_result = container.wait(timeout=3600)
            exit_code = int(wait_result.get("StatusCode", -1))
            results_path = run_root / "results.json"
            if not results_path.exists():
                return (
                    {
                        "error": f"Docker run finished with code {exit_code}, no results.json.",
                        "attempts": [],
                        "turns": 0,
                    },
                    stderr_lines,
                    exit_code,
                )
            result = json.loads(results_path.read_text())
            return result, stderr_lines, exit_code
        finally:
            if container is not None:
                with contextlib.suppress(DockerException):
                    container.remove(force=True)

    async def run_agent(
        self,
        spec: AgentRunSpec,
    ) -> dict:
        config = {
            "task_id": spec.task_id,
            "agent_id": spec.agent_id,
            "raw_task": spec.raw_task,
            "test_index": spec.test_index,
            "model": spec.model,
            "max_iterations": spec.max_iterations,
            "soft_training_feedback": spec.soft_training_feedback,
            "whole_task": spec.whole_task,
            "cli_type": spec.cli_type,
        }

        run_root = Path(tempfile.mkdtemp(prefix=f"arc-agent-{spec.agent_id[:24]}-"))
        container_name = self._sanitize_container_name(f"arc-agent-{spec.agent_id}-{int(time.time())}")
        run_start = time.time()

        try:
            (run_root / "config.json").write_text(json.dumps(config))
            app_dir = run_root / "app"
            app_dir.mkdir()
            shutil.copy(spec.root_path / "agent_runner.py", app_dir / "agent_runner.py")
            shutil.copy(spec.root_path / "log_protocol.py", app_dir / "log_protocol.py")
            shutil.copytree(spec.root_path / "cli_impl", app_dir / "cli_impl")

            result, stderr_lines, exit_code = await asyncio.to_thread(
                self._run_agent_container_sync,
                run_root=run_root,
                container_name=container_name,
                envs=spec.envs,
                cli_type=spec.cli_type,
                log_dir=spec.log_dir,
            )

            container_duration = time.time() - run_start
            if stderr_lines:
                existing_stderr = result.get("stderr", "")
                joined = "\n".join(stderr_lines)
                result["stderr"] = f"{existing_stderr}\n{joined}".strip()
            if exit_code != 0 and "error" not in result:
                result["error"] = f"Docker exited with code {exit_code}"
                logger.error(f"[docker-error] {spec.agent_id}: {result['error']}")

            result["backend_duration"] = container_duration
            result["backend_cost"] = 0.0
            result["total_cost"] = result.get("cost", 0)

            logger.info(
                f"[docker-cost] {spec.agent_id}: API=${result.get('cost', 0):.4f}, "
                f"Total=${result['total_cost']:.4f}, Duration={container_duration:.1f}s"
            )
            return result
        except Exception as e:
            err_msg = f"Docker error: {e}"
            logger.error(f"[docker-error] {spec.agent_id}: {err_msg}", exc_info=True)
            return {
                "task_id": spec.task_id,
                "agent_id": spec.agent_id,
                "test_index": spec.test_index,
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
