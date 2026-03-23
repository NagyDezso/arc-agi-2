import contextlib
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import aiodocker
import docker
from aiodocker.exceptions import DockerError

from src.backends.base import BackendRunner
from src.models import AgentConfig, AgentResultData, UsageTotals
from src.orchestrator import ROOT, SESSION_LOG_FILENAME, TRANSCRIPT_FILENAME

logger = logging.getLogger(__name__)

DOCKER_IMAGE = os.environ.get("ARC_SOLVER_DOCKER_IMAGE", "arc-solver:latest")
DOCKER_CPU_COUNT = int(os.environ.get("ARC_SOLVER_DOCKER_CPUS", "1"))
DOCKER_MEMORY = int(os.environ.get("ARC_SOLVER_DOCKER_MEMORY", "1")) * 1024 * 1024 * 1024


class DockerRunner(BackendRunner):

    def _sanitize_container_name(self, name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "-", name)
        return cleaned[:63] if cleaned else "arc-agent"

    def _ensure_docker_image(self, root_path: Path, cli_type: str) -> None:
        image_tag = f"arc-solver-{cli_type}:latest"
        logger.info(f"Building Docker image '{image_tag}' ...")
        dockerfile = f"Dockerfile.{cli_type}"
        client = docker.from_env()
        client.images.build(
            path=str(root_path),
            dockerfile=str(root_path / dockerfile),
            tag=image_tag,
            rm=True,
        )

    def setup(self, root_path: Path, cli_type: str) -> None:
        self._ensure_docker_image(root_path, cli_type)

    async def start_agent_backend(
        self,
        config: AgentConfig,
    ) -> AgentResultData:
        run_root = Path(tempfile.mkdtemp(prefix=f"arc-agent-{config.agent_id[:24]}-"))
        container_name = self._sanitize_container_name(f"arc-agent-{config.agent_id}-{int(time.time())}")
        run_start = time.time()
        container = None
        adocker: aiodocker.Docker | None = None

        try:
            (run_root / "config.json").write_text(config.model_dump_json(), encoding="utf-8")
            # Create src package structure for Python imports to work
            app = run_root / "app"
            app.mkdir()
            shutil.copy(ROOT / "__init__.py", app / "__init__.py")
            shutil.copy(ROOT / "agent_runner.py", app / "agent_runner.py")
            shutil.copy(ROOT / "models.py", app / "models.py")
            shutil.copytree(ROOT / "cli_impl", app / "cli_impl")

            image_tag = f"arc-solver-{config.cli_type}:latest"

            config.log_dir.mkdir(parents=True, exist_ok=True)
            session_log_path = config.log_dir / SESSION_LOG_FILENAME
            transcript_path = config.log_dir / TRANSCRIPT_FILENAME

            command = (
                "cp /workspace/config.json /root/config.json && "
                "cp -r /workspace/app/* /app/ && "
                "rm -rf /workspace/config.json /workspace/app && "
                "python3 /app/agent_runner.py"
            )

            container_config = {
                "Image": image_tag,
                "Cmd": ["bash", "-lc", command],
                "WorkingDir": "/workspace",
                "Env": [f"{key}={value}" for key, value in config.envs.items()],
                "HostConfig": {
                    "Binds": [f"{run_root.resolve()}:/workspace:rw"],
                    "CpuCount": DOCKER_CPU_COUNT,
                    "Memory": DOCKER_MEMORY,
                    "LogConfig": {"Type": "json-file", "Config": {}},
                },
            }

            adocker = aiodocker.Docker()
            container = await adocker.containers.create(config=container_config, name=container_name)
            await container.start()
            with (
                session_log_path.open("a", encoding="utf-8") as session_f,
                transcript_path.open("a", encoding="utf-8") as transcript_f,
            ):
                stdout_buffer = ""
                async for chunk in container.log(stdout=True, stderr=True, follow=True):
                    stdout_buffer += chunk
                    while "\n" in stdout_buffer:
                        line, stdout_buffer = stdout_buffer.split("\n", 1)
                        line = line.rstrip("\r")
                        if not line:
                            continue
                    self._route_agent_output_line(line, session_f, transcript_f)

            await container.wait(timeout=3600)
            results_path = run_root / "results.json"
            result = AgentResultData.model_validate_json(results_path.read_text())

            container_duration = time.time() - run_start

            # Add backend metadata
            result.backend_duration = container_duration

            logger.info(
                f"[docker-cost] {config.agent_id}: API=${result.cost:.4f}, "
                f"Total=${result.cost:.4f}, Duration={container_duration:.1f}s"
            )
        except Exception as e:
            err_msg = f"Docker error: {e}"
            logger.error(f"[docker-error] {config.agent_id}: {err_msg}")
            return AgentResultData(
                task_id=config.task_id,
                agent_id=config.agent_id,
                test_index=config.test_index,
                attempts=[],
                elapsed=0.0,
                cost=0.0,
                turns=0,
                usage=UsageTotals(input_tokens=0, cached_tokens=0, output_tokens=0),
                stderr=err_msg,
            )
        else:
            return result
        finally:
            if container is not None:
                with contextlib.suppress(DockerError):
                    await container.delete(force=True)
            shutil.rmtree(run_root, ignore_errors=True)
            if adocker is not None:
                await adocker.close()
