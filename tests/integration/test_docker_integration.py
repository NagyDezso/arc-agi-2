import tempfile
from pathlib import Path

import pytest

from src.backends.docker_runner import DockerRunner
from src.models import AgentConfig
from src.orchestrator import SESSION_LOG_FILENAME


@pytest.mark.asyncio
async def test_docker_run_agent_failure():
    root_path = Path(__file__).parent.parent.parent / "src"
    runner = DockerRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"

        raw_task = {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[1]]}]}

        runner.setup(root_path, cli_type="opencode")

        result = await runner.start_agent_backend(
            config=AgentConfig(
                task_id="test_docker_fail",
                agent_id="agent_docker_fail",
                test_index=0,
                log_dir=log_dir,
                raw_task=raw_task,
                model="invalid/model-does-not-exist",
                envs={},
                max_iterations=1,
                soft_training_feedback=False,
                whole_task=False,
                cli_type="opencode",
            ),
        )

        assert result.task_id == "test_docker_fail"

        assert (
            result.error is not None
            or result.stderr != ""
            or any("error" in str(line).lower() for line in result.raw_lines)
        ), f"Expected the run to capture an error state, but it didn't. Result: {result}"

        assert len(result.attempts) == 0


@pytest.mark.asyncio
async def test_docker_opencode_agent():
    root_path = Path(__file__).parent.parent.parent / "src"
    runner = DockerRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"

        raw_task = {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[1]]}]}

        runner.setup(root_path, cli_type="opencode")

        result = await runner.start_agent_backend(
            config=AgentConfig(
                task_id="test_docker_int",
                agent_id="agent_docker_int",
                test_index=0,
                log_dir=log_dir,
                raw_task=raw_task,
                model="opencode/big-pickle",
                envs={},
                max_iterations=1,
                soft_training_feedback=False,
                whole_task=False,
                cli_type="opencode",
            ),
        )

        assert result.error is None or result.error == "", f"Agent run failed: {result.error}"

        session_log_path = log_dir / SESSION_LOG_FILENAME
        assert session_log_path.exists(), f"{SESSION_LOG_FILENAME} was not created by Docker runner"

        content = session_log_path.read_text()
        assert content.strip() != "", f"{SESSION_LOG_FILENAME} is empty"

        assert " started " in content, "The agent 'started' marker was not found in the session log"
