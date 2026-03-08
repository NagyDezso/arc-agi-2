import tempfile
from pathlib import Path

import pytest

from src.backends.docker_runner import DockerRunner
from src.models import AgentRunSpec


@pytest.mark.asyncio
async def test_docker_run_agent_failure():
    # Setup paths
    root_path = Path(__file__).parent.parent.parent / "src"
    runner = DockerRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"

        # Fake task
        raw_task = {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[1]]}]}

        # Ensure docker image exists
        runner.setup(root_path, cli_type="opencode")

        # Run agent with an invalid model to trigger an API failure inside the container
        result = await runner.run_agent(
            spec=AgentRunSpec(
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
                root_path=root_path,
            ),
        )

        # Depending on how the error is surfaced, it should be in stderr or error.
        # But the orchestrator process itself must NOT crash.
        assert type(result) is dict
        assert "task_id" in result

        has_error_field = bool(result.get("error"))
        has_stderr = bool(result.get("stderr"))
        has_error_in_lines = any("error" in str(line).lower() for line in result.get("raw_lines", []))

        assert has_error_field or has_stderr or has_error_in_lines, (
            f"Expected the run to capture an error state, but it didn't. Result: {result}"
        )

        # Assert attempts is likely empty since it failed
        assert len(result.get("attempts", [])) == 0
