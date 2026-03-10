import tempfile
from pathlib import Path

import pytest

from src.backends.docker_runner import DockerRunner
from src.log_protocol import SESSION_LOG_FILENAME
from src.models import AgentRunSpec


@pytest.mark.asyncio
async def test_docker_real_execution():
    # Setup paths
    root_path = Path(__file__).parent.parent.parent / "src"
    runner = DockerRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"

        # Fake task
        raw_task = {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[1]]}]}

        # Ensure docker image exists
        runner.setup(root_path, cli_type="opencode")

        # Run agent
        result = await runner.run_agent(
            spec=AgentRunSpec(
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
                root_path=root_path,
            ),
        )

        # Verify run_agent outputs
        assert "error" not in result or result["error"] == "", f"Agent run failed: {result.get('error')}"

        # Verify session.log was created and populated
        session_log_path = log_dir / SESSION_LOG_FILENAME
        assert session_log_path.exists(), "session.log was not created by Docker runner"

        content = session_log_path.read_text()
        assert content.strip() != "", "session.log is empty"

        # Check if the runner logged the agent start marker to stdout and captured it
        assert " started " in content, "The agent 'started' marker was not found in the session log"
