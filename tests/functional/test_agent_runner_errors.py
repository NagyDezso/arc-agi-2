import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to sys path to resolve imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.agent_runner import FATAL_ERRORS, run_agent
from src.models import AgentConfig


@pytest.mark.functional
@pytest.mark.parametrize(
    "error_msg",
    [
        "Error: ModelNotFoundError: Requested entity was not found.",
        "Error: Invalid model name provided.",
        "Requested entity was not found",
        "QuotaExceeded",
        "Access denied",
    ],
)
def test_run_agent_should_stop_on_fatal_errors(error_msg, mock_cli_impl):
    """Verify that run_agent stops early when it encounters various fatal errors."""
    # Mock run_session to return the specific fatal error in stderr
    mock_cli_impl.run_session.return_value = (
        [],  # raw_lines
        0,  # turns
        error_msg,  # stderr
        {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0},  # stats
    )
    mock_cli_impl.calculate_cost.return_value = 0.0

    config = AgentConfig(
        task_id="test_task",
        agent_id="test_agent_ens0",
        raw_task={"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[2]]}]},
        test_index=0,
        model="some-model",
        max_iterations=5,
        cli_type="gemini",
        log_dir=Path("/tmp"),
        envs={"GEMINI_API_KEY": "test-key"},
        soft_training_feedback=False,
        whole_task=False,
    )

    with patch("src.agent_runner.prepare_workspace", return_value=Path("/tmp")):
        result = run_agent(config, mock_cli_impl)

    # It should only call run_session once because the error is fatal
    assert mock_cli_impl.run_session.call_count == 1
    assert any(fatal in result.stderr for fatal in FATAL_ERRORS)
