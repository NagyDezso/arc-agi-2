import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to sys path to resolve imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.agent_runner import run_agent


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
def test_run_agent_should_stop_on_fatal_errors(error_msg):
    """Verify that run_agent stops early when it encounters various fatal errors."""
    mock_impl = MagicMock()
    # Mock run_session to return the specific fatal error in stderr
    mock_impl.run_session.return_value = (
        [],  # raw_lines
        0,  # turns
        error_msg,  # stderr
        {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0},  # stats
        False,  # session_started_now
    )
    mock_impl.calculate_cost.return_value = 0.0

    config = {
        "task_id": "test_task",
        "agent_id": "test_agent_ens0",
        "raw_task": {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[2]]}]},
        "test_index": 0,
        "model": "some-model",
        "max_iterations": 5,
        "cli_type": "gemini",
    }

    with patch("cli_impl.get_cli_impl", return_value=mock_impl):
        with patch("src.agent_runner.prepare_workspace", return_value=Path("/tmp")):
            result = run_agent(config)

    # It should only call run_session once because the error is fatal
    assert mock_impl.run_session.call_count == 1
    assert any(
        fatal in result["stderr"]
        for fatal in ["ModelNotFoundError", "Invalid model", "Requested entity", "QuotaExceeded", "Access denied"]
    )
