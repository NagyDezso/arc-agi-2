import json
import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.models import UsageTotals


@pytest.fixture
def mock_task_data():
    return {
        "train": [{"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]}],
        "test": [{"input": [[0, 0], [0, 0]]}],
    }


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_raw_task_file(temp_workspace, mock_task_data):
    task_file = temp_workspace / "task.json"
    task_file.write_text(json.dumps(mock_task_data), encoding="utf-8")
    return task_file

@pytest.fixture
def mock_test_prompt() -> str:
    return f"""Hello this a test prompt."""


@pytest.fixture
def mock_cli_impl():
    """Create a MagicMock configured for CLI implementation."""
    mock = MagicMock()
    mock.workspace_extras.return_value = None
    mock.calculate_cost.return_value = 0.0
    mock.run_session.return_value = ([], 0, "", UsageTotals())
    mock.extract_grid_from_output.return_value = None
    mock.write_readable_log.side_effect = lambda rf, obj: rf.write(f"Parsed readable: {obj}\n")
    return mock
