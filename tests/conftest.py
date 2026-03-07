import pytest
import tempfile
from pathlib import Path
import json


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
    task_file.write_text(json.dumps(mock_task_data))
    return task_file
