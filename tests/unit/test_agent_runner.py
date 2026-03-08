import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Add src to sys path to resolve cli_impl import inside agent_runner
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.agent_runner import _format_diff, test_transform


def test_format_diff():
    expected = np.array([[1, 2], [3, 4]])
    actual = np.array([[1, 0], [3, 4]])
    diff = _format_diff(expected, actual)

    assert "Value mismatch: 1 cell(s) differ" in diff
    assert "  (0,1): 2 -> 0" in diff
    assert "[[1 2]" in diff


def test_format_diff_shape_mismatch():
    expected = np.array([[1, 2], [3, 4]])
    actual = np.array([[1, 2]])
    diff = _format_diff(expected, actual)

    assert "Shape mismatch: expected (2, 2), got (1, 2)" in diff


@patch("src.agent_runner.run_with_timeout")
def test_test_transform_success(mock_run, tmp_path):
    # Mock run_with_timeout to just call the function directly
    mock_run.side_effect = lambda fn, arg: fn(arg)

    transform_file = tmp_path / "transform.py"
    transform_file.write_text("""
import numpy as np
def transform(grid):
    return grid + 1
""")

    train_examples = [
        {"input": [[1, 2]], "output": [[2, 3]]},
        {"input": [[5, 6]], "output": [[6, 7]]},
    ]

    all_pass, msg, fn = test_transform(transform_file, train_examples)
    assert all_pass is True
    assert msg == "All training examples pass."
    assert fn is not None


@patch("src.agent_runner.run_with_timeout")
def test_test_transform_failure(mock_run, tmp_path):
    mock_run.side_effect = lambda fn, arg: fn(arg)

    transform_file = tmp_path / "transform.py"
    transform_file.write_text("""
import numpy as np
def transform(grid):
    return grid
""")

    train_examples = [{"input": [[1, 2]], "output": [[2, 3]]}]

    all_pass, msg, fn = test_transform(transform_file, train_examples)
    assert all_pass is False
    assert "Value mismatch" in msg
    assert fn is None
