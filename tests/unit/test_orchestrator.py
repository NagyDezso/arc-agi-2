import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.orchestrator import process_task, run_all


class MockCLIImpl:
    def workspace_extras(self, ws_path: Path) -> None:
        pass

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
    ) -> float:
        return 0.0

    def run_session(
        self,
        ws_path: Path,
        model: str,
        initial_prompt: str,
        feedback: str,
        iteration: int,
        session_started: bool,
        task_id: str,
        test_index: int,
        _status_cb: Any,
    ) -> tuple[list[str], int, str, dict, bool]:
        return ([], 0, "", {}, False)

    def extract_grid_from_output(self, raw_lines: list[str]) -> list[list[int]] | None:
        return None

    def parse_stream_json(self, raw_lines: list[str], task_id: str) -> list[dict]:
        return [{"parsed": True, "lines": len(raw_lines)}]

    def write_readable_log(self, rf: Any, line: str, obj: dict) -> None:
        rf.write(f"Parsed readable: {line}\n")


class SequenceBackend:
    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []
        self.current_runs = 0
        self.max_concurrent_runs = 0

    async def setup(self, root_path: Path, cli_type: str) -> None:
        pass

    async def run_agent(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        self.current_runs += 1
        self.max_concurrent_runs = max(self.max_concurrent_runs, self.current_runs)
        try:
            result = self._results.pop(0)
            if isinstance(result, BaseException):
                raise result
            log_dir = kwargs["log_dir"]
            log_dir.mkdir(parents=True, exist_ok=True)
            if "raw_lines" in result and result["raw_lines"]:
                (log_dir / "raw_stream.jsonl").write_text("\n".join(result["raw_lines"]) + "\n")
            return result
        finally:
            self.current_runs -= 1


class QueueTrackingBackend:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []

    async def setup(self, root_path: Path, cli_type: str) -> None:
        pass

    async def run_agent(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs["agent_id"])
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await __import__("asyncio").sleep(0.01)
            test_index = kwargs["test_index"]
            return {
                "task_id": kwargs["task_id"],
                "agent_id": kwargs["agent_id"],
                "test_index": test_index,
                "attempts": [{"test_index": test_index, "grid": [[test_index]]}],
                "elapsed": 0.2,
                "cost": 0.1,
                "backend_cost": 0.2,
                "backend_duration": 0.3,
                "total_cost": 0.3,
                "turns": 1,
                "usage": {
                    "input_tokens": 10,
                    "cached_tokens": 2,
                    "output_tokens": 3,
                },
                "raw_lines": ['{"event": "started"}'],
                "stderr": "",
            }
        finally:
            self.active -= 1


def _make_args(**overrides: Any) -> argparse.Namespace:
    base = {
        "tasks": "task_a,task_b,task_c",
        "backend": "docker",
        "cli": "opencode",
        "model": "test-model",
        "num_agents": 1,
        "max_iterations": 2,
        "soft_training_feedback": False,
        "whole_task": False,
        "resume": None,
        "name": "testrun",
        "limit": None,
        "concurrency": 0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _task_with_n_tests(count: int) -> dict[str, Any]:
    return {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[i]]} for i in range(count)],
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_task_aggregates_multi_agent_multi_test_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _make_args(num_agents=2, whole_task=False)

    backend = SequenceBackend(
        [
            {
                "task_id": "task_x",
                "agent_id": "task_x_ens0_t0",
                "test_index": 0,
                "attempts": [{"test_index": 0, "grid": [[1]]}],
                "elapsed": 1.2,
                "cost": 0.5,
                "backend_cost": 0.2,
                "backend_duration": 1.0,
                "total_cost": 0.7,
                "turns": 2,
                "usage": {
                    "input_tokens": 100,
                    "cached_tokens": 10,
                    "output_tokens": 20,
                },
                "raw_lines": ['{"event": "started"}'],
                "stderr": "",
            },
            {
                "task_id": "task_x",
                "agent_id": "task_x_ens1_t0",
                "test_index": 0,
                "attempts": [{"test_index": 0, "grid": [[2]]}],
                "elapsed": 1.0,
                "cost": 0.3,
                "backend_cost": 0.1,
                "backend_duration": 0.8,
                "total_cost": 0.4,
                "turns": 1,
                "usage": {
                    "input_tokens": 50,
                    "cached_tokens": 5,
                    "output_tokens": 7,
                },
                "raw_lines": ['{"event": "started"}'],
                "stderr": "",
            },
            {
                "task_id": "task_x",
                "agent_id": "task_x_ens0_t1",
                "test_index": 1,
                "attempts": [{"test_index": 1, "grid": [[3]]}],
                "elapsed": 2.5,
                "cost": 0.8,
                "backend_cost": 0.4,
                "backend_duration": 2.1,
                "total_cost": 1.2,
                "turns": 3,
                "usage": {
                    "input_tokens": 70,
                    "cached_tokens": 0,
                    "output_tokens": 9,
                },
                "raw_lines": ['{"event": "started"}'],
                "stderr": "",
            },
            {
                "task_id": "task_x",
                "agent_id": "task_x_ens1_t1",
                "test_index": 1,
                "attempts": [],
                "elapsed": 0.7,
                "cost": 0.1,
                "backend_cost": 0.05,
                "backend_duration": 0.6,
                "total_cost": 0.15,
                "turns": 1,
                "usage": {
                    "input_tokens": 30,
                    "cached_tokens": 3,
                    "output_tokens": 4,
                },
                "raw_lines": ['{"event": "started"}'],
                "stderr": "",
            },
        ]
    )

    with patch("src.orchestrator.load_task_json", return_value=_task_with_n_tests(2)):
        result = await process_task(
            "task_x",
            args,
            run_dir,
            None,
            backend,
            MockCLIImpl(),
        )

    assert result["task_id"] == "task_x"
    assert result["score"]["submitted"] == 2
    assert result["score"]["total"] == 2
    assert result["score"]["api_cost"] == pytest.approx(1.7)
    assert result["score"]["backend_cost"] == pytest.approx(0.75)
    assert result["score"]["total_cost"] == pytest.approx(2.45)
    assert result["score"]["elapsed"] == pytest.approx(2.5)
    assert result["score"]["usage"] == {
        "input_tokens": 250,
        "cached_tokens": 18,
        "output_tokens": 40,
    }

    task_result_path = run_dir / "task_results" / "task_x.json"
    task_result = json.loads(task_result_path.read_text())
    assert set(task_result["agents"]) == {
        "task_x_ens0_t0",
        "task_x_ens1_t0",
        "task_x_ens0_t1",
        "task_x_ens1_t1",
    }
    assert task_result["agents"]["task_x_ens0_t0"]["attempts"] == [[[1]]]
    assert task_result["agents"]["task_x_ens1_t1"]["attempts"] == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_task_whole_task_counts_submissions_from_multiple_test_indexes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _make_args(num_agents=2, whole_task=True)

    backend = SequenceBackend(
        [
            {
                "task_id": "task_whole",
                "agent_id": "task_whole_ens0",
                "test_index": 0,
                "attempts": [
                    {"test_index": 0, "grid": [[1]]},
                    {"test_index": 1, "grid": [[2]]},
                ],
                "elapsed": 1.5,
                "cost": 1.0,
                "backend_cost": 0.5,
                "backend_duration": 1.4,
                "total_cost": 1.5,
                "turns": 2,
                "usage": {
                    "input_tokens": 40,
                    "cached_tokens": 4,
                    "output_tokens": 8,
                },
                "raw_lines": ['{"event": "started"}'],
                "stderr": "",
            },
            {
                "task_id": "task_whole",
                "agent_id": "task_whole_ens1",
                "test_index": 0,
                "attempts": [{"test_index": 1, "grid": [[9]]}],
                "elapsed": 1.0,
                "cost": 0.4,
                "backend_cost": 0.2,
                "backend_duration": 0.9,
                "total_cost": 0.6,
                "turns": 1,
                "usage": {
                    "input_tokens": 10,
                    "cached_tokens": 1,
                    "output_tokens": 2,
                },
                "raw_lines": ['{"event": "started"}'],
                "stderr": "",
            },
        ]
    )

    with patch("src.orchestrator.load_task_json", return_value=_task_with_n_tests(2)):
        result = await process_task(
            "task_whole",
            args,
            run_dir,
            None,
            backend,
            MockCLIImpl(),
        )

    assert result["score"]["submitted"] == 2
    assert result["score"]["total"] == 2

    assert (run_dir / "logs" / "task_whole" / "agent0" / "readable.md").exists()
    assert (run_dir / "logs" / "task_whole" / "agent1" / "readable.md").exists()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_task_retries_empty_results_until_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _make_args(num_agents=1, whole_task=False)

    backend = SequenceBackend(
        [
            {
                "task_id": "task_retry",
                "agent_id": "task_retry_ens0_t0",
                "test_index": 0,
                "attempts": [],
                "elapsed": 0.1,
                "cost": 0.0,
                "backend_cost": 0.0,
                "backend_duration": 0.1,
                "total_cost": 0.0,
                "turns": 0,
                "usage": {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0},
                "raw_lines": [],
                "stderr": "",
            },
            {
                "task_id": "task_retry",
                "agent_id": "task_retry_ens0_t0",
                "test_index": 0,
                "attempts": [],
                "elapsed": 0.1,
                "cost": 0.0,
                "backend_cost": 0.0,
                "backend_duration": 0.1,
                "total_cost": 0.0,
                "turns": 0,
                "usage": {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0},
                "raw_lines": [],
                "stderr": "",
            },
            {
                "task_id": "task_retry",
                "agent_id": "task_retry_ens0_t0",
                "test_index": 0,
                "attempts": [{"test_index": 0, "grid": [[7]]}],
                "elapsed": 0.5,
                "cost": 0.2,
                "backend_cost": 0.1,
                "backend_duration": 0.4,
                "total_cost": 0.3,
                "turns": 1,
                "usage": {"input_tokens": 5, "cached_tokens": 0, "output_tokens": 1},
                "raw_lines": ['{"event": "started"}'],
                "stderr": "",
            },
        ]
    )

    with patch("src.orchestrator.load_task_json", return_value=_task_with_n_tests(1)):
        with patch("src.orchestrator.asyncio.sleep", new_callable=__import__("unittest").mock.AsyncMock) as sleep_mock:
            result = await process_task(
                "task_retry",
                args,
                run_dir,
                None,
                backend,
                MockCLIImpl(),
            )

    assert result["score"]["submitted"] == 1
    assert len(backend.calls) == 3
    assert sleep_mock.await_count == 2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_task_persists_exception_as_agent_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _make_args(num_agents=2, whole_task=False)

    backend = SequenceBackend(
        [
            RuntimeError("backend exploded"),
            {
                "task_id": "task_fail",
                "agent_id": "task_fail_ens1_t0",
                "test_index": 0,
                "attempts": [{"test_index": 0, "grid": [[5]]}],
                "elapsed": 0.9,
                "cost": 0.2,
                "backend_cost": 0.1,
                "backend_duration": 0.7,
                "total_cost": 0.3,
                "turns": 1,
                "usage": {"input_tokens": 11, "cached_tokens": 1, "output_tokens": 2},
                "raw_lines": ['{"event": "started"}'],
                "stderr": "",
            },
        ]
    )

    with patch("src.orchestrator.load_task_json", return_value=_task_with_n_tests(1)):
        result = await process_task(
            "task_fail",
            args,
            run_dir,
            None,
            backend,
            MockCLIImpl(),
        )

    assert result["score"]["submitted"] == 1
    task_result = json.loads((run_dir / "task_results" / "task_fail.json").read_text())
    failed_agent = task_result["agents"]["task_fail_ens0_t0"]
    ok_agent = task_result["agents"]["task_fail_ens1_t0"]

    assert failed_agent["attempts"] == []
    assert "backend exploded" in failed_agent["error"]
    assert ok_agent["attempts"] == [[[5]]]

    error_log = run_dir / "logs" / "task_fail" / "t0" / "agent0" / "error.log"
    assert error_log.exists()
    assert "backend exploded" in error_log.read_text()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_task_respects_backend_queue_concurrency_limit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _make_args(num_agents=3, whole_task=False)
    backend = QueueTrackingBackend()
    queue = __import__("asyncio").Queue()
    queue.put_nowait("slot")

    with patch("src.orchestrator.load_task_json", return_value=_task_with_n_tests(1)):
        result = await process_task(
            "task_queue",
            args,
            run_dir,
            queue,
            backend,
            MockCLIImpl(),
        )

    assert result["score"]["submitted"] == 1
    assert backend.max_active == 1
    assert len(backend.calls) == 3


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_all_skips_completed_tasks_and_writes_summary(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    resume_dir = results_dir / "resume_run"
    task_results_dir = resume_dir / "task_results"
    task_results_dir.mkdir(parents=True)

    (task_results_dir / "task_a.json").write_text(
        json.dumps(
            {
                "score": {
                    "submitted": 1,
                    "total": 1,
                    "elapsed": 0.5,
                    "api_cost": 0.1,
                    "backend_cost": 0.2,
                    "total_cost": 0.3,
                    "usage": {
                        "input_tokens": 1,
                        "cached_tokens": 0,
                        "output_tokens": 1,
                    },
                },
                "agents": {},
            }
        )
    )

    args = _make_args(
        tasks="task_a,task_b,task_c",
        resume=str(resume_dir),
        limit=1,
        concurrency=2,
        whole_task=False,
    )

    backend = SequenceBackend([])
    cli = MockCLIImpl()
    processed: list[str] = []

    async def fake_process_task(
        task_id: str,
        args_obj: Any,
        run_dir: Path,
        backend_queue: Any,
        backend_impl: Any,
        cli_impl: Any,
    ) -> dict[str, Any]:
        processed.append(task_id)
        assert run_dir == resume_dir
        assert backend_queue is not None
        return {
            "task_id": task_id,
            "score": {
                "submitted": 2,
                "total": 2,
                "elapsed": 1.5,
                "api_cost": 1.0,
                "backend_cost": 0.5,
                "total_cost": 1.5,
                "usage": {
                    "input_tokens": 20,
                    "cached_tokens": 2,
                    "output_tokens": 4,
                },
            },
        }

    with patch("src.orchestrator.RESULTS", results_dir):
        with patch("src.orchestrator.get_backend_runner", return_value=backend):
            with patch("src.orchestrator.get_cli_impl", return_value=cli):
                with patch(
                    "src.orchestrator.load_task_ids",
                    return_value=["task_a", "task_b", "task_c"],
                ):
                    with patch("src.orchestrator.setup_logging"):
                        with patch("src.orchestrator.process_task", side_effect=fake_process_task):
                            await run_all(args)

    assert processed == ["task_b"]

    summary = json.loads((resume_dir / "summary.json").read_text())
    assert summary["num_tasks"] == 3
    assert summary["total_tests"] == 3
    assert summary["total_cost"] == pytest.approx(1.8)
    assert summary["tasks"]["task_a"]["submitted"] == 1
    assert summary["tasks"]["task_b"]["submitted"] == 2
    assert "task_c" not in summary["tasks"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_all_continues_after_process_task_crash(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    run_dir = results_dir / "testrun_stamp"

    args = _make_args(
        tasks="task_a,task_b",
        name="testrun",
        resume=None,
        limit=None,
        concurrency=0,
    )

    backend = SequenceBackend([])
    cli = MockCLIImpl()

    async def fake_process_task(
        task_id: str,
        args_obj: Any,
        run_dir_arg: Path,
        backend_queue: Any,
        backend_impl: Any,
        cli_impl: Any,
    ) -> dict[str, Any]:
        if task_id == "task_a":
            raise RuntimeError("boom")
        return {
            "task_id": task_id,
            "score": {
                "submitted": 1,
                "total": 1,
                "elapsed": 0.7,
                "api_cost": 0.3,
                "backend_cost": 0.2,
                "total_cost": 0.5,
                "usage": {
                    "input_tokens": 4,
                    "cached_tokens": 0,
                    "output_tokens": 1,
                },
            },
        }

    with patch("src.orchestrator.RESULTS", results_dir):
        with patch("src.orchestrator.get_backend_runner", return_value=backend):
            with patch("src.orchestrator.get_cli_impl", return_value=cli):
                with patch("src.orchestrator.load_task_ids", return_value=["task_a", "task_b"]):
                    with patch("src.orchestrator.setup_logging"):
                        with patch("src.orchestrator.process_task", side_effect=fake_process_task):
                            with patch("src.orchestrator.random.shuffle", side_effect=lambda items: None):
                                with patch("src.orchestrator.datetime") as mock_datetime:
                                    mock_datetime.now.return_value.strftime.return_value = "stamp"
                                    await run_all(args)

    assert run_dir.exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["num_tasks"] == 2
    assert summary["total_tests"] == 1
    assert summary["total_cost"] == pytest.approx(0.5)
    assert "task_b" in summary["tasks"]
    assert "task_a" not in summary["tasks"]
