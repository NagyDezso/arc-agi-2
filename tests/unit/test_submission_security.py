"""Tests for submission.py's transcript suspicious-activity scanner."""

from pathlib import Path

from submission import check_transcripts


def _write_transcript(results_dir: Path, task_id: str, content: str) -> None:
    p = results_dir / "logs" / task_id / "t0" / "agent0" / "transcript.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_flags_web_search_tool_call(tmp_path):
    _write_transcript(
        tmp_path,
        "task_a",
        '{"type":"PLANNER_RESPONSE","tool_calls":[{"name":"search_web",'
        '"args":{"query":"fchollet/ARC 7,7,7"}}]}\n',
    )
    warnings = check_transcripts(tmp_path)
    assert len(warnings) == 1
    assert "search_web" in warnings[0]


def test_flags_grounded_search_result_domain(tmp_path):
    _write_transcript(
        tmp_path,
        "task_b",
        '{"content":"https://vertexaisearch.cloud.google.com/grounding-api-redirect/X"}\n',
    )
    warnings = check_transcripts(tmp_path)
    assert len(warnings) == 1
    assert "vertexaisearch" in warnings[0]


def test_does_not_flag_tool_permission_allowlist(tmp_path):
    # A tool permission allowlist (e.g. "- read_url(*): allowed") must not be
    # flagged — only actual invocations are.
    _write_transcript(
        tmp_path,
        "task_clean",
        '{"content":"Permissions:\\n- command(*): allowed\\n'
        '- read_url(*): allowed\\n- search_web(*): allowed\\n"}\n',
    )
    assert check_transcripts(tmp_path) == []


def test_clean_run_produces_no_warnings(tmp_path):
    _write_transcript(
        tmp_path,
        "task_clean",
        '{"type":"PLANNER_RESPONSE","tool_calls":[{"name":"write_to_file",'
        '"args":{"path":"transform.py"}}]}\n',
    )
    assert check_transcripts(tmp_path) == []


def test_env_inspection_regexes_match():
    def _flagged(content: str) -> bool:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_transcript(root, "t", content)
            return bool(check_transcripts(root))

    assert _flagged('{"content":"x = os.environ[\\"GEMINI_API_KEY\\"]"}\n')
    assert _flagged('{"content":"cat /proc/4321/environ"}\n')
    assert _flagged('{"content":"cat /proc/self/environ"}\n')
