import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import TextIO

from src.models import UsageTotals

from .base import BaseCLI, capture_raw_output_line, find_last_grid

_IGNORED_STDERR_SUBSTRINGS = (
    "Performing one time database migration, may take a few minutes...\n",
    "sqlite-migration:done\n",
    "Database migration complete.\n",
)


class OpenCodeCLI(BaseCLI):
    def __init__(self) -> None:
        self.PRICING = {
            "kilo/minimax/minimax-m2.5": (0.29, 1.20, 0.00),
            "kilo/minimax/minimax-m2.5:free": (0.29, 1.20, 0.00),
        }

    def workspace_extras(self) -> None:
        auth_path = Path("/root/.local/share/opencode/auth.json")
        config_path = Path("/root/.config/opencode/opencode.json")
        auth_path.mkdir(parents=True, exist_ok=True)
        auth_path.write_text(json.dumps({
            "github-copilot": {
                "type": "oauth",
                "access": "",
                "refresh": os.environ.get("GITHUB_TOKEN", ""),
                "expires": 0,
            }
        }), encoding="utf-8")
        config = {
            "$schema": "https://opencode.ai/config.json",
            "agent": {
                "arc_solver": {
                    "prompt": "You are participating in a puzzle solving competition. You are an expert at solving puzzles.",
                    "tools": {
                        "webfetch": False,
                        "skill": False,
                        "task": False,
                        "todowrite": False,
                        "question": False,
                        "grep": False,
                        "glob": False,
                        "bash": True,
                        "read": True,
                        "write": True,
                        "edit": True,
                    },
                }
            },
            "provider": {
                "lmstudio": {
                    "models": {
                        "qwen3.5:2b": {"name": "qwen3.5:2b"},
                        "qwen3.5:0.8b": {"name": "qwen3.5:0.8b"},
                        "qwen3.5:27b": {
                            "name": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2",
                            "limit": {
                                "context": 400000,
                                "output": 120000,
                            },
                        },
                    },
                    "name": "LMStudio (local)",
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {"baseURL": "host.docker.internal:4444/v1"},
                }
            },
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def run_session(
        self, ws_path: Path, model: str, initial_prompt: str, feedback: str, iteration: int
    ) -> tuple[list[str], int, str, UsageTotals]:
        # Resolve the opencode executable path not the same on Windows and Unix
        base_path = shutil.which("opencode")
        cmd = [base_path, "run", "--format", "json", "--agent", "arc_solver"]
        if iteration == 0:
            cmd.extend(
                [
                    "--model",
                    model,
                    initial_prompt,
                ]
            )
        else:
            cmd.extend(["--continue", feedback])

        proc = subprocess.Popen(
            cmd,
            cwd=str(ws_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise ValueError("Failed to open stdin, stdout or stderr")
        proc.stdin.close()
        raw_lines = []
        num_turns = 0
        token_stats = UsageTotals()
        for line in proc.stdout or []:
            obj = capture_raw_output_line(raw_lines, line)
            if obj is None:
                continue

            evt_type = obj.get("type", "")
            if evt_type == "step_start":
                num_turns += 1
            elif evt_type == "step_finish":
                part = obj.get("part", {})
                tokens = part.get("tokens", {})
                token_stats.input_tokens += tokens.get("input", 0)
                token_stats.cached_tokens += tokens.get("cache", {}).get("read", 0)
                token_stats.output_tokens += tokens.get("output", 0)

        stderr_text = proc.stderr.read()
        for ignored in _IGNORED_STDERR_SUBSTRINGS:
            stderr_text = stderr_text.replace(ignored, "")
        proc.wait(timeout=3500)

        return raw_lines, num_turns, stderr_text, token_stats

    def extract_grid_from_output(self, raw_lines: list[str]) -> list[list[int]] | None:
        all_text = ""
        for line in raw_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt_type = obj.get("type", "")
            if evt_type == "tool_use":
                part = obj.get("part", {})
                tool = part.get("tool", "")
                state = part.get("state", {})
                if tool.lower() in ("write", "edit"):
                    inp = state.get("input", {})
                    fpath = inp.get("filePath", inp.get("file_path", "")).lower()
                    if not fpath.endswith(".py") and any(
                        kw in fpath
                        for kw in (
                            "output",
                            "answer",
                            "result",
                            "solution",
                            "submission",
                        )
                    ):
                        content = inp.get("content", "")
                        all_text += content + "\n"
                elif tool.lower() == "bash":
                    output = state.get("output", "")
                    if output:
                        all_text += output + "\n"
            elif evt_type == "text":
                part = obj.get("part", {})
                text = part.get("text", "")
                if text:
                    all_text += text + "\n"
        return find_last_grid(all_text)

    def map_tool_params(self, tool_name: str, params: dict) -> dict:
        tool_lower = tool_name.lower()
        if tool_lower == "bash":
            return {
                "command": params.get("command", ""),
                "description": params.get("description", ""),
            }
        if tool_lower == "read":
            return {"file_path": params.get("file_path", "")}
        if tool_lower in ("write", "edit"):
            return {
                "file_path": params.get("file_path", ""),
                "content": params.get("content", ""),
            }
        if tool_lower == "glob":
            return {"pattern": params.get("pattern", "")}
        if tool_lower == "grep":
            return {
                "pattern": params.get("pattern", ""),
                "path": params.get("path", ""),
            }
        return params

    def write_readable_log(self, rf: TextIO, obj: dict):
        evt_type = obj.get("type", "")
        part = obj.get("part", {})
        if evt_type == "text":
            text = part.get("text", "")
            if text.strip():
                rf.write(f"\n**Assistant:**\n{text.strip()}\n\n")
        elif evt_type == "tool_use":
            tool_name = part.get("tool", "")
            state = part.get("state", {})
            inp = state.get("input", {})
            output = state.get("output", "")
            if tool_name.lower() == "bash":
                rf.write(f"\n\n**Tool: {tool_name}**\n```\n$ {inp.get('command', '')}\n```\n\n")
            else:
                input_str = json.dumps(inp, indent=2)[:500]
                rf.write(f"\n\n**Tool: {tool_name}**\n```\n{input_str}\n```\n\n")
            if output:
                truncated = output[:2000] if len(output) > 2000 else output
                rf.write(f"**Tool Result:**\n```\n{truncated}\n```\n\n")
        elif evt_type == "step_finish":
            tokens = part.get("tokens", {})
            rf.write(
                f"---\n**Step:** tokens={tokens.get('input', 0) + tokens.get('output', 0)}, reason={part.get('reason', '?')}\n"
            )
        elif evt_type == "harness_feedback":
            nxt = obj.get("for_iteration", "?")
            body = obj.get("text", "")
            hdr = f"\n\n**Harness feedback** (next session iteration {nxt}):\n```\n"
            rf.write(f"{hdr}{body}\n```\n\n")
