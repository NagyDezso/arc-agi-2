import asyncio
import json
import logging
import time
from pathlib import Path

from e2b import ALL_TRAFFIC, AsyncSandbox, SandboxNetworkOpts, Stderr, Stdout

from src.backends.base import BackendRunner
from src.models import AgentConfig, AgentResultData
from src.orchestrator import ROOT, SESSION_LOG_FILENAME, TRANSCRIPT_FILENAME

logger = logging.getLogger(__name__)

E2B_CPU_COUNT = 2
E2B_COST_PER_VCPU_HOUR = 0.05


class E2BRunner(BackendRunner):
    def setup(self, root_path: Path, cli_type: str) -> None:
        pass

    async def start_agent_backend(
        self,
        config: AgentConfig,
    ) -> AgentResultData:
        network: SandboxNetworkOpts = {
            "deny_out": [ALL_TRAFFIC],
            "allow_out": [
                "generativelanguage.googleapis.com",
                "api.github.com",
                "opencode.ai",
            ],
        }

        config.log_dir.mkdir(parents=True, exist_ok=True)
        session_log_path = config.log_dir / SESSION_LOG_FILENAME
        transcript_path = config.log_dir / TRANSCRIPT_FILENAME

        sandbox = None
        sandbox_start = time.time()
        for attempt in range(5):
            try:
                sandbox = await AsyncSandbox.create(
                    template="arc-solver",
                    envs=config.envs,
                    network=network,
                    timeout=43500,
                )
                sandbox_start = time.time()
                break
            except Exception as e:
                if attempt == 4:
                    logger.error(f"  [e2b] {config.agent_id}: sandbox create failed permanently: {e}")
                    raise
                wait = 2**attempt * 5
                logger.warning(
                    f"  [e2b] {config.agent_id}: sandbox create failed (attempt {attempt + 1}/5), retrying in {wait}s: {e}"
                )
                await asyncio.sleep(wait)

        if sandbox is None:
            msg = f"E2B sandbox creation failed for {config.agent_id}"
            raise RuntimeError(msg)

        try:
            await sandbox.files.write("/root/config.json", json.dumps(config))
            await sandbox.files.make_dir("/app/src")
            await sandbox.files.make_dir("/app/src/cli_impl")
            await sandbox.files.write("/app/src/__init__.py", (ROOT / "__init__.py").read_text())
            await sandbox.files.write("/app/src/agent_runner.py", (ROOT / "agent_runner.py").read_text())
            await sandbox.files.write("/app/src/models.py", (ROOT / "models.py").read_text())
            for f in (ROOT / "cli_impl").glob("*.py"):
                await sandbox.files.write(f"/app/src/cli_impl/{f.name}", f.read_text())

            session_f = session_log_path.open("a", encoding="utf-8")
            transcript_f = transcript_path.open("a", encoding="utf-8")
            stdout_buffer = ""

            def on_stdout(output: Stdout) -> None:
                nonlocal stdout_buffer
                stdout_buffer += str(output)
                while "\n" in stdout_buffer:
                    line, stdout_buffer = stdout_buffer.split("\n", 1)
                    line = line.rstrip("\r")
                    if not line:
                        continue
                    self._route_agent_output_line(line, session_f, transcript_f)

            def on_stderr(output: Stderr) -> None:
                if output.strip():
                    logger.error(output[:200])

            try:
                await sandbox.commands.run(
                    "PYTHONPATH=/app python3 /app/src/agent_runner.py",
                    user="root",
                    timeout=43200 + 120,
                    on_stdout=on_stdout,
                    on_stderr=on_stderr,
                )
            finally:
                if stdout_buffer.strip():
                    self._route_agent_output_line(stdout_buffer.rstrip("\r"), session_f, transcript_f)
                session_f.close()
                transcript_f.close()

            results_content = await sandbox.files.read("/workspace/results.json")
            result = AgentResultData.model_validate_json(results_content)

            sandbox_duration = time.time() - sandbox_start
            e2b_cost = (sandbox_duration / 3600) * E2B_CPU_COUNT * E2B_COST_PER_VCPU_HOUR

            result.backend_cost = e2b_cost
            result.backend_duration = sandbox_duration

            logger.info(
                f"[e2b-cost] API=${result.cost:.4f}, "
                f"E2B=${e2b_cost:.4f}, Total=${result.cost + e2b_cost:.4f}, "
                f"Duration={sandbox_duration:.1f}s"
            )
            return result

        except Exception as e:
            err_msg = f"E2B sandbox error: {e}"
            logger.error(f"[e2b-error] {err_msg}", exc_info=True)
            sandbox_duration = time.time() - sandbox_start
            e2b_cost = (sandbox_duration / 3600) * E2B_CPU_COUNT * E2B_COST_PER_VCPU_HOUR
            return AgentResultData(
                task_id=config.task_id,
                agent_id=config.agent_id,
                test_index=config.test_index,
                error=err_msg,
                backend_cost=e2b_cost,
                backend_duration=sandbox_duration,
            )
        finally:
            if sandbox:
                await sandbox.kill()
