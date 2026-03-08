import asyncio
import json
import logging
import time
from pathlib import Path

from e2b import ALL_TRAFFIC, AsyncSandbox, Stdout, Stderr, SandboxNetworkOpts

from src.backends.base import BackendRunner
from src.models import AgentRunSpec

logger = logging.getLogger(__name__)

E2B_CPU_COUNT = 2
E2B_COST_PER_VCPU_HOUR = 0.05


class E2BRunner(BackendRunner):
    def setup(self, root_path: Path, cli_type: str) -> None:
        pass

    async def run_agent(
        self,
        spec: AgentRunSpec,
    ) -> dict:

        config = {
            "task_id": spec.task_id,
            "agent_id": spec.agent_id,
            "raw_task": spec.raw_task,
            "test_index": spec.test_index,
            "model": spec.model,
            "max_iterations": spec.max_iterations,
            "soft_training_feedback": spec.soft_training_feedback,
            "whole_task": spec.whole_task,
            "cli_type": spec.cli_type,
        }

        network: SandboxNetworkOpts = {
            "deny_out": [ALL_TRAFFIC],
            "allow_out": [
                "generativelanguage.googleapis.com",
                "api.github.com",
                "opencode.ai",
            ],
        }

        spec.log_dir.mkdir(parents=True, exist_ok=True)
        raw_stream_path = spec.log_dir / "raw_stream.jsonl"

        sandbox = None
        sandbox_start = time.time()
        for attempt in range(5):
            try:
                sandbox = await AsyncSandbox.create(
                    template="arc-solver",
                    envs=spec.envs,
                    network=network,
                    timeout=43500,
                )
                sandbox_start = time.time()
                break
            except Exception as e:
                if attempt == 4:
                    logger.error(f"  [e2b] {spec.agent_id}: sandbox create failed permanently: {e}")
                    raise
                wait = 2**attempt * 5
                logger.warning(
                    f"  [e2b] {spec.agent_id}: sandbox create failed (attempt {attempt + 1}/5), retrying in {wait}s: {e}"
                )
                await asyncio.sleep(wait)

        if sandbox is None:
            msg = f"E2B sandbox creation failed for {spec.agent_id}"
            raise RuntimeError(msg)

        try:
            await sandbox.files.write("/root/config.json", json.dumps(config))
            await sandbox.files.write("/app/agent_runner.py", (spec.root_path / "agent_runner.py").read_text())
            await sandbox.files.make_dir("/app/cli_impl")
            for f in (spec.root_path / "cli_impl").glob("*.py"):
                await sandbox.files.write(f"/app/cli_impl/{f.name}", f.read_text())

            raw_f = raw_stream_path.open("a")

            def on_stdout(output: Stdout) -> None:
                if output.strip():
                    logger.info(f"[agent] {spec.agent_id}: {output}")
                    raw_f.write(output + "\n")
                    raw_f.flush()

            def on_stderr(output: Stderr) -> None:
                if output.strip():
                    logger.error(f"  [e2b-stderr] {spec.agent_id}: {output[:200]}")

            try:
                await sandbox.commands.run(
                    "python3 /app/agent_runner.py",
                    user="root",
                    timeout=43200 + 120,
                    on_stdout=on_stdout,
                    on_stderr=on_stderr,
                )
            finally:
                raw_f.close()

            results_content = await sandbox.files.read("/workspace/results.json")
            result = json.loads(results_content)

            sandbox_duration = time.time() - sandbox_start
            e2b_cost = (sandbox_duration / 3600) * E2B_CPU_COUNT * E2B_COST_PER_VCPU_HOUR
            result["backend_cost"] = e2b_cost
            result["backend_duration"] = sandbox_duration
            result["total_cost"] = result.get("cost", 0) + e2b_cost

            logger.info(
                f"  [e2b-cost] {spec.agent_id}: API=${result.get('cost', 0):.4f}, "
                f"E2B=${e2b_cost:.4f}, Total=${result['total_cost']:.4f}, "
                f"Duration={sandbox_duration:.1f}s"
            )
            return result

        except Exception as e:
            err_msg = f"E2B sandbox error: {e}"
            logger.error(f"  [e2b-error] {spec.agent_id}: {err_msg}", exc_info=True)
            sandbox_duration = time.time() - sandbox_start
            e2b_cost = (sandbox_duration / 3600) * E2B_CPU_COUNT * E2B_COST_PER_VCPU_HOUR
            return {
                "task_id": spec.task_id,
                "agent_id": spec.agent_id,
                "test_index": spec.test_index,
                "attempts": [],
                "elapsed": 0,
                "cost": 0,
                "backend_cost": e2b_cost,
                "backend_duration": sandbox_duration,
                "total_cost": e2b_cost,
                "turns": 0,
                "error": err_msg,
                "raw_lines": [],
                "stderr": "",
                "usage": {},
            }
        finally:
            if sandbox:
                await sandbox.kill()
