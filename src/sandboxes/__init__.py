"""Sandbox runners for docker and e2b."""

from src.sandboxes.base import SandboxRunner

__all__ = ["SandboxRunner", "get_sandbox_runner"]


def get_sandbox_runner(sandbox_name: str) -> SandboxRunner:
    """Return a SandboxRunner instance for the given sandbox name."""
    if sandbox_name == "docker":
        from src.sandboxes.docker_runner import DockerRunner

        return DockerRunner()
    if sandbox_name == "e2b":
        from src.sandboxes.e2b_runner import E2BRunner

        return E2BRunner()
    raise ValueError(f"Unknown sandbox name: {sandbox_name}")
