"""Backend runners for docker and e2b."""

from src.backends.base import BackendRunner

__all__ = ["BackendRunner", "get_backend_runner"]


def get_backend_runner(backend_name: str) -> BackendRunner:
    """Return a BackendRunner instance for the given backend name."""
    if backend_name == "docker":
        from src.backends.docker_runner import DockerRunner

        return DockerRunner()
    if backend_name == "e2b":
        from src.backends.e2b_runner import E2BRunner

        return E2BRunner()
    raise ValueError(f"Unknown backend name: {backend_name}")
