"""Backend runners for docker and e2b."""

def get_backend_runner(backend_name: str):
    if backend_name == "docker":
        import src.backends.docker_runner as runner
        return runner
    elif backend_name == "e2b":
        import src.backends.e2b_runner as runner
        return runner
    else:
        raise ValueError(f"Unknown backend name: {backend_name}")
