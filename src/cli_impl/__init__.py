"""CLI implementations for open_code and gemini."""

def get_cli_impl(cli_name: str):
    if cli_name == "opencode":
        import src.cli_impl.opencode as impl
        return impl
    elif cli_name == "gemini":
        import src.cli_impl.gemini as impl
        return impl
    else:
        raise ValueError(f"Unknown cli name: {cli_name}")
