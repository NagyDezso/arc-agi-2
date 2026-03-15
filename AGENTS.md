# Testing

You can run tests with: uv run pytest

- **unit/** tests: single function/class in isolation, fast, no side effects
- **functional/** tests: component workflows with mocked dependencies, fast
- **integration/** tests: real external dependencies (Docker, APIs), have side effects
