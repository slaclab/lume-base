# lume-base

Base classes and architecture for LUME Python projects

## Installation

Install using pip:

```bash
pip install lume-base
```

## Development

We recommend using [uv](https://github.com/astral-sh/uv) for development:

```bash
# Install uv
pip install uv

# Initialize uv project
uv init

# Install package in editable mode with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest
```

or alternatively run pip commands directly in uv:

```bash
# Install package in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```

### Linting and Formatting

This project uses [pre-commit](https://pre-commit.com/) to manage code quality checks, including [ruff](https://github.com/astral-sh/ruff) for linting and formatting. Pre-commit hooks will automatically run on every commit.

```bash
# Install pre-commit (if not already installed from dev dependencies)
uv pip install pre-commit

# Install the git hook scripts
pre-commit install

# (Optional) Run against all files manually
pre-commit run --all-files
```

Once installed, pre-commit will automatically run checks whenever you commit changes. If any checks fail, the commit will be blocked until you fix the issues.

**To manually run checks before committing:**

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Or run only ruff checks
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files
```

The CI pipeline will automatically run these checks on every push and pull request.

### Building Documentation

```bash
# Install with docs dependencies
uv sync --extra docs

# Build docs
uv run mkdocs build

# Serve docs locally
uv run mkdocs serve
```
