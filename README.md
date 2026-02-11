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

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and code formatting. Before committing your changes, please ensure your code passes all linting checks:

```bash
# Install ruff (if not already installed from dev dependencies)
uv pip install ruff

# Check for linting issues
ruff check .

# Automatically fix linting issues where possible
ruff check . --fix

# Check code formatting
ruff format --check .

# Format code
ruff format .
```

**Before pushing your changes**, run:

```bash
# Fix linting issues and format code
ruff check . --fix && ruff format .
```

The CI pipeline will automatically run these checks on every push and pull request. Your code must pass both `ruff check` and `ruff format --check` to merge.

### Building Documentation

```bash
# Install with docs dependencies
uv sync --extra docs

# Build docs
uv run mkdocs build

# Serve docs locally
uv run mkdocs serve
```

