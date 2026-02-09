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

### Building Documentation

```bash
# Install with docs dependencies
uv sync --extra docs

# Build docs
uv run mkdocs build

# Serve docs locally
uv run mkdocs serve
```

