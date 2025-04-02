# AI Workspace

Custom repo to work on ai related things

## Requirements

* [uv](https://docs.astral.sh/uv/) for Python package and environment management.

## General Workflow

By default, the dependencies are managed with [uv](https://docs.astral.sh/uv/), go there and install it.

From root, you can install all the dependencies with:

```bash
uv sync
```

Then you can activate the virtual environment with:

```bash
source .venv/bin/activate
```

Make sure your editor is using the correct Python virtual environment, with the interpreter at `.venv/bin/python`.

* You can run utility scripts from root e.g. `python manage.py format` or `python manage.py lint`

To format code:

```bash
# Run ruff check with fix
ruff check . --fix
# Run ruff format
ruff format .
```

To lint code:

```bash
# Run mypy type checking
mypy .
# Run ruff linting
ruff check .
# Run ruff format check
ruff format . --check
```

## Tag Convention

Todos and Warnings will be specified respectively inside code with `<ToDo>`, `<Warning>` tags and inside README.md files with `ToDos`, `Warnings` sections.
