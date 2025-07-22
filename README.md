# AI Workspace

Custom repo to work on ai related things

## 🚀 Quick Start

Install Dependencies with [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

Activate virtual environment

 ```bash
source .venv/bin/activate
# or, if you are using windows
.venv\Scripts\activate
```

### 🏗️ Development

Run `manage --help` to see available commands e.g. `manage lint` or `manage format`.

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

## 🛠️ Project Structure

### 🖊️ Naming Convention

* snake_case: used for Python files, functions, and variables.
* PascalCase: used for classes and exceptions.
* UPPER_CASE: used for constants.
* _snake_case: used for private variables and functions.
* `__snake_case__`: used for special variables and methods (dunder methods).

### 💡 Tag Convention

Todos and Warnings will be specified respectively inside code with `<ToDo>`, `<Warning>` tags and inside README.md file with `ToDos`, `Warnings` sections.

## Troubleshooting

If this error occurs:

```text
UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
```

install this:

```bash
sudo apt install python3-tk
```
