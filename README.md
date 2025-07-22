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

## 🚧 Troubleshooting

If ui related errors occurs like these in wsl:

```text
UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
```

```text
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in ".venv/lib/python3.12/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb.
```

install this:

```bash
sudo apt update
sudo apt install -y libx11-dev libxext-dev libxrender-dev libqt5x11extras5 libxcb-xinerama0
```
