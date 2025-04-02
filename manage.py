from argparse import ArgumentParser


def format():
    """Format code using ruff."""
    import subprocess

    try:
        # Run ruff check with fix
        subprocess.run(["ruff", "check", "app", "--fix"], check=True)
        # Run ruff format
        subprocess.run(["ruff", "format", "app"], check=True)
        print("✨ Code formatted successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error formatting code: {e}")


def lint():
    """Run linting and type checking."""
    import subprocess

    try:
        # Run mypy type checking
        subprocess.run(["mypy", "app"], check=True)
        # Run ruff linting
        subprocess.run(["ruff", "check", "app"], check=True)
        # Run ruff format check
        subprocess.run(["ruff", "format", "app", "--check"], check=True)
        print("✨ Code validation successful!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error validating code: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Project management commands")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Format command
    format_parser = subparsers.add_parser("format", help="Format code using ruff")

    # Lint command
    lint_parser = subparsers.add_parser("lint", help="Run linting and type checking")

    # args = parser.parse_args()
    known_args, unknown_args = parser.parse_known_args()

    if known_args.command == "format":
        format()
    elif known_args.command == "lint":
        lint()
