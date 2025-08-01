from argparse import ArgumentParser


def format() -> None:
    """Format code using ruff."""
    import subprocess

    try:
        # Run ruff check with fix
        subprocess.run(["ruff", "check", ".", "--fix"], check=True)
        # Run ruff format
        subprocess.run(["ruff", "format", "."], check=True)
        print("✨ Code formatted successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error formatting code: {e}")


def lint() -> None:
    """Run linting and type checking."""
    import subprocess

    try:
        # Run mypy type checking
        subprocess.run(["mypy", "."], check=True)
        # Run ruff linting
        subprocess.run(["ruff", "check", "."], check=True)
        # Run ruff format check
        subprocess.run(["ruff", "format", ".", "--check"], check=True)
        print("✨ Code validation successful!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error validating code: {e}")


def cli():
    parser = ArgumentParser(description="Project management commands")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Format command
    subparsers.add_parser("format", help="Format code using ruff")

    # Lint command
    subparsers.add_parser("lint", help="Run linting and type checking")

    # args = parser.parse_args()
    known_args, unknown_args = parser.parse_known_args()

    if known_args.command == "format":
        format()
    elif known_args.command == "lint":
        lint()


if __name__ == "__main__":
    cli()
