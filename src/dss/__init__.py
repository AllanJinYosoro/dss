"""
Command-line entry point for the appointment DSS simulation.
"""

from .cli import app


def main() -> None:
    # Delegate to Typer app so `uv run dss ...` works.
    app()
