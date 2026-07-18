"""
cli.py — Development shim for running Motif directly from the project root.

This file is for convenience during development:
    python cli.py          → launches the interactive REPL
    python cli.py ask "q"  → one-shot query

When installed via `uv tool install motif-rag` or pip, the `motif` command
calls rag.cli:main directly. This shim is not part of the installed package.
"""
from rag.cli import main

if __name__ == "__main__":
    main()
