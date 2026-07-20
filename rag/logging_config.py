"""
rag/logging_config.py — Persistent file logging for Motif.

Sets up a rotating file handler that writes to ~/.ragdb/motif.log.
Called once at startup in rag/cli.py.

Format:
    2026-07-18 16:00:01 INFO     rag.pipeline — Query complete: 4120 ms ...
    2026-07-18 16:00:01 WARNING  rag.retrieval.expander — HyDE failed: ...

Log levels:
    - File: DEBUG (captures all pipeline internals for diagnosis)
    - Console: WARNING (only user-relevant warnings appear in the terminal)

Rotation policy:
    - maxBytes=5_000_000 (5 MB per file)
    - backupCount=3      (up to ~15 MB total log history)
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig


_LOG_FILENAME = "motif.log"
_FILE_LOG_LEVEL = logging.DEBUG
_CONSOLE_LOG_LEVEL = logging.WARNING
_MAX_BYTES = 5_000_000
_BACKUP_COUNT = 3
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_FILE_FMT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
_CONSOLE_FMT = "%(levelname)s %(name)s — %(message)s"

# Avoid adding handlers multiple times if logging_config.setup() is called
# more than once (e.g., during tests).
_configured = False


def setup(config: "RAGConfig") -> None:
    """
    Configure the root logger with a rotating file handler.

    Must be called once at startup before any log messages are emitted.
    Idempotent — safe to call multiple times.

    Args:
        config: RAGConfig — reads storage.db_path for the log file location.
    """
    global _configured
    if _configured:
        return

    from rag.config import get_app_dir
    log_dir = get_app_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / _LOG_FILENAME

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # let handlers filter

    # ── Rotating file handler ─────────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(_FILE_LOG_LEVEL)
    file_handler.setFormatter(
        logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT)
    )
    root.addHandler(file_handler)

    # ── Console handler (warnings only — don't pollute the REPL output) ───────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(_CONSOLE_LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FMT))
    root.addHandler(console_handler)

    _configured = True
    logging.getLogger(__name__).debug(
        "Logging initialised. Log file: %s", log_path
    )
