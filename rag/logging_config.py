"""
rag/logging_config.py — Persistent File Logging for Motif.

Configures a rotating file handler that writes to ~/.motif/logs/motif.log.
All internal technical logs, tracebacks, and warnings are captured here.
NO log messages reach the terminal via Python's logging system.
"""
from __future__ import annotations

import logging
import logging.handlers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig


_LOG_FILENAME = "motif.log"
_FILE_LOG_LEVEL = logging.DEBUG
_MAX_BYTES = 5_000_000
_BACKUP_COUNT = 3
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_FILE_FMT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"

_configured = False


def setup(config: RAGConfig) -> None:
    """
    Configure the root logger with a rotating file handler.
    All terminal console handlers are purged to prevent developer log leakages.

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
    root.setLevel(logging.DEBUG)

    # ── Remove all existing handlers (console/stream handlers) ───────────────
    for handler in root.handlers[:]:
        root.removeHandler(handler)

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

    _configured = True
    logging.getLogger(__name__).debug(
        "Logging initialised. Log file: %s", log_path
    )
