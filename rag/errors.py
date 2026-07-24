"""
rag/errors.py — Centralized Human-Friendly Error Translation.

Translates technical exceptions into concise, user-facing notifications.
Raw exceptions, tracebacks, and library details are logged to ~/.motif/logs/motif.log.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def humanize_error(exc: Exception | str | None) -> str:
    """
    Convert a raw python exception or error string into a clean, concise,
    actionable one-line message suitable for display in the terminal.
    """
    if exc is None:
        return "An unknown error occurred."

    err_str = str(exc).strip()
    err_lower = err_str.lower()

    # Search index / database state issues
    if "qdrantlocal instance is closed" in err_lower or "qdrant" in err_lower and "closed" in err_lower:
        return "Search index connection closed. Please restart Motif."
    if "database is locked" in err_lower or "sqlite3.operationalerror" in err_lower:
        return "Storage database is currently busy. Please try again."

    # Model / Environment files
    if isinstance(exc, FileNotFoundError) or "file not found" in err_lower or "no such file" in err_lower:
        if "model" in err_lower or "gguf" in err_lower or "onnx" in err_lower:
            return "Model files not found. Run 'motif setup' to verify installation."
        return "Required file could not be found."

    # Memory / Hardware issues
    if "cuda" in err_lower or "out of memory" in err_lower or "oom" in err_lower:
        return "Insufficient GPU/System memory for this operation."

    # Permissions
    if isinstance(exc, PermissionError) or "permission denied" in err_lower:
        return "Permission denied accessing local storage or document path."

    # Fallback: single concise line without exception class names or tracebacks
    first_line = err_str.split("\n")[0]
    # Clean up standard Exception prefixes like "RuntimeError: "
    if ":" in first_line and any(first_line.startswith(cls_name) for cls_name in ["RuntimeError", "ValueError", "TypeError", "AttributeError"]):
        first_line = first_line.split(":", 1)[1].strip()

    if not first_line:
        return "Operation failed. Check ~/.motif/logs/motif.log for details."

    return first_line
