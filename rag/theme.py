"""
rag/theme.py — Centralized UI Theme for Motif

Defines the semantic color palette and pre-configures a rich Console.
Colors map to a modern, minimalist premium aesthetic using Motif Pink (#FF2E93).
"""
from rich.console import Console
from rich.theme import Theme

# Hex color definitions (Motif Primary Palette)
MOTIF_PRIMARY = "#FF2E93"   # Neon Pink
SUCCESS_GREEN = "#22c55e"   # Green
STRUCTURAL_GRAY = "#6b7280"
WARNING_YELLOW = "#f59e0b"
ERROR_RED = "#ef4444"

# Define semantic theme mapping
motif_theme = Theme({
    "accent": MOTIF_PRIMARY,
    "accent_bold": f"bold {MOTIF_PRIMARY}",
    "brand_green": f"bold {SUCCESS_GREEN}",
    "structure": STRUCTURAL_GRAY,
    "subtle": STRUCTURAL_GRAY,
    "success": SUCCESS_GREEN,
    "warning": WARNING_YELLOW,
    "error": ERROR_RED,
    "muted": "dim default",
    "muted_italic": "dim italic default",
    "separator": f"dim {STRUCTURAL_GRAY}",
    "citation": "dim default",
    "excerpt": f"dim italic {STRUCTURAL_GRAY}",

    # Markdown specific overrides to remove rainbow colors
    "markdown.code": "default on #2a2a2a",
    "markdown.code_block": "default on #1e1e1e",
    "markdown.block_quote": STRUCTURAL_GRAY,
    "markdown.list": "default",
    "markdown.item.number": "default",
    "markdown.h1": f"bold {MOTIF_PRIMARY}",
    "markdown.h2": "bold default",
    "markdown.h3": "bold default",
    "markdown.h4": "bold default",
    "markdown.h5": "bold default",
    "markdown.link": f"underline {MOTIF_PRIMARY}",
    "markdown.link_url": f"dim underline {STRUCTURAL_GRAY}",
    "markdown.table.border": STRUCTURAL_GRAY,
    "markdown.table.header": "bold default",
})

# A global console instance to be imported across the application
console = Console(theme=motif_theme)
