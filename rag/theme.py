"""
rag/theme.py — Centralized UI Theme for Motif

Defines the semantic color palette and pre-configures a rich Console.
Colors map to a modern, minimalist premium aesthetic using Motif Indigo.
"""
from rich.console import Console
from rich.theme import Theme

# Hex color definitions (Cyber Sunset Palette)
MOTIF_PURPLE = "#8A2BE2"  # Radiant Violet
MOTIF_GOLD = "#FFB800"    # Solar Gold
MOTIF_CYAN = "#00F0FF"    # Electric Cyan
MOTIF_GREEN = "#00FF87"   # Neon Emerald
STRUCTURAL_GRAY = "#6b7280"
WARNING_YELLOW = "#f59e0b"
ERROR_RED = "#ef4444"

# Define semantic theme mapping
motif_theme = Theme({
    "accent": MOTIF_PURPLE,
    "accent_bold": f"bold {MOTIF_PURPLE}",
    "brand_gold": f"bold {MOTIF_GOLD}",
    "brand_cyan": f"bold {MOTIF_CYAN}",
    "brand_green": f"bold {MOTIF_GREEN}",
    "structure": STRUCTURAL_GRAY,
    "success": MOTIF_GREEN,
    "warning": WARNING_YELLOW,
    "error": ERROR_RED,
    "dim": "dim", # Fallback for pure dimming without explicit gray

    # Markdown specific overrides to remove rainbow colors
    "markdown.code": "default on #2a2a2a",
    "markdown.code_block": "default on #1e1e1e",
    "markdown.block_quote": STRUCTURAL_GRAY,
    "markdown.list": "default",
    "markdown.item.number": "default",
    "markdown.h1": f"bold {MOTIF_PURPLE}",
    "markdown.h2": "bold default",
    "markdown.h3": "bold default",
    "markdown.h4": "bold default",
    "markdown.h5": "bold default",
    "markdown.link": f"underline {MOTIF_CYAN}",
    "markdown.link_url": f"dim underline {STRUCTURAL_GRAY}",
    "markdown.table.border": STRUCTURAL_GRAY,
    "markdown.table.header": "bold default",
})

# A global console instance to be imported across the application
console = Console(theme=motif_theme)
