"""
rag/config.py — Configuration dataclasses and hardware tier detection.

Loads config.toml from the project root (or a specified path).
Falls back to config.template.toml if config.toml does not exist.
"""
from __future__ import annotations

import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HardwareConfig:
    tier: str = "auto"  # "auto" | "T1" | "T2" | "T3"


@dataclass
class ModelsConfig:
    llm_path: str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    embed_model: str = "models/nomic-embed-text-v1.5"
    reranker: str = "models/MiniLM-L12-v2"
    whisper: str = "models/ggml-tiny-q5_1.bin"


@dataclass
class LLMConfig:
    n_gpu_layers: int = 0
    ctx_size: int = 2048
    max_tokens: int = 400
    temperature: float = 0.1
    threads: int = 4


@dataclass
class RetrievalConfig:
    top_k_retrieval: int = 20
    top_k_rerank: int = 3
    relevance_threshold: float = 0.3
    query_expansion: str = "none"   # "none" | "hyde"
    bm25_backend: str = "rank_bm25"


@dataclass
class ChunkingConfig:
    target_tokens: int = 512
    overlap_tokens: int = 64
    use_semantic: bool = False
    semantic_threshold: float = 0.3


@dataclass
class GenerationConfig:
    context_max_tokens: int = 2048
    streaming: bool = True
    history_turns: int = 3


@dataclass
class StorageConfig:
    db_path: str = "~/.ragdb"
    query_cache_enabled: bool = False


@dataclass
class ParsersConfig:
    use_moondream: bool = False
    image_density_threshold: float = 0.3


@dataclass
class RAGConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    parsers: ParsersConfig = field(default_factory=ParsersConfig)

    # Resolved after load — set by detect_hardware_tier()
    resolved_tier: str = "T1"

    @property
    def db_root(self) -> Path:
        """Expanded, absolute path to the database root directory."""
        return Path(self.storage.db_path).expanduser().resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Hardware Tier Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_hardware_tier() -> str:
    """
    Determine the hardware tier based on available GPU VRAM.

    Returns:
        "T1" — CPU only, or GPU with < 4 GB VRAM
        "T2" — GPU with 4–5.9 GB VRAM (GTX 1650 class)
        "T3" — GPU with >= 6 GB VRAM (RTX 3050 class and above)
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return "T1"

        lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        if not lines:
            return "T1"

        # Use the first GPU's VRAM (MiB)
        vram_mb = int(lines[0])
        if vram_mb >= 6000:
            return "T3"
        elif vram_mb >= 3800:
            return "T2"
        else:
            return "T1"

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return "T1"


# ─────────────────────────────────────────────────────────────────────────────
# Tier-specific defaults
# ─────────────────────────────────────────────────────────────────────────────

_TIER_DEFAULTS: dict[str, dict] = {
    "T1": {
        "llm": {"n_gpu_layers": 0, "ctx_size": 2048, "threads": 4},
        "retrieval": {"top_k_retrieval": 20, "top_k_rerank": 3, "query_expansion": "none"},
        "chunking": {"use_semantic": False},
        "generation": {"context_max_tokens": 2048},
        "models": {"llm_path": "models/Phi-3.5-mini-instruct-Q4_K_M.gguf", "reranker": "models/MiniLM-L12-v2"},
    },
    "T2": {
        "llm": {"n_gpu_layers": 20, "ctx_size": 3072, "threads": 6},
        "retrieval": {"top_k_retrieval": 25, "top_k_rerank": 5, "query_expansion": "hyde"},
        "chunking": {"use_semantic": True},
        "generation": {"context_max_tokens": 2048},
        "models": {"llm_path": "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf", "reranker": "models/MiniLM-L12-v2"},
    },
    "T3": {
        "llm": {"n_gpu_layers": 28, "ctx_size": 4096, "threads": 8},
        "retrieval": {"top_k_retrieval": 30, "top_k_rerank": 5, "query_expansion": "hyde"},
        "chunking": {"use_semantic": True},
        "generation": {"context_max_tokens": 3072},
        "models": {"llm_path": "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf", "reranker": "models/bge-reranker-base"},
    },
}


def _apply_tier_defaults(config: RAGConfig, tier: str) -> None:
    """Apply tier-specific defaults to fields not explicitly set in config.toml."""
    defaults = _TIER_DEFAULTS.get(tier, {})
    for section_name, values in defaults.items():
        section = getattr(config, section_name)
        for key, val in values.items():
            setattr(section, key, val)


# ─────────────────────────────────────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Path | None = None) -> RAGConfig:
    """
    Load and return a RAGConfig.

    Search order:
      1. config_path (if provided)
      2. config.toml in the current working directory
      3. config.toml next to this file (project root)
      4. config.template.toml next to this file (read-only defaults)
      5. Built-in dataclass defaults

    After loading, detects or resolves the hardware tier and applies
    tier-specific defaults for any values not explicitly set.
    """
    config = RAGConfig()
    raw: dict = {}

    candidates = []
    if config_path:
        candidates.append(Path(config_path))

    # Look in CWD and project root
    cwd = Path.cwd()
    pkg_root = Path(__file__).parent.parent  # rag/ → project root
    candidates += [
        cwd / "config.toml",
        pkg_root / "config.toml",
        pkg_root / "config.template.toml",
    ]

    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "rb") as f:
                raw = tomllib.load(f)
            break

    # Populate dataclass fields from the raw TOML dict
    _populate_section(config.hardware, raw.get("hardware", {}))
    _populate_section(config.models, raw.get("models", {}))
    _populate_section(config.llm, raw.get("llm", {}))
    _populate_section(config.retrieval, raw.get("retrieval", {}))
    _populate_section(config.chunking, raw.get("chunking", {}))
    _populate_section(config.generation, raw.get("generation", {}))
    _populate_section(config.storage, raw.get("storage", {}))
    _populate_section(config.parsers, raw.get("parsers", {}))

    # Resolve tier
    if config.hardware.tier == "auto":
        config.resolved_tier = detect_hardware_tier()
    else:
        config.resolved_tier = config.hardware.tier.upper()

    # Apply tier-specific defaults
    _apply_tier_defaults(config, config.resolved_tier)

    return config


def _populate_section(obj: object, data: dict) -> None:
    """Set attributes on a dataclass instance from a dict, ignoring unknown keys."""
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
