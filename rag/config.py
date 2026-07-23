"""
rag/config.py — Configuration dataclasses and hardware tier detection.

Loads config.toml from the project root (or a specified path).
Falls back to config.template.toml if config.toml does not exist.

Hardware Tier Summary
---------------------
T1  — CPU only / < 3800 MB VRAM / Mac < 8 GB unified memory
        Model: Phi-3.5-mini-instruct-Q4_K_M.gguf (2.2 GB)
        n_gpu_layers: 0

T2  — 3800–6000 MB VRAM / Mac 8–15 GB unified memory (GTX 1650 / M1 8 GB)
        Model: Qwen2.5-7B-Instruct-Q4_K_M.gguf (4.2 GB)
        n_gpu_layers: 20 (CUDA/Metal)

T3  — >= 6000 MB VRAM / Mac 16+ GB unified memory (RTX 3050+ / M2 Pro 16 GB+)
        Model: Qwen2.5-7B-Instruct-Q4_K_M.gguf (4.2 GB)
        n_gpu_layers: 28 (CUDA/Metal)

Backend Detection Priority
--------------------------
1. NVIDIA GPU  → nvidia-smi  (Windows / Linux)
2. Apple Silicon → sysctl hw.memsize  (macOS arm64 — Metal)
3. AMD ROCm GPU → rocm-smi  (Linux ROCm)
4. Fallback → CPU (T1)
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import platformdirs

# Hardware detection cache
_hw_cache: dict[str, str | None] = {}

# ── Windows DLL Directory Registration (Python 3.8+ ctypes fix for CUDA) ────
if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        for sub in ["bin", os.path.join("bin", "x64")]:
            p = os.path.join(cuda_path, sub)
            if os.path.isdir(p):
                try:
                    os.add_dll_directory(p)
                except Exception:
                    pass



# ─────────────────────────────────────────────────────────────────────────────
# Paths and Migration
# ─────────────────────────────────────────────────────────────────────────────

def get_app_dir() -> Path:
    """Return the global application directory using OS conventions."""
    # On Linux: ~/.local/share/motif
    # On Windows: %LOCALAPPDATA%/motif
    # On macOS: ~/Library/Application Support/motif
    return Path(platformdirs.user_data_dir("motif", appauthor=False)).resolve()

def migrate_if_needed(app_dir: Path | None = None) -> None:
    """Migrate legacy ~/.ragdb to the new <APP_DIR> structure atomically."""
    if app_dir is None:
        app_dir = get_app_dir()
        
    legacy_dir = Path(os.path.expanduser("~/.ragdb")).resolve()
    sentinel = app_dir / "migration.done"
    
    if sentinel.exists():
        return
        
    if legacy_dir.exists() and legacy_dir.is_dir():
        import logging
        log = logging.getLogger("rag.config")
        log.info(f"Migrating legacy database from {legacy_dir} to {app_dir}")
        
        # We ensure app_dir exists
        os.makedirs(str(app_dir), exist_ok=True)
        
        workspaces_dir = app_dir / "workspaces"
        os.makedirs(str(workspaces_dir), exist_ok=True)
        
        # Move everything inside legacy_dir to workspaces_dir (or specific dirs)
        # Note: legacy structure had ~/.ragdb/<workspace>/... and ~/.ragdb/query_cache.db and ~/.ragdb/motif.log
        for item in legacy_dir.iterdir():
            if item.name == "models":
                # Models go to app_dir/models
                shutil.move(str(item), str(app_dir / "models"))
            elif item.name == "query_cache.db":
                # Move into default workspace and rename
                default_ws = workspaces_dir / "default"
                os.makedirs(str(default_ws), exist_ok=True)
                shutil.move(str(item), str(default_ws / "query_cache.sqlite"))
            elif item.name == "motif.log":
                logs_dir = app_dir / "logs"
                os.makedirs(str(logs_dir), exist_ok=True)
                shutil.move(str(item), str(logs_dir / "motif.log"))
            else:
                if item.is_dir():
                    shutil.move(str(item), str(workspaces_dir / item.name))
        
    # Consolidate legacy SQLite files per workspace into motif_store.db
    workspaces_dir = app_dir / "workspaces"
    if workspaces_dir.exists():
        import sqlite3
        from rag.storage.db_manager import _CREATE_SCHEMA
        for ws in workspaces_dir.iterdir():
            if ws.is_dir():
                target_db = ws / "motif_store.db"
                legacy_chunks = ws / "chunks.db"
                legacy_tracker = ws / "ingestion_tracker.db"
                legacy_qc = ws / "query_cache.sqlite"

                if not target_db.exists() and (legacy_chunks.exists() or legacy_tracker.exists()):
                    try:
                        conn = sqlite3.connect(str(target_db))
                        conn.executescript(_CREATE_SCHEMA)

                        if legacy_chunks.exists():
                            src = sqlite3.connect(str(legacy_chunks))
                            rows = src.execute("SELECT * FROM chunks").fetchall()
                            if rows:
                                placeholders = ",".join(["?"] * len(rows[0]))
                                conn.executemany(f"INSERT OR REPLACE INTO chunks VALUES ({placeholders})", rows)
                            src.close()
                            legacy_chunks.rename(ws / "chunks.db.bak")

                        if legacy_tracker.exists():
                            src = sqlite3.connect(str(legacy_tracker))
                            rows = src.execute("SELECT * FROM files").fetchall()
                            for r in rows:
                                conn.execute("INSERT OR REPLACE INTO file_tracker VALUES (?, ?, ?, ?)", r)
                            src.close()
                            legacy_tracker.rename(ws / "ingestion_tracker.db.bak")

                        conn.commit()
                        conn.close()
                    except Exception as e:
                        import logging
                        logging.getLogger("rag.config").warning("Legacy DB migration warning for %s: %s", ws.name, e)

    # Mark as done
    os.makedirs(str(app_dir), exist_ok=True)
    sentinel.touch()

# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HardwareConfig:
    tier: str = "auto"    # "auto" | "T1" | "T2" | "T3"
    backend: str = "cpu"  # "cuda" | "metal" | "rocm" | "cpu"


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
    max_tokens: int = 150    # concise answers; reduces hallucination and repetition
    temperature: float = 0.1
    threads: int = 4


@dataclass
class RetrievalConfig:
    top_k_retrieval: int = 20
    top_k_rerank: int = 3
    relevance_threshold: float = 0.3
    chitchat_threshold: float = 0.80
    query_expansion: str = "none"   # "none" | "hyde"
    bm25_backend: str = "rank_bm25"
    use_raptor: bool = False
    use_parent_docs: bool = False
    use_flare: bool = False


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
    db_path: str = ""
    workspace: str = "default"
    query_cache_enabled: bool = True
    query_cache_ttl_hours: int = 24


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
        if self.storage.db_path and self.storage.db_path != "~/.ragdb":
            base = Path(os.path.expanduser(self.storage.db_path)).resolve()
            return base / self.storage.workspace
        return get_app_dir() / "workspaces" / self.storage.workspace
        
    def save(self) -> None:
        """Save current configuration to the global config.toml."""
        app_dir = get_app_dir()
        config_path = app_dir / "config.toml"
        # We only really modify storage.workspace dynamically via CLI.
        # But for correctness, we update the [storage] workspace key.
        import re
        if config_path.exists():
            content = config_path.read_text(encoding="utf-8")
            # Minimal sed-like replacement to avoid writing full toml
            content = re.sub(r'(?<=workspace = ")[^"]*(?=")', self.storage.workspace, content)
            config_path.write_text(content, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Hardware Tier Detection — Multi-Backend
# ─────────────────────────────────────────────────────────────────────────────

def _detect_nvidia_tier() -> str | None:
    """
    Return tier string if an NVIDIA GPU is found via nvidia-smi, else None.
    Uses the highest-VRAM GPU when multiple GPUs are present.
    """
    if "nvidia" in _hw_cache:
        return _hw_cache["nvidia"]

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
            return None

        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return None

        # Use the highest VRAM GPU (multi-GPU support)
        vram_mb = max(int(l) for l in lines)
        if vram_mb >= 6000:
            tier = "T3"
        elif vram_mb >= 3800:
            tier = "T2"
        else:
            tier = "T1"
        _hw_cache["nvidia"] = tier
        return tier
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        _hw_cache["nvidia"] = None
        return None


def _detect_apple_silicon_tier() -> str | None:
    """
    Return tier string for Apple Silicon Macs (arm64), else None.
    Uses unified memory size as the effective VRAM for Metal inference.

    Memory thresholds:
      8–15 GB unified memory → T2 (Qwen2.5-7B Q4 partial Metal offload)
      16 GB+  unified memory → T3 (full Metal offload)
    """
    if "apple" in _hw_cache:
        return _hw_cache["apple"]

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        _hw_cache["apple"] = None
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            return None
        ram_bytes = int(result.stdout.strip())
        ram_gb = ram_bytes / (1024 ** 3)
        if ram_gb >= 16:
            tier = "T3"
        elif ram_gb >= 8:
            tier = "T2"
        else:
            tier = "T1"
        _hw_cache["apple"] = tier
        return tier
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        _hw_cache["apple"] = None
        return None


def _detect_amd_tier() -> str | None:
    """
    Return tier string if an AMD ROCm GPU is found via rocm-smi, else None.
    Parses the CSV output to find maximum VRAM across all GPUs.
    """
    if "amd" in _hw_cache:
        return _hw_cache["amd"]

    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        max_vram_mb = 0
        for line in result.stdout.splitlines():
            # Skip header lines
            if line.startswith("card") or line.startswith("GPU") or not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    vram_bytes = int(parts[1].strip())
                    vram_mb = vram_bytes / (1024 ** 2)
                    max_vram_mb = max(max_vram_mb, int(vram_mb))
                except (ValueError, IndexError):
                    continue

        if max_vram_mb == 0:
            _hw_cache["amd"] = None
            return None
        if max_vram_mb >= 6000:
            tier = "T3"
        elif max_vram_mb >= 3800:
            tier = "T2"
        else:
            tier = "T1"
        _hw_cache["amd"] = tier
        return tier
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        _hw_cache["amd"] = None
        return None


def detect_hardware_tier() -> str:
    """
    Universal hardware tier detection.

    Priority order:
      1. NVIDIA GPU via nvidia-smi     (Windows / Linux / WSL)
      2. Apple Silicon via sysctl      (macOS arm64 — uses Metal)
      3. AMD GPU via rocm-smi          (Linux ROCm)
      4. Fallback: CPU-only T1

    Returns:
        "T1" — CPU only, or GPU with < 3800 MB VRAM
        "T2" — 3800–6000 MB VRAM / Mac 8–15 GB unified memory
        "T3" — >= 6000 MB VRAM / Mac 16+ GB unified memory
    """
    tier = _detect_nvidia_tier()
    if tier:
        return tier

    tier = _detect_apple_silicon_tier()
    if tier:
        return tier

    tier = _detect_amd_tier()
    if tier:
        return tier

    return "T1"


def _resolve_backend(tier: str) -> str:
    """Return the GPU backend string for a detected tier."""
    if tier == "T1":
        return "cpu"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal"
    if _detect_amd_tier() is not None:
        return "rocm"
    if _detect_nvidia_tier() is not None:
        return "cuda"
    return "cpu"


def _check_cuda_toolkit() -> bool:
    """Return True if CUDA Toolkit is installed and CUDA_PATH is set."""
    cuda_path = os.environ.get("CUDA_PATH", "")
    if bool(cuda_path) and Path(cuda_path).exists():
        return True

    if platform.system() == "Windows":
        known_paths = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if known_paths.exists() and any(known_paths.iterdir()):
            return True

    return False


def _get_models_dir() -> Path:
    """Resolve the models directory in the global app dir."""
    return get_app_dir() / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Tier-specific defaults
# ─────────────────────────────────────────────────────────────────────────────

_TIER_DEFAULTS: dict[str, dict] = {
    "T1": {
        "llm": {"n_gpu_layers": 0, "ctx_size": 2048, "max_tokens": 150, "threads": 4},
        "retrieval": {"top_k_retrieval": 20, "top_k_rerank": 3, "query_expansion": "none"},
        "chunking": {"use_semantic": False},
        "generation": {"context_max_tokens": 2048},
        "models": {"llm_path": "models/Phi-3.5-mini-instruct-Q4_K_M.gguf", "reranker": "models/MiniLM-L12-v2"},
    },
    "T2": {
        "llm": {"n_gpu_layers": 20, "ctx_size": 3072, "max_tokens": 150, "threads": 6},
        "retrieval": {"top_k_retrieval": 25, "top_k_rerank": 5, "query_expansion": "none"},
        "chunking": {"use_semantic": True},
        "generation": {"context_max_tokens": 2048},
        "models": {"llm_path": "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf", "reranker": "models/MiniLM-L12-v2"},
    },
    "T3": {
        "llm": {"n_gpu_layers": 28, "ctx_size": 4096, "max_tokens": 150, "threads": 8},
        "retrieval": {"top_k_retrieval": 30, "top_k_rerank": 5, "query_expansion": "none"},
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
      2. .motif/config.toml in the current working directory (project override)
      3. config.toml in the global APP_DIR
    """
    import logging as _log
    _logger = _log.getLogger("rag.config")

    config = RAGConfig()
    raw: dict = {}
    
    app_dir = get_app_dir()
    
    global_config = app_dir / "config.toml"
    if not global_config.exists():
        template_path = Path(__file__).parent / "data" / "config.template.toml"
        if template_path.exists():
            os.makedirs(str(app_dir), exist_ok=True)
            shutil.copy2(str(template_path), str(global_config))

    candidates = []
    if config_path:
        candidates.append(Path(config_path))

    cwd = Path.cwd()
    candidates += [
        cwd / ".motif" / "config.toml",  # Local override
        global_config,                   # Global config
    ]

    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "rb") as f:
                raw = tomllib.load(f)
            break

    # ── Step 1: Resolve the tier ────────────────────────────────────────────
    # Read hardware section from raw first to check for manual tier override
    _populate_section(config.hardware, raw.get("hardware", {}))

    if config.hardware.tier == "auto":
        config.resolved_tier = detect_hardware_tier()
    else:
        config.resolved_tier = config.hardware.tier.upper()

    # ── Step 2: Apply tier defaults as the baseline ──────────────────────────
    _apply_tier_defaults(config, config.resolved_tier)

    # ── Step 3: Apply user overrides from config.toml (user always wins) ────
    _populate_section(config.models, raw.get("models", {}))
    _populate_section(config.llm, raw.get("llm", {}))
    _populate_section(config.retrieval, raw.get("retrieval", {}))
    _populate_section(config.chunking, raw.get("chunking", {}))
    _populate_section(config.generation, raw.get("generation", {}))
    _populate_section(config.storage, raw.get("storage", {}))
    _populate_section(config.parsers, raw.get("parsers", {}))

    # ── Step 4: Resolve and set backend ──────────────────────────────────────
    config.hardware.backend = _resolve_backend(config.resolved_tier)

    # ── Step 5: CUDA toolkit warning for T2/T3 on Windows/Linux ──────────────
    if (
        config.resolved_tier in ("T2", "T3")
        and config.hardware.backend == "cuda"
        and not _check_cuda_toolkit()
    ):
        _logger.warning(
            "Tier %s detected but CUDA_PATH is not set. "
            "GPU inference will be silently disabled. "
            "Install CUDA Toolkit 12.x to enable GPU acceleration: "
            "https://developer.nvidia.com/cuda-downloads",
            config.resolved_tier,
        )

    return config


def _populate_section(obj: object, data: dict) -> None:
    """Set attributes on a dataclass instance from a dict, ignoring unknown keys."""
    import logging
    log = logging.getLogger("rag.config")
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            log.warning("Unknown configuration key '%s' in section '%s'", key, obj.__class__.__name__)
