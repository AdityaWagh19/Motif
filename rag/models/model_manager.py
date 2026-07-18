"""
rag/models/model_manager.py — Lazy-load singleton for all model instances.

No module instantiates Embedder, Reranker, or LLMClient directly.
All model access goes through get_model_manager().

Design:
  - Module-level singleton (_manager) — one instance per process.
  - Lazy imports inside each get_*() method — prevents circular imports and
    ensures model code is not imported until actually needed.
  - Lazy loading — models are not loaded until first use.
  - after_ingestion() handles the T1 memory constraint: unload the embedder
    before the LLM loads (combined they would exceed 6 GB on T1).

Dependency graph position:
    model_manager  →  rag.config  →  (stdlib)
    model_manager  --lazy--> rag.models.embedder
    model_manager  --lazy--> rag.models.reranker
    model_manager  --lazy--> rag.generation.llm_client

No model code is imported at module load time.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rag.config import RAGConfig

if TYPE_CHECKING:
    # These imports are only for type checkers — never executed at runtime here.
    from rag.models.embedder import Embedder
    from rag.models.reranker import Reranker
    from rag.generation.llm_client import LLMClient  # type: ignore[import]  # Phase 3

log = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager for Embedder, Reranker, and LLMClient instances.

    Models are loaded on first access and optionally unloaded to manage RAM.
    The singleton is accessed via get_model_manager() — never instantiate this
    class directly.
    """

    def __init__(self) -> None:
        self._embedder: Optional["Embedder"] = None
        self._reranker: Optional["Reranker"] = None
        self._llm: Optional["LLMClient"] = None

    # ------------------------------------------------------------------
    # Embedder
    # ------------------------------------------------------------------

    def get_embedder(self, config: RAGConfig) -> "Embedder":
        """
        Return the loaded Embedder, loading it on first call.

        Raises:
            FileNotFoundError: If the model directory does not exist.
                               Run `motif setup` to download models.
        """
        if self._embedder is None:
            from rag.models.embedder import Embedder  # lazy import

            model_path = Path(config.models.embed_model)
            if not model_path.is_absolute():
                model_path = Path(config.models.embed_model).resolve()

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Embedding model not found: {model_path}\n"
                    f"Run `motif setup` to download models."
                )
            log.info("Loading embedder from %s", model_path)
            self._embedder = Embedder(model_path)
            self._embedder._load()

        return self._embedder

    def unload_embedder(self) -> None:
        """Unload the embedder and release its memory."""
        if self._embedder is not None:
            log.debug("Unloading embedder")
            self._embedder.unload()
        self._embedder = None

    # ------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------

    def get_reranker(self, config: RAGConfig) -> "Reranker":
        """
        Return the loaded Reranker, loading it on first call.

        Raises:
            FileNotFoundError: If the model directory does not exist.
        """
        if self._reranker is None:
            from rag.models.reranker import Reranker  # lazy import

            model_path = Path(config.models.reranker)
            if not model_path.is_absolute():
                model_path = Path(config.models.reranker).resolve()

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Reranker model not found: {model_path}\n"
                    f"Run `motif setup` to download models."
                )
            log.info("Loading reranker from %s", model_path)
            self._reranker = Reranker(model_path)
            self._reranker._load()

        return self._reranker

    def unload_reranker(self) -> None:
        """Unload the reranker and release its memory."""
        if self._reranker is not None:
            log.debug("Unloading reranker")
            self._reranker.unload()
        self._reranker = None

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def get_llm(self, config: RAGConfig) -> "LLMClient":
        """
        Return the loaded LLMClient, loading it on first call.

        Raises:
            FileNotFoundError: If the GGUF model file does not exist.
        """
        if self._llm is None:
            from rag.generation.llm_client import LLMClient  # type: ignore[import]  # Phase 3

            model_path = Path(config.models.llm_path)
            if not model_path.is_absolute():
                model_path = Path(config.models.llm_path).resolve()

            if not model_path.exists():
                raise FileNotFoundError(
                    f"LLM model not found: {model_path}\n"
                    f"Run `motif setup` to download models."
                )
            log.info("Loading LLM from %s", model_path)
            self._llm = LLMClient(model_path, config)
            self._llm._load()

        return self._llm

    def unload_llm(self) -> None:
        """Unload the LLM and release its memory."""
        if self._llm is not None:
            log.debug("Unloading LLM")
            self._llm.unload()
        self._llm = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def unload_all(self) -> None:
        """Unload all models. Called on clean exit."""
        self.unload_embedder()
        self.unload_reranker()
        self.unload_llm()

    def after_ingestion(self, config: RAGConfig) -> None:
        """
        Post-ingestion memory management.

        T1 (CPU-only, ≤16 GB RAM):
            Unload the embedder (~550 MB) before the LLM loads.
            Combined RAM would exceed the T1 budget.

        T2 / T3 (GPU with VRAM, or ≥32 GB RAM):
            Keep the embedder loaded for faster follow-up queries.
        """
        tier = getattr(config, "resolved_tier", config.hardware.tier)
        if tier == "T1":
            log.debug("T1 post-ingestion: unloading embedder to free RAM")
            self.unload_embedder()

    def status(self) -> dict:
        """
        Return a dict indicating which models are currently loaded.

        Used by /status command and logging.
        """
        embedder_loaded = (
            self._embedder is not None and self._embedder.is_loaded()
        )
        reranker_loaded = (
            self._reranker is not None and self._reranker.is_loaded()
        )
        llm_loaded = self._llm is not None

        return {
            "embedder_loaded": embedder_loaded,
            "reranker_loaded": reranker_loaded,
            "llm_loaded": llm_loaded,
        }


# ---------------------------------------------------------------------------
# Module-level singleton — the single ModelManager instance for this process.
# All callers use get_model_manager() — never ModelManager() directly.
# ---------------------------------------------------------------------------

_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Return the process-level ModelManager singleton."""
    return _manager
