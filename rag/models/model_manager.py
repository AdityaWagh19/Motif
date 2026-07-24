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
from typing import TYPE_CHECKING

from rag.config import RAGConfig, resolve_model_path

if TYPE_CHECKING:
    # These imports are only for type checkers — never executed at runtime here.
    from rag.generation.llm_client import LLMClient  # type: ignore[import]  # Phase 3
    from rag.models.captioner import Captioner
    from rag.models.embedder import Embedder
    from rag.models.reranker import Reranker

log = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager for Embedder, Reranker, and LLMClient instances.

    Models are loaded on first access and optionally unloaded to manage RAM.
    The singleton is accessed via get_model_manager() — never instantiate this
    class directly.
    """

    def __init__(self) -> None:
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None
        self._llm: LLMClient | None = None
        self._captioner: Captioner | None = None
        self._whisper = None

    # ------------------------------------------------------------------
    # Embedder
    # ------------------------------------------------------------------

    def get_embedder(self, config: RAGConfig) -> Embedder:
        """
        Return the loaded Embedder, loading it on first call.

        Raises:
            FileNotFoundError: If the model directory does not exist.
                               Run `motif setup` to download models.
        """
        if self._embedder is None:
            from rag.models.embedder import Embedder  # lazy import

            model_path = resolve_model_path(config.models.embed_model)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Embedding model not found: {model_path}\n"
                    f"Run `motif setup` to download models."
                )
            log.info("Loading embedder from %s", model_path)
            self._embedder = Embedder(model_path)
            try:
                self._embedder._load()
            except Exception:
                self._embedder = None
                raise

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

    def get_reranker(self, config: RAGConfig) -> Reranker:
        """
        Return the loaded Reranker, loading it on first call.

        Supports both file-form and directory-form model paths:
          - File: models/MiniLM-L12-v2/onnx/model_O3.onnx  (explicit ONNX file)
          - Dir:  models/MiniLM-L12-v2/                     (auto-discovers ONNX inside)
          - Dir:  models/bge-reranker-base/                  (T3 model)

        Raises:
            FileNotFoundError: If the model directory/file does not exist.
        """
        if self._reranker is None:
            from rag.models.reranker import Reranker  # lazy import

            model_path = resolve_model_path(config.models.reranker)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Reranker model not found: {model_path}\n"
                    f"Run `motif setup` to download models."
                )

            log.info("Loading reranker from %s", model_path)
            self._reranker = Reranker(model_path)

            try:
                self._reranker._load()
            except Exception:
                self._reranker = None
                raise

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

    def get_llm(self, config: RAGConfig) -> LLMClient:
        """
        Return the loaded LLMClient, loading it on first call.

        Raises:
            FileNotFoundError: If the GGUF model file does not exist.
        """
        if self._llm is None:
            from rag.generation.llm_client import LLMClient  # type: ignore[import]  # Phase 3

            model_path = resolve_model_path(config.models.llm_path)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"LLM model not found: {model_path}\n"
                    f"Run `motif setup` to download models."
                )
            if config.hardware.backend in ("cuda", "metal", "rocm") and config.llm.n_gpu_layers == 0:
                log.warning(
                    f"GPU offload requested ({config.hardware.backend}), but "
                    f"n_gpu_layers is 0. Running in CPU mode."
                )

            log.info("Loading LLM from %s", model_path)
            self._llm = LLMClient(model_path, config)
            try:
                self._llm._load()
                if config.hardware.backend in ("cuda", "metal", "rocm") and config.llm.n_gpu_layers > 0:
                    try:
                        log.debug("Running LLM GPU warm-up pass...")
                        # Pass a dummy string and request 1 token
                        _ = self._llm.generate("hello", max_tokens=1)
                    except Exception:
                        pass
            except Exception:
                self._llm = None
                raise

        return self._llm

    def unload_llm(self) -> None:
        """Unload the LLM and release its memory."""
        if self._llm is not None:
            log.debug("Unloading LLM")
            self._llm.unload()
        self._llm = None

    # ------------------------------------------------------------------
    # Captioner
    # ------------------------------------------------------------------

    def get_captioner(self, config: RAGConfig) -> Captioner:
        """Lazy-load moondream2 captioning model."""
        from rag.models.captioner import Captioner
        if self._captioner is None:
            model_path = resolve_model_path(getattr(config.models, "captioner", "models/moondream2"))

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Captioning model not found: {model_path}\n"
                    f"Run `motif setup --tier T3 --captioning` to download."
                )
            self._captioner = Captioner(model_path)
            try:
                self._captioner._load()
            except Exception:
                self._captioner = None
                raise
        return self._captioner

    # ------------------------------------------------------------------
    # Whisper
    # ------------------------------------------------------------------

    def get_whisper(self, config: RAGConfig):
        """Lazy-load whisperx models (transcription and diarization)."""
        if self._whisper is None:
            try:
                import whisperx
            except ImportError as exc:
                raise RuntimeError("whisperx is not installed") from exc

            # whisperx downloads models via huggingface hub if not present.
            # In a fully offline setup, we rely on HF_HOME being populated.
            log.info("Loading WhisperX model (base)...")
            device = "cuda" if config.hardware.backend in ("cuda", "rocm") else "cpu"
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                log.warning("Config requests CUDA but Torch is not compiled with CUDA. Falling back to CPU.")
                device = "cpu"
                
            compute_type = "float16" if device == "cuda" else "int8"
            
            try:
                model = whisperx.load_model("base", device, compute_type=compute_type)
                self._whisper = {"model": model, "device": device, "compute_type": compute_type}
            except Exception as e:
                log.error(f"Failed to load WhisperX model: {e}")
                raise
                
        return self._whisper
        
    def unload_whisper(self) -> None:
        """Unload whisper model and release memory."""
        if self._whisper is not None:
            log.debug("Unloading Whisper")
        self._whisper = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def unload_all(self) -> None:
        """Unload all models. Called on clean exit."""
        self.unload_embedder()
        self.unload_reranker()
        self.unload_llm()
        self.unload_whisper()

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
