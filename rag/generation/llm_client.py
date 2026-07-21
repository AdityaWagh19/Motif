"""
rag/generation/llm_client.py — llama-cpp-python streaming wrapper.

Wraps the Llama class from llama-cpp-python for token-by-token streaming
and non-streaming generation. All model loading is lazy (inside _load()).

Backend: llama-cpp-python (Llama class)
Models:
  T1: Phi-3.5-mini-instruct-Q4_K_M.gguf  (2.2 GB)
  T2/T3: Qwen2.5-7B-Instruct-Q4_K_M.gguf (4.2 GB)

Memory policy:
  - use_mmap=True:  Memory-map the model file (does not load into RAM until needed).
  - use_mlock=False: Do not pin pages (allows OS to page out unused model weights).
  - verbose=False:  Suppress llama.cpp debug output to terminal.
  - n_gpu_layers:   From config.llm.n_gpu_layers (0 = CPU only, >0 = partial GPU).

Stop sequences include common chat template markers to prevent the model from
generating additional spurious turns.

Always accessed through ModelManager.get_llm() — never instantiate directly.
"""
from __future__ import annotations

import gc
import logging
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

# Stop sequences that signal end-of-response across multiple model families.
# Model-specific markers:
#   Phi-3.5:  <|end|>
#   Qwen2.5:  <|im_end|>
#   LLaMA 2:  </s>, [/INST]
#   General:  User: / \n\nUser (prevents multi-turn hallucination)
STOP_TOKENS: list[str] = [
    "</s>",
    "<|end|>",
    "<|im_end|>",
    "[/INST]",
    "User:",
    "\n\nUser",
    "<|endoftext|>",
]


class LLMClient:
    """
    Streaming wrapper around llama-cpp-python's Llama class.

    Usage (via ModelManager):
        llm = get_model_manager().get_llm(config)
        for token in llm.stream(prompt, max_tokens=400):
            print(token, end="", flush=True)
    """

    def __init__(self, model_path: Path, config: RAGConfig) -> None:
        """
        Store model path and config. Does NOT load the model.
        Call _load() (via ModelManager) before stream()/generate().

        Args:
            model_path: Absolute path to the GGUF model file.
            config:     RAGConfig — reads llm.* settings on load.
        """
        self._model_path = model_path
        self._config = config
        self._llm: object | None = None  # llama_cpp.Llama

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """
        Load the Llama model into memory.

        Raises:
            ImportError:      If llama-cpp-python is not installed.
            FileNotFoundError: If the GGUF file does not exist.
            RuntimeError:     If llama.cpp fails to initialise.
        """
        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed.\n"
                "Run: pip install llama-cpp-python\n"
                "Or with CUDA: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
            ) from exc

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"LLM model not found: {self._model_path}\n"
                "Run `motif setup` to download models."
            )

        cfg = self._config.llm
        log.info(
            "Loading LLM %s (ctx=%d, gpu_layers=%d, threads=%d)",
            self._model_path.name,
            cfg.ctx_size,
            cfg.n_gpu_layers,
            cfg.threads,
        )

        use_flare = getattr(self._config.retrieval, "use_flare", False)
        
        self._llm = Llama(  # type: ignore[attr-defined]
            model_path=str(self._model_path),
            n_ctx=cfg.ctx_size,
            n_gpu_layers=cfg.n_gpu_layers,
            n_threads=cfg.threads,
            verbose=False,       # suppress llama.cpp debug output
            use_mlock=False,     # don't pin memory
            use_mmap=True,       # memory-map model file
            logits_all=use_flare, # 7-C: Required for logprobs streaming
        )

        # ── GPU offload verification ─────────────────────────────────────────
        # Note: llama-cpp-python v0.3.34+ no longer exposes n_gpu_layers as an attribute
        # on the Llama object directly.
        n_layers_requested = cfg.n_gpu_layers
        
        if n_layers_requested > 0:
            log.info(
                "GPU offload enabled (requested %d layers on backend).",
                n_layers_requested,
            )

        log.info("LLM loaded — model: %s", self._model_path.name)

    def unload(self) -> None:
        """Release model memory and run GC."""
        del self._llm
        self._llm = None
        gc.collect()
        log.debug("LLM unloaded")

    def is_loaded(self) -> bool:
        """Return True if the model is loaded and ready."""
        return self._llm is not None

    # ── Inference ─────────────────────────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.1,
        stop: list[str] | None = None,
        return_logprobs: bool = False,
    ) -> Generator[str | tuple[str, float], None, None]:
        """
        Stream the LLM response token by token.

        Yields each token text string as it is generated.
        Caller is responsible for printing and flushing.

        Args:
            prompt:      Complete prompt string (assembled by ContextBuilder).
            max_tokens:  Maximum number of new tokens to generate.
            temperature: Sampling temperature (0.1 = near-deterministic).
            stop:        Additional stop sequences beyond STOP_TOKENS.

        Yields:
            Token text strings (may be empty — skip those in callers).

        Raises:
            RuntimeError: If _load() has not been called.
        """
        if self._llm is None:
            raise RuntimeError(
                "LLMClient is not loaded. Call _load() before stream()."
            )

        stop_seqs = list(STOP_TOKENS)
        if stop:
            stop_seqs.extend(stop)

        messages = [
            {"role": "user", "content": prompt}
        ]

        output = self._llm.create_chat_completion(  # type: ignore[operator]
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_seqs,
            stream=True,
            logprobs=True if return_logprobs else None,
            top_logprobs=1 if return_logprobs else None,
        )

        for chunk in output:
            choice = chunk["choices"][0]
            delta = choice["delta"]
            if "content" in delta:
                token = delta["content"]
                if return_logprobs:
                    # Retrieve logprob for this token
                    # In llama_cpp chat completions with logprobs, it's typically inside choice["logprobs"]["content"][0]["logprob"]
                    logprob = 0.0
                    if "logprobs" in choice and choice["logprobs"] and "content" in choice["logprobs"]:
                        content_logprobs = choice["logprobs"]["content"]
                        if content_logprobs and len(content_logprobs) > 0:
                            logprob = content_logprobs[0].get("logprob", 0.0)
                    yield (token, logprob)
                else:
                    yield token

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> str:
        """
        Non-streaming generation. Returns the complete response as a string.

        Used for:
          - HyDE hypothetical answer generation (Phase 4)
          - RAGAS evaluation (Phase 6)
          - Testing (deterministic output with temperature=0)

        Args:
            prompt:      Complete prompt string.
            max_tokens:  Maximum number of new tokens.
            temperature: Sampling temperature.
            stop:        Additional stop sequences.

        Returns:
            Full response string.
        """
        return "".join(self.stream(prompt, max_tokens, temperature, stop))
