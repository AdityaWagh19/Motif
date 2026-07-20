"""
rag/pipeline.py — QueryPipeline: the central RAG query coordinator.

Contains NO business logic. Delegates to specialist modules:
  - rag.retrieval.expander    → query expansion / embedding
  - rag.retrieval.fusion      → RRF multi-list fusion
  - rag.reranking.cross_encoder → cross-encoder reranking
  - rag.generation.context_builder → context assembly + anti-middle ordering
  - rag.generation.prompts    → prompt formatting + citation building
  - rag.models.model_manager  → lazy model loading / memory management

Query flow:
    User query
         │
         ▼
    QueryExpander.expand()    → query vector (768-dim float32)
         │
         ├──► VectorStore.search_dense()  → dense candidates [(id, score)]
         ├──► BM25Index.search()          → lexical candidates [(id, score)]
         │
         ▼
    rrf_fuse()                → merged candidates, RRF scored
         │
         ▼
    rrf_to_scored_passages()  → List[ScoredPassage] with Chunk objects
         │
         ▼
    rerank()                  → top-K ScoredPassage, cross-encoder scored
         │
         ▼
    ContextBuilder.build()    → (prompt, ordered_passages)
         │
         ▼
    LLMClient.stream()        → token-by-token answer to terminal
         │
         ▼
    build_citations()         → List[Citation]
         │
         ▼
    AnswerResult returned
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, TYPE_CHECKING

from rag.config import RAGConfig
from rag.types import AnswerResult
from rag.retrieval.fusion import rrf_fuse, rrf_to_scored_passages
from rag.retrieval.expander import QueryExpander
from rag.reranking.cross_encoder import rerank
from rag.generation.context_builder import ContextBuilder
from rag.generation.prompts import build_citations
from rag.models.model_manager import get_model_manager
from rag.storage.chunk_store import ChunkStore
from rag.retrieval.bm25_index import BM25Index
from rag.retrieval.vector_store import VectorStore

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class QueryPipeline:
    """
    Stateless* query executor. Instantiate once per REPL session.

    *Storage objects (ChunkStore, BM25Index, VectorStore) hold open handles
    but are themselves stateless between queries.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._chunk_store = ChunkStore(config)
        self._bm25 = BM25Index(config)
        self._vector_store = VectorStore(config)
        self._context_builder = ContextBuilder()
        self._expander = QueryExpander()
        self._intent_classifier = None
        # Cache is lazily initialised on first query if enabled.
        self._cache: object = None  # Optional[QueryCache]

    # ── Public API ─────────────────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        history: List[dict],
        file_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
        page_range: Optional[str] = None,
        use_hyde: bool = False,   # Phase 3: opt-in only. Pass True or use /hyde modifier.
        show_sources: bool = True,
    ) -> AnswerResult:
        """
        Run the full RAG pipeline for one query.

        Args:
            query:       Raw user question string.
            history:     Rolling conversation history (may be empty list).
            file_filter: Restrict retrieval to chunks from this filename/path.
            type_filter: Restrict to document type ("pdf", "md", "audio", …).
            page_range:  Restrict to page numbers, format "MIN-MAX" e.g. "20-40".
            use_hyde:    Phase 4 flag — ignored in Phase 3.
            show_sources: If True, print citation list after the answer.

        Returns:
            AnswerResult with text, citations, timing, and passage count.
        """
        from rag.theme import console
        t_start = time.monotonic()
        cfg = self._config

        # ── 0a. Intent Classification (Runs before cache) ──────────────────────
        try:
            embedder = get_model_manager().get_embedder(cfg)
        except FileNotFoundError as exc:
            console.print(f"[error]Embedder not available:[/error] {exc}")
            return AnswerResult(
                text=str(exc),
                citations=[],
                passages_used=0,
            )

        if self._intent_classifier is None:
            from rag.intent import IntentClassifier
            self._intent_classifier = IntentClassifier(embedder, threshold=cfg.retrieval.chitchat_threshold)
            
        from rag.intent import Intent
        intent = self._intent_classifier.classify(query)
        if intent == Intent.GREETING_FAST:
            result = AnswerResult(
                text="Hello! Ask me anything about your documents.",
                citations=[],
                passages_used=0,
                latency_ms=(time.monotonic() - t_start) * 1000,
                retrieval_latency_ms=0,
                generation_latency_ms=0,
                tier=cfg.resolved_tier,
            )
            console.print(f"\n{result.text}\n")
            return result
        elif intent == Intent.CHITCHAT:
            result = self._handle_chitchat(query, cfg, console, t_start)
            return result

        # ── 0b. Query cache check ──────────────────────────────────────────────
        if getattr(cfg.storage, "query_cache_enabled", False):
            from rag.storage.query_cache import QueryCache
            if self._cache is None:
                self._cache = QueryCache(cfg)  # type: ignore[assignment]
            cached = self._cache.get(  # type: ignore[union-attr]
                query, file_filter, type_filter, page_range
            )
            if cached is not None:
                log.info("Cache HIT for query: %.60s", query)
                if show_sources and cached.citations:
                    console.print()
                    console.print("[structure](cached)[/structure]")
                    console.print(cached.text)
                    console.print("\n[structure]Sources:[/structure]")
                    for c in cached.citations:
                        console.print(f"  [structure]{c.format()}[/structure]")
                return cached

        # ── 0b. Guard: index must be populated ────────────────────────────────
        if self._chunk_store.count() == 0:
            console.print(
                "[warning]No documents indexed.[/warning] "
                "Run [bold]/ingest PATH[/bold] first."
            )
            return AnswerResult(
                text="No documents are indexed. Run /ingest to add documents first.",
                citations=[],
                passages_used=0,
            )

        # ── 1. Expand query → embedding ──────────────────────────────────────────────
        query_vector, effective_query = self._expander.expand(query, cfg, embedder)

        # ── 2. Retrieve ────────────────────────────────────────────────────────
        t_retrieval_start = time.monotonic()
        top_k_retrieval: int = cfg.retrieval.top_k_retrieval
        filter_dict = self._build_filter(file_filter, type_filter, page_range)

        dense_results = self._vector_store.search_dense(
            query_vector,
            top_k=top_k_retrieval,
            filter_=filter_dict if filter_dict else None,
        )
        bm25_results = self._bm25.search(effective_query, top_k=top_k_retrieval)

        log.debug(
            "Retrieval: %d dense, %d BM25 candidates",
            len(dense_results),
            len(bm25_results),
        )

        # ── 3. RRF fusion ──────────────────────────────────────────────────────
        fused = rrf_fuse([dense_results, bm25_results], top_k=top_k_retrieval)
        candidates = rrf_to_scored_passages(fused, self._chunk_store)

        t_retrieval_ms = (time.monotonic() - t_retrieval_start) * 1000

        if not candidates:
            console.print(
                "[warning]No relevant passages found for your query.[/warning]"
            )
            return AnswerResult(
                text="I cannot find an answer to this in the available documents.",
                citations=[],
                passages_used=0,
                retrieval_latency_ms=t_retrieval_ms,
            )

        # ── 4. Rerank ──────────────────────────────────────────────────────────
        top_k_rerank: int = cfg.retrieval.top_k_rerank
        threshold: float = cfg.retrieval.relevance_threshold

        # ── Reranker candidate guard ───────────────────────────────────────
        MAX_EFFICIENT_RERANK = 20
        if len(candidates) > MAX_EFFICIENT_RERANK:
            log.debug(
                "Reranker received %d candidates (efficient max: %d). "
                "Each extra candidate adds ~8 ms reranking latency.",
                len(candidates),
                MAX_EFFICIENT_RERANK,
            )

        try:
            reranked = rerank(
                query,
                candidates,
                cfg,
                top_k=top_k_rerank,
                threshold=threshold,
            )
        except FileNotFoundError as exc:
            # Reranker model not downloaded — fall back to RRF scores
            log.warning("Reranker not available (%s) — using RRF scores.", exc)
            console.print(
                f"[warning]Reranker model not found.[/warning] "
                f"Falling back to retrieval scores.\n"
                f"Run [bold]motif setup[/bold] to download the reranker."
            )
            # Use top candidates from RRF directly
            from rag.types import ScoredPassage
            reranked = candidates[:top_k_rerank]

        if not reranked:
            log.warning("No passages met the relevance threshold. Falling back to top RRF candidates.")
            console.print(
                "[structure]No passages met the relevance threshold "
                f"({threshold:.2f}). Falling back to retrieval scores.[/structure]"
            )
            # Give the LLM a chance to extract information even if cross-encoder is pessimistic
            reranked = candidates[:top_k_rerank]

        # ── 5. Build context and prompt ────────────────────────────────────────
        history_context = self._get_history_context(history)
        prompt, passages_used = self._context_builder.build(
            reranked, query, history_context, cfg
        )

        # ── 6. Stream answer ───────────────────────────────────────────────────
        t_gen_start = time.monotonic()

        try:
            llm = get_model_manager().get_llm(cfg)
        except FileNotFoundError as exc:
            console.print(f"[error]LLM not available:[/error] {exc}")
            return AnswerResult(
                text=str(exc),
                citations=[],
                passages_used=len(passages_used),
                retrieval_latency_ms=t_retrieval_ms,
            )

        console.print()
        full_answer = ""
        ttft_ms = 0.0
        
        from rich.live import Live
        from rich.markdown import Markdown
        
        try:
            # refresh_per_second throttles terminal updates to prevent flickering
            with Live(Markdown(""), console=console, refresh_per_second=15, transient=False) as live:
                for i, token in enumerate(llm.stream(
                    prompt,
                    max_tokens=cfg.llm.max_tokens,
                    temperature=cfg.llm.temperature,
                )):
                    if i == 0:
                        ttft_ms = (time.monotonic() - t_gen_start) * 1000
                    full_answer += token
                    live.update(Markdown(full_answer))
        except Exception as e:
            console.print(f"[error]Error during generation: {e}[/error]")

        t_gen_ms = (time.monotonic() - t_gen_start) * 1000

        # ── 7. Citations ───────────────────────────────────────────────────────
        citations = build_citations(passages_used)

        if show_sources and citations:
            console.print()
            console.print("[structure]Sources:[/structure]")
            for c in citations:
                console.print(f"  [structure]{c.format()}[/structure]")

        t_total_ms = (time.monotonic() - t_start) * 1000

        log.info(
            "Query complete: %.1f ms total (retrieval %.1f ms, gen %.1f ms), "
            "%d passages used",
            t_total_ms,
            t_retrieval_ms,
            t_gen_ms,
            len(passages_used),
        )

        result = AnswerResult(
            text=full_answer,
            citations=citations,
            passages_used=len(passages_used),
            latency_ms=t_total_ms,
            ttft_ms=ttft_ms,
            retrieval_latency_ms=t_retrieval_ms,
            generation_latency_ms=t_gen_ms,
            tier=cfg.resolved_tier,
        )

        # ── 8. Store in cache ──────────────────────────────────────────────────
        if getattr(cfg.storage, "query_cache_enabled", False) and self._cache is not None:
            self._cache.put(  # type: ignore[union-attr]
                query, result, file_filter, type_filter, page_range
            )

        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _handle_chitchat(self, query: str, cfg, console, start_time: float) -> AnswerResult:
        from rag.generation.prompts import CHITCHAT_PROMPT
        prompt = CHITCHAT_PROMPT.format(query=query)
        
        t_gen_start = time.monotonic()
        try:
            llm = get_model_manager().get_llm(cfg)
        except FileNotFoundError as exc:
            console.print(f"[error]LLM not available:[/error] {exc}")
            return AnswerResult(text=str(exc), citations=[], passages_used=0)
            
        console.print()
        full_answer = ""
        
        from rich.live import Live
        from rich.markdown import Markdown
        
        try:
            with Live(Markdown(""), console=console, refresh_per_second=15, transient=False) as live:
                for token in llm.stream(prompt, max_tokens=cfg.llm.max_tokens, temperature=cfg.llm.temperature):
                    full_answer += token
                    live.update(Markdown(full_answer))
        except Exception as e:
            console.print(f"[error]Error during generation: {e}[/error]")
            
        t_gen_ms = (time.monotonic() - t_gen_start) * 1000
        t_total_ms = (time.monotonic() - start_time) * 1000
        
        log.info("Chit-chat query complete: %.1f ms total (gen %.1f ms)", t_total_ms, t_gen_ms)
        
        return AnswerResult(
            text=full_answer,
            citations=[],
            passages_used=0,
            latency_ms=t_total_ms,
            retrieval_latency_ms=0,
            generation_latency_ms=t_gen_ms,
            tier=cfg.resolved_tier,
        )

    def _get_history_context(self, full_history: List[dict]) -> List[dict]:
        """Return a rolling window of history that fits within the token budget."""
        from rag.session import Session
        tmp = Session(self._config)
        tmp.history = full_history
        return tmp.get_history_for_context(
            token_budget=self._config.generation.context_max_tokens,
            passage_tokens=500,  # conservative estimate before passages are selected
        )

    def _build_filter(
        self,
        file_filter: Optional[str],
        type_filter: Optional[str],
        page_range: Optional[str],
    ) -> dict:
        """
        Build a Qdrant metadata filter dict from inline modifiers.

        Returns an empty dict if no filters are specified (callers must check).
        """
        f: dict = {}
        if file_filter:
            f["source"] = file_filter
        if type_filter:
            f["source_type"] = type_filter
        if page_range and "-" in page_range:
            parts = page_range.split("-", 1)
            try:
                f["page_min"] = int(parts[0])
                f["page_max"] = int(parts[1])
            except ValueError:
                log.warning("Invalid page_range format: %r (expected 'MIN-MAX')", page_range)
        return f
