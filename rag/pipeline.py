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
from typing import TYPE_CHECKING

from rag.config import RAGConfig
from rag.generation.context_builder import ContextBuilder
from rag.generation.prompts import build_citations
from rag.models.model_manager import get_model_manager
from rag.reranking.cross_encoder import rerank
from rag.retrieval.bm25_index import BM25Index
from rag.retrieval.expander import QueryExpander
from rag.retrieval.fusion import rrf_fuse, rrf_to_scored_passages
from rag.retrieval.vector_store import VectorStore
from rag.storage.chunk_store import ChunkStore
from rag.types import AnswerResult

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
        history: list[dict],
        file_filter: str | None = None,
        type_filter: str | None = None,
        page_range: str | None = None,
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
            self._intent_classifier = IntentClassifier()
            
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
            from rag.generation.prompts import CHITCHAT_PROMPT
            return self._handle_llm_direct(query, CHITCHAT_PROMPT, cfg, console, t_start)

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

        # ── 0b. Guard: empty index fallback ─────────────────────────────────────
        if self._chunk_store.count() == 0:
            from rag.generation.prompts import FALLBACK_PROMPT_EMPTY_INDEX
            return self._handle_llm_direct(
                query, 
                FALLBACK_PROMPT_EMPTY_INDEX, 
                cfg, 
                console, 
                t_start
            )

        # ── 1. Rewrite query ─────────────────────────────────────────────────────────
        try:
            llm = get_model_manager().get_llm(cfg)
            from rag.generation.query_rewriter import rewrite_query
            search_query = rewrite_query(query, llm)
        except Exception as exc:
            log.warning("LLM not available for query rewrite (%s).", exc)
            search_query = query

        # ── 1b. Expand query → embedding ──────────────────────────────────────────────
        query_vector, effective_query = self._expander.expand(search_query, cfg, embedder)

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
            from rag.generation.prompts import CHITCHAT_PROMPT
            return self._handle_llm_direct(
                query, 
                CHITCHAT_PROMPT, 
                cfg, 
                console, 
                t_start, 
                retrieval_latency_ms=t_retrieval_ms
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
                search_query,
                candidates,
                cfg,
                top_k=top_k_rerank,
                threshold=threshold,
            )
        except FileNotFoundError as exc:
            # Reranker model not downloaded — fall back to RRF scores
            log.warning("Reranker not available (%s) — using RRF scores.", exc)
            console.print(
                "[warning]Reranker model not found.[/warning] "
                "Falling back to retrieval scores.\n"
                "Run [bold]motif setup[/bold] to download the reranker."
            )
            # Use top candidates from RRF directly
            reranked = candidates[:top_k_rerank]

        if not reranked:
            log.debug("No passages met the absolute relevance floor. Routing to fallback prompt.")
            from rag.generation.prompts import CHITCHAT_PROMPT
            return self._handle_llm_direct(
                query, 
                CHITCHAT_PROMPT, 
                cfg, 
                console, 
                t_start, 
                retrieval_latency_ms=t_retrieval_ms
            )
        # ── 4b. 7-B: Parent-Document Retrieval ─────────────────────────────────
        if getattr(cfg.retrieval, "use_parent_docs", False):
            for p in reranked:
                if p.chunk.parent_id:
                    parent_chunk = self._chunk_store.fetch_parent(p.chunk)
                    if parent_chunk:
                        log.debug("Swapping child chunk %s for parent chunk %s", p.chunk.id, parent_chunk.id)
                        p.chunk = parent_chunk

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
                
                use_flare = getattr(cfg.retrieval, "use_flare", False)
                if use_flare:
                    from rag.generation.flare import FlareController
                    
                    def _flare_retrieve(flare_query: str) -> str:
                        try:
                            q_vec, e_query = self._expander.expand(flare_query, cfg, embedder)
                            d_res = self._vector_store.search_dense(q_vec, top_k=top_k_retrieval, filter_=filter_dict if filter_dict else None)
                            b_res = self._bm25.search(e_query, top_k=top_k_retrieval)
                            fused = rrf_fuse([d_res, b_res], top_k=top_k_retrieval)
                            c = rrf_to_scored_passages(fused, self._chunk_store)
                            r = rerank(flare_query, c, cfg, top_k=top_k_rerank, threshold=threshold)
                            if getattr(cfg.retrieval, "use_parent_docs", False):
                                for p in r:
                                    if p.chunk.parent_id:
                                        parent_chunk = self._chunk_store.fetch_parent(p.chunk)
                                        if parent_chunk:
                                            p.chunk = parent_chunk
                            return "\n\n".join(f"---\n{p.chunk.text}" for p in r)
                        except Exception as e:
                            log.warning("FLARE retrieval failed: %s", e)
                            return ""
                            
                    controller = FlareController(
                        llm=llm,
                        base_prompt=prompt,
                        retrieve_fn=_flare_retrieve,
                        max_tokens=cfg.llm.max_tokens,
                        temperature=cfg.llm.temperature,
                    )
            import random
            thinking_phrases = [
                "Thinking...", "Analyzing...", "Understanding...", "Decoding...", 
                "Processing...", "Synthesizing...", "Evaluating...", "Investigating...",
                "Computing...", "Reasoning...", "Pondering...", "Scanning...",
                "Formulating...", "Correlating...", "Inferring...", "Exploring...",
                "Reviewing...", "Interpreting...", "Comprehending...", "Extrapolating...",
                "Parsing...", "Structuring...", "Resolving...", "Assembling...", "Distilling..."
            ]
            phrase = random.choice(thinking_phrases)

            use_flare = getattr(cfg.retrieval, "use_flare", False)
            if use_flare:
                from rag.generation.flare import FlareController
                
                def _flare_retrieve(flare_query: str) -> str:
                    try:
                        q_vec, e_query = self._expander.expand(flare_query, cfg, embedder)
                        d_res = self._vector_store.search_dense(q_vec, top_k=top_k_retrieval, filter_=filter_dict if filter_dict else None)
                        b_res = self._bm25.search(e_query, top_k=top_k_retrieval)
                        fused = rrf_fuse([d_res, b_res], top_k=top_k_retrieval)
                        c = rrf_to_scored_passages(fused, self._chunk_store)
                        r = rerank(flare_query, c, cfg, top_k=top_k_rerank, threshold=threshold)
                        if getattr(cfg.retrieval, "use_parent_docs", False):
                            for p in r:
                                if p.chunk.parent_id:
                                    parent_chunk = self._chunk_store.fetch_parent(p.chunk)
                                    if parent_chunk:
                                        p.chunk = parent_chunk
                        return "\n\n".join(f"---\n{p.chunk.text}" for p in r)
                    except Exception as e:
                        log.warning("FLARE retrieval failed: %s", e)
                        return ""
                        
                controller = FlareController(
                    llm=llm,
                    base_prompt=prompt,
                    retrieve_fn=_flare_retrieve,
                    max_tokens=cfg.llm.max_tokens,
                    temperature=cfg.llm.temperature,
                )
                stream_gen = controller.stream()
            else:
                stream_gen = llm.stream(
                    prompt,
                    max_tokens=cfg.llm.max_tokens,
                    temperature=cfg.llm.temperature,
                )

            first_token_data = None
            try:
                with console.status(f"[accent]{phrase}[/accent]", spinner="dots"):
                    first_token_data = next(stream_gen)
            except StopIteration:
                pass
                
            with Live(Markdown(""), console=console, refresh_per_second=15, transient=False) as live:
                if first_token_data is not None:
                    ttft_ms = (time.monotonic() - t_gen_start) * 1000
                    token_text = first_token_data[0] if isinstance(first_token_data, tuple) else first_token_data
                    full_answer += token_text
                    live.update(Markdown(full_answer))
                    
                for token_data in stream_gen:
                    token_text = token_data[0] if isinstance(token_data, tuple) else token_data
                    full_answer += token_text
                    live.update(Markdown(full_answer))
        except KeyboardInterrupt:
            full_answer += "\n\n*[Cancelled]*"
            console.print("\n[subtle]^C [Generation cancelled][/subtle]")
        except Exception as e:
            log.exception("Error during generation: %s", e)
            console.print(f"[error]✖ Generation notice:[/error] {e}")

        t_gen_ms = (time.monotonic() - t_gen_start) * 1000

        # ── 7. Citations ───────────────────────────────────────────────────────
        import re
        all_citations = build_citations(passages_used)
        
        # Extract all numbers inside square brackets, e.g., "[1]", "[2]"
        used_numbers = set(int(m) for m in re.findall(r'\[(\d+)\]', full_answer))
        
        # If the LLM declared it couldn't find an answer, explicitly wipe citations
        if "I cannot find an answer to this" in full_answer:
            used_numbers = set()
            
        citations = [c for c in all_citations if c.number in used_numbers]

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

    def _handle_llm_direct(
        self, query: str, prompt_template: str, cfg, console, start_time: float, retrieval_latency_ms: float = 0.0
    ) -> AnswerResult:
        prompt = prompt_template.format(query=query)
        
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
            import random
            thinking_phrases = [
                "Thinking...", "Analyzing...", "Understanding...", "Decoding...", 
                "Processing...", "Synthesizing...", "Evaluating...", "Investigating...",
                "Computing...", "Reasoning...", "Pondering...", "Scanning...",
                "Formulating...", "Correlating...", "Inferring...", "Exploring...",
                "Reviewing...", "Interpreting...", "Comprehending...", "Extrapolating...",
                "Parsing...", "Structuring...", "Resolving...", "Assembling...", "Distilling..."
            ]
            phrase = random.choice(thinking_phrases)
            
            stream_gen = llm.stream(prompt, max_tokens=cfg.llm.max_tokens, temperature=cfg.llm.temperature)
            
            first_token_data = None
            try:
                with console.status(f"[accent]{phrase}[/accent]", spinner="dots"):
                    first_token_data = next(stream_gen)
            except StopIteration:
                pass
                
            with Live(Markdown(""), console=console, refresh_per_second=15, transient=False) as live:
                if first_token_data is not None:
                    token_text = first_token_data[0] if isinstance(first_token_data, tuple) else first_token_data
                    full_answer += token_text
                    live.update(Markdown(full_answer))
                    
                for token_data in stream_gen:
                    token_text = token_data[0] if isinstance(token_data, tuple) else token_data
                    full_answer += token_text
                    live.update(Markdown(full_answer))
        except Exception as e:
            console.print(f"[error]Error during generation: {e}[/error]")
            
        t_gen_ms = (time.monotonic() - t_gen_start) * 1000
        t_total_ms = (time.monotonic() - start_time) * 1000
        
        log.info("LLM direct query complete: %.1f ms total (gen %.1f ms)", t_total_ms, t_gen_ms)
        
        return AnswerResult(
            text=full_answer,
            citations=[],
            passages_used=0,
            latency_ms=t_total_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=t_gen_ms,
            tier=cfg.resolved_tier,
        )

    def _get_history_context(self, full_history: list[dict]) -> list[dict]:
        """Return a rolling window of history that fits within the token budget."""
        from rag.session import get_history_for_context
        return get_history_for_context(
            history=full_history,
            token_budget=self._config.generation.context_max_tokens,
            passage_tokens=500,  # conservative estimate before passages are selected
        )

    def _build_filter(
        self,
        file_filter: str | None,
        type_filter: str | None,
        page_range: str | None,
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
