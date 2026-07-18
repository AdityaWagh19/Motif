# Phase 3 — Query Pipeline

> **Status:** Not started  
> **Prerequisite:** Phase 2 complete (`/ingest` populates index, `/status` shows real counts)  
> **Model downloads required:** LLM (2.2 GB T1 / 4.2 GB T2) + reranker MiniLM (~134 MB)  
> **Estimated scope:** 7 files created/modified, ~1,000 lines of implementation

---

## Objective

Build the complete end-to-end query pipeline. By the end of this phase, typing
any question in the REPL produces a streamed, grounded answer with numbered
citations. History follow-up works. The system refuses to answer questions that
are not grounded in the indexed documents.

This is the first time the system is usable as an actual RAG application.

**Query flow:**

```
User types query
       │
       ▼
QueryExpander.route()    →  decides: use HyDE or direct embed? (Phase 3: direct only)
       │
       ▼
Embedder.encode()        →  query vector (768-dim)
       │
       ├──► VectorStore.search_dense()  →  dense candidates
       ├──► BM25Index.search()          →  lexical candidates
       │
       ▼
rrf_fuse()               →  merged, re-scored candidates
       │
       ▼
CrossEncoder.rerank()    →  top-K passages by relevance score
       │
       ▼
ContextBuilder.build()   →  RAG context string (anti-middle order + history)
       │
       ▼
LLMClient.stream()       →  token-by-token answer to terminal
       │
       ▼
Citation formatter       →  numbered source list printed below answer
       │
       ▼
Session.add_turn()       →  history updated
```

---

## Scope

**In scope:**
- `rag/retrieval/fusion.py` — Reciprocal Rank Fusion (RRF)
- `rag/retrieval/expander.py` — routing heuristic (Phase 3: direct embed only)
- `rag/models/reranker.py` — full ONNX cross-encoder implementation
- `rag/reranking/cross_encoder.py` — reranking algorithm (calls ModelManager)
- `rag/generation/prompts.py` — RAG_PROMPT, HISTORY_SYSTEM_PROMPT, REFUSAL_PROMPT
- `rag/generation/context_builder.py` — assembly, anti-middle ordering, history injection
- `rag/generation/llm_client.py` — llama-cpp-python streaming wrapper
- `rag/pipeline.py` — `QueryPipeline.answer()` full implementation
- Update `rag/cli.py` — route plain-text queries to `QueryPipeline`, not the stub
- `tests/unit/test_fusion.py`
- `tests/unit/test_context_builder.py`
- `tests/unit/test_citation.py`
- `tests/integration/test_query.py`
- `tests/integration/test_history.py`

**Out of scope:**
- HyDE query expansion (Phase 4)
- Sparse/SPLADE vector search (Phase 4)
- Adjacent chunk merging / extractive compression (Phase 4)
- Metadata filters from inline modifiers (Phase 4)

---

## Model Download Requirement

```bash
# Download LLM and reranker
motif setup --tier T1   # Phi-3.5-mini + nomic-embed + MiniLM-L12 reranker
# OR for T2/T3:
motif setup --tier T2   # Qwen2.5-7B + nomic-embed + MiniLM-L12 reranker
```

---

## File Specifications

### `rag/retrieval/fusion.py`

**Algorithm:** Reciprocal Rank Fusion (RRF) with parameter k=60.

RRF score for a chunk `d` across ranking lists `L₁, L₂, ..., Lₙ`:
```
RRF(d) = Σᵢ 1 / (k + rank_i(d))
```
where `rank_i(d)` is the 1-based rank of `d` in list `i` (set to ∞ if absent).

```python
def rrf_fuse(
    ranked_lists: List[List[Tuple[str, float]]],
    top_k: int = 25,
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked lists of (chunk_id, score) tuples using RRF.

    Args:
        ranked_lists: Each inner list is one ranking (dense, sparse, bm25).
                      Assumed to be sorted descending by score.
        top_k:        Number of results to return.
        k:            RRF parameter. Higher k reduces impact of rank differences.

    Returns:
        List of (chunk_id, rrf_score) sorted descending by RRF score.
        Top top_k entries only.
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
```

```python
def rrf_to_scored_passages(
    fused: List[Tuple[str, float]],
    chunk_store: "ChunkStore",
) -> List[ScoredPassage]:
    """
    Fetch Chunk objects for fused IDs and wrap them in ScoredPassage.
    Drops any chunk_ids that no longer exist in ChunkStore (e.g., mid-sync deletion).
    """
    chunk_ids = [cid for cid, _ in fused]
    score_map = dict(fused)
    chunks = chunk_store.fetch_batch(chunk_ids)
    return [
        ScoredPassage(chunk=c, score=score_map[c.id], retrieval_method="fused")
        for c in chunks
        if c.id in score_map
    ]
```

---

### `rag/retrieval/expander.py`

**Phase 3 scope:** Routing heuristic only. HyDE is planned for Phase 4.

```python
def should_use_hyde(query: str, config: RAGConfig) -> bool:
    """
    Decide whether to use HyDE for this query.

    Phase 3: always return False.
    Phase 4: implement heuristic — return True if:
      - config.retrieval.query_expansion == "hyde"
      - AND query length > 10 words (factual short queries benefit less from HyDE)
      - AND query tier is T2 or T3 (T1 latency budget too tight)
    """
    return False


class QueryExpander:
    """
    Phase 3: thin wrapper — just encodes the query directly.
    Phase 4: implements HyDE (generate hypothetical answer, embed that).
    """

    def expand(
        self,
        query: str,
        config: RAGConfig,
        embedder: "Embedder",
    ) -> Tuple[np.ndarray, str]:
        """
        Expand a query for retrieval.

        Returns:
            (query_vector, effective_query_text)
            Phase 3: direct embedding of the original query.
            Phase 4: HyDE generates a hypothetical answer, embeds that instead.
        """
        vector = embedder.encode(query, prefix="search_query: ")
        return vector, query
```

---

### `rag/models/reranker.py` (full implementation)

**Replaces the Phase 0 skeleton.**

**Models:**
- T1/T2: `cross-encoder/ms-marco-MiniLM-L-12-v2` ONNX
- T3: `BAAI/bge-reranker-base` ONNX

**Model directory structure:**
```
models/MiniLM-L12-v2/
    tokenizer.json
    tokenizer_config.json
    model.onnx   (or model_quantized.onnx if INT8 converted)
```

```python
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MAX_SEQ_LEN = 512

class Reranker:

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._session: Optional[ort.InferenceSession] = None
        self._tokenizer: Optional[Tokenizer] = None

    def _load(self) -> None:
        # Try model_quantized.onnx first, then model.onnx
        for name in ["model_quantized.onnx", "model.onnx"]:
            onnx_path = self._model_dir / name
            if onnx_path.exists():
                break
        else:
            raise FileNotFoundError(f"No ONNX model found in {self._model_dir}")

        tok_path = self._model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(onnx_path), sess_opts, ["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def score(self, query: str, passages: List[str]) -> np.ndarray:
        """
        Score (query, passage) pairs.

        Returns float32 array of shape (len(passages),).
        Higher = more relevant.

        Implementation:
        1. For each passage: form pair text = query + " [SEP] " + passage
        2. Tokenize as sequence pairs with [CLS] query [SEP] passage [SEP]
        3. Run ONNX inference → logits shape (batch, 1) or (batch, 2)
        4. If shape (batch, 2): softmax and take column 1 (relevant class)
           If shape (batch, 1): raw logit (higher = more relevant)
        5. Return as float32 array
        """
        if self._session is None:
            raise RuntimeError("Reranker not loaded. Call _load() first.")

        # Process in batches of 16 to avoid OOM on long passages
        BATCH = 16
        all_scores = []

        for i in range(0, len(passages), BATCH):
            batch_passages = passages[i : i + BATCH]
            # Encode as pairs: tokenizer handles [CLS] query [SEP] passage [SEP]
            encodings = self._tokenizer.encode_batch(
                [[query, p] for p in batch_passages]
            )
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )
            logits = outputs[0]  # (batch, 1) or (batch, 2)
            if logits.shape[-1] == 2:
                # Binary classification: softmax and take positive class
                exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
                probs = exp / exp.sum(axis=-1, keepdims=True)
                scores = probs[:, 1]
            else:
                scores = logits[:, 0]

            all_scores.append(scores)

        return np.concatenate(all_scores).astype(np.float32)

    def is_loaded(self) -> bool:
        return self._session is not None

    def unload(self) -> None:
        self._session = None
        self._tokenizer = None
```

---

### `rag/reranking/cross_encoder.py`

**Purpose:** Reranking algorithm. Does NOT own model loading — uses ModelManager.

```python
def rerank(
    query: str,
    passages: List[ScoredPassage],
    config: RAGConfig,
    top_k: int = 5,
    threshold: float = 0.3,
) -> List[ScoredPassage]:
    """
    Rerank a list of ScoredPassage objects using the cross-encoder.

    Algorithm:
    1. Get reranker from ModelManager
    2. Score all passages: scores = reranker.score(query, [p.chunk.text for p in passages])
    3. Filter: drop passages with score < threshold
    4. Sort descending by score
    5. Return top_k, with each ScoredPassage.score replaced by cross-encoder score
       and .retrieval_method = "reranked"

    Args:
        query:    The user query.
        passages: Pre-retrieved candidates from RRF fusion.
        config:   For ModelManager and threshold config.
        top_k:    Max passages to return.
        threshold: Min relevance score (from config.retrieval.relevance_threshold).

    Returns:
        Reranked list, length <= top_k. May be empty if no passage exceeds threshold.
    """
    if not passages:
        return []

    reranker = get_model_manager().get_reranker(config)
    texts = [p.chunk.text for p in passages]
    scores = reranker.score(query, texts)

    reranked = []
    for passage, score in zip(passages, scores):
        if float(score) >= threshold:
            reranked.append(ScoredPassage(
                chunk=passage.chunk,
                score=float(score),
                retrieval_method="reranked",
            ))

    reranked.sort(key=lambda p: p.score, reverse=True)
    return reranked[:top_k]
```

---

### `rag/generation/prompts.py`

```python
# System prompt injected at the start of every query.
# {context} is replaced by the assembled retrieved passages.
# {query} is replaced by the user's question.
RAG_PROMPT = """\
You are a precise research assistant. Answer the question using ONLY the \
information in the provided context passages. Do not speculate or use outside \
knowledge.

If the answer is not present in the context, say exactly:
"I cannot find an answer to this in the available documents."

Cite each piece of information with its passage number in square brackets, \
e.g. [1], [2]. If multiple passages support a point, cite all of them, e.g. [1][3].

Context:
{context}

Question: {query}
Answer:"""


# System prompt used when conversation history is present.
# {history} is replaced by prior turns formatted as User:/Assistant: blocks.
HISTORY_SYSTEM_PROMPT = """\
You are a precise research assistant continuing a conversation. Prior context:

{history}

Answer the current question using ONLY the provided document passages. \
Maintain consistency with your previous answers. Cite sources with [N]."""


# Prompt used by HyDE (Phase 4) to generate a hypothetical answer.
HYDE_PROMPT = """\
Write a short, factual passage (2-3 sentences) that would answer the following \
question if it existed in a research document. Focus on the most likely answer.

Question: {query}
Passage:"""


def format_context(passages: List["ScoredPassage"]) -> str:
    """
    Format a list of ScoredPassage objects into a numbered context string.

    Format:
        [1] Source: filename.pdf (p.3 — Section Title)
        <passage text>

        [2] Source: notes.md (Introduction)
        <passage text>
        ...
    """
    lines = []
    for i, p in enumerate(passages, start=1):
        chunk = p.chunk
        loc_parts = []
        if chunk.page is not None:
            loc_parts.append(f"p.{chunk.page}")
        if chunk.section:
            loc_parts.append(chunk.section)
        loc = f" ({', '.join(loc_parts)})" if loc_parts else ""
        lines.append(f"[{i}] Source: {chunk.filename}{loc}")
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def format_history(history: List[dict]) -> str:
    """
    Format conversation history as alternating User/Assistant blocks.

    history is a list of {"role": "user"|"assistant", "content": str}.
    """
    parts = []
    for turn in history:
        prefix = "User" if turn["role"] == "user" else "Assistant"
        parts.append(f"{prefix}: {turn['content']}")
    return "\n\n".join(parts)


def build_prompt(
    query: str,
    passages: List["ScoredPassage"],
    history: List[dict],
) -> str:
    """
    Assemble the final LLM prompt.

    If history is non-empty: use HISTORY_SYSTEM_PROMPT + RAG context.
    If no history: use RAG_PROMPT only.
    """
    context = format_context(passages)

    if history:
        history_text = format_history(history)
        system = HISTORY_SYSTEM_PROMPT.format(history=history_text)
        return f"{system}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    else:
        return RAG_PROMPT.format(context=context, query=query)


def build_citations(passages: List["ScoredPassage"]) -> List["Citation"]:
    """
    Build Citation objects from the final reranked passages.
    Citation numbers correspond to [N] in the generated answer.
    """
    from rag.types import Citation
    citations = []
    for i, p in enumerate(passages, start=1):
        c = p.chunk
        citations.append(Citation(
            number=i,
            source_type=c.source_type,
            filepath=c.source,
            filename=c.filename,
            page=c.page,
            section=c.section,
            start_time=c.start_time,
            end_time=c.end_time,
            relevance_score=p.score,
            excerpt=c.text[:150],
        ))
    return citations
```

---

### `rag/generation/context_builder.py`

**Purpose:** Assemble retrieved passages into an LLM-ready context string.
Applies "anti-middle ordering" and token budget management.

**Anti-middle ordering (Lost-in-the-Middle mitigation):**
LLMs attend more strongly to the beginning and end of long contexts. To counteract
this, the most relevant passage goes first, the second most relevant goes last,
and remaining passages fill the middle.

```python
class ContextBuilder:

    def build(
        self,
        passages: List[ScoredPassage],
        query: str,
        history: List[dict],
        config: RAGConfig,
    ) -> Tuple[str, List[ScoredPassage]]:
        """
        Build the final LLM prompt and return the passages actually used.

        Steps:
        1. Apply token budget: drop passages until total fits in context_max_tokens
        2. Apply anti-middle ordering: reorder passages for best attention coverage
        3. Build the prompt string
        4. Return (prompt_str, ordered_passages_used)

        Token budget:
        - Total context budget: config.generation.context_max_tokens
        - History tokens: estimate from len(format_history(history)) // 4
        - Remaining: for passages
        - Prompt overhead: ~200 tokens (RAG_PROMPT template)
        """
        if not passages:
            return "", []

        # Step 1: Token budget — drop lowest-scoring passages first
        budget_words = config.generation.context_max_tokens * 3 // 4  # word approximation
        history_words = sum(len(t["content"].split()) for t in history)
        available_words = budget_words - history_words - 200  # 200 for prompt overhead

        selected = []
        used_words = 0
        for p in passages:  # already sorted by score desc
            words = len(p.chunk.text.split())
            if used_words + words <= available_words:
                selected.append(p)
                used_words += words
            else:
                break

        if not selected:
            selected = [passages[0]]  # always include at least one passage

        # Step 2: Anti-middle ordering
        ordered = _anti_middle_order(selected)

        # Step 3: Build prompt
        prompt = build_prompt(query, ordered, history)

        return prompt, ordered

def _anti_middle_order(passages: List[ScoredPassage]) -> List[ScoredPassage]:
    """
    Reorder passages so highest-scoring is first, second-highest is last,
    remaining fill middle positions.

    Example for 5 passages scored [1, 2, 3, 4, 5] (desc):
    Result order: [1, 3, 4, 5, 2]  — 1st best at start, 2nd best at end
    """
    if len(passages) <= 2:
        return passages

    sorted_desc = sorted(passages, key=lambda p: p.score, reverse=True)
    result = [None] * len(sorted_desc)
    result[0] = sorted_desc[0]   # best → first position
    result[-1] = sorted_desc[1]  # second best → last position

    middle_idx = 1
    for i in range(2, len(sorted_desc)):
        result[middle_idx] = sorted_desc[i]
        middle_idx += 1

    return [p for p in result if p is not None]
```

---

### `rag/generation/llm_client.py`

**Backend:** `llama-cpp-python` (`Llama` class)

```python
from llama_cpp import Llama

STOP_TOKENS = ["</s>", "<|end|>", "<|im_end|>", "[/INST]", "User:", "\n\nUser"]

class LLMClient:

    def __init__(self, model_path: Path, config: RAGConfig) -> None:
        self._model_path = model_path
        self._config = config
        self._llm: Optional[Llama] = None

    def _load(self) -> None:
        cfg = self._config.llm
        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=cfg.ctx_size,
            n_gpu_layers=cfg.n_gpu_layers,
            n_threads=cfg.threads,
            verbose=False,          # suppress llama.cpp debug output
            use_mlock=False,        # don't pin memory
            use_mmap=True,          # memory-map the model file
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.1,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream the LLM response token by token.

        Yields each token text string as it is generated.
        Stops at max_tokens or stop sequences.

        Usage in CLI:
            for token in client.stream(prompt, max_tokens=400):
                print(token, end="", flush=True)
        """
        if self._llm is None:
            raise RuntimeError("LLMClient not loaded. Call _load() first.")

        stop_seqs = (stop or []) + STOP_TOKENS
        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_seqs,
            stream=True,
            echo=False,
        )
        for chunk in output:
            token_text = chunk["choices"][0]["text"]
            if token_text:
                yield token_text

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.1,
    ) -> str:
        """
        Non-streaming generation. Returns the full response string.
        Used for HyDE (Phase 4) and eval (Phase 6).
        """
        return "".join(self.stream(prompt, max_tokens, temperature))

    def unload(self) -> None:
        del self._llm
        self._llm = None
        import gc; gc.collect()
```

---

### `rag/pipeline.py`

**The central coordinator. Contains NO business logic — delegates everything.**

```python
class QueryPipeline:

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._chunk_store = ChunkStore(config)
        self._bm25 = BM25Index(config)
        self._vector_store = VectorStore(config)
        self._context_builder = ContextBuilder()
        self._expander = QueryExpander()

    def answer(
        self,
        query: str,
        history: List[dict],
        file_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
        page_range: Optional[str] = None,   # "MIN-MAX" string
        use_hyde: bool = False,             # ignored in Phase 3
        show_sources: bool = True,
    ) -> AnswerResult:
        """
        Run the full RAG pipeline for one query.

        1. Validate index is not empty
        2. Expand query (Phase 3: direct embed)
        3. Retrieve: dense + BM25
        4. Fuse: RRF
        5. Rerank: cross-encoder
        6. Build context and prompt
        7. Stream answer to terminal
        8. Format citations
        9. Return AnswerResult

        Filtering: if file_filter or type_filter given, pass as Qdrant filter.
        page_range: parse "20-40" → {"page_min": 20, "page_max": 40}
        """
        from rich.console import Console
        console = Console()
        t_start = time.monotonic()

        # 0. Check index is populated
        if self._chunk_store.count() == 0:
            console.print("[yellow]No documents indexed.[/yellow] Run /ingest first.")
            return AnswerResult(
                text="No documents are indexed. Run /ingest to add documents.",
                citations=[],
                passages_used=0,
            )

        # 1. Get history context for token budget
        history_context = self._get_history_context(history)

        # 2. Expand query → get query vector
        cfg = self._config
        embedder = get_model_manager().get_embedder(cfg)
        query_vector, effective_query = self._expander.expand(query, cfg, embedder)

        t_retrieval_start = time.monotonic()

        # 3. Build metadata filter if provided
        filter_dict = self._build_filter(file_filter, type_filter, page_range)

        # 4. Retrieve
        top_k_retrieval = cfg.retrieval.top_k_retrieval
        dense_results = self._vector_store.search_dense(
            query_vector, top_k=top_k_retrieval, filter_=filter_dict or None
        )
        bm25_results = self._bm25.search(effective_query, top_k=top_k_retrieval)

        # 5. RRF fusion
        fused = rrf_fuse(
            [dense_results, bm25_results],
            top_k=top_k_retrieval,
        )
        candidates = rrf_to_scored_passages(fused, self._chunk_store)

        t_retrieval_ms = (time.monotonic() - t_retrieval_start) * 1000

        if not candidates:
            console.print("[yellow]No relevant passages found for your query.[/yellow]")
            return AnswerResult(
                text="I cannot find an answer to this in the available documents.",
                citations=[],
                passages_used=0,
            )

        # 6. Rerank
        top_k_rerank = cfg.retrieval.top_k_rerank
        threshold = cfg.retrieval.relevance_threshold
        reranked = rerank(query, candidates, cfg, top_k=top_k_rerank, threshold=threshold)

        if not reranked:
            # All passages below threshold — model thinks query is off-topic
            console.print("[dim]No passages met the relevance threshold.[/dim]")
            return AnswerResult(
                text="I cannot find an answer to this in the available documents.",
                citations=[],
                passages_used=0,
            )

        # 7. Build context and prompt
        prompt, passages_used = self._context_builder.build(
            reranked, query, history_context, cfg
        )

        # 8. Stream answer
        t_gen_start = time.monotonic()
        llm = get_model_manager().get_llm(cfg)

        console.print()
        full_answer = ""
        for token in llm.stream(prompt, max_tokens=cfg.llm.max_tokens, temperature=cfg.llm.temperature):
            print(token, end="", flush=True)
            full_answer += token
        print()  # newline after streamed answer

        t_gen_ms = (time.monotonic() - t_gen_start) * 1000

        # 9. Citations
        citations = build_citations(passages_used)
        if show_sources and citations:
            console.print()
            console.print("[dim]Sources:[/dim]")
            for c in citations:
                console.print(f"  [dim]{c.format()}[/dim]")

        t_total_ms = (time.monotonic() - t_start) * 1000

        return AnswerResult(
            text=full_answer,
            citations=citations,
            passages_used=len(passages_used),
            latency_ms=t_total_ms,
            retrieval_latency_ms=t_retrieval_ms,
            generation_latency_ms=t_gen_ms,
            tier=cfg.resolved_tier,
        )

    def _get_history_context(self, full_history: List[dict]) -> List[dict]:
        """Return the rolling window of history within token budget."""
        from rag.session import Session
        # Use a temporary Session to compute the rolling window
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
        f = {}
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
                pass
        return f
```

---

### Update `rag/cli.py`

In `_handle_query()`, replace the `ImportError` stub branch:

```python
from rag.pipeline import QueryPipeline

pipeline = QueryPipeline(config)
answer = pipeline.answer(
    query=query,
    history=history_context,
    file_filter=modifiers.get("file"),
    type_filter=modifiers.get("type"),
    page_range=modifiers.get("pages"),
    use_hyde=not modifiers.get("no-hyde", False),
    show_sources=not modifiers.get("no-sources", False),
)
if answer.text:
    session.add_turn(query, answer.text)
```

---

## Test Specifications

### `tests/unit/test_fusion.py`

```
test_rrf_prefers_items_appearing_in_all_lists:
    list1 = [("a", 1.0), ("b", 0.8), ("c", 0.5)]
    list2 = [("b", 0.9), ("c", 0.7), ("d", 0.4)]
    list3 = [("c", 0.6), ("b", 0.5), ("e", 0.3)]
    fused = rrf_fuse([list1, list2, list3], top_k=5)
    ids = [x[0] for x in fused]
    # "b" and "c" appear in all three lists → should rank above "a" and "d"
    assert ids.index("b") < ids.index("a")  or ids.index("c") < ids.index("a")

test_rrf_top_k_limits_results:
    lists = [[("a", 1.0), ("b", 0.5), ("c", 0.3)]]
    fused = rrf_fuse(lists, top_k=2)
    assert len(fused) == 2

test_rrf_single_list_preserves_order:
    ranked = [(str(i), 1.0 - i*0.1) for i in range(10)]
    fused = rrf_fuse([ranked], top_k=10)
    ids = [x[0] for x in fused]
    assert ids == [x[0] for x in ranked]  # original order preserved

test_rrf_empty_list:
    fused = rrf_fuse([], top_k=10)
    assert fused == []

test_rrf_scores_decrease:
    ranked = [("a", 1.0), ("b", 0.8), ("c", 0.5)]
    fused = rrf_fuse([ranked], top_k=3)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
```

### `tests/unit/test_context_builder.py`

```
test_anti_middle_order_best_first():
    passages = [ScoredPassage(chunk=Chunk(id=str(i), text=f"text{i}", ...), score=5-i, ...)
                for i in range(5)]
    ordered = _anti_middle_order(passages)
    assert ordered[0].score == 5   # best passage first
    assert ordered[-1].score == 4  # second best last

test_context_builder_respects_token_budget():
    # 10 passages × 200 words each. Budget = 512 tokens ≈ 384 words.
    # Only ~1-2 passages should fit.
    long_passages = [ScoredPassage(chunk=Chunk(id=str(i),
        text=" ".join(["word"] * 200), ...), score=10-i, ...)
        for i in range(10)]
    prompt, used = builder.build(long_passages, "query", [], config)
    assert len(used) <= 3  # budget enforced

test_context_builder_always_includes_one_passage():
    # Even with zero budget, at least one passage included
    very_long = ScoredPassage(chunk=Chunk(id="1", text=" ".join(["word"]*5000), ...), score=1.0, ...)
    config.generation.context_max_tokens = 100
    prompt, used = builder.build([very_long], "q", [], config)
    assert len(used) == 1
```

### `tests/unit/test_citation.py`

```
test_citation_format_pdf():
    c = Citation(number=1, source_type="pdf", filepath="/docs/thesis.pdf",
                 filename="thesis.pdf", page=42, section="Methods")
    assert c.format() == "[1] thesis.pdf (p.42, Methods)"

test_citation_format_md():
    c = Citation(number=2, source_type="md", filepath="/notes/ideas.md",
                 filename="ideas.md", section="Introduction")
    assert c.format() == "[2] ideas.md (Introduction)"

test_citation_format_audio():
    c = Citation(number=3, source_type="audio", filepath="/rec/talk.mp3",
                 filename="talk.mp3", start_time=125.0, end_time=137.5)
    assert c.format() == "[3] talk.mp3 @ 02:05–02:17"
```

### `tests/integration/test_query.py`

```python
@pytest.mark.slow
def test_end_to_end_ingest_and_query(minimal_config, sample_md):
    # Ingest
    ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
    assert ChunkStore(minimal_config).count() >= 1

    # Query
    pipeline = QueryPipeline(minimal_config)
    result = pipeline.answer(
        query="What does this document discuss?",
        history=[],
    )
    assert result.text
    assert len(result.text) > 20
    # Should not be a hallucination refusal for a query about the doc
    assert "cannot find" not in result.text.lower()

@pytest.mark.slow
def test_query_unanswerable_returns_refusal(minimal_config, sample_md):
    ingest_path(sample_md, config=minimal_config, recursive=False, console=None)
    pipeline = QueryPipeline(minimal_config)
    result = pipeline.answer(
        query="What is the molecular weight of caffeine?",
        history=[],
    )
    # Query about chemistry — not in our test doc
    assert "cannot find" in result.text.lower() or len(result.citations) == 0
```

### `tests/integration/test_history.py`

```python
def test_session_history_persists_across_restart(minimal_config, tmp_db_root):
    session1 = Session(minimal_config)
    session1.add_turn("first query", "first answer")
    session1.save()

    session2 = Session(minimal_config)
    loaded = session2.load()
    assert loaded is True
    assert session2.turn_count == 1
    assert session2.last_query == "first query"

def test_session_clear_deletes_file(minimal_config, tmp_db_root):
    session = Session(minimal_config)
    session.add_turn("q", "a")
    session.save()
    assert session.history_path.exists()
    session.clear()
    assert not session.history_path.exists()
    assert session.turn_count == 0

def test_session_new_archives(minimal_config, tmp_db_root):
    session = Session(minimal_config)
    session.add_turn("q1", "a1")
    session.save()
    archive_path = session.new()
    assert archive_path is not None
    assert archive_path.exists()
    assert session.turn_count == 0
```

---

## Validation Checklist

```bash
# 1. Module imports
python -c "from rag.pipeline import QueryPipeline; print('Pipeline OK')"
python -c "from rag.generation.llm_client import LLMClient; print('LLM OK')"
python -c "from rag.generation.context_builder import ContextBuilder; print('ContextBuilder OK')"
python -c "from rag.retrieval.fusion import rrf_fuse; print('RRF OK')"

# 2. Unit tests (no model needed)
pytest tests/unit/test_fusion.py tests/unit/test_context_builder.py tests/unit/test_citation.py -v

# 3. History integration tests (no model needed)
pytest tests/integration/test_history.py -v

# 4. Full end-to-end (needs LLM + embedder + reranker models)
pytest tests/integration/test_query.py -v -m slow

# 5. Manual REPL verification (must pass ALL items)
# motif
# Expected welcome screen: tier, model, index stats shown
#
# /ingest ./project-context -r
# Expected: "Ingestion complete. Files: 8  Chunks added: N  Skipped: 0"
#
# What does the context.md file describe?
# Expected: streamed answer referencing context.md content, [1] citation
#
# Expand on that.
# Expected: answer referencing prior context (history follow-up)
#
# What is the boiling point of water?
# Expected: "I cannot find an answer..." (off-topic refusal)
#
# exit
# Expected: "Session saved. Goodbye."
#
# motif [second launch]
# Expected: welcome screen shows "Resuming previous session — 2 exchanges"

# 6. Manual accuracy check (20 questions on a real corpus)
# Run 20 questions where you know the answer from indexed documents.
# Record: correct (grounded answer), wrong (hallucination), refusal (unanswerable treated correctly)
# Target: ≥ 70% correct / (correct + wrong)
# Record result in progress.md Metrics Snapshots
```

---

## Post-Phase Documentation Updates

**`project-context/progress.md`:**
- Mark all Phase 1 tasks ✅ (entire Foundation phase complete)
- Update Phase Status Overview: Phase 1 → ✅ Done with completion date
- Add Metrics Snapshot row: Phase 1 complete, manual accuracy ≥ 70%, P95 latency

**`project-context/tests.md`:**
- Mark RET-01, RET-02, RET-03 (dense, BM25, fusion) ✅
- Mark GEN-01, GEN-02, GEN-03 (context builder, prompt, generation) ✅
- Mark INT-01, INT-02 (end-to-end ingest+query, history persistence) ✅

**`project-context/progress.md` Deferred Decisions Log:**
- Update "HyDE vs multi-query" row: revisit at Phase 4 ✓ (Phase 3 establishes baseline)
