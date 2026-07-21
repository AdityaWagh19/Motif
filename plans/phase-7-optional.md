# Phase 7 — Optional Enhancements

> **Status:** Not planned until Phase 6 complete  
> **Prerequisite:** Phase 6 complete (all NFRs met, RAGAS baseline saved)  
> **Model downloads required:** Varies per feature (noted per item)  
> **Estimated scope:** Independent features — implement in any order

---

## Objective

Optional features that improve specific use cases but are not required for the
core RAG system. These are implemented based on user feedback after Phase 6.
Each item is self-contained and does not gate any other item.

---

## Features

### 7-A: RAPTOR Hierarchical Indexing

**Use case:** Document corpora with > 500 pages where high-level conceptual
questions perform poorly on chunk-level retrieval.

**What it does:**
1. After initial ingestion, cluster all chunks using k-means (k = √n)
2. For each cluster, prompt the LLM to write a 1-paragraph "summary chunk"
3. Index summary chunks alongside regular chunks (different Qdrant payload field: `is_summary=True`)
4. At query time: retrieve from both levels, merge, deduplicate

**Files to create:**
- `rag/ingestion/raptor.py` — clustering + summary generation
- Update `rag/pipeline.py` — add `use_raptor` flag to `answer()`

**When to enable:**
```toml
[retrieval]
use_raptor = true   # only if > 500 pages indexed
```

**Validation:**
- Run RAGAS on conceptual questions: e.g., "What is the overall thesis of this corpus?"
- RAPTOR should improve answer_relevancy by ≥ 3% on high-level questions

---

### 7-B: Parent-Document Retrieval

**Use case:** Chunks are short (e.g., 128 tokens) for precise retrieval, but the
LLM needs more context to give a complete answer.

**What it does:**
- At ingestion: store both "child" chunks (128 tokens) and "parent" chunks (512 tokens)
  in ChunkStore (parent_id field links them)
- At retrieval: retrieve by child chunk (precision), expand to parent chunk (context)
- Pass parent chunk text to LLM, cite child chunk location

**Files to create/modify:**
- Update `rag/types.py` — add `parent_id: Optional[str] = None` to Chunk
- Update `rag/ingestion/chunker.py` — add `ParentChunker`
- Update `rag/storage/chunk_store.py` — add `fetch_parent()` method
- Update `rag/pipeline.py` — add `use_parent_docs` flag

**Cost:** Storage doubles (child + parent chunks). Only recommended if
retrieval recall is measured to be insufficient.

---

### 7-C: FLARE Iterative Retrieval

**Use case:** Complex multi-hop questions (e.g., "Compare the methods in
document A with the results in document B") that require multiple retrieval steps.

**What it does:**
- During generation, when the LLM is about to generate an uncertain token
  (detected via low logit probability), pause and retrieve fresh context
- Re-inject retrieved passages and continue generation

**Blocker:** Requires token-level logit access from llama-cpp-python.
As of Phase 7, verify if `Llama.__call__` exposes `logprobs` in the output.
If available: implement. If not: defer.

**Files to create:**
- `rag/generation/flare.py` — iterative retrieval controller

**Validation:** Run RAGAS on a multi-hop eval set (requires creating one).
FLARE should improve faithfulness on multi-hop by ≥ 5%.
