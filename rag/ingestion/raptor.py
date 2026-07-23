"""
rag/ingestion/raptor.py — Phase 7-A RAPTOR Hierarchical Indexing.

Implements clustering and summarization of chunks to provide global context.
Because Motif avoids heavy dependencies (no scikit-learn), we implement a 
fast, lightweight k-means algorithm in NumPy.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rich.console import Console

    from rag.config import RAGConfig

log = logging.getLogger(__name__)

RAPTOR_PROMPT = """\
Summarize the core themes of these chunks in exactly one short sentence.

Excerpts:
{excerpts}

Summary:"""


def _kmeans(vectors: np.ndarray, k: int, max_iters: int = 50) -> np.ndarray:
    """
    Basic k-means clustering implementation in pure NumPy.
    Returns an array of cluster assignments (shape: N,).
    """
    n, dim = vectors.shape
    
    # Initialize centroids randomly from data points
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(n, k, replace=False)
    centroids = vectors[indices].copy()
    
    assignments = np.zeros(n, dtype=int)
    
    for _ in range(max_iters):
        # Broadcast vectors (N, 1, D) and centroids (1, K, D)
        # Compute distances: (N, K)
        dists = np.linalg.norm(vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        new_assignments = np.argmin(dists, axis=1)
        
        if np.array_equal(assignments, new_assignments):
            break
            
        assignments = new_assignments
        
        # Update centroids
        for i in range(k):
            cluster_pts = vectors[assignments == i]
            if len(cluster_pts) > 0:
                centroids[i] = cluster_pts.mean(axis=0)
                
    return assignments


def build_raptor_summaries(
    config: RAGConfig, 
    console: Console | None = None,
    chunk_store=None,
    bm25=None,
    vector_store=None,
    embedder=None
) -> None:
    """
    Run the RAPTOR process over all chunks in the store.
    1. Fetches all chunks and their vectors.
    2. Clusters them using k-means.
    3. Prompts the LLM to summarize each cluster.
    4. Ingests the new summary chunks back into the indices.
    """
    from rag.ingestion import _chunk_to_payload
    from rag.models.model_manager import get_model_manager
    from rag.retrieval.bm25_index import BM25Index
    from rag.retrieval.vector_store import VectorStore
    from rag.storage.chunk_store import ChunkStore
    from rag.types import Chunk

    if chunk_store is None:
        chunk_store = ChunkStore(config)
        
    rows = chunk_store._conn.execute(
        "SELECT id, text, source, filename FROM chunks WHERE parent_id IS NULL AND source_type != 'raptor'"
    ).fetchall()
    
    if not rows:
        if console:
            console.print("[yellow]No chunks available for hierarchical summarization.[/yellow]")
        return
        
    k = int(np.sqrt(len(rows)))
    if k < 2:
        return
        
    model_manager = get_model_manager()
    if embedder is None:
        embedder = model_manager.get_embedder(config)
    llm = model_manager.get_llm(config)
    
    # 1. Embed all chunks
    texts = [r[1] for r in rows]
    vectors = embedder.encode_batch(texts, prefix="search_document: ")
    
    # 2. Cluster
    assignments = _kmeans(vectors, k)
    
    # Group rows by cluster
    clusters: list[list[tuple]] = [[] for _ in range(k)]
    for i, row in enumerate(rows):
        cluster_idx = assignments[i]
        clusters[cluster_idx].append(row)
        
    summary_chunks: list[Chunk] = []
    
    if console:
        console.print(f"[dim]Generating {k} hierarchical summary chunks...[/dim]")
        
    # 3. Generate summaries
    for i, cluster in enumerate(clusters):
        if not cluster:
            continue
            
        # Combine texts, capping at ~1000 tokens to avoid LLM context overflow
        # Very rough approximation: 1 token ~ 4 chars
        MAX_CHARS = 4000
        combined_text = ""
        for row in cluster:
            if len(combined_text) > MAX_CHARS:
                break
            combined_text += f"---\n{row[1]}\n"
            
        prompt = RAPTOR_PROMPT.format(excerpts=combined_text)
        
        try:
            summary = llm.generate(prompt, max_tokens=50, temperature=0.1)
        except Exception as exc:
            log.warning("Hierarchical summary LLM generation failed for cluster %d: %s", i, exc)
            continue
            
        if not summary.strip():
            continue
            
        # Create a summary chunk
        # We inherit source/filename from the first chunk in the cluster for provenance
        base_row = cluster[0]
        chunk = Chunk(
            id=str(uuid.uuid4()),
            text=f"[SUMMARY] {summary.strip()}",
            source=base_row[2],
            filename=base_row[3],
            source_type="raptor",
            indexed_at=datetime.now(UTC).isoformat()
        )
        summary_chunks.append(chunk)
        
    if not summary_chunks:
        return
        
    # 4. Ingest summaries
    if console:
        console.print(f"[green]OK[/green] Indexing {len(summary_chunks)} summary chunks.")
        
    summary_vectors = embedder.encode_batch(
        [c.text for c in summary_chunks], 
        prefix="search_document: "
    )
    
    if bm25 is None:
        bm25 = BM25Index(config)
    if vector_store is None:
        vector_store = VectorStore(config)
    
    chunk_store.insert_batch(summary_chunks)
    bm25.add_batch(summary_chunks)
    
    payloads = []
    for c in summary_chunks:
        p = _chunk_to_payload(c)
        p["is_summary"] = True
        payloads.append(p)
        
    vector_store.upsert_batch(
        [c.id for c in summary_chunks],
        summary_vectors,
        payloads
    )
