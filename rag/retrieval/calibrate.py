"""
rag/retrieval/calibrate.py — Auto-calibrate the relevance threshold.

Phase 4 feature: on first startup with a non-empty index, run a small set of
representative queries against the reranker and find the score threshold that
maximises precision without being so aggressive it excludes valid passages.

Algorithm:
  1. Sample up to `n_probes` chunks randomly from ChunkStore.
  2. For each sampled chunk, use the first 20 words as a probe query
     (these are "on-topic" queries that should retrieve the chunk itself
     or close neighbours).
  3. Run dense retrieval + RRF + reranking for each probe.
  4. Collect all reranker scores from those probes.
  5. Set threshold to the 25th percentile of collected scores.
     This means 75% of "naturally relevant" passages will pass the filter.
  6. Clamp to [0.15, 0.55] — the TRD-required range.
  7. Persist the calibrated threshold in the StorageConfig so it survives restart.

The calibrated threshold is stored in `~/.ragdb/calibration.json`.
It is only recomputed when:
  - The file does not exist.
  - `force=True` is passed.
  - The index has grown by more than 50% since last calibration.

Public API:
    calibrate_threshold(config, n_probes, force) → float
    load_calibrated_threshold(config) → float | None
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

_CALIBRATION_FILE = "calibration.json"
_MIN_THRESHOLD = 0.15
_MAX_THRESHOLD = 0.55
_PERCENTILE = 25   # Use 25th percentile of scores as the threshold


def load_calibrated_threshold(config: "RAGConfig") -> Optional[float]:
    """
    Load a previously calibrated threshold from disk.

    Returns:
        float threshold if calibration file exists and index hasn't grown
        too much since last calibration, otherwise None.
    """
    cal_path = config.db_root / _CALIBRATION_FILE
    if not cal_path.exists():
        return None

    try:
        with open(cal_path, encoding="utf-8") as f:
            data = json.load(f)
        threshold = float(data["threshold"])
        cal_chunk_count = int(data.get("chunk_count", 0))

        # Recompute if index has grown more than 50% since calibration.
        from rag.storage.chunk_store import ChunkStore
        with ChunkStore(config) as chunk_store:
            current_count = chunk_store.count()
        if cal_chunk_count > 0 and current_count > cal_chunk_count * 1.5:
            log.info(
                "Index grew %.0f%% since last calibration — will recalibrate.",
                (current_count / cal_chunk_count - 1) * 100,
            )
            return None

        return max(_MIN_THRESHOLD, min(_MAX_THRESHOLD, threshold))

    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        log.warning("Calibration file corrupted (%s) — will recalibrate.", exc)
        return None


def _save_threshold(config: "RAGConfig", threshold: float, chunk_count: int) -> None:
    """Persist the calibrated threshold to disk."""
    cal_path = config.db_root / _CALIBRATION_FILE
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "chunk_count": chunk_count}, f)
    log.info("Saved calibrated threshold %.3f to %s", threshold, cal_path)


def calibrate_threshold(
    config: "RAGConfig",
    n_probes: int = 20,
    force: bool = False,
) -> float:
    """
    Auto-calibrate the relevance threshold from probe queries.

    Args:
        config:   RAGConfig — provides storage paths and model access.
        n_probes: Number of probe queries to run. More probes = more accurate
                  calibration, but slower. Default 20 ≈ < 30 seconds on T1.
        force:    If True, ignore any existing calibration file.

    Returns:
        Calibrated threshold in [0.15, 0.55].
        Falls back to config.retrieval.relevance_threshold (default 0.3)
        if calibration cannot be performed (e.g., index empty, model missing).
    """
    fallback = getattr(config.retrieval, "relevance_threshold", 0.3)

    if not force:
        cached = load_calibrated_threshold(config)
        if cached is not None:
            log.info("Using cached calibration threshold: %.3f", cached)
            return cached

    log.info("Calibrating relevance threshold (n_probes=%d)…", n_probes)

    try:
        from rag.storage.chunk_store import ChunkStore
        from rag.models.model_manager import get_model_manager
        from rag.retrieval.vector_store import VectorStore
        from rag.retrieval.bm25_index import BM25Index
        from rag.retrieval.fusion import rrf_fuse, rrf_to_scored_passages

        with ChunkStore(config) as chunk_store:
            count = chunk_store.count()
            if count == 0:
                log.warning("Index is empty — cannot calibrate. Using default %.3f.", fallback)
                return fallback

            # Sample probe chunks randomly.
            all_ids = chunk_store.list_ids()
            sample_ids = random.sample(all_ids, min(n_probes, len(all_ids)))
            probe_chunks = [c for c in [chunk_store.fetch(id_) for id_ in sample_ids] if c]

            if not probe_chunks:
                return fallback

            # Load models.
            mm = get_model_manager()
            embedder = mm.get_embedder(config)
            reranker = mm.get_reranker(config)
            vector_store = VectorStore(config)
            bm25 = BM25Index(config)

            all_scores: List[float] = []

            for chunk in probe_chunks:
                # Use first 20 words as a probe query.
                words = chunk.text.split()[:20]
                query = " ".join(words)
                if not query:
                    continue

                try:
                    qvec = embedder.encode(query, prefix="search_query: ")
                    dense = vector_store.search_dense(qvec, top_k=15)
                    bm25_res = bm25.search(query, top_k=15)
                    fused = rrf_fuse([dense, bm25_res], top_k=15)
                    candidates = rrf_to_scored_passages(fused, chunk_store)
                    if not candidates:
                        continue

                    texts = [p.chunk.text for p in candidates]
                    scores = reranker.score(query, texts)
                    all_scores.extend(float(s) for s in scores)

                except Exception as exc:
                    log.debug("Probe failed for chunk %s: %s", chunk.id, exc)
                    continue

        if not all_scores:
            log.warning("No probe scores collected — using default %.3f.", fallback)
            return fallback

        # 25th percentile threshold.
        sorted_scores = sorted(all_scores)
        idx = int(len(sorted_scores) * _PERCENTILE / 100)
        threshold = sorted_scores[max(0, idx)]
        threshold = max(_MIN_THRESHOLD, min(_MAX_THRESHOLD, threshold))

        log.info(
            "Calibrated threshold: %.3f (from %d scores, 25th-pct).",
            threshold,
            len(sorted_scores),
        )
        _save_threshold(config, threshold, count)
        return threshold

    except Exception as exc:
        log.warning("Calibration failed (%s) — using default %.3f.", exc, fallback)
        return fallback
