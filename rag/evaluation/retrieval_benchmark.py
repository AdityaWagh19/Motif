#!/usr/bin/env python3
"""
rag/evaluation/retrieval_benchmark.py — Retrieval-only evaluation.

Evaluates retrieval independently from generation, so you can tell whether
a bad end-to-end score is caused by the retriever or the generator.

Metrics:
    Recall@k         — Did the ground-truth chunk appear in the top-k results?
    Precision@k      — What % of retrieved chunks were relevant?
    MRR              — Mean Reciprocal Rank of the first relevant chunk
    NDCG@k           — Normalized Discounted Cumulative Gain

Ground-truth format (JSONL, one line per query):
    {
      "question": "What is self-attention?",
      "ground_truth_sources": ["attention_is_all_you_need.pdf"],
      "ground_truth_keywords": ["self-attention", "attention mechanism"]
    }

A chunk is considered relevant if its source matches a ground_truth_source
AND its text contains at least one ground_truth_keyword.

Usage:
    python -m rag.evaluation.retrieval_benchmark
    python -m rag.evaluation.retrieval_benchmark --questions eval_questions.jsonl --k 5 10
    python -m rag.evaluation.retrieval_benchmark --verbose
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Default evaluation questions
# Generated from the benchmark_dataset.json — can be replaced with a JSONL file.
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_QUESTIONS = [
    {
        "question": "What is the Transformer architecture?",
        "ground_truth_sources": [],          # Any source
        "ground_truth_keywords": ["transformer", "attention", "encoder", "decoder"],
    },
    {
        "question": "How does self-attention work?",
        "ground_truth_sources": [],
        "ground_truth_keywords": ["self-attention", "query", "key", "value"],
    },
    {
        "question": "What optimizer was used in training?",
        "ground_truth_sources": [],
        "ground_truth_keywords": ["adam", "optimizer", "learning rate"],
    },
    {
        "question": "What datasets were used for evaluation?",
        "ground_truth_sources": [],
        "ground_truth_keywords": ["wmt", "dataset", "newstest"],
    },
    {
        "question": "How many attention heads are there?",
        "ground_truth_sources": [],
        "ground_truth_keywords": ["8", "attention head", "multi-head"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


@dataclass
class QueryRetrievalResult:
    question: str
    k: int
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    relevant_indices: list[int] = field(default_factory=list)   # 0-based
    recall: float = 0.0
    precision: float = 0.0
    reciprocal_rank: float = 0.0
    ndcg: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkSummary:
    k: int
    n_queries: int
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    avg_latency_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Relevance judgment
# ─────────────────────────────────────────────────────────────────────────────

def _is_relevant(
    chunk: RetrievedChunk,
    ground_truth_sources: list[str],
    ground_truth_keywords: list[str],
) -> bool:
    """
    A chunk is relevant if:
      - No sources specified (any source is relevant), OR source matches
      - AND at least one keyword appears in the chunk text (case-insensitive)

    If no keywords are specified, any chunk from a matching source is relevant.
    """
    text_lower = chunk.text.lower()
    source_lower = chunk.source.lower()

    # Source filter (if specified)
    if ground_truth_sources:
        source_ok = any(s.lower() in source_lower for s in ground_truth_sources)
        if not source_ok:
            return False

    # Keyword filter (if specified)
    if ground_truth_keywords:
        return any(kw.lower() in text_lower for kw in ground_truth_keywords)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval runner
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve(query: str, k: int, pipeline, embedder, config) -> tuple[list[RetrievedChunk], float]:
    """
    Run the retrieval-only path (no generation) and return (chunks, latency_ms).
    Uses the same retriever as the full pipeline but skips LLM.
    """
    from rag.reranking.cross_encoder import rerank
    from rag.retrieval.fusion import rrf_fuse, rrf_to_scored_passages

    t0 = time.monotonic()

    # 1. Expand query
    query_vector, effective_query = pipeline._expander.expand(query, config, embedder)

    # 2. Hybrid search
    dense_results = pipeline._vector_store.search_dense(query_vector, top_k=k * 2)
    bm25_results = pipeline._bm25.search(effective_query, top_k=k * 2)

    # 3. RRF fusion
    fused = rrf_fuse([dense_results, bm25_results], top_k=k * 2)
    candidates = rrf_to_scored_passages(fused, pipeline._chunk_store)

    # 4. Rerank to top-k
    if candidates and len(candidates) > k:
        candidates = rerank(query, candidates, config, top_k=k, threshold=0.0)
    else:
        candidates = candidates[:k]

    latency_ms = (time.monotonic() - t0) * 1000

    chunks = [
        RetrievedChunk(
            text=c.chunk.text,
            source=c.chunk.source,
            score=c.score,
        )
        for c in candidates
    ]
    return chunks, latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ndcg(relevant_indices: list[int], k: int) -> float:
    """Compute NDCG@k for a single query. Binary relevance (0 or 1)."""
    if not relevant_indices:
        return 0.0
    dcg = sum(
        1.0 / math.log2(idx + 2)  # +2 because idx is 0-based → positions 1-based
        for idx in relevant_indices
        if idx < k
    )
    # Ideal DCG: all relevant docs at top
    n_relevant = len([i for i in relevant_indices if i < k])
    idcg = sum(1.0 / math.log2(pos + 2) for pos in range(min(n_relevant, k)))
    return dcg / idcg if idcg > 0 else 0.0


def _evaluate_query(
    question: str,
    ground_truth_sources: list[str],
    ground_truth_keywords: list[str],
    k: int,
    pipeline,
    embedder,
    config,
    verbose: bool = False,
) -> QueryRetrievalResult:
    """Run retrieval for one query and compute all metrics."""
    result = QueryRetrievalResult(question=question, k=k)

    try:
        chunks, latency_ms = _retrieve(question, k, pipeline, embedder, config)
        result.retrieved = chunks
        result.latency_ms = latency_ms
    except Exception as exc:
        result.error = str(exc)
        return result

    # Find relevant chunks
    relevant_indices = [
        i for i, c in enumerate(chunks)
        if _is_relevant(c, ground_truth_sources, ground_truth_keywords)
    ]
    result.relevant_indices = relevant_indices

    # Recall@k — was any relevant chunk retrieved?
    result.recall = 1.0 if relevant_indices else 0.0

    # Precision@k — fraction of retrieved chunks that are relevant
    result.precision = len(relevant_indices) / k if k > 0 else 0.0

    # MRR — reciprocal rank of first relevant chunk
    if relevant_indices:
        first_relevant = min(relevant_indices) + 1   # 1-indexed
        result.reciprocal_rank = 1.0 / first_relevant
    else:
        result.reciprocal_rank = 0.0

    # NDCG@k
    result.ndcg = _compute_ndcg(relevant_indices, k)

    if verbose:
        print(f"\n  Q: {question}")
        print(f"  Retrieved {len(chunks)} chunks in {latency_ms:.0f} ms")
        for i, c in enumerate(chunks[:5]):
            rel = "[REL]" if i in relevant_indices else "     "
            print(f"  {rel} [{i+1}] {c.source[:30]:<30}  score={c.score:.3f}  text={c.text[:60]!r}")
        print(f"  Recall@{k}={result.recall:.2f}  "
              f"Precision@{k}={result.precision:.2f}  "
              f"RR={result.reciprocal_rank:.3f}  "
              f"NDCG@{k}={result.ndcg:.3f}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_retrieval_benchmark(
    questions: list[dict],
    k_values: list[int],
    config,
    verbose: bool = False,
) -> tuple[list[QueryRetrievalResult], list[BenchmarkSummary]]:
    """
    Run retrieval benchmark for all questions and k values.

    Returns:
        (per_query_results, summaries_per_k)
    """
    from rag.models.model_manager import get_model_manager
    from rag.pipeline import QueryPipeline

    pipeline = QueryPipeline(config)
    embedder = get_model_manager().get_embedder(config)

    all_results: list[QueryRetrievalResult] = []
    summaries: list[BenchmarkSummary] = []

    for k in k_values:
        print(f"\n── k={k} ──────────────────────────────────────────────────────")
        k_results: list[QueryRetrievalResult] = []

        for q in questions:
            result = _evaluate_query(
                question=q["question"],
                ground_truth_sources=q.get("ground_truth_sources", []),
                ground_truth_keywords=q.get("ground_truth_keywords", []),
                k=k,
                pipeline=pipeline,
                embedder=embedder,
                config=config,
                verbose=verbose,
            )
            k_results.append(result)
            all_results.append(result)

            status = "[red]ERR[/red]" if result.error else (
                "[green]HIT[/green]" if result.recall > 0 else "[yellow]MISS[/yellow]"
            )
            if not verbose:
                msg = result.error[:50] if result.error else (
                    f"Recall={result.recall:.0%}  MRR={result.reciprocal_rank:.3f}  {result.latency_ms:.0f} ms"
                )
                print(f"  {status}  {q['question'][:55]:<55}  {msg}")

        # Compute summary
        valid = [r for r in k_results if not r.error]
        if valid:
            summary = BenchmarkSummary(
                k=k,
                n_queries=len(valid),
                recall_at_k=sum(r.recall for r in valid) / len(valid),
                precision_at_k=sum(r.precision for r in valid) / len(valid),
                mrr=sum(r.reciprocal_rank for r in valid) / len(valid),
                ndcg_at_k=sum(r.ndcg for r in valid) / len(valid),
                avg_latency_ms=sum(r.latency_ms for r in valid) / len(valid),
            )
            summaries.append(summary)

    return all_results, summaries


def _print_summary_table(summaries: list[BenchmarkSummary], console=None) -> None:
    from rich import box as rbox
    from rich.console import Console
    from rich.table import Table

    if console is None:
        console = Console()

    t = Table(
        title="Retrieval Benchmark Summary",
        box=rbox.ROUNDED,
        show_lines=True,
    )
    t.add_column("k",             justify="right")
    t.add_column("Recall@k",      justify="right")
    t.add_column("Precision@k",   justify="right")
    t.add_column("MRR",           justify="right")
    t.add_column("NDCG@k",        justify="right")
    t.add_column("Avg ms",        justify="right")
    t.add_column("Queries",       justify="right")

    for s in summaries:
        t.add_row(
            str(s.k),
            f"{s.recall_at_k:.1%}",
            f"{s.precision_at_k:.1%}",
            f"{s.mrr:.3f}",
            f"{s.ndcg_at_k:.3f}",
            f"{s.avg_latency_ms:.0f}",
            str(s.n_queries),
        )

    console.print(t)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval-only evaluation: Recall@k, Precision@k, MRR, NDCG@k"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="JSONL file with ground-truth questions (default: built-in set)",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        metavar="K",
        help="k values to evaluate (default: 3 5 10)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-chunk relevance decisions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: auto-named in project root)",
    )
    args = parser.parse_args()

    from rich.console import Console

    from rag.config import load_config
    console = Console()

    config = load_config()
    console.print(f"\n[bold]Retrieval Benchmark[/bold] — Tier {config.resolved_tier}")

    # Load questions
    if args.questions and args.questions.exists():
        raw_lines = args.questions.read_text(encoding="utf-8").splitlines()
        questions = [json.loads(l) for l in raw_lines if l.strip()]
        console.print(f"Loaded {len(questions)} questions from {args.questions}")
    else:
        questions = _DEFAULT_QUESTIONS
        console.print(f"Using {len(questions)} built-in evaluation questions")

    all_results, summaries = run_retrieval_benchmark(
        questions=questions,
        k_values=sorted(set(args.k)),
        config=config,
        verbose=args.verbose,
    )

    console.print()
    _print_summary_table(summaries, console=console)

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    output = args.output or (PROJECT_ROOT / f"retrieval_benchmark_{ts}.json")
    data = {
        "summaries": [asdict(s) for s in summaries],
        "per_query":  [
            {
                "question": r.question,
                "k": r.k,
                "recall": r.recall,
                "precision": r.precision,
                "mrr": r.reciprocal_rank,
                "ndcg": r.ndcg,
                "latency_ms": r.latency_ms,
                "relevant_positions": r.relevant_indices,
                "error": r.error,
            }
            for r in all_results
        ],
    }
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\nResults saved to [bold]{output}[/bold]")


if __name__ == "__main__":
    main()
