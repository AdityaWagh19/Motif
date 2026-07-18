"""
rag/evaluation/latency_test.py — P50/P95 query latency measurement.

Measures wall-clock latency over N queries (including warmup) and returns
structured statistics. Used for NFR validation and regression testing.

Usage (programmatic):
    from rag.evaluation.latency_test import run_latency_test
    results = run_latency_test(questions, config, warmup=2)

Usage (CLI):
    python -m rag.evaluation.latency_test --dataset eval_dataset.json -n 50
"""
from __future__ import annotations

import json
import logging
import time
import argparse
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)


def run_latency_test(
    questions: List[str],
    config: "RAGConfig",
    warmup: int = 2,
) -> dict:
    """
    Measure query latency for a list of questions.

    Runs `warmup` queries first (not measured) to warm up models, then
    measures wall-clock time for each remaining query.

    Args:
        questions:  List of query strings (at least warmup + 1 entries).
        config:     RAGConfig.
        warmup:     Number of warmup queries to run before measurement.
                    (Models are cold on first call — warmup avoids skewing P50.)

    Returns:
        {
            "p50_ms": float,
            "p95_ms": float,
            "p99_ms": float,
            "min_ms": float,
            "max_ms": float,
            "n_queries": int,
            "tier": str,
        }
    """
    from rag.pipeline import QueryPipeline

    if not questions:
        raise ValueError("questions list must not be empty")

    pipeline = QueryPipeline(config)
    latencies: List[float] = []

    all_questions = list(questions[:warmup]) + list(questions)
    log.info(
        "Latency test: %d warmup + %d measured queries (tier=%s)",
        warmup, len(questions), config.resolved_tier,
    )

    for i, q in enumerate(all_questions):
        t0 = time.monotonic()
        try:
            pipeline.answer(query=q, history=[], show_sources=False)
        except Exception as e:
            log.warning("Query failed during latency test: %s", e)
            continue
        elapsed_ms = (time.monotonic() - t0) * 1000

        if i >= warmup:  # skip warmup queries
            latencies.append(elapsed_ms)

    if not latencies:
        raise RuntimeError("No latency measurements collected — all queries failed.")

    latencies.sort()
    n = len(latencies)

    return {
        "p50_ms": latencies[int(n * 0.50)],
        "p95_ms": latencies[min(int(n * 0.95), n - 1)],
        "p99_ms": latencies[min(int(n * 0.99), n - 1)],
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
        "n_queries": n,
        "tier": config.resolved_tier,
    }


def measure_latency(dataset_path: Path, num_queries: int = 50) -> None:
    """CLI entry point: load dataset from file and run latency test."""
    from rag.config import load_config

    config = load_config()

    if not dataset_path.exists():
        log.error("Dataset not found at %s. Run test_generator.py first.", dataset_path)
        return

    with open(str(dataset_path), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not dataset:
        log.error("Dataset is empty.")
        return

    questions = [item["question"] for item in dataset[:num_queries] if item.get("question")]

    if not questions:
        log.error("No questions found in dataset.")
        return

    log.info("Running latency test on %d queries (Tier: %s)...", len(questions), config.resolved_tier)

    try:
        results = run_latency_test(questions, config, warmup=2)
    except RuntimeError as e:
        log.error("%s", e)
        return

    print("\n--- Latency Results ---")
    print(f"Tier          : {results['tier']}")
    print(f"Total queries : {results['n_queries']}")
    print(f"Min           : {results['min_ms']:.1f} ms")
    print(f"P50 (Median)  : {results['p50_ms']:.1f} ms")
    print(f"P95           : {results['p95_ms']:.1f} ms")
    print(f"P99           : {results['p99_ms']:.1f} ms")
    print(f"Max           : {results['max_ms']:.1f} ms")
    print("-----------------------")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run latency test on eval dataset.")
    parser.add_argument("--dataset", type=str, default="eval_dataset.json")
    parser.add_argument("-n", type=int, default=50)
    args = parser.parse_args()

    measure_latency(Path(args.dataset), num_queries=args.n)
