"""
rag/evaluation/latency_test.py — Measures P50 and P95 latency for queries.

Loads the evaluation dataset, runs queries without LLM streaming if possible, 
or just measures total pipeline turn-around, and outputs latency metrics.
"""
from __future__ import annotations

import json
import logging
import time
import argparse
from pathlib import Path
from typing import List

from rag.config import load_config
from rag.pipeline import QueryPipeline

log = logging.getLogger(__name__)

def measure_latency(dataset_path: Path, num_queries: int = 50) -> None:
    config = load_config()
    
    if not dataset_path.exists():
        log.error("Dataset not found at %s. Run test_generator.py first.", dataset_path)
        return
        
    with open(str(dataset_path), "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    if not dataset:
        log.error("Dataset is empty.")
        return
        
    pipeline = QueryPipeline(config)
    latencies: List[float] = []
    
    queries = dataset[:num_queries]
    log.info("Running latency test on %d queries (Tier: %s)...", len(queries), config.resolved_tier)
    
    for i, item in enumerate(queries):
        query = item.get("question")
        if not query:
            continue
            
        start_time = time.monotonic()
        
        # Suppress stdout to avoid clutter
        try:
            # By default pipeline.answer prints to console because of streaming.
            # In a real benchmarking scenario, we might want to disable streaming
            # or redirect output. We'll run it as is and measure wall-clock.
            pipeline.answer(query, show_sources=False)
        except Exception as e:
            log.warning("Query failed: %s", e)
            continue
            
        end_time = time.monotonic()
        duration = end_time - start_time
        latencies.append(duration)
        log.debug("Query %d: %.2fs", i + 1, duration)
        
    if not latencies:
        log.error("No valid queries run.")
        return
        
    latencies.sort()
    n = len(latencies)
    
    p50 = latencies[int(n * 0.50)]
    p90 = latencies[int(n * 0.90)]
    p95 = latencies[int(n * 0.95)]
    avg = sum(latencies) / n
    
    print("\n--- Latency Results ---")
    print(f"Total queries : {n}")
    print(f"Average       : {avg:.3f}s")
    print(f"P50 (Median)  : {p50:.3f}s")
    print(f"P90           : {p90:.3f}s")
    print(f"P95           : {p95:.3f}s")
    print(f"Max           : {latencies[-1]:.3f}s")
    print("-----------------------")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="eval_dataset.json")
    parser.add_argument("-n", type=int, default=50)
    args = parser.parse_args()
    
    measure_latency(Path(args.dataset), num_queries=args.n)
