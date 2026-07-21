"""
rag/evaluation/test_generator.py — Generates synthetic Q&A pairs for offline evaluation.

Algorithm:
1. Sample n chunks from ChunkStore.
2. Prompt the local LLM to generate one specific, answerable question about the chunk.
3. Save {question, ground_truth, source, source_type} as JSON.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import RAGConfig

log = logging.getLogger(__name__)

QUESTION_PROMPT = (
    "Generate one specific, answerable question whose answer is contained in this text. "
    "The question should be clear and factual.\n\nText:\n{text}\n\nQuestion:"
)

def create_eval_dataset(
    config: RAGConfig,
    n: int = 50,
    output_path: Path | None = None,
) -> list[dict]:
    from rag.models.model_manager import get_model_manager
    from rag.storage.chunk_store import ChunkStore

    store = ChunkStore(config)
    
    try:
        llm = get_model_manager().get_llm(config)
    except FileNotFoundError as exc:
        log.error("LLM not available: %s", exc)
        return []

    # Sample chunks
    conn = sqlite3.connect(str(config.db_root / "chunks.db"))
    rows = conn.execute(
        "SELECT id, text, filename, source_type FROM chunks ORDER BY RANDOM() LIMIT ?", (n,)
    ).fetchall()
    conn.close()

    if not rows:
        log.warning("No chunks found in index to generate questions from.")
        return []

    dataset = []
    log.info("Generating %d synthetic questions using %s...", len(rows), config.resolved_tier)
    
    for row in rows:
        chunk_id, text, filename, source_type = row
        try:
            question = llm.generate(
                QUESTION_PROMPT.format(text=text[:800]),  # truncate for prompt safety
                max_tokens=60,
                temperature=0.4,
            ).strip()
            
            if question:
                dataset.append({
                    "question": question,
                    "ground_truth": text,
                    "source": filename,
                    "source_type": source_type,
                    "chunk_id": chunk_id,
                })
        except Exception as e:
            log.warning("Failed to generate question for chunk %s: %s", chunk_id, e)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        log.info("Saved %d evaluation questions to %s", len(dataset), output_path)

    return dataset

if __name__ == "__main__":
    import argparse

    from rag.config import load_config
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("-o", "--output", type=str, default="eval_dataset.json")
    args = parser.parse_args()
    
    config = load_config()
    create_eval_dataset(config, n=args.n, output_path=Path(args.output))
