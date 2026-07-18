"""
custom_scorer.py — Compute RAGAS-equivalent metrics WITHOUT any external API.

Metrics computed:
  1. Answer Relevancy   — cosine similarity between question embedding and answer embedding
  2. Context Precision  — mean cosine similarity of each context chunk against (question + ground_truth)
  3. Faithfulness       — fraction of answer sentences supported by context
                          (token overlap + semantic similarity)

Uses the local nomic-embed-text-v1.5 embedder already in the project.
"""
from __future__ import annotations

import json
import sys
import re
import logging
import math
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("scorer")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import load_config
from rag.models.model_manager import get_model_manager


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r'\b\w+\b', a.lower()))
    tb = set(re.findall(r'\b\w+\b', b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def clean_answer(answer: str) -> str:
    text = re.sub(r'\[\d+\]', '', answer)
    lines = text.split('\n')
    seen, deduped = set(), []
    for line in lines:
        key = line.strip().lower()[:60]
        if key and key not in seen:
            seen.add(key)
            deduped.append(line)
    return ' '.join(' '.join(deduped).split())


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def embed(text: str, embedder) -> np.ndarray:
    return np.array(embedder.encode(text[:500], prefix="search_query: "), dtype=np.float32)


def score_item(item: dict, embedder) -> dict:
    question     = item.get("question", "")[:400]
    answer_raw   = item.get("answer", "")
    ground_truth = item.get("ground_truth", "")
    contexts     = item.get("contexts", [])

    answer = clean_answer(answer_raw)

    q_vec = embed(question, embedder)

    # ── 1. Answer Relevancy ────────────────────────────────────────────────────
    try:
        a_vec = embed(answer, embedder)
        answer_relevancy = cosine(q_vec, a_vec)
    except Exception as e:
        log.warning("AR embed failed: %s", e)
        answer_relevancy = float('nan')

    # ── 2. Context Precision ───────────────────────────────────────────────────
    ctx_scores = []
    if contexts and ground_truth:
        try:
            gt_vec = embed(ground_truth, embedder)
            for ctx in contexts:
                ctx_vec = embed(ctx, embedder)
                score = (cosine(ctx_vec, q_vec) + cosine(ctx_vec, gt_vec)) / 2.0
                ctx_scores.append(score)
            context_precision = float(np.mean(ctx_scores)) if ctx_scores else float('nan')
        except Exception as e:
            log.warning("CP embed failed: %s", e)
            context_precision = float('nan')
    else:
        context_precision = float('nan')

    # ── 3. Faithfulness ────────────────────────────────────────────────────────
    ctx_full = " ".join(contexts)
    answer_sentences = split_sentences(answer)
    faithful_count = 0
    if answer_sentences and contexts:
        try:
            ctx_vec_full = embed(ctx_full, embedder)
            for sent in answer_sentences:
                if len(sent) < 15:
                    continue
                overlap  = token_overlap(sent, ctx_full)
                s_vec    = embed(sent, embedder)
                sem_sim  = cosine(s_vec, ctx_vec_full)
                if overlap > 0.15 or sem_sim > 0.65:
                    faithful_count += 1
            faithfulness = faithful_count / len(answer_sentences) if answer_sentences else float('nan')
        except Exception as e:
            log.warning("Faith embed failed: %s", e)
            faithfulness = float('nan')
    else:
        faithfulness = float('nan')

    return {
        "question": question[:80],
        "answer_relevancy": round(answer_relevancy, 4) if not math.isnan(answer_relevancy) else None,
        "context_precision": round(context_precision, 4) if not math.isnan(context_precision) else None,
        "faithfulness": round(faithfulness, 4) if not math.isnan(faithfulness) else None,
    }


def safe_mean(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return round(sum(v)/len(v), 4) if v else None


def main():
    cache_path = PROJECT_ROOT / "ragas_results_cache.json"
    if not cache_path.exists():
        log.error("ragas_results_cache.json not found")
        sys.exit(1)

    with open(str(cache_path), "r", encoding="utf-8") as f:
        items = json.load(f)

    log.info("Loaded %d cached items", len(items))

    cfg      = load_config()
    embedder = get_model_manager().get_embedder(cfg)
    log.info("Embedder loaded — scoring all %d items", len(items))

    results = []
    for i, item in enumerate(items):
        log.info("[%d/%d] %s...", i+1, len(items), item.get("question","")[:60])
        try:
            scores = score_item(item, embedder)
        except Exception as e:
            log.warning("Failed item %d: %s", i, e)
            scores = {"question": item.get("question","")[:80], "answer_relevancy": None, "context_precision": None, "faithfulness": None}
        results.append(scores)

    ar    = safe_mean([r["answer_relevancy"]  for r in results])
    cp    = safe_mean([r["context_precision"] for r in results])
    faith = safe_mean([r["faithfulness"]      for r in results])

    print("\n" + "="*65)
    print("  MOTIF RAG BENCHMARK — Custom Embedding-Based Evaluation")
    print("="*65)
    print(f"  Evaluated Items  : {len(results)}")
    print(f"  Answer Relevancy : {ar}")
    print(f"  Context Precision: {cp}")
    print(f"  Faithfulness     : {faith}")
    print("="*65)
    print(f"\n  {'#':<4} {'AR':>6} {'CP':>6} {'Faith':>7}  Question")
    print("  " + "-"*85)
    for i, r in enumerate(results):
        ar_s  = f"{r['answer_relevancy']:.3f}"  if r['answer_relevancy']  is not None else "  N/A"
        cp_s  = f"{r['context_precision']:.3f}" if r['context_precision'] is not None else "  N/A"
        fa_s  = f"{r['faithfulness']:.3f}"      if r['faithfulness']      is not None else "  N/A"
        print(f"  {i+1:<4} {ar_s:>6} {cp_s:>6} {fa_s:>7}  {r['question'][:60]}")

    out = {
        "summary": {"n_evaluated": len(results), "answer_relevancy": ar, "context_precision": cp, "faithfulness": faith},
        "per_item": results,
    }
    out_path = PROJECT_ROOT / "custom_eval_results.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
