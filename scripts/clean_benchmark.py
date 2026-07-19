#!/usr/bin/env python3
"""
scripts/clean_benchmark.py — Strip injected system prompts from benchmark question sets.

Some benchmark datasets (e.g. those exported from chatbot APIs) contain
meta-instructions embedded in the question text:
    "You are a helpful assistant. [INST] What is X? [/INST]"
    "<|im_start|>system\nYou are..." 

These injections cause the RAG pipeline to receive a polluted query rather
than the plain user question. This script removes them, producing a clean
JSONL file of {question, ground_truth} pairs.

Input format (JSONL, one JSON object per line):
    {"question": "<injected text> Real question?", "ground_truth": "..."}
    {"question": "...", "answer": "...", "contexts": [...]}

Output format (JSONL, clean):
    {"question": "Real question?", "ground_truth": "...", "contexts": [...]}

Usage:
    python scripts/clean_benchmark.py                                    # uses default paths
    python scripts/clean_benchmark.py input.jsonl output.jsonl
    python scripts/clean_benchmark.py input.jsonl                        # prints to stdout
    python scripts/clean_benchmark.py --check input.jsonl                # count polluted rows
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ── Injection patterns to strip ───────────────────────────────────────────────

_PATTERNS: list[re.Pattern] = [
    # Llama 2 / Llama 3 chat template tags
    re.compile(r"<\|begin_of_text\|>.*?<\|end_header_id\|>\s*", re.DOTALL),
    re.compile(r"\[INST\].*?\[/INST\]", re.DOTALL),
    re.compile(r"\[SYS\].*?\[/SYS\]", re.DOTALL),
    # Phi-3 / Qwen / ChatML style
    re.compile(r"<\|im_start\|>system.*?<\|im_end\|>\s*", re.DOTALL),
    re.compile(r"<\|im_start\|>user\s*", re.DOTALL),
    re.compile(r"<\|im_end\|>", re.DOTALL),
    re.compile(r"<\|end\|>", re.DOTALL),
    re.compile(r"<\|endoftext\|>", re.DOTALL),
    # OpenAI system prompt wrappers
    re.compile(r"You are a helpful(?: and harmless)? assistant[.!]?\s*", re.IGNORECASE),
    re.compile(r"You are an? [\w\s]+ assistant[.!]?\s*", re.IGNORECASE),
    # RAGAS-style meta-instructions
    re.compile(r"(?:Question|Q):\s*", re.IGNORECASE),
    # HyDE / RAG pipeline injections
    re.compile(r"Write a short.*?passage.*?\n", re.IGNORECASE),
]


def _clean(text: str) -> str:
    """Strip all known injection patterns from text and normalise whitespace."""
    for pat in _PATTERNS:
        text = pat.sub("", text)
    # Collapse multiple whitespace / newlines to a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_polluted(text: str) -> bool:
    """Return True if any injection pattern is found in text."""
    for pat in _PATTERNS:
        if pat.search(text):
            return True
    return False


def _process_stream(lines: list[str]) -> tuple[list[dict], int, int]:
    """Parse and clean JSONL lines. Returns (records, total, n_cleaned)."""
    records: list[dict] = []
    n_cleaned = 0
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"  [warn] line {i}: invalid JSON — {exc}", file=sys.stderr)
            continue

        question = obj.get("question", "")
        original_question = question
        if _is_polluted(question):
            obj["question"] = _clean(question)
            n_cleaned += 1

        records.append(obj)

    return records, len(records), n_cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip injected system prompts from benchmark JSONL question sets."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Input JSONL file path (default: rag/evaluation/benchmark_dataset.json)",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output JSONL file path (default: stdout, or benchmark_dataset_clean.json if using default input)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Count polluted rows without writing output",
    )
    args = parser.parse_args()

    if not args.input:
        project_root = Path(__file__).parent.parent
        args.input = project_root / "rag" / "evaluation" / "benchmark_dataset.json"
        if not args.output:
            args.output = args.input.with_name("benchmark_dataset_clean.json")

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    lines = args.input.read_text(encoding="utf-8").splitlines()

    if args.check:
        polluted = sum(1 for l in lines if l.strip() and _is_polluted(json.loads(l).get("question", "")))
        total = sum(1 for l in lines if l.strip())
        print(f"Checked {total} records. Polluted: {polluted} ({100 * polluted / max(total, 1):.1f}%)")
        sys.exit(0)

    records, total, n_cleaned = _process_stream(lines)

    print(f"Processed {total} records. Cleaned {n_cleaned} ({100 * n_cleaned / max(total, 1):.1f}%).",
          file=sys.stderr)

    output_lines = [json.dumps(r, ensure_ascii=False) for r in records]
    output_text = "\n".join(output_lines) + "\n"

    if args.output:
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
