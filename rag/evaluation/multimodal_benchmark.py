"""
rag/evaluation/multimodal_benchmark.py — Full multimodal benchmark suite.

Measures ingestion throughput, parsing accuracy, and per-modality RAG quality
across all five document types: PDF, DOCX, Markdown, Image, Audio.

Metrics collected per modality:
    parse_time_ms         — wall-clock time for parser.parse()
    chunk_count           — number of chunks produced
    char_count            — total characters extracted
    embed_time_ms         — time to embed all chunks via Embedder
    ingest_time_ms        — total ingestion time (parse + chunk + embed + store)
    query_time_ms         — P50 query latency over N sample queries
    answer_relevance      — fraction of queries with non-empty answer
    retrieval_hit_rate    — fraction of queries returning ≥1 passage from this doc

Usage:
    python -m rag.evaluation.multimodal_benchmark --fixtures tests/fixtures/ --queries 5

Output:
    Console table (Rich) + JSON file: benchmark_multimodal_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParseResult:
    modality: str
    filename: str
    pages: int = 0
    chunks: int = 0
    chars: int = 0
    parse_time_ms: float = 0.0
    embed_time_ms: float = 0.0
    ingest_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class QueryResult:
    query: str
    modality: str
    latency_ms: float = 0.0
    passages_used: int = 0
    answer_non_empty: bool = False
    error: Optional[str] = None


@dataclass
class ModalitySummary:
    modality: str
    files_tested: int = 0
    total_chunks: int = 0
    total_chars: int = 0
    avg_parse_ms: float = 0.0
    avg_embed_ms: float = 0.0
    avg_ingest_ms: float = 0.0
    avg_query_ms: float = 0.0
    answer_rate: float = 0.0  # fraction of queries with non-empty answer
    retrieval_hit_rate: float = 0.0  # fraction of queries returning ≥1 passage
    errors: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

_MODALITY_QUERIES: dict[str, list[str]] = {
    "pdf": [
        "What is the main topic of this document?",
        "Summarise the key findings.",
        "What methods are described?",
    ],
    "docx": [
        "What is this document about?",
        "List the main sections.",
    ],
    "md": [
        "What does this document cover?",
        "What are the key points in section 2?",
    ],
    "image": [
        "What text appears in this image?",
        "Describe the content of this image.",
    ],
    "audio": [
        "What was discussed in this recording?",
        "Summarise the audio content.",
    ],
}

_FIXTURE_EXTENSIONS: dict[str, list[str]] = {
    "pdf":   [".pdf"],
    "docx":  [".docx"],
    "md":    [".md", ".txt", ".markdown"],
    "image": [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"],
    "audio": [".mp3", ".wav", ".m4a", ".flac", ".ogg"],
}


class MultimodalBenchmark:
    """
    Run ingestion + query benchmarks across all five modalities.

    Args:
        fixtures_dir: Directory containing sample files (one per modality minimum).
        n_queries:    Number of queries to run per modality (uses _MODALITY_QUERIES).
        config:       RAGConfig. If None, loads from config.toml.
    """

    def __init__(
        self,
        fixtures_dir: Path,
        n_queries: int = 3,
        config=None,
    ) -> None:
        self._fixtures = fixtures_dir
        self._n_queries = n_queries
        if config is None:
            from rag.config import load_config
            config = load_config()
        self._config = config
        self._parse_results: list[ParseResult] = []
        self._query_results: list[QueryResult] = []

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def run_ingestion(self, console=None) -> None:
        """Parse all fixture files and record timings."""
        from rag.ingestion.parsers.base import get_parser
        from rag.ingestion.chunker import SentenceChunker, ChunkerConfig
        from rag.models.model_manager import get_model_manager

        chunker = SentenceChunker(ChunkerConfig(target_tokens=512, overlap_tokens=64))

        for modality, exts in _FIXTURE_EXTENSIONS.items():
            files = []
            for ext in exts:
                files.extend(self._fixtures.glob(f"*{ext}"))

            if not files:
                if console:
                    console.print(f"  [yellow]skip[/yellow]  {modality}: no fixture files found")
                continue

            for fixture_file in files:
                result = ParseResult(modality=modality, filename=fixture_file.name)

                # Parse
                t0 = time.monotonic()
                try:
                    parser = get_parser(fixture_file, self._config)
                    pages = parser.parse(fixture_file)
                    result.parse_time_ms = (time.monotonic() - t0) * 1000
                    result.pages = len(pages)
                    result.chars = sum(len(p.text) for p in pages)

                    # Chunk
                    chunks = chunker.chunk_pages(
                        pages,
                        source=str(fixture_file),
                        filename=fixture_file.name,
                        source_type=modality,
                    )
                    result.chunks = len(chunks)

                    # Embed
                    t1 = time.monotonic()
                    try:
                        embedder = get_model_manager().get_embedder(self._config)
                        embedder.encode_batch([c.text for c in chunks])
                        result.embed_time_ms = (time.monotonic() - t1) * 1000
                    except Exception as emb_exc:
                        result.embed_time_ms = -1
                        result.error = f"embed: {emb_exc}"

                    result.ingest_time_ms = (time.monotonic() - t0) * 1000

                except Exception as exc:
                    result.parse_time_ms = (time.monotonic() - t0) * 1000
                    result.error = str(exc)

                self._parse_results.append(result)

                status = "[red]FAIL[/red]" if result.error else "[green]ok[/green]"
                if console:
                    console.print(
                        f"  {status}  {modality:<8}  {fixture_file.name:<30}  "
                        f"{result.parse_time_ms:.0f} ms parse  "
                        f"{result.chunks} chunks"
                    )

    # ── Query benchmarking ────────────────────────────────────────────────────

    def run_queries(self, console=None) -> None:
        """Run sample queries per modality and record latencies."""
        from rag.pipeline import QueryPipeline

        pipeline = QueryPipeline(self._config)

        for modality, queries in _MODALITY_QUERIES.items():
            for query_text in queries[: self._n_queries]:
                result = QueryResult(query=query_text, modality=modality)
                t0 = time.monotonic()
                try:
                    answer = pipeline.answer(
                        query=query_text,
                        history=[],
                        type_filter=modality,
                        use_hyde=False,
                        show_sources=False,
                    )
                    result.latency_ms = (time.monotonic() - t0) * 1000
                    result.passages_used = answer.passages_used
                    result.answer_non_empty = bool(answer.text.strip())
                except Exception as exc:
                    result.latency_ms = (time.monotonic() - t0) * 1000
                    result.error = str(exc)

                self._query_results.append(result)

                if console:
                    status = "[red]ERR[/red]" if result.error else (
                        "[green]ok[/green]" if result.answer_non_empty else "[yellow]empty[/yellow]"
                    )
                    console.print(
                        f"  {status}  {modality:<8}  {query_text[:50]:<50}  "
                        f"{result.latency_ms:.0f} ms"
                    )

    # ── Summary computation ───────────────────────────────────────────────────

    def summarise(self) -> list[ModalitySummary]:
        summaries: list[ModalitySummary] = []
        modalities = list(_FIXTURE_EXTENSIONS.keys())

        for mod in modalities:
            parse_recs = [r for r in self._parse_results if r.modality == mod]
            query_recs  = [r for r in self._query_results  if r.modality == mod]

            if not parse_recs and not query_recs:
                continue

            s = ModalitySummary(modality=mod)
            s.files_tested = len(parse_recs)
            s.total_chunks = sum(r.chunks for r in parse_recs)
            s.total_chars  = sum(r.chars  for r in parse_recs)
            s.errors       = [r.error for r in parse_recs if r.error]

            if parse_recs:
                s.avg_parse_ms = sum(r.parse_time_ms for r in parse_recs) / len(parse_recs)
                embed_times = [r.embed_time_ms for r in parse_recs if r.embed_time_ms >= 0]
                s.avg_embed_ms = sum(embed_times) / len(embed_times) if embed_times else 0
                s.avg_ingest_ms = sum(r.ingest_time_ms for r in parse_recs) / len(parse_recs)

            if query_recs:
                s.avg_query_ms = sum(r.latency_ms for r in query_recs) / len(query_recs)
                answered = [r for r in query_recs if not r.error]
                if answered:
                    s.answer_rate = sum(1 for r in answered if r.answer_non_empty) / len(answered)
                    s.retrieval_hit_rate = sum(1 for r in answered if r.passages_used > 0) / len(answered)

            summaries.append(s)

        return summaries

    # ── Rich table display ────────────────────────────────────────────────────

    def print_table(self, summaries: list[ModalitySummary], console=None) -> None:
        from rich.table import Table
        from rich.console import Console
        from rich import box as rbox

        if console is None:
            console = Console()

        t = Table(title="Multimodal Benchmark Summary", box=rbox.ROUNDED, show_lines=True)
        t.add_column("Modality",   style="bold")
        t.add_column("Files",      justify="right")
        t.add_column("Chunks",     justify="right")
        t.add_column("Parse ms",   justify="right")
        t.add_column("Embed ms",   justify="right")
        t.add_column("Query ms",   justify="right")
        t.add_column("Ans Rate",   justify="right")
        t.add_column("Hit Rate",   justify="right")
        t.add_column("Errors",     justify="left",  style="red")

        for s in summaries:
            t.add_row(
                s.modality,
                str(s.files_tested),
                str(s.total_chunks),
                f"{s.avg_parse_ms:.0f}",
                f"{s.avg_embed_ms:.0f}" if s.avg_embed_ms else "—",
                f"{s.avg_query_ms:.0f}" if s.avg_query_ms else "—",
                f"{s.answer_rate:.0%}" if s.answer_rate else "—",
                f"{s.retrieval_hit_rate:.0%}" if s.retrieval_hit_rate else "—",
                str(len(s.errors)) if s.errors else "0",
            )

        console.print(t)

    # ── JSON export ───────────────────────────────────────────────────────────

    def save_json(self, output_path: Path) -> None:
        data = {
            "parse_results":  [asdict(r) for r in self._parse_results],
            "query_results":  [asdict(r) for r in self._query_results],
            "summaries":      [asdict(s) for s in self.summarise()],
        }
        with open(str(output_path), "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Results written to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full multimodal ingestion + query benchmark."
    )
    parser.add_argument("--fixtures", type=Path, default=PROJECT_ROOT / "tests" / "fixtures",
                        help="Directory containing sample files (default: tests/fixtures/)")
    parser.add_argument("--queries", type=int, default=3,
                        help="Number of queries per modality (default: 3)")
    parser.add_argument("--no-queries", action="store_true",
                        help="Skip the query benchmark (ingestion only)")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output file path (default: auto-named in project root)")
    args = parser.parse_args()

    from rich.console import Console
    console = Console()

    if not args.fixtures.exists():
        console.print(f"[red]Fixtures directory not found:[/red] {args.fixtures}")
        console.print("Run: python scripts/test_multimodal.py --create-fixtures")
        sys.exit(1)

    bench = MultimodalBenchmark(fixtures_dir=args.fixtures, n_queries=args.queries)

    console.print("\n[bold]── Ingestion Phase ──[/bold]")
    bench.run_ingestion(console=console)

    if not args.no_queries:
        console.print("\n[bold]── Query Phase ──[/bold]")
        bench.run_queries(console=console)

    summaries = bench.summarise()
    console.print("\n[bold]── Summary ──[/bold]")
    bench.print_table(summaries, console=console)

    ts = time.strftime("%Y%m%d_%H%M%S")
    output = args.output or (PROJECT_ROOT / f"benchmark_multimodal_{ts}.json")
    bench.save_json(output)


if __name__ == "__main__":
    main()
