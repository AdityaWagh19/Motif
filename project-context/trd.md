# Technical Requirements Document — Motif Offline Multimodal RAG

> **Depends on:** `context.md`, `architecture.md`  
> **Purpose:** Machine-testable acceptance criteria for every subsystem. If it is not in this document, it is not a requirement.

---

## 1. Ingestion Requirements

### 1.1 Supported File Types

| Requirement | ID | Criterion |
|---|---|---|
| PDF (text) ingested correctly | ING-01 | Given a 10-page text PDF, all body text is extracted with no character loss on standard fonts |
| PDF (scanned) ingested on T2/T3 | ING-02 | Given a scanned PDF, PaddleOCR extracts ≥ 90% of body text characters (measured vs ground truth) |
| PDF (scanned) layout preserved on T3 | ING-03 | Surya correctly identifies column layout in a 2-column academic PDF |
| DOCX ingested correctly | ING-04 | Tables are serialized as markdown tables; headings become section_title in metadata |
| Markdown ingested correctly | ING-05 | Heading hierarchy preserved as section_title; code blocks extracted as separate chunks |
| Images ingested (OCR) | ING-06 | A PNG containing typed text yields ≥ 85% character accuracy via PaddleOCR |
| Audio transcribed correctly | ING-07 | whisper.cpp transcribes a 1-minute English MP3 with ≤ 8% WER |
| Timestamps in audio chunks | ING-08 | Each audio chunk has `start_time` and `end_time` populated in ChunkMetadata |
| Unsupported format rejected gracefully | ING-09 | A `.xlsx` file logs an error and continues ingestion without crashing |

### 1.2 Chunking

| Requirement | ID | Criterion |
|---|---|---|
| Chunk size within bounds | ING-10 | 95% of chunks have token_count between 64 and 640 tokens |
| No chunk truncates a sentence | ING-11 | No chunk ends mid-sentence (sentence splitter) or mid-semantic-unit (semantic chunker) |
| Overlap is applied | ING-12 | Consecutive chunks from the same document share ≥ 40 and ≤ 80 overlapping tokens |
| Tables kept intact | ING-13 | A detected table is never split across two chunks |
| Deduplication works | ING-14 | Re-ingesting the same document produces zero new chunks in the index |

### 1.3 Indexing

| Requirement | ID | Criterion |
|---|---|---|
| All chunks indexed in Qdrant | ING-15 | After ingestion, `VectorStore.count()` equals `ChunkStore.count()` |
| All chunks indexed in BM25 | ING-16 | After ingestion, BM25 returns results for exact phrases present in the corpus |
| Ingestion tracker updated | ING-17 | `IngestionTracker.is_indexed(filepath)` returns True after ingestion |
| Incremental ingestion correct | ING-18 | Running `ingest` twice on the same directory adds 0 new chunks on second run |
| Delete works | ING-19 | After `cli.py remove doc.pdf`, `VectorStore.count()` decreases by exact chunk count of that document |
| Sync detects deletions | ING-20 | After deleting a source file and running `cli.py sync`, all its chunks are removed |

### 1.4 Performance

| Requirement | ID | Criterion |
|---|---|---|
| Ingestion speed (T2/T3) | ING-21 | 100 typical PDF pages (text) indexed in ≤ 10 minutes |
| Ingestion speed (T1) | ING-22 | 100 typical PDF pages (text) indexed in ≤ 15 minutes |
| Peak RAM during ingestion (T1) | ING-23 | `psutil.Process().memory_info().rss` < 5.0 GB with LLM not loaded |
| moondream2 unloaded after ingestion | ING-24 | After `ModelManager.after_ingestion()`, moondream2 not in `ModelManager._models` |

---

## 2. Retrieval Requirements

### 2.1 Dense + Sparse + BM25 Hybrid

| Requirement | ID | Criterion |
|---|---|---|
| Dense retrieval returns results | RET-01 | Given any encoded query, Qdrant returns ≥ 1 result with a finite score |
| Sparse retrieval returns results | RET-02 | Qdrant sparse search returns results for queries with known corpus terms |
| BM25 handles exact matches | RET-03 | A product code or exact phrase from the corpus ranks in BM25 top-3 |
| RRF fusion produces unified ranking | RET-04 | `rrf_fuse()` returns exactly `min(top_k_retrieval, total_chunks)` results |
| RRF correctly boosts multi-list hits | RET-05 | A chunk appearing in all three lists ranks higher than one appearing in one |

### 2.2 Recall

| Requirement | ID | Criterion |
|---|---|---|
| Recall@20 on eval set | RET-06 | The ground-truth passage appears in top-20 retrieved for ≥ 75% of eval queries |
| Exact-match recall | RET-07 | For queries containing an exact 3+ word phrase from corpus, BM25 returns the correct chunk in top-5 |

### 2.3 Metadata Filtering

| Requirement | ID | Criterion |
|---|---|---|
| Filename filter works | RET-08 | `--file report.pdf` returns only chunks from `report.pdf` |
| Type filter works | RET-09 | `--type audio` returns only audio chunks |
| Page range filter works | RET-10 | `--pages 5-10` returns only chunks with `page_number` in [5, 10] |
| Filter does not crash on no results | RET-11 | An overly restrictive filter returns an empty list, not an exception |

### 2.4 Performance

| Requirement | ID | Criterion |
|---|---|---|
| Retrieval latency (T1) | RET-12 | Dense + sparse + BM25 + fusion completes in ≤ 500ms |
| Retrieval latency (T2/T3) | RET-13 | Dense + sparse + BM25 + fusion completes in ≤ 100ms |

---

## 3. Reranking Requirements

| Requirement | ID | Criterion |
|---|---|---|
| Reranker always runs | RER-01 | CrossEncoder is invoked on every query — no code path bypasses it |
| Precision improvement | RER-02 | Reranked top-5 has ≥ 10% higher precision than pre-rerank top-5 on eval set |
| Relevance threshold applied | RER-03 | Passages with reranker score < threshold are excluded from context |
| Threshold auto-calibration | RER-04 | On first run with eval queries, `calibrate_threshold()` sets a value in [0.2, 0.5] |
| Reranking latency (T2) | RER-05 | Reranking 20 query-passage pairs completes in ≤ 200ms |
| Reranking latency (T3) | RER-06 | Reranking 20 query-passage pairs completes in ≤ 100ms |

---

## 4. Generation Requirements

### 4.1 Answer Quality

| Requirement | ID | Criterion |
|---|---|---|
| RAGAS faithfulness (T2/T3) | GEN-01 | ≥ 85% on synthetic corpus eval set |
| RAGAS faithfulness (T1) | GEN-02 | ≥ 75% on synthetic corpus eval set |
| Answer relevancy (T2/T3) | GEN-03 | ≥ 85% RAGAS answer_relevancy |
| No hallucination on unanswerable | GEN-04 | Given a question with no relevant passages (threshold failure), LLM responds "not found in documents" |
| Citations present | GEN-05 | Every answer includes ≥ 1 citation unless the no-relevant-passage response is triggered |

### 4.2 Context Construction

| Requirement | ID | Criterion |
|---|---|---|
| Context within token budget | GEN-06 | Total context tokens ≤ `config.generation.context_max_tokens` on every query |
| Anti-middle ordering applied | GEN-07 | Rank-1 passage is always at position 0 in the context string |
| Adjacent chunks merged | GEN-08 | Two consecutive chunks from the same source appear as one block in context |
| Extractive compression triggers | GEN-09 | When raw context exceeds budget, compress to budget without dropping the top-1 passage |

### 4.3 Prompt Behavior

| Requirement | ID | Criterion |
|---|---|---|
| Low temperature | GEN-10 | `temperature = 0.1` on all tiers (hardcoded, not user-overridable) |
| System prompt enforced | GEN-11 | System prompt includes explicit "answer only from provided context" instruction |

### 4.4 Streaming

| Requirement | ID | Criterion |
|---|---|---|
| Streaming works | GEN-12 | First token appears in terminal ≤ 3s on T2 (before full answer completes) |
| Streaming does not corrupt citations | GEN-13 | Citations are appended only after streaming is complete |

### 4.5 Latency

| Requirement | ID | Criterion |
|---|---|---|
| P95 latency (T1, no HyDE) | GEN-14 | ≤ 13s end-to-end over 100 diverse queries |
| P95 latency (T2, adaptive HyDE) | GEN-15 | ≤ 8s end-to-end over 100 diverse queries |
| P95 latency (T3, adaptive HyDE) | GEN-16 | ≤ 5s end-to-end over 100 diverse queries |

---

## 5. Storage Requirements

| Requirement | ID | Criterion |
|---|---|---|
| SQLite WAL mode enabled | STO-01 | `PRAGMA journal_mode = WAL` verified on connection |
| Chunk fetch by ID | STO-02 | `ChunkStore.fetch(chunk_id)` returns in ≤ 5ms per chunk |
| Batch chunk fetch | STO-03 | Fetching 20 chunks by ID completes in ≤ 20ms |
| Index size scales linearly | STO-04 | 10K chunk index ≤ 50 MB; 100K chunk index ≤ 500 MB (Qdrant on_disk) |
| Query cache hit | STO-05 | Second identical query returns cached answer in ≤ 50ms |
| Cache privacy warning | STO-06 | On startup with `query_cache_enabled = true`, a yellow warning is printed once |

---

## 6. REPL & Interface Requirements

### 6.1 Interactive Session

| Requirement | ID | Criterion |
|---|---|---|
| REPL launches correctly | REPL-01 | `motif` starts without error; welcome screen renders within 3s |
| Welcome screen shows system info | REPL-02 | Welcome panel shows: version, tier, model name, chunk count, document count, working dir |
| Prompt accepts plain-text queries | REPL-03 | Typing a non-slash string and pressing Enter triggers the query pipeline |
| Prompt accepts slash commands | REPL-04 | Typing `/ingest ./docs` triggers the ingest command; output appears inline |
| Arrow-key history works | REPL-05 | Up arrow recalls previous query; Down arrow moves forward (prompt_toolkit built-in) |
| Tab completion for slash commands | REPL-06 | Typing `/i` + Tab completes to `/ingest` |
| Ctrl+C exits gracefully | REPL-07 | Ctrl+C during a streaming answer stops generation cleanly; Ctrl+C at prompt exits after saving history |
| `exit` / `quit` exits cleanly | REPL-08 | Typing `exit` or `quit` saves session history and exits with code 0 |
| One-shot mode works | REPL-09 | `motif ask "query"` prints answer and exits (for scripting) |
| One-shot ingest works | REPL-10 | `motif ingest ./docs` ingests and exits (for scripting) |

### 6.2 Slash Commands

| Command | ID | Criterion |
|---|---|---|
| `/ingest PATH` | CMD-01 | Ingests all supported files in PATH; prints count of files processed |
| `/ingest PATH -r` | CMD-02 | Recursively ingests all supported files in subdirectories |
| `/remove PATH` | CMD-03 | Removes all chunks for the file; prints count removed |
| `/sync DIR` | CMD-04 | Adds new files, removes deleted files, re-ingests changed files |
| `/status` | CMD-05 | Prints: document count, chunk count, index size, loaded models, detected tier |
| `/clear` | CMD-06 | Clears in-memory history list and deletes `~/.ragdb/history.json`; prints confirmation |
| `/new` | CMD-07 | Archives current history to `~/.ragdb/history_TIMESTAMP.json`; starts fresh session |
| `/setup` | CMD-08 | Runs model download for detected tier; equivalent to `motif setup` one-shot |
| `/help` | CMD-09 | Prints all available commands with one-line descriptions |
| Unknown command error | CMD-10 | Unknown `/xyz` prints "Unknown command: /xyz. Type /help for available commands." |

### 6.3 Conversation History

| Requirement | ID | Criterion |
|---|---|---|
| History appended after each query | HST-01 | After each answered query, `session.history` grows by exactly one `{role:user}` + one `{role:assistant}` entry |
| Rolling window enforced | HST-02 | History passed to LLM never includes more turns than fit within the remaining context budget after retrieved passages |
| Retrieved passages take priority | HST-03 | If budget forces a choice, passages are kept and oldest history turns are dropped first |
| History persisted on exit | HST-04 | `~/.ragdb/history.json` contains the full session history after clean exit |
| History loaded on restart | HST-05 | On next `motif` launch, history.json is loaded and last query is shown in welcome screen |
| Empty history valid | HST-06 | A fresh install with no history.json starts with empty history and no error |
| `/clear` resets completely | HST-07 | After `/clear`, `session.history == []` and `history.json` is deleted |

### 6.4 Installer

| Requirement | ID | Criterion |
|---|---|---|
| Linux/macOS install succeeds | INS-01 | `curl -fsSL .../install.sh \| bash` installs `motif` command on a clean Ubuntu 22.04 system |
| Windows install succeeds | INS-02 | `irm .../install.ps1 \| iex` installs `motif` command on a clean Windows 11 system |
| Python auto-installed if missing | INS-03 | If Python < 3.11 or absent, installer bootstraps uv which installs Python 3.11 automatically |
| CUDA wheel attempted on GPU systems | INS-04 | Installer detects CUDA via `nvidia-smi`; attempts pre-built CUDA wheel; falls back to CPU with warning |
| `motif setup` downloads correct models | INS-05 | After install, `motif setup` detects tier and downloads the correct model set with progress bars |
| Installer is idempotent | INS-06 | Running the install script twice does not break the existing installation |

---

## 7. Non-Functional Requirements

| Requirement | ID | Criterion |
|---|---|---|
| Fully offline | NFR-01 | No outbound network calls after model download; all inference is local |
| Disk footprint (T1) | NFR-02 | `du -sh models/` ≤ 3.0 GB |
| Disk footprint (T2) | NFR-03 | `du -sh models/` ≤ 5.0 GB |
| Disk footprint (T3 base) | NFR-04 | `du -sh models/` ≤ 5.0 GB (moondream2 is opt-in, not counted in base) |
| Python version | NFR-05 | `python --version` ≥ 3.11 |
| No server process | NFR-06 | Qdrant runs as embedded library; no background daemon |
| Config file in TOML | NFR-07 | `config.toml` is valid TOML and loads without error |
| Graceful shutdown | NFR-08 | Ctrl+C during ingestion does not corrupt the Qdrant index or SQLite database |
| Logging to file | NFR-09 | All log entries written to `~/.ragdb/motif.log` with timestamps |
| Single-user | NFR-10 | No concurrent access handling required; SQLite WAL is sufficient |
| `motif` on PATH after install | NFR-11 | After running install script, `which motif` (Linux/macOS) or `Get-Command motif` (Windows) returns a valid path |
