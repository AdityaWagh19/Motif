# Implementation Plans — Motif Offline Multimodal RAG

> **Purpose:** Self-contained, executable plans for each implementation phase.
> Each plan is a complete specification — no placeholder code, no TODOs, no
> ambiguity. A phase is only considered complete when every validation command
> in its acceptance checklist passes.

---

## Phase Breakdown Rationale

Progress.md tracks five coarse phases (0–5). The plans directory divides these
into seven implementation phases with strict dependency ordering. The
mapping is:

| Plan Phase | Plan Name | Maps to progress.md |
|---|---|---|
| **Phase 1** | Storage Foundation | progress Phase 1 (partial) |
| **Phase 2** | Text Ingestion Pipeline | progress Phase 1 (partial) |
| **Phase 3** | Query Pipeline | progress Phase 1 (complete) |
| **Phase 4** | Quality & Retrieval Hardening | progress Phase 2 |
| **Phase 5** | Multimodal Ingestion | progress Phase 3 |
| **Phase 6** | Evaluation & Production Hardening | progress Phase 4 |
| **Phase 7** | Optional Enhancements | progress Phase 5 |

Phase 0 (Infrastructure) is already complete.

---

## Why This Breakdown

### Dependency constraints

Every phase has strict prerequisites. The division is designed so that:

1. **Phase 1** (Storage) has zero model dependencies — no downloads required,
   no ONNX sessions, pure Python + SQLite. It can be built and fully tested
   immediately.

2. **Phase 2** (Ingestion) depends on Phase 1 storage but only needs the
   embedding model. It can be validated by running `/ingest` and checking
   `/status` counts — no LLM needed.

3. **Phase 3** (Query) depends on Phase 2 having a populated index. It
   introduces the LLM and produces the first end-to-end answer. This is the
   first time a user can ask a question.

4. **Phases 4–7** are quality and feature additions on a working baseline.

### No "big bang" integration

Each phase ends with a working, testable system. There is no phase where the
system is "mostly working but needs one more thing." Each phase gates the next.

---

## Implementation Dependency Graph

```
[Phase 0 — Done]
       │
       ▼
[Phase 1 — Storage]      ← No model deps. Tests run offline.
       │
       ▼
[Phase 2 — Ingestion]    ← Needs: nomic-embed (274 MB)
       │
       ▼
[Phase 3 — Query]        ← Needs: LLM (2.2–4.2 GB) + reranker (134 MB)
       │
       ▼
[Phase 4 — Quality]      ← Needs: Phase 3 + real corpus for eval
       │
       ▼
[Phase 5 — Multimodal]   ← Needs: Phase 4 + whisper (~75 MB) + paddleocr
       │
       ▼
[Phase 6 — Hardening]    ← Needs: Phase 5 + full corpus for RAGAS eval
       │
       ▼
[Phase 7 — Optional]     ← Independent features, order flexible
```

---

## Tracking Rules

After completing each phase:

1. Update `project-context/progress.md` — mark all phase tasks ✅
2. Update `project-context/tests.md` — mark passing tests ✅
3. Add a row to the Metrics Snapshots table in `progress.md`
4. If a deferred decision was resolved, update the Deferred Decisions Log
5. Update the Phase Status Overview table in `progress.md`

**Never mark a phase complete if any acceptance criterion fails.**
If a criterion cannot be met, open a blocker entry in `progress.md`.

---

## File Index

| File | Phase | Description |
|---|---|---|
| `plans/overview.md` | — | This file |
| `plans/phase-1-storage.md` | Phase 1 | Storage foundation |
| `plans/phase-2-ingestion.md` | Phase 2 | Text ingestion pipeline |
| `plans/phase-3-query-pipeline.md` | Phase 3 | Query pipeline |
| `plans/phase-4-quality.md` | Phase 4 | Quality hardening |
| `plans/phase-5-multimodal.md` | Phase 5 | Multimodal ingestion |
| `plans/phase-6-evaluation.md` | Phase 6 | Eval & production hardening |
| `plans/phase-7-optional.md` | Phase 7 | Optional enhancements |
