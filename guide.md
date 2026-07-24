# Motif RAG — End-User Installation & Verification Guide

This guide describes how an end user installs, sets up, and tests Motif RAG using the official single-line installer scripts. No manual virtual environments or developer tools are required.

---

## Installation Overview

The official installer performs the following automated steps:
1. **Bootstraps `uv`**: Downloads and installs Astral's `uv` package manager if not present.
2. **Tool Environment**: Installs Motif into an isolated global tool environment (`uv tool install`).
3. **Hardware & GPU Detection**: Inspects system GPU/CUDA capability (`nvidia-smi`) and automatically installs the matching pre-built `llama-cpp-python` CUDA wheel (or CPU fallback).
4. **PATH Configuration**: Configures your environment so `motif` is available as a command directly from any terminal.

---

## Step 1: Single-Line Installation

Open a terminal window and run the installer for your operating system:

### Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/AdityaWagh19/Motif/main/scripts/install.ps1 | iex
```

### Linux / macOS (Bash)
```bash
curl -fsSL https://raw.githubusercontent.com/AdityaWagh19/Motif/main/scripts/install.sh | bash
```

---

## Step 2: Automated Model Setup

After installation completes, run `motif setup` to download the models appropriate for your hardware:

```bash
motif setup
```

### What `motif setup` Automates:
- **Hardware Tier Resolution**:
  - **T1 (CPU / < 3.8 GB VRAM)**: Provisions `Phi-3.5-mini-instruct-Q4_K_M.gguf` (2.2 GB).
  - **T2 / T3 (GPU CUDA $\ge$ 4–6 GB VRAM)**: Provisions `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (4.2 GB).
- **Embeddings & Reranker**: Provisions `nomic-embed-text-v1.5` and cross-encoder models into application storage (`~/.ragdb/models`).

---

## Step 3: Verify Installation Status

Verify hardware tier detection and model load readiness:

```bash
motif status
```

### Expected Status Output:
- Displays detected Hardware Tier (`T1`, `T2`, or `T3`).
- Displays model load status (`✓ loaded` or `✓ on disk`).
- Lists active storage database root and workspace.

---

## Step 4: Test End-to-End RAG Usage

### 1. Ingest Documents
Ingest a directory containing markdown, PDF, or text files:

```bash
motif ingest /path/to/your/documents
```

### 2. Launch Interactive REPL Mode
Start the interactive search and query terminal:

```bash
motif
```

### 3. Verify Interactive Features:
- **Query answering**: Type a question about your documents. Answers stream with citations.
- **Chitchat routing**: Type short greetings (`hi`, `hello`) for instant 0ms responses.
- **Query Caching**: Re-run the same question to verify instant LRU cached answers.
- **Slash Commands**: Try `/status`, `/workspace list`, and `/help`.
- **Exit**: Type `/exit` to quit.

---

## Resetting for Fresh Re-testing (Optional)

To wipe all downloaded models and workspace state before re-running the installer test:

### Windows (PowerShell)
```powershell
Remove-Item -Path "models\*" -Recurse -Force -Exclude ".gitkeep" -ErrorAction SilentlyContinue
Remove-Item -Path "$env:LOCALAPPDATA\motif" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$HOME\.ragdb" -Recurse -Force -ErrorAction SilentlyContinue
```

### Linux / macOS
```bash
rm -rf ~/.ragdb ~/.local/share/motif
```
