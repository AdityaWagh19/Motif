#!/usr/bin/env bash
# Motif installer — Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/AdityaWagh19/Motif/main/install.sh | bash
set -euo pipefail

MOTIF_REPO="https://github.com/AdityaWagh19/Motif"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
LLAMA_CPP_CUDA_INDEX="https://abetlen.github.io/llama-cpp-python/whl"

# ── Formatting helpers ────────────────────────────────────────────────────────
bold()    { printf "\033[1m%s\033[0m\n" "$*"; }
info()    { printf "\033[34m-->\033[0m %s\n" "$*"; }
success() { printf "\033[32m  ok\033[0m %s\n" "$*"; }
warn()    { printf "\033[33m warn\033[0m %s\n" "$*"; }
die()     { printf "\033[31merror\033[0m %s\n" "$*" >&2; exit 1; }

# ── Header ────────────────────────────────────────────────────────────────────
echo ""
bold "Motif — offline multimodal RAG"
echo "  https://github.com/AdityaWagh19/Motif"
echo ""

# ── Step 1: Ensure uv is present ──────────────────────────────────────────────
if command -v uv &>/dev/null; then
    success "uv already installed: $(uv --version)"
else
    info "Installing uv (Python package manager)..."
    curl -LsSf "$UV_INSTALL_URL" | sh
    # Make uv available in this shell session
    export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
    command -v uv &>/dev/null || die "uv installation failed. Please install manually: https://docs.astral.sh/uv/"
    success "uv installed: $(uv --version)"
fi

# ── Step 2: Install Motif ─────────────────────────────────────────────────────
info "Installing motif..."
uv tool install "git+${MOTIF_REPO}" --force
success "motif installed"

# Ensure uv tool bin dir is on PATH
uv tool update-shell 2>/dev/null || true

# ── Step 3: GPU / CUDA detection ─────────────────────────────────────────────
CUDA_VERSION=""
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi 2>/dev/null \
        | grep -oP "CUDA Version: \K[\d.]+" \
        || echo "")
fi

if [ -n "$CUDA_VERSION" ]; then
    # Map "12.4" → "cu124", "12.1" → "cu121", etc.
    CUDA_TAG="cu$(echo "$CUDA_VERSION" | tr -d '.')"
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)

    info "NVIDIA GPU detected — CUDA ${CUDA_VERSION}."
    info "Installing GPU-enabled llama-cpp-python (${CUDA_TAG} pre-built wheel)..."

    # Use uv pip inside the tool's isolated environment
    MOTIF_ENV=$(uv tool dir motif 2>/dev/null || echo "")
    if [ -n "$MOTIF_ENV" ]; then
        uv pip install llama-cpp-python \
            --python "${MOTIF_ENV}/bin/python" \
            --extra-index-url "${LLAMA_CPP_CUDA_INDEX}/${CUDA_TAG}" \
            --force-reinstall \
            --quiet 2>/dev/null && \
        success "llama-cpp-python with CUDA ${CUDA_VERSION} support installed" || \
        warn "Pre-built CUDA wheel not found for ${CUDA_TAG}. Falling back to CPU inference."
    else
        warn "Could not locate Motif tool environment. CUDA wheel not installed."
        warn "Re-run with: uv pip install llama-cpp-python --extra-index-url ${LLAMA_CPP_CUDA_INDEX}/${CUDA_TAG} --force-reinstall"
    fi
else
    info "No NVIDIA GPU detected. CPU inference will be used (Tier 1)."
    info "Generation will work but will be slower (~11s P95 latency)."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
bold "Installation complete."
echo ""
echo "  Download models for your hardware:"
echo ""
echo "    motif setup"
echo ""
echo "  Then start using Motif:"
echo ""
echo "    motif"
echo ""

# Warn if motif is not yet on PATH (needs shell restart)
if ! command -v motif &>/dev/null; then
    warn "'motif' not found in current PATH."
    warn "Restart your terminal, or run one of:"
    warn "  source ~/.bashrc    (bash)"
    warn "  source ~/.zshrc     (zsh)"
fi
