#!/usr/bin/env bash
# Motif installer — Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/AdityaWagh19/Motif/main/scripts/install.sh | bash
set -euo pipefail

MOTIF_REPO="https://github.com/AdityaWagh19/Motif"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
LLAMA_CPP_CUDA_INDEX="https://abetlen.github.io/llama-cpp-python/whl"
LLAMA_CPP_ROCM_INDEX="https://abetlen.github.io/llama-cpp-python/whl/rocm"

# ── Formatting helpers ────────────────────────────────────────────────────────
bold()    { printf "\033[1m%s\033[0m\n" "$*"; }
info()    { printf "\033[34m-->\033[0m %s\n" "$*"; }
success() { printf "\033[32m  ok\033[0m %s\n" "$*"; }
warn()    { printf "\033[33m warn\033[0m %s\n" "$*"; }
die()     { printf "\033[31merror\033[0m %s\n" "$*" >&2; exit 1; }

# ── Header ────────────────────────────────────────────────────────────────────
echo ""
bold "Motif - offline multimodal RAG"
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

# ── Step 3: GPU / accelerator detection ──────────────────────────────────────
MOTIF_ENV="$(uv tool dir 2>/dev/null)/motif-rag"

# ── 3a. NVIDIA CUDA ───────────────────────────────────────────────────────────
CUDA_VERSION=""
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi 2>/dev/null \
        | grep -oP "CUDA Version: \K[\d.]+" \
        || echo "")
fi

if [ -n "$CUDA_VERSION" ]; then
    # Bug #6 fix: Take only major.minor (e.g. "12.4" not "12.4.0")
    # nvidia-smi sometimes reports a 3-part version string; cu1240 is invalid.
    CUDA_MAJOR_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f1,2)
    CUDA_TAG="cu$(echo "$CUDA_MAJOR_MINOR" | tr -d '.')"

    info "NVIDIA GPU detected - CUDA ${CUDA_VERSION} (wheel tag: ${CUDA_TAG})."
    info "Installing GPU-enabled llama-cpp-python (${CUDA_TAG} pre-built wheel)..."

    if [ -n "$MOTIF_ENV" ]; then
        uv pip install llama-cpp-python \
            --python "${MOTIF_ENV}/bin/python" \
            --extra-index-url "${LLAMA_CPP_CUDA_INDEX}/${CUDA_TAG}" \
            --force-reinstall \
            --only-binary llama-cpp-python \
            --quiet 2>/dev/null && \
        success "llama-cpp-python with CUDA ${CUDA_VERSION} support installed" || \
        warn "Pre-built CUDA wheel not found for ${CUDA_TAG}. Falling back to CPU inference."
    else
        warn "Could not locate Motif tool environment. CUDA wheel not installed."
        warn "Re-run with: uv pip install llama-cpp-python --extra-index-url ${LLAMA_CPP_CUDA_INDEX}/${CUDA_TAG} --force-reinstall"
    fi

# ── 3b. Apple Silicon (Metal) ─────────────────────────────────────────────────
elif [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    info "Apple Silicon detected - llama.cpp will use Metal (GPU) automatically."
    info "The standard llama-cpp-python wheel includes Metal support on macOS arm64."
    info "No additional install step needed."
    success "Metal GPU inference enabled (llama.cpp built-in)"

    # Retrieve unified memory size for user info
    RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    RAM_GB=$(( RAM_BYTES / 1073741824 ))
    if [ "$RAM_GB" -ge 16 ]; then
        info "Detected ${RAM_GB} GB unified memory → Tier T3 (full Metal offload)"
    elif [ "$RAM_GB" -ge 8 ]; then
        info "Detected ${RAM_GB} GB unified memory → Tier T2 (partial Metal offload)"
    else
        info "Detected ${RAM_GB} GB unified memory → Tier T1 (CPU only recommended)"
    fi

# ── 3c. AMD ROCm ──────────────────────────────────────────────────────────────
elif command -v rocm-smi &>/dev/null; then
    info "AMD ROCm GPU detected."
    info "Installing ROCm-enabled llama-cpp-python..."

    if [ -n "$MOTIF_ENV" ]; then
        uv pip install llama-cpp-python \
            --python "${MOTIF_ENV}/bin/python" \
            --extra-index-url "${LLAMA_CPP_ROCM_INDEX}" \
            --force-reinstall \
            --only-binary llama-cpp-python \
            --quiet 2>/dev/null && \
        success "llama-cpp-python with ROCm support installed" || \
        warn "Pre-built ROCm wheel not found. Falling back to CPU inference."
    else
        warn "Could not locate Motif tool environment. ROCm wheel not installed."
    fi

# ── 3d. CPU fallback ──────────────────────────────────────────────────────────
else
    info "No GPU accelerator detected. CPU inference will be used (Tier T1)."
    info "Generation will work but will be slower (~2-3 min P50 for 7B models)."
    info "Phi-3.5-mini (T1 model) is much faster: ~11 s P95 on modern CPUs."
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
