#!/usr/bin/env bash
# Motif installer — Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/AdityaWagh19/Motif/main/scripts/install.sh | bash
set -euo pipefail

MOTIF_REPO="${MOTIF_REPO:-https://github.com/AdityaWagh19/Motif}"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
LLAMA_CPP_CUDA_INDEX="https://abetlen.github.io/llama-cpp-python/whl"
LLAMA_CPP_ROCM_INDEX="https://abetlen.github.io/llama-cpp-python/whl/rocm"

# ── Colour helpers ────────────────────────────────────────────────────────────
BOLD="\033[1m"; RESET="\033[0m"; GREEN="\033[32m"; CYAN="\033[36m"
YELLOW="\033[33m"; RED="\033[31m"; GRAY="\033[90m"; WHITE="\033[97m"

step()    { printf "\n  ${GRAY}[${WHITE}%s${GRAY}]${RESET} ${CYAN}%s${RESET}\n" "$1" "$2"; }
ok()      { printf "  ${GRAY}[${GREEN}ok${GRAY}]${RESET} %s\n" "$*"; }
warn()    { printf "  ${GRAY}[${YELLOW}!!${GRAY}]${RESET} ${YELLOW}%s${RESET}\n" "$*"; }
die()     { printf "\n  ${GRAY}[${RED}!!${GRAY}]${RESET} ${RED}%s${RESET}\n" "$*" >&2; exit 1; }

spinner() {
    # Usage: spinner <pid> <message>
    local pid=$1 msg=$2
    local frames=("   [  ]" "   [. ]" "   [..]" "   [...]")
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r${GRAY}%s${RESET} ${CYAN}%s${RESET}" "${frames[$((i % 4))]}" "$msg"
        sleep 0.12
        (( i++ )) || true
    done
    # Erase the spinner line
    printf "\r%*s\r" $(( ${#msg} + 12 )) ""
}

# ── Banner ────────────────────────────────────────────────────────────────────
printf "\n  ${BOLD}${WHITE}Motif${RESET}  ${GRAY}—  offline multimodal RAG${RESET}\n"
printf "  ${GRAY}https://github.com/AdityaWagh19/Motif${RESET}\n"
printf "  ${GRAY}─────────────────────────────────────${RESET}\n"

# ── Step 1: Ensure uv ────────────────────────────────────────────────────────
step "1/3" "Checking package manager (uv)..."
if command -v uv &>/dev/null; then
    ok "uv already installed  ($(uv --version))"
else
    printf "      ${GRAY}Downloading uv...${RESET}\n"
    curl -LsSf "$UV_INSTALL_URL" | sh
    export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
    command -v uv &>/dev/null || die "uv installation failed. See: https://docs.astral.sh/uv/"
    ok "uv installed  ($(uv --version))"
fi

# ── Step 2: Install Motif ────────────────────────────────────────────────────
step "2/3" "Installing Motif..."

if [ "${MOTIF_REPO}" = "." ] || [ -d "${MOTIF_REPO}" ]; then
    INSTALL_SPEC="${MOTIF_REPO}"
elif [[ "${MOTIF_REPO}" == git+* ]]; then
    INSTALL_SPEC="${MOTIF_REPO}"
else
    INSTALL_SPEC="git+${MOTIF_REPO}"
fi

PYTHON_BIN="${PYTHON:-python3}"

if [ "$(uname)" = "Darwin" ]; then
    uv tool install "${INSTALL_SPEC}" --python "${PYTHON_BIN}" \
        --no-binary-package llama-cpp-python --force --quiet &
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        uv tool install "${INSTALL_SPEC}" --python "${PYTHON_BIN}" \
            --find-links "${LLAMA_CPP_CUDA_INDEX}/cu124/llama-cpp-python/" --force --quiet &
    elif [ -d "/opt/rocm" ] || lsmod 2>/dev/null | grep -q amdgpu; then
        uv tool install "${INSTALL_SPEC}" --python "${PYTHON_BIN}" \
            --find-links "${LLAMA_CPP_ROCM_INDEX}/llama-cpp-python/" --force --quiet &
    else
        uv tool install "${INSTALL_SPEC}" --python "${PYTHON_BIN}" \
            --find-links "${LLAMA_CPP_CUDA_INDEX}/cpu/llama-cpp-python/" --force --quiet &
    fi
fi

INSTALL_PID=$!
spinner $INSTALL_PID "Resolving and installing packages  (this takes ~1–2 min on first run)..."
wait $INSTALL_PID || die "Motif installation failed."
ok "Motif installed"

uv tool update-shell 2>/dev/null || true
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
MOTIF_ENV="$(uv tool dir 2>/dev/null)/motif-rag"

# ── Step 3: GPU / accelerator setup ─────────────────────────────────────────
step "3/3" "Detecting GPU accelerator..."

# ── 3a. NVIDIA CUDA ──────────────────────────────────────────────────────────
CUDA_VERSION=""
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi 2>/dev/null \
        | grep -oP "CUDA(?: UMD)? Version:\s*\K[\d.]+" \
        || echo "")
fi

if [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 13 ]; then
        CUDA_MAJOR_MINOR="12.4"
    else
        CUDA_MAJOR_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f1,2)
    fi
    CUDA_TAG="cu$(echo "$CUDA_MAJOR_MINOR" | tr -d '.')"

    ok "NVIDIA GPU detected  (CUDA ${CUDA_VERSION} → wheel tag: ${CUDA_TAG})"

    if [ -n "$MOTIF_ENV" ]; then
        uv pip install llama-cpp-python \
            --python "${MOTIF_ENV}/bin/python" \
            --extra-index-url "${LLAMA_CPP_CUDA_INDEX}/${CUDA_TAG}" \
            --force-reinstall --only-binary llama-cpp-python --quiet &
        GPU_PID=$!
        spinner $GPU_PID "Installing GPU-accelerated llama-cpp-python (${CUDA_TAG})..."
        wait $GPU_PID && ok "GPU-accelerated llama-cpp-python installed" \
            || warn "GPU wheel not found for ${CUDA_TAG} — falling back to CPU inference"
    else
        warn "Could not locate Motif environment. CUDA wheel not installed."
    fi

# ── 3b. Apple Silicon (Metal) ────────────────────────────────────────────────
elif [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    ok "Apple Silicon  —  Metal GPU enabled automatically"
    RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    RAM_GB=$(( RAM_BYTES / 1073741824 ))
    if   [ "$RAM_GB" -ge 16 ]; then ok "Unified memory: ${RAM_GB} GB  →  Tier T3 (full Metal offload)"
    elif [ "$RAM_GB" -ge 8  ]; then ok "Unified memory: ${RAM_GB} GB  →  Tier T2 (partial Metal offload)"
    else                             ok "Unified memory: ${RAM_GB} GB  →  Tier T1 (CPU)"
    fi

# ── 3c. AMD ROCm ─────────────────────────────────────────────────────────────
elif command -v rocm-smi &>/dev/null; then
    ok "AMD ROCm GPU detected"
    if [ -n "$MOTIF_ENV" ]; then
        uv pip install llama-cpp-python \
            --python "${MOTIF_ENV}/bin/python" \
            --extra-index-url "${LLAMA_CPP_ROCM_INDEX}" \
            --force-reinstall --only-binary llama-cpp-python --quiet &
        ROCM_PID=$!
        spinner $ROCM_PID "Installing ROCm llama-cpp-python..."
        wait $ROCM_PID && ok "ROCm llama-cpp-python installed" \
            || warn "ROCm wheel not found — falling back to CPU inference"
    else
        warn "Could not locate Motif environment. ROCm wheel not installed."
    fi

# ── 3d. CPU fallback ─────────────────────────────────────────────────────────
else
    warn "No GPU detected  —  CPU inference (Tier T1)"
    warn "Expect ~2–3 min per answer for 7B models. Phi-3.5-mini is ~11 s."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
printf "\n  ${GRAY}─────────────────────────────────────${RESET}\n"
printf "  ${GREEN}${BOLD}Installation complete.${RESET}\n\n"
printf "  ${WHITE}Next steps:${RESET}\n"
printf "    ${CYAN}motif setup${RESET}  ${GRAY}—  download models for your hardware${RESET}\n"
printf "    ${CYAN}motif${RESET}        ${GRAY}—  start chatting${RESET}\n\n"

if ! command -v motif &>/dev/null; then
    warn "Restart your terminal to pick up the new PATH, then run  motif setup"
fi
