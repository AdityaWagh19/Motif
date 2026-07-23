# Motif installer — Windows PowerShell
# Usage: irm https://raw.githubusercontent.com/AdityaWagh19/Motif/main/scripts/install.ps1 | iex
#Requires -Version 5.1

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$MotifRepo      = "https://github.com/AdityaWagh19/Motif"  # Add @<tag> here for version-pinning
$UvInstallUrl   = "https://astral.sh/uv/install.ps1"
$LlamaCppIndex  = "https://abetlen.github.io/llama-cpp-python/whl"
$LlamaCppRocm   = "https://abetlen.github.io/llama-cpp-python/whl/rocm"

# ── Formatting helpers ────────────────────────────────────────────────────────
function Write-Header($msg)  { Write-Host "`n$msg" -ForegroundColor White }
function Write-Info($msg)    { Write-Host "  --> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)      { Write-Host "   ok $msg" -ForegroundColor Green }
function Write-Warn($msg)    { Write-Host " warn $msg" -ForegroundColor Yellow }
function Write-Fail($msg)    { Write-Host "error $msg" -ForegroundColor Red; exit 1 }

# ── Header ────────────────────────────────────────────────────────────────────
Write-Header "Motif - offline multimodal RAG"
Write-Host "  https://github.com/AdityaWagh19/Motif`n"

# ── Step 1: Ensure uv is present ──────────────────────────────────────────────
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCmd) {
    $uvVer = & uv --version 2>&1
    Write-Ok "uv already installed: $uvVer"
} else {
    Write-Info "Installing uv (Python package manager)..."
    try {
        Invoke-RestMethod $UvInstallUrl | Invoke-Expression
    } catch {
        Write-Fail "Failed to install uv: $_"
    }
    # Add common uv locations to PATH for this session
    $uvPaths = @(
        "$env:USERPROFILE\.cargo\bin",
        "$env:USERPROFILE\.local\bin",
        "$env:APPDATA\uv\bin"
    )
    foreach ($p in $uvPaths) {
        if (Test-Path $p) { $env:PATH = "$p;$env:PATH" }
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Fail "uv installation failed. Please install manually: https://docs.astral.sh/uv/"
    }
    Write-Ok "uv installed: $(uv --version)"
}

# ── Step 2: Install Motif ─────────────────────────────────────────────────────
Write-Info "Installing motif..."
& uv tool install "git+$MotifRepo" --extra-index-url "$LlamaCppIndex/cpu" --upgrade
if ($LASTEXITCODE -ne 0) { Write-Fail "Motif installation failed." }
Write-Ok "motif installed"

# Add uv tool bin dir to PATH for this session
& uv tool update-shell 2>$null

$UvToolDir = uv tool dir 2>$null
if ($UvToolDir) {
    $MotifEnv = Join-Path $UvToolDir "motif-rag"
} else {
    $MotifEnv = $null
}

# ── Step 3: GPU / accelerator detection ──────────────────────────────────────
if ([string]::IsNullOrWhiteSpace($MotifEnv) -or -not (Test-Path $MotifEnv)) {
    Write-Fail "Could not determine Motif tool environment. Installation aborted."
}

# ── 3a. NVIDIA CUDA ───────────────────────────────────────────────────────────
$CudaVersion = ""
try {
    $NvSmiOut = & nvidia-smi 2>$null
    if ($NvSmiOut -match "CUDA Version:\s+([\d.]+)") {
        $CudaVersion = $matches[1]
    }
} catch { }

if ($CudaVersion) {
    # Bug #7 fix: Take only major.minor components to build the wheel tag.
    # "12.4.0" → "12.4" → "cu124"  (cu1240 is invalid and causes silent fallback to CPU)
    $CudaShort = ($CudaVersion -split '\.')[0..1] -join '.'
    $CudaTag = "cu" + $CudaShort.Replace(".", "")

    Write-Info "NVIDIA GPU detected - CUDA $CudaVersion (wheel tag: $CudaTag)."
    Write-Info "Installing GPU-enabled llama-cpp-python ($CudaTag pre-built wheel)..."

    if ($MotifEnv) {
        $pythonExe = Join-Path $MotifEnv "Scripts\python.exe"
        try {
            & uv pip install llama-cpp-python `
                --python $pythonExe `
                --extra-index-url "$LlamaCppIndex/$CudaTag" `
                --force-reinstall `
                --only-binary llama-cpp-python `
                --quiet
            Write-Ok "llama-cpp-python with CUDA $CudaVersion support installed"

            # Auto-provision CUDA 12 runtime DLLs if missing in environment
            $LlamaLib = Join-Path $MotifEnv "Lib\site-packages\llama_cpp\lib"
            if (Test-Path $LlamaLib) {
                $NeedDlls = @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
                $Missing = $NeedDlls | Where-Object { -not (Test-Path (Join-Path $LlamaLib $_)) }
                if ($Missing) {
                    Write-Info "Provisioning CUDA runtime DLLs for Windows GPU acceleration..."
                    & $pythonExe -c "
import urllib.request, zipfile, io, os
lib = r'$LlamaLib'
for url, dlls in [
    ('https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.4.127-archive.zip', ['cudart64_12.dll']),
    ('https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.4.5.8-archive.zip', ['cublas64_12.dll', 'cublasLt64_12.dll'])
]:
    try:
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read()))
        for m in z.namelist():
            if any(m.endswith(d) for d in dlls):
                with z.open(m) as src, open(os.path.join(lib, os.path.basename(m)), 'wb') as dst:
                    dst.write(src.read())
    except Exception as e:
        print('CUDA DLL provision notice:', e)
"
                    Write-Ok "CUDA runtime DLLs provisioned"
                }
            }
        } catch {
            Write-Warn "Pre-built CUDA wheel not found for $CudaTag. Falling back to CPU inference."
            Write-Warn "To retry manually:"
            Write-Warn "  uv pip install llama-cpp-python --extra-index-url $LlamaCppIndex/$CudaTag --force-reinstall"
        }
    } else {
        Write-Warn "Could not locate Motif tool environment. CUDA wheel not installed."
    }

# ── 3b. AMD ROCm ──────────────────────────────────────────────────────────────
} elseif (Get-Command rocm-smi -ErrorAction SilentlyContinue) {
    Write-Info "AMD ROCm GPU detected."
    Write-Info "Installing ROCm-enabled llama-cpp-python..."

    if ($MotifEnv) {
        $pythonExe = Join-Path $MotifEnv "Scripts\python.exe"
        try {
            & uv pip install llama-cpp-python `
                --python $pythonExe `
                --extra-index-url $LlamaCppRocm `
                --force-reinstall `
                --only-binary llama-cpp-python `
                --quiet
            Write-Ok "llama-cpp-python with ROCm support installed"
        } catch {
            Write-Warn "Pre-built ROCm wheel not found. Falling back to CPU inference."
            Write-Warn "To retry manually:"
            Write-Warn "  uv pip install llama-cpp-python --extra-index-url $LlamaCppRocm --force-reinstall"
        }
    } else {
        Write-Warn "Could not locate Motif tool environment. ROCm wheel not installed."
    }

# ── 3c. CPU fallback ──────────────────────────────────────────────────────────
} else {
    Write-Info "No GPU accelerator detected. CPU inference will be used (Tier T1)."
    Write-Info "Generation will work but will be slower (~2-3 min P50 for 7B models)."
    Write-Info "Phi-3.5-mini (T1 model) is much faster: ~11 s P95 on modern CPUs."
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host "`nInstallation complete.`n" -ForegroundColor Green
Write-Host "  Download models for your hardware:`n"
Write-Host "    motif setup`n"
Write-Host "  Then start using Motif:`n"
Write-Host "    motif`n"

# Warn if motif is not found yet
if (-not (Get-Command motif -ErrorAction SilentlyContinue)) {
    Write-Warn "'motif' not found in current PATH."
    Write-Warn "Restart PowerShell to apply PATH changes."
}
