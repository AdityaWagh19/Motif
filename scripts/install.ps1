# Motif installer - Windows PowerShell
# Usage: irm https://raw.githubusercontent.com/AdityaWagh19/Motif/main/scripts/install.ps1 | iex
#Requires -Version 5.1

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$MotifRepo     = if ($env:MOTIF_REPO) { $env:MOTIF_REPO } else { "https://github.com/AdityaWagh19/Motif" }
$UvInstallUrl  = "https://astral.sh/uv/install.ps1"
$LlamaCppIndex = "https://abetlen.github.io/llama-cpp-python/whl"
$LlamaCppRocm  = "https://abetlen.github.io/llama-cpp-python/whl/rocm"

# ── Colour / formatting helpers ───────────────────────────────────────────────
function c($t, $col) { Write-Host $t -ForegroundColor $col -NoNewline }
function nl { Write-Host "" }

function Write-Step($n, $total, $msg) {
    nl
    c "  [" "DarkGray"; c "$n/$total" "White"; c "] " "DarkGray"
    c $msg "Cyan"
    nl
}
function Write-Ok($msg)   { c "  [" "DarkGray"; c "ok" "Green";  c "] " "DarkGray"; c $msg "White"; nl }
function Write-Warn($msg) { c "  [" "DarkGray"; c "!!" "Yellow"; c "] " "DarkGray"; c $msg "Yellow"; nl }
function Write-Fail($msg) { nl; c "  [" "DarkGray"; c "!!" "Red"; c "] " "DarkGray"; c $msg "Red"; nl; exit 1 }

function Write-Spinner($job, $msg) {
    $frames = @("   [  ] ", "   [. ] ", "   [..] ", "   [...]")
    $i = 0
    while ($job.State -eq "Running") {
        $frame = $frames[$i % $frames.Length]
        Write-Host "`r$frame" -NoNewline -ForegroundColor DarkGray
        Write-Host $msg -NoNewline -ForegroundColor Cyan
        Start-Sleep -Milliseconds 120
        $i++
    }
    Write-Host "`r" -NoNewline
    Write-Host ("   " + (" " * ($msg.Length + 8))) -NoNewline  # erase the line
    Write-Host "`r" -NoNewline
}

# ── Banner ─────────────────────────────────────────────────────────────────────
nl
c "  Motif" "White"; c "  —  offline multimodal RAG" "DarkGray"; nl
c "  https://github.com/AdityaWagh19/Motif" "DarkGray"; nl
c "  ─────────────────────────────────────" "DarkGray"; nl

# ── Step 1: Ensure uv ─────────────────────────────────────────────────────────
Write-Step 1 3 "Checking package manager (uv)..."
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCmd) {
    $uvVer = & uv --version 2>&1
    Write-Ok "uv already installed  ($uvVer)"
} else {
    c "      Downloading uv..." "DarkGray"; nl
    try { Invoke-RestMethod $UvInstallUrl | Invoke-Expression }
    catch { Write-Fail "Failed to install uv: $_" }

    foreach ($p in @("$env:USERPROFILE\.cargo\bin", "$env:USERPROFILE\.local\bin", "$env:APPDATA\uv\bin")) {
        if (Test-Path $p) { $env:PATH = "$p;$env:PATH" }
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Fail "uv installation failed. See: https://docs.astral.sh/uv/"
    }
    Write-Ok "uv installed  ($(uv --version))"
}

# ── Step 2: Install Motif package ─────────────────────────────────────────────
Write-Step 2 3 "Installing Motif..."
$InstallSpec = if ($MotifRepo -eq "." -or (Test-Path $MotifRepo)) { $MotifRepo }
               elseif ($MotifRepo -like "git+*") { $MotifRepo }
               else { "git+$MotifRepo" }
$PythonArgs = if ($env:PYTHON) { @("--python", $env:PYTHON) } else { @() }

$installJob = Start-Job -ScriptBlock {
    param($spec, $pyArgs, $idx)
    & uv tool install $spec @pyArgs --find-links "$idx/cpu/llama-cpp-python/" --upgrade --quiet 2>&1
} -ArgumentList $InstallSpec, $PythonArgs, $LlamaCppIndex

Write-Spinner $installJob "Resolving and installing packages  (this takes ~1–2 min on first run)..."
$installOut = Receive-Job $installJob
Remove-Job $installJob
if ($LASTEXITCODE -ne 0 -and $installOut -match "error") { Write-Fail "Motif installation failed.`n$installOut" }
Write-Ok "Motif installed"

# Clean stale profile aliases
Remove-Item Alias:motif -ErrorAction SilentlyContinue
Remove-Item Function:motif -ErrorAction SilentlyContinue
if ($PROFILE -and (Test-Path $PROFILE)) {
    try {
        $pc = Get-Content $PROFILE -ErrorAction SilentlyContinue
        if ($pc -match "motif.*\.venv") {
            Set-Content -Path $PROFILE -Value ($pc | Where-Object { $_ -notmatch "motif.*\.venv" }) -Force
        }
    } catch { }
}

# Refresh PATH
foreach ($p in @("$env:USERPROFILE\.local\bin", "$env:USERPROFILE\.cargo\bin")) {
    if ((Test-Path $p) -and ($env:PATH -notlike "*$p*")) { $env:PATH = "$p;$env:PATH" }
}
& uv tool update-shell 2>$null

$UvToolDir = uv tool dir 2>$null
$MotifEnv  = if ($UvToolDir) { Join-Path $UvToolDir "motif-rag" } else { $null }
if ([string]::IsNullOrWhiteSpace($MotifEnv) -or -not (Test-Path $MotifEnv)) {
    Write-Fail "Could not locate Motif environment after install."
}

# ── Step 3: GPU / accelerator setup ──────────────────────────────────────────
Write-Step 3 3 "Detecting GPU accelerator..."

# Try all common nvidia-smi paths
$NvSmiPaths = @(
    "nvidia-smi",
    "$env:SystemRoot\System32\nvidia-smi.exe",
    "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    "$env:ProgramW6432\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
)
$CudaVersion = ""
foreach ($nvSmi in $NvSmiPaths) {
    try {
        $NvSmiOut = & $nvSmi 2>$null
        if ($NvSmiOut -match "CUDA(?: UMD)? Version:\s+([\d.]+)") {
            $CudaVersion = $Matches[1]
            break
        }
    } catch { }
}

if ($CudaVersion) {
    $CudaMajor = [int]($CudaVersion -split '\.')[0]
    $CudaShort = if ($CudaMajor -ge 13) { "12.4" } else { ($CudaVersion -split '\.')[0..1] -join '.' }
    $CudaTag   = "cu" + $CudaShort.Replace(".", "")

    Write-Ok "NVIDIA GPU detected  (CUDA $CudaVersion → wheel tag: $CudaTag)"

    $pythonExe = Join-Path $MotifEnv "Scripts\python.exe"
    $gpuJob = Start-Job -ScriptBlock {
        param($pyExe, $idx, $tag)
        & uv pip install llama-cpp-python `
            --python $pyExe `
            --extra-index-url "$idx/$tag" `
            --force-reinstall --only-binary llama-cpp-python --quiet 2>&1
    } -ArgumentList $pythonExe, $LlamaCppIndex, $CudaTag

    Write-Spinner $gpuJob "Installing GPU-accelerated llama-cpp-python ($CudaTag)..."
    $gpuOut = Receive-Job $gpuJob
    Remove-Job $gpuJob

    if ($LASTEXITCODE -eq 0) {
        Write-Ok "GPU-accelerated llama-cpp-python installed"

        # Provision CUDA runtime DLLs if missing
        $LlamaLib = Join-Path $MotifEnv "Lib\site-packages\llama_cpp\lib"
        if (Test-Path $LlamaLib) {
            $NeedDlls = @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
            $Missing  = $NeedDlls | Where-Object { -not (Test-Path (Join-Path $LlamaLib $_)) }
            if ($Missing) {
                $dllJob = Start-Job -ScriptBlock {
                    param($pyExe, $lib)
                    & $pyExe -c "
import urllib.request, zipfile, io, os
lib = r'$lib'
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
        print('notice:', e)
" 2>&1
                } -ArgumentList $pythonExe, $LlamaLib
                Write-Spinner $dllJob "Provisioning CUDA runtime DLLs..."
                Receive-Job $dllJob | Out-Null
                Remove-Job $dllJob
                Write-Ok "CUDA runtime DLLs provisioned"
            }
        }
    } else {
        Write-Warn "GPU wheel not found for $CudaTag — falling back to CPU inference"
        Write-Warn "Retry manually: uv pip install llama-cpp-python --extra-index-url $LlamaCppIndex/$CudaTag --force-reinstall"
    }

} elseif (Get-Command rocm-smi -ErrorAction SilentlyContinue) {
    Write-Ok "AMD ROCm GPU detected"
    $pythonExe = Join-Path $MotifEnv "Scripts\python.exe"
    $rocmJob = Start-Job -ScriptBlock {
        param($pyExe, $rocm)
        & uv pip install llama-cpp-python --python $pyExe --extra-index-url $rocm `
            --force-reinstall --only-binary llama-cpp-python --quiet 2>&1
    } -ArgumentList $pythonExe, $LlamaCppRocm
    Write-Spinner $rocmJob "Installing ROCm llama-cpp-python..."
    Receive-Job $rocmJob | Out-Null
    Remove-Job $rocmJob
    if ($LASTEXITCODE -eq 0) { Write-Ok "ROCm llama-cpp-python installed" }
    else { Write-Warn "ROCm wheel not found — falling back to CPU inference" }

} else {
    Write-Warn "No GPU detected  —  CPU inference (Tier 1)"
    Write-Warn "Expect ~2–3 min per answer for 7B models. Phi-3.5-mini is ~11 s."
}

# ── Done ──────────────────────────────────────────────────────────────────────
nl
c "  ─────────────────────────────────────" "DarkGray"; nl
c "  Installation complete." "Green"; nl
nl
c "  Next steps:" "White"; nl
c "    motif setup" "Cyan"; c "   — download models for your hardware" "DarkGray"; nl
c "    motif" "Cyan";       c "        — start chatting" "DarkGray"; nl
nl

if (-not (Get-Command motif -ErrorAction SilentlyContinue)) {
    Write-Warn "Restart PowerShell to pick up the new PATH, then run  motif setup"
}
