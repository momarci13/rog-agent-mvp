# ROG-Agent MVP setup script (Windows PowerShell).
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1
#
# If you prefer WSL2 (recommended on the X13 — better Linux tool support
# and proper resource limits in the sandbox), use setup.sh inside WSL.

$ErrorActionPreference = "Stop"
Write-Host "==> ROG-Agent MVP setup (Windows)" -ForegroundColor Cyan

# ---------- 1. Python venv ----------
if (-Not (Test-Path ".venv")) {
    Write-Host "[1/4] Creating Python venv..."
    python -m venv .venv
}
& ".venv\Scripts\Activate.ps1"

Write-Host "[1/4] Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools | Out-Null

Write-Host "[2/4] Installing requirements (this can take ~5 min)..."
pip install -r requirements.txt

# ---------- 2. Ollama check ----------
Write-Host "[3/4] Checking Ollama..."
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-Not $ollama) {
    Write-Host "  Ollama is NOT installed." -ForegroundColor Yellow
    Write-Host "  Download the Windows installer from: https://ollama.com/download/windows"
    Write-Host "  Run it, then re-run this script."
    exit 1
}

# On Windows the Ollama desktop app starts a tray service automatically.
# Verify it's listening.
try {
    $r = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3
    Write-Host "  Ollama service is running."
} catch {
    Write-Host "  Ollama is installed but not responding on :11434." -ForegroundColor Yellow
    Write-Host "  Open the Ollama app from the Start menu, then re-run this script."
    exit 1
}

# Read model name from config
$cfg = Get-Content configs\config.yaml -Raw
if ($cfg -match 'model:\s*"([^"]+)"') {
    $model = $matches[1]
} else {
    $model = "qwen2.5:7b-instruct-q4_K_M"
}
Write-Host "  Pulling model: $model  (~4.7 GB; one-time download)"
ollama pull $model

# ---------- 3. LaTeX (optional) ----------
Write-Host "[4/4] Checking LaTeX..."
$tectonic = Get-Command tectonic -ErrorAction SilentlyContinue
$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
if ($tectonic) {
    Write-Host "  tectonic OK"
} elseif ($pdflatex) {
    Write-Host "  pdflatex OK"
} else {
    Write-Host "  (optional) No LaTeX engine found. Writing tasks will save .tex but not .pdf."
    Write-Host "  To install tectonic on Windows:"
    Write-Host "    1. Download from https://github.com/tectonic-typesetting/tectonic/releases"
    Write-Host "    2. Unzip tectonic.exe to a folder on your PATH"
    Write-Host "  Or install MiKTeX (full LaTeX): https://miktex.org/download"
}

# ---------- 4. Healthcheck ----------
Write-Host ""
Write-Host "==> Running healthcheck..."
python run.py --healthcheck

Write-Host ""
Write-Host "==> Setup complete." -ForegroundColor Green
Write-Host "    Activate the venv:    .venv\Scripts\Activate.ps1"
Write-Host "    Try a task:           python run.py 'Compute volatility of SPY 2020-2024'"
Write-Host "    Build the KB:         python run.py --ingest data\papers\"
