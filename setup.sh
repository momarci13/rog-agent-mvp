#!/usr/bin/env bash
# ROG-Agent MVP setup script (Linux / macOS / WSL2).
# Usage:  bash setup.sh

set -e

echo "==> ROG-Agent MVP setup"

# ---------- 1. Python venv ----------
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating Python venv..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[1/4] Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools >/dev/null

echo "[2/4] Installing requirements (this can take ~5 min)..."
pip install -r requirements.txt

# ---------- 2. Ollama check ----------
echo "[3/4] Checking Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
    echo "  Ollama is NOT installed."
    echo "  Install it with:"
    echo "    Linux:  curl -fsSL https://ollama.com/install.sh | sh"
    echo "    macOS:  brew install ollama   (or download from https://ollama.com)"
    echo "  Then re-run this script."
    exit 1
fi

# Start ollama if not running (Linux). On macOS, the desktop app handles this.
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "  Starting ollama serve in the background..."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    sleep 3
fi

MODEL="$(grep -E '^\s*model:' configs/config.yaml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
if [ -z "$MODEL" ]; then
    MODEL="qwen2.5:7b-instruct-q4_K_M"
fi
echo "  Pulling model: $MODEL  (~4.7 GB; one-time download)"
ollama pull "$MODEL"

# ---------- 3. LaTeX (optional) ----------
echo "[4/4] Checking LaTeX..."
if command -v tectonic >/dev/null 2>&1; then
    echo "  tectonic OK"
elif command -v pdflatex >/dev/null 2>&1; then
    echo "  pdflatex OK"
else
    echo "  (optional) No LaTeX engine found. Writing tasks will save .tex but not .pdf."
    echo "  To install tectonic (recommended, single binary, auto-fetches packages):"
    echo "    curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh"
    echo "    sudo mv tectonic /usr/local/bin/"
fi

# ---------- 4. Healthcheck ----------
echo
echo "==> Running healthcheck..."
python run.py --healthcheck || true

echo
echo "==> Setup complete."
echo "    Activate the venv with:   source .venv/bin/activate"
echo "    Try a task with:          python run.py \"Compute volatility of SPY 2020-2024\""
echo "    Build the knowledge base: python run.py --ingest data/papers/"
