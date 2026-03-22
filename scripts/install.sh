#!/usr/bin/env bash
set -euo pipefail

REQ_FILE="requirements.txt"
VENV_DIR=".venv"
TORCH_MODE="auto"

usage() {
    cat <<'EOF'
Usage: ./scripts/install.sh [options]

Options:
  -r, --requirements PATH   Path to requirements.txt (default: requirements.txt)
  -v, --venv PATH           Virtual environment directory (default: .venv)
  --torch MODE              Torch install mode: auto, default, cpu, cuda, skip
  -h, --help                Show this help message

Examples:
  ./scripts/install.sh
  ./scripts/install.sh --requirements requirements.txt --torch cpu
  ./scripts/install.sh --venv .venv --torch cuda
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--requirements)
            REQ_FILE="$2"
            shift 2
            ;;
        -v|--venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --torch)
            TORCH_MODE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -f "$REQ_FILE" ]]; then
    echo "Requirements file not found: $REQ_FILE" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but was not found on PATH." >&2
    exit 1
fi

case "$TORCH_MODE" in
    auto)
        case "$(uname -s)" in
            Darwin) TORCH_MODE="default" ;;
            Linux) TORCH_MODE="cpu" ;;
            *) TORCH_MODE="default" ;;
        esac
        ;;
    default|cpu|cuda|skip)
        ;;
    *)
        echo "Invalid torch mode: $TORCH_MODE" >&2
        echo "Choose one of: auto, default, cpu, cuda, skip" >&2
        exit 1
        ;;
esac

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[INFO] Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "[INFO] Reusing virtual environment at $VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[INFO] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

case "$TORCH_MODE" in
    default)
        echo "[INFO] Installing torch + torchvision from PyPI"
        python -m pip install torch torchvision
        ;;
    cpu)
        echo "[INFO] Installing CPU-only torch + torchvision"
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        ;;
    cuda)
        echo "[INFO] Installing CUDA torch + torchvision (cu121)"
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ;;
    skip)
        echo "[INFO] Skipping torch installation"
        ;;
esac

echo "[INFO] Installing project dependencies from $REQ_FILE"
python -m pip install "numpy<2"
python -m pip install -r "$REQ_FILE"

cat <<EOF

[INFO] Installation complete.
Next steps:
  1. Activate the environment: source "$VENV_DIR/bin/activate"
  2. Create a .env file with OPENAI_API_KEY=...
  3. Add your files to data/
  4. Run a search, for example:
     python3 app.py --query "What is attention mechanism?"
EOF
