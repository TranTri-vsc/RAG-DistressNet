#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="$REPO_ROOT/third_party/llama.cpp"
BACKEND="cpu"

usage() {
    cat <<'EOF'
Usage: ./scripts/setup_llama_cpp.sh [options]

Options:
  --dir PATH        Install or update llama.cpp in PATH
  --backend TYPE    Build backend: cpu, cuda, or metal (default: cpu)
  -h, --help        Show this help message

Examples:
  ./scripts/setup_llama_cpp.sh
  ./scripts/setup_llama_cpp.sh --backend cuda
  ./scripts/setup_llama_cpp.sh --dir /opt/llama.cpp
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            LLAMA_DIR="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
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

for required_cmd in git cmake; do
    if ! command -v "$required_cmd" >/dev/null 2>&1; then
        echo "$required_cmd is required but was not found on PATH." >&2
        exit 1
    fi
done

case "$BACKEND" in
    cpu)
        CMAKE_ARGS=(-DLLAMA_CURL=ON)
        ;;
    cuda)
        CMAKE_ARGS=(-DGGML_CUDA=ON -DLLAMA_CURL=ON)
        ;;
    metal)
        CMAKE_ARGS=(-DGGML_METAL=ON -DLLAMA_CURL=ON)
        ;;
    *)
        echo "Unsupported backend: $BACKEND" >&2
        echo "Choose one of: cpu, cuda, metal" >&2
        exit 1
        ;;
esac

if [[ -d "$LLAMA_DIR/.git" ]]; then
    echo "[INFO] Updating existing llama.cpp checkout in $LLAMA_DIR"
    git -C "$LLAMA_DIR" pull --ff-only
else
    echo "[INFO] Cloning llama.cpp into $LLAMA_DIR"
    rm -rf "$LLAMA_DIR"
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

echo "[INFO] Configuring llama.cpp with backend=$BACKEND"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release "${CMAKE_ARGS[@]}"

echo "[INFO] Building llama.cpp binaries"
cmake --build "$LLAMA_DIR/build" --config Release -j

echo
echo "[INFO] llama.cpp is ready."
echo "Key binaries:"
echo "  $LLAMA_DIR/build/bin/llama-server"
echo "  $LLAMA_DIR/build/bin/llama-quantize"
echo "  $LLAMA_DIR/build/bin/llama-cli"
