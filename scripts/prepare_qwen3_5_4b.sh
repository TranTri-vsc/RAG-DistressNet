#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="$REPO_ROOT/third_party/llama.cpp"
MODELS_DIR="$REPO_ROOT/models"
HF_REPO="Qwen/Qwen3.5-4B"
QUANT="Q4_K_M"
KEEP_F16=1

usage() {
    cat <<'EOF'
Usage: ./scripts/prepare_qwen3_5_4b.sh [options]

Options:
  --llama-dir PATH   Path to the llama.cpp checkout (default: third_party/llama.cpp)
  --models-dir PATH  Directory to store downloaded and converted model files
  --hf-repo REPO     Hugging Face repo to download (default: Qwen/Qwen3.5-4B)
  --quant TYPE       Quantization target, e.g. Q4_K_M or Q8_0 (default: Q4_K_M)
  --discard-f16      Delete the intermediate f16 GGUF after quantization
  -h, --help         Show this help message

Examples:
  ./scripts/prepare_qwen3_5_4b.sh
  ./scripts/prepare_qwen3_5_4b.sh --quant Q8_0
  ./scripts/prepare_qwen3_5_4b.sh --models-dir /data/local-models
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llama-dir)
            LLAMA_DIR="$2"
            shift 2
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --hf-repo)
            HF_REPO="$2"
            shift 2
            ;;
        --quant)
            QUANT="$2"
            shift 2
            ;;
        --discard-f16)
            KEEP_F16=0
            shift
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

for required_cmd in python3; do
    if ! command -v "$required_cmd" >/dev/null 2>&1; then
        echo "$required_cmd is required but was not found on PATH." >&2
        exit 1
    fi
done

if [[ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]]; then
    echo "Expected llama-quantize at $LLAMA_DIR/build/bin/llama-quantize" >&2
    echo "Run ./scripts/setup_llama_cpp.sh first." >&2
    exit 1
fi

if [[ ! -f "$LLAMA_DIR/convert_hf_to_gguf.py" ]]; then
    echo "Expected convert_hf_to_gguf.py at $LLAMA_DIR/convert_hf_to_gguf.py" >&2
    echo "Run ./scripts/setup_llama_cpp.sh first." >&2
    exit 1
fi

if ! python3 - <<'PY' >/dev/null 2>&1
import huggingface_hub  # noqa: F401
PY
then
    echo "huggingface_hub is required. Install the project requirements first." >&2
    exit 1
fi

mkdir -p "$MODELS_DIR"

HF_DIR="$MODELS_DIR/Qwen3.5-4B-hf"
F16_GGUF="$MODELS_DIR/Qwen3.5-4B-f16.gguf"
QUANTIZED_GGUF="$MODELS_DIR/Qwen3.5-4B-${QUANT}.gguf"

echo "[INFO] Downloading $HF_REPO to $HF_DIR"
python3 - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${HF_REPO}",
    local_dir="${HF_DIR}",
)
PY

echo "[INFO] Converting Hugging Face weights to GGUF"
python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$HF_DIR" --outfile "$F16_GGUF" --outtype f16

echo "[INFO] Quantizing GGUF to $QUANT"
"$LLAMA_DIR/build/bin/llama-quantize" "$F16_GGUF" "$QUANTIZED_GGUF" "$QUANT"

if [[ "$KEEP_F16" -eq 0 ]]; then
    echo "[INFO] Removing intermediate $F16_GGUF"
    rm -f "$F16_GGUF"
fi

echo
echo "[INFO] Qwen3.5-4B is ready for llama.cpp."
echo "Quantized model:"
echo "  $QUANTIZED_GGUF"
echo
echo "Suggested server command:"
echo "  $LLAMA_DIR/build/bin/llama-server --model $QUANTIZED_GGUF --host 127.0.0.1 --port 8080 --ctx-size 8192"
