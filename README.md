# RAG-DistressNet

A lightweight RAG playground for document search and image search. The current CLI runtime lives in `app.py` + `src/` and supports:

- Document retrieval with FAISS + `all-MiniLM-L6-v2`
- Image retrieval with OpenCLIP `ViT-L-14`
- Document answer generation with either OpenAI `gpt-4o-mini` or a local `llama.cpp` server
- Image description with OpenAI `gpt-4o-mini`

## Architecture

### Document pipeline
```text
Documents in data/ -> LangChain loaders -> chunking -> all-MiniLM-L6-v2 embeddings
-> FAISS IndexFlatL2 -> top-k retrieved chunks -> selected LLM backend
   (OpenAI or local llama.cpp)
```

### Image pipeline
```text
Images in data/ -> OpenCLIP image embeddings -> FAISS IndexFlatIP
-> CLIP text query embedding -> top-k image matches -> gpt-4o-mini image description
```

## Prerequisites

- Python 3.11 or newer with `python3` available on your PATH
- An OpenAI API key with credits if you want the OpenAI-backed document or image paths
- `git` and `cmake` if you want to build the local `llama.cpp` path
- Optional: `chafa` for terminal image previews

If `chafa` is not installed, image search still works and the CLI will simply skip the preview.

## Local llama.cpp Quick Start

This branch now includes a first local-LLM milestone for document answering. The retrieval stack is unchanged, but the document generation step can point to a local `llama.cpp` server instead of OpenAI.

`Qwen/Qwen3.5-4B` is the target model for this first pass. Because it is a multimodal checkpoint, this repo currently uses it only for text-only document answers. Image description still follows the existing OpenAI path.

### 1. Install Python dependencies

```bash
chmod +x scripts/install.sh
./scripts/install.sh --torch cpu
```

### 2. Build `llama.cpp`

```bash
chmod +x scripts/setup_llama_cpp.sh
./scripts/setup_llama_cpp.sh --backend cpu
```

Use `--backend cuda` on a CUDA-capable NVIDIA machine or `--backend metal` on Apple Silicon.

### 3. Download and convert `Qwen3.5-4B`

```bash
chmod +x scripts/prepare_qwen3_5_4b.sh
./scripts/prepare_qwen3_5_4b.sh --quant Q4_K_M
```

This script:

- downloads `Qwen/Qwen3.5-4B` from Hugging Face
- converts the official checkpoint to GGUF with upstream `llama.cpp`
- quantizes it for local inference

### 4. Start the local server

```bash
./third_party/llama.cpp/build/bin/llama-server \
  --model ./models/Qwen3.5-4B-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 8192
```

### 5. Smoke test the local server

```bash
python3 scripts/smoke_test_llama_server.py \
  --model Qwen3.5-4B \
  --base-url http://127.0.0.1:8080/v1
```

### 6. Run document RAG against the local backend

```bash
python3 app.py \
  --query "Summarize the offer letter" \
  --llm-backend llama_cpp \
  --llm-model Qwen3.5-4B \
  --llm-base-url http://127.0.0.1:8080/v1
```

If you prefer environment variables over CLI flags, copy the values from `.env.example` into `.env` and set `LLM_BACKEND=llama_cpp`.

## Docker Quick Start

For moving this project to another machine with minimal setup, Docker is the fastest path. On Windows, install Docker Desktop and run these commands from PowerShell, Git Bash, or WSL in the repo root.

### 1. Create your environment file

```bash
cp .env.example .env
```

Then open `.env` and add your real OpenAI API key:

```env
OPENAI_API_KEY=sk-proj-your_actual_key_here
```

### 2. Put your files in `data/`

The Docker setup bind-mounts your local `data/` folder into the container, so whatever you place there will be available at runtime.

### 3. Build the image

```bash
docker compose build
```

### 4. Run a document query

```bash
docker compose run --rm rag --query "What is attention mechanism?"
```

### 5. Run an image query

```bash
docker compose run --rm rag --images --query "Show me the cat"
```

### 6. Rebuild the index when your data changes

```bash
docker compose run --rm rag --rebuild --query "Summarize the documents"
docker compose run --rm rag --images --rebuild --query "famous tower"
```

### Docker notes

- The container uses CPU-only PyTorch for portability.
- Document and image indexes are stored in Docker volumes, so they persist across runs.
- Hugging Face model downloads are cached in a Docker volume to speed up later runs.
- Terminal image preview is skipped in Docker because `chafa` is not installed in the image.
- To completely reset the Docker-side caches and indexes, run:

```bash
docker compose down -v
```

## Quick Start

### 1. Install dependencies

Use the install script from the project root:

```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

Useful options:

```bash
# Linux CPU-only
./scripts/install.sh --torch cpu

# Linux with NVIDIA CUDA 12.1 wheels
./scripts/install.sh --torch cuda

# Skip torch if you already installed it yourself
./scripts/install.sh --torch skip

# Use a different requirements file
./scripts/install.sh --requirements requirements.txt
```

The script:

- creates `.venv/` if needed
- upgrades `pip`, `setuptools`, and `wheel`
- installs `torch` / `torchvision`
- installs `numpy<2`
- installs everything from `requirements.txt`

### Install script guide

You can inspect the script's built-in help at any time:

```bash
./scripts/install.sh --help
```

Available options:

- `--requirements PATH`: use a different requirements file
- `--venv PATH`: create or reuse a virtual environment at a custom path
- `--torch MODE`: choose how PyTorch is installed

Supported `--torch` modes:

- `auto`: uses a sensible default for your OS
- `default`: installs standard PyPI wheels
- `cpu`: installs CPU-only wheels
- `cuda`: installs CUDA 12.1 wheels
- `skip`: skips PyTorch installation entirely

Examples:

```bash
# Use the default .venv and auto-detected torch mode
./scripts/install.sh

# Create a custom virtual environment
./scripts/install.sh --venv rag-env

# Reuse an existing environment but skip torch
./scripts/install.sh --venv .venv --torch skip

# Install from another requirements file
./scripts/install.sh --requirements requirements-dev.txt
```

It is safe to rerun the script. If the virtual environment already exists, the script reuses it and reinstalls/upgrades packages as needed.

### 2. Activate the environment

```bash
source .venv/bin/activate
```

### 3. Add your API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-proj-your_actual_key_here
```

If you are only testing the local document backend, you can skip `OPENAI_API_KEY` and instead set:

```env
LLM_BACKEND=llama_cpp
LLM_MODEL=Qwen3.5-4B
LLM_BASE_URL=http://127.0.0.1:8080/v1
LLM_API_KEY=EMPTY
```

### 4. Add files to `data/`

Supported document formats:

- `.pdf`
- `.txt`
- `.csv`
- `.xlsx`
- `.docx`
- `.json`

Supported image formats:

- `.png`
- `.jpg`
- `.jpeg`

Example:

```text
data/
├── paper1.pdf
├── notes.txt
├── results.csv
├── cat.png
└── subfolder/
    └── offerletter.pdf
```

## How To Run

### Document search

Document search is the default mode:

```bash
python3 app.py --query "What is attention mechanism?"
```

You can also pass `--pdfs` explicitly:

```bash
python3 app.py --pdfs --query "Summarize the offer letter"
```

To use the local `llama.cpp` backend for document answers:

```bash
python3 app.py \
  --query "Summarize the offer letter" \
  --llm-backend llama_cpp \
  --llm-model Qwen3.5-4B \
  --llm-base-url http://127.0.0.1:8080/v1
```

On the first run, the app builds `faiss_store/`. Later runs reuse the saved index.

### Image search

```bash
python3 app.py --images --query "Show me the cat"
python3 app.py --images --query "a person in a suit"
python3 app.py --images --query "famous tower"
```

On the first run, the app downloads the CLIP model and builds `faiss_store_images/`.

### Rebuild the index

If you add or remove files and want a clean rebuild:

```bash
# Rebuild document index
python3 app.py --rebuild --query "What changed in the documents?"

# Rebuild image index
python3 app.py --images --rebuild --query "show me the cat"
```

## Project Structure

```text
RAG-DistressNet/
├── Dockerfile
├── docs/
│   └── local_rag_plan.md
├── docker-compose.yml
├── app.py
├── requirements.txt
├── scripts/
│   ├── install.sh
│   ├── prepare_qwen3_5_4b.sh
│   ├── setup_llama_cpp.sh
│   └── smoke_test_llama_server.py
├── data/
├── faiss_store/            # created after first document run
├── faiss_store_images/     # created after first image run
├── models/                 # created by local model setup
├── third_party/            # created by local llama.cpp setup
└── src/
    ├── data_loader.py
    ├── embedding.py
    ├── llm.py
    ├── vectorstore.py
    ├── clip_store.py
    └── search.py
```

## Current Runtime Notes

- The main runtime uses `app.py` and `src/`.
- Experimental notebooks are included in the repo, but they are not used by the CLI.
- Document search now supports either the OpenAI backend or a local `llama.cpp` server.
- Image search retrieves with CLIP, then asks `gpt-4o-mini` to describe the retrieved image.
- The new local milestone is documented in `docs/local_rag_plan.md`.
- The Docker image is intended for quick CPU-based testing on other machines.

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Text embeddings | `all-MiniLM-L6-v2` | Embeds document chunks |
| Image embeddings | `ViT-L-14` via OpenCLIP | Embeds images and image queries |
| Document LLM | `gpt-4o-mini` or local `Qwen3.5-4B` via `llama.cpp` | Generates document answers |
| Image LLM | `gpt-4o-mini` | Describes retrieved images |

## Troubleshooting

### `python3: command not found`

Install Python 3.11+ and make sure `python3` is on your PATH.

### `OPENAI_API_KEY` errors

Make sure `.env` exists in the project root and contains:

```env
OPENAI_API_KEY=your_key_here
```

`OPENAI_API_KEY` is only required for the OpenAI-backed document path and for image mode.

### Local `llama.cpp` server is unreachable

Make sure `llama-server` is running and that the app is pointing at the right URL:

```bash
python3 app.py \
  --query "test" \
  --llm-backend llama_cpp \
  --llm-model Qwen3.5-4B \
  --llm-base-url http://127.0.0.1:8080/v1
```

You can validate the server independently with:

```bash
python3 scripts/smoke_test_llama_server.py \
  --model Qwen3.5-4B \
  --base-url http://127.0.0.1:8080/v1
```

### Image search shows no terminal preview

Install `chafa` if you want previews:

```bash
# Ubuntu / Debian
sudo apt install chafa

# Fedora / RHEL
sudo dnf install chafa

# macOS
brew install chafa
```

### You added new files but results did not change

Rebuild the relevant index:

```bash
python3 app.py --rebuild --query "your query"
python3 app.py --images --rebuild --query "your query"
```

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.
