# RAG-DistressNet

A lightweight RAG playground for document search and image search. The current CLI runtime lives in `app.py` + `src/` and supports:

- Document retrieval with FAISS + `all-MiniLM-L6-v2`
- Image retrieval with OpenCLIP `ViT-L-14`
- Answer generation / image description with OpenAI `gpt-4o-mini`

## Architecture

### Document pipeline
```text
Documents in data/ -> LangChain loaders -> chunking -> all-MiniLM-L6-v2 embeddings
-> FAISS IndexFlatL2 -> top-k retrieved chunks -> gpt-4o-mini answer
```

### Image pipeline
```text
Images in data/ -> OpenCLIP image embeddings -> FAISS IndexFlatIP
-> CLIP text query embedding -> top-k image matches -> gpt-4o-mini image description
```

## Prerequisites

- Python 3.11 or newer with `python3` available on your PATH
- An OpenAI API key with credits
- Optional: `chafa` for terminal image previews

If `chafa` is not installed, image search still works and the CLI will simply skip the preview.

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
├── app.py
├── requirements.txt
├── scripts/
│   └── install.sh
├── data/
├── faiss_store/            # created after first document run
├── faiss_store_images/     # created after first image run
└── src/
    ├── data_loader.py
    ├── embedding.py
    ├── vectorstore.py
    ├── clip_store.py
    └── search.py
```

## Current Runtime Notes

- The main runtime uses `app.py` and `src/`.
- Experimental notebooks are included in the repo, but they are not used by the CLI.
- Document search currently retrieves chunk text and sends it directly to `gpt-4o-mini` in a simple prompt.
- Image search retrieves with CLIP, then asks `gpt-4o-mini` to describe the retrieved image.

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Text embeddings | `all-MiniLM-L6-v2` | Embeds document chunks |
| Image embeddings | `ViT-L-14` via OpenCLIP | Embeds images and image queries |
| LLM | `gpt-4o-mini` | Generates document answers and image descriptions |

## Troubleshooting

### `python3: command not found`

Install Python 3.11+ and make sure `python3` is on your PATH.

### `OPENAI_API_KEY` errors

Make sure `.env` exists in the project root and contains:

```env
OPENAI_API_KEY=your_key_here
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
