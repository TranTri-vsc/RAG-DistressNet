# Local RAG Plan

## Goal

Move this project toward an on-device RAG stack that can run without cloud generation.

## Current Milestone

This branch implements the first milestone:

1. Keep the current FAISS retrieval pipeline intact so we do not change too many moving parts at once.
2. Add a pluggable document-generation backend so `app.py` can talk either to OpenAI or to a local `llama.cpp` server.
3. Standardize the local workflow around `Qwen/Qwen3.5-4B`:
   - download official Hugging Face weights
   - convert to GGUF with upstream `llama.cpp`
   - quantize for on-device inference
   - serve through `llama-server`
4. Add a standalone smoke test so we can validate local inference before wiring it into later RAG experiments.

## Scope Notes

- The document-answering path is now local-backend ready.
- The image search path still uses the existing OpenAI image-description step.
- This is intentional: `Qwen3.5-4B` is a multimodal checkpoint, and image support in `llama.cpp` adds projector and serving details that are better handled in a later milestone.

## Next Milestones

### Milestone 2: Fully Local Text RAG

- Replace OpenAI document generation in the default workflow with the local backend by default.
- Add simple benchmarking for latency, memory, and answer quality across quantization levels.
- Add a small evaluation set for regression testing.

### Milestone 3: Fully Local Retrieval Stack

- Replace or make optional the current embedding stack with a local embedding model chosen for on-device use.
- Benchmark retrieval quality and end-to-end answer quality under device constraints.
- Add configuration profiles for CPU-only vs GPU-assisted local runs.

### Milestone 4: Local Multimodal RAG

- Revisit image answering with a local multimodal model path.
- Decide whether `Qwen3.5-4B` remains the right checkpoint for local image reasoning or whether a smaller vision-specialized model is a better fit.
