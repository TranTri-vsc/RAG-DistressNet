import argparse
import os
import shutil
import subprocess
from src.llm import LLMConfig, SUPPORTED_LLM_BACKENDS
from src.search import RAGSearch


## surya - CLI flag support
def reset_index(index_dir):
    os.makedirs(index_dir, exist_ok=True)
    removed_any = False
    for entry in os.listdir(index_dir):
        path = os.path.join(index_dir, entry)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)
        removed_any = True
    if removed_any:
        print(f"[INFO] Cleared existing index contents in {index_dir}")
    else:
        print(f"[INFO] {index_dir} was already empty; building a fresh index.")


def render_image_preview(image_path):
    if shutil.which("chafa") is None:
        print("  Preview: skipped (install `chafa` for terminal image previews)")
        return
    subprocess.run(["chafa", "--size=40x20", image_path], check=False)


def build_llm_config(args):
    return LLMConfig.from_inputs(
        backend=args.llm_backend,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        temperature=args.llm_temperature,
        disable_thinking=not args.enable_thinking,
    )


def run_pdfs(query, rebuild=False, llm_config=None):
    if rebuild:
        reset_index("faiss_store")
    rag_search = RAGSearch(llm_config=llm_config)
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)

def run_images(query, rebuild=False):
    if rebuild:
        reset_index("faiss_store_images")
    from src.search import ImageRAGSearch
    rag_search = ImageRAGSearch()
    result = rag_search.search_and_summarize(query, top_k=3)
    if result["images"]:
        print(f"\nResults for: '{query}'")
        for img in result["images"]:
            print(f"\n  Image: {img['path']}")
            print(f"  Score: {img['score']}")
            print(f"  Description: {img['description']}")
            render_image_preview(img["path"])
    else:
        print("No relevant images found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Search")
    parser.add_argument("--pdfs", action="store_true", help="Search through PDFs")
    parser.add_argument("--images", action="store_true", help="Search through images")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the selected index before searching")
    parser.add_argument("--query", type=str, default="What is attention mechanism?", help="Search query")
    parser.add_argument(
        "--llm-backend",
        choices=SUPPORTED_LLM_BACKENDS,
        default=None,
        help="LLM backend for document answers: openai or llama_cpp",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override the LLM model name for the selected backend",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="Base URL for an OpenAI-compatible local server, e.g. http://127.0.0.1:8080/v1",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="Optional API key override. llama.cpp can usually use the default EMPTY value.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="Override the LLM temperature",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow reasoning / <think> output on models that support it",
    )
    args = parser.parse_args()

    if args.images:
        if any(
            value is not None
            for value in (
                args.llm_backend,
                args.llm_model,
                args.llm_base_url,
                args.llm_api_key,
                args.llm_temperature,
            )
        ) or args.enable_thinking:
            print(
                "[WARN] Image mode still uses the existing OpenAI image description path. "
                "The local llama.cpp backend currently applies to document mode only."
            )
        run_images(args.query, rebuild=args.rebuild)
    else:
        llm_config = build_llm_config(args)
        run_pdfs(args.query, rebuild=args.rebuild, llm_config=llm_config)
