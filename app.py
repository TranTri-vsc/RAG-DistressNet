import argparse
import os
import shutil
import subprocess
from src.search import RAGSearch


## surya - CLI flag support
def reset_index(index_dir):
    if os.path.isdir(index_dir):
        shutil.rmtree(index_dir)
        print(f"[INFO] Removed existing index at {index_dir}")
    else:
        print(f"[INFO] No existing index found at {index_dir}; building a fresh one.")


def render_image_preview(image_path):
    if shutil.which("chafa") is None:
        print("  Preview: skipped (install `chafa` for terminal image previews)")
        return
    subprocess.run(["chafa", "--size=40x20", image_path], check=False)


def run_pdfs(query, rebuild=False):
    if rebuild:
        reset_index("faiss_store")
    rag_search = RAGSearch()
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
    args = parser.parse_args()

    if args.images:
        run_images(args.query, rebuild=args.rebuild)
    else:
        run_pdfs(args.query, rebuild=args.rebuild)
## surya end
