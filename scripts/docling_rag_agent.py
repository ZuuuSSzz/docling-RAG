import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_docling.loader import ExportType

from docling_config import (
    DEFAULT_TOP_K,
    DEFAULT_MAX_CHUNK_TOKENS,
    DEFAULT_VLM_CONCURRENCY,
    DEFAULT_VLM_MAX_TOKENS,
    DEFAULT_VLM_PRESET,
    DEFAULT_VLM_RUNTIME,
    DEFAULT_VLM_URL,
    DEFAULT_QDRANT_MODE,
    SUPPORTED_FILE_EXTENSIONS,
)
from docling_rag_core import COLLECTION_NAME, initialize_rag, get_rag_chain, rag_answer
from rag_contract import normalize_chain_response


# Default document source: local arxiv PDFs
DEFAULT_DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "HF-eval" / "data" / "arxiv"
# Default Qdrant storage: local dir so data persists; run again with --reuse-collection to skip re-processing
DEFAULT_QDRANT_LOCATION = str(Path(__file__).resolve().parent.parent / "data" / "qdrant_data")


def _expand_file_paths(paths: List[str]) -> List[str]:
    """Expand directory paths to supported files; leave files/URLs as-is."""
    expanded: List[str] = []
    repo_root = Path(__file__).resolve().parents[3]
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            cwd_candidate = Path.cwd() / path
            repo_candidate = repo_root / path
            if cwd_candidate.exists():
                path = cwd_candidate
            elif repo_candidate.exists():
                path = repo_candidate
        if path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_FILE_EXTENSIONS:
                    expanded.append(str(f))
        else:
            expanded.append(str(path))
    return expanded if expanded else paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Docling + LangChain RAG example using Qdrant and HuggingFaceEndpoint.\n"
            "Follows Docling Retrieval with Qdrant and RAG with LangChain documentation."
        )
    )
    parser.add_argument(
        "--file-path",
        "-f",
        nargs="+",
        required=False,
        default=[str(DEFAULT_DOCS_DIR)],
        help="Directory of PDFs, or one or more PDF paths/URLs. Default: HF-eval/data/arxiv",
    )
    parser.add_argument(
        "--question",
        "-q",
        required=False,
        default="Which are the main AI models in Docling?",
        help="Question to ask the RAG pipeline.",
    )
    parser.add_argument(
        "--export-type",
        choices=["doc_chunks", "markdown"],
        default="doc_chunks",
        help="Docling export type: doc_chunks (default) or markdown.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved documents for RAG.",
    )
    parser.add_argument(
        "--qdrant-mode",
        choices=["embedded", "remote", "memory"],
        default=DEFAULT_QDRANT_MODE,
        help="Qdrant mode: embedded (local folder), remote (HTTP server), or memory.",
    )
    parser.add_argument(
        "--qdrant-location",
        type=str,
        default=DEFAULT_QDRANT_LOCATION,
        help="Qdrant target path or URL. Interpreted according to --qdrant-mode.",
    )
    parser.add_argument(
        "--reuse-collection",
        action="store_true",
        help="Use existing Qdrant collection if present; skip loading/chunking/embedding (faster re-runs).",
    )
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable hybrid search (do not set Qdrant sparse model).",
    )
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="Use Docling VLM pipeline for PDF conversion before chunking.",
    )
    parser.add_argument(
        "--vlm-preset",
        type=str,
        default=DEFAULT_VLM_PRESET,
        help="Docling VLM preset name (default: granite_docling).",
    )
    parser.add_argument(
        "--vlm-runtime",
        choices=["api", "ollama", "lmstudio"],
        default=DEFAULT_VLM_RUNTIME,
        help="Docling VLM runtime type. Use 'ollama' for local Ollama, 'api' for vLLM/OpenAI-compatible URL.",
    )
    parser.add_argument(
        "--vlm-url",
        type=str,
        default=DEFAULT_VLM_URL,
        help="VLM API URL (OpenAI-compatible /v1/chat/completions).",
    )
    parser.add_argument(
        "--vlm-concurrency",
        type=int,
        default=DEFAULT_VLM_CONCURRENCY,
        help="Concurrency for Docling VLM API engine options.",
    )
    parser.add_argument(
        "--vlm-max-tokens",
        type=int,
        default=DEFAULT_VLM_MAX_TOKENS,
        help="Max tokens for VLM API generation (avoid exceeding model context).",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default=None,
        help="Optional model name to send to the VLM API (if required by the server).",
    )
    parser.add_argument(
        "--no-contextualize",
        action="store_true",
        help="Do not use HybridChunker.contextualize(); embed raw chunk.text instead.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=DEFAULT_MAX_CHUNK_TOKENS,
        help="Max tokens per chunk (parse_doc-style; default %(default)s). Aligns with embedding model for RAG benchmarking.",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Use default HybridChunker without custom tokenizer (faster, less control over chunk size).",
    )
    parser.add_argument(
        "--no-merge-peers",
        action="store_true",
        help="Disable merging of undersized peer chunks.",
    )
    parser.add_argument(
        "--debug-print-chunks",
        type=int,
        default=0,
        help="If >0, print the first N chunks in Docling example format (chunk.text and contextualize()).",
    )
    return parser.parse_args()


def _export_type_from_string(name: str) -> ExportType:
    if name == "doc_chunks":
        return ExportType.DOC_CHUNKS
    if name == "markdown":
        return ExportType.MARKDOWN
    raise ValueError(f"Unsupported export type: {name}")


def _print_contract(title: str, payload: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(payload.get("answer", ""))
    sources = payload.get("sources", []) or []
    pages = payload.get("pages", []) or []
    warnings = payload.get("warnings", []) or []

    if sources:
        print("\nSources:")
        for src in sources[:10]:
            print(
                f"- source={src.get('source')} page={src.get('page')} "
                f"modality={src.get('modality')} chunk_id={src.get('chunk_id')}"
            )
    if pages:
        print("\nPages:")
        print(", ".join(str(p) for p in pages))
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")

    print("\nStructured JSON:")
    print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    file_paths = _expand_file_paths(args.file_path)
    if not file_paths:
        raise SystemExit("No PDF files found. Pass a directory with .pdf files or explicit file paths.")
    export_type = _export_type_from_string(args.export_type)
    # Initialize RAG components for direct RAG.
    initialize_rag(
        file_paths=file_paths,
        export_type=export_type,
        top_k=args.top_k,
        qdrant_mode=args.qdrant_mode,
        qdrant_location=args.qdrant_location,
        collection_name=COLLECTION_NAME,
        use_sparse_model=not args.no_sparse,
        use_vlm=args.use_vlm,
        vlm_preset=args.vlm_preset,
        vlm_runtime=args.vlm_runtime,
        vlm_url=args.vlm_url,
        vlm_concurrency=args.vlm_concurrency,
        vlm_max_tokens=args.vlm_max_tokens,
        vlm_model=args.vlm_model,
        contextualize_chunks=not args.no_contextualize,
        debug_print_chunks=args.debug_print_chunks,
        max_chunk_tokens=args.max_chunk_tokens,
        use_tokenizer=not args.no_tokenizer,
        merge_peers=not args.no_merge_peers,
        reuse_collection=args.reuse_collection,
    )

    print("=== Direct RAG run ===")
    rag_chain = get_rag_chain()
    direct_resp = rag_answer(rag_chain, args.question)
    direct_payload = normalize_chain_response(mode="direct", chain_response=direct_resp)
    _print_contract("Direct Answer", direct_payload)


if __name__ == "__main__":
    main()

