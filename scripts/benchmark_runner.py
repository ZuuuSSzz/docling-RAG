import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from docling_rag_agent import _expand_file_paths, _export_type_from_string
from docling_rag_core import initialize_rag, get_rag_chain_for_mode, rag_answer
from docling_loader import build_docling_loader
from docling_config import DEFAULT_QDRANT_MODE
from rag_contract import normalize_chain_response


DEFAULT_QDRANT_LOCATION = str(
    Path(__file__).resolve().parent.parent / "data" / "qdrant_data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark questions against Docling RAG pipeline."
    )
    parser.add_argument(
        "--docs",
        "-f",
        nargs="+",
        required=True,
        help="Document paths or directories.",
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="JSONL file where each line has {id, question, expected?, mode?}.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["mixed", "text", "visual"],
        default="mixed",
        help="Retriever mode for direct runs.",
    )
    parser.add_argument(
        "--export-type",
        choices=["doc_chunks", "markdown"],
        default="doc_chunks",
        help="Docling export type.",
    )
    parser.add_argument(
        "--qdrant-mode",
        choices=["embedded", "remote", "memory"],
        default=DEFAULT_QDRANT_MODE,
        help="Qdrant mode: embedded (local folder), remote (HTTP server), or memory.",
    )
    parser.add_argument(
        "--qdrant-location",
        default=DEFAULT_QDRANT_LOCATION,
        help="Qdrant location. Default is persistent local path under rag-benchmarking/data/qdrant_data.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k retrieval count.",
    )
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="Enable Docling VLM for PDF conversion.",
    )
    parser.add_argument(
        "--dump-chunks-path",
        default="",
        help="Optional JSONL path to dump chunk text+metadata for inspection.",
    )
    return parser.parse_args()


def _read_questions(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def main() -> None:
    args = parse_args()
    files = _expand_file_paths(args.docs)
    if not files:
        raise SystemExit("No supported files found for benchmark run.")
    files = [str(Path(f).resolve()) if "://" not in f else f for f in files]

    export_type = _export_type_from_string(args.export_type)
    if args.dump_chunks_path:
        chunks = build_docling_loader(
            file_paths=files,
            export_type=export_type,
            use_vlm=args.use_vlm,
        )
        dump_path = Path(args.dump_chunks_path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as fp:
            for idx, chunk in enumerate(chunks):
                fp.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "text": chunk.page_content,
                            "metadata": chunk.metadata or {},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    initialize_rag(
        file_paths=files,
        export_type=export_type,
        top_k=args.top_k,
        qdrant_mode=args.qdrant_mode,
        qdrant_location=args.qdrant_location,
        use_vlm=args.use_vlm,
    )

    questions = _read_questions(args.questions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fp:
        for row in questions:
            qid = row.get("id")
            question = row.get("question", "")
            expected = row.get("expected")
            retrieval_mode = row.get("mode", args.retrieval_mode)
            chain = get_rag_chain_for_mode(retrieval_mode)
            resp = rag_answer(chain, question)
            normalized = normalize_chain_response(
                mode=str(retrieval_mode),
                chain_response=resp,
            )
            record = {
                "id": qid,
                "question": question,
                "expected": expected,
                "answer": normalized.get("answer", ""),
                "mode": "direct",
                "retrieval_mode": retrieval_mode,
                "sources": normalized.get("sources", []),
                "pages": normalized.get("pages", []),
                "warnings": normalized.get("warnings", []),
            }

            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
