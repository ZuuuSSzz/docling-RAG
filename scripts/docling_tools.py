import json
from typing import Iterable, Optional

from langchain_core.tools import tool
from langchain_docling.loader import ExportType

from docling_config import DEFAULT_EXPORT_TYPE, DEFAULT_TOP_K, DEFAULT_QDRANT_MODE
from docling_loader import build_docling_loader
from docling_rag_core import (
    build_vectorstore,
    build_rag_chain,
    rag_answer,
    get_rag_chain,
    get_rag_chain_for_mode,
    get_rag_retriever_for_mode,
)
from rag_contract import normalize_chain_response


def _clip_text(text: str, threshold: int = 200) -> str:
    return f"{text[:threshold]}..." if len(text) > threshold else text


def run_rag_once(
    file_paths: Iterable[str],
    question: str,
    export_type: ExportType = DEFAULT_EXPORT_TYPE,
    top_k: int = DEFAULT_TOP_K,
    qdrant_mode: str = DEFAULT_QDRANT_MODE,
    qdrant_location: str = ":memory:",
    collection_name: Optional[str] = None,
    use_vlm: bool = False,
    vlm_preset: str = "granite_docling",
    vlm_runtime: str = "api",
    vlm_url: str = "http://localhost:8000/v1/chat/completions",
    vlm_concurrency: int = 64,
    vlm_max_tokens: int = 1024,
    vlm_model: Optional[str] = None,
    contextualize_chunks: bool = True,
    debug_print_chunks: int = 0,
    max_chunk_tokens: Optional[int] = None,
    use_tokenizer: bool = True,
    merge_peers: bool = True,
    embed_model_id: Optional[str] = None,
):
    """
    Convenience helper: load docs, build Qdrant vector store, run a single RAG query.
    """
    from docling_config import DEFAULT_MAX_CHUNK_TOKENS
    from docling_rag_core import COLLECTION_NAME

    _max_chunk_tokens = max_chunk_tokens if max_chunk_tokens is not None else DEFAULT_MAX_CHUNK_TOKENS
    splits = build_docling_loader(
        file_paths=file_paths,
        export_type=export_type,
        use_vlm=use_vlm,
        vlm_preset=vlm_preset,
        vlm_runtime=vlm_runtime,
        vlm_url=vlm_url,
        vlm_concurrency=vlm_concurrency,
        vlm_max_tokens=vlm_max_tokens,
        vlm_model=vlm_model,
        contextualize_chunks=contextualize_chunks,
        debug_print_chunks=debug_print_chunks,
        max_chunk_tokens=_max_chunk_tokens,
        use_tokenizer=use_tokenizer,
        merge_peers=merge_peers,
        embed_model_id=embed_model_id,
    )
    _, retriever = build_vectorstore(
        splits=splits,
        qdrant_mode=qdrant_mode,
        qdrant_location=qdrant_location,
        collection_name=collection_name or COLLECTION_NAME,
        top_k=top_k,
    )
    rag_chain = build_rag_chain(retriever=retriever)
    resp_dict = rag_answer(rag_chain=rag_chain, question=question)

    clipped_answer = _clip_text(resp_dict.get("answer", ""), threshold=200)
    print(f"Question:\n{resp_dict.get('input')}\n")
    print(f"Answer:\n{clipped_answer}")

    context_docs = resp_dict.get("context", [])
    for i, doc in enumerate(context_docs):
        print()
        print(f"Source {i + 1}:")
        print(f"  text: {json.dumps(_clip_text(doc.page_content, threshold=350))}")
        for key, val in doc.metadata.items():
            if key == "pk":
                continue
            clipped_val = _clip_text(val) if isinstance(val, str) else val
            print(f"  {key}: {clipped_val}")


@tool("docling_rag_tool")
def docling_rag_tool(question: str) -> str:
    """
    Answer questions grounded in the ingested Docling documents using a RAG pipeline.
    Use this tool when the user asks about the content of the loaded documents.
    """
    rag_chain = get_rag_chain()
    resp = rag_answer(rag_chain=rag_chain, question=question)
    payload = normalize_chain_response(mode="mixed", chain_response=resp)
    return json.dumps(payload, ensure_ascii=False)


@tool("docling_text_search")
def docling_text_search(question: str) -> str:
    """
    Answer questions using text/table chunks only.
    Prefer this for non-visual questions or general document QA.
    """
    rag_chain = get_rag_chain_for_mode("text")
    resp = rag_answer(rag_chain=rag_chain, question=question)
    return _format_tool_response(resp, mode="text")


@tool("docling_visual_search")
def docling_visual_search(question: str) -> str:
    """
    Answer questions using image/chart chunks only.
    Prefer this for figure, chart, diagram, and plot questions.
    """
    rag_chain = get_rag_chain_for_mode("visual")
    resp = rag_answer(rag_chain=rag_chain, question=question)
    return _format_tool_response(resp, mode="visual")


@tool("docling_mixed_search")
def docling_mixed_search(question: str) -> str:
    """
    Answer questions using all chunk modalities.
    Use this when a question may need both textual and visual evidence.
    """
    rag_chain = get_rag_chain_for_mode("mixed")
    resp = rag_answer(rag_chain=rag_chain, question=question)
    return _format_tool_response(resp, mode="mixed")


@tool("docling_list_similar_pages")
def docling_list_similar_pages(question: str) -> str:
    """
    Return compact retrieval metadata for debugging/evaluation.
    Useful to inspect which pages and modalities were retrieved.
    """
    retriever = get_rag_retriever_for_mode("mixed")
    docs = retriever.get_relevant_documents(question)
    rows = []
    for doc in docs:
        meta = doc.metadata or {}
        rows.append(
            {
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
                "modality": meta.get("modality", "text"),
                "chunk_id": meta.get("chunk_id"),
            }
        )
    return json.dumps({"question": question, "results": rows}, ensure_ascii=False)


@tool("docling_search_by_modality")
def docling_search_by_modality(payload_json: str) -> str:
    """
    Retrieve context with explicit modality control.
    Input JSON format: {"question":"...", "mode":"text|visual|mixed"}.
    """
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"invalid json: {exc}"}, ensure_ascii=False)

    question = str(payload.get("question", "")).strip()
    mode = str(payload.get("mode", "mixed")).strip().lower()
    if not question:
        return json.dumps({"error": "question is required"}, ensure_ascii=False)
    if mode not in {"text", "visual", "mixed"}:
        return json.dumps(
            {"error": "mode must be one of text|visual|mixed"},
            ensure_ascii=False,
        )

    rag_chain = get_rag_chain_for_mode(mode)
    resp = rag_answer(rag_chain=rag_chain, question=question)
    return _format_tool_response(resp, mode=mode)


def _format_tool_response(resp: dict, *, mode: str) -> str:
    payload = normalize_chain_response(mode=mode, chain_response=resp)
    return json.dumps(payload, ensure_ascii=False)

