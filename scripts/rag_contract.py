from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_CONTEXT_PROMPT_TEMPLATE = (
    "You are a grounded Docling RAG assistant.\n"
    "Use only the provided context when answering.\n"
    "If context is insufficient, say so briefly.\n"
    "Cite sources using available metadata (source, page, modality) when possible.\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Query: {input}\n"
    "Answer:\n"
)


DEFAULT_AGENT_RULES = (
    "You are a Docling RAG assistant.\n"
    "Tool selection policy:\n"
    "1) Use docling_visual_search for chart/image/figure/diagram questions.\n"
    "2) Use docling_text_search for purely textual questions.\n"
    "3) Use docling_mixed_search when evidence may be multimodal or unclear.\n"
    "4) Use docling_list_similar_pages when user asks for retrieval evidence/pages.\n"
    "Return concise grounded answers."
)


def normalize_response_contract(
    *,
    mode: str,
    answer: str,
    context_docs: List[Any] | None = None,
    warnings: List[str] | None = None,
) -> Dict[str, Any]:
    context = context_docs or []
    sources: List[Dict[str, Any]] = []
    pages: List[int] = []
    for doc in context:
        meta = getattr(doc, "metadata", None) or {}
        page = _safe_int(meta.get("page"))
        if page is not None:
            pages.append(page)
        sources.append(
            {
                "source": meta.get("source", "unknown"),
                "page": page,
                "modality": meta.get("modality", "text"),
                "chunk_id": meta.get("chunk_id"),
            }
        )
    pages = sorted(set(pages))
    return {
        "mode": mode,
        "answer": str(answer or "").strip(),
        "sources": sources,
        "pages": pages,
        "warnings": warnings or [],
    }


def normalize_chain_response(
    *,
    mode: str,
    chain_response: Dict[str, Any],
    warnings: List[str] | None = None,
) -> Dict[str, Any]:
    return normalize_response_contract(
        mode=mode,
        answer=str(chain_response.get("answer", "")),
        context_docs=chain_response.get("context", []) or [],
        warnings=warnings,
    )


def _safe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
