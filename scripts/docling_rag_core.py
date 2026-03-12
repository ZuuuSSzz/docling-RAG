from pathlib import Path
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaLLM
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_docling.loader import ExportType
from qdrant_client import QdrantClient
from qdrant_client.models import Document as QdrantDocument
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, PointStruct

from docling_config import (
    EMBED_MODEL_ID,
    GEN_MODEL_ID,
    DEFAULT_EXPORT_TYPE,
    DEFAULT_TOP_K,
    DEFAULT_MAX_CHUNK_TOKENS,
    DEFAULT_QDRANT_MODE,
    get_env,
)
from docling_loader import build_docling_loader
from rag_contract import DEFAULT_CONTEXT_PROMPT_TEMPLATE


# Collection names for Qdrant (single and split modality indexing)
COLLECTION_NAME = "docling"
TEXT_COLLECTION_NAME = "docling_text"
VISUAL_COLLECTION_NAME = "docling_visual"

_VECTORSTORE: Optional[QdrantClient] = None
_RETRIEVER: Any = None
_RAG_CHAIN: Any = None
_RETRIEVERS: Dict[str, Any] = {}
_RAG_CHAINS: Dict[str, Any] = {}


QDRANT_MODES = {"embedded", "remote", "memory"}


class QdrantRetriever(BaseRetriever):
    """
    Retriever that uses Qdrant client.query_points() and returns LangChain Documents.
    """

    client: QdrantClient
    collection_name: str
    top_k: int = DEFAULT_TOP_K
    modality_filter: Optional[Sequence[str]] = None
    vector_name: Optional[str] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        vector_name = self.vector_name or _get_dense_vector_name(
            client=self.client,
            collection_name=self.collection_name,
        )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=QdrantDocument(text=query, model=EMBED_MODEL_ID),
            using=vector_name,
            limit=self.top_k,
            query_filter=_build_modality_filter(self.modality_filter),
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", []) or []
        docs = []
        for p in points:
            content = getattr(p, "document", None) or getattr(p, "text", None)
            if not content and isinstance(getattr(p, "payload", None), dict):
                payload = p.payload
                content = payload.get("document") or payload.get("text") or ""
            meta = getattr(p, "metadata", None) or getattr(p, "payload", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            docs.append(Document(page_content=content or "", metadata=meta))
        return docs


def build_vectorstore(
    splits: List[Document],
    qdrant_location: str = ":memory:",
    qdrant_mode: str = DEFAULT_QDRANT_MODE,
    collection_name: str = COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
    use_sparse_model: bool = True,
    text_collection_name: str = TEXT_COLLECTION_NAME,
    visual_collection_name: str = VISUAL_COLLECTION_NAME,
    split_collections_by_modality: bool = True,
) -> Tuple[QdrantClient, QdrantRetriever]:
    """
    Build Qdrant vector store and retriever from Docling splits.

    Follows Docling "Retrieval with Qdrant" example:
    - QdrantClient(path=...) for embedded mode, or URL for remote
    - client.set_model("sentence-transformers/all-MiniLM-L6-v2")
    - client.set_sparse_model("Qdrant/bm25") for hybrid search (RRF)
    - client.upsert(...) with inferred vectors
    - Retrieval via client.query_points(...)
    """
    if not splits:
        raise ValueError("splits must not be empty.")

    resolved_mode = _validate_qdrant_mode(qdrant_mode)
    client = _build_qdrant_client(
        qdrant_mode=resolved_mode,
        qdrant_location=qdrant_location,
    )
    _health_check_qdrant(
        client=client,
        qdrant_mode=resolved_mode,
        qdrant_location=qdrant_location,
    )
    client.set_model(EMBED_MODEL_ID)
    if use_sparse_model:
        client.set_sparse_model("Qdrant/bm25")

    text_docs: List[Document] = []
    visual_docs: List[Document] = []
    for doc in splits:
        modality = str((doc.metadata or {}).get("modality", "text")).lower()
        if modality in {"image", "chart"}:
            visual_docs.append(doc)
        else:
            text_docs.append(doc)

    if split_collections_by_modality:
        _recreate_collection_if_needed(
            client, resolved_mode, text_collection_name
        )
        _recreate_collection_if_needed(
            client, resolved_mode, visual_collection_name
        )
        _add_docs_to_collection(
            client,
            text_collection_name,
            text_docs,
            use_sparse_model=use_sparse_model,
        )
        _add_docs_to_collection(
            client,
            visual_collection_name,
            visual_docs,
            use_sparse_model=use_sparse_model,
        )

        # Compatibility collection for mixed retrieval / existing tooling.
        _recreate_collection_if_needed(client, resolved_mode, collection_name)
        _add_docs_to_collection(
            client,
            collection_name,
            splits,
            use_sparse_model=use_sparse_model,
        )
    else:
        _recreate_collection_if_needed(client, resolved_mode, collection_name)
        _add_docs_to_collection(
            client,
            collection_name,
            splits,
            use_sparse_model=use_sparse_model,
        )

    retriever = QdrantRetriever(
        client=client,
        collection_name=collection_name,
        top_k=top_k,
        vector_name=_get_dense_vector_name(
            client=client,
            collection_name=collection_name,
        ),
    )
    return client, retriever


def build_rag_chain(
    retriever: Any,
    gen_model_id: str = GEN_MODEL_ID,
    hf_token: Optional[str] = None,
    prompt: Optional[PromptTemplate] = None,
):
    """
    Build a retrieval-augmented generation chain using:
    - Qdrant retriever
    - Local Ollama LLM (llama3.2:3b)
    - Stuff documents chain with a simple prompt
    """
    # Use local Ollama instead of HuggingFaceEndpoint.
    # Model name should match `ollama list`.
    llm = OllamaLLM(model=gen_model_id or "llama3.2:3b")

    if prompt is None:
        prompt = PromptTemplate.from_template(DEFAULT_CONTEXT_PROMPT_TEMPLATE)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


def build_text_retriever(
    client: QdrantClient,
    *,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = TEXT_COLLECTION_NAME,
) -> QdrantRetriever:
    collection = _collection_or_fallback(
        client=client,
        preferred=collection_name,
        fallback=COLLECTION_NAME,
    )
    return QdrantRetriever(
        client=client,
        collection_name=collection,
        top_k=top_k,
        modality_filter=("text", "table"),
        vector_name=_get_dense_vector_name(
            client=client,
            collection_name=collection,
        ),
    )


def build_visual_retriever(
    client: QdrantClient,
    *,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = VISUAL_COLLECTION_NAME,
) -> QdrantRetriever:
    collection = _collection_or_fallback(
        client=client,
        preferred=collection_name,
        fallback=COLLECTION_NAME,
    )
    return QdrantRetriever(
        client=client,
        collection_name=collection,
        top_k=top_k,
        modality_filter=("image", "chart"),
        vector_name=_get_dense_vector_name(
            client=client,
            collection_name=collection,
        ),
    )


def build_mixed_retriever(
    client: QdrantClient,
    *,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = COLLECTION_NAME,
) -> QdrantRetriever:
    collection = _collection_or_fallback(
        client=client,
        preferred=collection_name,
        fallback=COLLECTION_NAME,
    )
    return QdrantRetriever(
        client=client,
        collection_name=collection,
        top_k=top_k,
        vector_name=_get_dense_vector_name(
            client=client,
            collection_name=collection,
        ),
    )


def initialize_rag(
    file_paths: Iterable[str],
    export_type: ExportType = DEFAULT_EXPORT_TYPE,
    top_k: int = DEFAULT_TOP_K,
    qdrant_location: str = ":memory:",
    qdrant_mode: str = DEFAULT_QDRANT_MODE,
    collection_name: str = COLLECTION_NAME,
    use_sparse_model: bool = True,
    use_vlm: bool = False,
    vlm_preset: str = "granite_docling",
    vlm_runtime: str = "api",
    vlm_url: str = "http://localhost:8000/v1/chat/completions",
    vlm_concurrency: int = 64,
    vlm_max_tokens: int = 1024,
    vlm_model: Optional[str] = None,
    contextualize_chunks: bool = True,
    debug_print_chunks: int = 0,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    use_tokenizer: bool = True,
    merge_peers: bool = True,
    embed_model_id: Optional[str] = None,
    reuse_collection: bool = False,
    split_collections_by_modality: bool = True,
):
    """
    Initialize global RAG components (Qdrant client, retriever, chain).

    Chunking uses parse_doc-style alignment (tokenizer + max_tokens) for reproducible RAG benchmarking.
    When reuse_collection is True and qdrant_location is a local path with an existing collection,
    skips loading/chunking/embedding and reuses it (faster re-runs).
    """
    global _VECTORSTORE, _RETRIEVER, _RAG_CHAIN, _RETRIEVERS, _RAG_CHAINS

    resolved_mode = _validate_qdrant_mode(qdrant_mode)
    if reuse_collection and resolved_mode != "memory":
        client = _build_qdrant_client(
            qdrant_mode=resolved_mode,
            qdrant_location=qdrant_location,
        )
        _health_check_qdrant(
            client=client,
            qdrant_mode=resolved_mode,
            qdrant_location=qdrant_location,
        )
        if client.collection_exists(collection_name):
            try:
                result = client.count(collection_name)
                n = getattr(result, "count", 0) or 0
            except Exception:
                n = 0
            if n > 0:
                client.set_model(EMBED_MODEL_ID)
                if use_sparse_model:
                    client.set_sparse_model("Qdrant/bm25")
                retriever = QdrantRetriever(
                    client=client,
                    collection_name=collection_name,
                    top_k=top_k,
                    vector_name=_get_dense_vector_name(
                        client=client,
                        collection_name=collection_name,
                    ),
                )
                rag_chain = build_rag_chain(retriever=retriever)
                _VECTORSTORE = client
                _RETRIEVER = retriever
                _RAG_CHAIN = rag_chain
                _RETRIEVERS = {
                    "mixed": build_mixed_retriever(client, top_k=top_k),
                    "text": build_text_retriever(client, top_k=top_k),
                    "visual": build_visual_retriever(client, top_k=top_k),
                }
                _RAG_CHAINS = {
                    name: build_rag_chain(retriever=r)
                    for name, r in _RETRIEVERS.items()
                }
                return rag_chain

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
        max_chunk_tokens=max_chunk_tokens,
        use_tokenizer=use_tokenizer,
        merge_peers=merge_peers,
        embed_model_id=embed_model_id,
    )
    client, retriever = build_vectorstore(
        splits=splits,
        qdrant_location=qdrant_location,
        qdrant_mode=resolved_mode,
        collection_name=collection_name,
        top_k=top_k,
        use_sparse_model=use_sparse_model,
        split_collections_by_modality=split_collections_by_modality,
    )
    rag_chain = build_rag_chain(retriever=retriever)
    _VECTORSTORE = client
    _RETRIEVER = retriever
    _RAG_CHAIN = rag_chain
    _RETRIEVERS = {
        "mixed": build_mixed_retriever(client, top_k=top_k),
        "text": build_text_retriever(client, top_k=top_k),
        "visual": build_visual_retriever(client, top_k=top_k),
    }
    _RAG_CHAINS = {
        name: build_rag_chain(retriever=r)
        for name, r in _RETRIEVERS.items()
    }
    return rag_chain


def rag_answer(rag_chain, question: str) -> dict:
    """
    Invoke the RAG chain with a question.

    Returns a dict with keys:
    - input
    - answer
    - context (list of Documents)
    """
    resp = rag_chain.invoke({"input": question})
    return resp


def get_rag_chain():
    """
    Return the globally initialized RAG chain, or raise if not initialized.
    """
    if _RAG_CHAIN is None:
        raise RuntimeError(
            "RAG chain is not initialized. Call initialize_rag(...) first."
        )
    return _RAG_CHAIN


def get_rag_chain_for_mode(mode: str):
    mode_normalized = mode.lower().strip()
    if mode_normalized not in {"text", "visual", "mixed"}:
        raise ValueError("mode must be one of: text|visual|mixed")
    if not _RAG_CHAINS:
        raise RuntimeError(
            "RAG chains are not initialized. Call initialize_rag(...) first."
        )
    return _RAG_CHAINS[mode_normalized]


def get_rag_retriever_for_mode(mode: str):
    mode_normalized = mode.lower().strip()
    if mode_normalized not in {"text", "visual", "mixed"}:
        raise ValueError("mode must be one of: text|visual|mixed")
    if not _RETRIEVERS:
        raise RuntimeError(
            "RAG retrievers are not initialized. Call initialize_rag(...) first."
        )
    return _RETRIEVERS[mode_normalized]


def _recreate_collection_if_needed(
    client: QdrantClient,
    qdrant_mode: str,
    collection_name: str,
) -> None:
    # Auto-reset for embedded/memory modes only.
    if qdrant_mode in {"embedded", "memory"} and client.collection_exists(collection_name):
        client.delete_collection(collection_name)


def _add_docs_to_collection(
    client: QdrantClient,
    collection_name: str,
    docs: List[Document],
    *,
    use_sparse_model: bool,
) -> None:
    if not docs:
        return
    _ensure_collection_exists(
        client=client,
        collection_name=collection_name,
        use_sparse_model=use_sparse_model,
    )
    dense_vector_name = _get_dense_vector_name(
        client=client,
        collection_name=collection_name,
    )
    points: List[PointStruct] = []
    for i, doc in enumerate(docs):
        meta = doc.metadata or {}
        compact = _compact_metadata_for_qdrant(meta)
        payload = {
            k: (
                v
                if isinstance(v, (str, int, float, bool, type(None)))
                else str(v)
            )
            for k, v in compact.items()
        }
        payload["text"] = doc.page_content
        # Local embedded Qdrant validates string ids as UUIDs.
        # Keep readable chunk_id in payload, and use deterministic UUID for point id.
        point_key = str(payload.get("chunk_id") or f"{collection_name}:{i}")
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_key))
        points.append(
            PointStruct(
                id=point_id,
                vector={
                    dense_vector_name: QdrantDocument(
                        text=doc.page_content,
                        model=EMBED_MODEL_ID,
                    )
                },
                payload=payload,
            )
        )
    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )


def _build_modality_filter(
    modality_filter: Optional[Sequence[str]],
) -> Optional[Filter]:
    if not modality_filter:
        return None
    values = [str(v).lower() for v in modality_filter]
    if len(values) == 1:
        return Filter(
            must=[FieldCondition(key="modality", match=MatchValue(value=values[0]))]
        )
    return Filter(
        must=[FieldCondition(key="modality", match=MatchAny(any=values))]
    )


def _collection_or_fallback(
    *,
    client: QdrantClient,
    preferred: str,
    fallback: str,
) -> str:
    if client.collection_exists(preferred):
        return preferred
    return fallback


def _compact_metadata_for_qdrant(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep payload lean and query-friendly.
    Large nested fields stay in chunk dump artifacts, not vector payload.
    """
    keep_keys = {
        "source",
        "doc_id",
        "chunk_id",
        "chunk_index",
        "page",
        "page_start",
        "page_end",
        "modality",
        "heading",
        "char_count",
        "token_count_est",
        "vlm_preset",
    }
    compact = {k: meta[k] for k in keep_keys if k in meta}
    # Keep origin filename if available for easier debugging.
    dl_meta = meta.get("dl_meta")
    if isinstance(dl_meta, dict):
        origin = dl_meta.get("origin")
        if isinstance(origin, dict) and "filename" in origin:
            compact["origin_filename"] = origin.get("filename")
    return compact


def _ensure_collection_exists(
    *,
    client: QdrantClient,
    collection_name: str,
    use_sparse_model: bool,
) -> None:
    if client.collection_exists(collection_name):
        return
    create_kwargs = {
        "collection_name": collection_name,
        "vectors_config": client.get_fastembed_vector_params(),
    }
    if use_sparse_model:
        sparse_cfg = client.get_fastembed_sparse_vector_params()
        if sparse_cfg:
            create_kwargs["sparse_vectors_config"] = sparse_cfg
    client.create_collection(**create_kwargs)


def _get_dense_vector_name(
    *,
    client: QdrantClient,
    collection_name: str,
) -> str:
    """
    Resolve the active dense vector name used by the collection.
    """
    try:
        info = client.get_collection(collection_name=collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict) and vectors:
            return str(next(iter(vectors.keys())))
    except Exception:
        pass
    # Fallback to current fastembed model key.
    return str(next(iter(client.get_fastembed_vector_params().keys())))


def _validate_qdrant_mode(qdrant_mode: str) -> str:
    mode = (qdrant_mode or "").strip().lower()
    if mode not in QDRANT_MODES:
        raise ValueError(
            f"Unsupported qdrant_mode: {qdrant_mode}. Use one of {sorted(QDRANT_MODES)}."
        )
    return mode


def _build_qdrant_client(
    *,
    qdrant_mode: str,
    qdrant_location: str,
) -> QdrantClient:
    if qdrant_mode == "memory":
        return QdrantClient(location=":memory:")

    if qdrant_mode == "embedded":
        path = Path(qdrant_location).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path), check_compatibility=False)

    # remote mode
    url = qdrant_location.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        url = f"http://{url}"
    return QdrantClient(url=url)


def _health_check_qdrant(
    *,
    client: QdrantClient,
    qdrant_mode: str,
    qdrant_location: str,
) -> None:
    try:
        _ = client.get_collections()
    except Exception as exc:
        target = ":memory:" if qdrant_mode == "memory" else qdrant_location
        raise RuntimeError(
            f"Qdrant health check failed for mode={qdrant_mode}, target={target}. "
            "For remote mode, ensure the server URL is reachable. "
            "For embedded mode, ensure path is writable."
        ) from exc

