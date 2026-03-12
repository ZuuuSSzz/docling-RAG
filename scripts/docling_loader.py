from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from docling.chunking import HybridChunker
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from docling_config import (
    EMBED_MODEL_ID,
    DEFAULT_EXPORT_TYPE,
    DEFAULT_MAX_CHUNK_TOKENS,
    SUPPORTED_FILE_EXTENSIONS,
)


def build_chunker(
    *,
    embed_model_id: str = EMBED_MODEL_ID,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    use_tokenizer: bool = True,
    merge_peers: bool = True,
) -> HybridChunker:
    """
    Build HybridChunker with parse_doc-style tokenizer alignment (best for RAG benchmarking).

    When use_tokenizer is True, uses the same tokenizer as the embedding model and
    explicit max_tokens so chunk sizes are reproducible and aligned with embeddings.
    """
    if not use_tokenizer:
        return HybridChunker(merge_peers=merge_peers)
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embed_model_id),
        max_tokens=max_chunk_tokens,
    )
    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=merge_peers,
    )


def build_docling_loader(
    file_paths: Iterable[str],
    export_type: ExportType = DEFAULT_EXPORT_TYPE,
    *,
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
) -> List[Document]:
    """
    Build a DoclingLoader and return LangChain Document splits.

    Uses parse_doc-style chunking: tokenizer aligned with embedding model and
    explicit max_tokens for reproducible RAG benchmarking. Optional VLM for PDF conversion.
    """
    file_paths_list = _expand_input_sources(file_paths)
    if not file_paths_list:
        raise ValueError("file_paths must contain at least one path or URL.")

    chunker = build_chunker(
        embed_model_id=embed_model_id or EMBED_MODEL_ID,
        max_chunk_tokens=max_chunk_tokens,
        use_tokenizer=use_tokenizer,
        merge_peers=merge_peers,
    )

    if use_vlm:
        pdf_sources = [src for src in file_paths_list if str(src).lower().endswith(".pdf")]
        non_pdf_sources = [src for src in file_paths_list if src not in pdf_sources]

        docs: List[Document] = []
        if non_pdf_sources:
            # VLM pipeline currently targets PDFs; parse other formats with standard loader.
            loader = DoclingLoader(
                file_path=non_pdf_sources,
                export_type=export_type,
                chunker=chunker,
            )
            docs.extend(loader.load())

        if pdf_sources:
            docs.extend(
                _load_docs_with_vlm_pipeline(
                    file_paths_list=pdf_sources,
                    export_type=export_type,
                    chunker=chunker,
                    vlm_preset=vlm_preset,
                    vlm_runtime=vlm_runtime,
                    vlm_url=vlm_url,
                    vlm_concurrency=vlm_concurrency,
                    vlm_max_tokens=vlm_max_tokens,
                    vlm_model=vlm_model,
                    contextualize_chunks=contextualize_chunks,
                    debug_print_chunks=debug_print_chunks,
                )
            )
    else:
        loader = DoclingLoader(
            file_path=file_paths_list,
            export_type=export_type,
            chunker=chunker,
        )
        docs = loader.load()

    if export_type == ExportType.DOC_CHUNKS:
        splits = [
            _normalize_document_metadata(doc, chunk_index=i)
            for i, doc in enumerate(docs)
        ]
    elif export_type == ExportType.MARKDOWN:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        splits = []
        for doc in docs:
            for split in splitter.split_text(doc.page_content):
                split.metadata = {**(doc.metadata or {}), **(split.metadata or {})}
                splits.append(split)
        splits = [
            _normalize_document_metadata(doc, chunk_index=i)
            for i, doc in enumerate(splits)
        ]
    else:
        raise ValueError(f"Unexpected export type: {export_type}")

    return splits


def _load_docs_with_vlm_pipeline(
    *,
    file_paths_list: List[str],
    export_type: ExportType,
    chunker: HybridChunker,
    vlm_preset: str,
    vlm_runtime: str,
    vlm_url: str,
    vlm_concurrency: int,
    vlm_max_tokens: int,
    vlm_model: Optional[str],
    contextualize_chunks: bool,
    debug_print_chunks: int,
) -> List[Document]:
    """
    Load documents using Docling's VLM pipeline and return LangChain Documents.

    This follows Docling's VLM examples:
    - Minimal VLM pipeline (VlmPipeline + DocumentConverter)
      https://docling-project.github.io/docling/examples/minimal_vlm_pipeline/
    - VLM pipeline with remote API model (ApiVlmEngineOptions with VlmEngineType.API)
      https://docling-project.github.io/docling/examples/vlm_pipeline_api_model/
    """
    if export_type == ExportType.MARKDOWN:
        # Convert each PDF with VLM pipeline, export markdown, then split via Markdown headers.
        all_docs: List[Document] = []
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        for source in file_paths_list:
            docling_doc = _convert_pdf_with_vlm(
                source=source,
                vlm_preset=vlm_preset,
                vlm_runtime=vlm_runtime,
                vlm_url=vlm_url,
                vlm_concurrency=vlm_concurrency,
                vlm_max_tokens=vlm_max_tokens,
                vlm_model=vlm_model,
            )
            markdown = docling_doc.export_to_markdown()
            splits = splitter.split_text(markdown)
            for s in splits:
                s.metadata = {**(s.metadata or {}), "source": source, "vlm_preset": vlm_preset}
            all_docs.extend(splits)
        return all_docs

    # ExportType.DOC_CHUNKS (default): use Docling chunker and keep each chunk as a Document
    docs: List[Document] = []
    for source in file_paths_list:
        docling_doc = _convert_pdf_with_vlm(
            source=source,
            vlm_preset=vlm_preset,
            vlm_runtime=vlm_runtime,
            vlm_url=vlm_url,
            vlm_concurrency=vlm_concurrency,
            vlm_max_tokens=vlm_max_tokens,
            vlm_model=vlm_model,
        )
        # Follow Docling chunking example signature: chunker.chunk(dl_doc=doc)
        chunk_iter = list(chunker.chunk(dl_doc=docling_doc))
        if not chunk_iter:
            # Fallback: if chunker yields nothing, use markdown export as a single doc.
            # This helps debugging VLM conversion outputs.
            markdown = docling_doc.export_to_markdown()
            if markdown.strip():
                docs.append(
                    Document(
                        page_content=markdown,
                        metadata={"source": source, "vlm_preset": vlm_preset, "fallback": "markdown"},
                    )
                )
            continue

        # Prepare markdown file path to save contextualized chunks
        src_path = Path(source)
        out_md = src_path.with_suffix(".chunks.md")
        md_lines: list[str] = []

        for i, chunk in enumerate(chunk_iter):
            # Docling docs recommend embedding the contextualized text:
            # https://docling-project.github.io/docling/examples/hybrid_chunking/
            enriched_text = chunker.contextualize(chunk=chunk) if contextualize_chunks else chunk.text

            if debug_print_chunks and i < debug_print_chunks:
                # Mirror Docling hybrid chunking example output format
                print(f"=== {i} ===")
                print(f"chunk.text:\n{f'{chunk.text[:300]}…'!r}")
                print(f"chunker.contextualize(chunk):\n{f'{enriched_text[:300]}…'!r}")
                print()

            # Collect for markdown output
            md_lines.append(f"### Chunk {i}\n\n")
            md_lines.append(enriched_text)
            md_lines.append("\n\n---\n\n")

            raw_meta = chunk.meta.export_json_dict() if chunk.meta is not None else {}
            meta = _build_chunk_metadata(
                source=source,
                chunk_index=i,
                raw_meta=raw_meta,
                raw_chunk_text=chunk.text,
                vlm_preset=vlm_preset,
            )
            docs.append(Document(page_content=enriched_text, metadata=meta))

        # Write all contextualized chunks to a markdown file
        try:
            out_md.write_text("".join(md_lines), encoding="utf-8")
        except Exception:
            # If writing fails, continue without blocking the pipeline.
            pass
    return docs


def _convert_pdf_with_vlm(
    *,
    source: str,
    vlm_preset: str,
    vlm_runtime: str,
    vlm_url: str,
    vlm_concurrency: int,
    vlm_max_tokens: int,
    vlm_model: Optional[str],
):
    """
    Convert a PDF using Docling VlmPipeline with an API runtime (vLLM-style URL).

    Mirrors the pattern shown in Docling VLM API example for VLLM:
    https://docling-project.github.io/docling/examples/vlm_pipeline_api_model/
    """
    runtime_map = {
        "api": VlmEngineType.API,
        "ollama": VlmEngineType.API_OLLAMA,
        "lmstudio": VlmEngineType.API_LMSTUDIO,
    }
    runtime_type = runtime_map.get(vlm_runtime)
    if runtime_type is None:
        raise ValueError(
            f"Unsupported vlm_runtime: {vlm_runtime}. Use api|ollama|lmstudio."
        )

    # First, load preset so we can derive its runtime-specific model name when needed.
    preset_options = VlmConvertOptions.from_preset(vlm_preset)

    params = {
        # Avoid vLLM/OpenAI-style 400s when max_tokens exceeds remaining context.
        "max_tokens": int(vlm_max_tokens),
        # Some servers prefer/require this field name.
        "max_completion_tokens": int(vlm_max_tokens),
        "skip_special_tokens": True,
    }

    # Ollama / LM Studio runtimes expect a model name; Docling presets provide it.
    # Allow explicit override via --vlm-model.
    if vlm_model:
        params["model"] = vlm_model
    else:
        try:
            params["model"] = preset_options.model_spec.get_repo_id(runtime_type)
        except Exception:
            # If the preset can't provide it, leave it out (may still fail if runtime requires it).
            pass

    engine_kwargs = {
        "runtime_type": runtime_type,
        "concurrency": vlm_concurrency,
        "params": params,
    }
    # For generic API runtime, we must specify the URL (Docling VLLM example).
    if runtime_type == VlmEngineType.API:
        engine_kwargs["url"] = vlm_url

    vlm_options = VlmConvertOptions.from_preset(
        vlm_preset,
        engine_options=ApiVlmEngineOptions(**engine_kwargs),
    )

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )

    result = converter.convert(source=source)
    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(f"VLM conversion failed for {source}: status={result.status}")
    return result.document


def _expand_input_sources(file_paths: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for item in file_paths:
        p = Path(item)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_FILE_EXTENSIONS:
                    expanded.append(str(child))
            continue
        expanded.append(item)
    # Keep insertion order while deduplicating.
    deduped = list(dict.fromkeys(expanded))
    return deduped


def _normalize_document_metadata(doc: Document, *, chunk_index: int) -> Document:
    metadata = dict(doc.metadata or {})
    source = str(metadata.get("source", "")).strip() or "unknown"
    metadata["source"] = source
    doc_id = str(metadata.get("doc_id") or Path(source).name or "unknown-doc")
    metadata["doc_id"] = doc_id

    page_start, page_end = _extract_page_span(metadata)
    if page_start is not None:
        metadata["page_start"] = page_start
        metadata["page"] = page_start
    if page_end is not None:
        metadata["page_end"] = page_end

    if "chunk_id" not in metadata:
        metadata["chunk_id"] = _derive_chunk_id(
            source=source,
            doc_id=doc_id,
            chunk_index=chunk_index,
            page_start=page_start,
            page_end=page_end,
        )
    metadata["chunk_index"] = int(chunk_index)
    metadata["char_count"] = len(doc.page_content or "")
    metadata["token_count_est"] = len((doc.page_content or "").split())
    heading = _extract_first_heading(metadata)
    if heading:
        metadata["heading"] = heading
    metadata["modality"] = str(
        metadata.get("modality") or _infer_modality_from_meta(metadata)
    )
    doc.metadata = metadata
    return doc


def _build_chunk_metadata(
    *,
    source: str,
    chunk_index: int,
    raw_meta: Dict[str, Any],
    raw_chunk_text: str,
    vlm_preset: str,
) -> Dict[str, Any]:
    page_start, page_end = _extract_page_span(raw_meta)
    modality = _infer_modality_from_meta(raw_meta)
    doc_id = str(raw_meta.get("doc_id") or Path(source).name)
    heading = _extract_first_heading(raw_meta)
    metadata = {
        **raw_meta,
        "source": source,
        "doc_id": doc_id,
        "chunk_id": _derive_chunk_id(
            source=source,
            doc_id=doc_id,
            chunk_index=chunk_index,
            page_start=page_start,
            page_end=page_end,
        ),
        "chunk_index": chunk_index,
        "vlm_preset": vlm_preset,
        "raw_chunk_text": raw_chunk_text,
        "char_count": len(raw_chunk_text or ""),
        "token_count_est": len((raw_chunk_text or "").split()),
        "modality": modality,
    }
    if page_start is not None:
        metadata["page_start"] = page_start
        metadata["page"] = page_start
    if page_end is not None:
        metadata["page_end"] = page_end
    if heading:
        metadata["heading"] = heading
    return metadata


def _derive_chunk_id(
    *,
    source: str,
    doc_id: str,
    chunk_index: int,
    page_start: Optional[int],
    page_end: Optional[int],
) -> str:
    if page_start is None and page_end is None:
        return f"{doc_id}:c{chunk_index:05d}"
    if page_end is None or page_end == page_start:
        return f"{doc_id}:p{page_start}:c{chunk_index:05d}"
    return f"{doc_id}:p{page_start}-{page_end}:c{chunk_index:05d}"


def _extract_first_value(meta: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in meta and meta[key] not in (None, ""):
            return meta[key]
    # Also inspect nested dictionaries that appear in exported chunk metadata.
    for value in meta.values():
        if isinstance(value, dict):
            nested = _extract_first_value(value, keys)
            if nested is not None:
                return nested
    return None


def _extract_page_span(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    direct = _extract_first_value(meta, ("page_start", "page"))
    direct_end = _extract_first_value(meta, ("page_end",))
    if isinstance(direct, int):
        start = int(direct)
        end = int(direct_end) if isinstance(direct_end, int) else start
        return start, end

    pages: List[int] = []

    def _collect(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"page_no", "page", "page_number"} and isinstance(value, int):
                    pages.append(int(value))
                _collect(value)
        elif isinstance(node, list):
            for item in node:
                _collect(item)

    _collect(meta)
    if not pages:
        return None, None
    return min(pages), max(pages)


def _extract_first_heading(meta: Dict[str, Any]) -> Optional[str]:
    headings = meta.get("headings")
    if isinstance(headings, list) and headings:
        first = str(headings[0]).strip()
        if first:
            return first[:300]
    heading = meta.get("heading")
    if isinstance(heading, str) and heading.strip():
        return heading.strip()[:300]
    return None


def _infer_modality_from_meta(meta: Dict[str, Any]) -> str:
    flat = str(meta).lower()
    if any(term in flat for term in ("chart", "plot", "graph")):
        return "chart"
    if any(term in flat for term in ("figure", "image", "picture")):
        return "image"
    if "table" in flat:
        return "table"
    return "text"

