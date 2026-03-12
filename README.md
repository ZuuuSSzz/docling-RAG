# RAG benchmarking (Docling + LangChain)

Docling-based RAG pipeline with LangChain and **Qdrant** vector store (see [Docling Retrieval with Qdrant](https://docling-project.github.io/docling/examples/retrieval_qdrant/)). Default document source: `HF-eval/data/arxiv`.

Chunking uses **parse_doc-style** settings for reproducible RAG benchmarking: the same tokenizer as the embedding model and an explicit max tokens per chunk (default 512). Optional **VLM** (e.g. Granite) improves PDF conversion for complex layouts before chunking. Chunk metadata includes source/page/modality tags (text, table, image, chart) and is indexed to Qdrant payloads.

## Run with uv

From this directory (`rag-benchmarking`):

```bash
# Install dependencies (first time or after changing pyproject.toml)
uv sync

# Run the Docling RAG pipeline (uses HF-eval/data/arxiv by default)
# Vectors are stored locally in data/qdrant_data so they persist between runs
uv run python scripts/docling_rag_agent.py

# Explicit Qdrant mode examples
uv run python scripts/docling_rag_agent.py --qdrant-mode embedded --qdrant-location ./data/qdrant_data
uv run python scripts/docling_rag_agent.py --qdrant-mode remote --qdrant-location http://localhost:6333
uv run python scripts/docling_rag_agent.py --qdrant-mode memory

# Next time: reuse existing collection (no re-chunking or re-embedding)
uv run python scripts/docling_rag_agent.py --reuse-collection -q "Your question"

# Use a different path or Qdrant server (directory or file list)
uv run python scripts/docling_rag_agent.py -q "Your question" -f /path/to/docs --qdrant-location http://localhost:6333
```

Set `HF_TOKEN` in your environment (or `.env`) for HuggingFace Inference API.

## Using the Docling VLM pipeline

To convert PDFs using Docling's **VLM pipeline** (GraniteDocling preset with an API runtime), pass `--use-vlm` and point it at your OpenAI-compatible VLM endpoint:

```bash
uv run python scripts/docling_rag_agent.py \
  --use-vlm \
  --vlm-preset granite_docling \
  --vlm-url http://localhost:8000/v1/chat/completions \
  --vlm-concurrency 64
```

This follows Docling’s VLM examples:
- https://docling-project.github.io/docling/examples/minimal_vlm_pipeline/
- https://docling-project.github.io/docling/examples/vlm_pipeline_api_model/

For non-PDF inputs, standard Docling conversion is used. For mixed folders, PDFs can still use VLM mode while other formats are converted with the standard pipeline.

## Benchmark harness integration

Use `scripts/benchmark_runner.py` to evaluate a question set and write JSONL outputs with answers plus retrieval metadata:

```bash
uv run python scripts/benchmark_runner.py \
  --docs /home/user/development/HF-eval/data/arxiv \
  --questions /path/to/questions.jsonl \
  --output /path/to/results.jsonl \
  --retrieval-mode mixed
```

`questions.jsonl` lines should look like:
`{"id":"q1","question":"...","expected":"...","mode":"visual"}`  
where `mode` is optional and can be `text`, `visual`, or `mixed`.

By default, benchmark runner stores vectors in `rag-benchmarking/data/qdrant_data` (persistent).  
Use `--qdrant-location :memory:` only when you explicitly want in-memory storage.

To inspect chunking output directly, dump chunks while indexing:

```bash
uv run python scripts/benchmark_runner.py \
  --docs /home/user/development/HF-eval/data/arxiv \
  --questions /path/to/questions.jsonl \
  --output /path/to/results.jsonl \
  --dump-chunks-path /path/to/chunks.jsonl
```

To inspect stored vector payloads in Qdrant:

```bash
uv run python scripts/inspect_qdrant.py \
  --qdrant-mode embedded \
  --qdrant-location /home/user/development/IE-evaluation/rag-benchmarking/data/qdrant_data \
  --collection docling \
  --limit 20
```

## Local storage and reusing the collection

By default, Qdrant stores vectors in **`data/qdrant_data`** (under the `rag-benchmarking` directory), so data persists between runs. The first run will load PDFs, chunk, embed, and index. On later runs, pass **`--reuse-collection`** to skip loading/chunking/embedding and use the existing collection—you can just ask questions. To re-ingest (e.g. after adding or changing documents), run without `--reuse-collection`; the existing collection at that path will be replaced.

## Chunking options (RAG benchmarking)

Chunking is aligned with the embedding model for consistent retrieval. You can tune it via:

- `--max-chunk-tokens 512` – max tokens per chunk (default 512)
- `--no-tokenizer` – use default HybridChunker without a custom tokenizer (faster, less control)
- `--no-merge-peers` – disable merging of undersized peer chunks
