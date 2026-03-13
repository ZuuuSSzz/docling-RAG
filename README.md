# docling-RAG

A **RAG (Retrieval-Augmented Generation)** pipeline built with [Docling](https://github.com/DS4SD/docling), [LangChain](https://python.langchain.com/), and [Qdrant](https://qdrant.tech/). Ingest PDFs and other documents, chunk them with embedding-model–aligned tokenization, index to a local or remote vector store, and answer questions with an LLM.

## Features

- **Docling** for document conversion (PDF, Office, HTML, etc.) with optional **VLM** (e.g. Granite) for complex PDF layouts
- **Reproducible chunking**: tokenizer aligned with the embedding model and configurable max tokens per chunk (default 512)
- **Qdrant** for vector search: embedded (local), in-memory, or remote server
- **Benchmark harness**: run question sets and export answers + retrieval metadata to JSONL

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip

Optional:

- **Ollama** (or another LLM endpoint) for answer generation
- **Hugging Face token** (`HF_TOKEN`) if using HuggingFace Inference API for embeddings
- **VLM API** (e.g. vLLM with Granite) for `--use-vlm` PDF conversion

## Quick start

Clone the repo and run from the project root:

```bash
# Install dependencies
uv sync

# Run the RAG pipeline (interactive: you'll be prompted for a document path and questions)
uv run python scripts/docling_rag_agent.py
```

On first run you'll be asked for a **document path**: point it at a folder of PDFs (or other supported files) or a single file. Documents are chunked, embedded, and stored in `data/qdrant_data` by default. On later runs you can pass **`--reuse-collection`** to skip re-ingestion and only run queries.

## Basic usage

### Interactive Q&A

```bash
# Ingest docs from a folder and then ask questions
uv run python scripts/docling_rag_agent.py -f /path/to/your/docs -q "Your question"

# Reuse existing vector store (no re-chunking)
uv run python scripts/docling_rag_agent.py --reuse-collection -q "Your question"
```

### Qdrant modes

- **Embedded** (default): local files in `./data/qdrant_data`
- **Memory**: in-memory only (no persistence)
- **Remote**: connect to a Qdrant server

```bash
uv run python scripts/docling_rag_agent.py --qdrant-mode embedded --qdrant-location ./data/qdrant_data
uv run python scripts/docling_rag_agent.py --qdrant-mode memory
uv run python scripts/docling_rag_agent.py --qdrant-mode remote --qdrant-location http://localhost:6333
```

### Optional: HuggingFace

Set `HF_TOKEN` in your environment (or in a `.env` file in the project root) if you use HuggingFace for embeddings or inference.

## VLM pipeline (better PDFs)

For complex PDFs you can use Docling’s **VLM pipeline** (e.g. Granite) via an OpenAI-compatible API:

```bash
uv run python scripts/docling_rag_agent.py \
  --use-vlm \
  --vlm-preset granite_docling \
  --vlm-url http://localhost:8000/v1/chat/completions \
  --vlm-concurrency 64 \
  -f /path/to/pdfs
```

Non-PDF files still use the standard Docling pipeline. See Docling’s [VLM examples](https://docling-project.github.io/docling/examples/minimal_vlm_pipeline/) for server setup.

## Benchmark runner

Evaluate a set of questions and write results to JSONL:

```bash
uv run python scripts/benchmark_runner.py \
  --docs /path/to/your/docs \
  --questions /path/to/questions.jsonl \
  --output /path/to/results.jsonl \
  --retrieval-mode mixed
```

**Question file format** (`questions.jsonl`): one JSON object per line, e.g.:

```json
{"id": "q1", "question": "What is the main finding?", "expected": "Optional reference answer.", "mode": "mixed"}
```

`mode` is optional and can be `text`, `visual`, or `mixed`.

**Dump chunks** while indexing (for inspection):

```bash
uv run python scripts/benchmark_runner.py \
  --docs /path/to/docs \
  --questions /path/to/questions.jsonl \
  --output results.jsonl \
  --dump-chunks-path chunks.jsonl
```

## Inspect Qdrant contents

To peek at stored vectors and payloads:

```bash
uv run python scripts/inspect_qdrant.py \
  --qdrant-mode embedded \
  --qdrant-location ./data/qdrant_data \
  --collection docling \
  --limit 20
```

## Chunking options

Chunking is aligned with the embedding model for consistent retrieval. Useful flags:

| Option | Description |
|--------|-------------|
| `--max-chunk-tokens 512` | Max tokens per chunk (default 512) |
| `--no-tokenizer` | Use default HybridChunker (faster, less control) |
| `--no-merge-peers` | Disable merging of undersized peer chunks |

## Project layout

```
rag-benchmarking/
├── scripts/
│   ├── docling_rag_agent.py   # Main CLI: ingest + Q&A
│   ├── docling_rag_core.py    # RAG chain, retriever, Qdrant setup
│   ├── docling_loader.py      # Docling load + chunk
│   ├── docling_config.py      # Config and env
│   ├── benchmark_runner.py    # Batch evaluation
│   ├── inspect_qdrant.py      # Inspect vector store
│   └── rag_contract.py        # Response normalization
├── data/                      # Default Qdrant data (gitignored: data/qdrant_data/)
├── results/                   # Benchmark outputs (gitignored: results/*.jsonl)
├── pyproject.toml
└── README.md
```

## License

See the repository for license information.
