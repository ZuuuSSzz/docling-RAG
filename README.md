# docling-RAG

**RAG (Retrieval-Augmented Generation)** pipeline built with Docling + LangChain + Qdrant.

It:
- converts your documents to text (and optional visual chunks),
- chunks + embeds them into Qdrant,
- answers questions with a local LLM (via Ollama),
- optionally runs a small benchmark over a set of questions.

---

## 1. Methodology (how it works)

- **Ingestion**: Docling parses PDFs/Office/etc. into structured chunks (text, tables, images).
- **Chunking**: chunks are cut to a fixed token budget aligned with the embedding model.
- **Indexing**: chunks are embedded and stored in **Qdrant** (embedded, in-memory, or remote).
- **Retrieval**: for a question, top‑k relevant chunks are fetched from Qdrant.
- **Generation**: an Ollama LLM (default `llama3.2:3b`) answers using those chunks as context.

---

## 2. Requirements

- **Python**: 3.10+
- **Package manager**: `uv` (recommended) or `pip`
- **LLM**: [Ollama](https://ollama.com) running locally  
  - default model: `llama3.2:3b` (see `GEN_MODEL_ID` in `scripts/docling_config.py`)

Optional:
- **VLM backend** (Granite / vLLM / Ollama VLM / LM Studio) for complex PDFs (`--use-vlm`)
- **HF token** (`HF_TOKEN`) if you route anything through HuggingFace

---

## 3. Getting started (basic workflow)

From the repo root:

```bash
# 1) Install dependencies
uv sync

# 2) Make sure Ollama is running and model is pulled
ollama pull llama3.2:3b

# 3) Run RAG on your own docs
uv run python scripts/docling_rag_agent.py \
  -f /path/to/your/docs \
  -q "Your question"
```

What happens:
- documents under `/path/to/your/docs` are parsed and chunked,
- embeddings are stored in `data/qdrant_data`,
- the script prints the final answer + some basic source info.

To **reuse the same index** without re‑ingesting:

```bash
uv run python scripts/docling_rag_agent.py --reuse-collection -q "Another question"
```

---

## 4. Running the benchmark

You can evaluate the pipeline on a question set and store the results in JSONL.

```bash
uv run python scripts/benchmark_runner.py \
  --docs /path/to/your/docs \
  --questions /path/to/questions.jsonl \
  --output ./results/results.jsonl \
  --retrieval-mode mixed
```

Where each line in `questions.jsonl` looks like:

```json
{"id": "q1", "question": "What is the main finding?", "expected": "Optional reference answer.", "mode": "mixed"}
```

`mode` can be `text`, `visual`, or `mixed` (optional).

---

## 5. Useful options (quick reference)

- **Qdrant mode**:
  - `--qdrant-mode embedded --qdrant-location ./data/qdrant_data` (default, persistent)
  - `--qdrant-mode memory` (in‑memory only, throw‑away)
  - `--qdrant-mode remote --qdrant-location http://localhost:6333`
- **VLM for tough PDFs**:
  - add `--use-vlm` and set `--vlm-runtime` / `--vlm-url` / `--vlm-model` as needed.
- **Chunking** (for more control):
  - `--max-chunk-tokens 512`
  - `--no-tokenizer`
  - `--no-merge-peers`

---

## 6. Project layout

```text
rag-benchmarking/
├── scripts/
│   ├── docling_rag_agent.py    # Main CLI: ingest + Q&A
│   ├── docling_rag_core.py     # RAG chain, retriever, Qdrant setup
│   ├── docling_loader.py       # Docling load + chunk
│   ├── docling_config.py       # Config: models, defaults
│   ├── benchmark_runner.py     # Batch evaluation / benchmarking
│   ├── inspect_qdrant.py       # Inspect vector store contents
│   └── rag_contract.py         # Normalized answer + metadata format
├── data/                       # Default Qdrant data lives here
├── results/                    # Benchmark outputs (.jsonl)
├── pyproject.toml
└── README.md
```

---

## 7. When to enable VLM

- **Turn VLM ON (`--use-vlm`)** when:
  - your PDFs contain many **figures, charts, scanned pages, or screenshots**, and you care about visual reasoning.
  - you want `.chunks.md` exports for debugging complex visual documents.
- **Run with VLM (Ollama)** — minimal example:
  ```bash
  # 1) Start Ollama server and pull a VLM model
  ollama serve &
  ollama pull ibm/granite-docling:latest

  # 2) From the rag-benchmarking directory, run:
  uv run python scripts/docling_rag_agent.py \
    -f /path/to/your/docs/or/pdf \
    -q "Tell me about this document" \
    --use-vlm \
    --vlm-runtime ollama \
    --vlm-url http://localhost:11434/v1/chat/completions \
    --vlm-model ibm/granite-docling:latest
  ```
- **Leave VLM OFF** when:
  - documents are mostly **clean text + tables** and standard Docling parsing is enough.
  - you are doing large‑scale runs (hundreds/thousands of PDFs) and want to avoid the extra VLM latency/cost on the visual side.

