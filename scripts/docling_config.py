import os
from typing import Optional

from dotenv import load_dotenv
from langchain_docling.loader import ExportType


load_dotenv()


# Embedding model (Qdrant + Docling chunker)
# Qdrant FastEmbed does not support intfloat/e5-base-v2 directly.
# Use a supported E5-family model instead: intfloat/multilingual-e5-large
EMBED_MODEL_ID = "intfloat/multilingual-e5-large"

# Chunking: parse_doc-style alignment with embedding model (reproducible RAG benchmarking)
# Same tokenizer as embedding model + explicit max_tokens for consistent chunk sizes
DEFAULT_MAX_CHUNK_TOKENS = 512

# Generator LLM ID (kept for reference; actual generation uses local Ollama)
GEN_MODEL_ID = "llama3.2:3b"
DEFAULT_EXPORT_TYPE = ExportType.DOC_CHUNKS
DEFAULT_TOP_K = 3

# Qdrant connection defaults
DEFAULT_QDRANT_MODE = "embedded"  # embedded | remote | memory
DEFAULT_QDRANT_LOCATION = "data/qdrant_data"

# Input handling
SUPPORTED_FILE_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".html",
    ".xml",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".webp",
)

# VLM defaults (used by CLI and loader/runtime wiring)
DEFAULT_VLM_PRESET = "granite_docling"
DEFAULT_VLM_RUNTIME = "api"
DEFAULT_VLM_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_VLM_CONCURRENCY = 64
DEFAULT_VLM_MAX_TOKENS = 1024


def get_env(key: str) -> Optional[str]:
    return os.getenv(key)

