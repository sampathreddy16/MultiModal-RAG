# doc-parser

A full **multimodal RAG pipeline** — from raw PDFs to re-ranked answers — powered by **PP-DocLayout-V3** (layout detection), **GLM-OCR 0.9B** (text/table/formula recognition), **OpenAI embeddings**, and a pluggable re-ranker. Runs on the **Z.AI MaaS cloud API** with no GPU required.

Ranked **#1 on OmniDocBench V1.5** (94.62 score, March 2026).

---

## What This Does

Feed it a PDF → get back clean Markdown, structured JSON, RAG-ready chunks, and — after ingestion — answers to natural-language queries with re-ranked context.

```
PDF / Image
    ↓
Phase 1 — Parse
  Z.AI MaaS API
    ├── PP-DocLayout-V3  (detects 23 element categories)
    └── GLM-OCR 0.9B     (text, HTML tables, LaTeX formulas)
  Post-Processor → Markdown + JSON
  Structure-Aware Chunker → RAG chunks (text | image | table | formula)
    ↓
Phase 2 — Ingest
  GPT-4o image captioner (enriches image/figure chunks with text descriptions)
  Pluggable embeddings — OpenAI (text-embedding-3-large / 3-small) or Gemini (gemini-embedding-2-preview)
  Feature-hashed BM25 sparse vectors
  Qdrant hybrid vector store (dense + sparse, RRF fusion)
    ↓
Phase 3 — Search & Re-rank
  Hybrid Qdrant search → top-k candidates (dense + sparse RRF)
  Re-ranker → top-n results (OpenAI · Jina · BGE · Qwen VL)
    ↓
Phase 4 — REST API (FastAPI)
  POST /ingest    → parse → caption → embed → Qdrant upsert
  POST /search    → embed query → hybrid search → rerank → JSON
  GET  /health    → ping Qdrant + OpenAI
  GET  /collections → list Qdrant collections
    ↓
  LLM generation (GPT-4o)
```

**23 element categories detected:** document title, paragraph title, paragraph, abstract, table, formula, inline formula, figure, caption, header, footer, footnote, code block, algorithm, reference, page number, seal, and more.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12+ | Required |
| uv | latest | Package manager (replaces pip) |
| Docker | latest | For local Qdrant (optional — use Qdrant Cloud instead) |
| Z.AI API key | — | Get one at [z.ai](https://z.ai) |
| OpenAI API key | — | For embeddings, captioning, and generation |

### Install `uv` (if you don't have it)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Quick Start (5 minutes)

### Step 1 — Clone / navigate to the project

```bash
cd multi-modal-rag
```

### Step 2 — Create and activate a virtual environment

```bash
uv venv --python 3.12
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
uv pip install -e ".[dev]"
```

Optional extras:

```bash
uv pip install -e ".[bge]"    # BAAI/bge-reranker-v2-minicpm-layerwise (local, fast)
uv pip install -e ".[qwen]"   # Qwen3-VL-Reranker-2B (local, multimodal, heavier)
uv pip install -e ".[gemini]" # Google Gemini embeddings (gemini-embedding-2-preview)
uv pip install -e ".[layout]" # PP-DocLayout-V3 local layout detection (Ollama mode)
```

### Step 4 — Configure your API keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```dotenv
Z_AI_API_KEY=your-z-ai-key
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333   # or your Qdrant Cloud URL
```

> **Never commit `.env` to git.** It is already in `.gitignore`.

### Step 5 — Start Qdrant (local Docker)

```bash
docker-compose up -d qdrant
```

Or point `QDRANT_URL` at your [Qdrant Cloud](https://cloud.qdrant.io) cluster and set `QDRANT_API_KEY`.

### Step 6 — Parse a document

```bash
python scripts/parse.py path/to/paper.pdf --chunks
```

### Step 7 — Ingest into Qdrant

```bash
python scripts/ingest.py path/to/paper.pdf
```

### Step 8 — Search with re-ranking

```bash
python scripts/search.py "What is the transformer attention mechanism?"
```

### Step 9 — Start the REST API (optional)

```bash
python scripts/serve.py --reload      # dev mode with auto-reload
# or
python scripts/serve.py --port 8080   # custom port
```

Interactive docs open automatically at **http://localhost:8000/docs**.

---

## CLI Reference

### `scripts/parse.py` — Parse PDFs

```
python scripts/parse.py <input> [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | *(required)* | PDF/image file **or** directory of documents |
| `--output` | `./output/` | Where to save results |
| `--format` | `both` | `markdown`, `json`, or `both` |
| `--chunks` | off | Also write `{name}_chunks.json` |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, or `WARNING` |

```bash
# Parse a single PDF — Markdown + JSON
python scripts/parse.py paper.pdf

# Parse and also generate RAG chunks
python scripts/parse.py paper.pdf --chunks

# Parse a directory, Markdown only
python scripts/parse.py ./docs/ --format markdown --output ./parsed/
```

---

### `scripts/ingest.py` — Embed + Upsert to Qdrant

Accepts PDFs and images (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`).

```
python scripts/ingest.py <input> [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | *(required)* | Document file or directory of documents |
| `--no-captions` | off | Skip GPT-4o image captioning |
| `--collection` | from `.env` | Override Qdrant collection name |
| `--overwrite` | off | Delete and recreate the collection first |
| `--max-chunk-tokens` | `512` | Max tokens per chunk |

```bash
# Ingest a PDF
python scripts/ingest.py paper.pdf

# Ingest a standalone image
python scripts/ingest.py figure.png

# Ingest a mixed directory (PDFs + images)
python scripts/ingest.py ./docs/ --no-captions

# Recreate the collection from scratch
python scripts/ingest.py ./docs/ --overwrite

# Ingest into a named collection with smaller chunks
python scripts/ingest.py paper.pdf --collection my_collection --max-chunk-tokens 256
```

---

### `scripts/search.py` — Query + Re-rank

```
python scripts/search.py "<query>" [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `query` | *(required)* | Natural-language search query |
| `--top-k` | `20` | Candidates to retrieve before re-ranking |
| `--top-n` | from `.env` | Results to keep after re-ranking |
| `--backend` | from `.env` | Re-ranker: `openai`, `jina`, `bge`, `qwen` |
| `--filter-modality` | all | Restrict to `text`, `image`, `table`, or `formula` |
| `--no-rerank` | off | Print raw retrieval results, skip re-ranking |
| `--collection` | from `.env` | Override Qdrant collection name |

```bash
# Default (OpenAI re-ranker, already in stack)
python scripts/search.py "attention mechanism in transformers"

# Search for figures only, using Jina multimodal re-ranker
python scripts/search.py "bar chart comparing model performance" \
    --filter-modality image --backend jina

# Fast local re-ranking with BGE (needs: uv pip install '.[bge]')
python scripts/search.py "accuracy results table" --backend bge

# Skip re-ranking — raw retrieval results
python scripts/search.py "query" --no-rerank --top-k 10

# Compare raw vs re-ranked for the same query
python scripts/search.py "transformer self-attention" --no-rerank
python scripts/search.py "transformer self-attention" --backend openai
```

---

## REST API

The FastAPI server exposes the full pipeline as an HTTP API. Start it with:

```bash
python scripts/serve.py [--host HOST] [--port PORT] [--workers N] [--reload]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |
| `--workers` | `1` | Uvicorn worker count |
| `--reload` | off | Auto-reload on file changes (dev only) |

**Interactive docs:** `http://localhost:8000/docs`

---

### `GET /health`

Pings Qdrant and OpenAI. Returns the status of each dependency.

**Response `200 OK`:**

```json
{
  "status": "ok",
  "qdrant": "ok",
  "openai": "ok",
  "reranker_backend": "openai"
}
```

`status` is `"ok"` when both services are reachable, otherwise `"degraded"`.

```bash
curl http://localhost:8000/health
```

---

### `GET /collections`

Lists all Qdrant collection names.

**Response `200 OK`:**

```json
{
  "collections": ["documents", "papers"]
}
```

```bash
curl http://localhost:8000/collections
```

---

### `POST /search`

Hybrid search with optional reranking. Embeds the query, retrieves candidates from Qdrant, and optionally reranks them.

**Request body:**

```json
{
  "query": "transformer attention mechanism",
  "top_k": 20,
  "top_n": 5,
  "filter_modality": null,
  "rerank": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | *(required)* | Natural-language query |
| `top_k` | int | `20` | Candidates to retrieve from Qdrant (1–200) |
| `top_n` | int \| null | from `.env` | Results to keep after reranking |
| `filter_modality` | string \| null | `null` | Restrict to `"text"` \| `"image"` \| `"table"` \| `"formula"` |
| `rerank` | bool | `true` | Set `false` to return raw Qdrant results |

**Response `200 OK`:**

```json
{
  "query": "transformer attention mechanism",
  "backend": "openai",
  "total_candidates": 20,
  "latency_ms": 1243.5,
  "results": [
    {
      "chunk_id": "paper.pdf_3_1",
      "text": "The attention function maps a query and a set of key-value pairs...",
      "source_file": "paper.pdf",
      "page": 3,
      "modality": "text",
      "element_types": ["paragraph"],
      "bbox": null,
      "is_atomic": false,
      "caption": null,
      "rerank_score": 9.0,
      "image_base64": null
    }
  ]
}
```

```bash
# Default (rerank=true, top_k=20, top_n from .env)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism"}'

# Image chunks only, no reranking, top-10
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "accuracy chart", "filter_modality": "image", "rerank": false, "top_k": 10}'

# Custom top_k / top_n
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "loss function", "top_k": 50, "top_n": 3}'
```

---

### `POST /ingest` — by file path

Ingest a document (PDF or image) referenced by its local path on the server.

**Request body:**

```json
{
  "file_path": "/absolute/path/to/paper.pdf",
  "collection": null,
  "overwrite": false,
  "max_chunk_tokens": 512,
  "caption": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file_path` | string | *(required)* | Absolute path to the document (PDF or image) |
| `collection` | string \| null | from `.env` | Override Qdrant collection name |
| `overwrite` | bool | `false` | Delete and recreate collection before ingesting |
| `max_chunk_tokens` | int | `512` | Max tokens per text chunk (64–4096) |
| `caption` | bool | `true` | Run GPT-4o captioning on image chunks |

**Response `200 OK`:**

```json
{
  "source_file": "/path/to/paper.pdf",
  "collection": "documents",
  "chunks_upserted": 87,
  "modality_counts": {"text": 72, "image": 10, "table": 4, "formula": 1},
  "latency_ms": 18432.1
}
```

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/data/papers/attention.pdf"}'

# Ingest an image file
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/data/figures/chart.png"}'

# Overwrite collection, skip captions
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/data/paper.pdf", "overwrite": true, "caption": false}'
```

---

### `POST /ingest/file` — multipart file upload

Upload a PDF or image directly from the client (no server-side path required). Supported types: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`.

**Form fields** (multipart/form-data):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | *(required)* | PDF or image file to upload |
| `collection` | string | from `.env` | Override collection name |
| `overwrite` | bool | `false` | Recreate collection before ingesting |
| `max_chunk_tokens` | int | `512` | Max tokens per chunk |
| `caption` | bool | `true` | Run GPT-4o captioning |

**Response:** same as `POST /ingest`.

```bash
# Upload a PDF
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@/path/to/paper.pdf"

# Upload an image
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@figure.png"

# Upload with options
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@paper.pdf" \
  -F "overwrite=true" \
  -F "caption=false" \
  -F "max_chunk_tokens=256"
```

---

### Error responses

All endpoints return standard HTTP error codes:

| Status | Meaning |
|--------|---------|
| `400` | Bad request (e.g. unsupported file type uploaded) |
| `404` | File not found (path-based ingest) |
| `422` | Validation error (invalid request body fields) |
| `500` | Internal server error (parsing failed) |
| `502` | Upstream error (Qdrant or OpenAI call failed) |

Every response includes an `X-Request-Id` header (8-char UUID prefix) for log correlation.

---

## Embedding Providers

The embedding layer is provider-agnostic. Both ingestion and search use the same `BaseEmbedder` interface, so switching providers only requires changing `.env` (and recreating the Qdrant collection if the vector dimensions change).

| Provider | Model | Dimensions | Install |
|----------|-------|-----------|---------|
| `openai` (default) | `text-embedding-3-large` | 3072 | None |
| `openai` | `text-embedding-3-small` | 1536 | None |
| `gemini` | `gemini-embedding-2-preview` | 3072 | `.[gemini]` |

### Switch to `text-embedding-3-small`

```dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

```bash
# Recreate collection (dimension change requires --overwrite)
python scripts/ingest.py paper.pdf --overwrite
```

### Switch to Gemini

```bash
uv pip install -e ".[gemini]"
```

```dotenv
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-2-preview
EMBEDDING_DIMENSIONS=3072
GEMINI_API_KEY=AIzaSy...
```

```bash
python scripts/ingest.py paper.pdf --overwrite
```

> **Warning:** Changing `EMBEDDING_DIMENSIONS` after a collection has been created requires `--overwrite` (or recreating via `DELETE /collections/{name}`). Dense and sparse vectors from different providers are not compatible.

---

## Re-ranker Backends

The re-ranker sits between Qdrant retrieval and LLM generation. It re-scores the top-k candidates and narrows to top-n with higher precision.

| Backend | Type | Multimodal | Cost | Latency | Extra install |
|---------|------|-----------|------|---------|---------------|
| `openai` | GPT-4o-mini cross-encoder | Yes (vision) | ~$0.03–0.10/query | 800ms–2s | None (default) |
| `jina` | Jina Reranker M0 API | Yes (Qwen2-VL-2B) | ~$0.01–0.02/query | 500ms–2s | `JINA_API_KEY` |
| `bge` | BAAI/bge-reranker-v2-minicpm-layerwise | Text (uses captions) | Free | **50–100ms** | `.[bge]` |
| `qwen` | Qwen3-VL-Reranker-2B (local) | Yes (raw images) | Free | 400–800ms | `.[qwen]` |

Set the default in `.env`:

```dotenv
RERANKER_BACKEND=openai   # or jina, bge, qwen
RERANKER_TOP_N=5
JINA_API_KEY=             # only needed for jina backend
```

**How image chunks flow through the re-ranker:**
- Every image chunk in Qdrant carries both a GPT-4o `caption` (text) and the raw `image_base64` (PNG bytes).
- `openai` and `jina` backends pass the raw image inline for true visual scoring.
- `bge` uses the caption text — still high quality since GPT-4o captions are detailed.
- `qwen` decodes and passes the raw image to the local VLM.

---

## Output Format Examples

### Markdown output (`document.md`)

```markdown
# Deep Learning for Document Understanding

**Abstract:** We propose a two-stage pipeline combining layout detection with ...

## 1. Introduction

Document understanding is a fundamental challenge in NLP ...

## 2. Method

$$
\mathcal{L} = \sum_{i=1}^{N} \ell(y_i, \hat{y}_i)
$$

| Model | F1 | Notes |
|-------|----|-------|
| Ours  | 94.62 | OmniDocBench V1.5 |
```

### JSON output (`document.json`)

```json
{
  "source_file": "paper.pdf",
  "total_elements": 42,
  "pages": [
    {
      "page_num": 1,
      "markdown": "# Deep Learning for Document Understanding\n\n...",
      "elements": [
        {
          "label": "document_title",
          "text": "Deep Learning for Document Understanding",
          "bbox": [103, 52, 897, 118],
          "score": 1.0,
          "reading_order": 0
        }
      ]
    }
  ]
}
```

### Chunks output (`document_chunks.json`)

```json
[
  {
    "text": "# Introduction\n\nDocument understanding is a fundamental challenge ...",
    "chunk_id": "paper.pdf_1_0",
    "page": 1,
    "element_types": ["paragraph_title", "paragraph"],
    "bbox": null,
    "source_file": "paper.pdf",
    "is_atomic": false,
    "modality": "text",
    "image_base64": null,
    "caption": null
  },
  {
    "text": "Figure 3: Accuracy vs recall curve across all models.",
    "chunk_id": "paper.pdf_2_1",
    "page": 2,
    "element_types": ["figure"],
    "bbox": [100, 400, 900, 750],
    "source_file": "paper.pdf",
    "is_atomic": true,
    "modality": "image",
    "image_base64": "<base64-encoded PNG>",
    "caption": "Figure 3: Accuracy vs recall curve across all models."
  }
]
```

> **`is_atomic: true`** means the chunk is a table, formula, figure, or algorithm — it is never split mid-element.
>
> **`modality`** is one of `"text"`, `"image"`, `"table"`, or `"formula"`.
>
> **`image_base64`** is populated only for image/figure chunks (the crop is stored in the Qdrant payload so re-rankers can access raw pixels without a second lookup).

---

## Streamlit Visual Inspector

Two visualizers are available — one for cloud results, one for local Ollama results.

### Cloud API visualizer (`app.py`)

```bash
uv run streamlit run app.py
```

- **Upload any PDF** and parse it with a single button click (calls Z.AI cloud API)
- **Page slider** — jump to any page in a multi-page document
- **Color-coded bounding boxes** — each element category gets its own color (titles red, paragraphs green, tables orange, formulas purple, …)
- **Legend** — shows only element types present on the current page
- **Element breakdown** — count per element type per page
- **Element list** — expandable list in reading order with raw text and bbox coordinates
- **Page Markdown** — rendered Markdown for the current page
- **Full document Markdown** — collapsible expander at the bottom

> Bounding boxes are normalized to 0–1000 by the Z.AI API. Pixel formula: `pixel = bbox_value × image_dimension / 1000`.

### Ollama results visualizer (`ollama/visualize.py`)

Reads pre-saved `ollama/output/*_elements.json` files — no API key or running service needed.

```bash
uv run streamlit run ollama/visualize.py
```

- **Dropdown** — select any `*_elements.json` from `ollama/output/`
- **PDF auto-detected** from `data/raw/` by matching filename stem; fallback file uploader if not found
- **Color-coded bounding boxes** — same color scheme, extended for PP-DocLayoutV3 labels (`doc_title`, `aside_text`, `paragraph_title`, `footnote`, …)
- **Polygon overlays** — toggle precise polygon outlines (Ollama outputs polygon points in addition to bounding boxes)
- **Element content** — expandable list with content text, bbox coords, and polygon point count
- **Full document Markdown** — collapsible expander showing the paired `*.md` file

---

## Ollama / Local Mode (No Cloud API Key)

The `ollama/` folder provides a fully local alternative to the Z.AI cloud pipeline. It runs the same GLM-OCR model and PP-DocLayout-V3 layout detector on your machine via [Ollama](https://ollama.com).

### Prerequisites

```bash
# 1. Install and start Ollama
brew install ollama   # macOS; see ollama.com for Linux
ollama serve          # leave running in a terminal

# 2. Pull the model (~600 MB)
ollama pull glm-ocr:latest

# 3. Install layout detection dependencies
uv pip install -e ".[layout]"
# (downloads PP-DocLayout-V3 weights ~400 MB from HuggingFace on first run)
```

### Parse a document locally

```bash
uv run python ollama/test_parse.py data/raw/paper.pdf
uv run python ollama/test_parse.py data/raw/paper.pdf --show-elements
```

Output saved to `ollama/output/`:
- `paper.md` — extracted Markdown
- `paper_elements.json` — raw per-page element JSON (same `bbox_2d`/`polygon` schema)

### Visualize saved results

```bash
uv run streamlit run ollama/visualize.py
```

### Cloud vs Ollama comparison

| | Cloud API (Z.AI) | Ollama (local) |
|---|---|---|
| API key required | Yes (`Z_AI_API_KEY`) | No |
| Cost | API credits | Free |
| Speed | Fast (cloud GPU) | 5–30s/page (CPU/MPS/CUDA) |
| Privacy | Data sent to Z.AI | Fully local |
| Layout detection | PP-DocLayout-V3 | PP-DocLayout-V3 (same) |
| Element schema | `label`, `text`, `bbox`, `reading_order` | `label`, `content`, `bbox_2d`, `polygon`, `index` |

See `ollama/README.md` for full setup details.

---

## Using the Python API

### Parse a document

```python
import sys
sys.path.insert(0, "src")

from pathlib import Path
from doc_parser.pipeline import DocumentParser
from doc_parser.chunker import structure_aware_chunking

parser = DocumentParser()
result = parser.parse_file(Path("my_paper.pdf"))

all_chunks = []
for page in result.pages:
    chunks = structure_aware_chunking(
        page.elements,
        source_file="my_paper.pdf",
        page=page.page_num,
        max_chunk_tokens=512,
    )
    all_chunks.extend(chunks)
```

### Ingest into Qdrant

```python
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from doc_parser.config import get_settings
from doc_parser.ingestion.embedder import embed_chunks, get_embedder
from doc_parser.ingestion.image_captioner import enrich_image_chunks
from doc_parser.ingestion.vector_store import QdrantDocumentStore

async def ingest(chunks):
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    # Enrich image chunks with GPT-4o captions
    chunks = await enrich_image_chunks(chunks, pdf_path=Path("paper.pdf"), client=client)

    # Embed (dense + sparse) — provider determined by EMBEDDING_PROVIDER in .env
    embedder = get_embedder(settings)
    dense, sparse = await embed_chunks(chunks, embedder, settings)

    # Upsert to Qdrant
    store = QdrantDocumentStore(settings)
    await store.create_collection()
    await store.upsert_chunks(chunks, dense, sparse)

asyncio.run(ingest(all_chunks))
```

### Search and re-rank

```python
import asyncio
from doc_parser.config import get_settings
from doc_parser.ingestion.embedder import get_embedder
from doc_parser.ingestion.vector_store import QdrantDocumentStore
from doc_parser.retrieval.reranker import get_reranker

async def search(query: str):
    settings = get_settings()
    embedder = get_embedder(settings)   # provider from EMBEDDING_PROVIDER in .env

    store = QdrantDocumentStore(settings)
    candidates = await store.search(query, embedder, settings, top_k=20)

    reranker = get_reranker(settings)   # backend from RERANKER_BACKEND in .env
    results = await reranker.rerank(query, candidates, top_n=5)

    for r in results:
        print(f"[{r['rerank_score']:.3f}] ({r['modality']}) {r['text'][:120]}")

asyncio.run(search("transformer attention mechanism"))
```

---

## Project Structure

```
multi-modal-rag/
├── pyproject.toml              # Dependencies and tool config
├── config.yaml                 # GLM-OCR SDK cloud settings
├── docker-compose.yml          # Qdrant local service
├── .env.example                # Template for all API keys
├── .env                        # Your actual keys (never commit)
│
├── src/
│   └── doc_parser/
│       ├── config.py           # pydantic-settings: all env vars
│       ├── pipeline.py         # DocumentParser — wraps glmocr SDK
│       ├── post_processor.py   # Elements → Markdown
│       ├── chunker.py          # Structure-aware chunker (RAG)
│       │
│       ├── ingestion/          # Phase 2: embed + store
│       │   ├── embedder.py     # BaseEmbedder + OpenAI/Gemini providers + BM25 sparse
│       │   ├── image_captioner.py  # GPT-4o captions for figures
│       │   └── vector_store.py # Qdrant hybrid search wrapper
│       │
│       ├── retrieval/          # Phase 3: re-ranking
│       │   └── reranker.py     # BaseReranker + OpenAI/Jina/BGE/Qwen backends
│       │
│       ├── api/                # Phase 4: FastAPI REST layer
│       │   ├── app.py          # App factory + lifespan
│       │   ├── dependencies.py # Shared deps (OpenAI client, store, reranker)
│       │   ├── middleware.py   # Loguru request/response logging
│       │   ├── schemas.py      # Pydantic request/response models
│       │   └── routes/
│       │       ├── health.py   # GET /health, GET /collections
│       │       ├── ingest.py   # POST /ingest, POST /ingest/file
│       │       └── search.py   # POST /search
│       │
│       ├── logging_config.py   # Loguru setup + stdlib interception
│       └── utils/
│           └── pdf_utils.py    # PyMuPDF helpers
│
├── app.py                      # Streamlit visual inspector (cloud API results)
│
├── ollama/                     # Local GLM-OCR pipeline (no cloud API needed)
│   ├── config.yaml             # Ollama-specific glmocr SDK config
│   ├── test_parse.py           # CLI: parse PDF with local Ollama + PP-DocLayout-V3
│   ├── visualize.py            # Streamlit visualizer for saved Ollama results
│   └── output/                 # Saved results: *_elements.json + *.md
│
├── scripts/
│   ├── parse.py                # CLI: PDF → Markdown + JSON + chunks
│   ├── ingest.py               # CLI: PDF → embed → Qdrant
│   ├── search.py               # CLI: query → retrieve → rerank → display
│   └── serve.py                # Launch uvicorn API server
│
├── tests/
│   ├── conftest.py
│   ├── unit/                   # All mocked — no API key needed
│   │   ├── test_chunker.py
│   │   ├── test_embedder.py
│   │   ├── test_post_processor.py
│   │   ├── test_reranker.py    # 17 tests covering all 4 backends
│   │   ├── test_vector_store.py
│   │   └── test_api_schemas.py # 15 schema validation tests
│   └── integration/            # Require live API keys
│       ├── test_ingest_e2e.py
│       └── test_pipeline_e2e.py
│
├── notebooks/
│   └── 01_quickstart.ipynb
│
└── data/
    ├── raw/                    # Source documents
    └── processed/              # Intermediate files (gitignored)
```

---

## Running Tests

```bash
# Unit tests only (no API key needed — runs in ~2 seconds)
uv run pytest tests/unit/ -v

# API schema tests specifically
uv run pytest tests/unit/test_api_schemas.py -v

# Re-ranker tests specifically
uv run pytest tests/unit/test_reranker.py -v

# Integration tests (requires Z_AI_API_KEY + OPENAI_API_KEY + running Qdrant)
uv run pytest tests/integration/ -v

# All tests
uv run pytest -v
```

---

## Development

### Lint and format

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff check --fix src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
```

### Type checking

```bash
uv run mypy src/
```

---

## Configuration Reference

### `.env` variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PARSER_BACKEND` | No | `cloud` | Parser backend: `cloud` (Z.AI MaaS) \| `ollama` (local Ollama) |
| `Z_AI_API_KEY` | Yes* | — | Z.AI cloud API key (required when `PARSER_BACKEND=cloud`) |
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key (captioning, generation, and embeddings when `EMBEDDING_PROVIDER=openai`) |
| `OPENAI_LLM_MODEL` | No | `gpt-4o` | LLM model for generation |
| `EMBEDDING_PROVIDER` | No | `openai` | Embedding backend: `openai` \| `gemini` |
| `EMBEDDING_MODEL` | No | `text-embedding-3-large` | Embedding model name |
| `EMBEDDING_DIMENSIONS` | No | `3072` | Embedding vector size (1536 for `text-embedding-3-small`) |
| `GEMINI_API_KEY` | No† | — | Google Gemini API key (required when `EMBEDDING_PROVIDER=gemini`) |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | No | — | Qdrant Cloud API key |
| `QDRANT_COLLECTION_NAME` | No | `documents` | Collection name |
| `RERANKER_BACKEND` | No | `openai` | `openai` \| `jina` \| `bge` \| `qwen` |
| `RERANKER_TOP_N` | No | `5` | Results to keep after re-ranking |
| `JINA_API_KEY` | No | — | Required when `RERANKER_BACKEND=jina` |
| `IMAGE_CAPTION_ENABLED` | No | `true` | Enable GPT-4o captioning during ingestion |
| `LOG_LEVEL` | No | `INFO` | `DEBUG`, `INFO`, or `WARNING` |
| `LOG_JSON` | No | `false` | `true` = JSON-lines output (for log aggregators) |
| `OUTPUT_DIR` | No | `./output` | Default output directory |
| `API_HOST` | No | `0.0.0.0` | REST API bind address |
| `API_PORT` | No | `8000` | REST API bind port |
| `API_WORKERS` | No | `1` | Uvicorn worker count |

*`OPENAI_API_KEY` is always required for GPT-4o captioning and reranking. It is also needed for embeddings when `EMBEDDING_PROVIDER=openai` (the default).

†`GEMINI_API_KEY` is only required when `EMBEDDING_PROVIDER=gemini`. Also install `.[gemini]`.

### `config.yaml` settings

| Setting | Default | Description |
|---------|---------|-------------|
| `pipeline.maas.enabled` | `true` | Use cloud MaaS API |
| `pipeline.maas.model` | `glm-ocr` | Model name on Z.AI |
| `pipeline.layout.confidence_threshold` | `0.3` | Min detection confidence |
| `pipeline.layout.nms_threshold` | `0.5` | Non-max suppression threshold |
| `pipeline.output.include_bbox` | `true` | Include bboxes in output |
| `pipeline.output.max_tokens` | `8192` | Max tokens per API call |

---

## Chunking Behavior

The chunker never breaks a table mid-row or splits a formula:

| Element type | Behavior |
|---|---|
| `table` | Always its own atomic chunk (`is_atomic=True`, `modality="table"`) |
| `formula` / `inline_formula` | Always its own atomic chunk (`modality="formula"`) |
| `figure` / `image` | Always its own atomic chunk (`modality="image"`) |
| `algorithm` | Always its own atomic chunk |
| `document_title` / `paragraph_title` | Attaches to the next content element |
| `paragraph`, `text`, `references` | Accumulated up to `max_chunk_tokens` (default: 512) |

Token estimation: `len(text.split()) × 1.3`.

---

## Supported File Types

| Extension | Description |
|-----------|-------------|
| `.pdf` | PDF documents (all pages) |
| `.png` | PNG images |
| `.jpg` / `.jpeg` | JPEG images |
| `.tiff` | TIFF images |
| `.bmp` | Bitmap images |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'glmocr'`**
```bash
uv pip install -e ".[dev]"
```

**`ValidationError: z_ai_api_key field required`**
- Check `.env` contains `Z_AI_API_KEY=your-key`
- Or export: `export Z_AI_API_KEY=your-key`

**`ValidationError: openai_api_key field required`** (during ingestion or search)
- Add `OPENAI_API_KEY=sk-...` to `.env`

**`ValueError: JINA_API_KEY must be set`**
- Sign up at [jina.ai](https://jina.ai) and add `JINA_API_KEY=jina_...` to `.env`
- Or switch to the default backend: `RERANKER_BACKEND=openai`

**`ImportError: BGE reranker requires FlagEmbedding`**
```bash
uv pip install -e ".[bge]"
```

**`ImportError: Qwen VL reranker requires transformers and torch`**
```bash
uv pip install -e ".[qwen]"
```

**`ImportError: google-genai is required`**
```bash
uv pip install -e ".[gemini]"
```

**`ValueError: GEMINI_API_KEY must be set when EMBEDDING_PROVIDER=gemini`**
- Add `GEMINI_API_KEY=AIzaSy...` to `.env`

**Qdrant connection refused**
- Start local Qdrant: `docker-compose up -d qdrant`
- Or set `QDRANT_URL` to your Qdrant Cloud endpoint

**Integration tests not running (just skipped)**
- Expected when API keys are not set — add them to `.env`

**`ModuleNotFoundError: No module named 'fastapi'`**
```bash
uv pip install -e ".[dev]"
```

**`Address already in use` when starting the API server**
```bash
python scripts/serve.py --port 8001  # use a different port
```

**Bounding boxes appear misaligned in the Streamlit app**
- All `bbox_2d` values are normalized to 0–1000. Formula: `pixel_x = bbox_x × image_width / 1000`

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Layout detection + OCR (cloud) | `glmocr` | ≥0.1.0 |
| Layout detection + OCR (local) | `glmocr[layout]` + Ollama | ≥0.1.0 |
| PDF → image extraction | `pymupdf` | ≥1.27.2 |
| Image processing | `Pillow` | ≥12.1.1 |
| Config management | `pydantic-settings` | ≥2.8.0 |
| LLM + embeddings (OpenAI) | `openai` | ≥2.24.0 |
| Embeddings (Gemini, optional) | `google-genai` | ≥1.0.0 |
| Vector database | `qdrant-client` | ≥1.17.0 |
| Sparse vectors (BM25 proxy) | feature hashing | built-in |
| Async HTTP | `httpx` | ≥0.28.0 |
| Token counting | `tiktoken` | ≥0.9.0 |
| Local re-ranking (optional) | `FlagEmbedding` | ≥1.3.0 |
| Local VL re-ranking (optional) | `transformers` + `torch` | ≥4.51.0 / ≥2.7.0 |
| Progress bars | `tqdm` | ≥4.67.0 |
| Terminal output | `rich` | ≥14.0.0 |
| Visual inspector UI | `streamlit` | ≥1.40.0 |
| REST API framework | `fastapi` | ≥0.120.0 |
| ASGI server | `uvicorn[standard]` | ≥0.34.0 |
| Structured logging | `loguru` | ≥0.7.0 |
| Testing | `pytest` | ≥8.4.0 |
| Linting | `ruff` | ≥0.11.0 |
