# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

```bash
# Install (always use uv, never pip directly)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

# Optional extras
uv pip install -e ".[bge]"    # local BGE reranker
uv pip install -e ".[qwen]"   # local Qwen VL reranker
uv pip install -e ".[gemini]" # Gemini embeddings

# Unit tests (no API keys needed, ~1 second)
uv run pytest tests/unit/ -v

# Single test
uv run pytest tests/unit/test_embedder.py::TestOpenAIEmbedder -v

# Lint / format
uv run ruff check src/ tests/ scripts/
uv run ruff check --fix src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/

# Type check
uv run mypy src/

# Start API server (dev)
python scripts/serve.py --reload

# CLI tools
python scripts/parse.py paper.pdf --chunks
python scripts/ingest.py paper.pdf
python scripts/search.py "query text"
```

---

## Architecture

The pipeline has four sequential phases. Each phase has a clear boundary in the codebase.

### Phase 1 — Parse (`pipeline.py`, `post_processor.py`, `chunker.py`)

`DocumentParser.parse_file()` calls the GLM-OCR MaaS API (Z.AI cloud, no GPU needed) and returns a `ParseResult`. The SDK returns a `PipelineResult` with two fields:
- `json_result`: `list[list[dict]]` — per-page elements with `label`, `content`, `bbox_2d`, `index`
- `markdown_result`: full-document Markdown string

`ParseResult.from_sdk_result()` converts this into typed `PageResult → ParsedElement` dataclasses. `assemble_markdown()` reconstructs per-page Markdown from element labels.

`structure_aware_chunking()` walks sorted elements and produces `Chunk` objects. Key rules: atomic labels (`table`, `formula`, `algorithm`, `image`, `figure`) always get their own chunk with `is_atomic=True`; title labels attach forward to the next content element; text elements accumulate up to `max_chunk_tokens` (default 512, estimated at `word_count × 1.3`).

**Important PDF detail:** `parse_file()` must pass `start_page_id=0, end_page_id=N-1` explicitly to the SDK — without these, the SDK defaults to page 1 only.

### Phase 2 — Ingest (`ingestion/`)

Three components run sequentially:

1. **`image_captioner.py`** — For each `Chunk` with `modality="image"`, crops the region from the source file using PyMuPDF, encodes as base64 PNG, and sends to GPT-4o for a 1–2 sentence caption. Errors are swallowed per-chunk (sets `chunk.text = "[figure]"`). Uses an `asyncio.Semaphore` for concurrency control. Always requires `AsyncOpenAI` regardless of the embedding provider.

2. **`embedder.py`** — `BaseEmbedder` ABC with two concrete implementations: `OpenAIEmbedder` (wraps `embed_texts()`) and `GeminiEmbedder` (runs sync Gemini SDK in `run_in_executor`). `get_embedder(settings)` is the factory. `embed_chunks(chunks, embedder, settings)` combines dense embeddings from the provider with feature-hashed BM25 sparse vectors (`compute_sparse_vectors()`).

3. **`vector_store.py`** — `QdrantDocumentStore` manages an async Qdrant client. Collection uses two named vector spaces: `text_dense` (COSINE, size from `settings.embedding_dimensions`) and `bm25_sparse`. `search()` runs hybrid retrieval with `Prefetch` + `FusionQuery(RRF)`.

### Phase 3 — Retrieval (`retrieval/reranker.py`)

`BaseReranker` ABC with four backends: `OpenAIReranker` (GPT-4o-mini cross-encoder, default), `JinaReranker` (cloud API), `BGEReranker` (local, needs `.[bge]`), `QwenVLReranker` (local VLM, needs `.[qwen]`). `get_reranker(settings)` is the factory. Image chunks carry `image_base64`; the OpenAI and Jina backends pass images inline for visual scoring; BGE uses the caption text.

### Phase 4 — API (`api/`)

FastAPI app created via `create_app()` in `app.py`. Routes are mounted at `/ingest` and `/search`. Shared dependencies in `dependencies.py` use `@lru_cache` for singleton lifetime: `get_openai_client()`, `get_store()`, `get_reranker_dep()`, `get_embedder_dep()`.

`get_openai_client()` is kept separate from `get_embedder_dep()` because the OpenAI client is also needed for image captioning (`enrich_image_chunks`) regardless of the configured embedding provider.

`LoggingMiddleware` (`middleware.py`) attaches an `X-Request-Id` header (8-char UUID prefix) to every response for log correlation.

---

## Key Patterns

**Adding a new embedding provider:** subclass `BaseEmbedder`, implement `async embed(texts) -> list[list[float]]`, add to `_PROVIDERS` dict in `embedder.py`, add optional dep to `pyproject.toml`.

**Adding a new reranker backend:** subclass `BaseReranker`, implement `async rerank(query, candidates, top_n)`, add to `_BACKENDS` dict in `reranker.py`, add to `RERANKER_BACKEND` env var docs.

**Config:** all settings flow through `Settings` (pydantic-settings). The singleton is `get_settings()`. Never read `os.environ` directly.

**Async boundary:** `DocumentParser.parse_file()` and `parser.parse()` are synchronous (GLM-OCR SDK). In async contexts (FastAPI routes, `scripts/ingest.py`), they are offloaded via `loop.run_in_executor(None, parser.parse_file, path)`.

**Qdrant deprecated API:** never use `client.search()`, `client.upload_records()`, or `client.recommend()`. Always use `client.query_points()` and `client.upsert()`.

---

## Configuration

All env vars loaded from `.env`. Critical ones:

| Var | Purpose |
|-----|---------|
| `Z_AI_API_KEY` | GLM-OCR MaaS API (always required) |
| `OPENAI_API_KEY` | Captioning + reranking (always required) |
| `EMBEDDING_PROVIDER` | `openai` (default) or `gemini` |
| `EMBEDDING_MODEL` | Default: `text-embedding-3-large` |
| `EMBEDDING_DIMENSIONS` | Default: `3072` (1536 for `text-embedding-3-small`) |
| `GEMINI_API_KEY` | Required only when `EMBEDDING_PROVIDER=gemini` |
| `RERANKER_BACKEND` | `openai` \| `jina` \| `bge` \| `qwen` |

Changing `EMBEDDING_DIMENSIONS` after a collection is created requires `--overwrite` (vectors are incompatible across dimension changes).

---

## Tests

Unit tests in `tests/unit/` are fully mocked — no API keys or running Qdrant required. Integration tests in `tests/integration/` require live credentials.

`pytest-asyncio` is configured with `asyncio_mode = "auto"` — no need to add `@pytest.mark.asyncio` explicitly, but it is used in existing tests for clarity.

When mocking `OpenAIEmbedder` or `get_embedder`, patch `doc_parser.ingestion.embedder.AsyncOpenAI` to prevent the real client from raising when no `OPENAI_API_KEY` env var is set.
