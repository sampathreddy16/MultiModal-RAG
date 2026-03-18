"""POST /ingest endpoint — file upload and JSON-path variants."""
from __future__ import annotations

import asyncio
import tempfile
import time
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from doc_parser.api.dependencies import get_embedder_dep, get_openai_client, get_store
from doc_parser.api.schemas import IngestRequest, IngestResponse
from doc_parser.chunker import structure_aware_chunking
from doc_parser.config import get_settings
from doc_parser.ingestion.embedder import embed_chunks
from doc_parser.ingestion.image_captioner import enrich_image_chunks
from doc_parser.pipeline import DocumentParser

router = APIRouter()


async def _run_ingest(
    pdf_path: Path,
    collection_override: str | None,
    overwrite: bool,
    max_chunk_tokens: int,
    caption: bool,
) -> IngestResponse:
    """Core ingest logic shared by both endpoint variants."""
    settings = get_settings()
    client = get_openai_client()
    embedder = get_embedder_dep()
    store = get_store()

    # Override collection name when requested
    if collection_override:
        store._collection = collection_override
    collection = store._collection

    t0 = time.perf_counter()

    # 1. Parse PDF (synchronous SDK call — offload to thread pool)
    try:
        parser = DocumentParser()
        loop = asyncio.get_running_loop()
        parse_result = await loop.run_in_executor(None, parser.parse_file, pdf_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Parsing failed for {}: {}", pdf_path, exc)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {exc}") from exc

    # 2. Chunk
    chunks = structure_aware_chunking(parse_result, max_chunk_tokens=max_chunk_tokens)
    logger.info("Chunked {} → {} chunks", pdf_path.name, len(chunks))

    # 3. Caption image chunks (if enabled)
    if caption and settings.image_caption_enabled:
        chunks = await enrich_image_chunks(chunks, pdf_path=pdf_path, client=client)

    # 4. Embed
    dense, sparse = await embed_chunks(chunks, embedder, settings)

    # 5. Ensure collection exists then upsert
    await store.create_collection(overwrite=overwrite)
    upserted = await store.upsert_chunks(chunks, dense, sparse)

    latency_ms = (time.perf_counter() - t0) * 1000
    modality_counts = dict(Counter(c.modality for c in chunks))

    return IngestResponse(
        source_file=str(pdf_path),
        collection=collection,
        chunks_upserted=upserted,
        modality_counts=modality_counts,
        latency_ms=round(latency_ms, 2),
    )


_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"})


@router.post("/file", response_model=IngestResponse, summary="Ingest document via file upload")
async def ingest_file(
    file: UploadFile = File(..., description="Document file to ingest (PDF or image)."),
    collection: str | None = Form(None, description="Override collection name."),
    overwrite: bool = Form(False, description="Recreate collection before ingesting."),
    max_chunk_tokens: int = Form(512, ge=64, le=4096, description="Max tokens per chunk."),
    caption: bool = Form(True, description="Run GPT-4o captioning on image chunks."),
) -> IngestResponse:
    """Upload a PDF or image file and ingest it into the vector store."""
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if not file.filename or suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
        )

    # Save upload to a temp file (preserve original suffix for parser)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        return await _run_ingest(tmp_path, collection, overwrite, max_chunk_tokens, caption)
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("", response_model=IngestResponse, summary="Ingest document by file path")
async def ingest_by_path(req: IngestRequest) -> IngestResponse:
    """Ingest a document referenced by its local file path."""
    file_path = Path(req.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return await _run_ingest(file_path, req.collection, req.overwrite, req.max_chunk_tokens, req.caption)
